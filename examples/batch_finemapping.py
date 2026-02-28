#!/usr/bin/env python3
"""Batch fine-mapping of many loci with computational optimizations.

This script demonstrates how to fine-map hundreds or thousands of loci
efficiently.  The key optimizations over a naive per-locus approach:

    1. **Load genotypes once** into a single NumPy array so each locus only
       needs an array slice — no repeated VCF parsing.
    2. **Precompute chromosome indices** so finding markers in a genomic
       window is a fast array operation instead of a linear scan.
    3. **Reuse a single SuSiE estimator** across loci (avoids re-allocating
       internal buffers).
    4. **Checkpoint periodically** so long runs can be resumed after
       interruption.

The script supports two modes:

    - **Individual-level data** (``--phenotype``): fits SuSiE directly on
      genotypes × phenotype with ``model.fit(X, y)``.
    - **Summary statistics** (``--gwas``): computes LD from the genotype
      matrix, then fits with ``model.fit_from_summary_stats(z, R, n)``.

Input formats
-------------
Genotypes (``--geno-prefix PREFIX``):
    Three files produced by common preprocessing pipelines:
      - ``PREFIX.geno.npy``  — int8 array, shape (n_samples, n_markers),
        values 0/1/2 with -1 for missing.
      - ``PREFIX.map.csv``   — CSV with columns CHR, POS, SNP.
      - ``PREFIX.ind.txt``   — one sample ID per line.

    To convert a VCF to this format::

        import allel, numpy as np, pandas as pd
        callset = allel.read_vcf("input.vcf.gz",
                                 fields=["samples", "variants/*", "calldata/GT"])
        gt = allel.GenotypeArray(callset["calldata/GT"]).to_n_alt()
        gt[gt < 0] = -1
        np.save("prefix.geno.npy", gt.T.astype(np.int8))  # samples × markers
        pd.DataFrame({"CHR": callset["variants/CHROM"],
                       "POS": callset["variants/POS"],
                       "SNP": callset["variants/ID"]}).to_csv("prefix.map.csv", index=False)
        with open("prefix.ind.txt", "w") as f:
            f.writelines(s + "\\n" for s in callset["samples"])

Loci (``--loci``):
    CSV with columns: locus_id, chrom, start, end
    One row per region to fine-map.

Phenotype (``--phenotype``, individual-level mode):
    CSV where the first column is sample IDs and remaining columns are
    phenotypes.  Column names are matched to locus_id.

GWAS (``--gwas``, summary-stats mode):
    CSV with columns: locus_id, CHR, POS, SNP, BETA, SE

Usage examples
--------------
Individual-level (e.g., eQTL fine-mapping — one phenotype per gene)::

    python batch_finemapping.py \\
        --geno-prefix genotypes \\
        --loci loci.csv \\
        --phenotype expression.csv \\
        --output-dir results/

Summary statistics::

    python batch_finemapping.py \\
        --geno-prefix ref_panel \\
        --loci loci.csv \\
        --gwas gwas_sumstats.csv \\
        --n-samples 50000 \\
        --output-dir results/

Requirements beyond pysusie:
    pip install pandas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Genotype loading
# ---------------------------------------------------------------------------

def load_genotypes(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load precomputed genotype arrays from PREFIX.{geno.npy,map.csv,ind.txt}.

    Returns
    -------
    geno : ndarray, shape (n_samples, n_markers), int8
    chroms : ndarray of chromosome labels (str)
    positions : ndarray of base-pair positions (int)
    sample_ids : list of sample ID strings
    """
    geno = np.load(f"{prefix}.geno.npy")
    marker_map = pd.read_csv(f"{prefix}.map.csv")
    with open(f"{prefix}.ind.txt") as f:
        sample_ids = [line.strip() for line in f]

    chroms = marker_map["CHR"].astype(str).values
    positions = marker_map["POS"].values.astype(int)

    if geno.shape[0] != len(sample_ids):
        raise ValueError(
            f"geno.npy has {geno.shape[0]} rows but ind.txt has "
            f"{len(sample_ids)} samples"
        )
    if geno.shape[1] != len(positions):
        raise ValueError(
            f"geno.npy has {geno.shape[1]} columns but map.csv has "
            f"{len(positions)} markers"
        )

    return geno, chroms, positions, sample_ids


def build_chrom_index(chroms: np.ndarray) -> dict[str, np.ndarray]:
    """Precompute per-chromosome marker indices for fast locus extraction.

    Building this index once turns each locus lookup from O(n_markers) to
    O(n_markers_on_chrom), which matters when you have millions of markers.
    """
    index: dict[str, np.ndarray] = {}
    for c in np.unique(chroms):
        index[str(c)] = np.where(chroms == c)[0]
    return index


def extract_locus_genotypes(
    geno: np.ndarray,
    positions: np.ndarray,
    chrom_index: dict[str, np.ndarray],
    chrom: str,
    start: int,
    end: int,
    sample_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice genotypes for a genomic window.

    Returns
    -------
    X : ndarray, float64, shape (n_samples, n_locus_markers)
        Genotype dosages with missing values mean-imputed and monomorphic
        columns removed.
    marker_idx : ndarray of global marker indices that were kept.
    """
    chrom = str(chrom)
    if chrom not in chrom_index:
        return np.empty((0, 0)), np.array([], dtype=int)

    cidx = chrom_index[chrom]
    cpos = positions[cidx]
    in_window = (cpos >= start) & (cpos <= end)
    marker_idx = cidx[in_window]

    if marker_idx.size == 0:
        return np.empty((0, 0)), marker_idx

    # Slice rows (samples) and columns (markers) from the full array.
    if sample_idx is not None:
        X = geno[np.ix_(sample_idx, marker_idx)].astype(np.float64)
    else:
        X = geno[:, marker_idx].astype(np.float64)

    # Mean-impute missing genotypes (coded as -1)
    for j in range(X.shape[1]):
        col = X[:, j]
        missing = col < 0
        if missing.any():
            valid = col[~missing]
            col[missing] = valid.mean() if valid.size > 0 else 0.0

    # Drop monomorphic markers (zero variance → SuSiE can't use them)
    col_std = X.std(axis=0)
    polymorphic = col_std > 0
    if not polymorphic.all():
        X = X[:, polymorphic]
        marker_idx = marker_idx[polymorphic]

    return X, marker_idx


# ---------------------------------------------------------------------------
# Sample matching
# ---------------------------------------------------------------------------

def match_samples(
    geno_ids: list[str], pheno_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Find shared samples between genotype and phenotype files.

    Returns arrays of indices into geno_ids and pheno_ids respectively,
    in matched order.
    """
    pheno_lookup = {pid: i for i, pid in enumerate(pheno_ids)}
    geno_idx, pheno_idx = [], []
    for gi, gid in enumerate(geno_ids):
        if gid in pheno_lookup:
            geno_idx.append(gi)
            pheno_idx.append(pheno_lookup[gid])
    return np.array(geno_idx, dtype=int), np.array(pheno_idx, dtype=int)


# ---------------------------------------------------------------------------
# LD computation
# ---------------------------------------------------------------------------

def compute_ld(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix from a samples × markers genotype matrix."""
    centered = X - X.mean(axis=0, keepdims=True)
    stds = centered.std(axis=0, keepdims=True, ddof=0)
    stds[stds == 0] = 1.0  # guard (monomorphic already removed upstream)
    normed = centered / stds
    R = (normed.T @ normed) / X.shape[0]
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# Core fine-mapping routines
# ---------------------------------------------------------------------------

def finemap_individual(
    model, X: np.ndarray, y: np.ndarray
) -> tuple:
    """Fit SuSiE on individual-level data and extract credible sets."""
    model.fit(X, y)
    result = model.result_
    cs_list = result.get_credible_sets(X=X)
    return model, cs_list


def finemap_summary_stats(
    model, X: np.ndarray, z: np.ndarray, n: int
) -> tuple:
    """Compute LD from genotypes, then fit SuSiE on z-scores."""
    R = compute_ld(X)

    # Stage 1: fixed residual variance for a stable starting point
    model.fit_from_summary_stats(
        z=z, R=R, n=n, var_y=1.0,
        regularize_ld="auto",
        estimate_residual_variance=False,
    )
    # Stage 2: estimate residual variance and refine credible sets
    model.refine = True
    model.fit_from_summary_stats(
        z=z, R=R, n=n, var_y=1.0,
        regularize_ld="auto",
        estimate_residual_variance=True,
        model_init=model,
    )

    result = model.result_
    cs_list = result.get_credible_sets(R=R)
    return model, cs_list


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    summary_rows: list[dict],
    cs_rows: list[dict],
    checkpoint_dir: Path,
) -> None:
    """Write current progress to disk so a crashed run can resume."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            checkpoint_dir / "checkpoint_summary.csv", index=False
        )
    if cs_rows:
        pd.DataFrame(cs_rows).to_csv(
            checkpoint_dir / "checkpoint_credible_sets.csv", index=False
        )
    meta = {"n_loci": len(summary_rows), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(checkpoint_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(checkpoint_dir: Path) -> tuple[list[dict], list[dict], set[str]]:
    """Reload progress from a previous run."""
    summary_rows: list[dict] = []
    cs_rows: list[dict] = []
    done: set[str] = set()

    summary_path = checkpoint_dir / "checkpoint_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        summary_rows = df.to_dict("records")
        done = set(df["locus_id"].astype(str))

    cs_path = checkpoint_dir / "checkpoint_credible_sets.csv"
    if cs_path.exists():
        cs_rows = pd.read_csv(cs_path).to_dict("records")

    return summary_rows, cs_rows, done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch fine-mapping of many loci with pySuSiE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--geno-prefix", required=True,
        help="Path prefix for genotype files (PREFIX.geno.npy, .map.csv, .ind.txt)",
    )
    parser.add_argument("--loci", required=True,
                        help="CSV of loci to fine-map (locus_id, chrom, start, end)")
    parser.add_argument("--phenotype", default=None,
                        help="Phenotype CSV for individual-level fitting "
                             "(first column = sample ID, one column per locus_id)")
    parser.add_argument("--gwas", default=None,
                        help="GWAS summary stats CSV for summary-stats fitting "
                             "(locus_id, CHR, POS, SNP, BETA, SE)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="GWAS sample size (required for --gwas mode)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output CSVs")
    parser.add_argument("--n-effects", type=int, default=10,
                        help="Max causal signals per locus (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N loci (default: 100)")

    args = parser.parse_args(argv)

    if args.phenotype is None and args.gwas is None:
        parser.error("Provide --phenotype (individual-level) or --gwas (summary stats)")
    if args.gwas is not None and args.n_samples is None:
        parser.error("--n-samples is required with --gwas")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be >= 1")

    from pysusie import SuSiE

    use_summary = args.gwas is not None
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load all data upfront
    # ------------------------------------------------------------------
    # Loading the full genotype matrix into RAM once is the single biggest
    # optimization.  Each locus then only needs an array slice (~1 ms)
    # instead of VCF parsing (~seconds).
    t_start = time.time()
    print("Loading genotypes ...")
    geno, chroms, positions, geno_ids = load_genotypes(args.geno_prefix)
    print(f"  {geno.shape[0]} samples x {geno.shape[1]} markers "
          f"({time.time() - t_start:.1f}s)")

    # Build a per-chromosome index so we can quickly find markers in any
    # genomic window without scanning the full marker array.
    chrom_index = build_chrom_index(chroms)

    loci = pd.read_csv(args.loci)
    for col in ("locus_id", "chrom", "start", "end"):
        if col not in loci.columns:
            raise ValueError(f"Loci file missing column: {col}")
    print(f"  {len(loci)} loci to fine-map")

    # Mode-specific data loading
    sample_idx = None  # indices into geno rows
    pheno_df = None
    gwas_df = None

    if use_summary:
        gwas_df = pd.read_csv(args.gwas)
        for col in ("locus_id", "CHR", "POS", "BETA", "SE"):
            if col not in gwas_df.columns:
                raise ValueError(f"GWAS file missing column: {col}")
        gwas_df["locus_id"] = gwas_df["locus_id"].astype(str)
        gwas_df["CHR"] = gwas_df["CHR"].astype(str)
        gwas_df["POS"] = pd.to_numeric(gwas_df["POS"], errors="coerce")
        gwas_df["BETA"] = pd.to_numeric(gwas_df["BETA"], errors="coerce")
        gwas_df["SE"] = pd.to_numeric(gwas_df["SE"], errors="coerce")
        invalid = (
            ~np.isfinite(gwas_df["POS"].values)
            | ~np.isfinite(gwas_df["BETA"].values)
            | ~np.isfinite(gwas_df["SE"].values)
            | (gwas_df["SE"].values <= 0)
        )
        if invalid.any():
            n_bad = int(invalid.sum())
            raise ValueError(
                f"GWAS file has {n_bad} row(s) with invalid POS/BETA/SE "
                "(POS finite and SE > 0 required)"
            )
        gwas_df["POS"] = gwas_df["POS"].astype(int)
        gwas_df["Z"] = gwas_df["BETA"] / gwas_df["SE"]
        print(f"  {len(gwas_df)} GWAS rows loaded")
    else:
        pheno_df = pd.read_csv(args.phenotype)
        pheno_ids = pheno_df.iloc[:, 0].astype(str).tolist()
        pheno_df = pheno_df.set_index(pheno_df.columns[0])

        # Match samples once, not per-locus
        geno_idx, pheno_idx = match_samples(geno_ids, pheno_ids)
        if len(geno_idx) < 2:
            raise ValueError(
                "Need at least 2 overlapping samples between genotype and phenotype files"
            )
        sample_idx = geno_idx
        pheno_df = pheno_df.iloc[pheno_idx]
        print(f"  {len(geno_idx)} samples matched between genotype and phenotype")

    # ------------------------------------------------------------------
    # Step 2: Prepare estimator and checkpoint
    # ------------------------------------------------------------------
    # Reusing a single SuSiE instance avoids repeated __init__ overhead.
    model = SuSiE(
        n_effects=args.n_effects,
        estimate_prior_variance=True,
        estimate_residual_variance=True,
        standardize=True,
        intercept=True,
        max_iter=200,
        verbose=False,
    )

    summary_rows: list[dict] = []
    cs_rows: list[dict] = []
    done: set[str] = set()

    if args.resume:
        summary_rows, cs_rows, done = load_checkpoint(outdir)
        if done:
            print(f"  Resuming: {len(done)} loci already processed")

    # ------------------------------------------------------------------
    # Step 3: Loop over loci
    # ------------------------------------------------------------------
    n_total = len(loci)
    n_failed = 0
    timings: list[float] = []

    print(f"\nFine-mapping {n_total} loci ...")

    for row_i, locus_row in loci.iterrows():
        locus_id = str(locus_row["locus_id"])
        chrom = str(locus_row["chrom"])
        start = int(locus_row["start"])
        end = int(locus_row["end"])
        i = int(row_i) + 1  # 1-based progress counter

        if locus_id in done:
            continue

        t0 = time.time()

        # Extract genotypes for this window (fast array slice)
        X, marker_idx = extract_locus_genotypes(
            geno, positions, chrom_index, chrom, start, end,
            sample_idx=sample_idx,
        )

        if X.shape[1] < 2:
            n_failed += 1
            if X.shape[1] == 0:
                print(f"  [{i}/{n_total}] {locus_id}: SKIP (no markers in window)")
            else:
                print(f"  [{i}/{n_total}] {locus_id}: SKIP (only 1 marker)")
            continue

        # Run fine-mapping
        try:
            if use_summary:
                # Align GWAS z-scores to the extracted markers
                locus_gwas = gwas_df[
                    (gwas_df["locus_id"] == locus_id) & (gwas_df["CHR"] == chrom)
                ].copy()
                if locus_gwas.empty:
                    print(f"  [{i}/{n_total}] {locus_id}: SKIP (no GWAS rows for locus/chrom)")
                    n_failed += 1
                    continue

                # Require unique CHR:POS in both GWAS and marker map to avoid
                # ambiguous variant-to-zscore alignment.
                gwas_keys = list(
                    zip(
                        locus_gwas["CHR"].astype(str).values,
                        locus_gwas["POS"].astype(int).values,
                    )
                )
                if len(set(gwas_keys)) != len(gwas_keys):
                    raise ValueError(
                        "Duplicate GWAS rows for the same CHR:POS within locus; "
                        "deduplicate GWAS input first"
                    )
                locus_gwas = locus_gwas.reset_index(drop=True)
                gwas_lookup = {key: idx for idx, key in enumerate(gwas_keys)}

                marker_chrom = chroms[marker_idx].astype(str)
                marker_pos = positions[marker_idx].astype(int)
                marker_keys = list(zip(marker_chrom, marker_pos))
                if len(set(marker_keys)) != len(marker_keys):
                    raise ValueError(
                        "Duplicate marker CHR:POS in genotype map for this locus; "
                        "cannot align uniquely to GWAS rows"
                    )

                keep_marker = []
                keep_gwas = []
                for mi, key in enumerate(marker_keys):
                    if key in gwas_lookup:
                        keep_marker.append(mi)
                        keep_gwas.append(gwas_lookup[key])

                if len(keep_marker) < 2:
                    print(f"  [{i}/{n_total}] {locus_id}: SKIP "
                          f"(<2 shared variants with GWAS)")
                    n_failed += 1
                    continue

                X_aligned = X[:, keep_marker]
                z = locus_gwas.iloc[keep_gwas]["Z"].values.astype(float)
                model, cs_list = finemap_summary_stats(
                    model, X_aligned, z, args.n_samples
                )
            else:
                if locus_id not in pheno_df.columns:
                    print(f"  [{i}/{n_total}] {locus_id}: SKIP (no phenotype column)")
                    n_failed += 1
                    continue
                y = pheno_df[locus_id].values.astype(np.float64)
                model, cs_list = finemap_individual(model, X, y)

        except Exception as e:
            print(f"  [{i}/{n_total}] {locus_id}: FAILED ({e})")
            n_failed += 1
            continue

        elapsed = time.time() - t0
        timings.append(elapsed)
        result = model.result_

        # Collect results
        summary_rows.append({
            "locus_id": locus_id,
            "chrom": chrom,
            "start": start,
            "end": end,
            "n_markers": result.n_variables,
            "n_credible_sets": len(cs_list),
            "converged": result.converged,
            "n_iter": result.n_iter,
            "residual_variance": round(result.residual_variance, 6),
        })

        for ci, cs in enumerate(cs_list):
            lead = int(cs.variables[np.argmax(result.pip[cs.variables])])
            cs_rows.append({
                "locus_id": locus_id,
                "chrom": chrom,
                "cs_index": ci + 1,
                "n_variants": len(cs.variables),
                "coverage": round(cs.coverage, 4),
                "log_bf": round(cs.log_bayes_factor, 2),
                "purity_min": (round(cs.purity.min_abs_corr, 4)
                               if cs.purity else None),
                "lead_pip": round(float(result.pip[lead]), 4),
            })

        # Progress (every 100 loci or at the end)
        n_done = len(summary_rows)
        if n_done % 100 == 0 or i == n_total:
            mean_t = np.mean(timings[-100:]) if timings else 0
            remaining = (n_total - i) * mean_t / 60
            print(f"  [{i}/{n_total}] {locus_id}: {len(cs_list)} CS, "
                  f"{elapsed:.2f}s  (ETA {remaining:.1f} min)")

        # Periodic checkpoint
        if n_done % args.checkpoint_interval == 0:
            save_checkpoint(summary_rows, cs_rows, outdir)

    # ------------------------------------------------------------------
    # Step 4: Write final outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Writing outputs ...")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "finemapping_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  {summary_path} ({len(summary_df)} loci)")

    cs_df = pd.DataFrame(cs_rows)
    cs_path = outdir / "credible_sets.csv"
    cs_df.to_csv(cs_path, index=False)
    print(f"  {cs_path} ({len(cs_df)} credible sets)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("BATCH FINE-MAPPING COMPLETE")
    print(f"{'='*60}")
    print(f"  Loci processed:  {len(summary_df)}")
    print(f"  Loci failed:     {n_failed}")
    print(f"  Total CS found:  {len(cs_df)}")

    if len(summary_df) > 0:
        cs_counts = summary_df["n_credible_sets"]
        print(f"  CS per locus:    mean={cs_counts.mean():.2f}, "
              f"median={cs_counts.median():.0f}, max={cs_counts.max()}")
        print(f"  Converged:       {summary_df['converged'].sum()}/{len(summary_df)}")

    if timings:
        t_arr = np.array(timings)
        print(f"  Timing per locus: mean={t_arr.mean():.2f}s, "
              f"median={np.median(t_arr):.2f}s, max={t_arr.max():.2f}s")

    print(f"  Total time:      {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
