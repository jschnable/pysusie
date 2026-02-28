#!/usr/bin/env python3
"""Fine-map a GWAS locus using summary statistics and a VCF reference panel.

This script demonstrates a typical SuSiE fine-mapping workflow:

    1. Load GWAS summary statistics (BETA, SE columns) and filter to a locus.
    2. Read genotypes from a VCF file for the same region.
    3. Align variants between the GWAS results and the VCF.
    4. Compute an LD correlation matrix from genotype dosages.
    5. Run SuSiE with ``fit_from_summary_stats`` and LD regularization.
    6. Report credible sets and posterior inclusion probabilities.

Requirements beyond pysusie:
    pip install cyvcf2 pandas matplotlib

Usage:
    python finemapping_from_vcf.py \\
        --vcf reference_panel.vcf.gz \\
        --gwas gwas_results.csv \\
        --chrom 1 --start 1000000 --end 2000000 \\
        --n-samples 50000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gwas(path: str, chrom: str, start: int, end: int) -> "pd.DataFrame":
    """Load a GWAS summary-statistics CSV and filter to a genomic region.

    Expected columns: CHR, POS, REF, ALT, BETA, SE
    Additional columns (e.g. P, A1, A2, SNP) are preserved but not required.
    """
    import pandas as pd

    df = pd.read_csv(path)
    required = {"CHR", "POS", "REF", "ALT", "BETA", "SE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GWAS file missing columns: {missing}")

    df["CHR"] = df["CHR"].astype(str)
    chrom = str(chrom)

    mask = (df["CHR"] == chrom) & (df["POS"] >= start) & (df["POS"] <= end)
    locus = df.loc[mask].copy()
    if locus.empty:
        raise ValueError(f"No GWAS variants in {chrom}:{start}-{end}")

    # Z-score = BETA / SE
    locus["Z"] = locus["BETA"] / locus["SE"]

    # Unique variant ID for matching: chr:pos:ref:alt
    locus["VARID"] = (
        locus["CHR"] + ":" + locus["POS"].astype(str) + ":"
        + locus["REF"] + ":" + locus["ALT"]
    )

    return locus.reset_index(drop=True)


def read_vcf_genotypes(
    vcf_path: str, chrom: str, start: int, end: int
) -> tuple[np.ndarray, list[str]]:
    """Read genotype dosages from a VCF for a genomic region.

    Returns
    -------
    dosages : ndarray, shape (n_variants, n_samples)
        Alternate-allele dosage (0/1/2), with missing genotypes mean-imputed.
    var_ids : list[str]
        Variant IDs as chr:pos:ref:alt.

    Notes
    -----
    Uses cyvcf2, which is the most common Python VCF reader in genetics
    pipelines.  If you prefer scikit-allel, replace this function with::

        import allel
        callset = allel.read_vcf(vcf_path, region=f"{chrom}:{start}-{end}",
                                 fields=["variants/CHROM", "variants/POS",
                                         "variants/REF", "variants/ALT",
                                         "calldata/GT"])
        gt = allel.GenotypeArray(callset["calldata/GT"])
        dosages = gt.to_n_alt().astype(float)
        # ... then mean-impute missing (-1) and build var_ids
    """
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    dosages_list: list[np.ndarray] = []
    var_ids: list[str] = []

    for variant in vcf(f"{chrom}:{start}-{end}"):
        if len(variant.ALT) != 1:
            continue  # skip multi-allelics

        # Genotype dosage: count of ALT alleles per sample
        gt = variant.genotype.array()[:, :2]  # shape (n_samples, 2)
        dosage = gt.sum(axis=1).astype(float)

        # Mean-impute missing genotypes (coded as -1 by cyvcf2)
        missing = dosage < 0
        if missing.any():
            valid_mean = dosage[~missing].mean() if (~missing).any() else 0.0
            dosage[missing] = valid_mean

        dosages_list.append(dosage)
        var_ids.append(
            f"{variant.CHROM}:{variant.POS}:{variant.REF}:{variant.ALT[0]}"
        )

    vcf.close()

    if not dosages_list:
        raise ValueError(f"No biallelic variants found in VCF for {chrom}:{start}-{end}")

    return np.array(dosages_list), var_ids


def compute_ld(dosages: np.ndarray) -> np.ndarray:
    """Compute a Pearson correlation (LD) matrix from genotype dosages.

    Parameters
    ----------
    dosages : ndarray, shape (n_variants, n_samples)

    Returns
    -------
    R : ndarray, shape (n_variants, n_variants)
        LD correlation matrix.
    """
    # Center each variant
    means = dosages.mean(axis=1, keepdims=True)
    centered = dosages - means

    # Standard deviation per variant
    stds = centered.std(axis=1, keepdims=True, ddof=0)

    # Remove monomorphic SNPs (std == 0) — they carry no LD information
    mono = (stds.ravel() == 0)
    if mono.any():
        print(f"  Removing {mono.sum()} monomorphic variant(s) from LD calculation",
              file=sys.stderr)

    # Replace zero stds with 1 to avoid division by zero; these rows/cols
    # will produce zero correlations, which is correct.
    stds[stds == 0] = 1.0
    normed = centered / stds

    n_samples = dosages.shape[1]
    R = (normed @ normed.T) / n_samples

    # Ensure exact symmetry and unit diagonal
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)

    return R


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-map a GWAS locus with SuSiE using a VCF reference panel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vcf", required=True,
                        help="Path to VCF/BCF file (indexed with .tbi or .csi)")
    parser.add_argument("--gwas", required=True,
                        help="GWAS summary statistics CSV (CHR, POS, REF, ALT, BETA, SE)")
    parser.add_argument("--chrom", required=True,
                        help="Chromosome of the locus")
    parser.add_argument("--start", required=True, type=int,
                        help="Start position (bp, inclusive)")
    parser.add_argument("--end", required=True, type=int,
                        help="End position (bp, inclusive)")
    parser.add_argument("--n-samples", required=True, type=int,
                        help="GWAS sample size (n)")
    parser.add_argument("--var-y", type=float, default=1.0,
                        help="Phenotype variance; 1.0 for quantitative traits "
                             "on z-score scale (default: 1.0)")
    parser.add_argument("--n-effects", type=int, default=10,
                        help="Maximum number of causal signals, L (default: 10)")
    parser.add_argument("--output-prefix", default=None,
                        help="If set, write PIPs CSV to PREFIX_pips.csv "
                             "and a PIP plot to PREFIX_pip.png")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Step 1: Load and filter GWAS summary stats
    # ------------------------------------------------------------------
    print(f"Loading GWAS summary stats from {args.gwas} ...")
    gwas = load_gwas(args.gwas, args.chrom, args.start, args.end)
    print(f"  {len(gwas)} variants in locus {args.chrom}:{args.start}-{args.end}")

    # ------------------------------------------------------------------
    # Step 2: Read genotypes from VCF
    # ------------------------------------------------------------------
    print(f"Reading genotypes from {args.vcf} ...")
    dosages, vcf_ids = read_vcf_genotypes(args.vcf, args.chrom, args.start, args.end)
    print(f"  {len(vcf_ids)} biallelic variants, {dosages.shape[1]} samples in VCF")

    # ------------------------------------------------------------------
    # Step 3: Align variants between GWAS and VCF
    # ------------------------------------------------------------------
    # Match on chr:pos:ref:alt so allele coding is consistent.
    vcf_set = set(vcf_ids)
    gwas_set = set(gwas["VARID"])
    shared = sorted(vcf_set & gwas_set)

    if not shared:
        print("ERROR: No overlapping variants between GWAS and VCF.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(shared)} variants shared between GWAS and VCF")

    # Reorder both datasets to the shared variant order
    vcf_idx = {vid: i for i, vid in enumerate(vcf_ids)}
    vcf_order = [vcf_idx[v] for v in shared]
    dosages = dosages[vcf_order]

    gwas_idx = gwas.set_index("VARID")
    gwas_aligned = gwas_idx.loc[shared]
    z = gwas_aligned["Z"].values.astype(float)

    # ------------------------------------------------------------------
    # Step 4: Compute LD matrix
    # ------------------------------------------------------------------
    print("Computing LD matrix ...")

    # Remove monomorphic variants (no variation → undefined correlation)
    variant_std = dosages.std(axis=1)
    polymorphic = variant_std > 0
    if not polymorphic.all():
        n_mono = (~polymorphic).sum()
        print(f"  Removing {n_mono} monomorphic variant(s)")
        dosages = dosages[polymorphic]
        z = z[polymorphic]
        shared = [v for v, keep in zip(shared, polymorphic) if keep]

    R = compute_ld(dosages)
    print(f"  LD matrix: {R.shape[0]} x {R.shape[1]}")

    # ------------------------------------------------------------------
    # Step 5: Run SuSiE
    # ------------------------------------------------------------------
    # regularize_ld="auto" shrinks R toward the identity to stabilise the
    # model when the reference panel is smaller than the GWAS sample.
    # refine=True re-runs IBSS to escape local optima in credible sets.
    from pysusie import SuSiE

    print(f"Fitting SuSiE (L={args.n_effects}, n={args.n_samples}) ...")
    model = SuSiE(n_effects=args.n_effects)
    model.fit_from_summary_stats(
        z=z,
        R=R,
        n=args.n_samples,
        var_y=args.var_y,
        regularize_ld="auto",
        estimate_residual_variance=False,
    )
    # Enable refinement for the final stage only, after obtaining a stable
    # initial fit with fixed residual variance.
    model.refine = True
    model.fit_from_summary_stats(
        z=z,
        R=R,
        n=args.n_samples,
        var_y=args.var_y,
        regularize_ld="auto",
        estimate_residual_variance=True,
        model_init=model,
    )

    result = model.result_
    print(f"  Converged: {result.converged} ({result.n_iter} iterations)")

    # ------------------------------------------------------------------
    # Step 6: Extract credible sets
    # ------------------------------------------------------------------
    cs_list = result.get_credible_sets(R=R)
    print(f"\n{'='*60}")
    print(f"Found {len(cs_list)} credible set(s)")
    print(f"{'='*60}")

    for i, cs in enumerate(cs_list):
        top_idx = cs.variables[np.argmax(result.pip[cs.variables])]
        top_id = shared[top_idx]
        purity_str = (f"min_r={cs.purity.min_abs_corr:.3f}"
                      if cs.purity else "no purity")
        print(f"\n  CS {i+1}: {len(cs.variables)} variants, "
              f"coverage={cs.coverage:.3f}, {purity_str}")
        print(f"    Top variant: {top_id} (PIP={result.pip[top_idx]:.4f})")

    # Top PIPs regardless of CS membership
    top_k = min(10, len(z))
    top_order = np.argsort(-result.pip)[:top_k]
    print(f"\nTop {top_k} PIPs:")
    for rank, idx in enumerate(top_order, 1):
        print(f"  {rank}. {shared[idx]}  PIP={result.pip[idx]:.4f}  "
              f"z={z[idx]:.2f}")

    # ------------------------------------------------------------------
    # Step 7 (optional): Save outputs
    # ------------------------------------------------------------------
    if args.output_prefix:
        import pandas as pd

        pip_df = pd.DataFrame({
            "variant": shared,
            "pip": result.pip,
            "z": z,
            "coef": result.coef,
        })
        csv_path = f"{args.output_prefix}_pips.csv"
        pip_df.to_csv(csv_path, index=False)
        print(f"\nPIPs written to {csv_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            positions = [int(v.split(":")[1]) for v in shared]
            ax.scatter(positions, result.pip, s=8, alpha=0.7, color="steelblue")

            # Highlight credible-set variants
            for cs in cs_list:
                cs_pos = [positions[j] for j in cs.variables]
                cs_pip = result.pip[cs.variables]
                ax.scatter(cs_pos, cs_pip, s=20, alpha=0.9, color="red",
                           zorder=3)

            ax.set_xlabel("Position (bp)")
            ax.set_ylabel("PIP")
            ax.set_title(f"Fine-mapping: {args.chrom}:{args.start}-{args.end}")
            ax.set_ylim(-0.02, 1.05)
            fig.tight_layout()

            plot_path = f"{args.output_prefix}_pip.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"PIP plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
