#!/usr/bin/env python3
"""Sweep PHYA window sizes for pysusie runtime and memory limits."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DATA_BASE = Path(
    "/Users/jamesschnable/Projects/pymashr/SorghumMashRExample/KarlaGene/cis_eqtl_report_package"
)
GENO_PATH = (
    DATA_BASE
    / "category3_baseline_data_symlinks/genotype/SAP_filtered_GTonly_maf025.vcf.gz.panicle.v2.geno.npy"
)
MAP_PATH = (
    DATA_BASE
    / "category3_baseline_data_symlinks/genotype/SAP_filtered_GTonly_maf025.vcf.gz.panicle.v2.map.csv"
)
IND_PATH = (
    DATA_BASE
    / "category3_baseline_data_symlinks/genotype/SAP_filtered_GTonly_maf025.vcf.gz.panicle.v2.ind.txt"
)
EXPR_PATH = (
    DATA_BASE / "category1_analyses/outputs/gwas_cis_eqtl/expression_phenotypes.csv"
)
MATCH_PATH = (
    DATA_BASE
    / "category1_analyses/outputs/gwas_cis_eqtl/matched_sample_ids_post_concordance.csv"
)
WINDOWS_PATH = DATA_BASE / "category1_analyses/outputs/gwas_cis_eqtl/gene_windows.csv"


def _chrom_bounds(chrom: int) -> tuple[int, int]:
    pos_min: int | None = None
    pos_max: int | None = None
    for chunk in pd.read_csv(
        MAP_PATH,
        usecols=["CHROM", "POS"],
        chunksize=1_000_000,
        dtype={"CHROM": "int16", "POS": "int64"},
    ):
        sub = chunk.loc[chunk["CHROM"] == chrom, "POS"]
        if sub.empty:
            continue
        local_min = int(sub.min())
        local_max = int(sub.max())
        pos_min = local_min if pos_min is None else min(pos_min, local_min)
        pos_max = local_max if pos_max is None else max(pos_max, local_max)
    if pos_min is None or pos_max is None:
        raise RuntimeError(f"No markers found for chromosome {chrom}")
    return pos_min, pos_max


def _load_gene_window(gene_id: str) -> tuple[int, int, int]:
    row = pd.read_csv(
        WINDOWS_PATH,
        usecols=["gene_id", "chrom", "gene_start", "gene_end"],
    ).query("gene_id == @gene_id")
    if row.empty:
        raise RuntimeError(f"Gene {gene_id} not found in {WINDOWS_PATH}")
    r = row.iloc[0]
    chrom = int(r["chrom"])
    gene_start = int(r["gene_start"])
    gene_end = int(r["gene_end"])
    return chrom, gene_start, gene_end


def _select_marker_indices(chrom: int, start: int, end: int) -> np.ndarray:
    idx_parts: list[np.ndarray] = []
    offset = 0
    for chunk in pd.read_csv(
        MAP_PATH,
        usecols=["CHROM", "POS"],
        chunksize=1_000_000,
        dtype={"CHROM": "int16", "POS": "int64"},
    ):
        mask = (chunk["CHROM"] == chrom) & (chunk["POS"] >= start) & (chunk["POS"] <= end)
        if mask.any():
            idx_parts.append(np.flatnonzero(mask.to_numpy()) + offset)
        offset += len(chunk)
    if not idx_parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(idx_parts).astype(np.int64)


def _aligned_rows_and_y(gene_id: str) -> tuple[np.ndarray, np.ndarray]:
    with IND_PATH.open() as f:
        geno_ids = [line.strip() for line in f if line.strip()]
    geno_lookup = {gid: i for i, gid in enumerate(geno_ids)}

    expr = pd.read_csv(EXPR_PATH, usecols=["Taxa", gene_id])
    expr_lookup = dict(zip(expr["Taxa"].astype(str), expr[gene_id].astype(float)))

    match = pd.read_csv(MATCH_PATH, usecols=["genotype_id", "expression_accession"])
    row_idx: list[int] = []
    y_vals: list[float] = []
    for rec in match.itertuples(index=False):
        gi = geno_lookup.get(str(rec.genotype_id))
        yv = expr_lookup.get(str(rec.expression_accession))
        if gi is None or yv is None or not np.isfinite(yv):
            continue
        row_idx.append(gi)
        y_vals.append(float(yv))
    if not row_idx:
        raise RuntimeError("No aligned samples available")
    return np.asarray(row_idx, dtype=np.int64), np.asarray(y_vals, dtype=np.float64)


def run_single(
    *,
    backend: str,
    gene_id: str,
    chrom: int,
    center: int,
    half_window_bp: int,
    l_effects: int,
    max_iter: int,
    coverage: float,
    min_abs_corr: float,
) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    backend = backend.lower()
    if backend not in {"pysusie", "susier"}:
        raise RuntimeError("backend must be one of: pysusie, susieR")

    chrom_min, chrom_max = _chrom_bounds(chrom)
    start = max(chrom_min, center - half_window_bp)
    end = min(chrom_max, center + half_window_bp)

    t0_total = time.perf_counter()
    marker_idx = _select_marker_indices(chrom, start, end)
    if marker_idx.size == 0:
        raise RuntimeError("Selected window has no markers")

    row_idx, y = _aligned_rows_and_y(gene_id)
    geno = np.load(GENO_PATH, mmap_mode="r")
    X = np.asarray(geno[np.ix_(row_idx, marker_idx)], dtype=np.float64)

    n_markers_raw = int(X.shape[1])
    var = X.var(axis=0)
    keep = var > 0
    X = X[:, keep]
    n_markers = int(X.shape[1])
    n_markers_dropped = n_markers_raw - n_markers
    if n_markers == 0:
        raise RuntimeError("All markers are invariant in this window")

    if backend == "pysusie":
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from pysusie import SuSiE  # pylint: disable=import-error

        model = SuSiE(
            n_effects=l_effects,
            max_iter=max_iter,
            estimate_residual_variance=True,
            standardize=True,
            intercept=True,
            coverage=coverage,
            min_abs_corr=min_abs_corr,
            verbose=False,
        )
        t0_fit = time.perf_counter()
        model.fit(X, y)
        fit_sec = time.perf_counter() - t0_fit
        n_iter = int(model.result_.n_iter)
        converged = bool(model.result_.converged)
    else:
        os.environ.setdefault("R_HOME", "/opt/homebrew/Cellar/r/4.5.2_1/lib/R")
        os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
        from rpy2 import robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter

        robjects.r("suppressPackageStartupMessages(library(susieR))")
        with localconverter(robjects.default_converter + numpy2ri.converter):
            robjects.globalenv["X_py"] = X
            robjects.globalenv["y_py"] = y

        t0_fit = time.perf_counter()
        fit = robjects.r(
            "susie("
            f"X_py, y_py, L={int(l_effects)}, max_iter={int(max_iter)}, "
            f"coverage={float(coverage)}, min_abs_corr={float(min_abs_corr)}, "
            "estimate_residual_variance=TRUE, standardize=TRUE, intercept=TRUE, verbose=FALSE)"
        )
        fit_sec = time.perf_counter() - t0_fit
        n_iter = int(np.asarray(fit.rx2("niter"))[0])
        converged = bool(np.asarray(fit.rx2("converged"))[0])

    total_sec = time.perf_counter() - t0_total

    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "backend": backend,
        "gene_id": gene_id,
        "chrom": chrom,
        "center_pos": center,
        "window_start": int(start),
        "window_end": int(end),
        "half_window_bp": int(half_window_bp),
        "window_size_bp": int(end - start + 1),
        "n_samples": int(X.shape[0]),
        "n_markers_raw": n_markers_raw,
        "n_markers": n_markers,
        "n_markers_dropped_invariant": int(n_markers_dropped),
        "fit_runtime_sec": float(fit_sec),
        "total_runtime_sec": float(total_sec),
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gb": float(peak_rss_bytes / (1024**3)),
        "n_iter": n_iter,
        "converged": converged,
    }


def _parse_half_windows(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(int(float(tok) * 1_000_000))
    if not out:
        raise ValueError("No half-window sizes parsed")
    return sorted(set(out))


def _run_child(
    *,
    script_path: Path,
    backend: str,
    gene_id: str,
    chrom: int,
    center: int,
    half_window_bp: int,
    l_effects: int,
    max_iter: int,
    coverage: float,
    min_abs_corr: float,
    timeout_sec: int,
) -> dict:
    cmd = [
        sys.executable,
        str(script_path),
        "--single-run",
        "--backend",
        backend,
        "--gene-id",
        gene_id,
        "--chrom",
        str(chrom),
        "--center",
        str(center),
        "--half-window-bp",
        str(half_window_bp),
        "--l-effects",
        str(l_effects),
        "--max-iter",
        str(max_iter),
        "--coverage",
        str(coverage),
        "--min-abs-corr",
        str(min_abs_corr),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "half_window_bp": int(half_window_bp),
            "half_window_mb": float(half_window_bp / 1_000_000),
            "status": "timeout",
        }
    if proc.returncode != 0:
        return {
            "half_window_bp": int(half_window_bp),
            "half_window_mb": float(half_window_bp / 1_000_000),
            "status": "error",
            "stderr_tail": proc.stderr[-500:],
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "half_window_bp": int(half_window_bp),
            "half_window_mb": float(half_window_bp / 1_000_000),
            "status": "error",
            "stderr_tail": proc.stderr[-500:],
        }
    record = json.loads(lines[-1])
    record["status"] = "ok"
    record["half_window_mb"] = float(half_window_bp / 1_000_000)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--backend", default="pysusie", choices=["pysusie", "susieR"])
    parser.add_argument("--gene-id", default="Sobic.001G111500")
    parser.add_argument("--chrom", type=int)
    parser.add_argument("--center", type=int)
    parser.add_argument("--half-window-bp", type=int)
    parser.add_argument("--half-windows-mb", default="2,4,8,12,16,20,24,28,30,32,34,36")
    parser.add_argument("--l-effects", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--min-abs-corr", type=float, default=0.5)
    parser.add_argument("--max-runtime-sec", type=float, default=300.0)
    parser.add_argument("--max-memory-gb", type=float, default=10.0)
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    if args.single_run:
        if args.chrom is None or args.center is None or args.half_window_bp is None:
            raise SystemExit("--single-run requires --chrom, --center, and --half-window-bp")
        result = run_single(
            backend=str(args.backend),
            gene_id=args.gene_id,
            chrom=int(args.chrom),
            center=int(args.center),
            half_window_bp=int(args.half_window_bp),
            l_effects=int(args.l_effects),
            max_iter=int(args.max_iter),
            coverage=float(args.coverage),
            min_abs_corr=float(args.min_abs_corr),
        )
        print(json.dumps(result))
        return

    chrom, gene_start, gene_end = _load_gene_window(args.gene_id)
    center = (gene_start + gene_end) // 2

    requested = _parse_half_windows(args.half_windows_mb)
    chrom_min, chrom_max = _chrom_bounds(chrom)
    max_half_window = max(center - chrom_min, chrom_max - center)
    half_windows = [w for w in requested if w <= max_half_window]
    if max_half_window not in half_windows:
        half_windows.append(max_half_window)
    half_windows = sorted(set(half_windows))

    script_path = Path(__file__).resolve()
    records: list[dict] = []
    best: dict | None = None

    for half_window_bp in half_windows:
        rec = _run_child(
            script_path=script_path,
            backend=str(args.backend),
            gene_id=args.gene_id,
            chrom=chrom,
            center=center,
            half_window_bp=half_window_bp,
            l_effects=args.l_effects,
            max_iter=args.max_iter,
            coverage=args.coverage,
            min_abs_corr=args.min_abs_corr,
            timeout_sec=args.timeout_sec,
        )
        if rec.get("status") == "ok":
            rec["meets_runtime"] = rec["fit_runtime_sec"] < args.max_runtime_sec
            rec["meets_memory"] = rec["peak_rss_gb"] < args.max_memory_gb
            rec["meets_both"] = rec["meets_runtime"] and rec["meets_memory"]
            if rec["meets_both"]:
                best = rec
        else:
            rec["meets_runtime"] = False
            rec["meets_memory"] = False
            rec["meets_both"] = False
        records.append(rec)
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "half_window_mb": rec["half_window_mb"],
                    "status": rec["status"],
                    "n_markers": rec.get("n_markers"),
                    "fit_runtime_sec": rec.get("fit_runtime_sec"),
                    "peak_rss_gb": rec.get("peak_rss_gb"),
                    "meets_both": rec.get("meets_both"),
                }
            ),
            flush=True,
        )
        if (not args.full_sweep) and (not rec["meets_both"]):
            break

    out = {
        "backend": args.backend,
        "gene_id": args.gene_id,
        "chrom": chrom,
        "center_pos": int(center),
        "constraints": {
            "max_fit_runtime_sec": float(args.max_runtime_sec),
            "max_peak_rss_gb": float(args.max_memory_gb),
        },
        "params": {
            "l_effects": int(args.l_effects),
            "max_iter": int(args.max_iter),
            "coverage": float(args.coverage),
            "min_abs_corr": float(args.min_abs_corr),
        },
        "records": records,
        "best_window_meeting_constraints": best,
    }

    if args.output is None:
        output_path = Path(
            "/Users/jamesschnable/Projects/pySusie/pysusie/benchmarks/results/"
            f"phya_window_sweep_{args.backend.lower()}_results.json"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
