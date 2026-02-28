#!/usr/bin/env python3
"""
Fine-mapping analysis for chr9 flowering time QTL using pysusie.

This script runs SuSiE fine-mapping and performs validation checks
as recommended in the pysusie documentation.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set working directory
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_inputs():
    """Load fine-mapping inputs."""
    print("="*60)
    print("Loading fine-mapping inputs")
    print("="*60)

    R = np.load(SCRIPT_DIR / 'ld_matrix.npy')
    z = np.load(SCRIPT_DIR / 'z_scores.npy')
    bhat = np.load(SCRIPT_DIR / 'bhat.npy')
    shat = np.load(SCRIPT_DIR / 'shat.npy')
    snp_order = pd.read_csv(SCRIPT_DIR / 'snp_order.csv')

    with open(SCRIPT_DIR / 'locus_metadata.json') as f:
        meta = json.load(f)

    print(f"LD matrix shape: {R.shape}")
    print(f"Number of SNPs: {len(z)}")
    print(f"Sample size (n): {meta['n_samples']}")
    print(f"Phenotype variance: {meta['var_y']:.4f}")
    print(f"Top SNP: {meta['top_snp']} (Z={meta['top_snp_z']:.2f})")

    return R, z, bhat, shat, snp_order, meta


def check_ld_matrix(R):
    """Validate LD matrix properties."""
    print("\n" + "="*60)
    print("Validating LD matrix")
    print("="*60)

    # Check diagonal
    diag = np.diag(R)
    print(f"Diagonal range: {diag.min():.6f} to {diag.max():.6f}")

    # Check symmetry
    is_symmetric = np.allclose(R, R.T)
    print(f"Symmetric: {is_symmetric}")

    # Check positive semi-definite
    eigvals = np.linalg.eigvalsh(R)
    n_negative = np.sum(eigvals < -1e-10)
    min_eigval = eigvals.min()
    print(f"Min eigenvalue: {min_eigval:.6f}")
    print(f"Negative eigenvalues: {n_negative}")

    if n_negative > 0:
        print("WARNING: LD matrix has negative eigenvalues - may need regularization")

    return n_negative == 0


def run_susie(z, R, n, var_y, n_effects=10, regularize_ld="auto", refine=True):
    """Run SuSiE fine-mapping."""
    try:
        from pysusie import SuSiE
    except ImportError:
        print("\nERROR: pysusie not installed. Install with:")
        print("  pip install pysusie")
        return None

    print(f"\n{'='*60}")
    print(f"Running SuSiE (n_effects={n_effects}, regularize_ld={regularize_ld}, refine={refine})")
    print("="*60)

    m = SuSiE(n_effects=n_effects, refine=refine, max_iter=200, tol=1e-4)
    m.fit_from_summary_stats(z=z, R=R, n=n, var_y=var_y, regularize_ld=regularize_ld)

    res = m.result_

    # Validation
    print(f"\nConverged: {res.converged}")
    final_elbo = float(res.elbo[-1]) if len(res.elbo) else float("nan")
    print(f"Iterations: {res.n_iter}")
    print(f"Final ELBO: {final_elbo:.2f}")

    return res


def analyze_results(res, R, snp_order, meta):
    """Analyze and report SuSiE results."""
    print("\n" + "="*60)
    print("Analyzing results")
    print("="*60)

    # Get credible sets
    cs = res.get_credible_sets(R=R, coverage=0.95, min_abs_corr=0.5)
    print(f"\nNumber of credible sets: {len(cs)}")

    # Get PIPs
    pips = res.pip

    # Top SNPs by PIP
    top_idx = np.argsort(pips)[::-1][:20]

    print("\nTop 20 SNPs by PIP:")
    print("-"*80)
    print(f"{'Rank':<6}{'SNP':<25}{'Position':<15}{'PIP':<10}{'Z-score':<10}{'P-value':<12}")
    print("-"*80)

    for rank, idx in enumerate(top_idx, 1):
        snp_info = snp_order.iloc[idx]
        print(f"{rank:<6}{snp_info['SNP']:<25}{snp_info['POS']:<15,}{pips[idx]:<10.4f}{snp_info['Z']:<10.2f}{snp_info['P']:<12.2e}")

    # Credible set details
    if len(cs) > 0:
        print(f"\nCredible Set Details:")
        print("-"*60)
        for i, cs_obj in enumerate(cs):
            cs_indices = np.asarray(cs_obj.variables, dtype=int)
            cs_pips = pips[cs_indices]
            cs_snps = snp_order.iloc[cs_indices]

            if cs_obj.purity is not None:
                purity = cs_obj.purity.min_abs_corr
            else:
                cs_R = R[np.ix_(cs_indices, cs_indices)]
                purity = float(np.min(np.abs(cs_R)))

            print(f"\nCS {i+1}: {len(cs_indices)} SNPs, purity={purity:.3f}, coverage={cs_obj.coverage:.3f}")
            print(f"  Position range: {cs_snps['POS'].min():,} - {cs_snps['POS'].max():,}")
            print(f"  Total PIP: {cs_pips.sum():.3f}")
            print(f"  Effect index: {cs_obj.effect_index}, logBF={cs_obj.log_bayes_factor:.3f}")

            # Top SNP in CS
            top_in_cs = cs_indices[np.argmax(cs_pips)]
            top_snp = snp_order.iloc[top_in_cs]
            print(f"  Top SNP: {top_snp['SNP']} (PIP={pips[top_in_cs]:.3f})")

    return cs, pips


def sensitivity_analysis(z, R, n, var_y, snp_order):
    """Run sensitivity tests."""
    print("\n" + "="*60)
    print("Sensitivity Analysis")
    print("="*60)

    try:
        from pysusie import SuSiE
    except ImportError:
        print("Skipping - pysusie not installed")
        return

    results = []

    # Test different n_effects
    for n_eff in [5, 10, 15]:
        m = SuSiE(n_effects=n_eff, refine=True, max_iter=200)
        m.fit_from_summary_stats(z=z, R=R, n=n, var_y=var_y, regularize_ld="auto")
        res = m.result_
        cs = res.get_credible_sets(R=R, coverage=0.95, min_abs_corr=0.5)

        results.append({
            'n_effects': n_eff,
            'regularize_ld': 'auto',
            'refine': True,
            'converged': res.converged,
            'n_cs': len(cs),
            'top_pip': res.pip.max()
        })

    # Test regularize_ld=0
    m = SuSiE(n_effects=10, refine=True, max_iter=200)
    m.fit_from_summary_stats(z=z, R=R, n=n, var_y=var_y, regularize_ld=0)
    res = m.result_
    cs = res.get_credible_sets(R=R, coverage=0.95, min_abs_corr=0.5)
    results.append({
        'n_effects': 10,
        'regularize_ld': 0,
        'refine': True,
        'converged': res.converged,
        'n_cs': len(cs),
        'top_pip': res.pip.max()
    })

    # Test refine=False
    m = SuSiE(n_effects=10, refine=False, max_iter=200)
    m.fit_from_summary_stats(z=z, R=R, n=n, var_y=var_y, regularize_ld="auto")
    res = m.result_
    cs = res.get_credible_sets(R=R, coverage=0.95, min_abs_corr=0.5)
    results.append({
        'n_effects': 10,
        'regularize_ld': 'auto',
        'refine': False,
        'converged': res.converged,
        'n_cs': len(cs),
        'top_pip': res.pip.max()
    })

    # Report
    print("\nSensitivity test results:")
    print("-"*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    return results_df


def save_results(res, cs, pips, snp_order, output_dir):
    """Save fine-mapping results."""
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)

    # Save PIPs
    results_df = snp_order.copy()
    results_df['PIP'] = pips
    results_df.to_csv(output_dir / 'finemapping_results.csv', index=False)
    print(f"Saved PIPs to finemapping_results.csv")

    # Save credible sets
    cs_data = []
    for i, cs_obj in enumerate(cs):
        cs_indices = np.asarray(cs_obj.variables, dtype=int)
        for idx in cs_indices:
            cs_data.append({
                'CS': i + 1,
                'effect_index': int(cs_obj.effect_index),
                'coverage': float(cs_obj.coverage),
                'log_bayes_factor': float(cs_obj.log_bayes_factor),
                'SNP': snp_order.iloc[idx]['SNP'],
                'POS': snp_order.iloc[idx]['POS'],
                'PIP': pips[idx]
            })

    if cs_data:
        cs_df = pd.DataFrame(cs_data)
        cs_df.to_csv(output_dir / 'credible_sets.csv', index=False)
        print(f"Saved credible sets to credible_sets.csv")

    # Save numpy arrays
    np.save(output_dir / 'pip.npy', pips)
    print(f"Saved PIPs to pip.npy")


def main():
    """Main fine-mapping workflow."""
    print("\n" + "="*60)
    print("FINE-MAPPING ANALYSIS: Chr9 Flowering Time QTL")
    print("="*60)

    # Load inputs
    R, z, bhat, shat, snp_order, meta = load_inputs()
    n = meta['n_samples']
    var_y = meta['var_y']

    # Validate LD matrix
    ld_ok = check_ld_matrix(R)

    # Run SuSiE
    res = run_susie(z, R, n, var_y, n_effects=10, regularize_ld="auto", refine=True)

    if res is None:
        print("\nFine-mapping could not be run. Please install pysusie.")
        return

    # Analyze results
    cs, pips = analyze_results(res, R, snp_order, meta)

    # Save results
    save_results(res, cs, pips, snp_order, OUTPUT_DIR)

    # Sensitivity analysis
    sens_results = sensitivity_analysis(z, R, n, var_y, snp_order)
    if sens_results is not None:
        sens_results.to_csv(OUTPUT_DIR / 'sensitivity_results.csv', index=False)

    print("\n" + "="*60)
    print("FINE-MAPPING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
