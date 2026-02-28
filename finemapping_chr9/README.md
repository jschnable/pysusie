# Fine-Mapping Analysis: Chromosome 9 QTL (62.6 Mb)

## Overview

This folder contains inputs for fine-mapping the flowering time QTL on chromosome 9
at approximately 62.6 Mb using pysusie. The data comes from the 2021 High-N environment
GWAS (n=827 individuals).

## Locus Details

- **Chromosome**: 9
- **Peak position**: 62,649,644 bp
- **Locus window**: 62,149,731 - 63,149,323 bp (~1 Mb total)
- **Number of SNPs**: 4,719
- **Sample size**: 827
- **Phenotype variance**: 40.10

## Top Association

- **SNP**: 9:62581725:T:C
- **Position**: 62,581,725 bp
- **Effect**: 1.78 days
- **P-value**: 1.03e-14
- **Z-score**: 7.88

## Candidate Genes in Locus

Based on LD analysis, this locus contains:
- **sbi-MIR172a**: microRNA regulating flowering via the aging pathway (~40 kb from peak)
- **SbELF3** (Sobic.009G257300): EARLY FLOWERING 3, circadian clock component (~320 kb from peak)

## Files

| File | Description |
|------|-------------|
| `ld_matrix.npy` | Correlation matrix R (4719 x 4719), symmetric, diagonal=1 |
| `z_scores.npy` | Z-scores for each SNP (n=4719), order matches LD matrix |
| `bhat.npy` | Effect size estimates (beta-hat) |
| `shat.npy` | Standard errors of effect sizes |
| `snp_order.csv` | SNP identifiers and positions, in same order as LD matrix |
| `locus_summary_stats.csv` | Full summary statistics for locus |
| `locus_metadata.json` | Metadata including sample size, variance, etc. |

## QC Notes

1. **Genome build**: Sorghum bicolor v5.1
2. **Allele coding**: REF/ALT from VCF, genotypes coded as 0/1/2 (dosage)
3. **MAF filter**: >0.025 (applied during GWAS)
4. **LD computed from**: Same 827 individuals used in GWAS (in-sample LD)
5. **Missing genotypes**: Imputed to mean before LD calculation

## Usage with pysusie

```python
import numpy as np
import json
from pysusie import SuSiE

# Load inputs
R = np.load('ld_matrix.npy')
z = np.load('z_scores.npy')

with open('locus_metadata.json') as f:
    meta = json.load(f)

n = meta['n_samples']
var_y = meta['var_y']

# Fit SuSiE
m = SuSiE(n_effects=10, refine=True, max_iter=200, tol=1e-4)
m.fit_from_summary_stats(z=z, R=R, n=n, var_y=var_y, regularize_ld="auto")

# Get results
res = m.result_

# Check convergence
print(f"Converged: {res.converged}")
print(f"Final ELBO: {res.elbo}")

# Get credible sets
cs = res.get_credible_sets(R=R, coverage=0.95, min_abs_corr=0.5)
print(f"Number of credible sets: {len(cs)}")

# Summary table
tbl = res.summary(R=R)
print(tbl)
```

## Validation Checklist

- [ ] `res.converged` is True
- [ ] `res.elbo` is non-decreasing or stable
- [ ] CS sizes are reasonable (<50 SNPs)
- [ ] Purity is acceptable (>0.5)
- [ ] Top PIP SNPs are near candidate genes

## Sensitivity Tests to Run

1. Vary `n_effects` (5, 10, 15, 20)
2. Compare `regularize_ld=0` vs `"auto"`
3. Run with/without `refine=True`
4. Check if results stable with different random seeds

## Citation

If using this data, please cite:
- Original GWAS: [your paper]
- pysusie: Wang et al. (SuSiE methodology)
- Sorghum genome: McCormick et al.

## Date Generated

2024-02-16
