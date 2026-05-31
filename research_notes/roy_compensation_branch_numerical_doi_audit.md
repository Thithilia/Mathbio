# Compensation Branch Numerical and DOI Audit

## Purpose

This note records audit support added during the final cleanup pass for the compensation-branch manuscript and supplement. The audit is intended to reduce drift between manuscript wording, source CSV outputs, and the reference bibliography before external review.

## Numerical Value Audit

The script `experiments/29_roy_manuscript_audit.py` reads existing CSV outputs and writes `results/roy_compensation_branch_value_audit.csv`. It checks the manuscript-reported values:

- \(q^*(0)\approx0.6726\)
- \(s_{\mathrm{fixed}}\approx0.0696\)
- verified rescue window \(0.0726\le s\le0.1656\)
- branch interval approximately \((-0.1131,0.2324)\)
- \(n^*\approx4.8333\)
- \(w^*\approx0.6417\)
- nonlinear shape-grid counts: 11 robust, 11 partial, 4 no-compensation, 1 unresolved
- sampled continuous-\(\lambda\) scan uses 600 values
- discrete Neumann mode range is \(i,j=0,\ldots,64\)

The script exits nonzero if any numerical check fails.

## DOI Audit

The same script writes `results/roy_compensation_branch_reference_doi_audit.csv` by parsing `manuscript/references.bib` and recording whether each entry has a DOI field. This is an offline field-presence check only. DOI resolver validation should still be completed manually or with network access before journal submission.

## Scope

The audit does not rerun simulations, alter model equations, or change manuscript claims. It only checks consistency between reported values and existing source files.

## Before Submission

- Re-run `python experiments/29_roy_manuscript_audit.py --profile focused` after any manuscript wording or source-CSV change.
- Confirm DOI resolver links manually before journal submission.
- Archive the repository and update the data/code statement only after a release DOI exists.
