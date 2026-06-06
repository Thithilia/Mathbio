# Compensation-Branch Manuscript Reproducibility Checklist

## Purpose

This note records the review-facing reproducibility additions for the compensation-branch manuscript package. It is a reproducibility note, not a new model analysis.

## Snapshot and Environment Note

The pre-external-review manuscript snapshot is identified by tag `external-review-2026-06-06`; the exact snapshot commit is the tag target recorded in Git metadata. The current package was generated in a local Windows/PowerShell environment with Python and LaTeX tools on the system path. Python package versions were captured with `python -m pip freeze` in `environment/requirements-2026-06-06.txt`. The LaTeX PDFs were built with the local LaTeX installation. This is an environment freeze file, not a containerized reproducibility package.

## Regeneration Commands

Run the following from the repository root:

```powershell
python -m pip freeze > environment/requirements-2026-06-06.txt
python experiments/28_roy_counterfactual_and_spatial_evidence.py --profile focused
python experiments/30_roy_ecological_review_sensitivities.py --profile focused
python experiments/31_roy_mathematical_classification_tightening.py --profile focused
python experiments/32_roy_spatial_nonlinear_robustness_clarification.py --profile focused
python experiments/29_roy_manuscript_audit.py --profile focused
latexmk -pdf -outdir=manuscript -interaction=nonstopmode -halt-on-error manuscript/roy_compensation_branch_rescue_manuscript.tex
latexmk -pdf -outdir=manuscript -interaction=nonstopmode -halt-on-error manuscript/roy_compensation_branch_supplement.tex
git diff --check
```

## New Robustness Clarification Outputs

- `results/roy_pde_compensation_modal_routh_hurwitz_scan.csv`
- `results/roy_pde_compensation_modal_routh_hurwitz_summary.csv`
- `figures/roy_evo_spatial/report/fig72_pde_modal_routh_hurwitz_margins.png`

These outputs evaluate Routh-Hurwitz coefficient margins for the sampled modal matrix `J_F(U*) - lambda D`. The calculation is semi-analytic linear algebra over a dense sampled lambda grid; it does not run PDE simulations.

## Archival TODO

Before submission, prepare a Zenodo or OSF archive containing:

- repository source at the final manuscript commit;
- generated CSV outputs under `results/`;
- manuscript figures under `figures/roy_evo_spatial/report/`;
- main manuscript and supplement PDFs;
- `environment/requirements-2026-06-06.txt`;
- a short README identifying the exact regeneration command sequence.

The archive should state that the PDE spatial evidence is targeted and sampled, not a global proof over all domains, diffusion closures, or finite-amplitude heterogeneous initial conditions.

## Deferred Diagnostics

No new finite-amplitude PDE robustness reruns were added in this PR. Time-step, grid-resolution, and larger-domain PDE checks remain useful future diagnostics, but they would add simulations beyond the text and semi-analytic clarification scope of this follow-up.
