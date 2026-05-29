# PDE Spatial Stability and Non-Homogeneous Perturbation Tests

## Purpose

This analysis tests whether the spatial PDE merely preserves the homogeneous ODE compensation branch, or whether spatial structure can alter basin entry or produce spatially organized outcomes in the tested parameterization.

## Why Linear Stability Alone Is Not Enough

Linear mode stability checks infinitesimal perturbations around a homogeneous steady state. It cannot by itself rule out finite-amplitude heterogeneous initial conditions that cross basin boundaries. Therefore this analysis combines linear Neumann mode stability with targeted nonlinear PDE perturbation tests.

## Homogeneous Compensation Branch as a PDE Steady State

The homogeneous ODE compensation branch is also a homogeneous PDE steady-state branch because diffusion terms vanish for spatially constant fields. The tested branch uses `RoyEvoParams(b_u=0.08, b_v=0.02)` and the current diffusion coefficients `D_n=0.01`, `D_w=0.01`, and `D_q=0.005`.

## Linear Neumann Mode Stability

For each Neumann mode, the linearized PDE uses `J_F(U*) - lambda_mn D`. The current target stresses gave:

- s=0.11765625: max nonzero growth=-0.02376675, label=`linearly_spatially_stable`
- s=0.1584375: max nonzero growth=-0.01607895, label=`linearly_spatially_stable`
- s=0.16486816: max nonzero growth=-0.01483752, label=`linearly_spatially_stable`
- s=0.175: max nonzero growth=-0.01286663, label=`linearly_spatially_stable`

The full table is `results/roy_pde_compensation_spatial_modes_current.csv`, and the summary is `results/roy_pde_compensation_spatial_stability_summary.csv`.

## Diffusion-Ratio Linear Stability

The controlled diffusion-ratio grid varies only linear diffusion ratios and does not run nonlinear PDE simulations. Condition counts were:

- `linearly_spatially_stable`: 100
- `near_neutral_or_unresolved`: 0
- `linear_spatial_instability_detected`: 0

The grid is saved in `results/roy_pde_compensation_diffusion_ratio_grid.csv`.

## Non-Homogeneous Initial Conditions

The nonlinear PDE tests used localized predator patches, localized defense patches, sinusoidal spatial modes, smooth random heterogeneity, and basin-boundary heterogeneity. Each heterogeneous run was compared with a matched homogeneous PDE control at the same stress and baseline mean state.

## Nonlinear PDE Perturbation Results

Targeted nonlinear runs: `28`.

Runs with basin changes relative to matched homogeneous controls: `2`.

Runs with final persistent spatial pattern above threshold: `0`.

Transient or otherwise unresolved nonlinear classifications: `2`.

Basin-changing cases: `2`. Persistent-pattern cases: `0`.

The basin-changing cases were localized defense patches started from the basin-boundary mean state. They changed the matched homogeneous-control label from `persistent_basin` to `transient_basin` at the tested horizon, while final spatial CV values decayed below the persistence threshold. This is evidence for finite-amplitude sensitivity near boundary states, not evidence for a sustained spatial pattern.

## Final Label

`pde_homogeneous_branch_spatially_stable_but_finite_heterogeneity_affects_basin`

## Interpretation

Linear spatial modes are stable, but finite-amplitude heterogeneity changed the finite-horizon basin assignment in basin-boundary local-defense-patch cases. These changes were transient-label changes with final spatial CV decay, so they support a nuanced finite-amplitude basin-entry effect rather than persistent spatial-pattern-mediated rescue.

## Biological Meaning

In the tested setup, the spatially extended PDE is best interpreted as preserving the homogeneous reaction-level compensation mechanism unless the finite-amplitude tests document basin changes or persistent spatial patterns. This does not make the PDE irrelevant; it means the current spatial tests did not identify spatial patterning as the mechanism.

## What This Proves

This shows that the homogeneous compensation branch is linearly stable to the tested Neumann spatial modes under current diffusion coefficients, and it directly tests a targeted set of finite-amplitude heterogeneous initial fields.

## What This Does Not Prove

This is not a broad PDE parameter scan, not a proof for all diffusion coefficients, and not a theorem for all nonlinear trade-off forms. Transient basin-boundary cases remain lower-confidence than steady outcomes.

## Files

- `experiments/25_roy_pde_spatial_stability_and_nonhomogeneous_tests.py`
- `results/roy_pde_compensation_spatial_modes_current.csv`
- `results/roy_pde_compensation_spatial_stability_summary.csv`
- `results/roy_pde_compensation_diffusion_ratio_grid.csv`
- `results/roy_pde_nonhomogeneous_perturbation_summary.csv`
- `results/roy_pde_nonhomogeneous_mean_timeseries.csv`
- `results/roy_pde_nonhomogeneous_spatial_metrics.csv`
- `results/roy_pde_spatial_nonlinear_mechanism_decision.csv`
- `figures/roy_evo_spatial/report/fig45_pde_mode_growth_rates.png`
- `figures/roy_evo_spatial/report/fig46_pde_spatial_stability_along_branch.png`
- `figures/roy_evo_spatial/report/fig47_pde_diffusion_ratio_stability.png`
- `figures/roy_evo_spatial/report/fig48_pde_spatial_mode_schematic.png`
- `figures/roy_evo_spatial/report/fig49_nonhomogeneous_initial_conditions.png`
- `figures/roy_evo_spatial/report/fig50_nonhomogeneous_final_fields.png`
- `figures/roy_evo_spatial/report/fig51_nonhomogeneous_mean_timeseries.png`
- `figures/roy_evo_spatial/report/fig52_nonhomogeneous_spatial_metrics.png`
- `figures/roy_evo_spatial/report/fig53_pde_mechanism_decision.png`

## Next Step

Use these targeted results to decide whether any future PDE work should focus on finite-amplitude basin-boundary cases rather than broad spatial scans.

pde_homogeneous_branch_spatially_stable_but_finite_heterogeneity_affects_basin
