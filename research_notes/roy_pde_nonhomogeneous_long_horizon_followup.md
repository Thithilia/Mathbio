# Long-Horizon Follow-Up of Non-Homogeneous PDE Basin-Changing Cases

## Purpose

This note follows only the basin-changing non-homogeneous PDE cases from PR #23. It tests whether localized finite-amplitude defense heterogeneity caused persistent basin changes or only finite-horizon transient label changes.

## Cases Selected from PR #23

The selected cases are exactly rows in `results/roy_pde_nonhomogeneous_perturbation_summary.csv` with `basin_changed_relative_to_homogeneous_control == True`.

- `basin_boundary_state_local_defense_patch_s0.158438_seed20260702`: stress=0.158438, baseline=`basin_boundary_state`, perturbation=`local_defense_patch`, seed=20260702
- `basin_boundary_state_local_defense_patch_s0.164868_seed20260702`: stress=0.164868, baseline=`basin_boundary_state`, perturbation=`local_defense_patch`, seed=20260702

## Methods

The follow-up uses the same model equations, parameterization, diffusion coefficients, grid, timestep, baseline construction, and local-defense-patch construction as PR #23. It does not run broad PDE scans.

Each selected heterogeneous case is compared with a matched homogeneous control at the same stress, baseline mean state, seed convention, and horizon. The focused horizons are `1600; 3200; 6400`.

## Long-Horizon Mean Dynamics

At the longest focused horizon, the followed cases have the following status:

- `basin_boundary_state_local_defense_patch_s0.158438_seed20260702` at T=6400: classification=`persistent_steady`, basin=`persistent_basin`, control basin=`persistent_basin`, status=`resolved_to_control`
- `basin_boundary_state_local_defense_patch_s0.164868_seed20260702` at T=6400: classification=`persistent_steady`, basin=`persistent_basin`, control basin=`persistent_basin`, status=`resolved_to_control`

Detailed mean time series are saved in `results/roy_pde_nonhomogeneous_long_horizon_mean_timeseries.csv`.

## Spatial Pattern Decay

The maximum final CV across followed cases at the longest horizon is `1.82808e-14`. Final spatial pattern persistence is evaluated with the same `1e-3` final-CV threshold used in PR #23.

## Basin Resolution

At the longest focused horizon:

- resolved to homogeneous control: `2`
- persistent different basin without final spatial pattern: `0`
- persistent different basin with final spatial pattern: `0`
- unresolved: `0`

## Final Label

`nonhomogeneous_basin_changes_resolve_to_homogeneous_control`

## Interpretation

The PR #23 basin changes were finite-horizon transient effects near basin-boundary states rather than persistent spatial-pattern-mediated basin changes.

## Biological Meaning

The follow-up distinguishes finite-amplitude basin entry from persistent spatial patterning. A basin-label change without final spatial CV persistence is not evidence for a spatial-pattern-mediated rescue mechanism; it indicates finite-amplitude sensitivity near the tested basin boundary.

## What This Proves

This targeted follow-up tests whether the specific PR #23 basin-changing local-defense perturbations persist, resolve, or remain unresolved at longer horizons.

## What This Does Not Prove

This is not a broad PDE parameter scan, not a grid-convergence study, and not a proof for all heterogeneous initial conditions or diffusion coefficients.

## Files

- `experiments/26_roy_pde_nonhomogeneous_long_horizon_followup.py`
- `results/roy_pde_nonhomogeneous_long_horizon_summary.csv`
- `results/roy_pde_nonhomogeneous_long_horizon_mean_timeseries.csv`
- `results/roy_pde_nonhomogeneous_long_horizon_spatial_metrics.csv`
- `results/roy_pde_nonhomogeneous_long_horizon_decision.csv`
- `figures/roy_evo_spatial/report/fig54_long_horizon_mean_timeseries.png`
- `figures/roy_evo_spatial/report/fig55_long_horizon_spatial_metrics.png`
- `figures/roy_evo_spatial/report/fig56_long_horizon_final_fields.png`
- `figures/roy_evo_spatial/report/fig57_long_horizon_decision.png`

## Next Step

Use this result to decide whether any later work should refine local basin-boundary diagnostics rather than broaden PDE scanning.

nonhomogeneous_basin_changes_resolve_to_homogeneous_control
