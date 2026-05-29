# Representative PDE Solutions for the Roy Eco-Evolutionary Spatial Model

## Purpose

This note addresses a missing evidence requirement from the verification plan: actual PDE solution fields, spatial mean time series, and residual diagnostics for representative outcomes. The run is targeted and reuses existing basin-boundary scan rows; it is not a parameter scan and does not change the model equations.

## Selected Cases

| case | stress | q0 | w0_scale | prior classification | rerun classification | prior basin | rerun basin |
|---|---:|---:|---:|---|---|---|---|
| `persistent_case` | 0.1584375 | 0.1 | 0.1 | `persistent_steady` | `persistent_steady` | `persistent_basin` | `persistent_basin` |
| `extinct_case` | 0.16486816 | 0.9 | 0.02 | `extinct_steady` | `extinct_steady` | `extinct_basin` | `extinct_basin` |
| `transient_case` | 0.1584375 | 0.7 | 0.02 | `recovery_transient` | `recovery_transient` | `transient_basin` | `transient_basin` |

## Numerical Setup

The script was run with profile `focused`. The focused profile uses `T = 1600`, `n_x = n_y = 64`, `L_x = L_y = 20`, `D_n = 0.01`, `D_w = 0.01`, `D_q = 0.005`, `dt = 0.1`, seed `20260702`, and `RoyEvoParams(b_u=0.08, b_v=0.02)`. Initial means are constructed from the Step 09A burn-in baseline using the selected `q0` and `w0_scale` values.

Snapshots were saved at `t = 0`, `0.25T`, `0.50T`, `0.75T`, and `T` for `n(x,y,t)`, `w(x,y,t)`, and `q(x,y,t)`.

## Spatial Snapshots

The final spatial fields are shown in `figures/roy_evo_spatial/report/fig19_pde_solution_snapshots.png`. The full snapshot arrays are saved as compressed `.npz` archives in `results/`.

## Spatial Mean Dynamics

The spatial mean time series are saved in `results/roy_pde_evo_representative_mean_timeseries.csv` and plotted in `figures/roy_evo_spatial/report/fig20_pde_mean_timeseries.png`.

## Residual and Convergence Diagnostics

| case | tail_mean_w | relative_change_last_windows | normalized_residual | note |
|---|---:|---:|---:|---|
| `persistent_case` | 0.641667 | -3.96589e-06 | 7.70711e-13 | classification_matches_prior;basin_label_matches_prior |
| `extinct_case` | 8.98108e-12 | 4.56144 | 1.3098e-09 | classification_matches_prior;basin_label_matches_prior |
| `transient_case` | 0.450854 | 642.659 | 5.4867e-08 | classification_matches_prior;basin_label_matches_prior |

For near-extinct trajectories, the relative change between final windows can be numerically large because both windows have predator abundance close to zero. The extinction interpretation therefore uses tail abundance together with the residual check rather than relative change alone.

## Interpretation

The representative trajectories support the interpretation that the spatial PDE can realize different outcome classes from different initial states in the tested parameterization. This is direct PDE solution evidence, but it remains case-specific and does not prove general bistability.

The representative transient case remains classified as transient in the rerun, so it should not be interpreted as an asymptotic steady outcome.

If a rerun classification or basin label differs from the prior basin-boundary scan, the difference is retained in the summary table rather than hidden.

## Limitations

This analysis reruns only three representative cases from the existing basin-boundary scan. It does not test robustness to grid size, time step, diffusion coefficients, trade-off parameters, perturbation seeds, or longer horizons. Transient-heavy regions still require targeted follow-up.

## Files

- `experiments/19_roy_pde_evo_representative_solutions.py`
- `results/roy_pde_evo_representative_solution_summary.csv`
- `results/roy_pde_evo_representative_mean_timeseries.csv`
- `results/roy_pde_evo_representative_fields_persistent_case.npz`
- `results/roy_pde_evo_representative_fields_extinct_case.npz`
- `results/roy_pde_evo_representative_fields_transient_case.npz`
- `figures/roy_evo_spatial/report/fig19_pde_solution_snapshots.png`
- `figures/roy_evo_spatial/report/fig20_pde_mean_timeseries.png`
- `figures/roy_evo_spatial/report/fig21_pde_solution_residuals.png`

## Next Step

Use these representative fields to guide adaptive q0-w0 basin-boundary refinement and longer-horizon checks for transient grid points.
