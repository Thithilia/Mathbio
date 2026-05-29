# Nonlinear PDE Results 10: Spatial Eco-Evolutionary Threshold Gate

**Final Step 09B label: `spatially_suppressed_indirect_evolutionary_rescue`.** PDE evolution thresholds were below the ODE evolution threshold beyond tolerance.

- profile: `focused`
- tradeoff: `interior_low_conversion_tradeoff`

## Equations Implemented

`partial_t n = D_n Laplacian(n) + n * (r(q) * z - xi - a(q) * w)`

`partial_t w = D_w Laplacian(w) + w * (b(q) * n * z - (m + stress) - mu * w)`

`partial_t q = D_q Laplacian(q) + nu * q * (1 - q) * ((r_v - r_u) * z - (a_v - a_u) * w)`

`z = 1/kappa - n - w`; zero-flux Neumann boundaries are used.

## Parameter Values

| parameter | value |
|---|---:|
| `kappa` | 0.15 |
| `xi` | 0.55 |
| `r_u` | 1 |
| `r_v` | 0.65 |
| `a_u` | 1 |
| `a_v` | 0.35 |
| `b_u` | 0.08 |
| `b_v` | 0.02 |
| `m` | 0.1 |
| `mu` | 0.2 |
| `nu` | 0.05 |

Initial state came from the Step 09A unstressed evolving burn-in: `n=4.8333333`, `w=0.64166667`, `q=0.67261474`.

## Threshold Summary

- ODE no-evolution threshold: `0.069448242`
- ODE evolution threshold: `0.16486816`
- representative Stage B PDE no-evolution threshold: `0.06921875`
- representative Stage B PDE evolution threshold: `0.11765625`
- Delta_evo_ODE: `0.095419922`
- Delta_evo_PDE mean: `0.0484375`; range `[0.0484375, 0.0484375]`
- Delta_spatial_no_evo mean: `-0.00022949219`; interval `[-0.00072949219, 0.00027050781]`
- Delta_spatial_evo mean: `-0.047211914`; interval `[-0.047711914, -0.046711914]`

## q Response and Mechanism

- Stage B PDE evolution q tail mean near threshold: `0.31440243`
- Stage B PDE evolution q change near threshold: `-0.35821231`
- Stage B spatial covariance bonus near PDE evolution threshold: `-6.2847038e-14`
- Stage C: ran; Delta_spatial_evo = -0.0064306641
- q clipping in Stage B threshold-low endpoints: max count `0`, max violation `0`

The mechanism diagnostic is `spatial_growth - meanfield_growth`, where `spatial_growth = mean(w * A(n,w,q,z))` and `A = b(q) * n * z - (m + stress) - mu * w`.

## Interpretation

PDE evolution thresholds were below the ODE evolution threshold beyond tolerance. Spatial covariance diagnostics were near zero, so the result does not support spatial amplification through a positive covariance mechanism.

## Outputs

- `results\roy_evo_spatial_threshold_comparison.csv`
- `results\roy_evo_spatial_timeseries.csv`
- `results\roy_evo_spatial_mechanism.csv`

Next: no spatial amplification; write ODE-only indirect rescue result
