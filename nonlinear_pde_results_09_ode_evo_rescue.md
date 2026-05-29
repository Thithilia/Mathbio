# Nonlinear PDE Results 09: ODE Eco-Evolutionary Rescue Gate

**Final ODE label: `ODE_indirect_evolutionary_rescue_supported`.** Evolution increases the threshold, produces a persistence window, and lowers q after stress.

## Direct Answer

Prey defense evolution increases the predator mortality threshold in the well-mixed ODE for `interior_low_conversion_tradeoff`.

## Equations Implemented

`z = 1/kappa - n - w`

`r(q) = r_u * (1 - q) + r_v * q`

`a(q) = a_u * (1 - q) + a_v * q`

`b(q) = b_u * (1 - q) + b_v * q`

`dn/dt = n * (r(q) * z - xi - a(q) * w)`

`dw/dt = w * (b(q) * n * z - (m + stress) - mu * w)`

`dq/dt = nu * q * (1 - q) * ((r_v - r_u) * z - (a_v - a_u) * w)`

When `evolve=False`, `dq/dt = 0` and q is frozen at the baseline value.

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

The source dataclass keeps the setup-note defaults. This Step 09A run uses a single documented adjusted conversion tradeoff with `b_u=0.08` and `b_v=0.02`. The reason is to place the unstressed evolving baseline at an interior defense frequency, making the ODE gate test a genuine q-response problem rather than a boundary-pinned q=1 control.

## Initial Condition

Burn-in method: integrate the unstressed evolving ODE from `[n0, w0, q0] = [1.0, 0.2, 0.5]` to `T=3000`.

Baseline state used for stress tests: `n=4.8333333`, `w=0.64166667`, `q=0.67261474`, `z=1.1916667`.

Burn-in residual norm: `7.52822e-13`.

## Threshold Results

- no-evolution threshold: `0.069448242`
- evolution threshold: `0.16486816`
- Delta m_c_evo: `0.095419922`
- threshold tolerance: `1e-05`

| condition | bracket low | bracket high | threshold | persistent low | persistent high | q tail at low | q change at low |
|---|---:|---:|---:|---|---|---:|---:|
| no evolution | 0.069443359 | 0.069453125 | 0.069448242 | True | False | 0.67261474 | 3.3306691e-16 |
| evolution | 0.16486328 | 0.16487305 | 0.16486816 | True | False | 8.020852e-06 | -0.67260672 |

## Rescue Window Check

Stress between thresholds: `0.1171582`.

- no-evolution persistent: `False`; tail mean w = `-4.1710647e-15`
- evolution persistent: `True`; tail mean w = `0.64166667`
- baseline q: `0.67261474`
- evolving q tail mean in window: `0.33359896`
- q change in window: `-0.33901578`

## Interpretation

The ODE rescue claim is accepted only when the evolution threshold exceeds the no-evolution threshold by more than tolerance, a stress window exists where no-evolution goes extinct but evolution persists, and q decreases relative to the baseline in that window.

Evolution increases the threshold, produces a persistence window, and lowers q after stress.

## Outputs

- `results\roy_evo_ode_threshold_scan.csv`
- `results\roy_evo_ode_timeseries.csv`

## Next Step

Step 09B should implement the spatial PDE comparison using this ODE-supported tradeoff, with no-flux boundaries and the same threshold and q-response diagnostics.
