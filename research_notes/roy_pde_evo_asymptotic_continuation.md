# Research Note: Asymptotic and Continuation Check for PDE-Evolution Persistence

## Executive Summary

Final Step 12 label: `pde_evo_hysteresis_detected`.

Upward and downward continuation sweeps give different classifications at one or more stresses.

## Why This Was Needed

PR #5 found the PDE-evo persistence boundary unresolved under multi-horizon classification. Step 12 checks whether the unresolved boundary reflects long transients, hysteresis or multiple attractors, nonmonotone stress response, or unresolved asymptotics.

This task does not diagnose spatial suppression mechanism and does not change the model equations or previous PR conclusions.

## Setup

- profile run: `full`
- model: PDE-evo only
- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`
- baseline: unstressed Step 09A burn-in via `find_evo_equilibrium`
- grid: `64x64`, `L_x=L_y=20`, seed `20260702`
- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`
- integration: `dt=0.1`, `record_every=50`
- long-horizon stresses: `0.135, 0.141262205, 0.15, 0.1584375, 0.16486816, 0.175`
- long horizons run: `1600, 2400, 3200`
- continuation grid: `0.09, 0.105, 0.11765625, 0.125, 0.135, 0.141262205, 0.15, 0.1584375, 0.16486816, 0.175`
- continuation step horizon: `1200`
- steady criteria: `abs(relative_change) < 0.02` and `normalized_residual < 0.0001`

## Long-Horizon Results

| stress | T1600 classification | T2400 classification | T3200 classification | latest tail mean w | normalized residual | conclusion |
|---:|---|---|---|---:|---:|---|
| 0.135 | `persistent_steady` | `persistent_steady` | `persistent_steady` | 0.64166667 | 8.8660578e-16 | `persistent_steady` |
| 0.141262205 | `declining_transient` | `persistent_steady` | `persistent_steady` | 0.64166667 | 8.8690187e-16 | `persistent_steady` |
| 0.15 | `declining_transient` | `declining_transient` | `persistent_steady` | 0.64166667 | 8.8642712e-16 | `persistent_steady` |
| 0.1584375 | `recovery_transient` | `declining_transient` | `declining_transient` | 0.64158821 | 4.7088651e-11 | `declining_transient` |
| 0.16486816 | `recovery_transient` | `recovery_transient` | `recovery_transient` | 0.95440951 | 7.3135139e-11 | `recovery_transient` |
| 0.175 | `extinct_steady` | `extinct_steady` | `extinct_steady` | 5.6466274e-14 | 1.3513734e-16 | `extinct_steady` |

## Continuation Sweep Results

| stress | upward classification | downward classification | same? | note |
|---:|---|---|---|---|
| 0.09 | `persistent_steady` | `persistent_steady` | True | same |
| 0.105 | `persistent_steady` | `persistent_steady` | True | same |
| 0.11765625 | `persistent_steady` | `persistent_steady` | True | same |
| 0.125 | `persistent_steady` | `persistent_steady` | True | same |
| 0.135 | `persistent_steady` | `declining_transient` | False | direction-dependent |
| 0.141262205 | `persistent_steady` | `persistent_steady` | True | same |
| 0.15 | `persistent_steady` | `persistent_steady` | True | same |
| 0.1584375 | `persistent_steady` | `recovery_transient` | False | direction-dependent |
| 0.16486816 | `persistent_steady` | `extinct_steady` | False | direction-dependent |
| 0.175 | `persistent_steady` | `extinct_steady` | False | direction-dependent |

## Hysteresis Check

- direction-dependent stresses: `4`
- mismatches: `0.135: up=persistent_steady, down=declining_transient; 0.1584375: up=persistent_steady, down=recovery_transient; 0.16486816: up=persistent_steady, down=extinct_steady; 0.175: up=persistent_steady, down=extinct_steady`

## Boundary Diagnosis

- transient or non-steady long-horizon stresses at `T=3200`: `2` of `6`
- final label: `pde_evo_hysteresis_detected`

Upward and downward continuation sweeps give different classifications at one or more stresses.

## Implication for Threshold Claims

Spatial suppression mechanism should not be interpreted unless the asymptotic or continuation boundary is stable.

## Files

- `results/roy_pde_evo_long_horizon.csv`
- `results/roy_pde_evo_continuation.csv`

## Next Step

Next: map continuation-dependent regimes before making threshold claims.

pde_evo_hysteresis_detected
