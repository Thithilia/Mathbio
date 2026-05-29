# Research Note: Hysteresis and Basin Structure in the PDE-Evolution Model

## Executive Summary

Final Step 13 label: `pde_evo_bistability_mapped`.

At least one stress admits both persistent and extinct outcomes from different initial conditions.

A single scalar threshold is inappropriate because at least one stress has both persistent and extinct reachable outcomes.

## Why This Was Needed

PR #6 found `pde_evo_hysteresis_detected`: upward continuation kept a persistent branch to high stress, while downward continuation from high stress could remain extinct or transient at the same stresses.

Step 13 therefore asks which asymptotic outcomes are reachable at each stress and how they depend on initial condition or continuation path. It does not diagnose spatial covariance mechanism.

## Setup

- profile run: `focused`
- model: PDE-evo only
- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`
- grid: `64x64`, `L_x=L_y=20`, seed `20260702`
- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`
- integration: `dt=0.1`, `record_every=50`
- continuation horizon: `1200`
- basin scan horizon: `1600`
- basin stresses: `0.141262205, 0.15, 0.1584375, 0.16486816, 0.175`
- initial families: `baseline_burnin`, `persistent_branch`, `extinct_branch`, `low_predator`, `low_defense`, `high_defense`, `mixed_random`
- steady criteria: `abs(relative_change) < 0.02` and `normalized_residual < 0.0001`

## Continuation Hysteresis

| stress | upward classification | downward classification | direction-dependent? |
|---:|---|---|---|
| 0.125 | `persistent_steady` | `persistent_steady` | False |
| 0.135 | `persistent_steady` | `declining_transient` | True |
| 0.141262205 | `persistent_steady` | `persistent_steady` | False |
| 0.15 | `persistent_steady` | `persistent_steady` | False |
| 0.1584375 | `persistent_steady` | `recovery_transient` | True |
| 0.16486816 | `persistent_steady` | `extinct_steady` | True |
| 0.175 | `persistent_steady` | `extinct_steady` | True |

Direction-dependent continuation stresses: `4`.

## Initial-Condition Basin Scan

| stress | persistent basin count | extinct basin count | transient/unresolved count | regime label |
|---:|---:|---:|---:|---|
| 0.141262205 | 2 | 0 | 5 | `persistent_transient_mixed` |
| 0.15 | 3 | 0 | 4 | `persistent_transient_mixed` |
| 0.1584375 | 3 | 1 | 3 | `bistable_persistent_extinct` |
| 0.16486816 | 2 | 3 | 2 | `bistable_persistent_extinct` |
| 0.175 | 2 | 5 | 0 | `bistable_persistent_extinct` |

## Stress-Regime Map

0.141262205: `persistent_transient_mixed`, 0.15: `persistent_transient_mixed`, 0.1584375: `bistable_persistent_extinct`, 0.16486816: `bistable_persistent_extinct`, 0.175: `bistable_persistent_extinct`

## Interpretation

A single scalar threshold is inappropriate because at least one stress has both persistent and extinct reachable outcomes.

Mechanism interpretation remains out of scope until the reachable outcome regimes are better resolved.

## Files

- `results/roy_pde_evo_hysteresis_map.csv`
- `results/roy_pde_evo_basin_initial_condition_scan.csv`

## Next Step

Next: quantify basin boundaries within the mapped bistable interval.

pde_evo_bistability_mapped
