# Research Note: Stabilizing the PDE-Evolution Persistence Criterion

## Executive Summary

Final Step 11 label: `pde_evo_persistence_unresolved`.

The multi-horizon statuses do not form a clean stable boundary.

Spatial suppression mechanism should not be interpreted because the persistence boundary remains unresolved.

## Why This Was Needed

PR #4 found that the apparent PDE-evo threshold depended strongly on tail fraction, time horizon, and the tail-slope rule. It also found re-entry-like behavior in the focused stress grid.

That makes spatial suppression mechanism interpretation premature. Step 11 therefore stabilizes the persistence classification itself before any mechanism diagnosis is attempted.

## Setup

- profile run: `full`
- model: PDE-evo only
- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`
- initial state: unstressed Step 09A burn-in via `find_evo_equilibrium`
- grid: `64x64`, `L_x=L_y=20`, seed `20260702`
- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`
- integration: `dt=0.1`, `record_every=50`
- stress grid: `0.09, 0.105, 0.11765625, 0.125, 0.135, 0.141262205, 0.15, 0.1584375, 0.16486816, 0.175`
- horizons: `500, 800, 1200, 1600`
- tail fractions: `0.25`, `0.35`, `0.50`

## Classification Framework

For each horizon and tail fraction, Step 11 computes tail mean predator density, tail minimum predator density, tail slope, the tail-slope floor, tail mean q, q change, var(q), minimum free space, physicality, slope-rule persistence, and no-slope persistence.

The per-horizon classes are:

- `persistent_like`: positive predator tail that passes the density checks without an active declining-transient rejection.
- `extinct_like`: predator tail remains below epsilon and does not pass the no-slope persistence rule.
- `declining_transient`: predator density is still positive, but the slope rule rejects it because the tail is declining too fast.
- `recovery_transient`: predator density passes the slope rule and increases substantially relative to the previous shorter horizon.
- `indeterminate`: any ambiguous case.

The cross-horizon classes are `persistent_confirmed`, `extinct_confirmed`, `declining_transient`, `recovery_transient`, and `indeterminate`. This is better than one hard tail-slope threshold because it separates stable tails from long declining or recovery transients and makes horizon sensitivity explicit.

## Stress-by-Stress Results

| stress | final status | latest tail mean w | horizon sequence | q behavior | note |
|---:|---|---:|---|---|---|
| 0.09 | `persistent_confirmed` | 0.64166667 | `T500:persistent_like;T800:persistent_like;T1200:persistent_like;T1600:persistent_like` | stable persistent tail | `tail0.25=persistent_confirmed,tail0.35=persistent_confirmed,tail0.50=persistent_confirmed` |
| 0.105 | `persistent_confirmed` | 0.64166667 | `T500:persistent_like;T800:persistent_like;T1200:persistent_like;T1600:persistent_like` | stable persistent tail | `tail0.25=persistent_confirmed,tail0.35=persistent_confirmed,tail0.50=persistent_confirmed` |
| 0.11765625 | `persistent_confirmed` | 0.64166667 | `T500:declining_transient;T800:persistent_like;T1200:persistent_like;T1600:persistent_like` | stable persistent tail | `tail0.25=persistent_confirmed,tail0.35=persistent_confirmed,tail0.50=persistent_confirmed` |
| 0.125 | `persistent_confirmed` | 0.64166666 | `T500:declining_transient;T800:persistent_like;T1200:persistent_like;T1600:persistent_like` | stable persistent tail | `tail0.25=persistent_confirmed,tail0.35=persistent_confirmed,tail0.50=persistent_confirmed` |
| 0.135 | `persistent_confirmed` | 0.64166657 | `T500:persistent_like;T800:declining_transient;T1200:persistent_like;T1600:persistent_like` | stable persistent tail | `tail0.25=persistent_confirmed,tail0.35=persistent_confirmed,tail0.50=indeterminate` |
| 0.141262205 | `indeterminate` | 0.6416768 | `T500:persistent_like;T800:declining_transient;T1200:persistent_like;T1600:persistent_like` | see detail CSV | `tail0.25=persistent_confirmed,tail0.35=indeterminate,tail0.50=indeterminate` |
| 0.15 | `declining_transient` | 0.66853063 | `T500:persistent_like;T800:recovery_transient;T1200:declining_transient;T1600:persistent_like` | active during transient | `tail0.25=indeterminate,tail0.35=declining_transient,tail0.50=declining_transient` |
| 0.1584375 | `indeterminate` | 1.3078157 | `T500:extinct_like;T800:recovery_transient;T1200:recovery_transient;T1600:persistent_like` | see detail CSV | `tail0.25=recovery_transient,tail0.35=indeterminate,tail0.50=indeterminate` |
| 0.16486816 | `indeterminate` | 0.00014178281 | `T500:extinct_like;T800:extinct_like;T1200:extinct_like;T1600:recovery_transient` | see detail CSV | `tail0.25=indeterminate,tail0.35=indeterminate,tail0.50=extinct_confirmed` |
| 0.175 | `extinct_confirmed` | 1.17501e-10 | `T500:extinct_like;T800:extinct_like;T1200:extinct_like;T1600:extinct_like` | predator tail below epsilon | `tail0.25=extinct_confirmed,tail0.35=extinct_confirmed,tail0.50=extinct_confirmed` |

## Boundary Diagnosis

- final status sequence: `persistent_confirmed, persistent_confirmed, persistent_confirmed, persistent_confirmed, persistent_confirmed, indeterminate, declining_transient, indeterminate, indeterminate, extinct_confirmed`
- transient stress count at longest horizon: `3` of `10`
- final label: `pde_evo_persistence_unresolved`

The multi-horizon statuses do not form a clean stable boundary.

## Implication for Spatial Suppression

Spatial suppression mechanism should not be interpreted until this persistence criterion is stable, unless the final label establishes a stable boundary.

Spatial suppression mechanism should not be interpreted because the persistence boundary remains unresolved.

## Files

- `results/roy_pde_evo_persistence_stability.csv`
- `results/roy_pde_evo_persistence_stability_summary.csv`

## Next Step

Next: inspect numerical stability and physicality before further interpretation.
