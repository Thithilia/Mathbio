# Research Note: PDE-Evolution Threshold Monotonicity Check

## Executive Summary

Final Step 10 label: `pde_evo_threshold_classifier_sensitive`.

Persistence classifications change with tail fraction, time horizon, or the tail-slope rule.

The earlier Stage B threshold should not be treated as a stable monotone boundary without additional checks.

The apparent threshold is classifier-sensitive; the next step should stabilize the persistence criterion before interpreting spatial mechanism.

## Why This Check Was Needed

Step 10 originally tried to diagnose the spatial suppression mechanism. But the focused S3 run did not reproduce the expected ODE-persistent/PDE-failed contrast: PDE-evo also remained persistent at S3.

Therefore mechanism diagnosis is premature. This note tests whether PDE-evo persistence is monotone and stable near the reported PDE and ODE thresholds.

## Setup

- profile run: `focused`
- model: PDE-evo only
- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`
- initial state: unstressed Step 09A burn-in via `find_evo_equilibrium`
- grid: `64x64`, `L_x=L_y=20`, seed `20260702`
- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`
- integration: `dt=0.1`, `record_every=50`
- evaluated horizons: `500, 800`
- tail fractions: `0.25`, `0.35`, `0.50`
- base persistence rule: physical trajectory, tail mean predator density above `1e-4`, tail minimum above `2.5e-5`, and tail slope no more negative than the tail-slope floor
- relaxed rule: same density thresholds but without the tail-slope check

## Stress Grid

The focused grid was:

```text
0.09, 0.105, 0.11765625, 0.125, 0.135, 0.141262205, 0.15, 0.1584375, 0.16486816, 0.175
```

This covers the shared rescue region, the reported Stage B PDE threshold, the S3 midpoint, the Stage C/ODE-threshold region, and above the ODE threshold.

## Persistence Classification Results

Base classifier results for `T=500`, `tail_fraction=0.25`:

| stress | persistent | no-slope persistent | tail mean w | tail slope | slope floor | tail mean q | note |
|---:|---|---|---:|---:|---:|---:|---|
| 0.09 | True | True | 0.64300225 | -3.1566657e-05 | -0.0012860045 | 0.41199753 | `persistent` |
| 0.105 | True | True | 0.65309898 | -0.00025427601 | -0.001306198 | 0.36722232 | `persistent` |
| 0.11765625 | False | True | 0.74175158 | -0.0014974554 | -0.0014835032 | 0.3133891 | `slope_check_rejects` |
| 0.125 | False | True | 0.92820666 | -0.0026030541 | -0.0018564133 | 0.24142544 | `slope_check_rejects` |
| 0.135 | True | True | 1.168098 | 0.0030704485 | -0.002336196 | 0.075320101 | `persistent` |
| 0.141262205 | True | True | 0.40790958 | 0.011803306 | -0.00081581916 | 0.026979858 | `persistent` |
| 0.15 | True | True | 0.0019110449 | 3.0171023e-05 | -3.8220898e-06 | 0.023692232 | `persistent` |
| 0.1584375 | False | False | 4.4673973e-05 | 3.5680794e-07 | -8e-07 | 0.02346533 | `near_classifier_tolerance` |
| 0.16486816 | False | False | 2.7380992e-06 | 4.4407537e-09 | -8e-07 | 0.023342503 | `near_classifier_tolerance` |
| 0.175 | False | False | 3.7198705e-08 | -3.1479383e-10 | -8e-07 | 0.023182652 | `near_classifier_tolerance` |

Base classifier results for `T=800`, `tail_fraction=0.25`:

| stress | persistent | no-slope persistent | tail mean w | tail slope | slope floor | tail mean q | note |
|---:|---|---|---:|---:|---:|---:|---|
| 0.09 | True | True | 0.64166641 | 1.6903204e-09 | -0.00080208301 | 0.4121856 | `persistent` |
| 0.105 | True | True | 0.64164926 | 3.076418e-07 | -0.00080206157 | 0.36878341 | `persistent` |
| 0.11765625 | True | True | 0.64156573 | -1.625158e-07 | -0.00080195716 | 0.33219821 | `persistent` |
| 0.125 | True | True | 0.64451398 | -7.1696997e-05 | -0.00080564247 | 0.31078331 | `persistent` |
| 0.135 | False | True | 0.75638642 | -0.0014701187 | -0.00094548303 | 0.26096688 | `slope_check_rejects` |
| 0.141262205 | False | True | 1.0735241 | -0.0021108214 | -0.0013419052 | 0.14673681 | `slope_check_rejects` |
| 0.15 | True | True | 0.91739782 | 0.008076642 | -0.0011467473 | 0.0054049572 | `persistent` |
| 0.1584375 | True | True | 0.00066550025 | 6.5733068e-06 | -8.3187531e-07 | 0.0020895547 | `persistent` |
| 0.16486816 | False | False | 6.6431044e-06 | 2.6603904e-08 | -5e-07 | 0.0020771473 | `near_classifier_tolerance` |
| 0.175 | False | False | 6.2376484e-09 | -3.7164803e-11 | -5e-07 | 0.0020626714 | `near_classifier_tolerance` |

## Classifier Sensitivity

- tail-fraction disagreement groups: `4`
- horizon disagreement groups: `15`
- slope-rule disagreement rows: `12`
- near-tolerance rows: `17`

| stress | T | tail 0.25 | tail 0.35 | tail 0.50 | no-slope tail 0.25 |
|---:|---:|---|---|---|---|
| 0.09 | 500 | True | True | True | True |
| 0.09 | 800 | True | True | True | True |
| 0.105 | 500 | True | True | False | True |
| 0.105 | 800 | True | True | True | True |
| 0.11765625 | 500 | False | False | False | True |
| 0.11765625 | 800 | True | True | True | True |
| 0.125 | 500 | False | False | True | True |
| 0.125 | 800 | True | True | False | True |
| 0.135 | 500 | True | True | True | True |
| 0.135 | 800 | False | False | False | True |
| 0.141262205 | 500 | True | True | True | True |
| 0.141262205 | 800 | False | False | True | True |
| 0.15 | 500 | True | True | True | True |
| 0.15 | 800 | True | True | True | True |
| 0.1584375 | 500 | False | False | False | False |
| 0.1584375 | 800 | True | True | True | True |
| 0.16486816 | 500 | False | False | False | False |
| 0.16486816 | 800 | False | False | False | False |
| 0.175 | 500 | False | False | False | False |
| 0.175 | 800 | False | False | False | False |

## Monotonicity Diagnosis

- re-entry sequences: `6`
- clean transition sequences: `0`
- physical failures: `0`

Final Step 10 label: `pde_evo_threshold_classifier_sensitive`.

Persistence classifications change with tail fraction, time horizon, or the tail-slope rule.

## Implication for Spatial Suppression Mechanism

The spatial suppression mechanism should not be claimed from the current Step 10 diagnostics.

The apparent threshold is classifier-sensitive; the next step should stabilize the persistence criterion before interpreting spatial mechanism.

This check does not alter the PR #3 conclusions; it only shows that the PDE-evo threshold boundary used for mechanism localization needs a stability audit before mechanistic interpretation.

## Files

- `results/roy_spatial_suppression_monotonicity.csv`
- `results/roy_spatial_suppression_summary.csv`
- `results/roy_spatial_suppression_timeseries.csv`

## Next Step

Next: stabilize the PDE-evo persistence criterion and threshold boundary before interpreting spatial mechanism.
