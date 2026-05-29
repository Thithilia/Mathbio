# Research Note: Mechanism of Spatial Suppression in the Eco-Evolutionary Rescue Model

## Executive Summary

Final Step 10 label: `spatial_suppression_inconclusive`.

The focused S3 window did not reproduce ODE persistence with PDE failure.

The ODE evolutionary rescue result remains intact, and the Step 09B conclusion is unchanged: the tested spatial PDE has a lower evolution threshold than the well-mixed ODE.

## Question

Why does the spatial PDE suppress the indirect evolutionary rescue window relative to the well-mixed ODE?

The analysis compares ODE and PDE trajectories at a focused set of stresses: unstressed baseline, a shared rescue-window stress, the PDE evolution threshold, the suppression-window midpoint, and the ODE evolution threshold.

## Setup

- profile: `focused`
- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`
- focused PDE config: `64x64`, `L=20`, `D_n=0.01`, `D_w=0.01`, `D_q=0.005`, `T=500`, seed `20260702`
- ODE evolution threshold: `0.16486816`
- PDE evolution threshold: `0.11765625`
- suppression-window midpoint S3: `0.141262205`
- small-covariance threshold used for dilution: `1e-08`
- small var(q) threshold used for dilution: `1e-08`

## ODE-PDE Rescue Window Comparison

| run | stress | persistent | tail mean w | tail mean q | q change |
|---|---:|---|---:|---:|---:|
| `ODE_EVO_S1` | 0.09 | True | 0.64166667 | 0.41218552 | -0.26042923 |
| `PDE_EVO_S1` | 0.09 | True | 0.64300225 | 0.41199753 | -0.26061721 |
| `ODE_EVO_S2` | 0.11765625 | True | 0.64166667 | 0.33215778 | -0.34045696 |
| `PDE_EVO_S2` | 0.11765625 | False | 0.74175158 | 0.3133891 | -0.35922564 |
| `ODE_EVO_S3` | 0.14126221 | True | 0.64164882 | 0.26384758 | -0.40876716 |
| `PDE_EVO_S3` | 0.14126221 | True | 0.40790958 | 0.026979858 | -0.64563488 |
| `ODE_EVO_S4` | 0.16486816 | False | 9.9973402e-05 | 8.0985385e-06 | -0.67260664 |
| `PDE_EVO_S4` | 0.16486816 | False | 2.7380992e-06 | 0.023342503 | -0.64927224 |

At S3, the requested suppression-window contrast is not reproduced: ODE persistence is `True` and PDE persistence is `True`.

## Spatial Covariance Diagnostics

| PDE evo run | cov(w,q) | cov(w,b(q)) | cov(w,n) | cov(w,b(q)nz) | spatial covariance bonus | var(q) |
|---|---:|---:|---:|---:|---:|---:|
| `PDE_EVO_S1` | -7.3765299e-17 | 4.4259179e-18 | -8.3393297e-16 | 7.3761139e-17 | -4.3293418e-17 | 1.2233947e-17 |
| `PDE_EVO_S2` | -2.3322203e-13 | 1.3993322e-14 | -1.7873786e-12 | 1.7933173e-13 | -6.87114e-14 | 5.9284413e-14 |
| `PDE_EVO_S3` | 4.5746278e-13 | -2.7447767e-14 | -2.0387715e-10 | 1.5216889e-11 | -1.7888466e-11 | 1.1333309e-14 |
| `PDE_EVO_S4` | -1.3817961e-18 | 8.2907768e-20 | 2.6828766e-19 | 1.6362485e-19 | 1.6301101e-19 | 1.5539619e-15 |

## Low-Defense / High-Gain Patch Occupancy

| PDE evo run | predator low-q enrichment | predator high-gain enrichment | diagnosis |
|---|---:|---:|---|
| `PDE_EVO_S1` | 1 | 1 | `pde_rescue_active` |
| `PDE_EVO_S2` | 1.0000009 | 1.0000009 | `no_rescue` |
| `PDE_EVO_S3` | 1.0000012 | 1.000016 | `inconclusive` |
| `PDE_EVO_S4` | 1.0000113 | 1.0000113 | `no_rescue` |

## Diagnosis

Final Step 10 label: `spatial_suppression_inconclusive`.

The focused S3 window did not reproduce ODE persistence with PDE failure.

The S2 near-threshold PDE run has near-zero covariance diagnostics and fails persistence while ODE evolution persists, but the specified S3 midpoint remains persistent in the PDE. The focused diagnostics therefore do not distinguish mismatch from diffusion dilution under the requested decision rule.

The default focused profile did not run the optional Stage C-style check.

## Interpretation

q evolution is active, and spatial covariances remain near zero in the focused runs. However, because the specified S3 midpoint does not reproduce PDE failure, this run should not be used to choose between mismatch and dilution.

This result leaves the PR #3 conclusion unchanged: the spatial PDE did not amplify the ODE rescue window.

## Files

- `results\roy_spatial_suppression_timeseries.csv`
- `results\roy_spatial_suppression_summary.csv`

## Next Step

Next: improve focused diagnostics or physicality checks before interpretation.
