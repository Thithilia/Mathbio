# Outline: Eco-Evolutionary Rescue and Spatial Bistability in a Roy-Style Predator–Prey Model

## Working Abstract

This report studies indirect evolutionary rescue in a Roy-style predator-prey model with prey defense evolution. The fixed-defense spatial route did not produce robust predator rescue. The well-mixed eco-evolutionary ODE did support indirect evolutionary rescue through a decrease in prey defense frequency. In the spatial PDE, the current result is not a simple spatial suppression result. It is a bistability and basin-structure result: the same stress can lead to predator persistence or extinction depending on initial defense frequency and predator abundance.

## 1. Introduction

- Motivation: indirect evolutionary rescue requires prey trait or genotype frequencies to change after predator mortality stress.
- Fixed-defense spatial patterning alone cannot test the evolutionary rescue mechanism.
- Central question: how does spatial structure alter the ODE eco-evolutionary rescue mechanism?
- Current framing: reachable outcome basins, not scalar thresholds.

## 2. Model Sequence

- Fixed-defense Roy-style ODE/PDE.
- Eco-evolutionary ODE with total prey density `n`, predator density `w`, and defense frequency `q`.
- Eco-evolutionary PDE with diffusion in `n`, `w`, and `q`.
- Later diagnostic sequence: classifier sensitivity, multi-horizon persistence, continuation, hysteresis, basin mapping, and basin-boundary scanning.

## 3. Fixed-Defense Spatial Model

- Fixed-defense spatial Roy-style PDE did not produce robust `ODE extinct, PDE persistent` rescue.
- The best candidate at `D_w/D_u=150` was classified as `transient_or_numerical_candidate`.
- This route does not support a robust fixed-defense spatial rescue claim.

## 4. Eco-Evolutionary ODE Rescue

- ODE no-evolution threshold: `m_c^{ODE,no evo} = 0.069448242`.
- ODE evolution threshold: `m_c^{ODE,evo} = 0.16486816`.
- Evolutionary rescue effect: `Delta_evo_ODE = 0.095419922`.
- Defense frequency `q` decreased in the rescue window from about `0.6726` to about `0.3336`.
- Conclusion: `ODE_indirect_evolutionary_rescue_supported`.

## 5. Spatial PDE and the Failure of Scalar Threshold Language

- Initial threshold comparison suggested a lower PDE-evo threshold than ODE-evo.
- PR #4: `pde_evo_threshold_classifier_sensitive`.
- PR #5: `pde_evo_persistence_unresolved`.
- PR #6: `pde_evo_hysteresis_detected`.
- PR #7: `pde_evo_bistability_mapped`.
- PR #9: `basin_boundary_mapped`.
- Interpretation: the spatial PDE does not have a stable scalar persistence threshold in the tested stress range.

## 6. Hysteresis and Basin Structure

- Upward continuation can maintain a predator-persistent branch at stresses where downward continuation can remain extinct or transient.
- PR #7 mapped stress-response regimes and found bistable persistent/extinct outcomes.
- A single scalar threshold is inappropriate when persistent and extinct states are both reachable at the same stress.

## 7. Basin-Boundary Dependence on q0 and w0

- PR #9 varied initial defense frequency `q0` and predator abundance scale `w0_scale`.
- At `0.1584375`: persistent 20, extinct 7, transient 43, unresolved 0, nonphysical 0.
- At `0.16486816`: persistent 14, extinct 21, transient 35, unresolved 0, nonphysical 0.
- Low-to-intermediate `q0` values more often enter predator-persistent outcomes.
- Initial predator abundance modulates basin entry within that `q0` range.
- Transient outcomes remain common, so the basin boundary is not fully sharp at the current horizon.

## 8. Discussion

- The ODE result supports indirect evolutionary rescue through prey defense evolution.
- The spatial PDE result is not a simple amplification or suppression result.
- The current result is not a simple spatial suppression result. It is a bistability and basin-structure result.
- Basin structure should be treated as the primary spatial result.
- Future claims about spatial mechanism should be made only after the basin boundary is adaptively refined.

## 9. Figures Needed

- Figure 1: Model schematic for ODE/PDE eco-evolutionary variables `n`, `w`, `q`.
- Figure 2: ODE rescue threshold comparison.
- Figure 3: Fixed-defense spatial negative result.
- Figure 4: PDE-evo classifier sensitivity / hysteresis diagnostic.
- Figure 5: Basin regime map from PR #7.
- Figure 6: q0-w0 basin-boundary heatmap from PR #9.

## 10. Tables Needed

- Table 1: Parameter values.
- Table 2: Threshold quantities from ODE and early PDE screening.
- Table 3: Stress-regime map from bistability analysis.
- Table 4: Basin-boundary counts from q0-w0 scan.

## 11. Open Technical Checks

- Need adaptive refinement of q0-w0 separatrix.
- Need longer horizon checks for transient grid points.
- Need sensitivity to diffusion coefficients after basin boundary is quantified.
- Need paper-quality figure generation.

## 12. Next Simulation Step

Adaptive refinement of the basin boundary in q0-w0 space for stresses `0.1584375` and `0.16486816`.

## Verification requirements before publication-level manuscript

Before this outline is expanded into a publication-level manuscript, claims should be checked against `research_notes/roy_evo_spatial_verification_plan.md`.

Required evidence:

- Representative PDE snapshots of `n(x,y,t)`, `w(x,y,t)`, and `q(x,y,t)` for persistent, extinct, and transient or basin-boundary cases.
- Spatial mean time series for `n`, `w`, and `q`.
- Representative PDE snapshots and mean time series have been generated in `figures/roy_evo_spatial/report/fig19_pde_solution_snapshots.png` and `figures/roy_evo_spatial/report/fig20_pde_mean_timeseries.png`.
- Upward/downward continuation diagram with branch classifications.
- Residual and convergence diagnostics, including tail slopes, normalized PDE RHS residuals, and horizon sensitivity.
- Adaptive `q0`-`w0` basin-boundary refinement at `0.1584375` and `0.16486816`.
- Grid/time-step, diffusion-coefficient, trade-off-strength, and evolutionary-rate robustness checks.
- Analytical support through ODE equilibria, selection-gradient sign analysis, and stability or bifurcation notes where feasible.

Publication-level wording should remain proposition-based: assumptions, evidence, missing evidence, and allowed claim. General claims that spatial structure amplifies, suppresses, or generally causes bistability are not justified by the current evidence.
