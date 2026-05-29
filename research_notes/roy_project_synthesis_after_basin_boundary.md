# Research Note: Basin-Boundary Structure in the Spatial Eco-Evolutionary Rescue Model

## Executive Summary

The well-mixed ODE supports indirect evolutionary rescue through prey defense evolution. The spatial PDE does not admit a simple scalar rescue threshold in the tested stress range. Instead, it shows hysteresis and bistability. PR #9 shows that basin entry depends on the initial defense frequency `q0` and predator abundance scale `w0_scale`. Therefore, the spatial model is best described by reachable outcome basins rather than a single threshold.

## Current Endpoint

The current project endpoint is:

```text
ODE prey defense evolution supports indirect evolutionary rescue. In the spatial PDE, the evolutionary rescue response is path-dependent and bistable. Basin entry depends on initial defense frequency q0 and predator abundance w0, so the spatial model should be analyzed through basin structure rather than a single persistence threshold.
```

## Model Sequence

The fixed-defense spatial Roy-style PDE did not produce robust predator rescue. The best fixed-defense candidate at `D_w/D_u=150` was classified as `transient_or_numerical_candidate`.

The well-mixed eco-evolutionary ODE supported indirect evolutionary rescue:

```text
m_c^{ODE,no evo} = 0.069448242
m_c^{ODE,evo} = 0.16486816
Delta_evo_ODE = 0.095419922
```

In the ODE rescue window, prey defense frequency decreased from about `0.6726` to about `0.3336`, consistent with reduced defense and increased palatability supporting predator recovery.

The spatial eco-evolutionary PDE did not produce a stable scalar rescue threshold. Its response depends on persistence criteria, time horizon, continuation direction, and initial condition.

## Why Threshold Language Failed

The first threshold comparison suggested a lower PDE-evo threshold than the ODE-evo threshold. Later analyses showed that threshold language is insufficient because the system has multiple reachable outcomes at the same stress.

The correction sequence was:

```text
PR #4: pde_evo_threshold_classifier_sensitive
PR #5: pde_evo_persistence_unresolved
PR #6: pde_evo_hysteresis_detected
PR #7: pde_evo_bistability_mapped
PR #9: basin_boundary_mapped
```

## Bistability and Hysteresis

PR #7 showed that the spatial PDE has stress-response regimes rather than one monotone persistence boundary. The focused basin map included:

```text
0.141262205: persistent_transient_mixed
0.15: persistent_transient_mixed
0.1584375: bistable_persistent_extinct
0.16486816: bistable_persistent_extinct
0.175: bistable_persistent_extinct
```

This established that predator-persistent and predator-extinct outcomes can both be reachable at the same stress.

## Basin-Boundary Scan

PR #9 quantified the basin boundary by varying `q0` and `w0_scale` at two stresses inside the bistable interval.

| stress | persistent | extinct | transient | unresolved | nonphysical | regime |
|---:|---:|---:|---:|---:|---:|---|
| 0.1584375 | 20 | 7 | 43 | 0 | 0 | `bistable_persistent_extinct` |
| 0.16486816 | 14 | 21 | 35 | 0 | 0 | `bistable_persistent_extinct` |

![PDE-evolution basin boundary scan](../figures/roy_evo_spatial/17_basin_boundary_heatmap.png)

## Interpretation of q0 and w0 Dependence

Low-to-intermediate `q0` tends to favor persistent outcomes, so defense state is a major basin-entry coordinate.

Initial predator abundance modulates basin entry within that `q0` range. The same stress can lead to predator persistence, predator extinction, or a transient classification depending on the initial predator abundance scale.

Transient outcomes remain common, so the basin boundary is not fully sharp at the current horizon. The next step is adaptive basin-boundary refinement, not another scalar threshold scan.

## Current Scientific Conclusion

Current conclusion: the spatial eco-evolutionary model has a bistable rescue structure. Initial defense frequency and predator abundance help determine whether predator persistence or extinction is reached at the same stress.

The spatial PDE has bistable and path-dependent rescue dynamics. It should be analyzed through reachable outcome basins and basin boundaries, not through a single scalar persistence threshold.

## Files

Key notes:

```text
research_notes/roy_evo_spatial_rescue_summary.md
research_notes/roy_project_synthesis_after_bistability.md
research_notes/roy_pde_evo_basin_boundary_scan.md
research_notes/roy_project_synthesis_after_basin_boundary.md
```

Key result tables:

```text
results/roy_pde_evo_basin_boundary_scan.csv
results/roy_pde_evo_basin_boundary_summary.csv
```

Figure:

```text
figures/roy_evo_spatial/17_basin_boundary_heatmap.png
```

## Next Research Direction

The next research direction is adaptive refinement of the basin boundary, not another threshold scan. The next quantitative question is how the separatrix in q0-w0 space changes with stress.
