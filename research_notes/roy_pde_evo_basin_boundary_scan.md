# Research Note: Basin Boundary Scan for the PDE-Evolution Model

## Executive Summary

Final Step 15 label: `basin_boundary_mapped`.

The q0-w0 grid resolves interpretable basin regions for at least two target stresses.

## Why This Was Needed

PR #7 mapped bistability qualitatively: at least one stress admitted both predator-persistent and predator-extinct reachable outcomes. Step 15 turns that result into a focused q0-w0 basin-boundary map inside the bistable stress interval.

This step does not change model equations, does not run a broad parameter scan, and does not diagnose the spatial covariance mechanism.

## Setup

- profile run: `focused`
- model: PDE-evo only
- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`
- grid: `64x64`, `L_x=L_y=20`, seed `20260702`
- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`
- integration: `dt=0.1`, `record_every=50`
- horizon: `T=1600`
- perturbation amplitude: `1e-5`
- target stresses: `0.1584375, 0.16486816`
- steady criteria: `abs(relative_change) < 0.02` and `normalized_residual < 0.0001`

## Initial-Condition Grid

Each run starts from the Step 09A burn-in prey density and scales predator abundance while setting defense frequency:

```text
n0 = n_baseline
w0 = w_baseline * w0_scale
q0 = q0_value
```

`q0` values: `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`

`w0_scale` values: `0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5`

Physicality is checked before simulation. Nonphysical initial states are recorded as `nonphysical_initial_condition` and skipped.

## Basin Boundary Results

Figure: `figures/roy_evo_spatial/17_basin_boundary_heatmap.png`

![PDE-evolution basin boundary scan](../figures/roy_evo_spatial/17_basin_boundary_heatmap.png)

The basin boundary is two-dimensional in defense-predator initial state: persistent basins concentrate at low-to-intermediate q0, and larger predator initial abundance helps persistence within that range.

## Stress-Level Regime Summary

| stress | persistent | extinct | transient | unresolved | nonphysical | regime label |
|---:|---:|---:|---:|---:|---:|---|
| 0.1584375 | 20 | 7 | 43 | 0 | 0 | `bistable_persistent_extinct` |
| 0.16486816 | 14 | 21 | 35 | 0 | 0 | `bistable_persistent_extinct` |

## Interpretation

The scan directly varies initial defense frequency and predator abundance. Bistable stress labels indicate that predator-persistent and predator-extinct basins are both reached on the same q0-w0 grid.

Transient outcomes remain important: they mark grid points where the current horizon and steady-state residual criteria do not yet justify assigning a persistent or extinct basin.

## Files

- `results/roy_pde_evo_basin_boundary_scan.csv`
- `figures/roy_evo_spatial/17_basin_boundary_heatmap.png`
- `experiments/17_roy_pde_evo_basin_boundary_scan.py`

## Next Step

Refine the mapped basin boundary with targeted continuation in q0-w0 space before returning to spatial covariance mechanism interpretation.

basin_boundary_mapped
