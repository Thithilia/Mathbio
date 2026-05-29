# Homogeneous Eco-Evolutionary Mechanism Behind the Roy Rescue Basins

## Purpose

This note analyzes the homogeneous eco-evolutionary ODE mechanism that appears to control basin-dependent outcomes in the Roy rescue model.

## Why this follows from the ODE-PDE mechanism test

PR #18 showed the current basin dependence is reaction-dominated homogeneous multistability, not spatial-pattern-mediated rescue. Representative ODE and PDE classifications agreed 3/3, ODE-PDE basin labels agreed for 90 percent of the q0-w0 grid, all disagreements involved transient labels, direct persistent/extinct disagreements were zero, final spatial CV values were very small, and perturbation outcome changes were zero.

## Model and selection-gradient logic

The ODE uses total prey density n, predator density w, defense frequency q, and free space z = kappa^{-1} - n - w. The selection gradient is `(r_v-r_u) z - (a_v-a_u) w`, and the predator growth factor is `b(q) n z - (m+s) - mu w`.

## Representative ODE trajectories

The representative ODE trajectories are saved in `results/roy_ode_homogeneous_representative_timeseries.csv` and plotted in `figures/roy_evo_spatial/report/fig28_ode_representative_timeseries.png`.

## ODE q0-w0 basin map

The ODE basin map uses the same q0-w0 grid and target stresses as the PR #18 comparison. It is saved in `results/roy_ode_homogeneous_basin_map.csv` and plotted in `figures/roy_evo_spatial/report/fig29_ode_basin_map.png`.

## Equilibria and stability

The numerical equilibrium scan found 31 unique equilibria across 6 target stresses. This is numerical evidence only, not an analytical proof. The table is `results/roy_ode_homogeneous_equilibria.csv`, and the stability figure is `figures/roy_evo_spatial/report/fig31_ode_equilibria_stability.png`.

## Equilibrium compensation mechanism

Across the analyzed stress values, the numerical equilibrium scan finds a stable persistent equilibrium with approximately \(n^* \approx 4.8333\) and \(w^* \approx 0.6417\).

As stress increases, the equilibrium defense frequency \(q^*\) decreases:

- \(s=0\): \(q^*=0.6726\)
- \(s=0.069448242\): \(q^*=0.4717\)
- \(s=0.11765625\): \(q^*=0.3322\)
- \(s=0.1584375\): \(q^*=0.2142\)
- \(s=0.16486816\): \(q^*=0.1955\)
- \(s=0.175\): \(q^*=0.1662\)

This suggests a compensation mechanism: prey defense frequency shifts downward, increasing predator conversion opportunity enough to offset increased predator mortality at a positive predator equilibrium.

At \(s=0.175\), the scan also finds a stable near-extinct predator equilibrium, supporting homogeneous bistability at that stress.

This is numerical equilibrium evidence, not an analytical proof.

## Mechanism interpretation

Qualitative ODE mechanism label: `ode_homogeneous_basin_structure_supported`.

The current evidence supports the interpretation that the observed PDE basin structure is inherited from the homogeneous eco-evolutionary reaction system.

## What this means for the spatial PDE

The spatially extended PDE preserves basin-dependent outcomes, but persistent spatial patterning is not currently supported as the mechanism. The PDE remains useful as a spatially extended test of the reaction dynamics rather than as evidence for spatial-pattern-mediated rescue.

## What is still not general

This conclusion is limited to `RoyEvoParams(b_u=0.08, b_v=0.02)` and the tested diffusion and perturbation settings. It does not establish general behavior across trade-off forms, evolutionary rates, diffusion coefficients, or broader parameter regions.

## Files

- `experiments/21_roy_ode_homogeneous_mechanism.py`
- `results/roy_ode_homogeneous_representative_timeseries.csv`
- `results/roy_ode_homogeneous_basin_map.csv`
- `results/roy_ode_homogeneous_equilibria.csv`
- `results/roy_ode_homogeneous_mechanism_summary.csv`
- `figures/roy_evo_spatial/report/fig28_ode_representative_timeseries.png`
- `figures/roy_evo_spatial/report/fig29_ode_basin_map.png`
- `figures/roy_evo_spatial/report/fig30_selection_growth_phase.png`
- `figures/roy_evo_spatial/report/fig31_ode_equilibria_stability.png`
- `figures/roy_evo_spatial/report/fig32_updated_mechanism_diagram.png`

## Next step

Correct manuscript language first, then test robustness of the homogeneous basin structure across trade-off parameters and refine the analytical equilibrium/stability interpretation.
