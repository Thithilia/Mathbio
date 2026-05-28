# Nonlinear PDE Results 06

This run adds predator mortality stress to the Roy-style model as `delta_eff = delta + s` and compares ODE/PDE persistence using mean predator density.

## Baseline Thresholds

- `s_c^ODE = 0.653906`
- `s_c^PDE = 0.646875`
- `Delta s_c = s_c^PDE - s_c^ODE = -0.00703125`
- baseline Turing unstable: `True`

## Phase Diagrams

- phase points evaluated: `60`
- classification counts: `{'ODE persistent, PDE persistent': 39, 'ODE extinct, PDE extinct': 21}`
- rescue-like points before robustness filtering: `0`
- robustness rows evaluated: `0`

## Conservative Interpretation

- pattern-mediated rescue claimed: `False`

A claim requires baseline Turing instability, measurable nonlinear 2D patterning, ODE extinction with PDE persistence under the mean-density criterion, and robustness to final time, grid resolution, and perturbation seed. This script applies that rule conservatively.

Outputs:

- `results\roy_2d_threshold_scan.csv`
- `figures\roy_2d\06_phase_stress_mu.png`
- `figures\roy_2d\06_phase_stress_diffusion_ratio.png`
