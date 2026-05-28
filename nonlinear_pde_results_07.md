# Nonlinear PDE Results 07

This run tests whether the preliminary negative 2D Roy-style threshold result was caused by short integration time, coarse grid resolution, sparse stress sampling, or pattern collapse near the extinction transition.

## 1. Long-Time Pattern Persistence

- long-time runs completed: `35`
- runs at `T=200`: `8`
- `T=200` runs with persistent final-quarter pattern strength above tolerance: `2`
- unstressed `T=200` runs with persistent patterning: `2/2`
- stressed near-threshold `T=200` runs with persistent patterning: `0/6`

Answer: the pattern persists for the unstressed Roy-style baseline, but collapses near the predator extinction transition in the long-time runs.

## 2. Grid Resolution

- rows at `64x64` or higher: `42`
- `64x64` is treated as the minimum serious resolution for interpreting 2D pattern dynamics.
- `96x96` spot checks are included where explicit CFL cost remains feasible.

Answer: increasing from `36x36` to `64x64`, with selected `96x96` checks, does not reveal ODE-extinct/PDE-persistent behavior.

## 3. Fine Stress Scan

- fine threshold rows completed: `131`
- classification counts: `{'ODE extinct, PDE extinct': 127, 'ODE persistent, PDE persistent': 4}`
- ODE-extinct/PDE-persistent rows satisfying baseline Turing and persistent-pattern filters: `0`
- adaptive baseline 64x64 rows near the shifted long-time transition: `8`
- adaptive baseline rows with persistent measurable patterning: `0`
- adaptive baseline persistent rows: `4`; extinct rows: `4`

Answer: the finer scan finds the long-time baseline transition between approximately `s=0.50` and `s=0.525`, but both ODE and PDE switch together and no persistent-pattern rescue window appears.

## 4. Regimes Beyond mu = 0.85

The scan covers `mu = [0.60, 0.72, 0.80, 0.85, 0.89, 0.95]` and `D_w/D_u = [40, 70, 100, 150, 250, 400]` with a fine stress grid near the transition.

Answer: outside `mu=0.85`, the tested fine-stress points also show no ODE-extinct/PDE-persistent regime.

## 5. Conservative Conclusion

The previous negative result is **robust within tested ranges**. Pattern-mediated rescue is not claimed.

A claim would require baseline Turing instability, persistent measurable 2D patterning near threshold, ODE extinction with PDE persistence under the mean-density criterion, and survival under longer `T`, higher grid resolution, and at least two perturbation seeds.

Outputs:

- `results\roy_2d_longtime_pattern_scan.csv`
- `results\roy_2d_fine_threshold_scan.csv`
- `results\roy_2d_pattern_timeseries.csv`
- `figures\roy_2d_longtime\07_longtime_pattern_strength.png`
- `figures\roy_2d_longtime\07_fine_phase_stress_mu.png`
- `figures\roy_2d_longtime\07_fine_phase_stress_diffusion_ratio.png`
