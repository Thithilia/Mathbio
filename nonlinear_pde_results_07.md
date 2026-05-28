# Nonlinear PDE Results 07

This revision makes the predator-mortality threshold comparison the primary scientific output. Pattern morphology remains exploratory and is not used as the rescue criterion.

Persistence is classified from the final 25% of the predator-density time series. A trajectory is persistent when the tail mean is above `epsilon`, the tail minimum stays above `0.25 epsilon`, and the least-squares tail slope is not strongly negative:

`tail_slope >= -max(epsilon, 0.25 tail_mean) / tail_duration`.

## Core Threshold Question

The decisive quantity is

`Delta m_c = m_c_PDE - m_c_ODE`.

Positive values indicate that spatial structure expands the predator-persistence range under mortality stress; negative values indicate that spatial structure shrinks it; values within tolerance are treated as no measurable threshold effect.

## Baseline Regime

Baseline parameters use `mu=0.85` and `D_w/D_u=100`.

Quick Stage A estimate:

- `m_c^ODE = 0.389893`
- `m_c^PDE = 0.392578`
- `Delta m_c = 0.00268555`
- tolerance used for this row: `0.001`
- row status: `ok`

Validated `T=200`, `64x64` estimate from the first baseline seed:

- `m_c^ODE = 0.395752`
- `m_c^PDE = 0.394531`
- `Delta m_c = -0.0012207`
- tolerance used for this row: `0.00390625`
- row status: `ok`

## Stage A: Quick Threshold Finder

- completed threshold rows: `13`
- candidate spatial-rescue rows: `10`
- candidate spatial-inhibition rows: `0`
- scanned axes: `mu`, `D_w/D_u`, and a small exploratory `eta` axis.
- Stage A is a candidate finder only; its positive rows are not interpreted as rescue without Stage B validation.

## Closest Regime To A Rescue Candidate

- closest Stage A regime: `mu` with `mu=0.6`
- `m_c^ODE = 0.4375`
- `m_c^PDE = 0.4375`
- `Delta m_c = 0`
- tolerance used for this row: `0.001`
- row status: `ok`

## Stage B: Validation

Stage A produced `10` positive `Delta m_c` candidate rows at the quick `T=70`, `36x36` setting. Stage B therefore validated the baseline and the first positive candidate regimes using longer `T=200`, `64x64` grids, and three perturbation seeds.

- validated positive-threshold rows after Stage B: `0`

- `B_001`: axis `baseline`, seed `20260621`, T `200`, grid `64x64`, Delta `-0.0012207`, status `ok`
- `B_002`: axis `baseline`, seed `20260622`, T `200`, grid `64x64`, Delta `-0.0012207`, status `ok`
- `B_003`: axis `baseline`, seed `20260623`, T `200`, grid `64x64`, Delta `-0.0012207`, status `ok`
- `B_004`: axis `mu`, seed `20260621`, T `200`, grid `64x64`, Delta `-0.000732422`, status `ok`
- `B_005`: axis `mu`, seed `20260622`, T `200`, grid `64x64`, Delta `-0.000732422`, status `ok`
- `B_006`: axis `mu`, seed `20260623`, T `200`, grid `64x64`, Delta `-0.000732422`, status `ok`
- `B_007`: axis `mu`, seed `20260621`, T `200`, grid `64x64`, Delta `-0.0012207`, status `ok`
- `B_008`: axis `mu`, seed `20260622`, T `200`, grid `64x64`, Delta `-0.0012207`, status `ok`
- `B_009`: axis `mu`, seed `20260623`, T `200`, grid `64x64`, Delta `-0.0012207`, status `ok`

## Interpretation

no measurable spatial-rescue effect was found; threshold differences are within tolerance or negative.

Spatial rescue is not claimed unless a positive threshold difference survives validation. In this run, the conclusion is framed entirely around the sign and robustness of `Delta m_c`, not around the mere existence of spatial patterning.

## Secondary Pattern Diagnostics

- representative long-time rows: `8`
- representative fine-stress rows: `24`
- dominant wavelength and Fourier power remain exploratory diagnostics only.

Outputs:

- `results\roy_2d_threshold_comparison.csv`
- `results\roy_2d_longtime_pattern_scan.csv`
- `results\roy_2d_fine_threshold_scan.csv`
- `results\roy_2d_pattern_timeseries.csv`
- `figures\roy_2d_longtime\07_threshold_delta.png`
- `figures\roy_2d_longtime\07_longtime_pattern_strength.png`
- `figures\roy_2d_longtime\07_fine_phase_stress_mu.png`
- `figures\roy_2d_longtime\07_fine_phase_stress_diffusion_ratio.png`
