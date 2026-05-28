# Nonlinear PDE Results 05

This run ends the 1D validation phase and starts the Roy-style 2D phase. It only verifies nonlinear 2D pattern formation; it does not compute predator mortality thresholds.

## 2D Pattern Reproduction

- parameter line: `kappa=0.15`, `eta=0.005`, `mu=0.85`, `delta=0.1`
- grid: `64 x 64`, `T=80.0`, `dt=0.01`, seed `20260601`
- ODE stable at homogeneous equilibrium: `True`
- continuous max spatial growth: `0.208692` at `k=3.17604`
- discrete max spatial growth: `0.20864` at mode `20`
- final mean predator density: `1.05823`
- final spatial variances `(u, v, w)`: `(0.0652113, 0.133896, 0.000663182)`
- dominant Fourier wavelength: `1.96116`
- pattern measurable: `True`
- negative population values detected: `False`
- negative free-space values detected: `False`

Interpretation: this confirms that the validated Roy-style parameter line can produce a measurable 2D nonlinear spatial pattern under no-flux finite-difference simulation. It is not a rescue-threshold result.

Outputs:

- `results\roy_2d_pattern_reproduction.csv`
- `figures\roy_2d\05_final_maps_mu_0p85.png`
- `figures\roy_2d\05_mean_w_timeseries_mu_0p85.png`
- `figures\roy_2d\05_power_spectrum_mu_0p85.png`
