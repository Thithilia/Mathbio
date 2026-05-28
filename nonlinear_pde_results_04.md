# Nonlinear PDE Results 04

This run implements the Roy-style dimensionless model in a separate module and treats the previous minimal mass-action and Holling-II variants as negative-control models.

No predator rescue thresholds were computed in this step.

## Roy-Style Reproduction

- Roy-style parameter rows tested: `6`
- rows with continuous and discrete Turing instability: `6`
- rows with nonlinear 1D pattern confirmation: `1`

Chosen PDE verification row:

- `mu = 0.85`
- homogeneous equilibrium `(u*, v*, w*, z*) = (0.154405, 0.466487, 0.99248, 5.05329)`
- ODE max real eigenvalue = `-0.0555199`
- continuous max spatial growth = `0.208692` at `k = 3.17604`
- discrete max spatial growth = `0.20864` at Neumann mode `20`
- final variances `(u, v, w) = (0.0629259, 0.133149, 0.000697154)`
- pattern confirmed by variance threshold: `True`

Interpretation: the Roy-style implementation reproduces the intended linear Turing instability and produces a nonlinear spatially heterogeneous final state in a 1D no-flux verification run. This is only a model-reproduction result, not a pattern-mediated rescue result.

Outputs:

- `results\roy_turing_reproduction.csv`
- `figures\04_roy_final_profiles.png`
