# Nonlinear PDE Results 03

This audit optimized the linear Turing growth objective with `scipy.optimize.differential_evolution`; it did not run nonlinear PDE threshold simulations and does not claim pattern-mediated rescue.

## Audit settings

- restarts: `2`
- maxiter: `8`
- popsize: `5`
- continuous k range: `[0.0001, 20.0]` with `240` grid points
- discrete Neumann modes: `1..120`
- equilibrium grid: `350`
- parameter ranges: `r_D in [0.05,2]`, `r_U-r_D in [1e-3,5]`, `a_D in [1e-2,2]`, `a_U-a_D in [1e-3,8]`, `mu in [1e-6,1]`, `delta_U in [1e-7,10]`, diffusion ratios in `[1e-5,1e6]`
- Holling-II handling-time range, when run: `h in [1e-3,20]`

## Minimal model

No, this targeted optimization run did not find a minimal-model continuous-k Turing candidate.

- best rows saved: `16`
- continuous-k candidates among saved rows: `0`
- discrete-domain candidates among saved rows: `0`
- failure counts among saved rows: `{'no_spatial_instability': 16}`

The negative result is only robust within the parameter bounds and optimizer budget used here; it is stronger than the previous random scan because the objective was directly optimized, but it is not a mathematical non-existence proof.

## Holling-II variant

No, the Holling-II variant did not admit a continuous-k Turing candidate in this audit.

- best rows saved: `16`
- continuous-k candidates among saved rows: `0`
- discrete-domain candidates among saved rows: `0`
- failure counts among saved rows: `{'no_spatial_instability': 16}`

## Next model for nonlinear thresholds

Do not proceed to nonlinear threshold computation yet; broaden or analytically refine the feasibility audit first.

Output:

- `results\turing_feasibility_audit.csv`
