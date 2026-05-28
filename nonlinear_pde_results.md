# Nonlinear PDE Threshold Experiment 01

This run uses the defended/undefended prey reaction-diffusion model from the report on a 1D domain with no-flux boundary conditions. The analytic prey-only invasion threshold is reported only as a linear sanity check; it is not used as the nonlinear rescue threshold.

## Parameters

- `r_U=1.2`, `r_D=0.6` with `r_U>r_D`
- `a_U=1.4`, `a_D=0.35` with `a_U>a_D`
- `K=1.0`, `e=0.55`
- `mu_UD=0.04`, `mu_DU=0.04`
- `delta_U=0.001`, `delta_D=0.05`, `delta_P=0.5`
- `L=20.0`, `n_x=50`
- threshold bracket: `m_low=0.1`, `m_high=0.7`
- final time: `T=80.0`
- persistence threshold: `epsilon=0.0001`
- initial condition: coexistence equilibrium at `m=0.1` plus random perturbation amplitude `0.001`

## Linear checks

- prey-only linear invasion threshold: `m_inv_ODE = 0.481250`
- positive coexistence equilibrium at `m=0.22`:
  `q=0.167321`, `N=0.760908`, `U=0.127316`, `D=0.633592`, `P=0.318551`
- ODE local stability at that equilibrium: `True`
- ODE eigenvalues: `[-0.14183+0.07725j -0.14183-0.07725j -0.45638+0.j     ]`
- Turing scan: `No Turing-unstable mode was detected for this baseline parameter set.`
- dominant scanned mode: `n=1`, `k=0.157080`, growth `-1.499858e-01`

## Nonlinear threshold results

- `m_c^ODE = 0.531250`
- `m_c^PDE` using total predator biomass = 0.578125
- `m_c^PDE` using mean predator density = 0.531250
- `Delta m_c` using total predator biomass = 0.046875
- `Delta m_c` using mean predator density = 0.000000
- total-biomass threshold sign classification: `m_c^PDE > m_c^ODE`
- mean-density threshold sign classification: `no resolvable threshold difference at this tolerance`

Important interpretation: The final predator variance was small in the first PDE runs, so this run should not be interpreted as evidence of pattern-mediated rescue. The sign of `Delta m_c` in this first run is therefore a threshold diagnostic for this parameter set and persistence criterion, not yet a demonstrated pattern-mediated rescue mechanism. A biological classification as pattern-promoted or pattern-inhibited rescue should require a Turing window, measurable spatial pattern strength, and a positive density-normalized threshold difference.

## PDE diagnostics for sampled mortality values

| m | persistent total | persistent mean | B_P(T) | mean P(T) | B_U(T) | B_D(T) | O_PU | mean edible | var(P) | negative values |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.300 | True | True | 3.373307e+00 | 1.686653e-01 | 4.654118e+00 | 1.253915e+01 | 5.000000e-02 | 5.452233e-01 | 1.089633e-16 | False |
| 0.500 | True | True | 2.876448e-02 | 1.438224e-03 | 9.908965e+00 | 1.006243e+01 | 5.000000e-02 | 8.697202e-01 | 5.153858e-16 | False |
| 0.600 | False | False | 2.186870e-05 | 1.093435e-06 | 9.983468e+00 | 1.001651e+01 | 5.000000e-02 | 8.741316e-01 | 3.563784e-22 | False |

## Saved figures

- `figures\01_final_profiles.png`
- `figures\01_predator_biomass_timeseries.png`
- `figures\01_phase_diagram_m_deltaP_ratio.png`

## Next scan

The first parameter set did not show a Turing window. The next scan should broaden reaction parameters and diffusion ratios, then require all conditions before claiming pattern-mediated rescue: a locally stable ODE coexistence state, at least one unstable spatial mode, measurable nonlinear patterning, and a positive density-normalized threshold difference.
