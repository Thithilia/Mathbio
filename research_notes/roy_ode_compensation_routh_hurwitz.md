# Routh-Hurwitz Stability Conditions for the Roy ODE Compensation Branch

## Purpose

This note turns local stability of the homogeneous compensation branch from numerical eigenvalue reporting into Routh-Hurwitz coefficient inequalities for the cubic characteristic polynomial of the analytic Jacobian. It is ODE-only and does not run PDE simulations or change model equations.

## Compensation Branch Jacobian

At the interior compensation branch, the ODE balances satisfy `R=0`, `P=0`, and `G=0`. The analytic Jacobian therefore uses the simplified entries

```text
J11 = n R_n
J12 = n R_w
J13 = 0
J21 = w P_n
J22 = w P_w
J23 = w P_q
J31 = nu q(1-q) G_n
J32 = nu q(1-q) G_w
J33 = 0
```

where the derivatives are those of the linear trade-off ODE. The implementation reuses the Step 23 analytic Jacobian.

## Characteristic Polynomial

For the Jacobian \(J\), the characteristic polynomial is written as

```text
det(lambda I - J) = lambda^3 + A1 lambda^2 + A2 lambda + A3
```

with

```text
A1 = -tr(J)
A2 = 0.5 * ((tr J)^2 - tr(J^2))
A3 = -det(J)
```

These coefficients are computed directly from the analytic Jacobian.

## Routh-Hurwitz Conditions

For a cubic polynomial, local stability requires `A1>0`, `A2>0`, `A3>0`, and `A1*A2>A3`. The last inequality is recorded as the Routh-Hurwitz margin `A1A2_minus_A3`.

## Current Parameterization

For the current parameterization, all target stresses satisfy the Routh-Hurwitz inequalities and agree with the direct analytic eigenvalue stability classification.

- s=0: A1=3.946775, A2=0.3367315, A3=0.003540184, A1A2-A3=1.325463, stable=True
- s=0.069448242: A1=4.324127, A2=0.2752758, A3=0.00400629, A1A2-A3=1.186321, stable=True
- s=0.11765625: A1=4.586068, A2=0.214856, A3=0.003566306, A1A2-A3=0.9817778, stable=True
- s=0.1584375: A1=4.807656, A2=0.1523808, A3=0.002705573, A1A2-A3=0.729889, stable=True
- s=0.16486816: A1=4.842598, A2=0.1415787, A3=0.002528977, A1A2-A3=0.6830796, stable=True
- s=0.175: A1=4.89765, A2=0.1240339, A3=0.002228151, A1A2-A3=0.6052465, stable=True

The current table is `results/roy_ode_compensation_routh_hurwitz_current.csv`, and the coefficient figure is `figures/roy_evo_spatial/report/fig42_routh_hurwitz_terms.png`.

## Structured Condition Grid

The structured local grid is the same local trade-off grid used in the compensation-condition analysis. It evaluates analytic branch existence and Routh-Hurwitz stability at the two target rescue stresses, not simulated trajectories.

- `branch_stable`: 217
- `branch_unstable`: 18
- `no_branch`: 215
- `rh_eigenvalue_disagreement`: 0

The grid table is `results/roy_ode_compensation_routh_hurwitz_condition_grid.csv`, and the condition-region figure is `figures/roy_evo_spatial/report/fig43_stability_condition_region.png`.

## Final Label

`routh_hurwitz_conditions_supported`

## Interpretation

The Routh-Hurwitz inequalities support local stability of the current compensation branch throughout the tested target stresses. The condition-grid result shows that branch existence and Routh-Hurwitz stability hold in a nontrivial local region of the structured trade-off grid. This strengthens the homogeneous rescue interpretation for the tested linear trade-off ODE.

## What This Adds Beyond Eigenvalue Checks

The previous analysis reported analytic Jacobian eigenvalues. This analysis expresses the same local-stability decision as explicit cubic coefficient inequalities. Agreement between the Routh-Hurwitz conditions and eigenvalue stability checks provides an internal consistency check.

## What This Does Not Prove

This remains a local stability result, not a global basin proof. It does not establish global attraction, spatial-pattern-mediated mechanisms, or behavior under nonlinear trade-off forms. It also does not test PDE dynamics.

## Files

- `experiments/24_roy_ode_compensation_routh_hurwitz.py`
- `results/roy_ode_compensation_routh_hurwitz_current.csv`
- `results/roy_ode_compensation_routh_hurwitz_condition_grid.csv`
- `results/roy_ode_compensation_routh_hurwitz_summary.csv`
- `figures/roy_evo_spatial/report/fig42_routh_hurwitz_terms.png`
- `figures/roy_evo_spatial/report/fig43_stability_condition_region.png`
- `figures/roy_evo_spatial/report/fig44_routh_hurwitz_margin.png`

## Next Step

Use the Routh-Hurwitz margins to identify which trade-off terms control loss of local stability before doing any new numerical exploration.

routh_hurwitz_conditions_supported
