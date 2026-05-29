# Existence and Stability Conditions for the Roy ODE Compensation Branch

## Purpose

This note derives conditional existence and local-stability conditions for the homogeneous compensation branch in the linear trade-off Roy eco-evolutionary ODE. It is ODE-only and does not run PDE simulations or change model equations.

## Interior Equilibrium Conditions

For an interior equilibrium with \(n>0\), \(w>0\), and \(0<q<1\), the nontrivial equilibrium conditions are prey balance, predator balance, and zero selection gradient. With \(\Delta r=r_v-r_u\), \(\Delta a=a_v-a_u\), and \(\Delta b=b_v-b_u\), the selection-gradient condition is \(\Delta r z-\Delta a w=0\).

## Analytic Compensation Branch

If \(c=\Delta r/\Delta a>0\), then \(w^*=cz^*\). Because the trade-offs are linear, \(r(q)-c a(q)=r_u-ca_u\), so the prey equation gives

```text
z* = xi / (r_u - c a_u)
w* = c z*
n* = kappa^-1 - z* - w*
```

The predator equation then determines

```text
b_req(s) = (m+s+mu w*)/(n* z*)
q*(s) = (b_req(s)-b_u)/(b_v-b_u)
```

## Feasibility Inequalities

The branch exists only when the feasibility inequalities hold: `c>0`, `r_u-c a_u>0`, `z*>0`, `w*>0`, `n*>0`, and `0<q*(s)<1`. The current parameterization satisfies these conditions at all target stress values. The current-condition table is `results/roy_ode_compensation_conditions_current.csv`.

## Stress Interval

Solving the branch formula for stress gives

```text
s(q) = n* z* [b_u + Delta b q] - m - mu w*
```

so the interior branch is feasible on the open interval between `s_q0` and `s_q1`. For the current parameterization this interval is:

- low endpoint: `-0.11313888888888887`
- high endpoint: `0.23244444444444454`
- length: `0.3455833333333334`

The formula table is `results/roy_ode_compensation_stress_interval_formula.csv`, and the interval figure is `figures/roy_evo_spatial/report/fig40_compensation_stress_interval.png`.

## Local Stability

An analytic Jacobian was implemented for the ODE. On the current interior branch, the branch is locally stable at all tested target stresses. The tested eigenvalue summaries are:

- s=0: max Re(lambda)=-0.0122735, stable
- s=0.069448242: max Re(lambda)=-0.0223825, stable
- s=0.11765625: max Re(lambda)=-0.0235817, stable
- s=0.1584375: max Re(lambda)=-0.0158939, stable
- s=0.16486816: max Re(lambda)=-0.0146525, stable
- s=0.175: max Re(lambda)=-0.0126816, stable

The stability table is `results/roy_ode_compensation_stability_current.csv`, and the figure is `figures/roy_evo_spatial/report/fig39_compensation_stability_along_branch.png`.

## Condition Grid

A structured local analytic grid varied \(r_v\), \(a_v\), \(b_u\), and \(b_v/b_u\). This is a condition-grid evaluation, not a simulation scan. Counts by condition class are:

- `valid_stable_compensation_branch`: 107
- `valid_unstable_compensation_branch`: 10
- `valid_branch_outside_target_stress`: 108
- `invalid_tradeoff_geometry`: 0
- `invalid_feasibility`: 0

The condition grid is `results/roy_ode_compensation_condition_grid.csv`, and the figure is `figures/roy_evo_spatial/report/fig38_compensation_conditions_region.png`.

## Final Label

`compensation_conditions_derived_and_supported`

## Interpretation

The current compensation branch is now a conditional mathematical statement for the linear trade-off ODE: if the feasibility inequalities hold and \(0<q^*(s)<1\), then the branch exists; if the analytic Jacobian has negative maximum real eigenvalue, it is locally stable. In the tested parameterization, these conditions hold at the target stresses, and the analytic branch matches the numerical branch to machine precision.

## What This Proves

This derives the branch equations and feasibility conditions for the linear trade-off ODE and verifies local stability for the tested current branch. It is stronger than the earlier numerical observation because it explains why \(n^*\) and \(w^*\) remain fixed while \(q^*(s)\) shifts with stress.

## What This Does Not Prove

This is not a global theorem for all trade-off forms or all parameter values. It does not establish global basin geometry, global stability, or spatial-pattern-mediated mechanisms. It also does not test PDE dynamics.

## Files

- `experiments/23_roy_ode_compensation_conditions.py`
- `results/roy_ode_compensation_conditions_current.csv`
- `results/roy_ode_compensation_stress_interval_formula.csv`
- `results/roy_ode_compensation_stability_current.csv`
- `results/roy_ode_compensation_condition_grid.csv`
- `results/roy_ode_compensation_conditions_summary.csv`
- `figures/roy_evo_spatial/report/fig38_compensation_conditions_region.png`
- `figures/roy_evo_spatial/report/fig39_compensation_stability_along_branch.png`
- `figures/roy_evo_spatial/report/fig40_compensation_stress_interval.png`
- `figures/roy_evo_spatial/report/fig41_compensation_conditions_schematic.png`

## Next Step

Use these inequalities to state sharper parameter hypotheses before any further numerical exploration. The next mathematical step is a local Routh-Hurwitz or symbolic stability analysis of the branch Jacobian.
