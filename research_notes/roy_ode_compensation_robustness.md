# Robustness of the Homogeneous Compensation Mechanism in the Roy ODE

## Purpose

This note tests whether the stable persistent homogeneous branch identified after PR #19 follows from the ODE equilibrium equations and whether it persists under a local structured trade-off perturbation. The analysis is ODE-only and does not run PDE scans or change model equations.

## Analytic Compensation Branch

For an interior equilibrium with \(n>0\), \(w>0\), and \(0<q<1\), the ODE equilibrium conditions imply \(w=c z\), where \(c=(r_v-r_u)/(a_v-a_u)\). Because the growth and palatability trade-offs are linear, \(r(q)-c a(q)=r_u-c a_u\), giving \(z^*=\xi/(r_u-c a_u)\), \(w^*=c z^*\), and \(n^*=\kappa^{-1}-z^*-w^*\). The predator equation then determines the required conversion value \(b_{req}(s)\), and the linear conversion trade-off gives \(q^*(s)=(b_{req}(s)-b_u)/(b_v-b_u)\).

The branch exists as an interior compensation branch when \(n^*>0\), \(w^*>0\), \(z^*>0\), and \(0<q^*(s)<1\).

## Match to Numerical Equilibria

In the current parameterization, \(q^*(s)\) decreases with stress while \(n^*\) and \(w^*\) remain approximately fixed. Analytic branch values are:

s=0: q*=0.6726, s=0.069448242: q*=0.4717, s=0.11765625: q*=0.3322, s=0.1584375: q*=0.2142, s=0.16486816: q*=0.1955, s=0.175: q*=0.1662

The maximum absolute difference in \(q^*\) between the analytic branch and the existing stable persistent numerical equilibria is `7.688294445529209e-15`. The comparison table is `results/roy_ode_compensation_branch_current.csv`, and the figure is `figures/roy_evo_spatial/report/fig33_compensation_branch_current.png`.

## Stress Interval of the Interior Branch

For the current parameterization, the branch is interior between the stress values where \(q^*=1\) and \(q^*=0\):

- `s_at_q_equals_1 = -0.11313888888888887`
- `s_at_q_equals_0 = 0.23244444444444454`
- `interior_stress_interval_length = 0.3455833333333334`

This interval is the homogeneous compensation range in which changing \(q^*\) can offset increased mortality while retaining a positive predator equilibrium.

## Local Trade-Off Robustness

The structured local grid varies \(r_v\), \(a_v\), \(b_u\), and \(b_v/b_u\) around the current trade-off values. It is not a global parameter scan. The grid has `81` parameter sets; `72` have a compensation branch with a nonzero interval overlapping nonnegative stress, for a fraction `0.8888888888888888`.

The grid output is `results/roy_ode_compensation_tradeoff_grid.csv`, and the summary figure is `figures/roy_evo_spatial/report/fig34_compensation_interval_tradeoff_grid.png`.

## Selected Basin Maps

Selected ODE basin maps were computed for the current parameterization and two local trade-off variants. These ODE diagnostics use the same \(q_0\)--\(w_0\) grid as the earlier basin comparisons and do not run PDE scans.

- `current`: extinct_basin=28, persistent_basin=48, transient_basin=64
- `shorter_interval_present`: persistent_basin=117, transient_basin=23
- `weak_or_absent_branch`: extinct_basin=140

The table is `results/roy_ode_compensation_selected_basin_maps.csv`, and the figure is `figures/roy_evo_spatial/report/fig35_selected_tradeoff_basin_maps.png`.

## Selected Equilibria and Stability

Finite-difference Jacobian stability diagnostics were computed for the same selected parameter sets.

- `current`: stable equilibria found = 6
- `shorter_interval_present`: stable equilibria found = 5
- `weak_or_absent_branch`: stable equilibria found = 7

The table is `results/roy_ode_compensation_selected_equilibria.csv`, and the figure is `figures/roy_evo_spatial/report/fig36_selected_equilibria_stability.png`.

## Final Label

`ode_compensation_branch_supported`

## Interpretation

The analytic compensation branch matches the current numerical persistent equilibria and is present across a substantial fraction of the local structured trade-off grid. In the current parameterization, the branch is derived from the ODE equilibrium equations: as mortality stress increases, \(q^*(s)\) shifts downward, increasing predator conversion opportunity and restoring predator growth balance at a positive predator equilibrium. This explains the homogeneous reaction-level compensation mechanism behind indirect rescue.

## What Is Still Not General

The robustness check is local and structured. It does not establish global behavior over all trade-off forms, diffusion settings, evolutionary rates, or biological calibrations. It also does not replace the earlier conclusion that the spatial PDE is currently best interpreted as preserving basin-dependent homogeneous dynamics rather than generating persistent spatial-pattern-mediated rescue.

## Files

- `experiments/22_roy_ode_compensation_robustness.py`
- `results/roy_ode_compensation_branch_current.csv`
- `results/roy_ode_compensation_interval_current.csv`
- `results/roy_ode_compensation_tradeoff_grid.csv`
- `results/roy_ode_compensation_selected_basin_maps.csv`
- `results/roy_ode_compensation_selected_equilibria.csv`
- `results/roy_ode_compensation_robustness_summary.csv`
- `figures/roy_evo_spatial/report/fig33_compensation_branch_current.png`
- `figures/roy_evo_spatial/report/fig34_compensation_interval_tradeoff_grid.png`
- `figures/roy_evo_spatial/report/fig35_selected_tradeoff_basin_maps.png`
- `figures/roy_evo_spatial/report/fig36_selected_equilibria_stability.png`
- `figures/roy_evo_spatial/report/fig37_compensation_mechanism_diagram.png`

## Next Step

Use the analytic branch conditions to design narrower trade-off hypotheses before any broader parameter work. The next mathematical step is to connect branch existence and stability to explicit inequalities in the trade-off parameters.
