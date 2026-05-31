# Nonlinear Trade-Off Extension of the Homogeneous Compensation Mechanism

## Purpose

This note tests whether the homogeneous compensation mechanism remains visible under controlled endpoint-preserving nonlinear trade-off forms. It extends the previous linear-trade-off derivation without changing the baseline model implementation.

## Generalized Trade-Off Model

The nonlinear forms are

```text
r(q) = r_u + (r_v-r_u) q^gamma_r
a(q) = a_u + (a_v-a_u) q^gamma_a
b(q) = b_u + (b_v-b_u) q^gamma_b
```

The controlled shape grid uses gamma values `0.5`, `1.0`, and `2.0`, representing concave, linear, and convex endpoint-preserving trade-offs. This is a local structured shape check, not a broad random scan.

## Generalized Compensation Branch

For nonlinear trade-offs, the branch is naturally parameterized by `q`. The selection-gradient condition gives

```text
c(q) = r'(q) / a'(q)
```

when `a'(q)` is nonzero. Then

```text
z(q) = xi / [r(q) - a(q)c(q)]
w(q) = c(q)z(q)
n(q) = kappa^-1 - z(q) - w(q)
s(q) = b(q)n(q)z(q) - m - mu w(q)
```

A target stress has an interior branch state when a feasible `q` satisfies `s(q)=s0`.

## Recovery of the Linear Case

The gamma `(1,1,1)` case recovers the previously derived linear branch. The maximum absolute recovery difference in `q*` is `3.1633029529132273e-13`.

## Shape-Grid Branch and Stability Results

The 27-combination shape grid produced:

- robust compensation shapes: `11`
- partial compensation shapes: `11`
- no-compensation shapes: `4`
- unresolved shapes: `1`

Concave, convex, and mixed shape choices can change the stress interval, branch feasibility, and local stability. This should be interpreted as local parameter sensitivity rather than a global conclusion about all nonlinear trade-offs.

## Selected ODE Basin Maps

Selected ODE q0-w0 basin maps were generated for the linear reference, all-concave, all-convex, one mixed case, and a weak or failed shape. These maps test whether basin-dependent outcomes remain visible in homogeneous dynamics under selected nonlinear shape choices.

## PDE Spatial Stability for Selected Shapes

PDE spatial-mode stability was evaluated using `J_F(U*) - lambda_mn D` for selected nonlinear branch states. Stable selected branch rows: `16` of `16`. This remains a linear spatial-mode test around homogeneous branch states.

## Non-Homogeneous PDE Perturbation Tests

Targeted nonlinear PDE tests used homogeneous controls, local defense patches, random heterogeneity, and basin-boundary heterogeneity. Finite-horizon rows with basin changes relative to controls: `1`. Finite-horizon rows with final spatial CV above threshold: `1`. After using the longest available horizon for each heterogeneous case, terminal basin changes relative to controls: `0`. Terminal persistent spatial-pattern rows above threshold: `0`.

## Final Label

`nonlinear_tradeoff_compensation_supported`

## Interpretation

The controlled nonlinear extension supports compensation branches beyond the linear case, with targeted PDE tests showing no persistent spatial-pattern-mediated rescue.

## Biological Meaning

The compensation interpretation is that evolving defense can reduce predator resistance as mortality stress increases, thereby increasing conversion opportunity enough to preserve a positive predator equilibrium when the branch exists and is stable. Nonlinear trade-off shape changes can alter whether this compensation path is feasible.

## What Is Supported

- The generalized nonlinear formulation recovers the linear compensation branch when all shape exponents equal 1.
- The `q`-parameterized branch gives a reproducible way to test nonlinear trade-off shapes.
- Selected nonlinear shape cases can support feasible locally stable compensation branches.
- Targeted PDE checks do not by themselves show persistent spatial-pattern-mediated rescue for the selected stable branch cases.

## What Is Not Supported

- A global theorem over all nonlinear trade-off forms.
- A biological calibration of concave or convex shape exponents.
- A broad PDE parameter scan.
- A claim that spatial patterning is the rescue mechanism.

## Remaining Caveats

- The shape grid is small and structured.
- PDE tests are targeted and selected from branch diagnostics.
- Finite-difference Jacobians are numerical local-stability evidence, not symbolic proof.
- Nonlinear trade-off shapes near endpoints require safe derivative clipping.

## Files

- `experiments/27_roy_nonlinear_tradeoff_compensation.py`
- `results/roy_nonlinear_tradeoff_branch_recovery_linear.csv`
- `results/roy_nonlinear_tradeoff_branch_stability_grid.csv`
- `results/roy_nonlinear_tradeoff_shape_summary.csv`
- `results/roy_nonlinear_tradeoff_selected_ode_basin_maps.csv`
- `results/roy_nonlinear_tradeoff_pde_spatial_stability.csv`
- `results/roy_nonlinear_tradeoff_pde_nonhomogeneous_summary.csv`
- `results/roy_nonlinear_tradeoff_compensation_decision.csv`
- `figures/roy_evo_spatial/report/fig59_nonlinear_branch_curves.png`
- `figures/roy_evo_spatial/report/fig60_nonlinear_shape_grid_summary.png`
- `figures/roy_evo_spatial/report/fig61_nonlinear_selected_ode_basin_maps.png`
- `figures/roy_evo_spatial/report/fig62_nonlinear_pde_spatial_stability.png`
- `figures/roy_evo_spatial/report/fig63_nonlinear_nonhomogeneous_pde_tests.png`
- `figures/roy_evo_spatial/report/fig64_nonlinear_tradeoff_final_decision.png`

## Next Step

Use the nonlinear branch diagnostics to identify analytically interpretable shape regimes before considering any broader PDE work.

nonlinear_tradeoff_compensation_supported
