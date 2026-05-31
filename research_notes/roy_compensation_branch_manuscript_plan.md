# Manuscript Plan: Homogeneous Compensation Mechanism

## Central Claim

The consolidated paper should argue that indirect evolutionary rescue in the tested Roy-style eco-evolutionary model is driven by a homogeneous compensation branch. The spatial PDE preserves this reaction-level mechanism and is stable to tested spatial perturbations. Persistent spatial-pattern-mediated rescue is not supported in the tested setup.

## What Must Be Shown

- The well-mixed eco-evolutionary ODE supports predator rescue through prey defense evolution.
- The homogeneous compensation branch follows from the interior equilibrium equations.
- The branch exists only under explicit feasibility inequalities and only over the stress interval where \(0<q^*(s)<1\).
- The current branch is locally stable by Routh-Hurwitz conditions.
- The spatial PDE does not destabilize the homogeneous branch through the tested Neumann modes.
- Targeted non-homogeneous PDE perturbations do not generate persistent spatial patterning or persistent basin changes.
- Controlled nonlinear trade-off shapes recover and extend the compensation mechanism while remaining parameter-sensitive.

## Figure Plan

Main text figures should be selective:

1. `fig33_compensation_branch_current.png` for the analytic/numerical branch match.
2. `fig40_compensation_stress_interval.png` for the feasible stress interval.
3. `fig42_routh_hurwitz_terms.png` for local stability conditions.
4. `fig45_pde_mode_growth_rates.png` for PDE spatial-mode stability.
5. `fig57_long_horizon_decision.png` for basin-changing heterogeneous follow-up.
6. `fig59_nonlinear_branch_curves.png` for nonlinear branch examples.
7. `fig64_nonlinear_tradeoff_final_decision.png` for nonlinear extension summary.
8. `fig58_final_mechanism_synthesis.png` for the final mechanism diagram.

Other figures should remain available in supporting reports or appendices rather than overloading the core manuscript.

## Main Text Logic

1. Introduce indirect evolutionary rescue and prey defense evolution.
2. Explain why spatial structure and scalar thresholds were initially considered.
3. Present the ODE and PDE models.
4. Derive the homogeneous compensation branch.
5. State feasibility conditions and current stress interval.
6. Present local stability via Routh-Hurwitz.
7. Test whether the PDE destabilizes the branch or generates persistent spatial patterns.
8. Extend the branch logic to controlled nonlinear trade-off shapes.
9. Interpret the mechanism biologically.
10. State limitations before the conclusion.

## Appendix Logic

Appendix A should contain the algebraic branch derivation.

Appendix B should define the Jacobian summary and Routh-Hurwitz coefficient formulas.

Appendix C should summarize classification protocols and emphasize that the manuscript consolidates existing outputs rather than adding new simulations.

## Claims Allowed

- In the tested parameterization, prey defense evolution supports indirect evolutionary rescue.
- The linear trade-off ODE has an interior compensation branch when the stated feasibility inequalities hold.
- The current branch is locally stable at tested stresses.
- The spatial PDE branch is linearly stable to tested Neumann modes under the current diffusion coefficients.
- Targeted non-homogeneous PDE perturbations do not support persistent spatial-pattern-mediated rescue.
- Controlled nonlinear trade-off tests support compensation beyond the linear case in selected shapes.

## Claims Forbidden

- Spatial patterning causes the rescue mechanism.
- Spatial structure generally amplifies or suppresses rescue.
- The branch is globally stable.
- The result is a general theorem for all nonlinear trade-off forms.
- The model is biologically calibrated or empirically fitted.
- All heterogeneous initial conditions have been exhausted.
- The nonlinear shape grid is a broad parameter scan.

## Remaining Work Before Submission

- Tighten manuscript prose and reduce overlap with older technical reports.
- Decide whether to move some figures into appendices or supplementary material.
- Check all figure captions against the final CSV-backed metrics.
- Add a compact parameter table if the paper is prepared for external submission.
- Improve bibliography only with verified sources; do not add placeholder citations.
- Consider a separate housekeeping cleanup for any remaining manuscript encoding issues.
