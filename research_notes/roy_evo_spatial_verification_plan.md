# Verification Plan for the Roy Eco-Evolutionary Spatial Rescue Model

## Purpose

The goal is to prevent overclaiming. The current results should be treated as a structured case study unless robustness, convergence, and basin-boundary analyses support broader claims.

The project must be organized as proposition -> assumptions -> evidence -> missing evidence -> allowed claim.

This plan is not a manuscript and does not add new simulations. It separates what has actually been shown from what is suggested by one parameterization, what is not yet justified, and what additional analysis is required before publication-level claims.

## Current State of the Project

Current factual results:

- Fixed-defense spatial Roy-style PDE did not produce robust predator rescue.
- The best fixed-defense candidate was classified as transient/numerical.
- ODE eco-evolutionary prey defense model produced indirect evolutionary rescue.
- ODE no-evolution threshold: `0.069448242`.
- ODE evolution threshold: `0.16486816`.
- `Delta_evo_ODE = 0.095419922`.
- Defense frequency `q` decreased in the ODE rescue window from about `0.6726` to about `0.3336`.
- Spatial PDE scalar threshold interpretation failed.
- Spatial PDE showed classifier sensitivity, long transients, continuation dependence, and basin dependence.
- Basin-boundary scan at `0.1584375` gave persistent 20, extinct 7, transient 43.
- Basin-boundary scan at `0.16486816` gave persistent 14, extinct 21, transient 35.
- Current evidence suggests basin-dependent dynamics but does not yet prove a general theorem or robust parameter-region result.

Source-material note: the current checkout contains the synthesis notes and basin CSVs, but `results/roy_pde_evo_continuation.csv` and `results/roy_pde_evo_long_horizon.csv` were not present during this verification-plan pass. The corresponding continuation/long-horizon evidence is therefore currently available through the research notes, not through inspectable CSVs in this checkout.

## Scientific Claims and Their Status

| Claim | Current status | Evidence available | Missing evidence | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|
| Well-mixed prey defense evolution can support indirect evolutionary rescue. | Established within tested ODE parameterization. | ODE thresholds `0.069448242` vs `0.16486816`; `Delta_evo_ODE = 0.095419922`; `q` decreases in rescue window. | Trade-off robustness; equilibrium/stability analysis; sensitivity to `nu`. | In the tested ODE parameterization, prey defense evolution supports indirect evolutionary rescue. | Prey defense evolution generally rescues predators in this model class. |
| Fixed-defense spatial patterning did not robustly rescue predators in the tested setup. | Established for tested fixed-defense setup. | Fixed-defense threshold validation; `D_w/D_u=150` candidate failed mechanism validation and was classified transient/numerical. | Broader fixed-defense parameter space; domain/grid convergence documentation. | In the tested fixed-defense setup, spatial patterning alone did not produce robust rescue. | Spatial patterning cannot rescue predators. |
| Spatial PDE dynamics are not captured by a single scalar threshold in the tested setup. | Strongly supported for tested spatial eco-evolutionary setup. | Classifier sensitivity; long transients; continuation dependence; basin scans. | Standardized classification protocol; complete source CSV availability; visual field evidence. | For the tested spatial eco-evolutionary setup, a single scalar threshold was not a stable descriptor. | PDE thresholds are meaningless in general. |
| Spatial PDE shows bistability/path dependence. | Supported as apparent bistability or basin-dependent dynamics in tested stress range. | Upward/downward continuation differences; persistent and extinct outcomes at same stress in basin scans. | Adaptive basin-boundary refinement; longer horizons for transient-heavy regions; grid/time-step convergence; field snapshots. | The tested spatial PDE shows evidence of basin-dependent and bistable dynamics. | The PDE has rigorously proven bistability. |
| `q0` and `w0` influence basin entry. | Moderately supported. | Focused `q0`-`w0_scale` scan at `0.1584375` and `0.16486816`; both stress grids show persistent and extinct outcomes. | Adaptive separatrix; `n0` dependence; repeated seeds; perturbation-amplitude checks; longer horizons for transient outcomes. | In the focused basin scan, `q0` and `w0` influence basin entry. | `q0` and `w0` fully determine basin entry. |
| Spatial structure generally causes bistability in indirect evolutionary rescue. | Not justified. | One tested spatial eco-evolutionary parameterization suggests basin-dependent outcomes. | Parameter-region mapping; diffusion sensitivity; trade-off-form sensitivity; analytical support. | This tested parameterization motivates basin-based analysis. | Spatial structure generally causes bistability. |
| Spatial structure generally amplifies or suppresses rescue. | Not justified. | Early scalar threshold screen suggested lower PDE threshold, but later diagnostics invalidated scalar-threshold interpretation. | Stable scalar boundary or basin-measure comparison across robust parameter ranges. | The scalar threshold screen was not a stable final descriptor of the spatial system. | Spatial structure suppresses indirect evolutionary rescue; spatial structure amplifies indirect evolutionary rescue; the PDE threshold is lower than the ODE threshold. |

## Proposition 1: ODE prey defense evolution can generate indirect evolutionary rescue

### Statement

In the well-mixed eco-evolutionary model, prey defense evolution can increase the predator mortality stress tolerated by the system.

### Assumptions

- The selected defense-growth/conversion trade-off is meaningful.
- `q`-evolution is governed by the chosen selection gradient.
- Persistence classification is reliable for ODE.

### Current evidence

- No-evolution threshold: `m_c^{ODE,no evo} = 0.069448242`.
- Evolution threshold: `m_c^{ODE,evo} = 0.16486816`.
- `Delta_evo_ODE = 0.095419922`.
- In the ODE rescue window, `q` decreased from about `0.6726` to about `0.3336`.

### Missing evidence

- Robustness across trade-off forms and strengths.
- Analytical equilibrium/stability characterization.
- Sensitivity to evolutionary rate `nu`.

### Required next checks

- ODE trade-off parameter sweep.
- Equilibrium/stability analysis.
- Derive conditions under which `q` decreases after predator stress.

### Allowed claim

In the tested ODE parameterization, prey defense evolution supports indirect evolutionary rescue.

### Not allowed yet

Prey defense evolution generally rescues predators in this model class.

## Proposition 2: Fixed-defense spatial patterning does not produce robust rescue in the tested setup

### Statement

Fixed-defense spatial patterning does not produce robust predator rescue in the tested Roy-style spatial setup.

### Assumptions

- The fixed-defense model and validation stages adequately represent the tested setup.
- The persistence classifier and covariance diagnostic used for the candidate are reliable enough to reject the weak candidate.
- The tested grids/domains cover the intended fixed-defense candidate validation, not all possible spatial regimes.

### Current evidence

- Fixed-defense threshold comparison did not show robust `ODE extinct, PDE persistent` behavior.
- The best weak candidate was `D_w/D_u=150`.
- The `D_w/D_u=150` candidate failed validation.
- Candidate mechanism analysis classified it as transient/numerical rather than a robust covariance-supported mechanism.

### Missing evidence

- Broader fixed-defense parameter space.
- Alternative domains.
- Stronger convergence documentation.

### Required next checks

- Only if fixed-defense claims become central again: targeted fixed-defense robustness checks across domain/grid choices.
- Do not prioritize this unless the project returns to fixed-defense spatial rescue.

### Allowed claim

In the tested fixed-defense setup, spatial patterning alone did not produce robust rescue.

### Not allowed yet

Spatial patterning cannot rescue predators.

## Proposition 3: The spatial eco-evolutionary PDE is not well described by one scalar persistence threshold

### Statement

The spatial eco-evolutionary PDE is not well described by one scalar persistence threshold in the tested setup.

### Assumptions

- The Step 09/10/11/12 diagnostic sequence used comparable model equations and parameterization.
- Tail-window and horizon sensitivity are scientifically relevant, not merely bookkeeping artifacts.
- Continuation and basin-dependence indicate that a single threshold is an unstable descriptor for the tested spatial setup.

### Current evidence

- Initial scalar threshold comparison was later contradicted by focused stress checks.
- Persistence classification was sensitive to tail fraction and time horizon.
- Multi-horizon classification did not produce a clean stable boundary.
- Continuation tests showed direction-dependent outcomes.
- Basin scans found multiple reachable outcomes at the same stress.

### Missing evidence

- Final-field PDE solution visualization.
- Residual/convergence plots for representative trajectories.
- Standardized classification protocol.
- Complete source CSV availability in the current checkout for long-horizon and continuation data.

### Required next checks

- Produce representative spatial snapshots and mean time series for persistent, extinct, and transient outcomes.
- Save normalized PDE RHS residuals and tail metrics alongside classification.
- Standardize and document the persistence classification protocol.

### Allowed claim

For the tested spatial eco-evolutionary setup, a single scalar threshold was not a stable descriptor.

### Not allowed yet

PDE thresholds are meaningless in general.

## Proposition 4: The spatial PDE exhibits bistability or basin-dependent rescue dynamics

### Statement

The spatial PDE exhibits bistability or basin-dependent rescue dynamics in the tested stress range.

### Assumptions

- Persistent and extinct basin labels correspond to distinct asymptotic or long-lived outcomes, not unresolved transients.
- The continuation and basin scans are sufficiently comparable to interpret direction and initial-condition dependence.
- The tested stress range is within the sensitive regime created by the selected parameterization.

### Current evidence

- Upward and downward continuation produced different outcomes at several stresses.
- Initial-condition family scans produced persistent and extinct outcomes at the same stress.
- Stress-regime map classified `0.1584375`, `0.16486816`, and `0.175` as `bistable_persistent_extinct`.
- Focused `q0`-`w0` basin scans also found persistent and extinct outcomes at `0.1584375` and `0.16486816`.

### Missing evidence

- Adaptive basin-boundary refinement.
- Longer-horizon confirmation for transient-heavy regions.
- Grid/time-step convergence for bistable outcomes.
- Representative spatial field snapshots.

### Required next checks

- Verify persistent/extinct basin outcomes at multiple grid sizes and smaller time step.
- Extend horizon for selected transient-heavy basin points.
- Save spatial fields for representative persistent, extinct, and transient runs.
- Quantify whether apparent bistability persists under perturbation/seed changes.

### Allowed claim

The tested spatial PDE shows evidence of basin-dependent and bistable dynamics.

Where asymptotic evidence is incomplete, use: apparent bistability or bistability-like behavior.

### Not allowed yet

The PDE has rigorously proven bistability.

## Proposition 5: Initial defense frequency and predator abundance influence basin entry

### Statement

Initial defense frequency `q0` and predator abundance `w0` influence basin entry.

### Assumptions

- The focused `q0`-`w0_scale` grid represents meaningful initial-condition variation.
- Persistent/extinct labels are sufficiently stable for non-transient grid points.
- Baseline `n0` and perturbation pattern do not dominate the observed basin labels.

### Current evidence

- `q0`-`w0` basin boundary scan varied initial defense frequency and predator abundance scale.
- Two target stresses were both classified as `bistable_persistent_extinct`.
- At `0.1584375`: persistent 20, extinct 7, transient 43, unresolved 0, nonphysical 0.
- At `0.16486816`: persistent 14, extinct 21, transient 35, unresolved 0, nonphysical 0.
- Synthesis notes report persistent basins concentrated at low-to-intermediate `q0` and modulated by predator initial abundance.

### Missing evidence

- Adaptive separatrix.
- `n0`-dependence.
- Perturbation amplitude dependence.
- Repeated seeds.
- Longer horizons for transient outcomes.

### Required next checks

- Adaptive refinement of the `q0`-`w0` separatrix at `0.1584375` and `0.16486816`.
- Repeat selected boundary points with longer horizon and residual checks.
- Test seed and perturbation-amplitude sensitivity.
- Add a small `n0` perturbation check after the `q0`-`w0` separatrix is sharper.

### Allowed claim

In the focused basin scan, `q0` and `w0` influence basin entry.

### Not allowed yet

`q0` and `w0` fully determine basin entry.

## What Is Not Yet General

The current result is not yet general because:

- only one main trade-off parameterization has been studied;
- `(b_u=0.08, b_v=0.02)` were chosen to obtain an interior defense baseline;
- diffusion sensitivity remains untested after basin mapping;
- many basin-scan points remain transient;
- exact separatrix is unresolved;
- PDE fields are not yet visualized sufficiently;
- no analytical bifurcation/stability analysis has been completed.

## Required Evidence Before Publication-Level Claims

Publication-level claims require the following evidence:

1. Representative PDE solution snapshots:
   - `n(x,y,t)`, `w(x,y,t)`, `q(x,y,t)`;
   - persistent, extinct, transient/basin-boundary cases.
2. Spatial mean time series:
   - `\bar n(t)`, `\bar w(t)`, `\bar q(t)`.
3. Continuation diagram:
   - upward vs downward branches.
4. Residual/convergence diagnostics:
   - tail slopes;
   - normalized PDE RHS residual;
   - horizon sensitivity.
5. Adaptive basin-boundary refinement:
   - `q0`-`w0` separatrix at `0.1584375` and `0.16486816`.
6. Robustness:
   - grid/time-step;
   - diffusion coefficients;
   - trade-off strength/form;
   - evolutionary rate `nu`.
7. Analytical support:
   - homogeneous ODE equilibria;
   - linear stability of homogeneous states if feasible;
   - explanation of possible multiple attractors.

## Required PDE Visual Evidence

| Figure | Purpose | Existing file if available | Missing data if any | Required source CSV/field output |
|---|---|---|---|---|
| Figure A: Model schematic. | Define variables and eco-evolutionary feedbacks. | `figures/roy_evo_spatial/report/fig01_model_schematic.png` | None for schematic. | Not data-driven. |
| Figure B: ODE rescue time series and threshold comparison. | Show ODE evolutionary rescue and `q` decline in rescue window. | `figures/roy_evo_spatial/report/fig02_ode_thresholds.png`; `results/roy_evo_ode_threshold_scan.csv`. | Publication-ready time-series panel should be checked against `results/roy_evo_ode_timeseries.csv` if included. | ODE threshold CSV and ODE timeseries CSV. |
| Figure C: Spatial snapshots of `n`, `w`, `q` for representative persistent and extinct outcomes. | Show whether the PDE result has real spatial structure or mostly homogeneous multistability. | Not available in current report figures. | Final/intermediate PDE fields for persistent and extinct cases. | Spatial field arrays or saved snapshot files for `n`, `w`, `q`. |
| Figure D: Spatial mean time series for persistent, extinct, and transient trajectories. | Show trajectory classes and transient behavior. | Some prior time-series CSVs exist for earlier stages, but not a final publication panel. | Representative runs tied to final classification protocol. | Time series with mean `n`, `w`, `q`, tail metrics, residuals. |
| Figure E: Continuation diagram showing upward/downward branches. | Visualize path dependence and hysteresis. | Hysteresis note table exists; current checkout did not include `results/roy_pde_evo_continuation.csv`. | Inspectable continuation CSV or regenerated non-simulation artifact from archived source data. | Upward/downward continuation summary with stress, classification, tail mean, residual. |
| Figure F: Basin-regime map across stress. | Show stress regimes and multiple reachable outcomes. | `figures/roy_evo_spatial/report/fig04_basin_regime_map.png`; `results/roy_pde_evo_basin_initial_condition_scan.csv`. | Add uncertainty/transient markers if needed. | Basin initial-condition scan CSV. |
| Figure G: `q0`-`w0` basin-boundary heatmap. | Show focused basin-boundary dependence on initial defense and predator abundance. | `figures/roy_evo_spatial/report/fig05_basin_boundary_heatmap.png`; `figures/roy_evo_spatial/17_basin_boundary_heatmap.png`; `results/roy_pde_evo_basin_boundary_scan.csv`. | Adaptive refinement needed near separatrix. | Basin-boundary scan CSV and refined scan CSV. |
| Figure H: residual/convergence diagnostic. | Distinguish steady states from long transients and numerical artifacts. | Not available as report figure. | Tail slope, horizon sensitivity, normalized RHS residual per representative run. | Residual/convergence CSV, ideally including final-state RHS norms. |
| Figure I: adaptive separatrix after refinement. | Provide publication-level basin boundary rather than coarse grid. | Not available. | Adaptive refinement data. | Refined `q0`-`w0` scan with classification confidence. |

## Required Numerical Robustness Checks

- Repeat representative basin outcomes at multiple grid sizes.
- Repeat at smaller time step or stricter CFL if explicit scheme.
- Extend time horizon for transient-heavy basin points.
- Save final and intermediate spatial fields.
- Compute normalized PDE RHS residual.
- Repeat basin-boundary scan across seeds/perturbation amplitudes.
- Test diffusion ratios `(D_w/D_n)`, `(D_q/D_n)`.
- Test trade-off strengths `(b_v/b_u)`, `(r_v/r_u)`, `(a_v/a_u)`.

## Required Analytical or Dynamical-Systems Checks

- Analyze ODE nullclines or equilibria.
- Determine when the `q`-selection gradient changes sign:

```text
G(n,w,q)=(r_v-r_u)z-(a_v-a_u)w.
```

- Derive qualitative condition for `q` to decrease after predator stress.
- Identify homogeneous persistent and predator-free equilibria.
- Study local stability of homogeneous equilibria.
- If feasible, relate PDE multistability to homogeneous attractors or diffusion-induced basin deformation.

## Claim Hierarchy

### Tier 1: Established within tested parameterization

- In the tested ODE parameterization, prey defense evolution increases predator mortality tolerance.
- In the tested fixed-defense setup, spatial patterning alone did not produce robust predator rescue.
- In the tested spatial eco-evolutionary setup, scalar PDE threshold language was not stable.

### Tier 2: Strongly suggested but needs robustness

- The tested spatial PDE shows apparent bistability or basin-dependent rescue dynamics.
- Initial defense frequency and predator abundance influence basin entry in the focused scan.
- Basin-based analysis is a better organizing framework than scalar-threshold comparison for this tested setup.

### Tier 3: Hypotheses for future work

- The basin boundary may be organized by a separatrix in `q0`-`w0` space.
- Diffusion may deform basins without necessarily generating a positive spatial covariance mechanism.
- Spatial fields may reveal whether the observed basin dependence is genuinely spatial or mostly homogeneous multistability with diffusion.

### Tier 4: Not allowed claims

- Spatial structure generally causes bistability.
- Spatial structure suppresses indirect evolutionary rescue.
- Spatial structure amplifies indirect evolutionary rescue.
- The PDE threshold is lower than the ODE threshold.
- The PDE has rigorously proven bistability.
- `q0` and `w0` fully determine basin entry.

## Recommended Next Work Packages

### Work Package 1: Save and visualize representative PDE solutions

Deliverables:

- spatial snapshots;
- mean time series;
- residual table.

### Work Package 2: Adaptive `q0`-`w0` basin-boundary refinement

Deliverables:

- refined separatrix;
- reduced transient ambiguity;
- updated heatmap.

### Work Package 3: Numerical robustness of bistability

Deliverables:

- grid/time-step convergence;
- longer horizons;
- perturbation/seed robustness.

### Work Package 4: Trade-off and diffusion sensitivity

Deliverables:

- structured, not broad random, parameter maps.

### Work Package 5: Analytical support

Deliverables:

- ODE equilibria;
- selection-gradient sign analysis;
- stability or bifurcation notes.

## Stop Conditions

- If PDE solution snapshots show no real spatial structure, revise interpretation toward homogeneous multistability rather than spatially organized rescue.
- If longer horizons collapse transient/persistent basins into one outcome, revise bistability claim.
- If grid/time-step convergence fails, stop publication claims and fix numerics.
- If bistability only exists for a tiny tuned parameter region, present it as a narrow case study.
- If trade-off sensitivity destroys the ODE rescue mechanism, revisit model formulation.
