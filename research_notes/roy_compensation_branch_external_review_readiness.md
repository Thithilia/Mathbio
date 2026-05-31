# External Review Readiness Note

## Main improvements

- Added a Methods / Numerical protocols section that separates ODE integration, branch computation, stability checks, PDE spatial-mode tests, finite-amplitude PDE perturbations, nonlinear trade-off shape-grid design, and classification logic.
- Added a main-text parameter table with symbol, meaning, value, and model role for the ODE, PDE, and domain values used in the tested case study.
- Reduced main-text figure architecture to the core mechanism, branch, stability, spatial, long-horizon, and nonlinear-extension figures.
- Added a separate supplement with numerical protocol details, diagnostic figures, and a claim-to-source CSV mapping.
- Added a frozen-\(q\) no-evolution counterfactual for the linear ODE. The manuscript now states indirect evolutionary rescue only over the verified grid window where frozen-\(q\) trajectories are extinct and evolving-\(q\) trajectories persist.
- Added the analytic frozen-\(q\) predator-invasion threshold \(s_{\mathrm{fixed}}\approx0.0696\), saved it to the counterfactual summary CSV, and marked it in the counterfactual figure.
- Added Roy-style/free-space/aposematic-prey source-model citations and clarified that \(q\)-diffusion is a phenomenological composition-smoothing term.
- Added full ODE Jacobian entries and a Routh--Hurwitz margin table to the supplement.
- Added a continuous \(\lambda\)-scan for \(J_F(U^*)-\lambda D\) over the tested Neumann-mode range, strengthening the spatial-stability evidence without claiming all modes are analytically stable.
- Polished the main compensation-branch figure to remove axis-offset ambiguity and replaced the nonlinear trade-off summary with a faceted shape-grid heatmap matching the 11/11/4/1 class counts.
- Removed process/revision language from the main paper, replaced placeholder authors with a clearer draft author-list placeholder, and added an external-review version tag to the data/code statement.
- Redesigned the main PDE evidence figure as a three-panel discrete-mode, continuous-\(\lambda\), and long-horizon-decision figure.
- Polished the no-evolution counterfactual figure with direct \(s_{\mathrm{fixed}}\), rescue-window, and representative \(q^*(s)\) annotations.
- Replaced the main Routh--Hurwitz figure with a clearer margin/eigenvalue summary and moved the full four-term coefficient plot to the supplement.
- Split crowded supplement PDE and nonlinear diagnostic mosaics into larger, one-message-per-figure panels with updated figure references and claim-source mapping; regenerated PDE diagnostics with larger fonts, external legends, and clearer colorbars.
- Made the nonlinear shape-grid heatmap grayscale-safe with larger R/P/N/U cell letters and explicit legend labels.
- Added final cleanup fixes for Figure 6 float placement, sampled continuous-\(\lambda\) wording, fixed-\(q\) threshold interpretation, supplement field-map color-scale captions, and supplement float placement.
- Added `experiments/29_roy_manuscript_audit.py` plus value and DOI audit CSV outputs to check manuscript-reported numerical values and bibliography DOI-field presence against source files.
- Clarified that \(q\)-diffusion is a phenomenological composition-smoothing closure and that PDE spatial-stability conclusions could depend on this closure, while the homogeneous branch and frozen-\(q\) counterfactual do not depend on \(D_q\).
- Clarified that the no-evolution rescue window is reachable from the pre-stress branch initial condition and is not the full analytic branch feasibility interval.
- Added trait-boundary endpoint language for the analytic interior branch and avoided classifying those endpoints as saddle-node or transcritical bifurcations.
- Added a finite-time evolutionary-rate caveat: the branch equilibrium is independent of \(\nu\), but rescue from an initial condition can fail if prey evolution is too slow.
- Added a supplemental no-compensation/unresolved nonlinear-shape failure-mode table generated from existing outputs.
- Verified the Roy et al. DOI `10.1038/s42005-025-02434-1` against the Nature/Communications Physics article page and updated the DOI audit note/status.
- Reformatted supplement source-map tables in landscape with compact path names for readability.
- Strengthened the biological interpretation around lower prey defense, predator conversion opportunity, indirect evolutionary rescue, and the stylized nature of the assumptions.
- Added a Claims and scope section separating established results, controlled extensions, and claims not made.

## Current manuscript status

The manuscript package now consists of a clean main paper, a supplement, and a submission checklist. The main paper is structured as a theoretical ecology / mathematical biology article rather than a project report. The supplement carries diagnostic ODE/PDE figures, nonlinear trade-off diagnostics, full Jacobian entries, Routh--Hurwitz margins, and the CSV source map needed for review traceability. The data/code statement contains a repository URL and the external-review version tag `external-review-polish-2026-05-31`; a DOI is still needed before journal submission.

## Remaining weaknesses

- Main figures are cleaner, but some are still generated from diagnostic workflows and may need journal-specific redesign before formal submission.
- The nonlinear trade-off extension is controlled and local, not a theorem over all nonlinear forms.
- PDE perturbation tests are targeted and do not exhaust all heterogeneous initial conditions.
- The continuous \(\lambda\)-scan strengthens the tested modal range but is still not an analytic proof for all diffusion coefficients or all spatial settings.
- The no-evolution rescue window is a finite grid/horizon result, not a complete bifurcation analysis of all frozen-\(q\) and evolving-\(q\) basins.
- Finite-time rescue can depend on the evolutionary-rate parameter \(\nu\); this manuscript states the caveat but does not add a new \(\nu\)-sensitivity simulation.
- The \(q\)-diffusion term is a phenomenological closure rather than a derivation from separate defended/undefended prey densities.
- The repository has not yet been archived as a release with a DOI.
- The model has no empirical calibration or species-specific parameter fitting.
- The Roy et al. DOI was resolver-checked; other references should be rechecked if the bibliography changes before submission.

## Recommended target article type

mathematical biology / theoretical ecology short article or methods-style theoretical note, not an empirical ecology paper.

## What should not be claimed

- Spatial patterning causes the rescue mechanism.
- Spatial structure generally amplifies or suppresses indirect evolutionary rescue.
- The compensation branch is globally stable.
- The nonlinear extension is fully general.
- All heterogeneous perturbations decay.
- The model makes empirical predictions without calibration.

## What to check before actual submission

- Recheck every numerical value and caption against the final CSV outputs.
- Run `python experiments/29_roy_manuscript_audit.py --profile focused` after any future manuscript or CSV edit.
- Decide the target outlet and adapt figure count, figure dimensions, and reference style.
- Convert diagnostic figures into publication-ready panels if needed.
- Add only verified literature if the framing is expanded.
- Decide whether a small \(\nu\)-sensitivity diagnostic is worth adding for the target outlet; it is currently deferred to avoid expanding the evidence package.
- Replace placeholder author information with the intended author list and affiliations.
