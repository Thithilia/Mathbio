# Nonlinear PDE Results 08: Candidate Mechanism

**Mechanism conclusion: `transient_or_numerical_candidate`.** The candidate did not retain a positive high-grid covariance signal and Stage D already failed to preserve the threshold sign beyond tolerance.

## Question

PR #3 found an inconclusive threshold-shift candidate at `D_w/D_u=150`: Stage C was positive, but Stage D did not preserve the sign beyond tolerance. This analysis asks whether that signal has a real spatial-growth mechanism.

## Diagnostic quantity

For each PDE snapshot, the predator growth term is decomposed as:

`spatial_growth = mean(w * A(u, v, w, z))`

`meanfield_growth = mean(w) * A(mean(u), mean(v), mean(w), mean(z))`

`spatial_covariance_bonus = spatial_growth - meanfield_growth`

where `A = ((beta1*u + beta2*v)/(1+u+v))*z - delta - mu*w` under the stressed parameters.

A robust spatial-rescue mechanism requires a positive tail covariance bonus that also survives threshold validation. Positive covariance alone is not enough.

## Runs

- profile: `focused`
- run summaries: `7`

| run_id | label | D_w/D_u | grid | L | stress | tail mean w | tail covariance bonus | threshold reference | diagnosis |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| M_001 | baseline_64_near_threshold | 100 | 64x64 | 20 | 0.395761 | 0.017707 | 1.14154e-19 | no_measurable_effect | positive_covariance_without_threshold_support |
| M_002 | candidate_64_near_threshold | 150 | 64x64 | 20 | 0.396621 | 0.0164538 | -3.66961e-19 | inconclusive_candidate | tail_predator_not_persistent |
| M_003 | candidate_96_grid_escalation | 150 | 96x96 | 20 | 0.396621 | 0.013417 | -7.62876e-20 | inconclusive_candidate | tail_predator_not_persistent |
| M_004 | ratio_120_64_same_stress_window | 120 | 64x64 | 20 | 0.396621 | 0.0164115 | -1.17845e-18 | none | tail_predator_not_persistent |
| M_005 | ratio_140_64_same_stress_window | 140 | 64x64 | 20 | 0.396621 | 0.0164043 | -1.25926e-19 | none | tail_predator_not_persistent |
| M_006 | ratio_160_64_same_stress_window | 160 | 64x64 | 20 | 0.396621 | 0.0164461 | -8.13152e-20 | none | tail_predator_not_persistent |
| M_007 | ratio_180_64_same_stress_window | 180 | 64x64 | 20 | 0.396621 | 0.0164164 | 2.50066e-19 | none | tail_predator_not_persistent |

## Interpretation rule

- If the covariance bonus is positive but the threshold pipeline remains `inconclusive_candidate`, the result is a weak spatial mechanism, not rescue.
- If covariance vanishes or changes sign under higher grid/domain checks, the candidate is interpreted as transient or numerical.
- Only a positive covariance signal plus validated positive `Delta m_c` would support a spatial-rescue mechanism.

## Outputs

- `results\roy_2d_candidate_mechanism.csv`
- `results\roy_2d_candidate_convergence.csv`
