# Nonlinear PDE Results 07

**Final conclusion: inconclusive candidate, not claimed.** A threshold-shift candidate appeared, but it did not survive the full validation/escalation rule; no rescue or inhibition claim is made.

## Core Criterion

The decisive quantity is `Delta m_c = m_c_PDE - m_c_ODE`, where `m_c_ODE` is the mortality-stress threshold for predator persistence in the well-mixed ODE and `m_c_PDE` is the corresponding threshold in the spatial PDE.

- `Delta m_c > 0`: spatial structure expands the predator-persistence / indirect-rescue range.
- `Delta m_c < 0`: spatial structure shrinks that range.
- values within the row tolerance are treated as no measurable threshold effect.

## Persistence Rule

Persistence is evaluated on the final 25% of the predator-density trajectory. A trajectory is persistent only if all three conditions hold:

- `tail_mean > epsilon`
- `tail_min > 0.25 * epsilon`
- `tail_slope >= -max(epsilon, 0.25 * tail_mean) / max(tail_duration, 1e-12)`

PDE runs are also rejected as nonpersistent if negative state values, negative free space `z`, or nonfinite diagnostic time series are detected. ODE runs are rejected if integration fails or produces nonfinite output.

## Stage A Results

- Stage A rows: `13`
- positive candidates: `12`
- negative candidates: `0`
- largest positive Stage A `Delta m_c`: `0.00842285`
- largest negative Stage A `Delta m_c`: `nan`
- closest-to-zero row: `mu=0.6|DwDu=100|eta=0.005|gamma=3.73|beta1=0.5` with `Delta m_c = -0.00012207`

Stage A is candidate discovery only; it is not used as evidence for rescue or inhibition.

## Stage B/C Validation

- Stage B precision-screen rows: `13`
- Stage C seed-validation rows: `6`
- Stage D grid-escalation rows: `1`

| group_id | mu | D_w/D_u | eta | Stage C seeds | mean Delta | Delta range | interval range | group conclusion | Stage D result | final conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| mu=0.85\|DwDu=100\|eta=0.005\|gamma=3.73\|beta1=0.5 | 0.85 | 100 | 0.005 | 3 | -0.000153068 | [-0.000153068, -0.000153068] | [-0.000653068, 0.000346932] | no_measurable_effect | nan (not_required) | no_measurable_effect |
| mu=0.85\|DwDu=150\|eta=0.005\|gamma=3.73\|beta1=0.5 | 0.85 | 150 | 0.005 | 3 | 0.000724594 | [0.000724594, 0.000724594] | [0.000224594, 0.00122459] | rescue_supported | 0.000244583 (ok) | inconclusive_candidate |

## Final Conclusion

**Final conclusion: inconclusive candidate, not claimed.** The final classification is based on the group summary intervals in `results/roy_2d_threshold_group_summary.csv`, not on pattern morphology or sparse stress classification counts.

## Secondary Diagnostics

Pattern morphology, Fourier power, and dominant wavelength remain exploratory diagnostics. They are not part of the rescue criterion in this PR.

Outputs:

- `results\roy_2d_threshold_comparison.csv`
- `results\roy_2d_threshold_group_summary.csv`
- `figures\roy_2d_longtime\07_threshold_delta.png`
