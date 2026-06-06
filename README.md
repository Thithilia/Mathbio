# Mathbio

Numerical and LaTeX materials for a predator--prey reaction--diffusion project on
indirect evolutionary rescue.

## Current compensation-branch manuscript

The current manuscript is titled "A Homogeneous Compensation Branch for Indirect
Evolutionary Rescue in a Reduced Free-Space Predator--Prey Model." Its supported
claim is conservative: the tested reduced free-space model supports a
homogeneous compensation branch, and a frozen-`q` counterfactual provides
evidence for indirect evolutionary rescue over the verified deterministic rescue
window. The threshold-safe window is treated as the more conservative ecological
interpretation because the extinction threshold is a deterministic
quasi-extinction diagnostic, not an empirical extinction probability.

Sampled spatial diagnostics and targeted finite-amplitude perturbation tests did
not provide evidence for persistent spatial-pattern-mediated rescue in the
tested reduced setup. This does not claim that the reduced model is equivalent to
the full Roy/Sasmal defended-prey PDE, and it is not evidence against spatial
pattern formation in those fuller systems.

Reproducibility packaging is still pending before journal submission: archive an
exact repository snapshot with a DOI, record the exact manuscript commit hash,
freeze the computational environment, remove the draft author placeholder,
perform a final DOI/reference check, and ensure CSV outputs, scripts, figures,
and source mappings are included in the archive.

## Legacy project materials

Earlier code computes nonlinear predator persistence thresholds in a defended /
undefended prey model:

- `src/turing_rescue_model.py`: model parameters, reaction terms, coexistence
  equilibria, Jacobian, local stability, and Turing spectral scan.
- `src/simulate_pde_1d.py`: 1D method-of-lines PDE solver, ODE/PDE threshold
  bisection, and persistence diagnostics.
- `experiments/01_turing_window_and_thresholds.py`: first runnable numerical
  experiment.
- `nonlinear_pde_results.md`: summary of the first nonlinear PDE threshold run.
- `turing_rescue_latex/main.tex`: current research report source.

Run the checks:

```bash
pytest -q
```

Run the first experiment:

```bash
python experiments/01_turing_window_and_thresholds.py
```

The analytic prey-only invasion threshold in the code is only a linear sanity check.
The nonlinear thresholds `m_c^ODE` and `m_c^PDE` are computed numerically from
long-time persistence criteria.
