# Mathbio

Numerical and LaTeX materials for a predator--prey reaction--diffusion project on
pattern-mediated indirect evolutionary rescue.

The current code computes nonlinear predator persistence thresholds in a defended /
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

