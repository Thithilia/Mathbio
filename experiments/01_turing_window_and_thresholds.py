"""First nonlinear threshold experiment for the defended-prey rescue model.

Run from the repository root:

    python experiments/01_turing_window_and_thresholds.py

The script intentionally treats the analytic prey-only invasion threshold as a sanity
check only.  The nonlinear thresholds are computed from long-time ODE/PDE simulations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.simulate_pde_1d import (
    classify_delta,
    compute_mc_ode,
    compute_mc_pde,
    default_threshold_initial_state,
    simulate_pde,
)
from src.turing_rescue_model import (
    RescueParams,
    jacobian_reaction,
    ode_local_stability,
    prey_only_invasion_threshold,
    require_single_equilibrium,
    turing_scan,
)


FIG_DIR = ROOT / "figures"
RESULTS_MD = ROOT / "nonlinear_pde_results.md"


def baseline_params() -> RescueParams:
    return RescueParams(
        r_U=1.2,
        r_D=0.6,
        a_U=1.4,
        a_D=0.35,
        K=1.0,
        e=0.55,
        m=0.22,
        mu_UD=0.04,
        mu_DU=0.04,
        delta_U=1.0e-3,
        delta_D=5.0e-2,
        delta_P=5.0e-1,
        L=20.0,
        n_x=50,
    )


def plot_final_profiles(results_by_m: dict[float, object], path: Path) -> None:
    fig, axes = plt.subplots(len(results_by_m), 1, figsize=(7.0, 2.4 * len(results_by_m)), sharex=True)
    if len(results_by_m) == 1:
        axes = [axes]
    for ax, (m_value, result) in zip(axes, results_by_m.items()):
        ax.plot(result.x, result.U, label="Undefended prey U", lw=1.8)
        ax.plot(result.x, result.D, label="Defended prey D", lw=1.8)
        ax.plot(result.x, result.P, label="Predator P", lw=1.8)
        ax.set_ylabel("density")
        ax.set_title(f"m = {m_value:.3f}, B_P(T) = {result.diagnostics.B_P:.3e}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("x")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_biomass_timeseries(results_by_m: dict[float, object], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for m_value, result in results_by_m.items():
        ax.plot(result.t, result.predator_biomass_time, label=f"m={m_value:.3f}", lw=1.8)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$B_P(t)=\int_\Omega P(x,t)\,dx$")
    ax.set_yscale("symlog", linthresh=1.0e-6)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase_diagram(m_grid: np.ndarray, ratio_grid: np.ndarray, persistent: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(
        persistent.astype(float),
        origin="lower",
        aspect="auto",
        extent=[ratio_grid[0], ratio_grid[-1], m_grid[0], m_grid[-1]],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"diffusion ratio $\delta_P/\delta_U$")
    ax.set_ylabel("predator mortality m")
    ax.set_title("PDE predator persistence at final time")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["lost", "persistent"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    params = baseline_params()
    m_low = 0.10
    m_high = 0.70
    epsilon = 1.0e-4
    T = 80.0
    perturbation = 1.0e-3
    seed = 4

    eq = require_single_equilibrium(params)
    J = jacobian_reaction(eq.U, eq.D, eq.P, params)
    ode_stable, ode_eigs = ode_local_stability(J)
    scan = turing_scan(params, eq, n_max=80)
    m_inv = prey_only_invasion_threshold(params)

    y0_ode, initial_profile = default_threshold_initial_state(
        params,
        m_reference=m_low,
        perturbation_amplitude=perturbation,
        seed=seed,
    )

    m_values = [0.30, 0.50, 0.60]
    pde_results = {}
    for m_value in m_values:
        pde_results[m_value] = simulate_pde(
            params.with_updates(m=m_value),
            T=T,
            initial_state=initial_profile,
            epsilon=epsilon,
            rtol=1.0e-5,
            atol=1.0e-7,
            n_time=160,
        )

    ode_threshold = compute_mc_ode(
        params,
        m_low=m_low,
        m_high=m_high,
        epsilon=epsilon,
        T=T,
        y0=y0_ode,
        tol_m=1.0e-2,
        max_iter=10,
    )
    pde_threshold = compute_mc_pde(
        params,
        m_low=m_low,
        m_high=m_high,
        epsilon=epsilon,
        T=T,
        initial_state=initial_profile,
        tol_m=1.0e-2,
        max_iter=10,
        rtol=1.0e-5,
        atol=1.0e-7,
        n_time=120,
    )
    delta_mc = pde_threshold.threshold - ode_threshold.threshold

    m_grid = np.linspace(0.45, 0.75, 5)
    ratio_grid = np.array([1.0, 10.0, 100.0, 500.0, 1000.0])
    persistent = np.zeros((len(m_grid), len(ratio_grid)), dtype=bool)
    phase_var_p = np.zeros_like(persistent, dtype=float)
    for i, m_value in enumerate(m_grid):
        for j, ratio in enumerate(ratio_grid):
            phase_params = params.with_updates(m=float(m_value), delta_P=params.delta_U * float(ratio))
            result = simulate_pde(
                phase_params,
                T=T,
                initial_state=initial_profile,
                epsilon=epsilon,
                rtol=1.0e-5,
                atol=1.0e-7,
                n_time=80,
            )
            persistent[i, j] = result.diagnostics.persistent
            phase_var_p[i, j] = result.diagnostics.var_P

    final_profiles_path = FIG_DIR / "01_final_profiles.png"
    biomass_path = FIG_DIR / "01_predator_biomass_timeseries.png"
    phase_path = FIG_DIR / "01_phase_diagram_m_deltaP_ratio.png"
    plot_final_profiles(pde_results, final_profiles_path)
    plot_biomass_timeseries(pde_results, biomass_path)
    plot_phase_diagram(m_grid, ratio_grid, persistent, phase_path)

    max_profile_var = max(result.diagnostics.var_P for result in pde_results.values())
    turing_note = (
        "Turing-unstable modes were detected."
        if scan.turing_unstable
        else "No Turing-unstable mode was detected for this baseline parameter set."
    )
    pattern_note = (
        "The final predator variance was small in the first PDE runs, so this run should not be interpreted as evidence of pattern-mediated rescue."
        if max_profile_var < 1.0e-6
        else "The final predator variance was non-negligible, so spatial pattern strength should be inspected alongside the threshold comparison."
    )

    md = f"""# Nonlinear PDE Threshold Experiment 01

This run uses the defended/undefended prey reaction-diffusion model from the report on a 1D domain with no-flux boundary conditions. The analytic prey-only invasion threshold is reported only as a linear sanity check; it is not used as the nonlinear rescue threshold.

## Parameters

- `r_U={params.r_U}`, `r_D={params.r_D}` with `r_U>r_D`
- `a_U={params.a_U}`, `a_D={params.a_D}` with `a_U>a_D`
- `K={params.K}`, `e={params.e}`
- `mu_UD={params.mu_UD}`, `mu_DU={params.mu_DU}`
- `delta_U={params.delta_U}`, `delta_D={params.delta_D}`, `delta_P={params.delta_P}`
- `L={params.L}`, `n_x={params.n_x}`
- threshold bracket: `m_low={m_low}`, `m_high={m_high}`
- final time: `T={T}`
- persistence threshold: `epsilon={epsilon}`
- initial condition: coexistence equilibrium at `m={m_low}` plus random perturbation amplitude `{perturbation}`

## Linear checks

- prey-only linear invasion threshold: `m_inv_ODE = {m_inv:.6f}`
- positive coexistence equilibrium at `m={params.m}`:
  `q={eq.q:.6f}`, `N={eq.N:.6f}`, `U={eq.U:.6f}`, `D={eq.D:.6f}`, `P={eq.P:.6f}`
- ODE local stability at that equilibrium: `{ode_stable}`
- ODE eigenvalues: `{np.array2string(ode_eigs, precision=5)}`
- Turing scan: `{turing_note}`
- dominant scanned mode: `n={scan.dominant_mode}`, `k={scan.dominant_k:.6f}`, growth `{scan.dominant_growth:.6e}`

## Nonlinear threshold results

- `m_c^ODE = {ode_threshold.threshold:.6f}`
- `m_c^PDE = {pde_threshold.threshold:.6f}`
- `Delta m_c = {delta_mc:.6f}`
- threshold sign classification: `{classify_delta(delta_mc, tol=1.0e-2)}`

Important interpretation: {pattern_note} The sign of `Delta m_c` in this first run is therefore a threshold diagnostic for this parameter set and persistence criterion, not yet a demonstrated pattern-mediated rescue mechanism. A biological classification as pattern-promoted or pattern-inhibited rescue should require a Turing window or measurable spatial pattern strength.

## PDE diagnostics for sampled mortality values

| m | persistent | B_P(T) | B_U(T) | B_D(T) | O_PU | mean edible | var(P) | negative values |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
"""
    for m_value, result in pde_results.items():
        d = result.diagnostics
        md += (
            f"| {m_value:.3f} | {d.persistent} | {d.B_P:.6e} | {d.B_U:.6e} | "
            f"{d.B_D:.6e} | {d.O_PU:.6e} | {d.mean_edible:.6e} | "
            f"{d.var_P:.6e} | {d.negative_detected} |\n"
        )

    md += f"""
## Saved figures

- `{final_profiles_path.relative_to(ROOT)}`
- `{biomass_path.relative_to(ROOT)}`
- `{phase_path.relative_to(ROOT)}`

## Next scan

The first parameter set did not show a Turing window. The next scan should broaden reaction parameters and diffusion ratios, then require both conditions before claiming pattern-mediated rescue: a locally stable ODE coexistence state and at least one unstable spatial mode. A density-normalized persistence threshold should also be considered as a control for the fact that the PDE criterion uses total predator biomass over a finite domain.
"""

    RESULTS_MD.write_text(md, encoding="utf-8")

    print(md)


if __name__ == "__main__":
    main()
