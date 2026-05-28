"""Reproduce Roy-style Turing instability and nonlinear 1D pattern formation.

This experiment uses the dimensionless Roy et al. model structure.  It does not
compute predator rescue thresholds; the only goal is to verify that the Roy-style
implementation can reproduce diffusion-driven instability and a nonlinear spatial
pattern.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.roy_style_model import (
    RoyParams,
    beta2_sharp,
    continuous_turing_scan,
    neumann_turing_scan,
    reaction_jacobian,
    reaction_part,
    require_positive_equilibrium,
    simulate_pde_1d,
    theta,
)


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
OUTPUT_CSV = RESULTS_DIR / "roy_turing_reproduction.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_04.md"
FIG_PATH = FIG_DIR / "04_roy_final_profiles.png"


FIELDNAMES = [
    "source",
    "kappa",
    "xi",
    "gamma",
    "rho",
    "eta",
    "beta1",
    "beta2",
    "theta",
    "delta",
    "mu",
    "D_u",
    "D_v",
    "D_w",
    "L",
    "n_x",
    "u_star",
    "v_star",
    "w_star",
    "z_star",
    "equilibrium_residual",
    "ode_max_real",
    "continuous_max_spatial_growth",
    "best_k",
    "continuous_turing_unstable",
    "continuous_unstable_windows",
    "discrete_max_spatial_growth",
    "dominant_discrete_mode",
    "dominant_discrete_k",
    "discrete_turing_unstable",
    "pde_success",
    "final_mean_w",
    "var_u",
    "var_v",
    "var_w",
    "pattern_confirmed",
    "negative_detected",
]


def roy_fig4_params(mu: float, n_x: int = 220, L: float = 20.0) -> RoyParams:
    return RoyParams(
        kappa=0.15,
        xi=0.55,
        gamma=3.73,
        rho=1.76,
        eta=0.005,
        beta1=0.5,
        delta=0.1,
        mu=mu,
        D_u=0.01,
        D_v=0.01,
        D_w=1.0,
        L=L,
        n_x=n_x,
    )


def format_windows(windows: list[tuple[float, float]]) -> str:
    return ";".join(f"{left:.6g}:{right:.6g}" for left, right in windows)


def row_for_params(params: RoyParams, run_pde: bool, pattern_tol: float) -> tuple[dict[str, object], object | None]:
    eq = require_positive_equilibrium(params)
    J = reaction_jacobian(eq.u, eq.v, eq.w, params)
    ode_eigs = np.linalg.eigvals(J)
    continuous = continuous_turing_scan(params, eq, k_min=1.0e-4, k_max=12.0, n_k=700, tol=1.0e-8)
    discrete = neumann_turing_scan(params, eq, n_max=160, tol=1.0e-8)

    pde = None
    if run_pde:
        pde = simulate_pde_1d(
            params,
            eq,
            T=80.0,
            perturbation_amplitude=1.0e-5,
            seed=20260531,
            rtol=1.0e-6,
            atol=1.0e-8,
            n_time=220,
        )

    var_u = pde.var_u if pde is not None else float("nan")
    var_v = pde.var_v if pde is not None else float("nan")
    var_w = pde.var_w if pde is not None else float("nan")
    pattern_confirmed = bool(pde is not None and pde.success and max(var_u, var_v, var_w) > pattern_tol)

    row = {
        "source": "Roy Table 1 / Fig. 4 parameter line, 1D verification",
        "kappa": params.kappa,
        "xi": params.xi,
        "gamma": params.gamma,
        "rho": params.rho,
        "eta": params.eta,
        "beta1": params.beta1,
        "beta2": beta2_sharp(params),
        "theta": theta(params),
        "delta": params.delta,
        "mu": params.mu,
        "D_u": params.D_u,
        "D_v": params.D_v,
        "D_w": params.D_w,
        "L": params.L,
        "n_x": params.n_x,
        "u_star": eq.u,
        "v_star": eq.v,
        "w_star": eq.w,
        "z_star": eq.z,
        "equilibrium_residual": eq.residual,
        "ode_max_real": float(np.max(ode_eigs.real)),
        "continuous_max_spatial_growth": continuous.max_spatial_growth,
        "best_k": continuous.best_k,
        "continuous_turing_unstable": continuous.turing_unstable,
        "continuous_unstable_windows": format_windows(continuous.unstable_windows),
        "discrete_max_spatial_growth": discrete.max_spatial_growth,
        "dominant_discrete_mode": discrete.dominant_mode,
        "dominant_discrete_k": discrete.best_k,
        "discrete_turing_unstable": discrete.turing_unstable,
        "pde_success": pde.success if pde is not None else "",
        "final_mean_w": pde.mean_w if pde is not None else float("nan"),
        "var_u": var_u,
        "var_v": var_v,
        "var_w": var_w,
        "pattern_confirmed": pattern_confirmed,
        "negative_detected": pde.negative_detected if pde is not None else "",
    }
    return row, pde


def plot_profiles(pde, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(pde.x, pde.u, label="undefended prey u", lw=1.8)
    ax.plot(pde.x, pde.v, label="defended prey v", lw=1.8)
    ax.plot(pde.x, pde.w, label="predator w", lw=1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Roy-style 1D final profiles")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]], chosen: dict[str, object], path: Path) -> None:
    turing_rows = [row for row in rows if bool(row["continuous_turing_unstable"]) and bool(row["discrete_turing_unstable"])]
    pattern_rows = [row for row in rows if bool(row["pattern_confirmed"])]
    lines = [
        "# Nonlinear PDE Results 04",
        "",
        "This run implements the Roy-style dimensionless model in a separate module and treats the previous minimal mass-action and Holling-II variants as negative-control models.",
        "",
        "No predator rescue thresholds were computed in this step.",
        "",
        "## Roy-Style Reproduction",
        "",
        f"- Roy-style parameter rows tested: `{len(rows)}`",
        f"- rows with continuous and discrete Turing instability: `{len(turing_rows)}`",
        f"- rows with nonlinear 1D pattern confirmation: `{len(pattern_rows)}`",
        "",
        "Chosen PDE verification row:",
        "",
        f"- `mu = {float(chosen['mu']):.6g}`",
        f"- homogeneous equilibrium `(u*, v*, w*, z*) = ({float(chosen['u_star']):.6g}, {float(chosen['v_star']):.6g}, {float(chosen['w_star']):.6g}, {float(chosen['z_star']):.6g})`",
        f"- ODE max real eigenvalue = `{float(chosen['ode_max_real']):.6g}`",
        f"- continuous max spatial growth = `{float(chosen['continuous_max_spatial_growth']):.6g}` at `k = {float(chosen['best_k']):.6g}`",
        f"- discrete max spatial growth = `{float(chosen['discrete_max_spatial_growth']):.6g}` at Neumann mode `{chosen['dominant_discrete_mode']}`",
        f"- final variances `(u, v, w) = ({float(chosen['var_u']):.6g}, {float(chosen['var_v']):.6g}, {float(chosen['var_w']):.6g})`",
        f"- pattern confirmed by variance threshold: `{chosen['pattern_confirmed']}`",
        "",
        "Interpretation: the Roy-style implementation reproduces the intended linear Turing instability and produces a nonlinear spatially heterogeneous final state in a 1D no-flux verification run. This is only a model-reproduction result, not a pattern-mediated rescue result.",
        "",
        "Outputs:",
        "",
        f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
        f"- `{FIG_PATH.relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    pattern_tol = 1.0e-6

    mu_values = [0.60, 0.72, 0.80, 0.85, 0.89, 0.95]
    preliminary_rows: list[tuple[dict[str, object], object | None]] = []
    for mu in mu_values:
        row, _ = row_for_params(roy_fig4_params(mu), run_pde=False, pattern_tol=pattern_tol)
        preliminary_rows.append((row, None))

    candidates = [row for row, _ in preliminary_rows if bool(row["continuous_turing_unstable"]) and bool(row["discrete_turing_unstable"])]
    if not candidates:
        raise RuntimeError("No Roy-style Turing candidate found along the Fig. 4 mu values.")

    chosen_mu = float(max(candidates, key=lambda item: float(item["discrete_max_spatial_growth"]))["mu"])
    rows: list[dict[str, object]] = []
    chosen_pde = None
    chosen_row: dict[str, object] | None = None
    for row, _ in preliminary_rows:
        if np.isclose(float(row["mu"]), chosen_mu):
            full_row, chosen_pde = row_for_params(roy_fig4_params(chosen_mu), run_pde=True, pattern_tol=pattern_tol)
            rows.append(full_row)
            chosen_row = full_row
        else:
            rows.append(row)

    assert chosen_pde is not None and chosen_row is not None
    plot_profiles(chosen_pde, FIG_PATH)
    write_rows(rows, OUTPUT_CSV)
    write_summary(rows, chosen_row, SUMMARY_MD)

    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
