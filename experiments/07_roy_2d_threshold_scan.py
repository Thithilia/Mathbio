"""Roy-style 2D predator mortality stress thresholds and phase diagrams."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from src.roy_style_2d import (
    Roy2DConfig,
    compute_stress_threshold_ode,
    compute_stress_threshold_pde,
    simulate_ode_stress,
    simulate_pde_2d,
)
from src.roy_style_model import RoyParams, continuous_turing_scan, require_positive_equilibrium


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_2d"
OUTPUT_CSV = RESULTS_DIR / "roy_2d_threshold_scan.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_06.md"


FIELDNAMES = [
    "row_type",
    "axis",
    "mu",
    "D_w_over_D_u",
    "stress",
    "epsilon",
    "T",
    "dt",
    "n_x",
    "n_y",
    "seed",
    "s_c_ode",
    "s_c_pde",
    "delta_s_c",
    "ode_persistent",
    "pde_persistent",
    "classification",
    "baseline_turing_unstable",
    "baseline_growth",
    "final_mean_w_ode",
    "final_mean_w_pde",
    "var_u",
    "var_v",
    "var_w",
    "dominant_wavelength",
    "pattern_measurable",
    "robustness_passed",
    "status",
]


CLASS_TO_INT = {
    "ODE persistent, PDE persistent": 0,
    "ODE extinct, PDE persistent": 1,
    "ODE persistent, PDE extinct": 2,
    "ODE extinct, PDE extinct": 3,
    "failed": 4,
}


def baseline_params(mu: float = 0.85, D_w_ratio: float = 100.0) -> RoyParams:
    D_u = 0.01
    return RoyParams(
        kappa=0.15,
        xi=0.55,
        gamma=3.73,
        rho=1.76,
        eta=0.005,
        beta1=0.5,
        delta=0.1,
        mu=mu,
        D_u=D_u,
        D_v=D_u,
        D_w=D_u * D_w_ratio,
    )


def classify(ode_persistent: bool, pde_persistent: bool) -> str:
    if ode_persistent and pde_persistent:
        return "ODE persistent, PDE persistent"
    if (not ode_persistent) and pde_persistent:
        return "ODE extinct, PDE persistent"
    if ode_persistent and (not pde_persistent):
        return "ODE persistent, PDE extinct"
    return "ODE extinct, PDE extinct"


def stable_config_for(params: RoyParams, config: Roy2DConfig) -> Roy2DConfig:
    dx = config.L_x / (config.n_x - 1)
    dy = config.L_y / (config.n_y - 1)
    max_diffusion = max(params.D_u, params.D_v, params.D_w)
    stable_dt = 0.22 / (max_diffusion * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if config.dt <= stable_dt:
        return config
    return replace(config, dt=0.9 * stable_dt)


def turing_status(params: RoyParams) -> tuple[bool, float]:
    eq = require_positive_equilibrium(params)
    scan = continuous_turing_scan(params, eq, k_min=1.0e-4, k_max=12.0, n_k=500, tol=1.0e-8)
    return scan.turing_unstable, scan.max_spatial_growth


def evaluate_point(
    axis: str,
    params: RoyParams,
    stress: float,
    config: Roy2DConfig,
    epsilon: float,
) -> dict[str, object]:
    try:
        run_config = stable_config_for(params, config)
        eq = require_positive_equilibrium(params)
        baseline_turing, baseline_growth = turing_status(params)
        ode = simulate_ode_stress(params, stress, T=run_config.T, epsilon=epsilon, y0=np.array([eq.u, eq.v, eq.w]))
        pde = simulate_pde_2d(params, run_config, stress=stress, equilibrium=eq)
        ode_persistent = bool(ode.success and ode.final_w > epsilon)
        pde_persistent = bool(pde.diagnostics.mean_w > epsilon)
        label = classify(ode_persistent, pde_persistent)
        pattern_measurable = bool(max(pde.diagnostics.var_u, pde.diagnostics.var_v, pde.diagnostics.var_w) > 1.0e-5)
        status = "ok"
        return {
            "row_type": "phase_point",
            "axis": axis,
            "mu": params.mu,
            "D_w_over_D_u": params.D_w / params.D_u,
            "stress": stress,
            "epsilon": epsilon,
            "T": run_config.T,
            "dt": run_config.dt,
            "n_x": run_config.n_x,
            "n_y": run_config.n_y,
            "seed": run_config.seed,
            "s_c_ode": "",
            "s_c_pde": "",
            "delta_s_c": "",
            "ode_persistent": ode_persistent,
            "pde_persistent": pde_persistent,
            "classification": label,
            "baseline_turing_unstable": baseline_turing,
            "baseline_growth": baseline_growth,
            "final_mean_w_ode": ode.final_w,
            "final_mean_w_pde": pde.diagnostics.mean_w,
            "var_u": pde.diagnostics.var_u,
            "var_v": pde.diagnostics.var_v,
            "var_w": pde.diagnostics.var_w,
            "dominant_wavelength": pde.diagnostics.dominant_wavelength,
            "pattern_measurable": pattern_measurable,
            "robustness_passed": "",
            "status": status,
        }
    except Exception as exc:
        return {
            "row_type": "phase_point",
            "axis": axis,
            "mu": params.mu,
            "D_w_over_D_u": params.D_w / params.D_u,
            "stress": stress,
            "epsilon": epsilon,
            "T": config.T,
            "dt": config.dt,
            "n_x": config.n_x,
            "n_y": config.n_y,
            "seed": config.seed,
            "s_c_ode": "",
            "s_c_pde": "",
            "delta_s_c": "",
            "ode_persistent": False,
            "pde_persistent": False,
            "classification": "failed",
            "baseline_turing_unstable": False,
            "baseline_growth": np.nan,
            "final_mean_w_ode": np.nan,
            "final_mean_w_pde": np.nan,
            "var_u": np.nan,
            "var_v": np.nan,
            "var_w": np.nan,
            "dominant_wavelength": np.nan,
            "pattern_measurable": False,
            "robustness_passed": "",
            "status": f"failed: {exc}",
        }


def threshold_row(params: RoyParams, config: Roy2DConfig, epsilon: float) -> dict[str, object]:
    ode = compute_stress_threshold_ode(params, s_low=0.0, s_high=0.9, epsilon=epsilon, T=config.T, tol_s=0.01, max_iter=9)
    pde = compute_stress_threshold_pde(params, config, s_low=0.0, s_high=0.9, epsilon=epsilon, tol_s=0.015, max_iter=8)
    baseline_turing, baseline_growth = turing_status(params)
    return {
        "row_type": "threshold",
        "axis": "baseline",
        "mu": params.mu,
        "D_w_over_D_u": params.D_w / params.D_u,
        "stress": "",
        "epsilon": epsilon,
        "T": config.T,
        "dt": config.dt,
        "n_x": config.n_x,
        "n_y": config.n_y,
        "seed": config.seed,
        "s_c_ode": ode.threshold,
        "s_c_pde": pde.threshold,
        "delta_s_c": pde.threshold - ode.threshold,
        "ode_persistent": "",
        "pde_persistent": "",
        "classification": "",
        "baseline_turing_unstable": baseline_turing,
        "baseline_growth": baseline_growth,
        "final_mean_w_ode": "",
        "final_mean_w_pde": "",
        "var_u": "",
        "var_v": "",
        "var_w": "",
        "dominant_wavelength": "",
        "pattern_measurable": "",
        "robustness_passed": "",
        "status": "ok",
    }


def robustness_rows(candidate_rows: list[dict[str, object]], epsilon: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = [
        row
        for row in candidate_rows
        if row["classification"] == "ODE extinct, PDE persistent"
        and bool(row["baseline_turing_unstable"])
        and bool(row["pattern_measurable"])
    ]
    for candidate in candidates[:3]:
        params = baseline_params(mu=float(candidate["mu"]), D_w_ratio=float(candidate["D_w_over_D_u"]))
        stress = float(candidate["stress"])
        checks = [
            Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=45.0, dt=0.025, seed=20260603),
            Roy2DConfig(n_x=44, n_y=44, L_x=20.0, L_y=20.0, T=45.0, dt=0.02, seed=20260604),
            Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=60.0, dt=0.025, seed=20260605),
        ]
        pass_all = True
        for config in checks:
            row = evaluate_point("robustness", params, stress, config, epsilon)
            pass_condition = (
                row["classification"] == "ODE extinct, PDE persistent"
                and bool(row["baseline_turing_unstable"])
                and bool(row["pattern_measurable"])
            )
            pass_all = pass_all and pass_condition
            row["row_type"] = "robustness"
            row["robustness_passed"] = pass_condition
            rows.append(row)
        for row in rows:
            if row["row_type"] == "robustness" and float(row["mu"]) == params.mu and float(row["stress"]) == stress:
                row["robustness_passed"] = pass_all
    return rows


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def plot_phase(rows: list[dict[str, object]], axis: str, path: Path) -> None:
    subset = [row for row in rows if row["row_type"] == "phase_point" and row["axis"] == axis]
    if not subset:
        return
    stress_values = sorted({float(row["stress"]) for row in subset})
    if axis == "stress_mu":
        x_values = sorted({float(row["mu"]) for row in subset})
        x_label = "mu"
        title = "Classification over stress and mu"
    else:
        x_values = sorted({float(row["D_w_over_D_u"]) for row in subset})
        x_label = "D_w / D_u"
        title = "Classification over stress and diffusion ratio"
    matrix = np.full((len(stress_values), len(x_values)), 4, dtype=float)
    for row in subset:
        i = stress_values.index(float(row["stress"]))
        key = float(row["mu"]) if axis == "stress_mu" else float(row["D_w_over_D_u"])
        j = x_values.index(key)
        matrix[i, j] = CLASS_TO_INT[str(row["classification"])]

    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#7f7f7f", "#d62728"]
    labels = list(CLASS_TO_INT.keys())
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=4, cmap=ListedColormap(colors))
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:g}" for value in x_values], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(stress_values)))
    ax.set_yticklabels([f"{value:g}" for value in stress_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel("stress s")
    ax.set_title(title)
    handles = [Patch(facecolor=colors[value], label=label) for label, value in CLASS_TO_INT.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], paths: dict[str, Path], path: Path) -> None:
    threshold = next(row for row in rows if row["row_type"] == "threshold")
    phase_rows = [row for row in rows if row["row_type"] == "phase_point"]
    rescue_like = [
        row
        for row in phase_rows
        if row["classification"] == "ODE extinct, PDE persistent"
        and bool(row["baseline_turing_unstable"])
        and bool(row["pattern_measurable"])
    ]
    robust_rows = [row for row in rows if row["row_type"] == "robustness"]
    robust_positive = bool(robust_rows) and all(bool(row["robustness_passed"]) for row in robust_rows)
    claim = bool(rescue_like and robust_positive)
    counts: dict[str, int] = {}
    for row in phase_rows:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1

    lines = [
        "# Nonlinear PDE Results 06",
        "",
        "This run adds predator mortality stress to the Roy-style model as `delta_eff = delta + s` and compares ODE/PDE persistence using mean predator density.",
        "",
        "## Baseline Thresholds",
        "",
        f"- `s_c^ODE = {float(threshold['s_c_ode']):.6g}`",
        f"- `s_c^PDE = {float(threshold['s_c_pde']):.6g}`",
        f"- `Delta s_c = s_c^PDE - s_c^ODE = {float(threshold['delta_s_c']):.6g}`",
        f"- baseline Turing unstable: `{threshold['baseline_turing_unstable']}`",
        "",
        "## Phase Diagrams",
        "",
        f"- phase points evaluated: `{len(phase_rows)}`",
        f"- classification counts: `{counts}`",
        f"- rescue-like points before robustness filtering: `{len(rescue_like)}`",
        f"- robustness rows evaluated: `{len(robust_rows)}`",
        "",
        "## Conservative Interpretation",
        "",
        f"- pattern-mediated rescue claimed: `{claim}`",
        "",
        "A claim requires baseline Turing instability, measurable nonlinear 2D patterning, ODE extinction with PDE persistence under the mean-density criterion, and robustness to final time, grid resolution, and perturbation seed. This script applies that rule conservatively.",
        "",
        "Outputs:",
        "",
        f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
        f"- `{paths['stress_mu'].relative_to(ROOT)}`",
        f"- `{paths['stress_ratio'].relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    epsilon = 1.0e-4
    config = Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=35.0, dt=0.025, record_every=20, seed=20260602)
    stress_values = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85]
    mu_values = [0.72, 0.80, 0.85, 0.89, 0.95]
    ratio_values = [40.0, 70.0, 100.0, 140.0, 200.0]

    rows: list[dict[str, object]] = [threshold_row(baseline_params(), config, epsilon)]
    phase_rows: list[dict[str, object]] = []

    for mu in mu_values:
        params = baseline_params(mu=mu)
        for stress in stress_values:
            phase_rows.append(evaluate_point("stress_mu", params, stress, config, epsilon))

    for ratio in ratio_values:
        params = baseline_params(mu=0.85, D_w_ratio=ratio)
        for stress in stress_values:
            phase_rows.append(evaluate_point("stress_ratio", params, stress, config, epsilon))

    rows.extend(phase_rows)
    rows.extend(robustness_rows(phase_rows, epsilon))

    stress_mu_path = FIG_DIR / "06_phase_stress_mu.png"
    stress_ratio_path = FIG_DIR / "06_phase_stress_diffusion_ratio.png"
    plot_phase(rows, "stress_mu", stress_mu_path)
    plot_phase(rows, "stress_ratio", stress_ratio_path)
    write_rows(rows, OUTPUT_CSV)
    write_summary(rows, {"stress_mu": stress_mu_path, "stress_ratio": stress_ratio_path}, SUMMARY_MD)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
