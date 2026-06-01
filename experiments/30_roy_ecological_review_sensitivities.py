#!/usr/bin/env python
"""Small ecological-review sensitivity diagnostics.

This script addresses reviewer-facing caveats without changing model equations,
baseline parameters, existing classifications, or central claims. It runs:

1. A finite-time ODE sensitivity diagnostic over the defense-evolution rate nu.
2. A linear modal diagnostic over the phenomenological q-diffusion coefficient.

It does not run PDE simulations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (  # noqa: E402
    RoyEvoParams,
    classify_evo_trajectory,
    free_space_evo,
    simulate_ode_evo,
)


def load_step23_module():
    path = Path(__file__).resolve().with_name("23_roy_ode_compensation_conditions.py")
    spec = importlib.util.spec_from_file_location("step23_compensation_conditions_runtime", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP23 = load_step23_module()

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
NU_VALUES = (0.005, 0.01, 0.025, 0.05, 0.1)
NU_STRESSES = (0.08, 0.10, 0.1191, 0.14, 0.16)
REPRESENTATIVE_STRESS = 0.1191

D_N = 0.01
D_W = 0.01
D_Q_VALUES = (0.0, 0.001, 0.005, 0.01, 0.05)
D_Q_TARGET_STRESSES = (0.11765625, 0.1584375, 0.16486816, 0.175)
L_X = 20.0
L_Y = 20.0
MODE_MAX_TESTED = 64
LAMBDA_SCAN_POINTS = 600
POSITIVE_INSTABILITY_TOL = 1.0e-8

ODE_T = 3000.0
ODE_N_EVAL = 1201
ODE_RTOL = 1.0e-8
ODE_ATOL = 1.0e-10
TAIL_FRACTION = 0.25
EXTINCTION_EPSILON = 1.0e-4

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"

NU_SUMMARY_CSV = RESULTS_DIR / "roy_ode_nu_sensitivity_summary.csv"
NU_TIMESERIES_CSV = RESULTS_DIR / "roy_ode_nu_sensitivity_timeseries.csv"
D_Q_SCAN_CSV = RESULTS_DIR / "roy_pde_dq_modal_sensitivity_scan.csv"
D_Q_SUMMARY_CSV = RESULTS_DIR / "roy_pde_dq_modal_sensitivity_summary.csv"

FIG_NU_PATH = FIG_DIR / "fig69_ode_nu_sensitivity.png"
FIG_DQ_PATH = FIG_DIR / "fig70_pde_dq_modal_sensitivity.png"

NU_SUMMARY_FIELDS = [
    "nu",
    "stress",
    "classification",
    "persistent_predator",
    "threshold_crossed",
    "finite_time_status",
    "tail_mean_w",
    "min_w",
    "final_w",
    "final_q",
    "tail_mean_q",
    "q_change_from_initial",
    "solver_success",
    "physical",
    "notes",
]
NU_TIMESERIES_FIELDS = ["nu", "stress", "time", "n", "w", "q", "z"]
D_Q_SCAN_FIELDS = [
    "D_n",
    "D_w",
    "D_q",
    "stress",
    "lambda_value",
    "q_star",
    "max_real_growth",
    "eigenvalues_real",
    "eigenvalues_imag",
    "positive_lambda",
]
D_Q_SUMMARY_FIELDS = [
    "D_n",
    "D_w",
    "D_q",
    "stress",
    "lambda_min",
    "lambda_max",
    "lambda_points",
    "mode_range_i_j",
    "zero_mode_growth",
    "max_growth_for_positive_lambda",
    "lambda_at_max_positive_growth",
    "positive_growth_count",
    "positive_lambda_stable",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_float(value: float, digits: int = 8) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def eigen_text(values: np.ndarray) -> tuple[str, str]:
    return (
        ";".join(format_float(float(v), 10) for v in np.real(values)),
        ";".join(format_float(float(v), 10) for v in np.imag(values)),
    )


def branch_state_at(stress: float, params: RoyEvoParams = PARAMS) -> tuple[float, float, float]:
    n, w, q = STEP23.branch_state(params, stress)
    return float(n), float(w), float(q)


def initial_state() -> np.ndarray:
    return np.asarray(branch_state_at(0.0), dtype=float)


def classify_nu_run(metrics: dict[str, Any], solver_success: bool) -> str:
    if not solver_success:
        return "nonphysical"
    tail_mean_w = float(metrics.get("tail_mean_w", math.nan))
    final_or_min_w = min(float(metrics.get("min_w", math.nan)), tail_mean_w)
    if not bool(metrics.get("physical", False)) and np.isfinite(final_or_min_w) and final_or_min_w <= EXTINCTION_EPSILON:
        return "extinct_numerical_boundary_crossing"
    if not bool(metrics.get("physical", False)):
        return "nonphysical"
    if bool(metrics.get("persistent_predator", False)):
        return "persistent"
    tail_slope_w = float(metrics.get("tail_slope_w", math.nan))
    if np.isfinite(tail_mean_w) and tail_mean_w <= EXTINCTION_EPSILON:
        return "extinct"
    if np.isfinite(tail_mean_w) and np.isfinite(tail_slope_w) and tail_slope_w < 0.0:
        return "declining_transient"
    return "transient_or_unresolved"


def finite_time_status_from_threshold(classification: str, threshold_crossed: bool) -> str:
    """Ecological status that distinguishes tail recovery from threshold-safe persistence."""
    if classification.startswith("extinct") or classification == "nonphysical":
        return "extinct_or_boundary_crossing"
    if classification == "persistent":
        if threshold_crossed:
            return "recovered_after_threshold_crossing"
        return "persistent_above_threshold"
    return "transient_or_unresolved"


def run_nu_sensitivity() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y0 = initial_state()
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for nu in NU_VALUES:
        params = PARAMS.with_updates(nu=nu)
        for stress in NU_STRESSES:
            trajectory = simulate_ode_evo(
                params,
                y0,
                stress=stress,
                evolve=True,
                T=ODE_T,
                n_eval=ODE_N_EVAL,
                rtol=ODE_RTOL,
                atol=ODE_ATOL,
            )
            metrics = classify_evo_trajectory(
                trajectory.t,
                trajectory.y,
                epsilon=EXTINCTION_EPSILON,
                tail_fraction=TAIL_FRACTION,
                params=params,
            )
            n, w, q = trajectory.y
            z = free_space_evo(n, w, params)
            classification = classify_nu_run(metrics, trajectory.success)
            threshold_crossed = bool(float(metrics["min_w"]) <= EXTINCTION_EPSILON)
            finite_time_status = finite_time_status_from_threshold(classification, threshold_crossed)
            notes = "finite_time_nu_sensitivity_from_pre_stress_branch_state"
            if classification == "extinct_numerical_boundary_crossing":
                notes = "predator_lost_before_adaptation_solver_crossed_boundary"
            elif finite_time_status == "recovered_after_threshold_crossing":
                notes = "tail_persistent_but_crossed_extinction_threshold"
            summary_rows.append(
                {
                    "nu": nu,
                    "stress": stress,
                    "classification": classification,
                    "persistent_predator": bool(trajectory.success and metrics["physical"] and metrics["persistent_predator"]),
                    "threshold_crossed": threshold_crossed,
                    "finite_time_status": finite_time_status,
                    "tail_mean_w": metrics["tail_mean_w"],
                    "min_w": metrics["min_w"],
                    "final_w": float(w[-1]),
                    "final_q": float(q[-1]),
                    "tail_mean_q": metrics["tail_mean_q"],
                    "q_change_from_initial": metrics["q_change_from_initial"],
                    "solver_success": trajectory.success,
                    "physical": metrics["physical"],
                    "notes": notes,
                }
            )
            for idx, time in enumerate(trajectory.t):
                timeseries_rows.append(
                    {
                        "nu": nu,
                        "stress": stress,
                        "time": float(time),
                        "n": float(n[idx]),
                        "w": float(w[idx]),
                        "q": float(q[idx]),
                        "z": float(z[idx]),
                    }
                )

    write_csv(NU_SUMMARY_CSV, summary_rows, NU_SUMMARY_FIELDS)
    write_csv(NU_TIMESERIES_CSV, timeseries_rows, NU_TIMESERIES_FIELDS)
    make_nu_figure(summary_rows, timeseries_rows)
    return summary_rows, timeseries_rows


def lambda_max_for_mode_range() -> float:
    return 2.0 * (MODE_MAX_TESTED * math.pi / L_X) ** 2


def run_dq_modal_sensitivity() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lambda_values = np.linspace(0.0, lambda_max_for_mode_range(), LAMBDA_SCAN_POINTS)
    scan_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for d_q in D_Q_VALUES:
        diffusion = np.diag([D_N, D_W, d_q])
        for stress in D_Q_TARGET_STRESSES:
            n, w, q = branch_state_at(stress)
            jacobian = STEP23.ode_rhs_jacobian(n, w, q, stress, PARAMS)
            growth_values: list[float] = []
            for lambda_value in lambda_values:
                matrix = np.asarray(jacobian, dtype=float) - float(lambda_value) * diffusion
                eigenvalues = np.linalg.eigvals(matrix)
                max_growth = float(np.max(np.real(eigenvalues)))
                growth_values.append(max_growth)
                real_text, imag_text = eigen_text(eigenvalues)
                scan_rows.append(
                    {
                        "D_n": D_N,
                        "D_w": D_W,
                        "D_q": d_q,
                        "stress": stress,
                        "lambda_value": float(lambda_value),
                        "q_star": q,
                        "max_real_growth": max_growth,
                        "eigenvalues_real": real_text,
                        "eigenvalues_imag": imag_text,
                        "positive_lambda": bool(lambda_value > 0.0),
                    }
                )
            nonzero_growth = np.asarray(growth_values[1:], dtype=float)
            max_idx = int(np.argmax(nonzero_growth)) + 1
            positive_count = int(np.sum(nonzero_growth > POSITIVE_INSTABILITY_TOL))
            summary_rows.append(
                {
                    "D_n": D_N,
                    "D_w": D_W,
                    "D_q": d_q,
                    "stress": stress,
                    "lambda_min": float(lambda_values[0]),
                    "lambda_max": float(lambda_values[-1]),
                    "lambda_points": len(lambda_values),
                    "mode_range_i_j": f"0..{MODE_MAX_TESTED}",
                    "zero_mode_growth": growth_values[0],
                    "max_growth_for_positive_lambda": float(growth_values[max_idx]),
                    "lambda_at_max_positive_growth": float(lambda_values[max_idx]),
                    "positive_growth_count": positive_count,
                    "positive_lambda_stable": bool(positive_count == 0),
                    "notes": "linear_modal_dq_sensitivity_no_pde_simulation",
                }
            )

    write_csv(D_Q_SCAN_CSV, scan_rows, D_Q_SCAN_FIELDS)
    write_csv(D_Q_SUMMARY_CSV, summary_rows, D_Q_SUMMARY_FIELDS)
    make_dq_figure(summary_rows)
    return scan_rows, summary_rows


def finite_time_status_code(status: str) -> int:
    if status == "persistent_above_threshold":
        return 0
    if status == "recovered_after_threshold_crossing":
        return 1
    if status == "extinct_or_boundary_crossing":
        return 2
    if status == "transient_or_unresolved":
        return 3
    return 3


def make_nu_figure(summary_rows: list[dict[str, Any]], timeseries_rows: list[dict[str, Any]]) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    ax_w, ax_q, ax_class, ax_tail = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0.12, 0.90, len(NU_VALUES)))
    for nu, color in zip(NU_VALUES, colors):
        subset = [
            row
            for row in timeseries_rows
            if abs(float(row["stress"]) - REPRESENTATIVE_STRESS) < 1.0e-12 and abs(float(row["nu"]) - nu) < 1.0e-12
        ]
        ax_w.plot([float(row["time"]) for row in subset], [float(row["w"]) for row in subset], color=color, lw=1.8, label=rf"$\nu={nu:g}$")
        ax_q.plot([float(row["time"]) for row in subset], [float(row["q"]) for row in subset], color=color, lw=1.8)

    ax_w.set_title(r"A. Predator density at $s=0.1191$", loc="left", fontweight="bold")
    ax_w.axhline(EXTINCTION_EPSILON, color="#111111", lw=1.0, ls=":", label=r"$w=10^{-4}$")
    ax_w.set_xlabel("time")
    ax_w.set_ylabel(r"$w(t)$")
    ax_w.grid(alpha=0.22)
    ax_w.legend(frameon=False, fontsize=8, ncol=2)

    ax_q.set_title(r"B. Defense frequency at $s=0.1191$", loc="left", fontweight="bold")
    ax_q.set_xlabel("time")
    ax_q.set_ylabel(r"$q(t)$")
    ax_q.grid(alpha=0.22)

    stress_values = list(NU_STRESSES)
    class_matrix = np.full((len(NU_VALUES), len(stress_values)), np.nan)
    tail_matrix = np.full((len(NU_VALUES), len(stress_values)), np.nan)
    for row in summary_rows:
        i = NU_VALUES.index(float(row["nu"]))
        j = stress_values.index(float(row["stress"]))
        class_matrix[i, j] = finite_time_status_code(str(row["finite_time_status"]))
        tail_matrix[i, j] = max(float(row["tail_mean_w"]), 1.0e-10)

    class_cmap = ListedColormap(["#2f6fbb", "#8c6bb1", "#d9d9d9", "#fdae61"])
    norm = BoundaryNorm(np.arange(5) - 0.5, class_cmap.N)
    ax_class.imshow(class_matrix, origin="lower", aspect="auto", cmap=class_cmap, norm=norm)
    ax_class.set_title(r"C. Finite-time status over stress and $\nu$", loc="left", fontweight="bold")
    ax_class.set_xlabel("stress s")
    ax_class.set_ylabel(r"$\nu$")
    ax_class.set_xticks(range(len(stress_values)))
    ax_class.set_xticklabels([f"{value:g}" for value in stress_values])
    ax_class.set_yticks(range(len(NU_VALUES)))
    ax_class.set_yticklabels([f"{value:g}" for value in NU_VALUES])
    letters = {0: "P", 1: "R", 2: "E", 3: "T"}
    for i in range(len(NU_VALUES)):
        for j in range(len(stress_values)):
            code = int(class_matrix[i, j])
            ax_class.text(j, i, letters[code], ha="center", va="center", fontsize=12, fontweight="bold")
    handles = [
        Patch(facecolor="#2f6fbb", edgecolor="#222222", label="P persistent above threshold"),
        Patch(facecolor="#8c6bb1", edgecolor="#222222", label="R recovered after threshold crossing"),
        Patch(facecolor="#d9d9d9", edgecolor="#222222", label="E extinct/boundary crossing"),
        Patch(facecolor="#fdae61", edgecolor="#222222", label="T transient/unresolved"),
    ]
    ax_class.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.02, -0.16), ncol=1)

    image = ax_tail.imshow(tail_matrix, origin="lower", aspect="auto", cmap="magma", norm=LogNorm(vmin=1.0e-4, vmax=max(0.7, float(np.nanmax(tail_matrix)))))
    ax_tail.set_title(r"D. Tail mean predator density", loc="left", fontweight="bold")
    ax_tail.set_xlabel("stress s")
    ax_tail.set_ylabel(r"$\nu$")
    ax_tail.set_xticks(range(len(stress_values)))
    ax_tail.set_xticklabels([f"{value:g}" for value in stress_values])
    ax_tail.set_yticks(range(len(NU_VALUES)))
    ax_tail.set_yticklabels([f"{value:g}" for value in NU_VALUES])
    cbar = fig.colorbar(image, ax=ax_tail, fraction=0.046, pad=0.04)
    cbar.set_label(r"tail mean $w$")

    fig.suptitle("Finite-time evolutionary-rate sensitivity from the pre-stress branch state", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    FIG_NU_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_NU_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_dq_figure(summary_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax_growth, ax_count = axes
    colors = plt.cm.plasma(np.linspace(0.12, 0.90, len(D_Q_VALUES)))
    for d_q, color in zip(D_Q_VALUES, colors):
        subset = sorted([row for row in summary_rows if abs(float(row["D_q"]) - d_q) < 1.0e-12], key=lambda row: float(row["stress"]))
        stresses = [float(row["stress"]) for row in subset]
        growth = [float(row["max_growth_for_positive_lambda"]) for row in subset]
        counts = [int(float(row["positive_growth_count"])) for row in subset]
        ax_growth.plot(stresses, growth, marker="o", color=color, lw=1.8, label=rf"$D_q={d_q:g}$")
        ax_count.plot(stresses, counts, marker="s", color=color, lw=1.8)

    ax_growth.axhline(0.0, color="#111111", lw=1.0, ls="--")
    ax_growth.set_title(r"A. Max nonzero modal growth", loc="left", fontweight="bold")
    ax_growth.set_xlabel("stress s")
    ax_growth.set_ylabel(r"max Re eig for $\lambda>0$")
    ax_growth.grid(alpha=0.22)
    ax_growth.legend(frameon=False, fontsize=8)

    ax_count.set_title("B. Positive-growth sample count", loc="left", fontweight="bold")
    ax_count.set_xlabel("stress s")
    ax_count.set_ylabel("count")
    ax_count.grid(alpha=0.22)
    ax_count.set_ylim(-0.05, 1.05)
    if all(int(float(row["positive_growth_count"])) == 0 for row in summary_rows):
        ax_count.text(0.5, 0.82, "all sampled counts = 0", transform=ax_count.transAxes, ha="center", fontsize=10)

    fig.suptitle(r"Linear modal sensitivity to phenomenological $q$-diffusion", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    FIG_DQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DQ_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused",), default="focused")
    return parser.parse_args()


def main() -> None:
    parse_args()
    nu_summary, _nu_timeseries = run_nu_sensitivity()
    _dq_scan, dq_summary = run_dq_modal_sensitivity()
    persistent_count = sum(1 for row in nu_summary if str(row["classification"]) == "persistent")
    threshold_crossed_count = sum(1 for row in nu_summary if bool(row["threshold_crossed"]))
    dq_unstable = sum(1 for row in dq_summary if not bool(row["positive_lambda_stable"]))
    print(
        f"Wrote {NU_SUMMARY_CSV.relative_to(ROOT)} with {len(nu_summary)} rows "
        f"({persistent_count} tail-persistent classifications; {threshold_crossed_count} threshold crossings)."
    )
    print(f"Wrote {NU_TIMESERIES_CSV.relative_to(ROOT)}.")
    print(f"Wrote {D_Q_SUMMARY_CSV.relative_to(ROOT)} with {len(dq_summary)} rows ({dq_unstable} unstable rows).")
    print(f"Wrote {D_Q_SCAN_CSV.relative_to(ROOT)}.")
    print(f"Wrote {FIG_NU_PATH.relative_to(ROOT)} and {FIG_DQ_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
