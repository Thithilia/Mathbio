#!/usr/bin/env python
"""Counterfactual and linear-spatial evidence for the compensation manuscript.

This script adds evidence requested during external-review preparation:

1. A no-evolution/frozen-q ODE counterfactual for the linear Roy-style model.
2. A continuous lambda scan for the PDE modal matrix J - lambda D.
3. A compact Routh-Hurwitz margin table copied from existing coefficient outputs.

It does not run PDE simulations, change model equations, or introduce broad
parameter scans.
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
    b_of_q,
    classify_evo_trajectory,
    free_space_evo,
    r_of_q,
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
TARGET_STRESSES = (0.0, 0.069448242, 0.11765625, 0.1584375, 0.16486816, 0.175)
LAMBDA_TARGET_STRESSES = (0.11765625, 0.1584375, 0.16486816, 0.175)

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"

COUNTERFACTUAL_SWEEP_CSV = RESULTS_DIR / "roy_ode_no_evolution_counterfactual_stress_sweep.csv"
COUNTERFACTUAL_TIMESERIES_CSV = RESULTS_DIR / "roy_ode_no_evolution_counterfactual_timeseries.csv"
COUNTERFACTUAL_SUMMARY_CSV = RESULTS_DIR / "roy_ode_no_evolution_counterfactual_summary.csv"
RH_MARGINS_CSV = RESULTS_DIR / "roy_ode_compensation_routh_hurwitz_margins.csv"
LAMBDA_SCAN_CSV = RESULTS_DIR / "roy_pde_compensation_lambda_scan.csv"
LAMBDA_SUMMARY_CSV = RESULTS_DIR / "roy_pde_compensation_lambda_scan_summary.csv"

RH_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_routh_hurwitz_current.csv"
BRANCH_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_branch_current.csv"
NONLINEAR_SHAPE_SUMMARY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_shape_summary.csv"

FIG33_PATH = FIG_DIR / "fig33_compensation_branch_current.png"
FIG49_PATH = FIG_DIR / "fig49_nonhomogeneous_initial_conditions.png"
FIG50_PATH = FIG_DIR / "fig50_nonhomogeneous_final_fields.png"
FIG51_PATH = FIG_DIR / "fig51_nonhomogeneous_mean_timeseries.png"
FIG52_PATH = FIG_DIR / "fig52_nonhomogeneous_spatial_metrics.png"
FIG54_PATH = FIG_DIR / "fig54_long_horizon_mean_timeseries.png"
FIG55_PATH = FIG_DIR / "fig55_long_horizon_spatial_metrics.png"
FIG56_PATH = FIG_DIR / "fig56_long_horizon_final_fields.png"
FIG62_PATH = FIG_DIR / "fig62_nonlinear_pde_spatial_stability.png"
FIG63_PATH = FIG_DIR / "fig63_nonlinear_nonhomogeneous_pde_tests.png"
FIG64_PATH = FIG_DIR / "fig64_nonlinear_tradeoff_final_decision.png"
FIG65_PATH = FIG_DIR / "fig65_no_evolution_counterfactual.png"
FIG66_PATH = FIG_DIR / "fig66_pde_continuous_lambda_scan.png"
FIG67_PATH = FIG_DIR / "fig67_pde_spatial_stability_long_horizon_summary.png"
FIG68_PATH = FIG_DIR / "fig68_routh_hurwitz_margin_eigenvalue.png"

SPATIAL_MODES_CSV = RESULTS_DIR / "roy_pde_compensation_spatial_modes_current.csv"
LONG_HORIZON_DECISION_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_decision.csv"
NONHOMOG_MEAN_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_mean_timeseries.csv"
NONHOMOG_SPATIAL_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_spatial_metrics.csv"
LONG_HORIZON_MEAN_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_mean_timeseries.csv"
LONG_HORIZON_SPATIAL_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_spatial_metrics.csv"
NONLINEAR_PDE_STABILITY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_pde_spatial_stability.csv"
NONLINEAR_PDE_NONHOMOG_SUMMARY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_pde_nonhomogeneous_summary.csv"

ODE_T = 3000.0
ODE_N_EVAL = 1201
ODE_METHOD = "LSODA"
ODE_RTOL = 1.0e-8
ODE_ATOL = 1.0e-10
TAIL_FRACTION = 0.25
EXTINCTION_EPSILON = 1.0e-4
STRESS_GRID_POINTS = 81

D_N = 0.01
D_W = 0.01
D_Q = 0.005
L_X = 20.0
L_Y = 20.0
MODE_MAX_TESTED = 64
LAMBDA_SCAN_POINTS = 600
POSITIVE_INSTABILITY_TOL = 1.0e-8

SWEEP_FIELDS = [
    "stress",
    "treatment",
    "evolve_q",
    "initial_n",
    "initial_w",
    "initial_q",
    "frozen_q_value",
    "solver",
    "rtol",
    "atol",
    "T",
    "n_eval",
    "classification",
    "persistent_predator",
    "tail_mean_w",
    "tail_min_w",
    "tail_slope_w",
    "tail_slope_floor_w",
    "tail_mean_q",
    "q_change_from_initial",
    "final_n",
    "final_w",
    "final_q",
    "min_z",
    "physical",
    "solver_success",
    "notes",
]
TIMESERIES_FIELDS = [
    "representative_stress",
    "treatment",
    "evolve_q",
    "time",
    "n",
    "w",
    "q",
    "z",
]
SUMMARY_FIELDS = ["metric", "value", "interpretation"]
RH_MARGIN_FIELDS = [
    "stress",
    "A1",
    "A2",
    "A3",
    "A1A2_minus_A3",
    "minimum_rh_margin",
    "routh_hurwitz_stable",
    "max_real_eigenvalue",
]
LAMBDA_FIELDS = [
    "stress",
    "lambda_value",
    "n_star",
    "w_star",
    "q_star",
    "zero_mode",
    "max_real_growth",
    "eigenvalues_real",
    "eigenvalues_imag",
    "within_discrete_mode_range",
    "notes",
]
LAMBDA_SUMMARY_FIELDS = [
    "stress",
    "lambda_min",
    "lambda_max",
    "lambda_points",
    "mode_range_i_j",
    "zero_mode_growth",
    "max_growth_for_positive_lambda",
    "lambda_at_max_positive_growth",
    "positive_lambda_stable",
    "positive_growth_count",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_float(value: float, digits: int = 8) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def metric_dict(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in read_csv(path)}


def clean_label(value: str) -> str:
    return str(value).replace("_", " ")


def eigen_text(values: np.ndarray) -> tuple[str, str]:
    return (
        ";".join(format_float(float(v), 10) for v in np.real(values)),
        ";".join(format_float(float(v), 10) for v in np.imag(values)),
    )


def branch_interval(params: RoyEvoParams = PARAMS) -> dict[str, float]:
    interval = STEP23.stress_interval_for_q_in_unit_interval(params)
    return {
        "low": float(interval["interior_stress_interval_low"]),
        "high": float(interval["interior_stress_interval_high"]),
        "length": float(interval["interior_interval_length"]),
    }


def branch_state_at(stress: float, params: RoyEvoParams = PARAMS) -> tuple[float, float, float]:
    n, w, q = STEP23.branch_state(params, stress)
    return float(n), float(w), float(q)


def fixed_q_invasion_threshold(q0: float, params: RoyEvoParams = PARAMS) -> dict[str, float]:
    """Predator invasion threshold at the prey-only equilibrium for fixed q."""
    r_q = float(r_of_q(q0, params))
    b_q = float(b_of_q(q0, params))
    z0 = params.xi / r_q
    n0 = 1.0 / params.kappa - z0
    threshold = b_q * n0 * z0 - params.m
    return {
        "q0": float(q0),
        "z0": float(z0),
        "n0": float(n0),
        "b_q0": float(b_q),
        "r_q0": float(r_q),
        "s_fixed": float(threshold),
    }


def classify_treatment(metrics: dict[str, Any]) -> str:
    if not bool(metrics.get("physical", False)):
        return "nonphysical"
    if bool(metrics.get("persistent_predator", False)):
        return "persistent"
    tail_mean_w = float(metrics.get("tail_mean_w", math.nan))
    tail_min_w = float(metrics.get("tail_min_w", math.nan))
    slope_w = float(metrics.get("tail_slope_w", math.nan))
    if np.isfinite(tail_mean_w) and tail_mean_w <= EXTINCTION_EPSILON:
        return "extinct"
    if np.isfinite(tail_mean_w) and np.isfinite(tail_min_w) and np.isfinite(slope_w):
        if tail_mean_w > EXTINCTION_EPSILON and slope_w < 0.0:
            return "declining_transient"
    return "transient_or_unresolved"


def simulate_counterfactual(stress: float, evolve_q: bool, initial_state: np.ndarray) -> dict[str, Any]:
    treatment = "evolving_q" if evolve_q else "frozen_q"
    trajectory = simulate_ode_evo(
        PARAMS,
        initial_state,
        stress=stress,
        evolve=evolve_q,
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
        params=PARAMS,
    )
    classification = classify_treatment(metrics)
    final = trajectory.y[:, -1]
    return {
        "stress": float(stress),
        "treatment": treatment,
        "evolve_q": bool(evolve_q),
        "initial_n": float(initial_state[0]),
        "initial_w": float(initial_state[1]),
        "initial_q": float(initial_state[2]),
        "frozen_q_value": float(initial_state[2]),
        "solver": ODE_METHOD,
        "rtol": ODE_RTOL,
        "atol": ODE_ATOL,
        "T": ODE_T,
        "n_eval": ODE_N_EVAL,
        "classification": classification,
        "persistent_predator": bool(trajectory.success and metrics["physical"] and metrics["persistent_predator"]),
        "tail_mean_w": metrics["tail_mean_w"],
        "tail_min_w": metrics["tail_min_w"],
        "tail_slope_w": metrics["tail_slope_w"],
        "tail_slope_floor_w": metrics["tail_slope_floor_w"],
        "tail_mean_q": metrics["tail_mean_q"],
        "q_change_from_initial": metrics["q_change_from_initial"],
        "final_n": float(final[0]),
        "final_w": float(final[1]),
        "final_q": float(final[2]),
        "min_z": metrics["min_z"],
        "physical": metrics["physical"],
        "solver_success": trajectory.success,
        "notes": "q evolves under selection" if evolve_q else "q frozen at q_star_s0",
        "_trajectory": trajectory,
    }


def stress_grid() -> np.ndarray:
    interval = branch_interval()
    high = max(0.0, interval["high"] - 1.0e-5)
    return np.linspace(0.0, high, STRESS_GRID_POINTS)


def find_rescue_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stress: dict[float, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_stress.setdefault(float(row["stress"]), {})[str(row["treatment"])] = row
    rescue_stresses: list[float] = []
    for stress, pair in sorted(by_stress.items()):
        frozen = pair.get("frozen_q")
        evolving = pair.get("evolving_q")
        if not frozen or not evolving:
            continue
        frozen_extinct = str(frozen["classification"]) == "extinct"
        evolving_persistent = bool(evolving["persistent_predator"])
        if frozen_extinct and evolving_persistent:
            rescue_stresses.append(stress)
    if not rescue_stresses:
        return {
            "exists": False,
            "low": math.nan,
            "high": math.nan,
            "representative": math.nan,
            "count": 0,
        }
    low = min(rescue_stresses)
    high = max(rescue_stresses)
    representative = rescue_stresses[len(rescue_stresses) // 2]
    return {
        "exists": True,
        "low": low,
        "high": high,
        "representative": representative,
        "count": len(rescue_stresses),
    }


def run_counterfactual() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    initial_state = np.asarray(branch_state_at(0.0), dtype=float)
    fixed_threshold = fixed_q_invasion_threshold(float(initial_state[2]))
    rows: list[dict[str, Any]] = []
    for stress in stress_grid():
        for evolve_q in (False, True):
            rows.append(simulate_counterfactual(float(stress), evolve_q, initial_state))
    rescue = find_rescue_window(rows)

    representative = float(rescue["representative"])
    if not np.isfinite(representative):
        # Still produce a diagnostic timeseries if no rescue window is found.
        representative = 0.5 * branch_interval()["high"]

    timeseries_rows: list[dict[str, Any]] = []
    for evolve_q in (False, True):
        record = simulate_counterfactual(representative, evolve_q, initial_state)
        trajectory = record.pop("_trajectory")
        n, w, q = trajectory.y
        z = free_space_evo(n, w, PARAMS)
        treatment = "evolving_q" if evolve_q else "frozen_q"
        for idx, time in enumerate(trajectory.t):
            timeseries_rows.append(
                {
                    "representative_stress": representative,
                    "treatment": treatment,
                    "evolve_q": bool(evolve_q),
                    "time": float(time),
                    "n": float(n[idx]),
                    "w": float(w[idx]),
                    "q": float(q[idx]),
                    "z": float(z[idx]),
                }
            )

    output_rows = [{k: v for k, v in row.items() if k != "_trajectory"} for row in rows]
    write_csv(COUNTERFACTUAL_SWEEP_CSV, output_rows, SWEEP_FIELDS)
    write_csv(COUNTERFACTUAL_TIMESERIES_CSV, timeseries_rows, TIMESERIES_FIELDS)

    interval = branch_interval()
    summary_rows = [
        {
            "metric": "initial_state",
            "value": ";".join(format_float(v, 10) for v in initial_state),
            "interpretation": "pre-stress branch state n*,w*,q*(s=0); q is frozen at this value in the no-evolution treatment",
        },
        {
            "metric": "stress_grid_points",
            "value": STRESS_GRID_POINTS,
            "interpretation": "stress values swept from s=0 to just below the branch upper endpoint",
        },
        {
            "metric": "branch_interval_low",
            "value": interval["low"],
            "interpretation": "lower endpoint for 0<q*(s)<1",
        },
        {
            "metric": "branch_interval_high",
            "value": interval["high"],
            "interpretation": "upper endpoint for 0<q*(s)<1",
        },
        {
            "metric": "fixed_q_invasion_threshold",
            "value": fixed_threshold["s_fixed"],
            "interpretation": "analytic prey-only predator invasion threshold for q frozen at q*(s=0)",
        },
        {
            "metric": "fixed_q_threshold_state",
            "value": ";".join(format_float(fixed_threshold[key], 10) for key in ("q0", "n0", "z0", "r_q0", "b_q0")),
            "interpretation": "q0,n0,z0,r(q0),b(q0) used in s_fixed=b(q0)n0z0-m",
        },
        {
            "metric": "rescue_window_exists",
            "value": rescue["exists"],
            "interpretation": "True when frozen-q is extinct but evolving-q persists on at least one stress-grid point",
        },
        {
            "metric": "rescue_window_low_grid",
            "value": rescue["low"],
            "interpretation": "lowest grid stress where frozen-q is extinct and evolving-q persists",
        },
        {
            "metric": "rescue_window_high_grid",
            "value": rescue["high"],
            "interpretation": "highest grid stress where frozen-q is extinct and evolving-q persists",
        },
        {
            "metric": "representative_rescue_stress",
            "value": representative,
            "interpretation": "stress used for the representative trajectory panels",
        },
        {
            "metric": "ode_solver",
            "value": ODE_METHOD,
            "interpretation": f"solve_ivp method with rtol={ODE_RTOL:g}, atol={ODE_ATOL:g}",
        },
        {
            "metric": "integration_horizon",
            "value": ODE_T,
            "interpretation": "ODE horizon used for stress sweep and representative trajectories",
        },
        {
            "metric": "extinction_threshold",
            "value": EXTINCTION_EPSILON,
            "interpretation": "predator tail-mean threshold used by existing classifier",
        },
        {
            "metric": "tail_fraction",
            "value": TAIL_FRACTION,
            "interpretation": "fraction of terminal trajectory used for tail means and slope diagnostics",
        },
    ]
    write_csv(COUNTERFACTUAL_SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    make_counterfactual_figure(output_rows, timeseries_rows, rescue, representative, fixed_threshold["s_fixed"])
    return output_rows, timeseries_rows, summary_rows


def make_counterfactual_figure(
    sweep_rows: list[dict[str, Any]],
    timeseries_rows: list[dict[str, Any]],
    rescue: dict[str, Any],
    representative: float,
    fixed_q_threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    ax_w, ax_q, ax_final, ax_window = axes.flatten()
    colors = {"frozen_q": "#c23b3b", "evolving_q": "#2f6fbb"}
    labels = {"frozen_q": "frozen q", "evolving_q": "evolving q"}

    for treatment in ("frozen_q", "evolving_q"):
        rows = [row for row in timeseries_rows if row["treatment"] == treatment]
        ax_w.plot([row["time"] for row in rows], [row["w"] for row in rows], color=colors[treatment], label=labels[treatment], lw=2.0)
    ax_w.set_title("A. Predator density at rescue stress")
    ax_w.set_xlabel("time")
    ax_w.set_ylabel("w(t)")
    ax_w.legend(frameon=False)

    evo_rows = [row for row in timeseries_rows if row["treatment"] == "evolving_q"]
    frozen_rows = [row for row in timeseries_rows if row["treatment"] == "frozen_q"]
    ax_q.plot([row["time"] for row in evo_rows], [row["q"] for row in evo_rows], color=colors["evolving_q"], lw=2.0)
    if frozen_rows:
        ax_q.axhline(float(frozen_rows[0]["q"]), color=colors["frozen_q"], lw=1.5, ls="--", label="frozen q")
    q_rep = branch_state_at(representative)[2]
    ax_q.axhline(q_rep, color="#222222", lw=1.3, ls=":", label=rf"$q^*({representative:.4f})$")
    ax_q.annotate(
        rf"$q^*={q_rep:.3f}$",
        xy=(0.70 * ODE_T, q_rep),
        xytext=(0.70 * ODE_T, q_rep + 0.08),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#222222"},
        fontsize=8,
        ha="left",
    )
    ax_q.legend(frameon=False, fontsize=8)
    ax_q.set_title("B. Defense response in evolving-q run")
    ax_q.set_xlabel("time")
    ax_q.set_ylabel("q(t)")

    by_treatment = {t: sorted([row for row in sweep_rows if row["treatment"] == t], key=lambda row: float(row["stress"])) for t in colors}
    plot_floor = 1.0e-8
    for treatment, rows in by_treatment.items():
        tail_means = [max(float(row["tail_mean_w"]), plot_floor) for row in rows]
        ax_final.plot([row["stress"] for row in rows], tail_means, marker="o", ms=2.8, lw=1.6, color=colors[treatment], label=labels[treatment])
    ax_final.axhline(EXTINCTION_EPSILON, color="#444444", lw=1.0, ls=":", label="extinction threshold")
    ax_final.axvline(fixed_q_threshold, color="#6b4c9a", lw=1.4, ls="-.", label=r"$s_{\mathrm{fixed}}$")
    ax_final.annotate(
        rf"$s_{{\mathrm{{fixed}}}}={fixed_q_threshold:.4f}$",
        xy=(fixed_q_threshold, EXTINCTION_EPSILON * 1.4),
        xytext=(fixed_q_threshold + 0.018, EXTINCTION_EPSILON * 20),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#6b4c9a"},
        fontsize=8,
        color="#4f3677",
    )
    ax_final.set_title("C. Tail predator density across stress")
    ax_final.set_xlabel("stress s")
    ax_final.set_ylabel("tail mean w")
    ax_final.set_yscale("log")
    ax_final.legend(frameon=False, fontsize=8)

    for treatment, rows in by_treatment.items():
        persistent = [1.0 if row["persistent_predator"] else 0.0 for row in rows]
        ax_window.step([row["stress"] for row in rows], persistent, where="mid", color=colors[treatment], label=labels[treatment], lw=2.0)
    if rescue["exists"]:
        ax_window.axvspan(float(rescue["low"]), float(rescue["high"]), color="#f0c419", alpha=0.35, label="verified rescue window")
        ax_window.axvline(representative, color="#111111", lw=1.0, ls="--", label="representative stress")
        ax_window.text(
            0.5 * (float(rescue["low"]) + float(rescue["high"])),
            0.55,
            "verified\nrescue window",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d8b000", "alpha": 0.9},
        )
    ax_window.axvline(fixed_q_threshold, color="#6b4c9a", lw=1.4, ls="-.", label=r"$s_{\mathrm{fixed}}$")
    ax_window.set_title("D. Frozen-q extinct, evolving-q persistent")
    ax_window.set_xlabel("stress s")
    ax_window.set_yticks([0, 1])
    ax_window.set_yticklabels(["not persistent", "persistent"])
    ax_window.set_ylim(-0.12, 1.12)
    ax_window.legend(frameon=False, fontsize=8, loc="center right")

    fig.suptitle("No-evolution counterfactual for indirect evolutionary rescue", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG65_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG65_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_compensation_branch_figure() -> None:
    rows = read_csv(BRANCH_CURRENT_CSV)
    interval = branch_interval()
    stress_values = np.linspace(interval["low"], interval["high"], 300)
    branch_states = np.asarray([branch_state_at(float(stress)) for stress in stress_values])
    n_values = branch_states[:, 0]
    w_values = branch_states[:, 1]
    q_values = branch_states[:, 2]
    n_mean = float(np.mean(n_values))
    w_mean = float(np.mean(w_values))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    ax_q, ax_levels = axes
    ax_q.axvspan(interval["low"], interval["high"], color="#e8eef7", alpha=0.7, label="interior interval")
    ax_q.plot(stress_values, q_values, color="#1f5f9f", lw=2.2, label=r"analytic $q^*(s)$")
    numeric_stress = np.asarray([float(row["stress"]) for row in rows])
    numeric_q = np.asarray([float(row["q_star_numerical"]) for row in rows])
    ax_q.scatter(numeric_stress, numeric_q, color="#d05a2a", edgecolor="white", linewidth=0.7, s=48, zorder=4, label="numerical stable equilibria")
    ax_q.scatter([interval["low"], interval["high"]], [1.0, 0.0], marker="D", color="#444444", s=42, zorder=5, label="branch endpoints")
    ax_q.axvline(0.0, color="#333333", lw=0.9, ls=":")
    ax_q.set_title(r"A. Defense frequency along branch")
    ax_q.set_xlabel("stress s")
    ax_q.set_ylabel(r"$q^*(s)$")
    ax_q.set_ylim(-0.05, 1.05)
    ax_q.legend(frameon=False, fontsize=8, loc="upper right")

    ax_levels.axvspan(interval["low"], interval["high"], color="#e8eef7", alpha=0.7)
    ax_levels.plot(stress_values, n_values / n_mean, color="#2a7f62", lw=2.2, label=r"$n^*/\bar n^*$")
    ax_levels.plot(stress_values, w_values / w_mean, color="#8a4f9e", lw=2.2, ls="--", label=r"$w^*/\bar w^*$")
    ax_levels.scatter(numeric_stress, np.asarray([float(row["n_star_numerical"]) for row in rows]) / n_mean, color="#2a7f62", edgecolor="white", linewidth=0.6, s=34, zorder=4)
    ax_levels.scatter(numeric_stress, np.asarray([float(row["w_star_numerical"]) for row in rows]) / w_mean, color="#8a4f9e", edgecolor="white", linewidth=0.6, s=34, zorder=4)
    ax_levels.axhline(1.0, color="#333333", lw=0.8, ls=":")
    ax_levels.axvline(0.0, color="#333333", lw=0.9, ls=":")
    ax_levels.set_title(r"B. Positive densities remain fixed")
    ax_levels.set_xlabel("stress s")
    ax_levels.set_ylabel("normalized equilibrium value")
    ax_levels.set_ylim(0.96, 1.04)
    ax_levels.ticklabel_format(useOffset=False, axis="y")
    ax_levels.text(
        0.04,
        0.07,
        rf"$n^*\approx {n_mean:.4f}$" + "\n" + rf"$w^*\approx {w_mean:.4f}$",
        transform=ax_levels.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    ax_levels.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Linear compensation branch: defense changes while positive densities stay fixed", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG33_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_nonlinear_tradeoff_figure() -> None:
    rows = read_csv(NONLINEAR_SHAPE_SUMMARY_CSV)
    gamma_values = [0.5, 1.0, 2.0]
    class_order = [
        "robust_compensation_shape",
        "partial_compensation_shape",
        "no_compensation_shape",
        "unresolved_shape",
    ]
    class_to_code = {name: idx for idx, name in enumerate(class_order)}
    colors = ["#222222", "#707070", "#d0d0d0", "#ffffff"]
    labels = {
        "robust_compensation_shape": "R robust compensation",
        "partial_compensation_shape": "P partial compensation",
        "no_compensation_shape": "N no compensation",
        "unresolved_shape": "U unresolved",
    }
    counts = {name: 0 for name in class_order}
    lookup: dict[tuple[float, float, float], str] = {}
    for row in rows:
        key = (float(row["gamma_r"]), float(row["gamma_a"]), float(row["gamma_b"]))
        shape_class = row["shape_class"]
        lookup[key] = shape_class
        counts[shape_class] = counts.get(shape_class, 0) + 1

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(class_order) + 1) - 0.5, cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), sharey=True)
    for ax, gamma_b in zip(axes, gamma_values):
        matrix = np.full((len(gamma_values), len(gamma_values)), np.nan)
        for y_idx, gamma_a in enumerate(gamma_values):
            for x_idx, gamma_r in enumerate(gamma_values):
                matrix[y_idx, x_idx] = class_to_code[lookup[(gamma_r, gamma_a, gamma_b)]]
        ax.imshow(matrix, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        ax.set_title(rf"$\gamma_b={gamma_b:g}$")
        ax.set_xticks(range(len(gamma_values)))
        ax.set_xticklabels([f"{value:g}" for value in gamma_values])
        ax.set_yticks(range(len(gamma_values)))
        ax.set_yticklabels([f"{value:g}" for value in gamma_values])
        ax.set_xlabel(r"$\gamma_r$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\gamma_a$")
        for y_idx in range(len(gamma_values)):
            for x_idx in range(len(gamma_values)):
                code = int(matrix[y_idx, x_idx])
                letter = labels[class_order[code]][0]
                text_color = "white" if code in (0, 1) else "black"
                ax.text(x_idx, y_idx, letter, ha="center", va="center", color=text_color, fontsize=17, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(gamma_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(gamma_values), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, len(gamma_values) - 0.5)
        ax.set_ylim(-0.5, len(gamma_values) - 0.5)

    handles = [Patch(facecolor=colors[idx], edgecolor="#222222", label=f"{labels[name]}: {counts.get(name, 0)} shapes") for idx, name in enumerate(class_order)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Controlled nonlinear trade-off shape grid", fontsize=13)
    fig.text(0.5, 0.06, "Grayscale-safe cell letters identify each class: R, P, N, U", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.14, 1, 0.92])
    fig.savefig(FIG64_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def reaction_jacobian(params: RoyEvoParams, stress: float) -> np.ndarray:
    n, w, q = branch_state_at(stress, params)
    return np.asarray(STEP23.ode_rhs_jacobian(n, w, q, stress, params), dtype=float)


def lambda_max_for_mode_range() -> float:
    return 2.0 * (MODE_MAX_TESTED * math.pi / L_X) ** 2


def run_lambda_scan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diffusion = np.diag([D_N, D_W, D_Q])
    lambda_values = np.linspace(0.0, lambda_max_for_mode_range(), LAMBDA_SCAN_POINTS)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for stress in LAMBDA_TARGET_STRESSES:
        n, w, q = branch_state_at(stress)
        jacobian = reaction_jacobian(PARAMS, stress)
        growth_values: list[float] = []
        for lambda_value in lambda_values:
            matrix = jacobian - float(lambda_value) * diffusion
            eigenvalues = np.linalg.eigvals(matrix)
            max_growth = float(np.max(np.real(eigenvalues)))
            real_text, imag_text = eigen_text(eigenvalues)
            growth_values.append(max_growth)
            rows.append(
                {
                    "stress": stress,
                    "lambda_value": float(lambda_value),
                    "n_star": n,
                    "w_star": w,
                    "q_star": q,
                    "zero_mode": bool(abs(float(lambda_value)) < 1.0e-14),
                    "max_real_growth": max_growth,
                    "eigenvalues_real": real_text,
                    "eigenvalues_imag": imag_text,
                    "within_discrete_mode_range": bool(lambda_value <= lambda_max_for_mode_range()),
                    "notes": f"continuous lambda scan over mode-equivalent range i,j=0..{MODE_MAX_TESTED}",
                }
            )
        positive_indices = [idx for idx, growth in enumerate(growth_values) if idx > 0 and growth > POSITIVE_INSTABILITY_TOL]
        nonzero_growth = np.asarray(growth_values[1:], dtype=float)
        max_idx = int(np.argmax(nonzero_growth)) + 1
        summary_rows.append(
            {
                "stress": stress,
                "lambda_min": float(lambda_values[0]),
                "lambda_max": float(lambda_values[-1]),
                "lambda_points": len(lambda_values),
                "mode_range_i_j": f"0..{MODE_MAX_TESTED}",
                "zero_mode_growth": growth_values[0],
                "max_growth_for_positive_lambda": float(growth_values[max_idx]),
                "lambda_at_max_positive_growth": float(lambda_values[max_idx]),
                "positive_lambda_stable": bool(len(positive_indices) == 0),
                "positive_growth_count": len(positive_indices),
                "notes": "continuous scan supports tested lambda range" if len(positive_indices) == 0 else "positive growth detected in continuous scan",
            }
        )
    write_csv(LAMBDA_SCAN_CSV, rows, LAMBDA_FIELDS)
    write_csv(LAMBDA_SUMMARY_CSV, summary_rows, LAMBDA_SUMMARY_FIELDS)
    make_lambda_figure(rows, summary_rows)
    return rows, summary_rows


def make_lambda_figure(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for stress in LAMBDA_TARGET_STRESSES:
        subset = [row for row in rows if abs(float(row["stress"]) - stress) < 1.0e-12]
        ax.plot([row["lambda_value"] for row in subset], [row["max_real_growth"] for row in subset], lw=1.8, label=f"s={stress:g}")
    ax.axhline(0.0, color="#111111", lw=1.0, ls="--")
    ax.set_xlabel(r"continuous modal value $\lambda$")
    ax.set_ylabel(r"max Re eig$(J_F(U^*)-\lambda D)$")
    ax.set_title("Continuous lambda scan over tested Neumann-mode range")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.02,
        0.04,
        f"Range matches i,j=0..{MODE_MAX_TESTED}; D=({D_N:g},{D_W:g},{D_Q:g})",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    FIG66_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG66_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_rh_margin_eigenvalue_figure(rows: list[dict[str, Any]]) -> None:
    stress = np.asarray([float(row["stress"]) for row in rows])
    min_margin = np.asarray([float(row["minimum_rh_margin"]) for row in rows])
    max_real = np.asarray([float(row["max_real_eigenvalue"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True)
    ax_margin, ax_eig = axes
    ax_margin.plot(stress, min_margin, marker="o", color="#2f6fbb", lw=2.2)
    ax_margin.axhline(0.0, color="#222222", lw=1.0, ls="--")
    ax_margin.set_title("A. Minimum Routh-Hurwitz margin")
    ax_margin.set_xlabel("stress s")
    ax_margin.set_ylabel(r"$\min(A_1,A_2,A_3,A_1A_2-A_3)$")
    ax_margin.grid(alpha=0.25)

    ax_eig.plot(stress, max_real, marker="s", color="#9b4a97", lw=2.2)
    ax_eig.axhline(0.0, color="#222222", lw=1.0, ls="--")
    ax_eig.set_title("B. Maximum eigenvalue real part")
    ax_eig.set_xlabel("stress s")
    ax_eig.set_ylabel(r"$\max \Re(\lambda_J)$")
    ax_eig.grid(alpha=0.25)

    fig.suptitle("Local stability margins along the compensation branch", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIG68_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_main_pde_evidence_figure() -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    mode_rows = read_csv(SPATIAL_MODES_CSV)
    lambda_rows = read_csv(LAMBDA_SCAN_CSV)
    decision = metric_dict(LONG_HORIZON_DECISION_CSV)
    stresses = list(LAMBDA_TARGET_STRESSES)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(stresses)))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1.25, 1.25, 0.95]})
    ax_modes, ax_lambda, ax_decision = axes

    for stress, color in zip(stresses, colors):
        subset = [row for row in mode_rows if abs(float(row["stress"]) - stress) < 1.0e-12 and not as_bool(row["zero_mode"])]
        ax_modes.scatter(
            [float(row["lambda_mn"]) for row in subset],
            [float(row["max_real_growth"]) for row in subset],
            s=5,
            alpha=0.32,
            color=color,
            edgecolors="none",
            label=f"s={stress:g}",
        )
    ax_modes.axhline(0.0, color="#111111", lw=1.0, ls="--")
    ax_modes.set_title("A. Discrete Neumann modes", loc="left", fontweight="bold")
    ax_modes.set_xlabel(r"mode eigenvalue $\lambda_{ij}$")
    ax_modes.set_ylabel("max real growth")
    ax_modes.grid(alpha=0.22)
    ax_modes.legend(frameon=False, fontsize=7, loc="upper right")

    for stress, color in zip(stresses, colors):
        subset = [row for row in lambda_rows if abs(float(row["stress"]) - stress) < 1.0e-12]
        ax_lambda.plot(
            [float(row["lambda_value"]) for row in subset],
            [float(row["max_real_growth"]) for row in subset],
            lw=2.0,
            color=color,
            label=f"s={stress:g}",
        )
    ax_lambda.axhline(0.0, color="#111111", lw=1.0, ls="--")
    ax_lambda.set_title(r"B. Continuous $\lambda$ scan", loc="left", fontweight="bold")
    ax_lambda.set_xlabel(r"continuous $\lambda$")
    ax_lambda.set_ylabel("max real growth")
    ax_lambda.grid(alpha=0.22)

    inset = inset_axes(ax_lambda, width="46%", height="46%", loc="upper right", borderpad=1.1)
    for stress, color in zip(stresses, colors):
        subset = [row for row in lambda_rows if abs(float(row["stress"]) - stress) < 1.0e-12]
        inset.plot([float(row["lambda_value"]) for row in subset], [float(row["max_real_growth"]) for row in subset], lw=1.3, color=color)
    inset.axhline(0.0, color="#111111", lw=0.8, ls="--")
    inset.set_xlim(0.0, 2.0)
    inset.set_ylim(-0.045, -0.010)
    inset.set_title(r"near $\lambda=0$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.18)
    mark_inset(ax_lambda, inset, loc1=2, loc2=4, fc="none", ec="#777777", lw=0.8)

    categories = [
        ("resolved", int(float(decision.get("cases_resolved_to_homogeneous_control", 0)))),
        ("different\nno pattern", int(float(decision.get("cases_persist_different_basin_without_pattern", 0)))),
        ("different\nwith pattern", int(float(decision.get("cases_persist_different_basin_with_pattern", 0)))),
        ("unresolved", int(float(decision.get("cases_unresolved", 0)))),
    ]
    followed = int(float(decision.get("followed_cases_count", 0)))
    bars = ax_decision.bar(
        [label for label, _ in categories],
        [value for _, value in categories],
        color=["#2f6fbb", "#aaaaaa", "#777777", "#cccccc"],
        edgecolor="#222222",
        linewidth=0.8,
    )
    for bar, (_, value) in zip(bars, categories):
        ax_decision.text(bar.get_x() + bar.get_width() / 2, value + 0.04, str(value), ha="center", va="bottom", fontsize=9)
    resolved = int(float(decision.get("cases_resolved_to_homogeneous_control", 0)))
    ax_decision.text(
        0.5,
        0.86,
        f"{resolved}/{followed} resolved",
        transform=ax_decision.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#2f6fbb", "linewidth": 1.1},
    )
    ax_decision.set_title("C. Long-horizon decision", loc="left", fontweight="bold")
    ax_decision.set_ylabel("followed cases")
    ax_decision.set_ylim(0, max(2.4, followed + 0.5))
    ax_decision.tick_params(axis="x", labelsize=8)
    ax_decision.grid(axis="y", alpha=0.22)

    fig.suptitle("PDE spatial stability and finite-amplitude follow-up", fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG67_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def nonhomogeneous_field_paths() -> list[Path]:
    names = [
        "local_predator_patch",
        "local_defense_patch",
        "sinusoidal_mode",
        "random_heterogeneity",
        "basin_boundary_heterogeneity",
    ]
    return [RESULTS_DIR / f"roy_pde_nonhomogeneous_fields_{name}.npz" for name in names]


def make_field_snapshot_figure(path: Path, snapshot_index: int, output_path: Path, title: str) -> None:
    files = nonhomogeneous_field_paths()
    variables = [("n", "n_snapshots"), ("w", "w_snapshots"), ("q", "q_snapshots")]
    arrays_by_var: dict[str, list[np.ndarray]] = {key: [] for _, key in variables}
    labels: list[str] = []
    for file_path in files:
        data = np.load(file_path)
        labels.append(clean_label(str(data["perturbation_type"])))
        for _, key in variables:
            arrays_by_var[key].append(np.asarray(data[key][snapshot_index], dtype=float))

    fig, axes = plt.subplots(len(files), len(variables), figsize=(10.8, 12.0), constrained_layout=True)
    for col, (var_label, key) in enumerate(variables):
        vmin = min(float(np.min(arr)) for arr in arrays_by_var[key])
        vmax = max(float(np.max(arr)) for arr in arrays_by_var[key])
        for row, arr in enumerate(arrays_by_var[key]):
            ax = axes[row, col]
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(var_label, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=10)
        cbar = fig.colorbar(im, ax=axes[:, col], fraction=0.028, pad=0.015)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def selected_nonhomogeneous_case_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    selected_types = {
        "homogeneous_control",
        "local_predator_patch",
        "local_defense_patch",
        "sinusoidal_mode",
        "random_heterogeneity",
        "basin_boundary_heterogeneity",
    }
    selected = [
        row
        for row in rows
        if row["stress"] == "0.1584375"
        and row["baseline_state"] == "basin_boundary_state"
        and row["perturbation_type"] in selected_types
        and row["seed"] == "20260702"
    ]
    return sorted(selected, key=lambda row: (row["perturbation_type"], float(row["time"])))


def make_nonhomogeneous_mean_figure() -> None:
    rows = selected_nonhomogeneous_case_rows(NONHOMOG_MEAN_CSV)
    types = sorted({row["perturbation_type"] for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    color_map = dict(zip(types, colors))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for perturbation in types:
        subset = [row for row in rows if row["perturbation_type"] == perturbation]
        label = clean_label(perturbation)
        axes[0].plot([float(row["time"]) for row in subset], [float(row["mean_w"]) for row in subset], lw=1.9, color=color_map[perturbation], label=label)
        axes[1].plot([float(row["time"]) for row in subset], [float(row["mean_q"]) for row in subset], lw=1.9, color=color_map[perturbation], label=label)
    axes[0].set_title("A. Mean predator density")
    axes[1].set_title("B. Mean defense frequency")
    for ax in axes:
        ax.set_xlabel("time")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\bar w(t)$")
    axes[1].set_ylabel(r"$\bar q(t)$")
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("Targeted non-homogeneous PDE mean dynamics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.84, 0.92])
    fig.savefig(FIG51_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_nonhomogeneous_cv_figure() -> None:
    rows = selected_nonhomogeneous_case_rows(NONHOMOG_SPATIAL_CSV)
    types = sorted({row["perturbation_type"] for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    color_map = dict(zip(types, colors))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    floor = 1.0e-14
    for perturbation in types:
        subset = [row for row in rows if row["perturbation_type"] == perturbation]
        max_cv = [max(float(row["cv_n"]), float(row["cv_w"]), float(row["cv_q"]), floor) for row in subset]
        ax.plot([float(row["time"]) for row in subset], max_cv, lw=2.0, color=color_map[perturbation], label=clean_label(perturbation))
    ax.axhline(1.0e-3, color="#222222", lw=1.0, ls="--", label="final pattern threshold")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("max CV across n, w, q")
    ax.set_title("Spatial heterogeneity decay in targeted PDE tests")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    fig.savefig(FIG52_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_long_horizon_mean_figure() -> None:
    rows = read_csv(LONG_HORIZON_MEAN_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True)
    colors = {"0.1584375": "#2f6fbb", "0.16486816": "#9b4a97"}
    styles = {"homogeneous_control": "--", "local_defense_patch": "-"}
    for stress in sorted({row["stress"] for row in rows}, key=float):
        for role in ("homogeneous_control", "local_defense_patch"):
            subset = [row for row in rows if row["stress"] == stress and row["perturbation_type"] == role]
            if not subset:
                continue
            label = f"s={float(stress):.4f}, {clean_label(role)}"
            axes[0].plot([float(row["time"]) for row in subset], [float(row["mean_w"]) for row in subset], color=colors[stress], ls=styles[role], lw=1.9, label=label)
            axes[1].plot([float(row["time"]) for row in subset], [float(row["mean_q"]) for row in subset], color=colors[stress], ls=styles[role], lw=1.9, label=label)
    axes[0].set_title("A. Long-horizon mean predator density")
    axes[1].set_title("B. Long-horizon mean defense frequency")
    for ax in axes:
        ax.set_xlabel("time")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\bar w(t)$")
    axes[1].set_ylabel(r"$\bar q(t)$")
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    fig.savefig(FIG54_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_long_horizon_cv_figure() -> None:
    rows = read_csv(LONG_HORIZON_SPATIAL_CSV)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    colors = {"0.1584375": "#2f6fbb", "0.16486816": "#9b4a97"}
    styles = {"homogeneous_control": "--", "local_defense_patch": "-"}
    floor = 1.0e-14
    for stress in sorted({row["stress"] for row in rows}, key=float):
        for role in ("homogeneous_control", "local_defense_patch"):
            subset = [row for row in rows if row["stress"] == stress and row["perturbation_type"] == role]
            if not subset:
                continue
            max_cv = [max(float(row["cv_n"]), float(row["cv_w"]), float(row["cv_q"]), floor) for row in subset]
            ax.plot([float(row["time"]) for row in subset], max_cv, color=colors[stress], ls=styles[role], lw=2.0, label=f"s={float(stress):.4f}, {clean_label(role)}")
    ax.axhline(1.0e-3, color="#222222", lw=1.0, ls="--", label="final pattern threshold")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("max CV across n, w, q")
    ax.set_title("Long-horizon spatial heterogeneity decay")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(FIG55_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_long_horizon_final_fields() -> None:
    files = sorted(RESULTS_DIR.glob("roy_pde_nonhomogeneous_long_horizon_fields_*.npz"))
    variables = [("n", "n_snapshots"), ("w", "w_snapshots"), ("q", "q_snapshots")]
    arrays_by_var: dict[str, list[np.ndarray]] = {key: [] for _, key in variables}
    row_labels: list[str] = []
    for path in files:
        data = np.load(path)
        row_labels.append(f"s={float(data['stress']):.4f}")
        for _, key in variables:
            arrays_by_var[key].append(np.asarray(data[key][-1], dtype=float))
    fig, axes = plt.subplots(len(files), len(variables), figsize=(10.2, 5.4), constrained_layout=True)
    for col, (var_label, key) in enumerate(variables):
        vmin = min(float(np.min(arr)) for arr in arrays_by_var[key])
        vmax = max(float(np.max(arr)) for arr in arrays_by_var[key])
        for row, arr in enumerate(arrays_by_var[key]):
            ax = axes[row, col]
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(var_label, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)
        cbar = fig.colorbar(im, ax=axes[:, col], fraction=0.035, pad=0.018)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Long-horizon final fields for followed basin-changing cases", fontsize=13, fontweight="bold")
    fig.savefig(FIG56_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_nonlinear_pde_stability_figure() -> None:
    rows = read_csv(NONLINEAR_PDE_STABILITY_CSV)
    shapes = list(dict.fromkeys(row["shape_label"] for row in rows))
    colors = plt.cm.tab10(np.linspace(0, 1, len(shapes)))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for shape, color in zip(shapes, colors):
        subset = sorted([row for row in rows if row["shape_label"] == shape], key=lambda row: float(row["stress"]))
        ax.plot([float(row["stress"]) for row in subset], [float(row["max_nonzero_mode_growth"]) for row in subset], marker="o", lw=2.0, color=color, label=clean_label(shape))
    ax.axhline(0.0, color="#222222", lw=1.0, ls="--")
    ax.set_xlabel("stress s")
    ax.set_ylabel("max nonzero spatial-mode growth")
    ax.set_title("Selected nonlinear PDE spatial-stability diagnostics")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(FIG62_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_nonlinear_nonhomogeneous_figure() -> None:
    rows = read_csv(NONLINEAR_PDE_NONHOMOG_SUMMARY_CSV)
    shapes = list(dict.fromkeys(row["shape_label"] for row in rows))
    max_final_cv = []
    basin_change_counts = []
    for shape in shapes:
        subset = [row for row in rows if row["shape_label"] == shape]
        max_final_cv.append(max(max(float(row["final_cv_n"]), float(row["final_cv_w"]), float(row["final_cv_q"])) for row in subset))
        basin_change_counts.append(sum(1 for row in subset if as_bool(row["basin_changed_relative_to_control"])))
    x = np.arange(len(shapes))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(x, basin_change_counts, color="#6b6b6b", edgecolor="#222222")
    axes[0].set_title("A. Basin changes vs control")
    axes[0].set_ylabel("count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([clean_label(shape) for shape in shapes], rotation=25, ha="right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, np.maximum(max_final_cv, 1.0e-14), color="#2f6fbb", edgecolor="#222222")
    axes[1].axhline(1.0e-3, color="#222222", lw=1.0, ls="--", label="pattern threshold")
    axes[1].set_yscale("log")
    axes[1].set_title("B. Max final spatial CV")
    axes[1].set_ylabel("max final CV")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([clean_label(shape) for shape in shapes], rotation=25, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Selected nonlinear non-homogeneous PDE diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIG63_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_publication_supplement_figures() -> None:
    make_field_snapshot_figure(FIG49_PATH, 0, FIG49_PATH, "Initial fields for targeted non-homogeneous PDE perturbations")
    make_field_snapshot_figure(FIG50_PATH, -1, FIG50_PATH, "Final fields for targeted non-homogeneous PDE perturbations")
    make_nonhomogeneous_mean_figure()
    make_nonhomogeneous_cv_figure()
    make_long_horizon_mean_figure()
    make_long_horizon_cv_figure()
    make_long_horizon_final_fields()
    make_nonlinear_pde_stability_figure()
    make_nonlinear_nonhomogeneous_figure()


def run_rh_margin_export() -> list[dict[str, Any]]:
    source_rows = read_csv(RH_CURRENT_CSV)
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        A1 = float(source["A1"])
        A2 = float(source["A2"])
        A3 = float(source["A3"])
        margin = float(source["A1A2_minus_A3"])
        rows.append(
            {
                "stress": float(source["stress"]),
                "A1": A1,
                "A2": A2,
                "A3": A3,
                "A1A2_minus_A3": margin,
                "minimum_rh_margin": min(A1, A2, A3, margin),
                "routh_hurwitz_stable": source["routh_hurwitz_stable"],
                "max_real_eigenvalue": source["max_real_eigenvalue"],
            }
        )
    write_csv(RH_MARGINS_CSV, rows, RH_MARGIN_FIELDS)
    make_rh_margin_eigenvalue_figure(rows)
    return rows


def run(profile: str) -> None:
    del profile
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    run_counterfactual()
    run_lambda_scan()
    run_rh_margin_export()
    make_compensation_branch_figure()
    make_nonlinear_tradeoff_figure()
    make_main_pde_evidence_figure()
    make_publication_supplement_figures()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused",), default="focused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
