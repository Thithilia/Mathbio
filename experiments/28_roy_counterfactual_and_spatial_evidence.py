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
FIG64_PATH = FIG_DIR / "fig64_nonlinear_tradeoff_final_decision.png"
FIG65_PATH = FIG_DIR / "fig65_no_evolution_counterfactual.png"
FIG66_PATH = FIG_DIR / "fig66_pde_continuous_lambda_scan.png"

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
        ax_q.legend(frameon=False)
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
    colors = ["#2f7d57", "#f0b44c", "#c95b59", "#8b8b8b"]
    labels = {
        "robust_compensation_shape": "robust",
        "partial_compensation_shape": "partial",
        "no_compensation_shape": "no compensation",
        "unresolved_shape": "unresolved",
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
                ax.text(x_idx, y_idx, labels[class_order[code]][0].upper(), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, len(gamma_values) - 0.5)
        ax.set_ylim(-0.5, len(gamma_values) - 0.5)

    handles = [Patch(facecolor=colors[idx], edgecolor="none", label=f"{labels[name]} ({counts.get(name, 0)})") for idx, name in enumerate(class_order)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Controlled nonlinear trade-off shape grid", fontsize=13)
    fig.text(0.5, 0.06, "Cell letters: R robust, P partial, N no compensation, U unresolved", ha="center", fontsize=8)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused",), default="focused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
