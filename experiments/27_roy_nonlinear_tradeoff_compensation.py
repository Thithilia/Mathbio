"""Controlled nonlinear trade-off extension of the Roy compensation branch.

This script keeps the baseline model implementation unchanged. It defines
endpoint-preserving nonlinear trade-off helpers locally, verifies recovery of
the linear compensation branch, evaluates a small structured shape grid, and
runs targeted ODE/PDE diagnostics for selected shapes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (  # noqa: E402
    RoyEvoPDEConfig,
    RoyEvoParams,
    find_evo_equilibrium,
    free_space_evo,
    grid_2d_evo,
    laplacian_neumann_2d_evo,
)


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTES_DIR = ROOT / "research_notes"
MANUSCRIPT_DIR = ROOT / "manuscript"

BRANCH_RECOVERY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_branch_recovery_linear.csv"
BRANCH_GRID_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_branch_stability_grid.csv"
SHAPE_SUMMARY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_shape_summary.csv"
BASIN_MAP_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_selected_ode_basin_maps.csv"
PDE_SPATIAL_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_pde_spatial_stability.csv"
PDE_NONHOM_SUMMARY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_pde_nonhomogeneous_summary.csv"
PDE_NONHOM_TIMESERIES_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_pde_nonhomogeneous_timeseries.csv"
PDE_NONHOM_SPATIAL_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_pde_nonhomogeneous_spatial_metrics.csv"
DECISION_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_compensation_decision.csv"
NOTE_PATH = NOTES_DIR / "roy_nonlinear_tradeoff_compensation_extension.md"

FIG59_PATH = FIG_DIR / "fig59_nonlinear_branch_curves.png"
FIG60_PATH = FIG_DIR / "fig60_nonlinear_shape_grid_summary.png"
FIG61_PATH = FIG_DIR / "fig61_nonlinear_selected_ode_basin_maps.png"
FIG62_PATH = FIG_DIR / "fig62_nonlinear_pde_spatial_stability.png"
FIG63_PATH = FIG_DIR / "fig63_nonlinear_nonhomogeneous_pde_tests.png"
FIG64_PATH = FIG_DIR / "fig64_nonlinear_tradeoff_final_decision.png"

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
TARGET_STRESSES = (0.11765625, 0.1584375, 0.16486816, 0.175)
BASIN_STRESSES = (0.1584375, 0.16486816)
PDE_STRESSES_FOCUSED = (0.1584375,)
GAMMA_VALUES = (0.5, 1.0, 2.0)
Q0_VALUES = tuple(round(0.1 * idx, 1) for idx in range(10))
W0_SCALES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5)

BASE_D_N = 0.01
BASE_D_W = 0.01
BASE_D_Q = 0.005
L_X = 20.0
L_Y = 20.0
N_X = 64
N_Y = 64
DT = 0.1
PDE_T_FOCUSED = 3200.0
PDE_T_LONG = 6400.0
RECORD_EVERY = 400
SEED = 20260702

Q_EPS = 1.0e-6
EPSILON = 1.0e-4
EXTREME_EXTINCTION_W = 1.0e-8
TAIL_FRACTION = 0.25
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 5.0e-5
FINAL_CV_THRESHOLD = 1.0e-3
MAX_CV_THRESHOLD = 1.0e-2
POSITIVE_MODE_TOL = 1.0e-8
NEAR_ZERO_TOL = 1.0e-7

BASIN_LABELS = ("persistent_basin", "extinct_basin", "transient_basin", "unresolved_basin")
BASIN_COLORS = {
    "persistent_basin": "#2f6fbb",
    "extinct_basin": "#c23b3b",
    "transient_basin": "#d9a441",
    "unresolved_basin": "#7f7f7f",
}

BRANCH_RECOVERY_FIELDS = (
    "stress",
    "q_star_generalized",
    "q_star_linear_reference",
    "abs_delta_q",
    "n_star_generalized",
    "n_star_linear_reference",
    "abs_delta_n",
    "w_star_generalized",
    "w_star_linear_reference",
    "abs_delta_w",
    "recovered_linear_branch",
)

BRANCH_GRID_FIELDS = (
    "gamma_r",
    "gamma_a",
    "gamma_b",
    "stress",
    "branch_found",
    "q_star",
    "n_star",
    "w_star",
    "z_star",
    "s_of_q_error",
    "max_real_eigenvalue",
    "locally_stable",
    "feasible",
    "branch_class",
    "notes",
)

SHAPE_SUMMARY_FIELDS = (
    "gamma_r",
    "gamma_a",
    "gamma_b",
    "target_stresses_total",
    "target_stresses_with_branch",
    "target_stresses_stable",
    "min_q_star",
    "max_q_star",
    "stress_interval_low_estimate",
    "stress_interval_high_estimate",
    "shape_class",
)

BASIN_MAP_FIELDS = (
    "shape_label",
    "gamma_r",
    "gamma_a",
    "gamma_b",
    "stress",
    "q0",
    "w0_scale",
    "classification",
    "basin_label",
    "tail_mean_w",
    "tail_mean_q",
    "q_change",
    "physical",
    "notes",
)

PDE_SPATIAL_FIELDS = (
    "shape_label",
    "gamma_r",
    "gamma_a",
    "gamma_b",
    "stress",
    "q_star",
    "zero_mode_growth",
    "max_nonzero_mode_growth",
    "most_unstable_m",
    "most_unstable_n",
    "positive_modes_count",
    "near_zero_modes_count",
    "linearly_spatially_stable",
    "notes",
)

PDE_NONHOM_FIELDS = (
    "shape_label",
    "gamma_r",
    "gamma_a",
    "gamma_b",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "T",
    "classification",
    "basin_label",
    "homogeneous_control_basin_label",
    "basin_changed_relative_to_control",
    "tail_mean_w",
    "tail_mean_q",
    "relative_change_between_last_windows",
    "normalized_residual",
    "initial_cv_n",
    "initial_cv_w",
    "initial_cv_q",
    "final_cv_n",
    "final_cv_w",
    "final_cv_q",
    "max_cv_n",
    "max_cv_w",
    "max_cv_q",
    "spatial_pattern_persisted",
    "physical",
    "notes",
)

PDE_TIMESERIES_FIELDS = (
    "shape_label",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "T",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
)

PDE_SPATIAL_METRIC_FIELDS = (
    "shape_label",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "T",
    "time",
    "var_n",
    "var_w",
    "var_q",
    "cv_n",
    "cv_w",
    "cv_q",
)


@dataclass(frozen=True)
class ShapeSpec:
    label: str
    gamma_r: float
    gamma_a: float
    gamma_b: float


@dataclass
class SnapshotPDEResult:
    times: np.ndarray
    mean_n: np.ndarray
    mean_w: np.ndarray
    mean_q: np.ndarray
    var_n: np.ndarray
    var_w: np.ndarray
    var_q: np.ndarray
    n_final: np.ndarray
    w_final: np.ndarray
    q_final: np.ndarray
    n_snapshots: np.ndarray
    w_snapshots: np.ndarray
    q_snapshots: np.ndarray
    snapshot_times: np.ndarray
    min_n: float
    min_w: float
    min_q: float
    max_q: float
    min_z: float
    q_clip_max_violation: float
    completed: bool
    nonfinite_detected: bool


def format_float(value: float, digits: int = 7) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def read_summary_metric(path: Path, metric: str, default: str = "") -> str:
    if not path.exists():
        return default
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == metric:
                return str(row.get("value", default))
    return default


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def shape_function(q: np.ndarray | float, gamma: float) -> np.ndarray | float:
    q_arr = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    return np.power(q_arr, float(gamma))


def shape_derivative(q: np.ndarray | float, gamma: float) -> np.ndarray | float:
    q_arr = np.clip(np.asarray(q, dtype=float), Q_EPS, 1.0 - Q_EPS)
    return float(gamma) * np.power(q_arr, float(gamma) - 1.0)


def tradeoffs(
    q: np.ndarray | float,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    r = params.r_u + (params.r_v - params.r_u) * shape_function(q, gamma_r)
    a = params.a_u + (params.a_v - params.a_u) * shape_function(q, gamma_a)
    b = params.b_u + (params.b_v - params.b_u) * shape_function(q, gamma_b)
    return r, a, b


def tradeoff_derivatives(
    q: np.ndarray | float,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    r_prime = (params.r_v - params.r_u) * shape_derivative(q, gamma_r)
    a_prime = (params.a_v - params.a_u) * shape_derivative(q, gamma_a)
    b_prime = (params.b_v - params.b_u) * shape_derivative(q, gamma_b)
    return r_prime, a_prime, b_prime


def branch_point_from_q(
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    q: float,
) -> dict[str, Any]:
    r, a, b = [float(value) for value in tradeoffs(q, params, gamma_r, gamma_a, gamma_b)]
    r_prime, a_prime, _b_prime = [
        float(value) for value in tradeoff_derivatives(q, params, gamma_r, gamma_a, gamma_b)
    ]
    if abs(a_prime) <= 1.0e-14:
        return {
            "q": q,
            "c": math.nan,
            "z": math.nan,
            "w": math.nan,
            "n": math.nan,
            "s": math.nan,
            "feasible": False,
            "notes": "a_prime_zero",
        }
    c = r_prime / a_prime
    denominator = r - a * c
    if abs(denominator) <= 1.0e-14:
        return {
            "q": q,
            "c": c,
            "z": math.nan,
            "w": math.nan,
            "n": math.nan,
            "s": math.nan,
            "feasible": False,
            "notes": "prey_balance_denominator_zero",
        }
    z = params.xi / denominator
    w = c * z
    n = 1.0 / params.kappa - z - w
    s = b * n * z - params.m - params.mu * w
    feasible = bool(0.0 < q < 1.0 and z > 0.0 and w > 0.0 and n > 0.0 and np.isfinite(s))
    return {
        "q": q,
        "c": c,
        "z": z,
        "w": w,
        "n": n,
        "s": s,
        "feasible": feasible,
        "notes": "feasible" if feasible else "failed_positive_feasibility",
    }


def generalized_branch_curve(
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    q_grid: np.ndarray,
) -> list[dict[str, Any]]:
    return [branch_point_from_q(params, gamma_r, gamma_a, gamma_b, float(q)) for q in q_grid]


def generalized_ode_rhs(
    _t: float,
    y: np.ndarray,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> np.ndarray:
    n, w, q = np.asarray(y, dtype=float)
    q_unit = float(np.clip(q, 0.0, 1.0))
    z = float(free_space_evo(n, w, params))
    r, a, b = [float(value) for value in tradeoffs(q_unit, params, gamma_r, gamma_a, gamma_b)]
    r_prime, a_prime, _b_prime = [
        float(value) for value in tradeoff_derivatives(q_unit, params, gamma_r, gamma_a, gamma_b)
    ]
    selection = r_prime * z - a_prime * w
    dn = n * (r * z - params.xi - a * w)
    dw = w * (b * n * z - (params.m + stress) - params.mu * w)
    dq = params.nu * q_unit * (1.0 - q_unit) * selection
    return np.array([dn, dw, dq], dtype=float)


def generalized_reaction_part_arrays(
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_unit = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    z = free_space_evo(n, w, params)
    r, a, b = tradeoffs(q_unit, params, gamma_r, gamma_a, gamma_b)
    r_prime, a_prime, _b_prime = tradeoff_derivatives(q_unit, params, gamma_r, gamma_a, gamma_b)
    selection = r_prime * z - a_prime * w
    dn = n * (r * z - params.xi - a * w)
    dw = w * (b * n * z - (params.m + stress) - params.mu * w)
    dq = params.nu * q_unit * (1.0 - q_unit) * selection
    return np.asarray(dn, dtype=float), np.asarray(dw, dtype=float), np.asarray(dq, dtype=float)


def generalized_jacobian_fd(
    y: np.ndarray,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
    step: float = 1.0e-6,
) -> np.ndarray:
    x = np.asarray(y, dtype=float)
    f0 = generalized_ode_rhs(0.0, x, params, gamma_r, gamma_a, gamma_b, stress)
    jac = np.empty((f0.size, x.size), dtype=float)
    for idx in range(x.size):
        delta = step * max(1.0, abs(float(x[idx])))
        plus = x.copy()
        minus = x.copy()
        plus[idx] += delta
        minus[idx] -= delta
        if idx == 2:
            plus[idx] = min(1.0 - Q_EPS, plus[idx])
            minus[idx] = max(Q_EPS, minus[idx])
        jac[:, idx] = (
            generalized_ode_rhs(0.0, plus, params, gamma_r, gamma_a, gamma_b, stress)
            - generalized_ode_rhs(0.0, minus, params, gamma_r, gamma_a, gamma_b, stress)
        ) / (plus[idx] - minus[idx])
    return jac


def eigen_max_real(jacobian: np.ndarray) -> float:
    eig = np.linalg.eigvals(jacobian)
    return float(np.max(np.real(eig)))


def linear_reference_branch(params: RoyEvoParams, stress: float) -> dict[str, float]:
    delta_r = params.r_v - params.r_u
    delta_a = params.a_v - params.a_u
    delta_b = params.b_v - params.b_u
    c = delta_r / delta_a
    z = params.xi / (params.r_u - c * params.a_u)
    w = c * z
    n = 1.0 / params.kappa - z - w
    b_req = (params.m + stress + params.mu * w) / (n * z)
    q = (b_req - params.b_u) / delta_b
    return {"n": n, "w": w, "q": q, "z": z, "b_req": b_req}


def find_branch_candidates_at_stress(
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
    q_grid: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    q_values = np.linspace(Q_EPS, 1.0 - Q_EPS, 4001) if q_grid is None else np.asarray(q_grid, dtype=float)
    curve = generalized_branch_curve(params, gamma_r, gamma_a, gamma_b, q_values)
    feasible = [row for row in curve if bool(row["feasible"]) and np.isfinite(float(row["s"]))]
    if not feasible:
        return []

    def s_minus_target(q: float) -> float:
        point = branch_point_from_q(params, gamma_r, gamma_a, gamma_b, q)
        return float(point["s"]) - stress

    roots: list[float] = []
    last = feasible[0]
    last_value = float(last["s"]) - stress
    if abs(last_value) < 1.0e-7:
        roots.append(float(last["q"]))
    for row in feasible[1:]:
        value = float(row["s"]) - stress
        if abs(value) < 1.0e-7:
            roots.append(float(row["q"]))
        elif np.isfinite(value) and np.isfinite(last_value) and value * last_value < 0.0:
            try:
                roots.append(float(brentq(s_minus_target, float(last["q"]), float(row["q"]), maxiter=100)))
            except ValueError:
                pass
        last = row
        last_value = value

    if not roots:
        nearest = min(feasible, key=lambda row: abs(float(row["s"]) - stress))
        if abs(float(nearest["s"]) - stress) < 1.0e-5:
            roots.append(float(nearest["q"]))

    candidates: list[dict[str, Any]] = []
    for q in sorted(set(round(root, 12) for root in roots)):
        point = branch_point_from_q(params, gamma_r, gamma_a, gamma_b, q)
        if not bool(point["feasible"]):
            continue
        state = np.array([float(point["n"]), float(point["w"]), q], dtype=float)
        jac = generalized_jacobian_fd(state, params, gamma_r, gamma_a, gamma_b, stress)
        max_real = eigen_max_real(jac)
        candidates.append(
            {
                **point,
                "q": q,
                "s_of_q_error": float(point["s"]) - stress,
                "max_real_eigenvalue": max_real,
                "locally_stable": bool(max_real < -1.0e-8),
            }
        )
    return candidates


def find_branch_state_at_stress(
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> dict[str, Any]:
    candidates = find_branch_candidates_at_stress(params, gamma_r, gamma_a, gamma_b, stress)
    if not candidates:
        return {
            "branch_found": False,
            "q_star": math.nan,
            "n_star": math.nan,
            "w_star": math.nan,
            "z_star": math.nan,
            "s_of_q_error": math.nan,
            "max_real_eigenvalue": math.nan,
            "locally_stable": False,
            "feasible": False,
            "notes": "no_feasible_q_root",
        }
    stable = [candidate for candidate in candidates if bool(candidate["locally_stable"])]
    chosen = min(stable or candidates, key=lambda row: (abs(float(row["s_of_q_error"])), float(row["q"])))
    return {
        "branch_found": True,
        "q_star": float(chosen["q"]),
        "n_star": float(chosen["n"]),
        "w_star": float(chosen["w"]),
        "z_star": float(chosen["z"]),
        "s_of_q_error": float(chosen["s_of_q_error"]),
        "max_real_eigenvalue": float(chosen["max_real_eigenvalue"]),
        "locally_stable": bool(chosen["locally_stable"]),
        "feasible": bool(chosen["feasible"]),
        "notes": f"candidate_roots={len(candidates)}",
    }


def branch_class(row: dict[str, Any]) -> str:
    if not bool(row["branch_found"]):
        return "no_feasible_branch"
    if not bool(row["feasible"]):
        return "branch_search_failed"
    if bool(row["locally_stable"]):
        return "feasible_stable_branch"
    return "feasible_unstable_branch"


def write_linear_recovery() -> tuple[list[dict[str, Any]], bool, float]:
    rows: list[dict[str, Any]] = []
    max_delta_q = 0.0
    for stress in TARGET_STRESSES:
        generalized = find_branch_state_at_stress(PARAMS, 1.0, 1.0, 1.0, stress)
        linear = linear_reference_branch(PARAMS, stress)
        delta_q = abs(float(generalized["q_star"]) - float(linear["q"])) if generalized["branch_found"] else math.inf
        delta_n = abs(float(generalized["n_star"]) - float(linear["n"])) if generalized["branch_found"] else math.inf
        delta_w = abs(float(generalized["w_star"]) - float(linear["w"])) if generalized["branch_found"] else math.inf
        max_delta_q = max(max_delta_q, delta_q)
        rows.append(
            {
                "stress": stress,
                "q_star_generalized": generalized["q_star"],
                "q_star_linear_reference": linear["q"],
                "abs_delta_q": delta_q,
                "n_star_generalized": generalized["n_star"],
                "n_star_linear_reference": linear["n"],
                "abs_delta_n": delta_n,
                "w_star_generalized": generalized["w_star"],
                "w_star_linear_reference": linear["w"],
                "abs_delta_w": delta_w,
                "recovered_linear_branch": bool(delta_q < 1.0e-5 and delta_n < 1.0e-5 and delta_w < 1.0e-5),
            }
        )
    write_csv(BRANCH_RECOVERY_CSV, rows, BRANCH_RECOVERY_FIELDS)
    return rows, all(bool(row["recovered_linear_branch"]) for row in rows), max_delta_q


def run_branch_stability_grid() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for gamma_r in GAMMA_VALUES:
        for gamma_a in GAMMA_VALUES:
            for gamma_b in GAMMA_VALUES:
                for stress in TARGET_STRESSES:
                    branch = find_branch_state_at_stress(PARAMS, gamma_r, gamma_a, gamma_b, stress)
                    row = {
                        "gamma_r": gamma_r,
                        "gamma_a": gamma_a,
                        "gamma_b": gamma_b,
                        "stress": stress,
                        **branch,
                    }
                    row["branch_class"] = branch_class(row)
                    rows.append(row)
    write_csv(BRANCH_GRID_CSV, rows, BRANCH_GRID_FIELDS)

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["gamma_r"]), float(row["gamma_a"]), float(row["gamma_b"]))].append(row)
    q_grid = np.linspace(Q_EPS, 1.0 - Q_EPS, 4001)
    for (gamma_r, gamma_a, gamma_b), group in sorted(grouped.items()):
        with_branch = [row for row in group if bool(row["branch_found"])]
        stable = [row for row in with_branch if bool(row["locally_stable"])]
        qs = [float(row["q_star"]) for row in with_branch if np.isfinite(float(row["q_star"]))]
        curve = generalized_branch_curve(PARAMS, gamma_r, gamma_a, gamma_b, q_grid)
        feasible_s = [float(row["s"]) for row in curve if bool(row["feasible"]) and np.isfinite(float(row["s"]))]
        if len(stable) == len(TARGET_STRESSES):
            shape_class = "robust_compensation_shape"
        elif stable:
            shape_class = "partial_compensation_shape"
        elif with_branch:
            shape_class = "unresolved_shape"
        else:
            shape_class = "no_compensation_shape"
        summary_rows.append(
            {
                "gamma_r": gamma_r,
                "gamma_a": gamma_a,
                "gamma_b": gamma_b,
                "target_stresses_total": len(TARGET_STRESSES),
                "target_stresses_with_branch": len(with_branch),
                "target_stresses_stable": len(stable),
                "min_q_star": min(qs) if qs else math.nan,
                "max_q_star": max(qs) if qs else math.nan,
                "stress_interval_low_estimate": min(feasible_s) if feasible_s else math.nan,
                "stress_interval_high_estimate": max(feasible_s) if feasible_s else math.nan,
                "shape_class": shape_class,
            }
        )
    write_csv(SHAPE_SUMMARY_CSV, summary_rows, SHAPE_SUMMARY_FIELDS)
    return rows, summary_rows


def baseline_burnin_state() -> tuple[float, float, float]:
    eq = find_evo_equilibrium(PARAMS)
    return float(eq["n"]), float(eq["w"]), float(eq["q"])


def tail_mask(times: np.ndarray, fraction: float = TAIL_FRACTION) -> np.ndarray:
    cutoff = float(times[-1]) - fraction * float(times[-1] - times[0])
    mask = times >= cutoff
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(times, dtype=bool)
        mask[-2:] = True
    return mask


def previous_window_mask(times: np.ndarray, fraction: float = TAIL_FRACTION) -> np.ndarray:
    span = float(times[-1] - times[0])
    latest_start = float(times[-1]) - fraction * span
    previous_start = float(times[-1]) - 2.0 * fraction * span
    mask = (times >= previous_start) & (times < latest_start)
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(times, dtype=bool)
        midpoint = max(1, len(times) // 2)
        mask[max(0, midpoint - 2) : midpoint] = True
    return mask


def tail_slope(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    x = times[mask]
    y = values[mask]
    if len(x) < 2:
        return 0.0
    centered = x - float(np.mean(x))
    denom = float(np.dot(centered, centered))
    return 0.0 if denom <= 0.0 else float(np.dot(centered, y - float(np.mean(y))) / denom)


def relative_change(previous: float, latest: float) -> float:
    return float((latest - previous) / max(abs(previous), EPSILON))


def ode_rhs_residual(
    state: np.ndarray,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> dict[str, float]:
    rhs = generalized_ode_rhs(0.0, state, params, gamma_r, gamma_a, gamma_b, stress)
    rhs_norm = float(np.linalg.norm(rhs))
    state_norm = float(np.linalg.norm(state))
    return {"rhs_norm": rhs_norm, "state_norm": state_norm, "normalized_residual": rhs_norm / max(state_norm, 1.0e-12)}


def basin_label_from_classification(classification: str) -> str:
    if classification == "persistent_steady":
        return "persistent_basin"
    if classification == "extinct_steady":
        return "extinct_basin"
    if classification in {"persistent_transient", "extinct_transient", "recovery_transient", "declining_transient"}:
        return "transient_basin"
    if classification == "nonphysical":
        return "unresolved_basin"
    return "unresolved_basin"


def classify_asymptotic(metrics: dict[str, Any]) -> str:
    physical = bool(metrics["physical"])
    tail_mean_w = float(metrics["tail_mean_w"])
    tail_min_w = float(metrics.get("tail_min_w", tail_mean_w))
    previous_w = float(metrics["previous_window_mean_w"])
    latest_w = float(metrics["latest_window_mean_w"])
    rel_change = float(metrics["relative_change_between_last_windows"])
    normalized_residual = float(metrics["normalized_residual"])
    persistent_without_slope = bool(metrics.get("persistent_without_slope_rule", False))
    persistent_with_slope = bool(metrics.get("persistent_with_slope_rule", False))
    if not physical:
        return "nonphysical"
    if (
        tail_mean_w > EPSILON
        and tail_min_w > 0.25 * EPSILON
        and persistent_without_slope
        and abs(rel_change) < STEADY_REL_CHANGE_TOL
        and normalized_residual < STEADY_RESIDUAL_TOL
    ):
        return "persistent_steady"
    if (
        tail_mean_w < EPSILON
        and latest_w < EPSILON
        and previous_w < EPSILON
        and (normalized_residual < STEADY_RESIDUAL_TOL or tail_mean_w < EXTREME_EXTINCTION_W)
    ):
        return "extinct_steady"
    if tail_mean_w > EPSILON and rel_change < -STEADY_REL_CHANGE_TOL:
        return "declining_transient"
    if tail_mean_w > EPSILON and persistent_without_slope and not persistent_with_slope:
        return "declining_transient"
    if tail_mean_w > EPSILON and rel_change > STEADY_REL_CHANGE_TOL:
        return "recovery_transient"
    if tail_mean_w > EPSILON and persistent_without_slope:
        return "persistent_transient"
    if tail_mean_w <= EPSILON:
        return "extinct_transient"
    return "indeterminate"


def classify_ode_trajectory(
    times: np.ndarray,
    values: np.ndarray,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> dict[str, Any]:
    n, w, q = values
    z = np.asarray(free_space_evo(n, w, params), dtype=float)
    physical = bool(
        np.all(np.isfinite(values))
        and np.all(n >= -1.0e-8)
        and np.all(w >= -1.0e-8)
        and np.all(q >= -1.0e-6)
        and np.all(q <= 1.0 + 1.0e-6)
        and np.all(z >= -1.0e-5)
    )
    mask = tail_mask(times)
    prev = previous_window_mask(times)
    tail_t = times[mask]
    tail_w = w[mask]
    tail_q = q[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(times, w, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_w = float(np.mean(w[prev]))
    residual = ode_rhs_residual(values[:, -1], params, gamma_r, gamma_a, gamma_b, stress)
    persistent_without_slope = bool(physical and tail_mean_w > EPSILON and tail_min_w > 0.25 * EPSILON)
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    metrics: dict[str, Any] = {
        "physical": physical,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": slope_floor,
        "tail_mean_q": float(np.mean(tail_q)),
        "previous_window_mean_w": previous_w,
        "latest_window_mean_w": tail_mean_w,
        "relative_change_between_last_windows": relative_change(previous_w, tail_mean_w),
        "persistent_without_slope_rule": persistent_without_slope,
        "persistent_with_slope_rule": persistent_with_slope,
        **residual,
    }
    metrics["classification"] = classify_asymptotic(metrics)
    metrics["basin_label"] = basin_label_from_classification(str(metrics["classification"]))
    return metrics


def simulate_generalized_ode(
    initial_state: np.ndarray,
    params: RoyEvoParams,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
    T: float,
    n_eval: int,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    t_eval = np.linspace(0.0, T, n_eval)
    result = solve_ivp(
        lambda t, y: generalized_ode_rhs(t, y, params, gamma_r, gamma_a, gamma_b, stress),
        (0.0, T),
        np.asarray(initial_state, dtype=float),
        t_eval=t_eval,
        method="LSODA",
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    return result.t, result.y, bool(result.success), str(result.message)


def select_shape_specs(shape_summary_rows: list[dict[str, Any]]) -> list[ShapeSpec]:
    fixed = [
        ShapeSpec("linear_reference", 1.0, 1.0, 1.0),
        ShapeSpec("all_concave", 0.5, 0.5, 0.5),
        ShapeSpec("all_convex", 2.0, 2.0, 2.0),
        ShapeSpec("mixed_growth_concave_conversion_convex", 0.5, 0.5, 2.0),
    ]
    used = {(spec.gamma_r, spec.gamma_a, spec.gamma_b) for spec in fixed}
    candidates = [row for row in shape_summary_rows if (float(row["gamma_r"]), float(row["gamma_a"]), float(row["gamma_b"])) not in used]
    no_branch = [row for row in candidates if str(row["shape_class"]) == "no_compensation_shape"]
    partial = [row for row in candidates if str(row["shape_class"]) in {"partial_compensation_shape", "unresolved_shape"}]
    pool = no_branch or sorted(partial or candidates, key=lambda row: (int(row["target_stresses_stable"]), float(row["stress_interval_high_estimate"]) - float(row["stress_interval_low_estimate"]) if np.isfinite(float(row["stress_interval_high_estimate"])) and np.isfinite(float(row["stress_interval_low_estimate"])) else math.inf))
    if pool:
        chosen = pool[0]
        fixed.append(
            ShapeSpec(
                "weak_or_failed_shape",
                float(chosen["gamma_r"]),
                float(chosen["gamma_a"]),
                float(chosen["gamma_b"]),
            )
        )
    return fixed


def run_selected_ode_basin_maps(shape_specs: list[ShapeSpec], profile: str) -> list[dict[str, Any]]:
    T = 1600.0 if profile == "focused" else 400.0
    n_eval = 321 if profile == "focused" else 101
    baseline_n, baseline_w, _baseline_q = baseline_burnin_state()
    rows: list[dict[str, Any]] = []
    for spec in shape_specs:
        for stress in BASIN_STRESSES:
            for q0 in Q0_VALUES:
                for w0_scale in W0_SCALES:
                    initial = np.array([baseline_n, baseline_w * w0_scale, q0], dtype=float)
                    z0 = float(free_space_evo(initial[0], initial[1], PARAMS))
                    if z0 < 0.0:
                        rows.append(
                            {
                                "shape_label": spec.label,
                                "gamma_r": spec.gamma_r,
                                "gamma_a": spec.gamma_a,
                                "gamma_b": spec.gamma_b,
                                "stress": stress,
                                "q0": q0,
                                "w0_scale": w0_scale,
                                "classification": "nonphysical",
                                "basin_label": "unresolved_basin",
                                "tail_mean_w": math.nan,
                                "tail_mean_q": math.nan,
                                "q_change": math.nan,
                                "physical": False,
                                "notes": "skipped_nonphysical_initial_condition",
                            }
                        )
                        continue
                    times, values, success, message = simulate_generalized_ode(
                        initial, PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, stress, T, n_eval
                    )
                    metrics = classify_ode_trajectory(times, values, PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, stress)
                    rows.append(
                        {
                            "shape_label": spec.label,
                            "gamma_r": spec.gamma_r,
                            "gamma_a": spec.gamma_a,
                            "gamma_b": spec.gamma_b,
                            "stress": stress,
                            "q0": q0,
                            "w0_scale": w0_scale,
                            "classification": metrics["classification"] if success else "indeterminate",
                            "basin_label": metrics["basin_label"],
                            "tail_mean_w": metrics["tail_mean_w"],
                            "tail_mean_q": metrics["tail_mean_q"],
                            "q_change": float(metrics["tail_mean_q"] - q0),
                            "physical": metrics["physical"],
                            "notes": "ode_solve_success" if success else message,
                        }
                    )
    write_csv(BASIN_MAP_CSV, rows, BASIN_MAP_FIELDS)
    return rows


def neumann_eigenvalue(m: int, n: int, L_x: float = L_X, L_y: float = L_Y) -> float:
    return float((m * math.pi / L_x) ** 2 + (n * math.pi / L_y) ** 2)


def modal_matrix(jacobian: np.ndarray, lambda_mn: float, diffusion: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(jacobian, dtype=float) - float(lambda_mn) * np.diag(np.asarray(diffusion, dtype=float))


def pde_spatial_stability_detects_instability(growths: np.ndarray, tol: float = POSITIVE_MODE_TOL) -> bool:
    values = np.asarray(growths, dtype=float)
    return bool(values.size > 0 and np.any(values > tol))


def run_pde_spatial_stability(shape_specs: list[ShapeSpec], mode_max: int = 32) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diffusion = (BASE_D_N, BASE_D_W, BASE_D_Q)
    for spec in shape_specs:
        for stress in TARGET_STRESSES:
            branch = find_branch_state_at_stress(PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, stress)
            if not (bool(branch["branch_found"]) and bool(branch["locally_stable"])):
                rows.append(
                    {
                        "shape_label": spec.label,
                        "gamma_r": spec.gamma_r,
                        "gamma_a": spec.gamma_a,
                        "gamma_b": spec.gamma_b,
                        "stress": stress,
                        "q_star": branch["q_star"],
                        "zero_mode_growth": branch["max_real_eigenvalue"],
                        "max_nonzero_mode_growth": math.nan,
                        "most_unstable_m": "",
                        "most_unstable_n": "",
                        "positive_modes_count": "",
                        "near_zero_modes_count": "",
                        "linearly_spatially_stable": False,
                        "notes": "skipped_no_locally_stable_branch",
                    }
                )
                continue
            state = np.array([float(branch["n_star"]), float(branch["w_star"]), float(branch["q_star"])])
            jac = generalized_jacobian_fd(state, PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, stress)
            zero_growth = eigen_max_real(jac)
            nonzero_rows: list[dict[str, Any]] = []
            for m in range(mode_max + 1):
                for n in range(mode_max + 1):
                    if m == 0 and n == 0:
                        continue
                    lam = neumann_eigenvalue(m, n)
                    growth = eigen_max_real(modal_matrix(jac, lam, diffusion))
                    nonzero_rows.append({"m": m, "n": n, "growth": growth, "lambda": lam})
            growths = np.array([float(row["growth"]) for row in nonzero_rows], dtype=float)
            max_row = max(nonzero_rows, key=lambda row: float(row["growth"]))
            positive_count = int(np.count_nonzero(growths > POSITIVE_MODE_TOL))
            near_zero_count = int(np.count_nonzero(np.abs(growths) <= NEAR_ZERO_TOL))
            rows.append(
                {
                    "shape_label": spec.label,
                    "gamma_r": spec.gamma_r,
                    "gamma_a": spec.gamma_a,
                    "gamma_b": spec.gamma_b,
                    "stress": stress,
                    "q_star": branch["q_star"],
                    "zero_mode_growth": zero_growth,
                    "max_nonzero_mode_growth": max_row["growth"],
                    "most_unstable_m": max_row["m"],
                    "most_unstable_n": max_row["n"],
                    "positive_modes_count": positive_count,
                    "near_zero_modes_count": near_zero_count,
                    "linearly_spatially_stable": positive_count == 0,
                    "notes": "nonzero_neumann_modes_tested",
                }
            )
    write_csv(PDE_SPATIAL_CSV, rows, PDE_SPATIAL_FIELDS)
    return rows


def pde_config(T: float) -> RoyEvoPDEConfig:
    return RoyEvoPDEConfig(
        n_x=N_X,
        n_y=N_Y,
        L_x=L_X,
        L_y=L_Y,
        dt=DT,
        T=T,
        record_every=RECORD_EVERY,
        D_n=BASE_D_N,
        D_w=BASE_D_W,
        D_q=BASE_D_Q,
        perturbation_amplitude=0.0,
        seed=SEED,
        clip_q=True,
    )


def coefficient_of_variation(field: np.ndarray, mean_value: float | None = None) -> float:
    array = np.asarray(field, dtype=float)
    mean = float(np.mean(array)) if mean_value is None else float(mean_value)
    return float(np.std(array) / max(abs(mean), 1.0e-12))


def smooth_mean_zero_noise(shape: tuple[int, int], rng: np.random.Generator, passes: int = 8) -> np.ndarray:
    noise = rng.standard_normal(shape)
    for _ in range(passes):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0
    noise = noise - float(np.mean(noise))
    scale = float(np.max(np.abs(noise)))
    return noise / scale if scale > 0.0 else noise


def enforce_physical(n: np.ndarray, w: np.ndarray, q: np.ndarray, params: RoyEvoParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.maximum(np.asarray(n, dtype=float), 1.0e-12)
    w = np.maximum(np.asarray(w, dtype=float), 1.0e-12)
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    capacity = 1.0 / params.kappa - 1.0e-8
    total = n + w
    scale = np.minimum(1.0, capacity / np.maximum(total, 1.0e-12))
    return n * scale, w * scale, q


def mean_state_for_baseline(
    spec: ShapeSpec,
    stress: float,
    baseline_state: str,
    baseline: tuple[float, float, float],
) -> tuple[float, float, float]:
    if baseline_state == "compensation_branch_state":
        branch = find_branch_state_at_stress(PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, stress)
        if not bool(branch["branch_found"]):
            raise ValueError(f"no compensation branch for {spec.label} at {stress}")
        return float(branch["n_star"]), float(branch["w_star"]), float(branch["q_star"])
    if baseline_state == "basin_boundary_state":
        n0, w0, _q0 = baseline
        return n0, w0 * 0.1, 0.0
    raise ValueError(f"unknown baseline_state: {baseline_state}")


def make_initial_fields(
    mean_state: tuple[float, float, float],
    perturbation_type: str,
    seed: int,
    config: RoyEvoPDEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    n_mean, w_mean, q_mean = mean_state
    x, y, _dx, _dy = grid_2d_evo(config)
    xx, yy = np.meshgrid(x, y)
    n = np.full((config.n_y, config.n_x), n_mean, dtype=float)
    w = np.full_like(n, w_mean)
    q = np.full_like(n, q_mean)
    amplitude = 0.0
    q_amplitude = 0.0
    rng = np.random.default_rng(seed)
    if perturbation_type == "homogeneous_control":
        return n, w, q, amplitude, q_amplitude
    if perturbation_type == "local_defense_patch":
        q_amplitude = 0.2
        sigma = config.L_x / 10.0
        gaussian = np.exp(-((xx - 0.5 * config.L_x) ** 2 + (yy - 0.5 * config.L_y) ** 2) / (2.0 * sigma * sigma))
        q = q_mean + q_amplitude * gaussian
    elif perturbation_type == "random_heterogeneity":
        amplitude = 0.1
        q_amplitude = 0.1
        n = n_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        w = w_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        q = q_mean + q_amplitude * smooth_mean_zero_noise(n.shape, rng)
    elif perturbation_type == "basin_boundary_heterogeneity":
        amplitude = 0.25
        q_amplitude = 0.25
        n = n_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        w = w_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        q = q_mean + q_amplitude * smooth_mean_zero_noise(n.shape, rng)
    else:
        raise ValueError(f"unknown perturbation_type: {perturbation_type}")
    return (*enforce_physical(n, w, q, PARAMS), amplitude, q_amplitude)


def _record_series(series: dict[str, list[float]], time: float, n: np.ndarray, w: np.ndarray, q: np.ndarray, params: RoyEvoParams) -> None:
    series["time"].append(float(time))
    series["mean_n"].append(float(np.mean(n)))
    series["mean_w"].append(float(np.mean(w)))
    series["mean_q"].append(float(np.mean(q)))
    series["var_n"].append(float(np.var(n)))
    series["var_w"].append(float(np.var(w)))
    series["var_q"].append(float(np.var(q)))
    series["min_z"].append(float(np.min(free_space_evo(n, w, params))))


def simulate_pde_with_snapshots(
    *,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
    snapshot_times: np.ndarray,
) -> SnapshotPDEResult:
    n, w, q = (np.array(field, dtype=float, copy=True) for field in initial_state)
    _x, _y, dx, dy = grid_2d_evo(config)
    n_steps = int(math.ceil(config.T / config.dt))
    snapshot_steps = {int(round(float(t) / config.dt)): idx for idx, t in enumerate(snapshot_times)}
    n_snapshots = np.empty((len(snapshot_times), config.n_y, config.n_x), dtype=float)
    w_snapshots = np.empty_like(n_snapshots)
    q_snapshots = np.empty_like(n_snapshots)
    series: dict[str, list[float]] = {key: [] for key in ("time", "mean_n", "mean_w", "mean_q", "var_n", "var_w", "var_q", "min_z")}
    min_n = float(np.min(n))
    min_w = float(np.min(w))
    min_q = float(np.min(q))
    max_q = float(np.max(q))
    min_z = float(np.min(free_space_evo(n, w, params)))
    q_clip_max_violation = 0.0
    nonfinite_detected = False
    completed = True

    def capture(step: int) -> None:
        if step in snapshot_steps:
            idx = snapshot_steps[step]
            n_snapshots[idx] = n
            w_snapshots[idx] = w
            q_snapshots[idx] = q

    capture(0)
    _record_series(series, 0.0, n, w, q, params)
    for step in range(1, n_steps + 1):
        dn_reaction, dw_reaction, dq_reaction = generalized_reaction_part_arrays(
            n, w, q, params, gamma_r, gamma_a, gamma_b, stress
        )
        n = n + config.dt * (config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + dn_reaction)
        w = w + config.dt * (config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + dw_reaction)
        q_raw = q + config.dt * (config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + dq_reaction)
        violation = np.maximum(np.maximum(0.0, -q_raw), np.maximum(0.0, q_raw - 1.0))
        q_clip_max_violation = max(q_clip_max_violation, float(np.max(violation)))
        q = np.clip(q_raw, 0.0, 1.0)
        z = free_space_evo(n, w, params)
        min_n = min(min_n, float(np.min(n)))
        min_w = min(min_w, float(np.min(w)))
        min_q = min(min_q, float(np.min(q)))
        max_q = max(max_q, float(np.max(q)))
        min_z = min(min_z, float(np.min(z)))
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(w)) and np.all(np.isfinite(q)) and np.all(np.isfinite(z))):
            nonfinite_detected = True
            completed = False
            break
        if step % config.record_every == 0 or step == n_steps or step in snapshot_steps:
            _record_series(series, min(step * config.dt, config.T), n, w, q, params)
        capture(step)
    if n_steps not in snapshot_steps:
        n_snapshots[-1] = n
        w_snapshots[-1] = w
        q_snapshots[-1] = q
    return SnapshotPDEResult(
        times=np.array(series["time"], dtype=float),
        mean_n=np.array(series["mean_n"], dtype=float),
        mean_w=np.array(series["mean_w"], dtype=float),
        mean_q=np.array(series["mean_q"], dtype=float),
        var_n=np.array(series["var_n"], dtype=float),
        var_w=np.array(series["var_w"], dtype=float),
        var_q=np.array(series["var_q"], dtype=float),
        n_final=n,
        w_final=w,
        q_final=q,
        n_snapshots=n_snapshots,
        w_snapshots=w_snapshots,
        q_snapshots=q_snapshots,
        snapshot_times=np.array(snapshot_times, dtype=float),
        min_n=min_n,
        min_w=min_w,
        min_q=min_q,
        max_q=max_q,
        min_z=min_z,
        q_clip_max_violation=q_clip_max_violation,
        completed=completed,
        nonfinite_detected=nonfinite_detected,
    )


def pde_rhs_residual(
    result: SnapshotPDEResult,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> dict[str, float]:
    _x, _y, dx, dy = grid_2d_evo(config)
    dn_reaction, dw_reaction, dq_reaction = generalized_reaction_part_arrays(
        result.n_final, result.w_final, result.q_final, params, gamma_r, gamma_a, gamma_b, stress
    )
    dn_dt = config.D_n * laplacian_neumann_2d_evo(result.n_final, dx, dy) + dn_reaction
    dw_dt = config.D_w * laplacian_neumann_2d_evo(result.w_final, dx, dy) + dw_reaction
    dq_dt = config.D_q * laplacian_neumann_2d_evo(result.q_final, dx, dy) + dq_reaction
    rhs_norm = float(np.sqrt(np.mean(dn_dt**2) + np.mean(dw_dt**2) + np.mean(dq_dt**2)))
    state_norm = float(np.sqrt(np.mean(result.n_final**2) + np.mean(result.w_final**2) + np.mean(result.q_final**2)))
    return {"rhs_norm": rhs_norm, "state_norm": state_norm, "normalized_residual": rhs_norm / max(state_norm, 1.0e-12)}


def classify_pde_result(
    result: SnapshotPDEResult,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    gamma_r: float,
    gamma_a: float,
    gamma_b: float,
    stress: float,
) -> dict[str, Any]:
    mask = tail_mask(result.times)
    prev = previous_window_mask(result.times)
    tail_t = result.times[mask]
    tail_w = result.mean_w[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(result.times, result.mean_w, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_w = float(np.mean(result.mean_w[prev]))
    residual = pde_rhs_residual(result, params, config, gamma_r, gamma_a, gamma_b, stress)
    physical = (
        bool(result.completed)
        and not bool(result.nonfinite_detected)
        and result.min_n >= -1.0e-8
        and result.min_w >= -1.0e-8
        and result.min_q >= -1.0e-6
        and result.max_q <= 1.0 + 1.0e-6
        and result.min_z >= -1.0e-5
        and result.q_clip_max_violation <= 1.0e-4
    )
    persistent_without_slope = bool(physical and tail_mean_w > EPSILON and tail_min_w > 0.25 * EPSILON)
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    metrics: dict[str, Any] = {
        "physical": physical,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_mean_q": float(np.mean(result.mean_q[mask])),
        "previous_window_mean_w": previous_w,
        "latest_window_mean_w": tail_mean_w,
        "relative_change_between_last_windows": relative_change(previous_w, tail_mean_w),
        "persistent_without_slope_rule": persistent_without_slope,
        "persistent_with_slope_rule": persistent_with_slope,
        **residual,
    }
    metrics["classification"] = classify_asymptotic(metrics)
    metrics["basin_label"] = basin_label_from_classification(str(metrics["classification"]))
    return metrics


def save_field_archive(path: Path, spec: dict[str, Any], result: SnapshotPDEResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snapshot_times=result.snapshot_times,
        n_snapshots=result.n_snapshots,
        w_snapshots=result.w_snapshots,
        q_snapshots=result.q_snapshots,
        **{key: spec[key] for key in ("shape_label", "gamma_r", "gamma_a", "gamma_b", "stress", "baseline_state", "perturbation_type", "seed", "T")},
    )


def build_pde_specs(shape_specs: list[ShapeSpec]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec in shape_specs:
        branch = find_branch_state_at_stress(PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, PDE_STRESSES_FOCUSED[0])
        if not (bool(branch["branch_found"]) and bool(branch["locally_stable"])):
            continue
        for stress in PDE_STRESSES_FOCUSED:
            for baseline_state in ("compensation_branch_state", "basin_boundary_state"):
                specs.append(
                    {
                        "shape_label": spec.label,
                        "gamma_r": spec.gamma_r,
                        "gamma_a": spec.gamma_a,
                        "gamma_b": spec.gamma_b,
                        "stress": stress,
                        "baseline_state": baseline_state,
                        "perturbation_type": "homogeneous_control",
                        "seed": SEED,
                        "T": PDE_T_FOCUSED,
                    }
                )
                if baseline_state == "compensation_branch_state":
                    for perturbation_type in ("local_defense_patch", "random_heterogeneity"):
                        specs.append(
                            {
                                "shape_label": spec.label,
                                "gamma_r": spec.gamma_r,
                                "gamma_a": spec.gamma_a,
                                "gamma_b": spec.gamma_b,
                                "stress": stress,
                                "baseline_state": baseline_state,
                                "perturbation_type": perturbation_type,
                                "seed": SEED,
                                "T": PDE_T_FOCUSED,
                            }
                        )
                else:
                    specs.append(
                        {
                            "shape_label": spec.label,
                            "gamma_r": spec.gamma_r,
                            "gamma_a": spec.gamma_a,
                            "gamma_b": spec.gamma_b,
                            "stress": stress,
                            "baseline_state": baseline_state,
                            "perturbation_type": "basin_boundary_heterogeneity",
                            "seed": SEED,
                            "T": PDE_T_FOCUSED,
                        }
                    )
    return specs


def run_one_pde_spec(
    spec: dict[str, Any],
    baseline: tuple[float, float, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], SnapshotPDEResult]:
    shape = ShapeSpec(str(spec["shape_label"]), float(spec["gamma_r"]), float(spec["gamma_a"]), float(spec["gamma_b"]))
    config = pde_config(float(spec["T"]))
    snapshot_times = np.array([0.0, 0.25 * config.T, 0.5 * config.T, 0.75 * config.T, config.T], dtype=float)
    mean_state = mean_state_for_baseline(shape, float(spec["stress"]), str(spec["baseline_state"]), baseline)
    n0, w0, q0, _amplitude, _q_amplitude = make_initial_fields(mean_state, str(spec["perturbation_type"]), int(spec["seed"]), config)
    result = simulate_pde_with_snapshots(
        params=PARAMS,
        config=config,
        initial_state=(n0, w0, q0),
        gamma_r=shape.gamma_r,
        gamma_a=shape.gamma_a,
        gamma_b=shape.gamma_b,
        stress=float(spec["stress"]),
        snapshot_times=snapshot_times,
    )
    metrics = classify_pde_result(result, PARAMS, config, shape.gamma_r, shape.gamma_a, shape.gamma_b, float(spec["stress"]))
    cv_n = np.sqrt(result.var_n) / np.maximum(np.abs(result.mean_n), 1.0e-12)
    cv_w = np.sqrt(result.var_w) / np.maximum(np.abs(result.mean_w), 1.0e-12)
    cv_q = np.sqrt(result.var_q) / np.maximum(np.abs(result.mean_q), 1.0e-12)
    summary = {
        **spec,
        "classification": metrics["classification"],
        "basin_label": metrics["basin_label"],
        "homogeneous_control_basin_label": "",
        "basin_changed_relative_to_control": False,
        "tail_mean_w": metrics["tail_mean_w"],
        "tail_mean_q": metrics["tail_mean_q"],
        "relative_change_between_last_windows": metrics["relative_change_between_last_windows"],
        "normalized_residual": metrics["normalized_residual"],
        "initial_cv_n": coefficient_of_variation(n0),
        "initial_cv_w": coefficient_of_variation(w0),
        "initial_cv_q": coefficient_of_variation(q0),
        "final_cv_n": float(cv_n[-1]),
        "final_cv_w": float(cv_w[-1]),
        "final_cv_q": float(cv_q[-1]),
        "max_cv_n": float(np.max(cv_n)),
        "max_cv_w": float(np.max(cv_w)),
        "max_cv_q": float(np.max(cv_q)),
        "spatial_pattern_persisted": bool(max(float(cv_n[-1]), float(cv_w[-1]), float(cv_q[-1])) > FINAL_CV_THRESHOLD),
        "physical": metrics["physical"],
        "notes": "targeted_nonlinear_pde",
    }
    timeseries_rows = [
        {
            "shape_label": spec["shape_label"],
            "stress": spec["stress"],
            "baseline_state": spec["baseline_state"],
            "perturbation_type": spec["perturbation_type"],
            "seed": spec["seed"],
            "T": spec["T"],
            "time": float(time),
            "mean_n": float(result.mean_n[idx]),
            "mean_w": float(result.mean_w[idx]),
            "mean_q": float(result.mean_q[idx]),
        }
        for idx, time in enumerate(result.times)
    ]
    spatial_rows = [
        {
            "shape_label": spec["shape_label"],
            "stress": spec["stress"],
            "baseline_state": spec["baseline_state"],
            "perturbation_type": spec["perturbation_type"],
            "seed": spec["seed"],
            "T": spec["T"],
            "time": float(time),
            "var_n": float(result.var_n[idx]),
            "var_w": float(result.var_w[idx]),
            "var_q": float(result.var_q[idx]),
            "cv_n": float(cv_n[idx]),
            "cv_w": float(cv_w[idx]),
            "cv_q": float(cv_q[idx]),
        }
        for idx, time in enumerate(result.times)
    ]
    return summary, timeseries_rows, spatial_rows, result


def run_pde_nonhomogeneous_tests(shape_specs: list[ShapeSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if PDE_NONHOM_SUMMARY_CSV.exists() and PDE_NONHOM_TIMESERIES_CSV.exists() and PDE_NONHOM_SPATIAL_CSV.exists():
        print("Using existing targeted nonlinear PDE non-homogeneous outputs.")
        return read_csv(PDE_NONHOM_SUMMARY_CSV), read_csv(PDE_NONHOM_TIMESERIES_CSV), read_csv(PDE_NONHOM_SPATIAL_CSV)

    baseline = baseline_burnin_state()
    specs = build_pde_specs(shape_specs)
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    control_basin: dict[tuple[str, float, str, float], str] = {}
    saved_field_keys: set[tuple[str, str]] = set()

    idx = 0
    while idx < len(specs):
        spec = specs[idx]
        idx += 1
        print(
            "Running nonlinear PDE "
            f"{idx}/{len(specs)}: {spec['shape_label']} {spec['baseline_state']} {spec['perturbation_type']} T={spec['T']}"
        )
        summary, t_rows, s_rows, result = run_one_pde_spec(spec, baseline)
        control_key = (
            str(summary["shape_label"]),
            float(summary["stress"]),
            str(summary["baseline_state"]),
            float(summary["T"]),
        )
        if summary["perturbation_type"] == "homogeneous_control":
            control_basin[control_key] = str(summary["basin_label"])
            summary["homogeneous_control_basin_label"] = str(summary["basin_label"])
        else:
            control = control_basin.get(control_key, "")
            summary["homogeneous_control_basin_label"] = control
            summary["basin_changed_relative_to_control"] = bool(control and str(summary["basin_label"]) != control)
            if (summary["basin_changed_relative_to_control"] or "transient" in str(summary["classification"])) and float(summary["T"]) < PDE_T_LONG:
                long_spec = dict(spec)
                long_spec["T"] = PDE_T_LONG
                control_long = dict(spec)
                control_long["perturbation_type"] = "homogeneous_control"
                control_long["T"] = PDE_T_LONG
                specs.append(control_long)
                specs.append(long_spec)
        summary_rows.append(summary)
        timeseries_rows.extend(t_rows)
        spatial_rows.extend(s_rows)
        field_key = (str(summary["shape_label"]), str(summary["perturbation_type"]))
        if summary["perturbation_type"] != "homogeneous_control" and field_key not in saved_field_keys:
            field_path = RESULTS_DIR / f"roy_nonlinear_tradeoff_pde_fields_{safe_filename(summary['shape_label'])}_{safe_filename(summary['perturbation_type'])}.npz"
            save_field_archive(field_path, summary, result)
            saved_field_keys.add(field_key)

    for row in summary_rows:
        if row["perturbation_type"] != "homogeneous_control" and not row["homogeneous_control_basin_label"]:
            control_key = (
                str(row["shape_label"]),
                float(row["stress"]),
                str(row["baseline_state"]),
                float(row["T"]),
            )
            control = control_basin.get(control_key, "")
            row["homogeneous_control_basin_label"] = control
            row["basin_changed_relative_to_control"] = bool(control and str(row["basin_label"]) != control)

    write_csv(PDE_NONHOM_SUMMARY_CSV, summary_rows, PDE_NONHOM_FIELDS)
    write_csv(PDE_NONHOM_TIMESERIES_CSV, timeseries_rows, PDE_TIMESERIES_FIELDS)
    write_csv(PDE_NONHOM_SPATIAL_CSV, spatial_rows, PDE_SPATIAL_METRIC_FIELDS)
    return summary_rows, timeseries_rows, spatial_rows


def terminal_heterogeneous_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal: dict[tuple[str, float, str, str, int], dict[str, Any]] = {}
    for row in rows:
        if str(row["perturbation_type"]) == "homogeneous_control":
            continue
        key = (
            str(row["shape_label"]),
            float(row["stress"]),
            str(row["baseline_state"]),
            str(row["perturbation_type"]),
            int(row["seed"]),
        )
        if key not in terminal or float(row["T"]) > float(terminal[key]["T"]):
            terminal[key] = row
    return list(terminal.values())


def decide_final_label(
    *,
    linear_branch_recovered: bool,
    has_concave_stable: bool,
    has_convex_stable: bool,
    has_mixed_stable: bool = True,
    pde_spatial_instability_count: int,
    persistent_pattern_count: int,
) -> str:
    if not linear_branch_recovered:
        return "nonlinear_tradeoff_compensation_unresolved"
    if (
        has_concave_stable
        and has_convex_stable
        and has_mixed_stable
        and pde_spatial_instability_count == 0
        and persistent_pattern_count == 0
    ):
        return "nonlinear_tradeoff_compensation_supported"
    return "nonlinear_tradeoff_compensation_parameter_sensitive"


def write_decision_summary(
    linear_recovered: bool,
    max_delta_q: float,
    shape_summary_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    pde_spatial_rows: list[dict[str, Any]],
    pde_nonhom_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    class_counts = Counter(str(row["shape_class"]) for row in shape_summary_rows)
    selected_spatial_completed = sum(str(row["notes"]) == "nonzero_neumann_modes_tested" for row in pde_spatial_rows)
    pde_instability_count = sum(
        str(row["notes"]) == "nonzero_neumann_modes_tested" and not as_bool(row["linearly_spatially_stable"])
        for row in pde_spatial_rows
    )
    nonhom_total = len(pde_nonhom_rows)
    terminal_rows = terminal_heterogeneous_rows(pde_nonhom_rows)
    finite_horizon_basin_changes = sum(as_bool(row["basin_changed_relative_to_control"]) for row in pde_nonhom_rows)
    finite_horizon_patterns = sum(as_bool(row["spatial_pattern_persisted"]) for row in pde_nonhom_rows)
    basin_change_count = sum(as_bool(row["basin_changed_relative_to_control"]) for row in terminal_rows)
    pattern_count = sum(as_bool(row["spatial_pattern_persisted"]) for row in terminal_rows)
    concave_stable = any(
        float(row["gamma_r"]) == 0.5
        and float(row["gamma_a"]) == 0.5
        and float(row["gamma_b"]) == 0.5
        and int(row["target_stresses_stable"]) > 0
        for row in shape_summary_rows
    )
    convex_stable = any(
        float(row["gamma_r"]) == 2.0
        and float(row["gamma_a"]) == 2.0
        and float(row["gamma_b"]) == 2.0
        and int(row["target_stresses_stable"]) > 0
        for row in shape_summary_rows
    )
    mixed_stable = any(
        len({float(row["gamma_r"]), float(row["gamma_a"]), float(row["gamma_b"])}) > 1
        and int(row["target_stresses_stable"]) > 0
        for row in shape_summary_rows
    )
    final_label = decide_final_label(
        linear_branch_recovered=linear_recovered,
        has_concave_stable=concave_stable,
        has_convex_stable=convex_stable,
        has_mixed_stable=mixed_stable,
        pde_spatial_instability_count=pde_instability_count,
        persistent_pattern_count=pattern_count,
    )
    rows = [
        {"metric": "linear_branch_recovered", "value": linear_recovered, "interpretation": "generalized nonlinear formulation recovers gamma=1 branch"},
        {"metric": "max_linear_recovery_abs_delta_q", "value": max_delta_q, "interpretation": "maximum q difference versus closed-form linear branch"},
        {"metric": "shape_grid_total", "value": len(shape_summary_rows), "interpretation": "controlled gamma_r,gamma_a,gamma_b shape combinations"},
        {"metric": "shape_grid_robust_count", "value": class_counts["robust_compensation_shape"], "interpretation": "shape combinations stable at all target stresses"},
        {"metric": "shape_grid_partial_count", "value": class_counts["partial_compensation_shape"], "interpretation": "shape combinations stable at some target stresses"},
        {"metric": "shape_grid_no_compensation_count", "value": class_counts["no_compensation_shape"], "interpretation": "shape combinations with no feasible target branch"},
        {"metric": "selected_ode_basin_maps_completed", "value": len(basin_rows) > 0, "interpretation": "selected nonlinear ODE q0-w0 basin maps generated"},
        {"metric": "selected_pde_spatial_stability_completed", "value": selected_spatial_completed, "interpretation": "selected stable branch stress rows tested for PDE spatial modes"},
        {"metric": "selected_pde_spatial_instability_count", "value": pde_instability_count, "interpretation": "selected nonlinear branch rows with positive nonzero spatial mode"},
        {"metric": "selected_nonhomogeneous_runs_total", "value": nonhom_total, "interpretation": "targeted nonlinear PDE nonhomogeneous runs including controls"},
        {"metric": "selected_nonhomogeneous_finite_horizon_basin_change_count", "value": finite_horizon_basin_changes, "interpretation": "all recorded heterogeneous rows changing basin label before terminal-horizon filtering"},
        {"metric": "selected_nonhomogeneous_finite_horizon_persistent_pattern_count", "value": finite_horizon_patterns, "interpretation": "all recorded heterogeneous rows with final spatial CV above threshold before terminal-horizon filtering"},
        {"metric": "selected_nonhomogeneous_basin_change_count", "value": basin_change_count, "interpretation": "terminal-horizon heterogeneous cases changing basin label relative to matched control"},
        {"metric": "selected_nonhomogeneous_persistent_pattern_count", "value": pattern_count, "interpretation": "terminal-horizon heterogeneous cases with final spatial CV above threshold"},
        {"metric": "final_label", "value": final_label, "interpretation": "allowed nonlinear trade-off extension label"},
    ]
    write_csv(DECISION_CSV, rows, ("metric", "value", "interpretation"))
    return rows, final_label


def plot_branch_curves(shape_specs: list[ShapeSpec]) -> None:
    q_grid = np.linspace(Q_EPS, 1.0 - Q_EPS, 1200)
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    for spec in shape_specs:
        curve = generalized_branch_curve(PARAMS, spec.gamma_r, spec.gamma_a, spec.gamma_b, q_grid)
        feasible = [row for row in curve if bool(row["feasible"]) and np.isfinite(float(row["s"]))]
        if not feasible:
            continue
        ax.plot([float(row["q"]) for row in feasible], [float(row["s"]) for row in feasible], linewidth=1.8, label=spec.label.replace("_", " "))
    for stress in TARGET_STRESSES:
        ax.axhline(stress, color="#777777", linewidth=0.8, linestyle="--", alpha=0.65)
    ax.set_xlabel("defense frequency q")
    ax.set_ylabel("branch stress s(q)")
    ax.set_title("Nonlinear compensation branch curves")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, FIG59_PATH)


def plot_shape_grid(shape_summary_rows: list[dict[str, Any]]) -> None:
    class_to_value = {
        "no_compensation_shape": 0,
        "unresolved_shape": 1,
        "partial_compensation_shape": 2,
        "robust_compensation_shape": 3,
    }
    gammas = list(GAMMA_VALUES)
    fig, axes = plt.subplots(1, len(gammas), figsize=(10.0, 4.3), constrained_layout=False)
    cmap = ListedColormap(["#bdbdbd", "#d9a441", "#8fb9dd", "#2f6fbb"])
    for ax, gamma_b in zip(axes, gammas):
        matrix = np.zeros((len(gammas), len(gammas)), dtype=float)
        for row in shape_summary_rows:
            if float(row["gamma_b"]) != gamma_b:
                continue
            y = gammas.index(float(row["gamma_a"]))
            x = gammas.index(float(row["gamma_r"]))
            matrix[y, x] = class_to_value[str(row["shape_class"])]
        ax.imshow(matrix, origin="lower", cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks(range(len(gammas)), [str(value) for value in gammas])
        ax.set_yticks(range(len(gammas)), [str(value) for value in gammas])
        ax.set_xlabel("gamma_r")
        ax.set_title(f"gamma_b={gamma_b:g}")
    axes[0].set_ylabel("gamma_a")
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=color, markeredgecolor=color, label=label.replace("_", " "))
        for label, color in zip(
            ("no_compensation_shape", "unresolved_shape", "partial_compensation_shape", "robust_compensation_shape"),
            ("#bdbdbd", "#d9a441", "#8fb9dd", "#2f6fbb"),
        )
    ]
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.24, wspace=0.35)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Nonlinear shape-grid compensation class", y=0.98)
    save_figure(fig, FIG60_PATH)


def plot_basin_maps(basin_rows: list[dict[str, Any]], shape_specs: list[ShapeSpec]) -> None:
    selected_labels = [spec.label for spec in shape_specs]
    category_index = {label: idx for idx, label in enumerate(BASIN_LABELS)}
    cmap = ListedColormap([BASIN_COLORS[label] for label in BASIN_LABELS])
    fig, axes = plt.subplots(len(selected_labels), len(BASIN_STRESSES), figsize=(8.2, 2.25 * len(selected_labels)), constrained_layout=False)
    if len(selected_labels) == 1:
        axes = np.array([axes])
    for row_idx, label in enumerate(selected_labels):
        for col_idx, stress in enumerate(BASIN_STRESSES):
            ax = axes[row_idx, col_idx]
            subset = [row for row in basin_rows if str(row["shape_label"]) == label and math.isclose(float(row["stress"]), stress)]
            matrix = np.full((len(W0_SCALES), len(Q0_VALUES)), category_index["unresolved_basin"])
            for row in subset:
                x = Q0_VALUES.index(round(float(row["q0"]), 1))
                y = W0_SCALES.index(float(row["w0_scale"]))
                matrix[y, x] = category_index.get(str(row["basin_label"]), category_index["unresolved_basin"])
            ax.imshow(matrix, origin="lower", cmap=cmap, vmin=0, vmax=len(BASIN_LABELS) - 1, aspect="auto")
            ax.set_xticks(range(len(Q0_VALUES)), [str(value) for value in Q0_VALUES], fontsize=6)
            ax.set_yticks(range(len(W0_SCALES)), [str(value) for value in W0_SCALES], fontsize=6)
            if row_idx == len(selected_labels) - 1:
                ax.set_xlabel("q0")
            if col_idx == 0:
                ax.set_ylabel(label.replace("_", "\n") + "\nw0 scale", fontsize=7)
            ax.set_title(f"s={stress:g}", fontsize=8)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=BASIN_COLORS[label], markeredgecolor=BASIN_COLORS[label], label=label.replace("_", " "))
        for label in BASIN_LABELS
    ]
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.075, hspace=0.45, wspace=0.18)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, fontsize=7, frameon=False)
    fig.suptitle("Selected nonlinear ODE q0-w0 basin maps", y=0.992)
    save_figure(fig, FIG61_PATH)


def plot_pde_spatial(pde_rows: list[dict[str, Any]]) -> None:
    subset = [row for row in pde_rows if str(row["notes"]) == "nonzero_neumann_modes_tested"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    labels = [f"{row['shape_label']}\ns={float(row['stress']):.3g}" for row in subset]
    values = [float(row["max_nonzero_mode_growth"]) for row in subset]
    ax.bar(range(len(values)), values, color="#5b8cc0")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(range(len(values)), labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("max nonzero spatial growth")
    ax.set_title("PDE spatial-mode stability for selected nonlinear branches")
    save_figure(fig, FIG62_PATH)


def plot_nonhomogeneous_summary(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    hetero = [row for row in rows if str(row["perturbation_type"]) != "homogeneous_control"]
    labels = ["basin changes", "persistent patterns"]
    values = [
        sum(as_bool(row["basin_changed_relative_to_control"]) for row in hetero),
        sum(as_bool(row["spatial_pattern_persisted"]) for row in hetero),
    ]
    axes[0].bar(labels, values, color=["#d9a441", "#c23b3b"])
    axes[0].set_title("Targeted nonlinear PDE outcomes")
    axes[0].set_ylabel("run count")
    cv_values = [max(float(row["final_cv_n"]), float(row["final_cv_w"]), float(row["final_cv_q"])) for row in hetero]
    axes[1].scatter(range(len(cv_values)), cv_values, color="#2f6fbb", s=24)
    axes[1].axhline(FINAL_CV_THRESHOLD, color="#c23b3b", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title("Final spatial CV")
    axes[1].set_xlabel("heterogeneous run")
    save_figure(fig, FIG63_PATH)


def plot_final_decision(decision_rows: list[dict[str, Any]]) -> None:
    data = {str(row["metric"]): row["value"] for row in decision_rows}
    metrics = [
        ("robust shapes", float(data["shape_grid_robust_count"])),
        ("partial shapes", float(data["shape_grid_partial_count"])),
        ("PDE instabilities", float(data["selected_pde_spatial_instability_count"])),
        ("persistent patterns", float(data["selected_nonhomogeneous_persistent_pattern_count"])),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.bar([item[0] for item in metrics], [item[1] for item in metrics], color=["#2f6fbb", "#8fb9dd", "#c23b3b", "#c23b3b"])
    ax.set_ylabel("count")
    ax.set_title(f"Nonlinear trade-off decision: {data['final_label']}")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, FIG64_PATH)


def write_research_note(
    decision_rows: list[dict[str, Any]],
    final_label: str,
    shape_summary_rows: list[dict[str, Any]],
    pde_rows: list[dict[str, Any]],
    nonhom_rows: list[dict[str, Any]],
) -> None:
    counts = Counter(str(row["shape_class"]) for row in shape_summary_rows)
    stable_spatial = sum(str(row["notes"]) == "nonzero_neumann_modes_tested" and as_bool(row["linearly_spatially_stable"]) for row in pde_rows)
    spatial_total = sum(str(row["notes"]) == "nonzero_neumann_modes_tested" for row in pde_rows)
    terminal_rows = terminal_heterogeneous_rows(nonhom_rows)
    finite_basin_changes = sum(as_bool(row["basin_changed_relative_to_control"]) for row in nonhom_rows)
    finite_patterns = sum(as_bool(row["spatial_pattern_persisted"]) for row in nonhom_rows)
    basin_changes = sum(as_bool(row["basin_changed_relative_to_control"]) for row in terminal_rows)
    patterns = sum(as_bool(row["spatial_pattern_persisted"]) for row in terminal_rows)
    if final_label == "nonlinear_tradeoff_compensation_supported":
        interpretation = "The controlled nonlinear extension supports compensation branches beyond the linear case, with targeted PDE tests showing no persistent spatial-pattern-mediated rescue."
    elif final_label == "nonlinear_tradeoff_compensation_parameter_sensitive":
        interpretation = "The generalized branch recovers the linear result, but nonlinear shape changes make compensation branch existence or stability parameter-sensitive."
    else:
        interpretation = "The nonlinear extension did not produce a consistent enough branch or stability diagnosis for a mechanism decision."
    text = f"""# Nonlinear Trade-Off Extension of the Homogeneous Compensation Mechanism

## Purpose

This note tests whether the homogeneous compensation mechanism remains visible under controlled endpoint-preserving nonlinear trade-off forms. It extends the previous linear-trade-off derivation without changing the baseline model implementation.

## Generalized Trade-Off Model

The nonlinear forms are

```text
r(q) = r_u + (r_v-r_u) q^gamma_r
a(q) = a_u + (a_v-a_u) q^gamma_a
b(q) = b_u + (b_v-b_u) q^gamma_b
```

The controlled shape grid uses gamma values `0.5`, `1.0`, and `2.0`, representing concave, linear, and convex endpoint-preserving trade-offs. This is a local structured shape check, not a broad random scan.

## Generalized Compensation Branch

For nonlinear trade-offs, the branch is naturally parameterized by `q`. The selection-gradient condition gives

```text
c(q) = r'(q) / a'(q)
```

when `a'(q)` is nonzero. Then

```text
z(q) = xi / [r(q) - a(q)c(q)]
w(q) = c(q)z(q)
n(q) = kappa^-1 - z(q) - w(q)
s(q) = b(q)n(q)z(q) - m - mu w(q)
```

A target stress has an interior branch state when a feasible `q` satisfies `s(q)=s0`.

## Recovery of the Linear Case

The gamma `(1,1,1)` case recovers the previously derived linear branch. The maximum absolute recovery difference in `q*` is `{read_summary_metric(DECISION_CSV, 'max_linear_recovery_abs_delta_q', 'not available')}`.

## Shape-Grid Branch and Stability Results

The 27-combination shape grid produced:

- robust compensation shapes: `{counts['robust_compensation_shape']}`
- partial compensation shapes: `{counts['partial_compensation_shape']}`
- no-compensation shapes: `{counts['no_compensation_shape']}`
- unresolved shapes: `{counts['unresolved_shape']}`

Concave, convex, and mixed shape choices can change the stress interval, branch feasibility, and local stability. This should be interpreted as local parameter sensitivity rather than a global conclusion about all nonlinear trade-offs.

## Selected ODE Basin Maps

Selected ODE q0-w0 basin maps were generated for the linear reference, all-concave, all-convex, one mixed case, and a weak or failed shape. These maps test whether basin-dependent outcomes remain visible in homogeneous dynamics under selected nonlinear shape choices.

## PDE Spatial Stability for Selected Shapes

PDE spatial-mode stability was evaluated using `J_F(U*) - lambda_mn D` for selected nonlinear branch states. Stable selected branch rows: `{stable_spatial}` of `{spatial_total}`. This remains a linear spatial-mode test around homogeneous branch states.

## Non-Homogeneous PDE Perturbation Tests

Targeted nonlinear PDE tests used homogeneous controls, local defense patches, random heterogeneity, and basin-boundary heterogeneity. Finite-horizon rows with basin changes relative to controls: `{finite_basin_changes}`. Finite-horizon rows with final spatial CV above threshold: `{finite_patterns}`. After using the longest available horizon for each heterogeneous case, terminal basin changes relative to controls: `{basin_changes}`. Terminal persistent spatial-pattern rows above threshold: `{patterns}`.

## Final Label

`{final_label}`

## Interpretation

{interpretation}

## Biological Meaning

The compensation interpretation is that evolving defense can reduce predator resistance as mortality stress increases, thereby increasing conversion opportunity enough to preserve a positive predator equilibrium when the branch exists and is stable. Nonlinear trade-off shape changes can alter whether this compensation path is feasible.

## What Is Supported

- The generalized nonlinear formulation recovers the linear compensation branch when all shape exponents equal 1.
- The `q`-parameterized branch gives a reproducible way to test nonlinear trade-off shapes.
- Selected nonlinear shape cases can support feasible locally stable compensation branches.
- Targeted PDE checks do not by themselves show persistent spatial-pattern-mediated rescue for the selected stable branch cases.

## What Is Not Supported

- A global theorem over all nonlinear trade-off forms.
- A biological calibration of concave or convex shape exponents.
- A broad PDE parameter scan.
- A claim that spatial patterning is the rescue mechanism.

## Remaining Caveats

- The shape grid is small and structured.
- PDE tests are targeted and selected from branch diagnostics.
- Finite-difference Jacobians are numerical local-stability evidence, not symbolic proof.
- Nonlinear trade-off shapes near endpoints require safe derivative clipping.

## Files

- `experiments/27_roy_nonlinear_tradeoff_compensation.py`
- `results/roy_nonlinear_tradeoff_branch_recovery_linear.csv`
- `results/roy_nonlinear_tradeoff_branch_stability_grid.csv`
- `results/roy_nonlinear_tradeoff_shape_summary.csv`
- `results/roy_nonlinear_tradeoff_selected_ode_basin_maps.csv`
- `results/roy_nonlinear_tradeoff_pde_spatial_stability.csv`
- `results/roy_nonlinear_tradeoff_pde_nonhomogeneous_summary.csv`
- `results/roy_nonlinear_tradeoff_compensation_decision.csv`
- `figures/roy_evo_spatial/report/fig59_nonlinear_branch_curves.png`
- `figures/roy_evo_spatial/report/fig60_nonlinear_shape_grid_summary.png`
- `figures/roy_evo_spatial/report/fig61_nonlinear_selected_ode_basin_maps.png`
- `figures/roy_evo_spatial/report/fig62_nonlinear_pde_spatial_stability.png`
- `figures/roy_evo_spatial/report/fig63_nonlinear_nonhomogeneous_pde_tests.png`
- `figures/roy_evo_spatial/report/fig64_nonlinear_tradeoff_final_decision.png`

## Next Step

Use the nonlinear branch diagnostics to identify analytically interpretable shape regimes before considering any broader PDE work.

{final_label}
"""
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(text, encoding="utf-8")


def run(profile: str) -> None:
    q_grid_size = 4001 if profile == "focused" else 1001
    del q_grid_size
    recovery_rows, linear_recovered, max_delta_q = write_linear_recovery()
    del recovery_rows
    branch_rows, shape_summary_rows = run_branch_stability_grid()
    selected_shapes = select_shape_specs(shape_summary_rows)
    basin_rows = run_selected_ode_basin_maps(selected_shapes, profile)
    pde_spatial_rows = run_pde_spatial_stability(selected_shapes, mode_max=32 if profile == "focused" else 8)
    pde_nonhom_rows, pde_timeseries_rows, pde_spatial_metric_rows = run_pde_nonhomogeneous_tests(selected_shapes)
    del branch_rows, pde_timeseries_rows, pde_spatial_metric_rows
    decision_rows, final_label = write_decision_summary(
        linear_recovered,
        max_delta_q,
        shape_summary_rows,
        basin_rows,
        pde_spatial_rows,
        pde_nonhom_rows,
    )
    plot_branch_curves(selected_shapes)
    plot_shape_grid(shape_summary_rows)
    plot_basin_maps(basin_rows, selected_shapes)
    plot_pde_spatial(pde_spatial_rows)
    plot_nonhomogeneous_summary(pde_nonhom_rows)
    plot_final_decision(decision_rows)
    write_research_note(decision_rows, final_label, shape_summary_rows, pde_spatial_rows, pde_nonhom_rows)
    print(f"Final label: {final_label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("minimal", "focused"), default="focused")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
