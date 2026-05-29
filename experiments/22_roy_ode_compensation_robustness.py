#!/usr/bin/env python
"""Test the homogeneous ODE compensation branch and local robustness.

This experiment is ODE-only. It derives the interior compensation branch from
the homogeneous equilibrium equations, compares it with existing numerical
equilibria, and performs a small structured trade-off robustness check.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (  # noqa: E402
    RoyEvoParams,
    a_of_q,
    b_of_q,
    free_space_evo,
    predator_growth_factor_evo,
    r_of_q,
    reaction_ode_evo,
    selection_gradient,
    simulate_ode_evo,
)


PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
TAIL_FRACTION = 0.25
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 1.0e-4
EXTREME_EXTINCTION_W = 1.0e-8

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTES_DIR = ROOT / "research_notes"

NUMERICAL_EQUILIBRIA_CSV = RESULTS_DIR / "roy_ode_homogeneous_equilibria.csv"
BRANCH_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_branch_current.csv"
INTERVAL_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_interval_current.csv"
TRADEOFF_GRID_CSV = RESULTS_DIR / "roy_ode_compensation_tradeoff_grid.csv"
SELECTED_BASIN_MAPS_CSV = RESULTS_DIR / "roy_ode_compensation_selected_basin_maps.csv"
SELECTED_EQUILIBRIA_CSV = RESULTS_DIR / "roy_ode_compensation_selected_equilibria.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_ode_compensation_robustness_summary.csv"
NOTE_PATH = NOTES_DIR / "roy_ode_compensation_robustness.md"

FIG33_PATH = FIG_DIR / "fig33_compensation_branch_current.png"
FIG34_PATH = FIG_DIR / "fig34_compensation_interval_tradeoff_grid.png"
FIG35_PATH = FIG_DIR / "fig35_selected_tradeoff_basin_maps.png"
FIG36_PATH = FIG_DIR / "fig36_selected_equilibria_stability.png"
FIG37_PATH = FIG_DIR / "fig37_compensation_mechanism_diagram.png"

TARGET_STRESSES = (0.0, 0.069448242, 0.11765625, 0.1584375, 0.16486816, 0.175)
BASIN_STRESSES = (0.1584375, 0.16486816)
SELECTED_EQUILIBRIUM_STRESSES = (0.0, 0.11765625, 0.1584375, 0.16486816, 0.175)
Q0_VALUES = tuple(round(0.1 * idx, 1) for idx in range(10))
W0_SCALES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5)

R_V_VALUES = (0.55, 0.65, 0.75)
A_V_VALUES = (0.25, 0.35, 0.45)
B_U_VALUES = (0.06, 0.08, 0.10)
B_RATIO_VALUES = (0.25, 0.50, 0.75)

BRANCH_CURRENT_FIELDS = [
    "stress",
    "c",
    "z_star_analytic",
    "w_star_analytic",
    "n_star_analytic",
    "q_star_analytic",
    "b_req",
    "interior_exists",
    "existence_failure_reason",
    "numerical_equilibrium_id",
    "n_star_numerical",
    "w_star_numerical",
    "q_star_numerical",
    "z_star_numerical",
    "abs_delta_n",
    "abs_delta_w",
    "abs_delta_q",
    "abs_delta_z",
]
INTERVAL_FIELDS = ["s_at_q_equals_1", "s_at_q_equals_0", "interior_stress_interval_length", "interpretation"]
TRADEOFF_FIELDS = [
    "r_v",
    "a_v",
    "b_u",
    "b_v",
    "b_v_over_b_u",
    "c",
    "valid_selection_ratio",
    "z_star",
    "w_star",
    "n_star",
    "s_at_q_equals_1",
    "s_at_q_equals_0",
    "interior_interval_length",
    "nonnegative_interval_length",
    "branch_exists_at_s_0",
    "branch_exists_at_s_0p1584375",
    "branch_exists_at_s_0p16486816",
    "branch_exists_at_s_0p175",
    "qualitative_class",
    "notes",
]
SELECTED_BASIN_FIELDS = [
    "parameter_set_label",
    "r_v",
    "a_v",
    "b_u",
    "b_v",
    "stress",
    "q0",
    "w0_scale",
    "classification",
    "basin_label",
    "tail_mean_w",
    "tail_mean_q",
    "q_change",
    "selection_gradient_tail_mean",
    "predator_growth_factor_tail_mean",
    "physical",
    "notes",
]
SELECTED_EQUILIBRIA_FIELDS = [
    "parameter_set_label",
    "stress",
    "equilibrium_id",
    "n_star",
    "w_star",
    "q_star",
    "z_star",
    "selection_gradient",
    "predator_growth_factor",
    "max_real_eigenvalue",
    "stability_label",
    "physical",
    "notes",
]
SUMMARY_FIELDS = ["metric", "value", "interpretation"]
BASIN_LABELS = ("persistent_basin", "extinct_basin", "transient_basin", "unresolved_basin")
BASIN_COLORS = {
    "persistent_basin": "#2f6fbb",
    "extinct_basin": "#c23b3b",
    "transient_basin": "#d9a441",
    "unresolved_basin": "#7f7f7f",
}
STABILITY_COLORS = {
    "stable": "#1b9e77",
    "unstable": "#d95f02",
    "marginal_or_unresolved": "#7570b3",
    "nonphysical": "#777777",
}


@dataclass(frozen=True)
class BranchPoint:
    stress: float
    c: float
    z_star: float
    w_star: float
    n_star: float
    q_star: float
    b_req: float
    interior_exists: bool
    existence_failure_reason: str


@dataclass(frozen=True)
class Interval:
    s_at_q_equals_1: float
    s_at_q_equals_0: float
    interior_stress_interval_length: float
    nonnegative_interval_length: float
    valid: bool
    failure_reason: str


@dataclass(frozen=True)
class ParameterSet:
    label: str
    params: RoyEvoParams
    selection_note: str


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def format_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def branch_geometry(params: RoyEvoParams) -> tuple[float, float, str]:
    delta_r = params.r_v - params.r_u
    delta_a = params.a_v - params.a_u
    if abs(delta_a) < 1.0e-14:
        return math.nan, math.nan, "delta_a_zero"
    c = delta_r / delta_a
    if not np.isfinite(c) or c <= 0.0:
        return c, math.nan, "invalid_selection_ratio"
    denominator = params.r_u - c * params.a_u
    if denominator <= 0.0:
        return c, denominator, "nonpositive_prey_balance_denominator"
    return c, denominator, ""


def analytic_compensation_branch(params: RoyEvoParams, stress: float) -> BranchPoint:
    c, denominator, failure = branch_geometry(params)
    if failure:
        return BranchPoint(stress, c, math.nan, math.nan, math.nan, math.nan, math.nan, False, failure)
    z_star = params.xi / denominator
    w_star = c * z_star
    n_star = 1.0 / params.kappa - z_star - w_star
    if z_star <= 0.0 or w_star <= 0.0 or n_star <= 0.0:
        return BranchPoint(stress, c, z_star, w_star, n_star, math.nan, math.nan, False, "nonpositive_n_w_or_z")
    denom = n_star * z_star
    if denom <= 0.0:
        return BranchPoint(stress, c, z_star, w_star, n_star, math.nan, math.nan, False, "nonpositive_predator_balance_denominator")
    b_req = (params.m + stress + params.mu * w_star) / denom
    b_span = params.b_v - params.b_u
    if abs(b_span) < 1.0e-14:
        return BranchPoint(stress, c, z_star, w_star, n_star, math.nan, b_req, False, "conversion_tradeoff_flat")
    q_star = (b_req - params.b_u) / b_span
    interior = bool(0.0 < q_star < 1.0)
    reason = "" if interior else "q_star_outside_interior_interval"
    return BranchPoint(stress, c, z_star, w_star, n_star, q_star, b_req, interior, reason)


def branch_exists_at(params: RoyEvoParams, stress: float) -> bool:
    return analytic_compensation_branch(params, stress).interior_exists


def compensation_interval(params: RoyEvoParams) -> Interval:
    point = analytic_compensation_branch(params, 0.0)
    if not np.isfinite(point.n_star) or not np.isfinite(point.z_star) or not np.isfinite(point.w_star):
        return Interval(math.nan, math.nan, 0.0, 0.0, False, point.existence_failure_reason)
    denom = point.n_star * point.z_star
    if denom <= 0.0:
        return Interval(math.nan, math.nan, 0.0, 0.0, False, "nonpositive_predator_balance_denominator")
    if abs(params.b_v - params.b_u) < 1.0e-14:
        return Interval(math.nan, math.nan, 0.0, 0.0, False, "conversion_tradeoff_flat")
    s_at_q_equals_1 = params.b_v * denom - params.m - params.mu * point.w_star
    s_at_q_equals_0 = params.b_u * denom - params.m - params.mu * point.w_star
    low = min(s_at_q_equals_1, s_at_q_equals_0)
    high = max(s_at_q_equals_1, s_at_q_equals_0)
    interval_length = high - low
    nonnegative_length = max(0.0, high - max(low, 0.0))
    valid = bool(interval_length > 0.0 and np.isfinite(interval_length))
    return Interval(s_at_q_equals_1, s_at_q_equals_0, interval_length, nonnegative_length, valid, "" if valid else "zero_or_invalid_interval")


def qualitative_tradeoff_class(params: RoyEvoParams, interval: Interval) -> str:
    c, _denominator, failure = branch_geometry(params)
    if failure or not np.isfinite(c) or c <= 0.0:
        return "invalid_tradeoff_geometry"
    if interval.valid and interval.nonnegative_interval_length > 0.0:
        return "compensation_branch_present"
    return "compensation_branch_absent"


def current_numerical_stable_branch(rows: list[dict[str, str]], stress: float, branch: BranchPoint) -> dict[str, str] | None:
    candidates = []
    for row in rows:
        if not math.isclose(as_float(row.get("stress")), stress, abs_tol=1.0e-9):
            continue
        if row.get("stability_label") != "stable":
            continue
        w = as_float(row.get("w_star"))
        q = as_float(row.get("q_star"))
        if w <= EPSILON or not (1.0e-8 < q < 1.0 - 1.0e-8):
            continue
        distance = abs(as_float(row.get("n_star")) - branch.n_star) + abs(w - branch.w_star) + abs(q - branch.q_star)
        candidates.append((distance, row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def compare_branch_to_numerical(params: RoyEvoParams = PARAMS) -> list[dict[str, Any]]:
    numerical_rows = read_csv(NUMERICAL_EQUILIBRIA_CSV)
    rows: list[dict[str, Any]] = []
    for stress in TARGET_STRESSES:
        branch = analytic_compensation_branch(params, stress)
        numerical = current_numerical_stable_branch(numerical_rows, stress, branch)
        n_num = as_float(numerical.get("n_star")) if numerical else math.nan
        w_num = as_float(numerical.get("w_star")) if numerical else math.nan
        q_num = as_float(numerical.get("q_star")) if numerical else math.nan
        z_num = as_float(numerical.get("z_star")) if numerical else math.nan
        rows.append(
            {
                "stress": stress,
                "c": branch.c,
                "z_star_analytic": branch.z_star,
                "w_star_analytic": branch.w_star,
                "n_star_analytic": branch.n_star,
                "q_star_analytic": branch.q_star,
                "b_req": branch.b_req,
                "interior_exists": branch.interior_exists,
                "existence_failure_reason": branch.existence_failure_reason,
                "numerical_equilibrium_id": numerical.get("equilibrium_id", "") if numerical else "",
                "n_star_numerical": n_num,
                "w_star_numerical": w_num,
                "q_star_numerical": q_num,
                "z_star_numerical": z_num,
                "abs_delta_n": abs(branch.n_star - n_num) if np.isfinite(n_num) else math.nan,
                "abs_delta_w": abs(branch.w_star - w_num) if np.isfinite(w_num) else math.nan,
                "abs_delta_q": abs(branch.q_star - q_num) if np.isfinite(q_num) else math.nan,
                "abs_delta_z": abs(branch.z_star - z_num) if np.isfinite(z_num) else math.nan,
            }
        )
    write_csv(BRANCH_CURRENT_CSV, rows, BRANCH_CURRENT_FIELDS)
    return rows


def write_current_interval(params: RoyEvoParams = PARAMS) -> dict[str, Any]:
    interval = compensation_interval(params)
    row = {
        "s_at_q_equals_1": interval.s_at_q_equals_1,
        "s_at_q_equals_0": interval.s_at_q_equals_0,
        "interior_stress_interval_length": interval.interior_stress_interval_length,
        "interpretation": "interior_branch_exists_for_stresses_between_boundary_q_values" if interval.valid else interval.failure_reason,
    }
    write_csv(INTERVAL_CURRENT_CSV, [row], INTERVAL_FIELDS)
    return row


def tradeoff_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r_v in R_V_VALUES:
        for a_v in A_V_VALUES:
            for b_u in B_U_VALUES:
                for ratio in B_RATIO_VALUES:
                    params = PARAMS.with_updates(r_v=r_v, a_v=a_v, b_u=b_u, b_v=b_u * ratio)
                    point = analytic_compensation_branch(params, 0.0)
                    interval = compensation_interval(params)
                    qualitative = qualitative_tradeoff_class(params, interval)
                    rows.append(
                        {
                            "r_v": r_v,
                            "a_v": a_v,
                            "b_u": b_u,
                            "b_v": params.b_v,
                            "b_v_over_b_u": ratio,
                            "c": point.c,
                            "valid_selection_ratio": bool(np.isfinite(point.c) and point.c > 0.0),
                            "z_star": point.z_star,
                            "w_star": point.w_star,
                            "n_star": point.n_star,
                            "s_at_q_equals_1": interval.s_at_q_equals_1,
                            "s_at_q_equals_0": interval.s_at_q_equals_0,
                            "interior_interval_length": interval.interior_stress_interval_length,
                            "nonnegative_interval_length": interval.nonnegative_interval_length,
                            "branch_exists_at_s_0": branch_exists_at(params, 0.0),
                            "branch_exists_at_s_0p1584375": branch_exists_at(params, 0.1584375),
                            "branch_exists_at_s_0p16486816": branch_exists_at(params, 0.16486816),
                            "branch_exists_at_s_0p175": branch_exists_at(params, 0.175),
                            "qualitative_class": qualitative,
                            "notes": point.existence_failure_reason or interval.failure_reason or "analytic_grid_entry",
                        }
                    )
    write_csv(TRADEOFF_GRID_CSV, rows, TRADEOFF_FIELDS)
    return rows


def branch_state_for_initialization(params: RoyEvoParams) -> np.ndarray:
    point = analytic_compensation_branch(params, 0.0)
    if np.isfinite(point.n_star) and np.isfinite(point.w_star):
        q0 = float(np.clip(point.q_star if np.isfinite(point.q_star) else 0.5, 0.0, 1.0))
        return np.array([max(point.n_star, 1.0e-8), max(point.w_star, 1.0e-8), q0], dtype=float)
    current = analytic_compensation_branch(PARAMS, 0.0)
    return np.array([current.n_star, current.w_star, current.q_star], dtype=float)


def selected_parameter_sets(grid_rows: list[dict[str, Any]]) -> list[ParameterSet]:
    current = ParameterSet("current", PARAMS, "current_parameterization")

    def same_as_current(row: dict[str, Any]) -> bool:
        return (
            math.isclose(as_float(row["r_v"]), PARAMS.r_v)
            and math.isclose(as_float(row["a_v"]), PARAMS.a_v)
            and math.isclose(as_float(row["b_u"]), PARAMS.b_u)
            and math.isclose(as_float(row["b_v"]), PARAMS.b_v)
        )

    present_target = [
        row
        for row in grid_rows
        if row["qualitative_class"] == "compensation_branch_present"
        and bool(row["branch_exists_at_s_0p1584375"])
        and bool(row["branch_exists_at_s_0p16486816"])
        and not same_as_current(row)
    ]
    present_target.sort(key=lambda row: as_float(row["nonnegative_interval_length"]))
    weaker_row = present_target[0] if present_target else min((row for row in grid_rows if not same_as_current(row)), key=lambda row: as_float(row["nonnegative_interval_length"]))

    absent_or_invalid = [
        row
        for row in grid_rows
        if row["qualitative_class"] != "compensation_branch_present"
        or not bool(row["branch_exists_at_s_0p175"])
    ]
    absent_or_invalid = [row for row in absent_or_invalid if not same_as_current(row) and row is not weaker_row]
    if absent_or_invalid:
        absent_row = min(absent_or_invalid, key=lambda row: as_float(row["nonnegative_interval_length"], math.inf))
        absent_note = "absent_or_target_missing_compensation_branch"
    else:
        candidates = [row for row in grid_rows if not same_as_current(row) and row is not weaker_row]
        absent_row = min(candidates, key=lambda row: as_float(row["nonnegative_interval_length"]))
        absent_note = "weakest_interval_substitute_no_absent_case"

    def to_param_set(row: dict[str, Any], label: str, note: str) -> ParameterSet:
        return ParameterSet(
            label,
            PARAMS.with_updates(
                r_v=as_float(row["r_v"]),
                a_v=as_float(row["a_v"]),
                b_u=as_float(row["b_u"]),
                b_v=as_float(row["b_v"]),
            ),
            note,
        )

    return [
        current,
        to_param_set(weaker_row, "shorter_interval_present", "present_branch_with_shorter_or_weaker_interval"),
        to_param_set(absent_row, "weak_or_absent_branch", absent_note),
    ]


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
    tail_t = times[mask]
    tail_values = values[mask]
    centered_t = tail_t - float(np.mean(tail_t))
    denom = float(np.dot(centered_t, centered_t))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(centered_t, tail_values - float(np.mean(tail_values))) / denom)


def relative_change(previous: float, latest: float) -> float:
    return float((latest - previous) / max(abs(previous), EPSILON))


def basin_label_from_classification(classification: str) -> str:
    if classification == "persistent_steady":
        return "persistent_basin"
    if classification == "extinct_steady":
        return "extinct_basin"
    if classification in {"persistent_transient", "extinct_transient", "recovery_transient", "declining_transient"}:
        return "transient_basin"
    return "unresolved_basin"


def row_note(classification: str) -> str:
    if classification.endswith("_steady"):
        return "steady_by_tail_change_and_rhs_residual"
    if "transient" in classification:
        return "window_change_indicates_transient"
    return classification


def ode_rhs_residual(state: np.ndarray, params: RoyEvoParams, stress: float) -> dict[str, float]:
    rhs = reaction_ode_evo(0.0, state, params, stress=stress, evolve=True)
    rhs_norm = float(np.linalg.norm(rhs))
    state_norm = float(np.linalg.norm(state))
    return {"rhs_norm": rhs_norm, "state_norm": state_norm, "normalized_residual": rhs_norm / max(state_norm, 1.0e-12)}


def classify_asymptotic(metrics: dict[str, Any]) -> str:
    if not bool(metrics["physical"]):
        return "nonphysical"
    tail_mean_w = float(metrics["tail_mean_w"])
    tail_min_w = float(metrics.get("tail_min_w", tail_mean_w))
    previous_w = float(metrics["previous_window_mean_w"])
    latest_w = float(metrics["latest_window_mean_w"])
    rel_change = float(metrics["relative_change_between_last_windows"])
    normalized_residual = float(metrics["normalized_residual"])
    persistent_without_slope = bool(metrics.get("persistent_without_slope_rule", False))
    persistent_with_slope = bool(metrics.get("persistent_with_slope_rule", False))
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


def classify_ode_trajectory(times: np.ndarray, values: np.ndarray, params: RoyEvoParams, stress: float) -> dict[str, Any]:
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
    previous = previous_window_mask(times)
    tail_t = times[mask]
    tail_w = w[mask]
    tail_q = q[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(times, w, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_w = float(np.mean(w[previous]))
    residual = ode_rhs_residual(values[:, -1], params, stress)
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


def state_quantities(n: np.ndarray | float, w: np.ndarray | float, q: np.ndarray | float, params: RoyEvoParams, stress: float) -> dict[str, np.ndarray]:
    return {
        "selection_gradient": np.asarray(selection_gradient(n, w, q, params), dtype=float),
        "predator_growth_factor": np.asarray(predator_growth_factor_evo(n, w, q, params, stress=stress), dtype=float),
    }


def run_selected_basin_maps(parameter_sets: list[ParameterSet], profile: str) -> list[dict[str, Any]]:
    T = 1600.0 if profile == "focused" else 500.0
    n_eval = 321 if profile == "focused" else 121
    rows: list[dict[str, Any]] = []
    for selected in parameter_sets:
        baseline = branch_state_for_initialization(selected.params)
        for stress in BASIN_STRESSES:
            for q0 in Q0_VALUES:
                for w0_scale in W0_SCALES:
                    initial = np.array([baseline[0], baseline[1] * w0_scale, q0], dtype=float)
                    z0 = float(free_space_evo(initial[0], initial[1], selected.params))
                    if z0 < 0.0:
                        rows.append(
                            {
                                "parameter_set_label": selected.label,
                                "r_v": selected.params.r_v,
                                "a_v": selected.params.a_v,
                                "b_u": selected.params.b_u,
                                "b_v": selected.params.b_v,
                                "stress": stress,
                                "q0": q0,
                                "w0_scale": w0_scale,
                                "classification": "nonphysical",
                                "basin_label": "unresolved_basin",
                                "physical": False,
                                "notes": "skipped_nonphysical_initial_condition",
                            }
                        )
                        continue
                    trajectory = simulate_ode_evo(selected.params, initial, stress=stress, evolve=True, T=T, n_eval=n_eval)
                    metrics = classify_ode_trajectory(trajectory.t, trajectory.y, selected.params, stress)
                    mask = tail_mask(trajectory.t)
                    quantities = state_quantities(trajectory.y[0], trajectory.y[1], trajectory.y[2], selected.params, stress)
                    rows.append(
                        {
                            "parameter_set_label": selected.label,
                            "r_v": selected.params.r_v,
                            "a_v": selected.params.a_v,
                            "b_u": selected.params.b_u,
                            "b_v": selected.params.b_v,
                            "stress": stress,
                            "q0": q0,
                            "w0_scale": w0_scale,
                            "classification": metrics["classification"],
                            "basin_label": metrics["basin_label"],
                            "tail_mean_w": metrics["tail_mean_w"],
                            "tail_mean_q": metrics["tail_mean_q"],
                            "q_change": float(metrics["tail_mean_q"] - q0),
                            "selection_gradient_tail_mean": float(np.mean(quantities["selection_gradient"][mask])),
                            "predator_growth_factor_tail_mean": float(np.mean(quantities["predator_growth_factor"][mask])),
                            "physical": metrics["physical"],
                            "notes": f"{selected.selection_note};{row_note(str(metrics['classification']))}",
                        }
                    )
    write_csv(SELECTED_BASIN_MAPS_CSV, rows, SELECTED_BASIN_FIELDS)
    return rows


def finite_difference_jacobian(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    jac = np.empty((f0.size, x.size), dtype=float)
    for idx in range(x.size):
        delta = step * max(1.0, abs(float(x[idx])))
        plus = x.copy()
        minus = x.copy()
        plus[idx] += delta
        minus[idx] -= delta
        jac[:, idx] = (np.asarray(func(plus), dtype=float) - np.asarray(func(minus), dtype=float)) / (2.0 * delta)
    return jac


def stability_label(max_real: float, physical: bool) -> str:
    if not physical:
        return "nonphysical"
    if max_real < -1.0e-6:
        return "stable"
    if max_real > 1.0e-6:
        return "unstable"
    return "marginal_or_unresolved"


def equilibrium_physical(state: np.ndarray, params: RoyEvoParams) -> bool:
    n, w, q = state
    z = float(free_space_evo(n, w, params))
    return bool(np.all(np.isfinite(state)) and n >= -1.0e-8 and w >= -1.0e-8 and -1.0e-6 <= q <= 1.0 + 1.0e-6 and z >= -1.0e-6)


def raw_equilibrium_guesses(params: RoyEvoParams, stress: float) -> list[np.ndarray]:
    carrying = 1.0 / params.kappa
    branch = analytic_compensation_branch(params, stress)
    branch0 = branch_state_for_initialization(params)
    q0_n = carrying - params.xi / params.r_u
    q1_n = carrying - params.xi / params.r_v
    guesses = [
        branch0,
        np.array([branch.n_star, branch.w_star, np.clip(branch.q_star, 0.0, 1.0)]) if np.isfinite(branch.n_star) else branch0,
        np.array([branch0[0], 1.0e-8, branch0[2]]),
        np.array([branch0[0], branch0[1] * 1.8, branch0[2]]),
        np.array([branch0[0], branch0[1], 0.05]),
        np.array([branch0[0], branch0[1], 0.95]),
        np.array([max(q0_n, 1.0e-8), 1.0e-8, 0.0]),
        np.array([max(q1_n, 1.0e-8), 1.0e-8, 1.0]),
        np.array([max(q0_n, 1.0e-8), 1.0e-6, 0.02]),
        np.array([max(q1_n, 1.0e-8), 1.0e-6, 0.98]),
        np.array([1.0e-8, 1.0e-8, 0.0]),
        np.array([1.0e-8, 1.0e-8, 1.0]),
        np.array([branch0[0] * 0.75, max(branch0[1] * 0.05, 1.0e-8), 0.2]),
        np.array([branch0[0] * 1.1, max(branch0[1] * 0.05, 1.0e-8), 0.8]),
    ]
    return [np.clip(guess, [0.0, 0.0, 0.0], [carrying, carrying, 1.0]) for guess in guesses]


def deduplicate_equilibria(equilibria: list[dict[str, Any]], tol: float = 1.0e-5) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for eq in sorted(equilibria, key=lambda item: as_float(item.get("residual_norm"), math.inf)):
        state = np.array([eq["n_star"], eq["w_star"], eq["q_star"]], dtype=float)
        if any(np.linalg.norm(state - np.array([other["n_star"], other["w_star"], other["q_star"]], dtype=float)) <= tol for other in unique):
            continue
        unique.append(eq)
    return unique


def find_equilibria_for_stress(params: RoyEvoParams, stress: float, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0 / params.kappa, 1.0 / params.kappa, 1.0])

    def rhs(state: np.ndarray) -> np.ndarray:
        return reaction_ode_evo(0.0, state, params, stress=stress, evolve=True)

    for guess in raw_equilibrium_guesses(params, stress):
        result = least_squares(
            rhs,
            np.clip(guess, lower, upper),
            bounds=(lower, upper),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=2000,
        )
        state = np.clip(result.x, lower, upper)
        residual = rhs(state)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > 1.0e-5:
            continue
        physical = equilibrium_physical(state, params)
        jac = finite_difference_jacobian(rhs, state)
        eigenvalues = np.linalg.eigvals(jac)
        max_real = float(np.max(np.real(eigenvalues)))
        n, w, q = [float(value) for value in state]
        rows.append(
            {
                "parameter_set_label": label,
                "stress": stress,
                "n_star": n,
                "w_star": w,
                "q_star": q,
                "z_star": float(free_space_evo(n, w, params)),
                "selection_gradient": float(selection_gradient(n, w, q, params)),
                "predator_growth_factor": float(predator_growth_factor_evo(n, w, q, params, stress=stress)),
                "max_real_eigenvalue": max_real,
                "stability_label": stability_label(max_real, physical),
                "physical": physical,
                "notes": f"least_squares_residual={residual_norm:.3e}",
                "residual_norm": residual_norm,
            }
        )
    unique = deduplicate_equilibria(rows)
    for idx, row in enumerate(unique, start=1):
        row["equilibrium_id"] = f"{label}_s{stress:.9g}_eq{idx:02d}".replace(".", "p")
    return unique


def run_selected_equilibria(parameter_sets: list[ParameterSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected in parameter_sets:
        for stress in SELECTED_EQUILIBRIUM_STRESSES:
            rows.extend(find_equilibria_for_stress(selected.params, stress, selected.label))
    write_csv(SELECTED_EQUILIBRIA_CSV, rows, SELECTED_EQUILIBRIA_FIELDS)
    return rows


def analytic_match_current(rows: list[dict[str, Any]], q_tol: float = 1.0e-4) -> bool:
    deltas = [as_float(row.get("abs_delta_q")) for row in rows if row.get("interior_exists") in {True, "True", "true", "1"}]
    return bool(deltas and max(deltas) < q_tol)


def decide_final_label(
    *,
    analytic_matches_current: bool,
    max_abs_delta_q: float,
    branch_present_fraction: float,
    basin_maps_completed: bool,
    equilibria_completed: bool,
) -> str:
    if not analytic_matches_current or not np.isfinite(max_abs_delta_q):
        return "ode_compensation_branch_unresolved"
    if not basin_maps_completed or not equilibria_completed:
        return "ode_compensation_branch_unresolved"
    if max_abs_delta_q < 1.0e-4 and branch_present_fraction >= 0.40:
        return "ode_compensation_branch_supported"
    return "ode_compensation_branch_parameter_sensitive"


def write_summary(
    branch_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    equilibria_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    q_deltas = [as_float(row.get("abs_delta_q")) for row in branch_rows if bool(row.get("interior_exists"))]
    max_abs_delta_q = max(q_deltas) if q_deltas else math.nan
    matches = analytic_match_current(branch_rows)
    present_count = sum(row["qualitative_class"] == "compensation_branch_present" for row in grid_rows)
    present_at_targets = sum(
        row["qualitative_class"] == "compensation_branch_present"
        and bool(row["branch_exists_at_s_0p1584375"])
        and bool(row["branch_exists_at_s_0p16486816"])
        and bool(row["branch_exists_at_s_0p175"])
        for row in grid_rows
    )
    total = len(grid_rows)
    present_fraction = present_count / total if total else 0.0
    basin_completed = bool(basin_rows) and len({row["parameter_set_label"] for row in basin_rows}) >= 3
    equilibria_completed = bool(equilibria_rows) and len({row["parameter_set_label"] for row in equilibria_rows}) >= 3
    final_label = decide_final_label(
        analytic_matches_current=matches,
        max_abs_delta_q=max_abs_delta_q,
        branch_present_fraction=present_fraction,
        basin_maps_completed=basin_completed,
        equilibria_completed=equilibria_completed,
    )
    rows = [
        {
            "metric": "analytic_matches_current_numerical_branch",
            "value": matches,
            "interpretation": "analytic branch matches existing stable persistent numerical equilibria",
        },
        {
            "metric": "max_abs_delta_q_current_branch",
            "value": max_abs_delta_q,
            "interpretation": "maximum q difference between analytic and numerical current-parameter branch",
        },
        {
            "metric": "focused_grid_total_parameter_sets",
            "value": total,
            "interpretation": "structured local trade-off combinations tested analytically",
        },
        {
            "metric": "focused_grid_branch_present_count",
            "value": present_count,
            "interpretation": "parameter sets with valid compensation interval overlapping nonnegative stress",
        },
        {
            "metric": "focused_grid_branch_present_fraction",
            "value": present_fraction,
            "interpretation": "local structured robustness fraction, not global generality",
        },
        {
            "metric": "focused_grid_branch_present_at_target_stress_count",
            "value": present_at_targets,
            "interpretation": "parameter sets with branch present at all target stress diagnostics",
        },
        {
            "metric": "selected_basin_maps_completed",
            "value": basin_completed,
            "interpretation": "selected ODE basin maps completed for three parameter sets",
        },
        {
            "metric": "selected_equilibria_completed",
            "value": equilibria_completed,
            "interpretation": "selected numerical equilibrium/stability diagnostics completed",
        },
        {
            "metric": "final_label",
            "value": final_label,
            "interpretation": "allowed final label for homogeneous compensation robustness",
        },
    ]
    write_csv(SUMMARY_CSV, rows, SUMMARY_FIELDS)
    return rows, final_label


def plot_current_branch(branch_rows: list[dict[str, Any]]) -> None:
    stress = [as_float(row["stress"]) for row in branch_rows]
    q_analytic = [as_float(row["q_star_analytic"]) for row in branch_rows]
    q_numerical = [as_float(row["q_star_numerical"]) for row in branch_rows]
    n_analytic = [as_float(row["n_star_analytic"]) for row in branch_rows]
    n_numerical = [as_float(row["n_star_numerical"]) for row in branch_rows]
    w_analytic = [as_float(row["w_star_analytic"]) for row in branch_rows]
    w_numerical = [as_float(row["w_star_numerical"]) for row in branch_rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.8), sharex=True, constrained_layout=True)
    axes[0].plot(stress, q_analytic, color="#1b9e77", marker="o", label="analytic q*")
    axes[0].scatter(stress, q_numerical, color="#ffffff", edgecolor="#1b9e77", s=52, label="numerical stable q*")
    axes[1].plot(stress, n_analytic, color="#2f6fbb", marker="o", label="analytic n*")
    axes[1].scatter(stress, n_numerical, color="#ffffff", edgecolor="#2f6fbb", s=52, label="numerical n*")
    axes[2].plot(stress, w_analytic, color="#7b3294", marker="o", label="analytic w*")
    axes[2].scatter(stress, w_numerical, color="#ffffff", edgecolor="#7b3294", s=52, label="numerical w*")
    axes[0].set_ylabel("defense frequency q*")
    axes[1].set_ylabel("prey density n*")
    axes[2].set_ylabel("predator density w*")
    axes[2].set_xlabel("mortality stress")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("Analytic homogeneous compensation branch matches numerical equilibria")
    save_figure(fig, FIG33_PATH)


def plot_tradeoff_grid(grid_rows: list[dict[str, Any]]) -> None:
    y_labels = []
    x_labels = []
    for r_v in R_V_VALUES:
        for a_v in A_V_VALUES:
            y_labels.append(f"r_v={r_v:.2f}\na_v={a_v:.2f}")
    for b_u in B_U_VALUES:
        for ratio in B_RATIO_VALUES:
            x_labels.append(f"b_u={b_u:.2f}\nratio={ratio:.2f}")
    matrix = np.zeros((len(y_labels), len(x_labels)))
    for row in grid_rows:
        y = list((r, a) for r in R_V_VALUES for a in A_V_VALUES).index((as_float(row["r_v"]), as_float(row["a_v"])))
        x = list((b, ratio) for b in B_U_VALUES for ratio in B_RATIO_VALUES).index((as_float(row["b_u"]), as_float(row["b_v_over_b_u"])))
        matrix[y, x] = as_float(row["nonnegative_interval_length"], 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [3.0, 1.0]})
    im = axes[0].imshow(matrix, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(x_labels)))
    axes[0].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(y_labels)))
    axes[0].set_yticklabels(y_labels, fontsize=8)
    axes[0].set_title("Nonnegative interior compensation interval")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="stress interval length")

    by_ratio: dict[float, list[bool]] = {ratio: [] for ratio in B_RATIO_VALUES}
    for row in grid_rows:
        by_ratio[as_float(row["b_v_over_b_u"])].append(row["qualitative_class"] == "compensation_branch_present")
    ratios = list(by_ratio)
    fractions = [sum(values) / len(values) for values in by_ratio.values()]
    axes[1].bar([str(ratio) for ratio in ratios], fractions, color="#2f6fbb")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("b_v / b_u")
    axes[1].set_ylabel("present fraction")
    axes[1].set_title("Branch presence")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Structured local trade-off robustness of the compensation branch")
    save_figure(fig, FIG34_PATH)


def plot_selected_basin_maps(rows: list[dict[str, Any]], parameter_sets: list[ParameterSet]) -> None:
    q_values = sorted({as_float(row["q0"]) for row in rows})
    w_values = sorted({as_float(row["w0_scale"]) for row in rows})
    category_index = {label: idx for idx, label in enumerate(BASIN_LABELS)}
    cmap = mcolors.ListedColormap([BASIN_COLORS[label] for label in BASIN_LABELS])
    norm = mcolors.BoundaryNorm(np.arange(len(BASIN_LABELS) + 1) - 0.5, len(BASIN_LABELS))
    fig, axes = plt.subplots(len(parameter_sets), len(BASIN_STRESSES), figsize=(10.8, 9.2), sharex=True, sharey=True, constrained_layout=True)
    for row_idx, selected in enumerate(parameter_sets):
        for col_idx, stress in enumerate(BASIN_STRESSES):
            ax = axes[row_idx, col_idx]
            matrix = np.full((len(w_values), len(q_values)), category_index["unresolved_basin"])
            for row in rows:
                if row["parameter_set_label"] != selected.label or not math.isclose(as_float(row["stress"]), stress):
                    continue
                x = q_values.index(as_float(row["q0"]))
                y = w_values.index(as_float(row["w0_scale"]))
                matrix[y, x] = category_index.get(str(row["basin_label"]), category_index["unresolved_basin"])
            ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
            ax.set_title(f"{selected.label}\ns={stress:.9g}", fontsize=9)
            ax.set_xticks(range(len(q_values)))
            ax.set_xticklabels([format_float(value, 3) for value in q_values], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(w_values)))
            ax.set_yticklabels([format_float(value, 3) for value in w_values], fontsize=8)
            ax.grid(color="white", linewidth=0.7)
            if col_idx == 0:
                ax.set_ylabel("w0 scale")
            if row_idx == len(parameter_sets) - 1:
                ax.set_xlabel("q0")
    handles = [mpatches.Patch(color=BASIN_COLORS[label], label=label.replace("_", " ")) for label in BASIN_LABELS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Selected homogeneous ODE basin maps across trade-off settings")
    save_figure(fig, FIG35_PATH)


def plot_selected_equilibria(rows: list[dict[str, Any]], parameter_sets: list[ParameterSet]) -> None:
    markers = {"current": "o", "shorter_interval_present": "s", "weak_or_absent_branch": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    for selected in parameter_sets:
        for row in rows:
            if row["parameter_set_label"] != selected.label:
                continue
            label = str(row["stability_label"])
            color = STABILITY_COLORS.get(label, "#777777")
            axes[0].scatter(as_float(row["stress"]), as_float(row["q_star"]), color=color, edgecolor="black", s=56, marker=markers.get(selected.label, "o"))
            axes[1].scatter(as_float(row["stress"]), as_float(row["w_star"]), color=color, edgecolor="black", s=56, marker=markers.get(selected.label, "o"))
    axes[0].set_ylabel("equilibrium q*")
    axes[1].set_ylabel("equilibrium w*")
    for ax in axes:
        ax.set_xlabel("stress")
        ax.grid(alpha=0.25)
    stability_handles = [mpatches.Patch(color=color, label=label.replace("_", " ")) for label, color in STABILITY_COLORS.items()]
    set_handles = [
        plt.Line2D([0], [0], color="black", marker=markers.get(selected.label, "o"), linestyle="", label=selected.label)
        for selected in parameter_sets
    ]
    fig.legend(handles=stability_handles + set_handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Selected homogeneous equilibria and finite-difference stability")
    save_figure(fig, FIG36_PATH)


def plot_mechanism_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x: float, y: float, text: str, width: float = 1.72, color: str = "#f4f7fb") -> None:
        patch = mpatches.FancyBboxPatch(
            (x - width / 2, y - 0.38),
            width,
            0.76,
            boxstyle="round,pad=0.06",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)

    def arrow(x0: float, y0: float, x1: float, y1: float, text: str = "") -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#333333"})
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.24, text, ha="center", fontsize=8)

    y = 3.0
    box(1.0, y, "stress up", color="#fbe9e7")
    box(2.8, y, "predator\npressure down")
    box(4.6, y, "selection on q\nshifts")
    box(6.4, y, "q* down")
    box(8.2, y, "b(q*) up")
    box(10.2, y, "predator growth\nbalance restored", width=2.2, color="#e8f5e9")
    for x0, x1 in [(1.86, 1.94), (3.66, 3.74), (5.46, 5.54), (7.26, 7.34), (9.06, 9.14)]:
        arrow(x0, y, x1, y)
    ax.text(
        6.0,
        1.2,
        "Homogeneous reaction-level compensation: no persistent spatial patterning is required for this branch.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    fig.suptitle("Analytic compensation mechanism in the homogeneous ODE")
    save_figure(fig, FIG37_PATH)


def write_research_note(
    branch_rows: list[dict[str, Any]],
    interval_row: dict[str, Any],
    grid_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    equilibria_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    final_label: str,
) -> None:
    summary = {row["metric"]: row["value"] for row in summary_rows}
    q_values = ", ".join(f"s={as_float(row['stress']):.9g}: q*={as_float(row['q_star_analytic']):.4f}" for row in branch_rows)
    basin_counts: dict[str, Counter[str]] = {}
    for row in basin_rows:
        basin_counts.setdefault(str(row["parameter_set_label"]), Counter())[str(row["basin_label"])] += 1
    basin_lines = "\n".join(
        f"- `{label}`: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        for label, counts in sorted(basin_counts.items())
    )
    stable_counts = Counter(row["parameter_set_label"] for row in equilibria_rows if row["stability_label"] == "stable")
    stable_lines = "\n".join(f"- `{label}`: stable equilibria found = {count}" for label, count in sorted(stable_counts.items()))
    interpretation = (
        "The analytic compensation branch matches the current numerical persistent equilibria and is present across a substantial fraction of the local structured trade-off grid."
        if final_label == "ode_compensation_branch_supported"
        else "The analytic compensation branch matches the current numerical equilibria, but the local structured grid indicates parameter sensitivity."
        if final_label == "ode_compensation_branch_parameter_sensitive"
        else "The current diagnostics do not yet resolve the homogeneous compensation branch."
    )
    text = f"""# Robustness of the Homogeneous Compensation Mechanism in the Roy ODE

## Purpose

This note tests whether the stable persistent homogeneous branch identified after PR #19 follows from the ODE equilibrium equations and whether it persists under a local structured trade-off perturbation. The analysis is ODE-only and does not run PDE scans or change model equations.

## Analytic Compensation Branch

For an interior equilibrium with \(n>0\), \(w>0\), and \(0<q<1\), the ODE equilibrium conditions imply \(w=c z\), where \(c=(r_v-r_u)/(a_v-a_u)\). Because the growth and palatability trade-offs are linear, \(r(q)-c a(q)=r_u-c a_u\), giving \(z^*=\\xi/(r_u-c a_u)\), \(w^*=c z^*\), and \(n^*=\\kappa^{{-1}}-z^*-w^*\). The predator equation then determines the required conversion value \(b_{{req}}(s)\), and the linear conversion trade-off gives \(q^*(s)=(b_{{req}}(s)-b_u)/(b_v-b_u)\).

The branch exists as an interior compensation branch when \(n^*>0\), \(w^*>0\), \(z^*>0\), and \(0<q^*(s)<1\).

## Match to Numerical Equilibria

In the current parameterization, \(q^*(s)\) decreases with stress while \(n^*\) and \(w^*\) remain approximately fixed. Analytic branch values are:

{q_values}

The maximum absolute difference in \(q^*\) between the analytic branch and the existing stable persistent numerical equilibria is `{summary.get("max_abs_delta_q_current_branch")}`. The comparison table is `results/roy_ode_compensation_branch_current.csv`, and the figure is `figures/roy_evo_spatial/report/fig33_compensation_branch_current.png`.

## Stress Interval of the Interior Branch

For the current parameterization, the branch is interior between the stress values where \(q^*=1\) and \(q^*=0\):

- `s_at_q_equals_1 = {interval_row["s_at_q_equals_1"]}`
- `s_at_q_equals_0 = {interval_row["s_at_q_equals_0"]}`
- `interior_stress_interval_length = {interval_row["interior_stress_interval_length"]}`

This interval is the homogeneous compensation range in which changing \(q^*\) can offset increased mortality while retaining a positive predator equilibrium.

## Local Trade-Off Robustness

The structured local grid varies \(r_v\), \(a_v\), \(b_u\), and \(b_v/b_u\) around the current trade-off values. It is not a global parameter scan. The grid has `{summary.get("focused_grid_total_parameter_sets")}` parameter sets; `{summary.get("focused_grid_branch_present_count")}` have a compensation branch with a nonzero interval overlapping nonnegative stress, for a fraction `{summary.get("focused_grid_branch_present_fraction")}`.

The grid output is `results/roy_ode_compensation_tradeoff_grid.csv`, and the summary figure is `figures/roy_evo_spatial/report/fig34_compensation_interval_tradeoff_grid.png`.

## Selected Basin Maps

Selected ODE basin maps were computed for the current parameterization and two local trade-off variants. These ODE diagnostics use the same \(q_0\)--\(w_0\) grid as the earlier basin comparisons and do not run PDE scans.

{basin_lines}

The table is `results/roy_ode_compensation_selected_basin_maps.csv`, and the figure is `figures/roy_evo_spatial/report/fig35_selected_tradeoff_basin_maps.png`.

## Selected Equilibria and Stability

Finite-difference Jacobian stability diagnostics were computed for the same selected parameter sets.

{stable_lines}

The table is `results/roy_ode_compensation_selected_equilibria.csv`, and the figure is `figures/roy_evo_spatial/report/fig36_selected_equilibria_stability.png`.

## Final Label

`{final_label}`

## Interpretation

{interpretation} In the current parameterization, the branch is derived from the ODE equilibrium equations: as mortality stress increases, \(q^*(s)\) shifts downward, increasing predator conversion opportunity and restoring predator growth balance at a positive predator equilibrium. This explains the homogeneous reaction-level compensation mechanism behind indirect rescue.

## What Is Still Not General

The robustness check is local and structured. It does not establish global behavior over all trade-off forms, diffusion settings, evolutionary rates, or biological calibrations. It also does not replace the earlier conclusion that the spatial PDE is currently best interpreted as preserving basin-dependent homogeneous dynamics rather than generating persistent spatial-pattern-mediated rescue.

## Files

- `experiments/22_roy_ode_compensation_robustness.py`
- `results/roy_ode_compensation_branch_current.csv`
- `results/roy_ode_compensation_interval_current.csv`
- `results/roy_ode_compensation_tradeoff_grid.csv`
- `results/roy_ode_compensation_selected_basin_maps.csv`
- `results/roy_ode_compensation_selected_equilibria.csv`
- `results/roy_ode_compensation_robustness_summary.csv`
- `figures/roy_evo_spatial/report/fig33_compensation_branch_current.png`
- `figures/roy_evo_spatial/report/fig34_compensation_interval_tradeoff_grid.png`
- `figures/roy_evo_spatial/report/fig35_selected_tradeoff_basin_maps.png`
- `figures/roy_evo_spatial/report/fig36_selected_equilibria_stability.png`
- `figures/roy_evo_spatial/report/fig37_compensation_mechanism_diagram.png`

## Next Step

Use the analytic branch conditions to design narrower trade-off hypotheses before any broader parameter work. The next mathematical step is to connect branch existence and stability to explicit inequalities in the trade-off parameters.
"""
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(text, encoding="utf-8")


def run(profile: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    branch_rows = compare_branch_to_numerical(PARAMS)
    interval_row = write_current_interval(PARAMS)
    grid_rows = tradeoff_grid()
    parameter_sets = selected_parameter_sets(grid_rows)
    basin_rows = run_selected_basin_maps(parameter_sets, profile)
    equilibria_rows = run_selected_equilibria(parameter_sets)
    summary_rows, final_label = write_summary(branch_rows, grid_rows, basin_rows, equilibria_rows)
    plot_current_branch(branch_rows)
    plot_tradeoff_grid(grid_rows)
    plot_selected_basin_maps(basin_rows, parameter_sets)
    plot_selected_equilibria(equilibria_rows, parameter_sets)
    plot_mechanism_diagram()
    write_research_note(branch_rows, interval_row, grid_rows, basin_rows, equilibria_rows, summary_rows, final_label)
    for path in [
        BRANCH_CURRENT_CSV,
        INTERVAL_CURRENT_CSV,
        TRADEOFF_GRID_CSV,
        SELECTED_BASIN_MAPS_CSV,
        SELECTED_EQUILIBRIA_CSV,
        SUMMARY_CSV,
        FIG33_PATH,
        FIG34_PATH,
        FIG35_PATH,
        FIG36_PATH,
        FIG37_PATH,
        NOTE_PATH,
    ]:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label: {final_label}")
    return final_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused", "full"), default="focused")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
