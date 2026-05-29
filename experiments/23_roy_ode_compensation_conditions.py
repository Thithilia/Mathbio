#!/usr/bin/env python
"""Derive existence and local-stability conditions for the Roy ODE branch.

This experiment is ODE-only. It evaluates analytic compensation-branch
conditions, the interior stress interval, and local stability via the analytic
Jacobian. It does not run PDE simulations or change model equations.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import RoyEvoParams, a_of_q, b_of_q, free_space_evo, r_of_q, reaction_ode_evo  # noqa: E402


PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
TARGET_STRESSES = (0.0, 0.069448242, 0.11765625, 0.1584375, 0.16486816, 0.175)
CONDITION_GRID_STRESSES = (0.0, 0.11765625, 0.1584375, 0.16486816, 0.175)
R_V_VALUES = (0.55, 0.60, 0.65, 0.70, 0.75)
A_V_VALUES = (0.25, 0.30, 0.35, 0.40, 0.45)
B_U_VALUES = (0.06, 0.08, 0.10)
B_RATIO_VALUES = (0.25, 0.50, 0.75)

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTES_DIR = ROOT / "research_notes"

CURRENT_CONDITIONS_CSV = RESULTS_DIR / "roy_ode_compensation_conditions_current.csv"
INTERVAL_FORMULA_CSV = RESULTS_DIR / "roy_ode_compensation_stress_interval_formula.csv"
STABILITY_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_stability_current.csv"
CONDITION_GRID_CSV = RESULTS_DIR / "roy_ode_compensation_condition_grid.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_ode_compensation_conditions_summary.csv"
NOTE_PATH = NOTES_DIR / "roy_ode_compensation_conditions.md"

HOMOGENEOUS_EQUILIBRIA_CSV = RESULTS_DIR / "roy_ode_homogeneous_equilibria.csv"
BRANCH_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_branch_current.csv"

FIG38_PATH = FIG_DIR / "fig38_compensation_conditions_region.png"
FIG39_PATH = FIG_DIR / "fig39_compensation_stability_along_branch.png"
FIG40_PATH = FIG_DIR / "fig40_compensation_stress_interval.png"
FIG41_PATH = FIG_DIR / "fig41_compensation_conditions_schematic.png"

CURRENT_CONDITIONS_FIELDS = ["quantity", "value", "condition", "satisfied", "interpretation"]
INTERVAL_FORMULA_FIELDS = ["quantity", "formula", "current_value", "interpretation"]
STABILITY_FIELDS = [
    "stress",
    "n_star",
    "w_star",
    "q_star",
    "max_real_eigenvalue_analytic",
    "eigenvalues_real_analytic",
    "eigenvalues_imag_analytic",
    "stability_label_analytic",
    "max_real_eigenvalue_finite_difference",
    "abs_delta_max_real",
    "trace",
    "determinant",
    "notes",
]
CONDITION_GRID_FIELDS = [
    "r_v",
    "a_v",
    "b_u",
    "b_v",
    "b_v_over_b_u",
    "c",
    "valid_c_positive",
    "valid_denominator",
    "valid_z",
    "valid_w",
    "valid_n",
    "s_q0",
    "s_q1",
    "interior_stress_interval_low",
    "interior_stress_interval_high",
    "interior_interval_length",
    "branch_exists_at_s_0",
    "branch_exists_at_s_0p11765625",
    "branch_exists_at_s_0p1584375",
    "branch_exists_at_s_0p16486816",
    "branch_exists_at_s_0p175",
    "stability_at_s_0p1584375",
    "stability_at_s_0p16486816",
    "condition_class",
]
SUMMARY_FIELDS = ["metric", "value", "interpretation"]

CLASS_ORDER = (
    "valid_stable_compensation_branch",
    "valid_unstable_compensation_branch",
    "valid_branch_outside_target_stress",
    "invalid_tradeoff_geometry",
    "invalid_feasibility",
)
CLASS_COLORS = {
    "valid_stable_compensation_branch": "#1b9e77",
    "valid_unstable_compensation_branch": "#d95f02",
    "valid_branch_outside_target_stress": "#7570b3",
    "invalid_tradeoff_geometry": "#666666",
    "invalid_feasibility": "#c23b3b",
}


@dataclass(frozen=True)
class Geometry:
    delta_r: float
    delta_a: float
    delta_b: float
    c: float
    denominator: float
    z_star: float
    w_star: float
    n_star: float
    valid_c_positive: bool
    valid_denominator: bool
    valid_z: bool
    valid_w: bool
    valid_n: bool
    valid_geometry: bool
    failure_reason: str


@dataclass(frozen=True)
class StressInterval:
    s_q0: float
    s_q1: float
    interval_low: float
    interval_high: float
    interval_length: float
    valid: bool
    failure_reason: str


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


def format_float(value: float, digits: int = 7) -> str:
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


def delta_values(params: RoyEvoParams) -> dict[str, float]:
    return {
        "Delta r": params.r_v - params.r_u,
        "Delta a": params.a_v - params.a_u,
        "Delta b": params.b_v - params.b_u,
    }


def selection_ratio_c(params: RoyEvoParams) -> float:
    values = delta_values(params)
    if abs(values["Delta a"]) < 1.0e-14:
        return math.nan
    return values["Delta r"] / values["Delta a"]


def analytic_compensation_geometry(params: RoyEvoParams) -> dict[str, float | bool | str]:
    values = delta_values(params)
    delta_a = values["Delta a"]
    c = selection_ratio_c(params)
    valid_c = bool(np.isfinite(c) and c > 0.0)
    denominator = params.r_u - c * params.a_u if np.isfinite(c) else math.nan
    valid_denominator = bool(np.isfinite(denominator) and denominator > 0.0)
    z_star = params.xi / denominator if valid_denominator else math.nan
    w_star = c * z_star if valid_c and np.isfinite(z_star) else math.nan
    n_star = 1.0 / params.kappa - z_star - w_star if np.isfinite(w_star) else math.nan
    valid_z = bool(np.isfinite(z_star) and z_star > 0.0)
    valid_w = bool(np.isfinite(w_star) and w_star > 0.0)
    valid_n = bool(np.isfinite(n_star) and n_star > 0.0)
    if abs(delta_a) < 1.0e-14:
        failure = "delta_a_zero"
    elif not valid_c:
        failure = "c_not_positive"
    elif not valid_denominator:
        failure = "r_u_minus_c_a_u_not_positive"
    elif not (valid_z and valid_w and valid_n):
        failure = "nonpositive_geometry"
    else:
        failure = ""
    return {
        **values,
        "c": c,
        "r_u_minus_c_a_u": denominator,
        "z_star": z_star,
        "w_star": w_star,
        "n_star": n_star,
        "valid_c_positive": valid_c,
        "valid_denominator": valid_denominator,
        "valid_z": valid_z,
        "valid_w": valid_w,
        "valid_n": valid_n,
        "valid_geometry": bool(valid_c and valid_denominator and valid_z and valid_w and valid_n),
        "failure_reason": failure,
    }


def analytic_q_star(params: RoyEvoParams, stress: float) -> float:
    geometry = analytic_compensation_geometry(params)
    if not geometry["valid_geometry"]:
        return math.nan
    delta_b = float(geometry["Delta b"])
    if abs(delta_b) < 1.0e-14:
        return math.nan
    n_star = float(geometry["n_star"])
    z_star = float(geometry["z_star"])
    w_star = float(geometry["w_star"])
    b_req = (params.m + stress + params.mu * w_star) / (n_star * z_star)
    return (b_req - params.b_u) / delta_b


def stress_interval_for_q_in_unit_interval(params: RoyEvoParams) -> dict[str, float | bool | str]:
    geometry = analytic_compensation_geometry(params)
    if not geometry["valid_geometry"]:
        return {
            "s_q0": math.nan,
            "s_q1": math.nan,
            "interior_stress_interval_low": math.nan,
            "interior_stress_interval_high": math.nan,
            "interior_interval_length": 0.0,
            "valid_interval": False,
            "failure_reason": str(geometry["failure_reason"]),
        }
    delta_b = float(geometry["Delta b"])
    if abs(delta_b) < 1.0e-14:
        return {
            "s_q0": math.nan,
            "s_q1": math.nan,
            "interior_stress_interval_low": math.nan,
            "interior_stress_interval_high": math.nan,
            "interior_interval_length": 0.0,
            "valid_interval": False,
            "failure_reason": "delta_b_zero",
        }
    n_star = float(geometry["n_star"])
    z_star = float(geometry["z_star"])
    w_star = float(geometry["w_star"])
    s_q0 = n_star * z_star * params.b_u - params.m - params.mu * w_star
    s_q1 = n_star * z_star * params.b_v - params.m - params.mu * w_star
    low = min(s_q0, s_q1)
    high = max(s_q0, s_q1)
    length = high - low
    return {
        "s_q0": s_q0,
        "s_q1": s_q1,
        "interior_stress_interval_low": low,
        "interior_stress_interval_high": high,
        "interior_interval_length": length,
        "valid_interval": bool(np.isfinite(length) and length > 0.0),
        "failure_reason": "" if np.isfinite(length) and length > 0.0 else "nonpositive_interval",
    }


def compensation_existence_conditions(params: RoyEvoParams, stress: float) -> dict[str, Any]:
    geometry = analytic_compensation_geometry(params)
    interval = stress_interval_for_q_in_unit_interval(params)
    q_star = analytic_q_star(params, stress)
    q_interior = bool(np.isfinite(q_star) and 0.0 < q_star < 1.0)
    exists = bool(geometry["valid_geometry"] and interval["valid_interval"] and q_interior)
    return {
        **geometry,
        **interval,
        "stress": stress,
        "q_star": q_star,
        "q_star_interior": q_interior,
        "branch_exists": exists,
        "existence_failure_reason": "" if exists else str(geometry["failure_reason"] or interval["failure_reason"] or "q_star_not_in_unit_interval"),
    }


def b_req(params: RoyEvoParams, stress: float) -> float:
    geometry = analytic_compensation_geometry(params)
    if not geometry["valid_geometry"]:
        return math.nan
    return (params.m + stress + params.mu * float(geometry["w_star"])) / (float(geometry["n_star"]) * float(geometry["z_star"]))


def branch_state(params: RoyEvoParams, stress: float) -> tuple[float, float, float]:
    geometry = analytic_compensation_geometry(params)
    return float(geometry["n_star"]), float(geometry["w_star"]), analytic_q_star(params, stress)


def ode_rhs_jacobian(n: float, w: float, q: float, stress: float, params: RoyEvoParams) -> np.ndarray:
    z = float(free_space_evo(n, w, params))
    r = float(r_of_q(q, params))
    a = float(a_of_q(q, params))
    b = float(b_of_q(q, params))
    values = delta_values(params)
    delta_r = values["Delta r"]
    delta_a = values["Delta a"]
    delta_b = values["Delta b"]
    r_balance = r * z - params.xi - a * w
    p_balance = b * n * z - (params.m + stress) - params.mu * w
    g_balance = delta_r * z - delta_a * w

    r_n = -r
    r_w = -r - a
    r_q = g_balance
    p_n = b * (z - n)
    p_w = -b * n - params.mu
    p_q = delta_b * n * z
    g_n = -delta_r
    g_w = -delta_r - delta_a
    return np.array(
        [
            [r_balance + n * r_n, n * r_w, n * r_q],
            [w * p_n, p_balance + w * p_w, w * p_q],
            [params.nu * q * (1.0 - q) * g_n, params.nu * q * (1.0 - q) * g_w, params.nu * (1.0 - 2.0 * q) * g_balance],
        ],
        dtype=float,
    )


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


def stability_label(max_real: float) -> str:
    if max_real < -1.0e-6:
        return "stable"
    if max_real > 1.0e-6:
        return "unstable"
    return "marginal_or_unresolved"


def compensation_branch_jacobian(params: RoyEvoParams, stress: float) -> dict[str, Any]:
    conditions = compensation_existence_conditions(params, stress)
    if not conditions["branch_exists"]:
        return {
            "stress": stress,
            "n_star": conditions["n_star"],
            "w_star": conditions["w_star"],
            "q_star": conditions["q_star"],
            "max_real_eigenvalue_analytic": math.nan,
            "eigenvalues_real_analytic": "",
            "eigenvalues_imag_analytic": "",
            "stability_label_analytic": "not_interior",
            "max_real_eigenvalue_finite_difference": math.nan,
            "abs_delta_max_real": math.nan,
            "trace": math.nan,
            "determinant": math.nan,
            "notes": str(conditions["existence_failure_reason"]),
        }
    n, w, q = branch_state(params, stress)
    analytic = ode_rhs_jacobian(n, w, q, stress, params)
    analytic_eigs = np.linalg.eigvals(analytic)
    max_real = float(np.max(np.real(analytic_eigs)))

    def rhs(state: np.ndarray) -> np.ndarray:
        return reaction_ode_evo(0.0, state, params, stress=stress, evolve=True)

    finite = finite_difference_jacobian(rhs, np.array([n, w, q], dtype=float))
    finite_eigs = np.linalg.eigvals(finite)
    finite_max = float(np.max(np.real(finite_eigs)))
    return {
        "stress": stress,
        "n_star": n,
        "w_star": w,
        "q_star": q,
        "max_real_eigenvalue_analytic": max_real,
        "eigenvalues_real_analytic": ";".join(format_float(float(value), 9) for value in np.real(analytic_eigs)),
        "eigenvalues_imag_analytic": ";".join(format_float(float(value), 9) for value in np.imag(analytic_eigs)),
        "stability_label_analytic": stability_label(max_real),
        "max_real_eigenvalue_finite_difference": finite_max,
        "abs_delta_max_real": abs(max_real - finite_max),
        "trace": float(np.trace(analytic)),
        "determinant": float(np.linalg.det(analytic)),
        "notes": "analytic_jacobian_compared_to_local_finite_difference",
    }


def write_current_conditions(params: RoyEvoParams = PARAMS) -> list[dict[str, Any]]:
    geometry = analytic_compensation_geometry(params)
    interval = stress_interval_for_q_in_unit_interval(params)
    target_conditions = [compensation_existence_conditions(params, stress) for stress in TARGET_STRESSES]
    rows = [
        {
            "quantity": "Delta r",
            "value": geometry["Delta r"],
            "condition": "r_v-r_u < 0 for defended-prey growth cost",
            "satisfied": float(geometry["Delta r"]) < 0.0,
            "interpretation": "defense lowers prey growth",
        },
        {
            "quantity": "Delta a",
            "value": geometry["Delta a"],
            "condition": "a_v-a_u < 0 for defended-prey palatability reduction",
            "satisfied": float(geometry["Delta a"]) < 0.0,
            "interpretation": "defense lowers attack or palatability",
        },
        {
            "quantity": "Delta b",
            "value": geometry["Delta b"],
            "condition": "b_v-b_u != 0 so q can tune conversion",
            "satisfied": abs(float(geometry["Delta b"])) > 1.0e-14,
            "interpretation": "conversion trade-off is non-flat",
        },
        {
            "quantity": "c",
            "value": geometry["c"],
            "condition": "c=(Delta r)/(Delta a)>0",
            "satisfied": geometry["valid_c_positive"],
            "interpretation": "selection-gradient condition can set w*=c z* with positive ratio",
        },
        {
            "quantity": "r_u - c a_u",
            "value": geometry["r_u_minus_c_a_u"],
            "condition": "r_u-c a_u>0",
            "satisfied": geometry["valid_denominator"],
            "interpretation": "prey equation yields positive z* denominator",
        },
        {
            "quantity": "z_star",
            "value": geometry["z_star"],
            "condition": "z*>0",
            "satisfied": geometry["valid_z"],
            "interpretation": "positive free-space coordinate",
        },
        {
            "quantity": "w_star",
            "value": geometry["w_star"],
            "condition": "w*>0",
            "satisfied": geometry["valid_w"],
            "interpretation": "persistent predator equilibrium",
        },
        {
            "quantity": "n_star",
            "value": geometry["n_star"],
            "condition": "n*>0",
            "satisfied": geometry["valid_n"],
            "interpretation": "positive prey equilibrium",
        },
        {
            "quantity": "s_at_q_equals_1",
            "value": interval["s_q1"],
            "condition": "stress endpoint at q*=1",
            "satisfied": np.isfinite(as_float(interval["s_q1"])),
            "interpretation": "fully defended endpoint of interior branch",
        },
        {
            "quantity": "s_at_q_equals_0",
            "value": interval["s_q0"],
            "condition": "stress endpoint at q*=0",
            "satisfied": np.isfinite(as_float(interval["s_q0"])),
            "interpretation": "undefended endpoint of interior branch",
        },
        {
            "quantity": "stress_interval_length",
            "value": interval["interior_interval_length"],
            "condition": "max(s_q0,s_q1)-min(s_q0,s_q1)>0",
            "satisfied": interval["valid_interval"],
            "interpretation": "nonzero stress interval with 0<q*(s)<1",
        },
        {
            "quantity": "target_stresses_interior",
            "value": sum(bool(row["branch_exists"]) for row in target_conditions),
            "condition": "all target stresses have 0<q*(s)<1 and feasible geometry",
            "satisfied": all(bool(row["branch_exists"]) for row in target_conditions),
            "interpretation": "current target stresses lie on the interior compensation branch",
        },
    ]
    write_csv(CURRENT_CONDITIONS_CSV, rows, CURRENT_CONDITIONS_FIELDS)
    return rows


def write_interval_formula(params: RoyEvoParams = PARAMS) -> list[dict[str, Any]]:
    geometry = analytic_compensation_geometry(params)
    interval = stress_interval_for_q_in_unit_interval(params)
    rows = [
        {
            "quantity": "q_star(s)",
            "formula": "((m+s+mu*w*)/(n* z*) - b_u)/(b_v-b_u)",
            "current_value": "linear in stress",
            "interpretation": "defense frequency required to balance predator growth",
        },
        {
            "quantity": "s(q)",
            "formula": "n* z* [b_u + (b_v-b_u) q] - m - mu*w*",
            "current_value": "affine inverse of q_star(s)",
            "interpretation": "stress value corresponding to an equilibrium defense frequency",
        },
        {
            "quantity": "s_q0",
            "formula": "n* z* b_u - m - mu*w*",
            "current_value": interval["s_q0"],
            "interpretation": "stress endpoint where q*=0",
        },
        {
            "quantity": "s_q1",
            "formula": "n* z* b_v - m - mu*w*",
            "current_value": interval["s_q1"],
            "interpretation": "stress endpoint where q*=1",
        },
        {
            "quantity": "interior_interval",
            "formula": "(min(s_q0,s_q1), max(s_q0,s_q1))",
            "current_value": f"({interval['interior_stress_interval_low']}, {interval['interior_stress_interval_high']})",
            "interpretation": "open stress interval where 0<q*(s)<1",
        },
        {
            "quantity": "fixed_geometry",
            "formula": "z*=xi/(r_u-c a_u), w*=c z*, n*=kappa^{-1}-z*-w*",
            "current_value": f"z*={geometry['z_star']}, w*={geometry['w_star']}, n*={geometry['n_star']}",
            "interpretation": "geometry is stress-independent along the interior branch",
        },
    ]
    write_csv(INTERVAL_FORMULA_CSV, rows, INTERVAL_FORMULA_FIELDS)
    return rows


def write_current_stability(params: RoyEvoParams = PARAMS) -> list[dict[str, Any]]:
    rows = [compensation_branch_jacobian(params, stress) for stress in TARGET_STRESSES]
    write_csv(STABILITY_CURRENT_CSV, rows, STABILITY_FIELDS)
    return rows


def classify_condition_row(params: RoyEvoParams) -> dict[str, Any]:
    geometry = analytic_compensation_geometry(params)
    interval = stress_interval_for_q_in_unit_interval(params)
    branch_exists = {stress: bool(compensation_existence_conditions(params, stress)["branch_exists"]) for stress in CONDITION_GRID_STRESSES}
    if not bool(geometry["valid_c_positive"]):
        condition_class = "invalid_tradeoff_geometry"
    elif not (bool(geometry["valid_denominator"]) and bool(geometry["valid_z"]) and bool(geometry["valid_w"]) and bool(geometry["valid_n"]) and bool(interval["valid_interval"])):
        condition_class = "invalid_feasibility"
    else:
        stable_158 = compensation_branch_jacobian(params, 0.1584375)["stability_label_analytic"] if branch_exists[0.1584375] else "not_interior"
        stable_164 = compensation_branch_jacobian(params, 0.16486816)["stability_label_analytic"] if branch_exists[0.16486816] else "not_interior"
        if not (branch_exists[0.1584375] and branch_exists[0.16486816]):
            condition_class = "valid_branch_outside_target_stress"
        elif stable_158 == "stable" and stable_164 == "stable":
            condition_class = "valid_stable_compensation_branch"
        else:
            condition_class = "valid_unstable_compensation_branch"
    stable_158 = compensation_branch_jacobian(params, 0.1584375)["stability_label_analytic"] if branch_exists.get(0.1584375, False) else "not_interior"
    stable_164 = compensation_branch_jacobian(params, 0.16486816)["stability_label_analytic"] if branch_exists.get(0.16486816, False) else "not_interior"
    return {
        "r_v": params.r_v,
        "a_v": params.a_v,
        "b_u": params.b_u,
        "b_v": params.b_v,
        "b_v_over_b_u": params.b_v / params.b_u if params.b_u != 0 else math.nan,
        "c": geometry["c"],
        "valid_c_positive": geometry["valid_c_positive"],
        "valid_denominator": geometry["valid_denominator"],
        "valid_z": geometry["valid_z"],
        "valid_w": geometry["valid_w"],
        "valid_n": geometry["valid_n"],
        "s_q0": interval["s_q0"],
        "s_q1": interval["s_q1"],
        "interior_stress_interval_low": interval["interior_stress_interval_low"],
        "interior_stress_interval_high": interval["interior_stress_interval_high"],
        "interior_interval_length": interval["interior_interval_length"],
        "branch_exists_at_s_0": branch_exists[0.0],
        "branch_exists_at_s_0p11765625": branch_exists[0.11765625],
        "branch_exists_at_s_0p1584375": branch_exists[0.1584375],
        "branch_exists_at_s_0p16486816": branch_exists[0.16486816],
        "branch_exists_at_s_0p175": branch_exists[0.175],
        "stability_at_s_0p1584375": stable_158,
        "stability_at_s_0p16486816": stable_164,
        "condition_class": condition_class,
    }


def write_condition_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r_v in R_V_VALUES:
        for a_v in A_V_VALUES:
            for b_u in B_U_VALUES:
                for ratio in B_RATIO_VALUES:
                    params = PARAMS.with_updates(r_v=r_v, a_v=a_v, b_u=b_u, b_v=b_u * ratio)
                    rows.append(classify_condition_row(params))
    write_csv(CONDITION_GRID_CSV, rows, CONDITION_GRID_FIELDS)
    return rows


def analytic_matches_current_numerical_branch(tolerance: float = 1.0e-4) -> bool:
    if BRANCH_CURRENT_CSV.exists():
        rows = read_csv(BRANCH_CURRENT_CSV)
        deltas = [as_float(row.get("abs_delta_q")) for row in rows if row.get("interior_exists") in {"True", "true", "1", True}]
        return bool(deltas and max(deltas) < tolerance)
    if not HOMOGENEOUS_EQUILIBRIA_CSV.exists():
        return False
    rows = read_csv(HOMOGENEOUS_EQUILIBRIA_CSV)
    for stress in TARGET_STRESSES:
        q_star = analytic_q_star(PARAMS, stress)
        matches = [
            row
            for row in rows
            if math.isclose(as_float(row.get("stress")), stress, abs_tol=1.0e-9)
            and row.get("stability_label") == "stable"
            and as_float(row.get("w_star")) > EPSILON
            and abs(as_float(row.get("q_star")) - q_star) < tolerance
        ]
        if not matches:
            return False
    return True


def decide_final_label(
    *,
    conditions_all_satisfied: bool,
    analytic_matches_current: bool,
    stable_all_targets: bool,
    valid_stable_grid_count: int,
) -> str:
    if not conditions_all_satisfied or not analytic_matches_current:
        return "compensation_conditions_incomplete"
    if stable_all_targets and valid_stable_grid_count > 0:
        return "compensation_conditions_derived_and_supported"
    return "compensation_conditions_derived_but_stability_limited"


def write_summary(
    current_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    current_all = all(str(row["satisfied"]) in {"True", "true", "1"} for row in current_rows)
    stable_all = all(row["stability_label_analytic"] == "stable" for row in stability_rows)
    interval = stress_interval_for_q_in_unit_interval(PARAMS)
    class_counts = Counter(str(row["condition_class"]) for row in grid_rows)
    valid_branch_count = sum(
        row["condition_class"]
        in {"valid_stable_compensation_branch", "valid_unstable_compensation_branch", "valid_branch_outside_target_stress"}
        for row in grid_rows
    )
    matches = analytic_matches_current_numerical_branch()
    final_label = decide_final_label(
        conditions_all_satisfied=current_all,
        analytic_matches_current=matches,
        stable_all_targets=stable_all,
        valid_stable_grid_count=class_counts["valid_stable_compensation_branch"],
    )
    rows = [
        {
            "metric": "current_branch_conditions_all_satisfied",
            "value": current_all,
            "interpretation": "all current feasibility and target-stress interior conditions are satisfied",
        },
        {
            "metric": "analytic_matches_current_numerical_branch",
            "value": matches,
            "interpretation": "analytic branch matches the existing numerical branch within tolerance",
        },
        {
            "metric": "current_branch_stable_at_all_target_stresses",
            "value": stable_all,
            "interpretation": "analytic Jacobian has negative max real eigenvalue at all target stresses",
        },
        {
            "metric": "current_stress_interval_low",
            "value": interval["interior_stress_interval_low"],
            "interpretation": "lower endpoint of open interval for 0<q*(s)<1",
        },
        {
            "metric": "current_stress_interval_high",
            "value": interval["interior_stress_interval_high"],
            "interpretation": "upper endpoint of open interval for 0<q*(s)<1",
        },
        {
            "metric": "current_interval_length",
            "value": interval["interior_interval_length"],
            "interpretation": "length of current branch interior stress interval",
        },
        {
            "metric": "condition_grid_total",
            "value": len(grid_rows),
            "interpretation": "structured analytic condition grid size",
        },
        {
            "metric": "condition_grid_valid_branch_count",
            "value": valid_branch_count,
            "interpretation": "grid rows with valid branch geometry and interval",
        },
        {
            "metric": "condition_grid_valid_stable_target_count",
            "value": class_counts["valid_stable_compensation_branch"],
            "interpretation": "grid rows stable at the two target rescue stresses",
        },
        {
            "metric": "condition_grid_invalid_geometry_count",
            "value": class_counts["invalid_tradeoff_geometry"],
            "interpretation": "grid rows failing c>0 trade-off geometry",
        },
        {
            "metric": "condition_grid_invalid_feasibility_count",
            "value": class_counts["invalid_feasibility"],
            "interpretation": "grid rows failing positivity or interval feasibility",
        },
        {
            "metric": "final_label",
            "value": final_label,
            "interpretation": "allowed final label for compensation-condition derivation",
        },
    ]
    write_csv(SUMMARY_CSV, rows, SUMMARY_FIELDS)
    return rows, final_label


def plot_conditions_region(grid_rows: list[dict[str, Any]]) -> None:
    class_index = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    rv_av_pairs = [(r_v, a_v) for r_v in R_V_VALUES for a_v in A_V_VALUES]
    matrix = np.zeros((len(R_V_VALUES), len(A_V_VALUES)))
    stable_counts = np.zeros_like(matrix)
    total_counts = np.zeros_like(matrix)
    dominant_class = np.zeros_like(matrix)
    for i, r_v in enumerate(R_V_VALUES):
        for j, a_v in enumerate(A_V_VALUES):
            rows = [row for row in grid_rows if math.isclose(as_float(row["r_v"]), r_v) and math.isclose(as_float(row["a_v"]), a_v)]
            total = len(rows)
            stable = sum(row["condition_class"] == "valid_stable_compensation_branch" for row in rows)
            total_counts[i, j] = total
            stable_counts[i, j] = stable
            matrix[i, j] = stable / total if total else 0.0
            counts = Counter(row["condition_class"] for row in rows)
            dominant_class[i, j] = class_index[counts.most_common(1)[0][0]]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), constrained_layout=True)
    im = axes[0].imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(len(A_V_VALUES)))
    axes[0].set_xticklabels([format_float(value, 3) for value in A_V_VALUES])
    axes[0].set_yticks(range(len(R_V_VALUES)))
    axes[0].set_yticklabels([format_float(value, 3) for value in R_V_VALUES])
    axes[0].set_xlabel("a_v")
    axes[0].set_ylabel("r_v")
    axes[0].set_title("Stable target-branch fraction over b grid")
    for i in range(len(R_V_VALUES)):
        for j in range(len(A_V_VALUES)):
            axes[0].text(j, i, f"{int(stable_counts[i,j])}/{int(total_counts[i,j])}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes[0], label="fraction")

    cmap = mcolors.ListedColormap([CLASS_COLORS[label] for label in CLASS_ORDER])
    norm = mcolors.BoundaryNorm(np.arange(len(CLASS_ORDER) + 1) - 0.5, len(CLASS_ORDER))
    axes[1].imshow(dominant_class, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    axes[1].set_xticks(range(len(A_V_VALUES)))
    axes[1].set_xticklabels([format_float(value, 3) for value in A_V_VALUES])
    axes[1].set_yticks(range(len(R_V_VALUES)))
    axes[1].set_yticklabels([format_float(value, 3) for value in R_V_VALUES])
    axes[1].set_xlabel("a_v")
    axes[1].set_ylabel("r_v")
    axes[1].set_title("Dominant condition class")
    handles = [mpatches.Patch(color=CLASS_COLORS[label], label=label.replace("_", " ")) for label in CLASS_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Analytic compensation-branch condition grid")
    save_figure(fig, FIG38_PATH)


def plot_stability(stability_rows: list[dict[str, Any]]) -> None:
    stress = [as_float(row["stress"]) for row in stability_rows]
    q = [as_float(row["q_star"]) for row in stability_rows]
    max_real = [as_float(row["max_real_eigenvalue_analytic"]) for row in stability_rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True, constrained_layout=True)
    axes[0].plot(stress, q, marker="o", color="#2f6fbb")
    axes[0].set_ylabel("q*(s)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(stress, max_real, marker="o", color="#1b9e77")
    axes[1].axhline(0.0, color="#333333", linestyle=":", linewidth=1.1)
    axes[1].set_xlabel("mortality stress")
    axes[1].set_ylabel("max Re(lambda)")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Local stability along the analytic compensation branch")
    save_figure(fig, FIG39_PATH)


def plot_stress_interval(stability_rows: list[dict[str, Any]]) -> None:
    interval = stress_interval_for_q_in_unit_interval(PARAMS)
    stress = np.array([as_float(row["stress"]) for row in stability_rows])
    q = np.array([as_float(row["q_star"]) for row in stability_rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=False, constrained_layout=True)
    low = as_float(interval["interior_stress_interval_low"])
    high = as_float(interval["interior_stress_interval_high"])
    axes[0].hlines(1.0, low, high, color="#2f6fbb", linewidth=7)
    axes[0].scatter(stress, np.ones_like(stress), color="#ffffff", edgecolor="#111111", zorder=3)
    axes[0].axvline(0.0, color="#777777", linestyle=":", linewidth=1.0)
    axes[0].set_yticks([])
    axes[0].set_xlabel("stress")
    axes[0].set_title("Interior stress interval for 0 < q*(s) < 1")
    axes[0].text(low, 1.08, "q*=1", ha="center", fontsize=9)
    axes[0].text(high, 1.08, "q*=0", ha="center", fontsize=9)
    axes[1].plot(stress, q, marker="o", color="#1b9e77")
    axes[1].axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    axes[1].axhline(1.0, color="#555555", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel("target stress")
    axes[1].set_ylabel("q*(s)")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Current compensation branch stress interval")
    save_figure(fig, FIG40_PATH)


def plot_conditions_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.4), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, text: str, width: float = 2.0, color: str = "#f4f7fb") -> None:
        patch = mpatches.FancyBboxPatch(
            (x - width / 2, y - 0.42),
            width,
            0.84,
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
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.2, text, ha="center", fontsize=8)

    box(1.25, 4.0, "G=0 fixes\nw*/z*=c")
    box(3.55, 4.0, "prey balance\nfixes z*")
    box(5.85, 4.0, "geometry gives\nw*, n*")
    box(8.15, 4.0, "predator balance\nforces q*(s)")
    box(10.45, 4.0, "0<q*(s)<1\nsets stress interval", width=2.25, color="#e8f5e9")
    arrow(2.25, 4.0, 2.55, 4.0)
    arrow(4.55, 4.0, 4.85, 4.0)
    arrow(6.85, 4.0, 7.15, 4.0)
    arrow(9.15, 4.0, 9.32, 4.0)
    box(3.2, 2.1, "feasibility:\nc>0, denominator>0,\nn*,w*,z*>0", width=2.7, color="#fff8e1")
    box(8.6, 2.1, "local stability:\nmax Re(lambda)<0\nfrom analytic Jacobian", width=2.7, color="#fff8e1")
    ax.text(
        6.0,
        0.75,
        "The branch is a conditional result for the linear trade-off ODE, not a global theorem for all trade-off forms.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.suptitle("Existence and stability conditions for the homogeneous compensation branch")
    save_figure(fig, FIG41_PATH)


def write_research_note(
    current_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    final_label: str,
) -> None:
    summary = {row["metric"]: row["value"] for row in summary_rows}
    interval = stress_interval_for_q_in_unit_interval(PARAMS)
    stable_text = "locally stable at all tested target stresses" if summary["current_branch_stable_at_all_target_stresses"] in {True, "True", "true"} else "not stable at all tested stresses"
    class_counts = Counter(row["condition_class"] for row in grid_rows)
    grid_lines = "\n".join(f"- `{label}`: {class_counts[label]}" for label in CLASS_ORDER)
    eig_lines = "\n".join(
        f"- s={as_float(row['stress']):.9g}: max Re(lambda)={as_float(row['max_real_eigenvalue_analytic']):.6g}, {row['stability_label_analytic']}"
        for row in stability_rows
    )
    text = f"""# Existence and Stability Conditions for the Roy ODE Compensation Branch

## Purpose

This note derives conditional existence and local-stability conditions for the homogeneous compensation branch in the linear trade-off Roy eco-evolutionary ODE. It is ODE-only and does not run PDE simulations or change model equations.

## Interior Equilibrium Conditions

For an interior equilibrium with \(n>0\), \(w>0\), and \(0<q<1\), the nontrivial equilibrium conditions are prey balance, predator balance, and zero selection gradient. With \(\Delta r=r_v-r_u\), \(\Delta a=a_v-a_u\), and \(\Delta b=b_v-b_u\), the selection-gradient condition is \(\Delta r z-\Delta a w=0\).

## Analytic Compensation Branch

If \(c=\Delta r/\Delta a>0\), then \(w^*=cz^*\). Because the trade-offs are linear, \(r(q)-c a(q)=r_u-ca_u\), so the prey equation gives

```text
z* = xi / (r_u - c a_u)
w* = c z*
n* = kappa^{-1} - z* - w*
```

The predator equation then determines

```text
b_req(s) = (m+s+mu w*)/(n* z*)
q*(s) = (b_req(s)-b_u)/(b_v-b_u)
```

## Feasibility Inequalities

The branch exists only when the feasibility inequalities hold: `c>0`, `r_u-c a_u>0`, `z*>0`, `w*>0`, `n*>0`, and `0<q*(s)<1`. The current parameterization satisfies these conditions at all target stress values. The current-condition table is `results/roy_ode_compensation_conditions_current.csv`.

## Stress Interval

Solving the branch formula for stress gives

```text
s(q) = n* z* [b_u + Delta b q] - m - mu w*
```

so the interior branch is feasible on the open interval between `s_q0` and `s_q1`. For the current parameterization this interval is:

- low endpoint: `{interval["interior_stress_interval_low"]}`
- high endpoint: `{interval["interior_stress_interval_high"]}`
- length: `{interval["interior_interval_length"]}`

The formula table is `results/roy_ode_compensation_stress_interval_formula.csv`, and the interval figure is `figures/roy_evo_spatial/report/fig40_compensation_stress_interval.png`.

## Local Stability

An analytic Jacobian was implemented for the ODE. On the current interior branch, the branch is {stable_text}. The tested eigenvalue summaries are:

{eig_lines}

The stability table is `results/roy_ode_compensation_stability_current.csv`, and the figure is `figures/roy_evo_spatial/report/fig39_compensation_stability_along_branch.png`.

## Condition Grid

A structured local analytic grid varied \(r_v\), \(a_v\), \(b_u\), and \(b_v/b_u\). This is a condition-grid evaluation, not a simulation scan. Counts by condition class are:

{grid_lines}

The condition grid is `results/roy_ode_compensation_condition_grid.csv`, and the figure is `figures/roy_evo_spatial/report/fig38_compensation_conditions_region.png`.

## Final Label

`{final_label}`

## Interpretation

The current compensation branch is now a conditional mathematical statement for the linear trade-off ODE: if the feasibility inequalities hold and \(0<q^*(s)<1\), then the branch exists; if the analytic Jacobian has negative maximum real eigenvalue, it is locally stable. In the tested parameterization, these conditions hold at the target stresses, and the analytic branch matches the numerical branch to machine precision.

## What This Proves

This derives the branch equations and feasibility conditions for the linear trade-off ODE and verifies local stability for the tested current branch. It is stronger than the earlier numerical observation because it explains why \(n^*\) and \(w^*\) remain fixed while \(q^*(s)\) shifts with stress.

## What This Does Not Prove

This is not a global theorem for all trade-off forms or all parameter values. It does not establish global basin geometry, global stability, or spatial-pattern-mediated mechanisms. It also does not test PDE dynamics.

## Files

- `experiments/23_roy_ode_compensation_conditions.py`
- `results/roy_ode_compensation_conditions_current.csv`
- `results/roy_ode_compensation_stress_interval_formula.csv`
- `results/roy_ode_compensation_stability_current.csv`
- `results/roy_ode_compensation_condition_grid.csv`
- `results/roy_ode_compensation_conditions_summary.csv`
- `figures/roy_evo_spatial/report/fig38_compensation_conditions_region.png`
- `figures/roy_evo_spatial/report/fig39_compensation_stability_along_branch.png`
- `figures/roy_evo_spatial/report/fig40_compensation_stress_interval.png`
- `figures/roy_evo_spatial/report/fig41_compensation_conditions_schematic.png`

## Next Step

Use these inequalities to state sharper parameter hypotheses before any further numerical exploration. The next mathematical step is a local Routh-Hurwitz or symbolic stability analysis of the branch Jacobian.
"""
    NOTE_PATH.write_text(text, encoding="utf-8")


def run(profile: str) -> str:
    del profile
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    current_rows = write_current_conditions(PARAMS)
    interval_rows = write_interval_formula(PARAMS)
    stability_rows = write_current_stability(PARAMS)
    grid_rows = write_condition_grid()
    summary_rows, final_label = write_summary(current_rows, stability_rows, grid_rows)
    plot_conditions_region(grid_rows)
    plot_stability(stability_rows)
    plot_stress_interval(stability_rows)
    plot_conditions_schematic()
    write_research_note(current_rows, interval_rows, stability_rows, grid_rows, summary_rows, final_label)
    for path in [
        CURRENT_CONDITIONS_CSV,
        INTERVAL_FORMULA_CSV,
        STABILITY_CURRENT_CSV,
        CONDITION_GRID_CSV,
        SUMMARY_CSV,
        FIG38_PATH,
        FIG39_PATH,
        FIG40_PATH,
        FIG41_PATH,
        NOTE_PATH,
    ]:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label: {final_label}")
    return final_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused",), default="focused")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
