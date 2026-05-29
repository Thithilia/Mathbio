#!/usr/bin/env python
"""Analyze the homogeneous Roy eco-evolutionary ODE mechanism.

This script follows PR #18's mechanism diagnosis: the mapped PDE basins are
primarily inherited from homogeneous reaction dynamics. It therefore analyzes
only the ODE/reaction system and does not run PDE scans.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

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
    find_evo_equilibrium,
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

REPRESENTATIVE_TIMESERIES_CSV = RESULTS_DIR / "roy_ode_homogeneous_representative_timeseries.csv"
BASIN_MAP_CSV = RESULTS_DIR / "roy_ode_homogeneous_basin_map.csv"
EQUILIBRIA_CSV = RESULTS_DIR / "roy_ode_homogeneous_equilibria.csv"
MECHANISM_SUMMARY_CSV = RESULTS_DIR / "roy_ode_homogeneous_mechanism_summary.csv"
PR18_DECISION_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_decision_summary.csv"

FIG28_PATH = FIG_DIR / "fig28_ode_representative_timeseries.png"
FIG29_PATH = FIG_DIR / "fig29_ode_basin_map.png"
FIG30_PATH = FIG_DIR / "fig30_selection_growth_phase.png"
FIG31_PATH = FIG_DIR / "fig31_ode_equilibria_stability.png"
FIG32_PATH = FIG_DIR / "fig32_updated_mechanism_diagram.png"
NOTE_PATH = NOTES_DIR / "roy_ode_homogeneous_mechanism.md"

REPRESENTATIVE_CASES = {
    "persistent_case": {"stress": 0.1584375, "q0": 0.1, "w0_scale": 0.1},
    "extinct_case": {"stress": 0.16486816, "q0": 0.9, "w0_scale": 0.02},
    "transient_case": {"stress": 0.1584375, "q0": 0.7, "w0_scale": 0.02},
}
CASE_ORDER = ("persistent_case", "extinct_case", "transient_case")
BASIN_STRESSES = (0.1584375, 0.16486816)
Q0_VALUES = tuple(round(0.1 * idx, 1) for idx in range(10))
W0_SCALES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5)
EQUILIBRIUM_STRESSES = (0.0, 0.069448242, 0.11765625, 0.1584375, 0.16486816, 0.175)

REPRESENTATIVE_FIELDS = [
    "case_label",
    "stress",
    "time",
    "n",
    "w",
    "q",
    "z",
    "r_q",
    "a_q",
    "b_q",
    "selection_gradient",
    "predator_growth_factor",
    "classification",
]

BASIN_FIELDS = [
    "stress",
    "q0",
    "w0_scale",
    "n0",
    "w0",
    "z0",
    "classification",
    "basin_label",
    "tail_mean_w",
    "tail_mean_q",
    "q_change",
    "selection_gradient_tail_mean",
    "predator_growth_factor_tail_mean",
    "relative_change_between_last_windows",
    "physical",
    "notes",
]

EQUILIBRIA_FIELDS = [
    "stress",
    "equilibrium_id",
    "n_star",
    "w_star",
    "q_star",
    "z_star",
    "selection_gradient",
    "predator_growth_factor",
    "max_real_eigenvalue",
    "eigenvalues_real",
    "eigenvalues_imag",
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


@dataclass(frozen=True)
class Baseline:
    n: float
    w: float
    q: float


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


def format_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def baseline_values() -> Baseline:
    eq = find_evo_equilibrium(PARAMS)
    return Baseline(float(eq["n"]), float(eq["w"]), float(eq["q"]))


def initial_mean(baseline: Baseline, q0: float, w0_scale: float) -> np.ndarray:
    return np.array([baseline.n, baseline.w * w0_scale, q0], dtype=float)


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


def state_quantities(n: np.ndarray | float, w: np.ndarray | float, q: np.ndarray | float, stress: float) -> dict[str, np.ndarray]:
    return {
        "z": np.asarray(free_space_evo(n, w, PARAMS), dtype=float),
        "r_q": np.asarray(r_of_q(q, PARAMS), dtype=float),
        "a_q": np.asarray(a_of_q(q, PARAMS), dtype=float),
        "b_q": np.asarray(b_of_q(q, PARAMS), dtype=float),
        "selection_gradient": np.asarray(selection_gradient(n, w, q, PARAMS), dtype=float),
        "predator_growth_factor": np.asarray(predator_growth_factor_evo(n, w, q, PARAMS, stress=stress), dtype=float),
    }


def selection_gradient_value(n: float, w: float, q: float, params: RoyEvoParams = PARAMS) -> float:
    return float(selection_gradient(n, w, q, params))


def ode_rhs_residual(state: np.ndarray, stress: float) -> dict[str, float]:
    rhs = reaction_ode_evo(0.0, state, PARAMS, stress=stress, evolve=True)
    rhs_norm = float(np.linalg.norm(rhs))
    state_norm = float(np.linalg.norm(state))
    return {"rhs_norm": rhs_norm, "state_norm": state_norm, "normalized_residual": rhs_norm / max(state_norm, 1.0e-12)}


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


def classify_ode_trajectory(times: np.ndarray, values: np.ndarray, stress: float) -> dict[str, Any]:
    n, w, q = values
    z = np.asarray(free_space_evo(n, w, PARAMS), dtype=float)
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
    residual = ode_rhs_residual(values[:, -1], stress)
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


def run_representative_trajectories(baseline: Baseline, profile: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    T = 1600.0 if profile == "focused" else 400.0
    n_eval = 321 if profile == "focused" else 101
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for case_label in CASE_ORDER:
        case = REPRESENTATIVE_CASES[case_label]
        stress = float(case["stress"])
        initial = initial_mean(baseline, float(case["q0"]), float(case["w0_scale"]))
        trajectory = simulate_ode_evo(PARAMS, initial, stress=stress, evolve=True, T=T, n_eval=n_eval)
        metrics = classify_ode_trajectory(trajectory.t, trajectory.y, stress)
        quantities = state_quantities(trajectory.y[0], trajectory.y[1], trajectory.y[2], stress)
        for idx, time in enumerate(trajectory.t):
            rows.append(
                {
                    "case_label": case_label,
                    "stress": stress,
                    "time": float(time),
                    "n": float(trajectory.y[0, idx]),
                    "w": float(trajectory.y[1, idx]),
                    "q": float(trajectory.y[2, idx]),
                    "z": float(quantities["z"][idx]),
                    "r_q": float(quantities["r_q"][idx]),
                    "a_q": float(quantities["a_q"][idx]),
                    "b_q": float(quantities["b_q"][idx]),
                    "selection_gradient": float(quantities["selection_gradient"][idx]),
                    "predator_growth_factor": float(quantities["predator_growth_factor"][idx]),
                    "classification": metrics["classification"],
                }
            )
        payload[case_label] = {"t": trajectory.t, "y": trajectory.y, "quantities": quantities, "classification": metrics["classification"]}
    write_csv(REPRESENTATIVE_TIMESERIES_CSV, rows, REPRESENTATIVE_FIELDS)
    return rows, payload


def run_basin_map(baseline: Baseline, profile: str) -> list[dict[str, Any]]:
    T = 1600.0 if profile == "focused" else 400.0
    n_eval = 321 if profile == "focused" else 101
    rows: list[dict[str, Any]] = []
    for stress in BASIN_STRESSES:
        for q0 in Q0_VALUES:
            for w0_scale in W0_SCALES:
                initial = initial_mean(baseline, q0, w0_scale)
                z0 = float(free_space_evo(initial[0], initial[1], PARAMS))
                if z0 < 0.0:
                    rows.append(
                        {
                            "stress": stress,
                            "q0": q0,
                            "w0_scale": w0_scale,
                            "n0": initial[0],
                            "w0": initial[1],
                            "z0": z0,
                            "classification": "nonphysical",
                            "basin_label": "unresolved_basin",
                            "physical": False,
                            "notes": "skipped_nonphysical_initial_condition",
                        }
                    )
                    continue
                trajectory = simulate_ode_evo(PARAMS, initial, stress=stress, evolve=True, T=T, n_eval=n_eval)
                metrics = classify_ode_trajectory(trajectory.t, trajectory.y, stress)
                mask = tail_mask(trajectory.t)
                quantities = state_quantities(trajectory.y[0], trajectory.y[1], trajectory.y[2], stress)
                rows.append(
                    {
                        "stress": stress,
                        "q0": q0,
                        "w0_scale": w0_scale,
                        "n0": initial[0],
                        "w0": initial[1],
                        "z0": z0,
                        "classification": metrics["classification"],
                        "basin_label": metrics["basin_label"],
                        "tail_mean_w": metrics["tail_mean_w"],
                        "tail_mean_q": metrics["tail_mean_q"],
                        "q_change": float(metrics["tail_mean_q"] - q0),
                        "selection_gradient_tail_mean": float(np.mean(quantities["selection_gradient"][mask])),
                        "predator_growth_factor_tail_mean": float(np.mean(quantities["predator_growth_factor"][mask])),
                        "relative_change_between_last_windows": metrics["relative_change_between_last_windows"],
                        "physical": metrics["physical"],
                        "notes": row_note(str(metrics["classification"])),
                    }
                )
    write_csv(BASIN_MAP_CSV, rows, BASIN_FIELDS)
    return rows


def row_note(classification: str) -> str:
    if classification.endswith("_steady"):
        return "steady_by_tail_change_and_rhs_residual"
    if "transient" in classification:
        return "window_change_indicates_transient"
    return classification


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


def raw_equilibrium_guesses(baseline: Baseline) -> list[np.ndarray]:
    carrying = 1.0 / PARAMS.kappa
    q0_n = carrying - PARAMS.xi / PARAMS.r_u
    q1_n = carrying - PARAMS.xi / PARAMS.r_v
    guesses = [
        np.array([baseline.n, baseline.w, baseline.q]),
        np.array([baseline.n, 1.0e-8, baseline.q]),
        np.array([baseline.n, baseline.w * 1.8, baseline.q]),
        np.array([baseline.n, baseline.w, 0.05]),
        np.array([baseline.n, baseline.w, 0.95]),
        np.array([q0_n, 1.0e-8, 0.0]),
        np.array([q1_n, 1.0e-8, 1.0]),
        np.array([max(q0_n, 1.0e-8), 1.0e-6, 0.02]),
        np.array([max(q1_n, 1.0e-8), 1.0e-6, 0.98]),
        np.array([1.0e-8, 1.0e-8, 0.0]),
        np.array([1.0e-8, 1.0e-8, 1.0]),
        np.array([baseline.n * 0.75, baseline.w * 0.05, 0.2]),
        np.array([baseline.n * 1.1, baseline.w * 0.05, 0.8]),
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


def stability_label(max_real: float, physical: bool) -> str:
    if not physical:
        return "nonphysical"
    if max_real < -1.0e-6:
        return "stable"
    if max_real > 1.0e-6:
        return "unstable"
    return "marginal_or_unresolved"


def equilibrium_physical(state: np.ndarray) -> bool:
    n, w, q = state
    z = float(free_space_evo(n, w, PARAMS))
    return bool(np.all(np.isfinite(state)) and n >= -1.0e-8 and w >= -1.0e-8 and -1.0e-6 <= q <= 1.0 + 1.0e-6 and z >= -1.0e-6)


def find_equilibria_for_stress(stress: float, baseline: Baseline) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0 / PARAMS.kappa, 1.0 / PARAMS.kappa, 1.0])

    def rhs(state: np.ndarray) -> np.ndarray:
        return reaction_ode_evo(0.0, state, PARAMS, stress=stress, evolve=True)

    for guess in raw_equilibrium_guesses(baseline):
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
        physical = equilibrium_physical(state)
        jac = finite_difference_jacobian(rhs, state)
        eigenvalues = np.linalg.eigvals(jac)
        max_real = float(np.max(np.real(eigenvalues)))
        q = float(state[2])
        quantities = state_quantities(float(state[0]), float(state[1]), q, stress)
        rows.append(
            {
                "stress": stress,
                "n_star": float(state[0]),
                "w_star": float(state[1]),
                "q_star": q,
                "z_star": float(quantities["z"]),
                "selection_gradient": float(quantities["selection_gradient"]),
                "predator_growth_factor": float(quantities["predator_growth_factor"]),
                "max_real_eigenvalue": max_real,
                "eigenvalues_real": ";".join(format_float(float(value), 8) for value in np.real(eigenvalues)),
                "eigenvalues_imag": ";".join(format_float(float(value), 8) for value in np.imag(eigenvalues)),
                "stability_label": stability_label(max_real, physical),
                "physical": physical,
                "notes": f"least_squares_residual={residual_norm:.3e}",
                "residual_norm": residual_norm,
            }
        )
    unique = deduplicate_equilibria(rows)
    for idx, row in enumerate(unique, start=1):
        row["equilibrium_id"] = f"s{stress:.9g}_eq{idx:02d}".replace(".", "p")
    return unique


def run_equilibrium_scan(baseline: Baseline) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stress in EQUILIBRIUM_STRESSES:
        rows.extend(find_equilibria_for_stress(stress, baseline))
    write_csv(EQUILIBRIA_CSV, rows, EQUILIBRIA_FIELDS)
    return rows


def mechanism_label_from_inputs(
    *,
    basin_counts: dict[float, Counter[str]],
    ode_pde_agreement_fraction: float,
    representative_classes: set[str],
    stable_persistent_equilibria: int,
    stable_extinct_equilibria: int,
) -> str:
    basin_structure = all(
        counts.get("persistent_basin", 0) > 0 and counts.get("extinct_basin", 0) > 0 and counts.get("transient_basin", 0) > 0
        for counts in basin_counts.values()
    )
    representative_structure = {"persistent_steady", "extinct_steady"} <= representative_classes and any(
        "transient" in label for label in representative_classes
    )
    if basin_structure and ode_pde_agreement_fraction >= 0.85 and representative_structure and stable_persistent_equilibria > 0 and stable_extinct_equilibria > 0:
        return "ode_homogeneous_basin_structure_supported"
    if basin_structure and ode_pde_agreement_fraction >= 0.85 and representative_structure:
        return "ode_homogeneous_mechanism_partially_supported"
    return "ode_homogeneous_mechanism_unresolved"


def write_mechanism_summary(
    representative_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    equilibrium_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    basin_counts: dict[float, Counter[str]] = defaultdict(Counter)
    for row in basin_rows:
        basin_counts[as_float(row["stress"])][str(row["basin_label"])] += 1
    decision = read_csv(PR18_DECISION_CSV)[0] if PR18_DECISION_CSV.exists() else {}
    ode_pde_agreement = as_float(decision.get("basin_grid_agreement_fraction"), 0.0)
    representative_classes = {str(row["classification"]) for row in representative_rows}
    stable_persistent = sum(row["stability_label"] == "stable" and as_float(row["w_star"]) > EPSILON for row in equilibrium_rows)
    stable_extinct = sum(row["stability_label"] == "stable" and as_float(row["w_star"]) <= EPSILON for row in equilibrium_rows)
    label = mechanism_label_from_inputs(
        basin_counts=basin_counts,
        ode_pde_agreement_fraction=ode_pde_agreement,
        representative_classes=representative_classes,
        stable_persistent_equilibria=stable_persistent,
        stable_extinct_equilibria=stable_extinct,
    )
    rows: list[dict[str, Any]] = [
        {"metric": "number_of_stress_values_analyzed", "value": len(EQUILIBRIUM_STRESSES), "interpretation": "targeted ODE equilibrium stresses"},
        {"metric": "number_of_equilibria_found", "value": len(equilibrium_rows), "interpretation": "unique numerical equilibria across target stresses"},
        {"metric": "number_of_stable_persistent_equilibria", "value": stable_persistent, "interpretation": "stable equilibria with predator abundance above epsilon"},
        {"metric": "number_of_stable_extinct_or_predator_free_equilibria", "value": stable_extinct, "interpretation": "stable equilibria with predator abundance at or below epsilon"},
        {"metric": "ode_pde_basin_agreement_fraction_from_pr18", "value": ode_pde_agreement, "interpretation": "PR18 high agreement supports ODE control of PDE basin labels"},
    ]
    for stress in sorted(basin_counts):
        counts = basin_counts[stress]
        rows.extend(
            [
                {"metric": f"basin_map_persistent_count_stress_{stress:.9g}", "value": counts.get("persistent_basin", 0), "interpretation": "ODE q0-w0 basin map persistent count"},
                {"metric": f"basin_map_extinct_count_stress_{stress:.9g}", "value": counts.get("extinct_basin", 0), "interpretation": "ODE q0-w0 basin map extinct count"},
                {"metric": f"basin_map_transient_count_stress_{stress:.9g}", "value": counts.get("transient_basin", 0), "interpretation": "ODE q0-w0 basin map transient count"},
            ]
        )
    rows.append({"metric": "qualitative_mechanism_label", "value": label, "interpretation": "conditional label for homogeneous ODE mechanism evidence"})
    write_csv(MECHANISM_SUMMARY_CSV, rows, SUMMARY_FIELDS)
    return rows, label


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_timeseries(payload: dict[str, Any]) -> None:
    colors = {"persistent_case": "#1b9e77", "extinct_case": "#d95f02", "transient_case": "#7570b3"}
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 9.6), sharex=True, constrained_layout=True)
    panels = [
        ("w", "w(t)"),
        ("q", "q(t)"),
        ("selection_gradient", "selection gradient"),
        ("predator_growth_factor", "predator growth factor"),
    ]
    for case_label in CASE_ORDER:
        data = payload[case_label]
        t = data["t"]
        y = data["y"]
        quantities = data["quantities"]
        series = {
            "w": y[1],
            "q": y[2],
            "selection_gradient": quantities["selection_gradient"],
            "predator_growth_factor": quantities["predator_growth_factor"],
        }
        for ax, (key, ylabel) in zip(axes, panels, strict=True):
            ax.plot(t, series[key], label=case_label.replace("_", " "), color=colors[case_label], linewidth=1.6)
            ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Homogeneous ODE representative mechanism trajectories")
    save_figure(fig, FIG28_PATH)


def plot_basin_map(rows: list[dict[str, Any]]) -> None:
    q_values = sorted({as_float(row["q0"]) for row in rows})
    w_values = sorted({as_float(row["w0_scale"]) for row in rows})
    stresses = sorted({as_float(row["stress"]) for row in rows})
    category_index = {label: idx for idx, label in enumerate(BASIN_LABELS)}
    cmap = mcolors.ListedColormap([BASIN_COLORS[label] for label in BASIN_LABELS])
    norm = mcolors.BoundaryNorm(np.arange(len(BASIN_LABELS) + 1) - 0.5, len(BASIN_LABELS))
    fig, axes = plt.subplots(1, len(stresses), figsize=(11.6, 5.0), sharey=True, constrained_layout=True)
    if len(stresses) == 1:
        axes = [axes]
    for ax, stress in zip(axes, stresses, strict=True):
        matrix = np.full((len(w_values), len(q_values)), category_index["unresolved_basin"])
        for row in rows:
            if not math.isclose(as_float(row["stress"]), stress):
                continue
            x = q_values.index(as_float(row["q0"]))
            y = w_values.index(as_float(row["w0_scale"]))
            matrix[y, x] = category_index.get(str(row["basin_label"]), category_index["unresolved_basin"])
        ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"stress = {stress:.9g}")
        ax.set_xticks(range(len(q_values)))
        ax.set_xticklabels([format_float(value, 3) for value in q_values], rotation=45, ha="right")
        ax.set_yticks(range(len(w_values)))
        ax.set_yticklabels([format_float(value, 3) for value in w_values])
        ax.set_xlabel("initial defense frequency q0")
        ax.grid(color="white", linewidth=0.7)
    axes[0].set_ylabel("initial predator scale w0_scale")
    handles = [mpatches.Patch(color=BASIN_COLORS[label], label=label.replace("_", " ")) for label in BASIN_LABELS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Homogeneous ODE q0-w0 basin map")
    save_figure(fig, FIG29_PATH)


def plot_selection_growth_phase(rows: list[dict[str, Any]]) -> None:
    colors = {"persistent_case": "#1b9e77", "extinct_case": "#d95f02", "transient_case": "#7570b3"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    for case_label in CASE_ORDER:
        case_rows = [row for row in rows if row["case_label"] == case_label]
        q = [as_float(row["q"]) for row in case_rows]
        gradient = [as_float(row["selection_gradient"]) for row in case_rows]
        growth = [as_float(row["predator_growth_factor"]) for row in case_rows]
        axes[0].plot(q, gradient, color=colors[case_label], linewidth=1.5, label=case_label.replace("_", " "))
        axes[1].plot(q, growth, color=colors[case_label], linewidth=1.5, label=case_label.replace("_", " "))
        axes[0].scatter([q[0], q[-1]], [gradient[0], gradient[-1]], color=colors[case_label], edgecolor="black", s=28)
        axes[1].scatter([q[0], q[-1]], [growth[0], growth[-1]], color=colors[case_label], edgecolor="black", s=28)
    axes[0].axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    axes[1].axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("defense frequency q")
    axes[1].set_xlabel("defense frequency q")
    axes[0].set_ylabel("selection gradient")
    axes[1].set_ylabel("predator growth factor")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("Selection-gradient and predator-growth trajectories in homogeneous ODE")
    save_figure(fig, FIG30_PATH)


def plot_equilibria(rows: list[dict[str, Any]]) -> None:
    colors = {"stable": "#1b9e77", "unstable": "#d95f02", "marginal_or_unresolved": "#7570b3", "nonphysical": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for row in rows:
        label = str(row["stability_label"])
        stress = as_float(row["stress"])
        axes[0].scatter(stress, as_float(row["w_star"]), color=colors.get(label, "#777777"), edgecolor="black", s=58)
        axes[1].scatter(stress, as_float(row["q_star"]), color=colors.get(label, "#777777"), edgecolor="black", s=58)
    axes[0].set_ylabel("equilibrium predator density w*")
    axes[1].set_ylabel("equilibrium defense frequency q*")
    for ax in axes:
        ax.set_xlabel("stress")
        ax.grid(alpha=0.25)
    handles = [mpatches.Patch(color=color, label=label.replace("_", " ")) for label, color in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Numerical homogeneous ODE equilibria and stability")
    save_figure(fig, FIG31_PATH)


def plot_updated_mechanism_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, text: str, width: float = 1.7, height: float = 0.72, color: str = "#f4f7fb") -> None:
        patch = mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
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
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, text, ha="center", fontsize=8)

    box(1.0, 3.5, "mortality\nstress", color="#f7e7da")
    box(2.7, 3.5, "reduced\npredator pressure")
    box(4.4, 3.5, "selection\non q")
    box(6.1, 3.5, "lower defense\nor boundary q")
    box(7.9, 3.5, "changed prey\ngrowth and conversion")
    box(9.2, 4.45, "persistent\nbasin", color="#dfecfb")
    box(9.2, 2.55, "extinct/transient\nbasin", color="#f9e1d9")
    box(4.9, 1.25, "spatial PDE:\npreserves basin structure\nbut fields remain nearly homogeneous", width=4.2, height=1.0, color="#edf4ea")

    arrow(1.85, 3.5, 1.95, 3.5)
    arrow(3.55, 3.5, 3.55, 3.5)
    arrow(5.25, 3.5, 5.25, 3.5)
    arrow(6.95, 3.5, 7.0, 3.5)
    arrow(8.62, 3.65, 8.65, 4.25, "positive tail growth")
    arrow(8.62, 3.35, 8.65, 2.75, "negative or delayed tail growth")
    arrow(4.9, 2.75, 4.9, 1.8)
    ax.text(5.0, 5.35, "Current mechanism diagnosis: reaction-dominated homogeneous multistability", ha="center", fontsize=13, fontweight="bold")
    save_figure(fig, FIG32_PATH)


def write_research_note(summary_rows: list[dict[str, Any]], mechanism_label: str) -> None:
    summary = {row["metric"]: row for row in summary_rows}
    note = [
        "# Homogeneous Eco-Evolutionary Mechanism Behind the Roy Rescue Basins",
        "",
        "## Purpose",
        "",
        "This note analyzes the homogeneous eco-evolutionary ODE mechanism that appears to control basin-dependent outcomes in the Roy rescue model.",
        "",
        "## Why this follows from the ODE-PDE mechanism test",
        "",
        "PR #18 showed the current basin dependence is reaction-dominated homogeneous multistability, not spatial-pattern-mediated rescue. Representative ODE and PDE classifications agreed 3/3, ODE-PDE basin labels agreed for 90 percent of the q0-w0 grid, all disagreements involved transient labels, direct persistent/extinct disagreements were zero, final spatial CV values were very small, and perturbation outcome changes were zero.",
        "",
        "## Model and selection-gradient logic",
        "",
        "The ODE uses total prey density n, predator density w, defense frequency q, and free space z = kappa^{-1} - n - w. The selection gradient is `(r_v-r_u) z - (a_v-a_u) w`, and the predator growth factor is `b(q) n z - (m+s) - mu w`.",
        "",
        "## Representative ODE trajectories",
        "",
        "The representative ODE trajectories are saved in `results/roy_ode_homogeneous_representative_timeseries.csv` and plotted in `figures/roy_evo_spatial/report/fig28_ode_representative_timeseries.png`.",
        "",
        "## ODE q0-w0 basin map",
        "",
        "The ODE basin map uses the same q0-w0 grid and target stresses as the PR #18 comparison. It is saved in `results/roy_ode_homogeneous_basin_map.csv` and plotted in `figures/roy_evo_spatial/report/fig29_ode_basin_map.png`.",
        "",
        "## Equilibria and stability",
        "",
        f"The numerical equilibrium scan found {summary['number_of_equilibria_found']['value']} unique equilibria across {summary['number_of_stress_values_analyzed']['value']} target stresses. This is numerical evidence only, not an analytical proof. The table is `results/roy_ode_homogeneous_equilibria.csv`, and the stability figure is `figures/roy_evo_spatial/report/fig31_ode_equilibria_stability.png`.",
        "",
        "## Mechanism interpretation",
        "",
        f"Qualitative ODE mechanism label: `{mechanism_label}`.",
        "",
    ]
    if mechanism_label == "ode_homogeneous_basin_structure_supported":
        note.append("The current evidence supports the interpretation that the observed PDE basin structure is inherited from the homogeneous eco-evolutionary reaction system.")
    elif mechanism_label == "ode_homogeneous_mechanism_partially_supported":
        note.append("The current evidence supports a homogeneous ODE basin-structure interpretation, but the numerical equilibrium/stability evidence remains incomplete.")
    else:
        note.append("The homogeneous ODE mechanism remains unresolved under the current diagnostics.")
    note.extend(
        [
            "",
            "## What this means for the spatial PDE",
            "",
            "The spatially extended PDE preserves basin-dependent outcomes, but persistent spatial patterning is not currently supported as the mechanism. The PDE remains useful as a spatially extended test of the reaction dynamics rather than as evidence for spatial-pattern-mediated rescue.",
            "",
            "## What is still not general",
            "",
            "This conclusion is limited to `RoyEvoParams(b_u=0.08, b_v=0.02)` and the tested diffusion and perturbation settings. It does not establish general behavior across trade-off forms, evolutionary rates, diffusion coefficients, or broader parameter regions.",
            "",
            "## Files",
            "",
            "- `experiments/21_roy_ode_homogeneous_mechanism.py`",
            "- `results/roy_ode_homogeneous_representative_timeseries.csv`",
            "- `results/roy_ode_homogeneous_basin_map.csv`",
            "- `results/roy_ode_homogeneous_equilibria.csv`",
            "- `results/roy_ode_homogeneous_mechanism_summary.csv`",
            "- `figures/roy_evo_spatial/report/fig28_ode_representative_timeseries.png`",
            "- `figures/roy_evo_spatial/report/fig29_ode_basin_map.png`",
            "- `figures/roy_evo_spatial/report/fig30_selection_growth_phase.png`",
            "- `figures/roy_evo_spatial/report/fig31_ode_equilibria_stability.png`",
            "- `figures/roy_evo_spatial/report/fig32_updated_mechanism_diagram.png`",
            "",
            "## Next step",
            "",
            "Correct manuscript language first, then test robustness of the homogeneous basin structure across trade-off parameters and refine the analytical equilibrium/stability interpretation.",
            "",
        ]
    )
    NOTE_PATH.write_text("\n".join(note), encoding="utf-8")


def run(profile: str) -> str:
    baseline = baseline_values()
    representative_rows, representative_payload = run_representative_trajectories(baseline, profile)
    basin_rows = run_basin_map(baseline, profile)
    equilibrium_rows = run_equilibrium_scan(baseline)
    summary_rows, mechanism_label = write_mechanism_summary(representative_rows, basin_rows, equilibrium_rows)
    plot_representative_timeseries(representative_payload)
    plot_basin_map(basin_rows)
    plot_selection_growth_phase(representative_rows)
    plot_equilibria(equilibrium_rows)
    plot_updated_mechanism_diagram()
    write_research_note(summary_rows, mechanism_label)
    for path in [
        REPRESENTATIVE_TIMESERIES_CSV,
        BASIN_MAP_CSV,
        EQUILIBRIA_CSV,
        MECHANISM_SUMMARY_CSV,
        FIG28_PATH,
        FIG29_PATH,
        FIG30_PATH,
        FIG31_PATH,
        FIG32_PATH,
        NOTE_PATH,
    ]:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Mechanism label: {mechanism_label}")
    return mechanism_label


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused", "minimal"), default="focused")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.profile)


if __name__ == "__main__":
    main()
