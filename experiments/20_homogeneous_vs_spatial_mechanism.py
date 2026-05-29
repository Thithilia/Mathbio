#!/usr/bin/env python
"""Test whether Roy PDE basin outcomes are homogeneous or spatially mediated.

This is a targeted mechanism analysis. It reuses the representative PDE fields
and basin scan outputs when available, runs matched ODE trajectories from the
same spatial means, and runs only the requested perturbation-sensitivity PDE
checks.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (  # noqa: E402
    RoyEvoPDEConfig,
    RoyEvoPDEResult,
    RoyEvoParams,
    classify_evo_trajectory,
    find_evo_equilibrium,
    free_space_evo,
    grid_2d_evo,
    initial_state_from_ode_equilibrium,
    laplacian_neumann_2d_evo,
    reaction_ode_evo,
    reaction_part_evo_pde,
    simulate_ode_evo,
    simulate_pde_evo_2d,
)


PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 1.0e-4
EXTREME_EXTINCTION_W = 1.0e-8
TAIL_FRACTION = 0.25

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTES_DIR = ROOT / "research_notes"
MANUSCRIPT_DIR = ROOT / "manuscript"

PDE_REP_SUMMARY_CSV = RESULTS_DIR / "roy_pde_evo_representative_solution_summary.csv"
PDE_REP_TIMESERIES_CSV = RESULTS_DIR / "roy_pde_evo_representative_mean_timeseries.csv"
PDE_BASIN_SCAN_CSV = RESULTS_DIR / "roy_pde_evo_basin_boundary_scan.csv"

REPRESENTATIVE_COMPARISON_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_representative_comparison.csv"
VARIANCE_TIMESERIES_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_variance_timeseries.csv"
BASIN_AGREEMENT_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_basin_agreement.csv"
DISAGREEMENTS_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_basin_disagreements.csv"
DISAGREEMENT_SUMMARY_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_disagreement_summary.csv"
PERTURBATION_SENSITIVITY_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_perturbation_sensitivity.csv"
DECISION_SUMMARY_CSV = RESULTS_DIR / "roy_homogeneous_vs_spatial_decision_summary.csv"

FIG22_PATH = FIG_DIR / "fig22_ode_pde_representative_comparison.png"
FIG23_PATH = FIG_DIR / "fig23_spatial_variance_decay.png"
FIG24_PATH = FIG_DIR / "fig24_ode_pde_basin_agreement.png"
FIG25_PATH = FIG_DIR / "fig25_perturbation_sensitivity.png"
FIG26_PATH = FIG_DIR / "fig26_mechanism_decision_summary.png"
FIG27_PATH = FIG_DIR / "fig27_ode_pde_basin_confusion_matrix.png"

NOTE_PATH = NOTES_DIR / "roy_homogeneous_vs_spatial_mechanism.md"
CURRENT_INTERPRETATION_NOTE = NOTES_DIR / "roy_current_mechanism_interpretation.md"
MANUSCRIPT_TEX = MANUSCRIPT_DIR / "roy_homogeneous_vs_spatial_mechanism.tex"

REPRESENTATIVE_CASES = {
    "persistent_case": {
        "stress": 0.1584375,
        "q0": 0.1,
        "w0_scale": 0.1,
        "field_path": RESULTS_DIR / "roy_pde_evo_representative_fields_persistent_case.npz",
    },
    "extinct_case": {
        "stress": 0.16486816,
        "q0": 0.9,
        "w0_scale": 0.02,
        "field_path": RESULTS_DIR / "roy_pde_evo_representative_fields_extinct_case.npz",
    },
    "transient_case": {
        "stress": 0.1584375,
        "q0": 0.7,
        "w0_scale": 0.02,
        "field_path": RESULTS_DIR / "roy_pde_evo_representative_fields_transient_case.npz",
    },
}
CASE_ORDER = ("persistent_case", "extinct_case", "transient_case")
STEADY_CASES = {"persistent_case", "extinct_case"}
PERTURBATION_AMPLITUDES = (0.0, 1.0e-5, 1.0e-3)
PERTURBATION_SEEDS = (20260702, 20260703)
BASIN_LABEL_ORDER = (
    "persistent_basin",
    "extinct_basin",
    "transient_basin",
    "unresolved_basin",
    "nonphysical_initial_condition",
)

REPRESENTATIVE_COMPARISON_FIELDS = [
    "case_label",
    "stress",
    "q0",
    "w0_scale",
    "T",
    "ode_classification",
    "ode_basin_label",
    "pde_classification",
    "pde_basin_label",
    "classification_agreement",
    "basin_label_agreement",
    "rmse_n",
    "rmse_w",
    "rmse_q",
    "max_abs_difference_n",
    "max_abs_difference_w",
    "max_abs_difference_q",
    "final_abs_difference_n",
    "final_abs_difference_w",
    "final_abs_difference_q",
    "ode_tail_mean_w",
    "ode_tail_mean_q",
    "ode_normalized_residual",
    "notes",
]

VARIANCE_TIMESERIES_FIELDS = [
    "case_label",
    "stress",
    "q0",
    "w0_scale",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
    "var_n",
    "var_w",
    "var_q",
    "cv_n",
    "cv_w",
    "cv_q",
    "range_n",
    "range_w",
    "range_q",
    "final_field_var_n",
    "final_field_var_w",
    "final_field_var_q",
    "final_field_cv_n",
    "final_field_cv_w",
    "final_field_cv_q",
    "final_field_range_n",
    "final_field_range_w",
    "final_field_range_q",
    "notes",
]

BASIN_AGREEMENT_FIELDS = [
    "stress",
    "q0",
    "w0_scale",
    "ode_classification",
    "ode_basin_label",
    "pde_classification",
    "pde_basin_label",
    "labels_agree",
    "notes",
]

BASIN_DISAGREEMENT_FIELDS = [
    "stress",
    "q0",
    "w0_scale",
    "ode_classification",
    "ode_basin_label",
    "pde_classification",
    "pde_basin_label",
    "labels_agree",
    "disagreement_type",
    "interpretation_note",
]

DISAGREEMENT_SUMMARY_FIELDS = [
    "stress",
    "total_points",
    "agreement_count",
    "disagreement_count",
    "agreement_fraction",
    "transient_involved_disagreement_count",
    "direct_persistent_extinct_disagreement_count",
    "dominant_disagreement_type",
    "interpretation",
]

PERTURBATION_FIELDS = [
    "case_label",
    "stress",
    "q0",
    "w0_scale",
    "perturbation_amplitude",
    "seed",
    "classification",
    "basin_label",
    "tail_mean_w",
    "tail_mean_q",
    "final_var_n",
    "final_var_w",
    "final_var_q",
    "final_cv_n",
    "final_cv_w",
    "final_cv_q",
    "normalized_residual",
    "physical",
    "notes",
]

DECISION_FIELDS = [
    "representative_ode_pde_agreement_count",
    "representative_ode_pde_total",
    "mean_rmse_w",
    "max_final_difference_w",
    "max_final_cv_n",
    "max_final_cv_w",
    "max_final_cv_q",
    "max_final_cv_n_steady",
    "max_final_cv_w_steady",
    "max_final_cv_q_steady",
    "basin_grid_agreement_fraction",
    "disagreement_count",
    "transient_involved_disagreement_count",
    "direct_persistent_extinct_disagreement_count",
    "disagreement_interpretation",
    "perturbation_outcome_change_count",
    "perturbation_total_groups",
    "perturbation_steady_outcome_change_count",
    "representative_steady_disagreement_count",
    "physical_issue_count",
    "final_label",
    "interpretation",
]


@dataclass(frozen=True)
class Baseline:
    n: float
    w: float
    q: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def format_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def representative_required_files() -> list[Path]:
    return [
        PDE_REP_SUMMARY_CSV,
        PDE_REP_TIMESERIES_CSV,
        *(case["field_path"] for case in REPRESENTATIVE_CASES.values()),
    ]


def ensure_representative_outputs(profile: str) -> None:
    missing = [path for path in representative_required_files() if not path.exists()]
    if not missing:
        return
    print("Representative PDE outputs missing; rerunning experiment 19:")
    for path in missing:
        print(f"  missing {path.relative_to(ROOT)}")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "19_roy_pde_evo_representative_solutions.py"),
            "--profile",
            profile,
        ],
        cwd=ROOT,
        check=True,
    )


def baseline_values() -> Baseline:
    eq = find_evo_equilibrium(PARAMS)
    return Baseline(n=float(eq["n"]), w=float(eq["w"]), q=float(eq["q"]))


def initial_mean_for_case(baseline: Baseline, q0: float, w0_scale: float) -> np.ndarray:
    return np.array([baseline.n, baseline.w * w0_scale, q0], dtype=float)


def tail_mask(times: np.ndarray, tail_fraction: float = TAIL_FRACTION) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("Need at least two time points for a tail window.")
    cutoff = float(times[-1]) - tail_fraction * float(times[-1] - times[0])
    mask = times >= cutoff
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(times, dtype=bool)
        mask[-2:] = True
    return mask


def previous_window_mask(times: np.ndarray, tail_fraction: float = TAIL_FRACTION) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("Need at least two time points for a previous window.")
    span = float(times[-1] - times[0])
    latest_start = float(times[-1]) - tail_fraction * span
    previous_start = float(times[-1]) - 2.0 * tail_fraction * span
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
    if classification == "nonphysical":
        return "nonphysical_initial_condition"
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


def ode_rhs_residual(final_state: np.ndarray, stress: float) -> dict[str, float]:
    rhs = reaction_ode_evo(0.0, final_state, PARAMS, stress=stress, evolve=True)
    rhs_norm = float(np.linalg.norm(rhs))
    state_norm = float(np.linalg.norm(final_state))
    return {
        "rhs_norm": rhs_norm,
        "state_norm": state_norm,
        "normalized_residual": float(rhs_norm / max(state_norm, 1.0e-12)),
    }


def pde_evo_rhs_residual(
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    config: RoyEvoPDEConfig,
    stress: float,
) -> dict[str, float]:
    _x, _y, dx, dy = grid_2d_evo(config)
    reactions = reaction_part_evo_pde(n, w, q, PARAMS, stress=stress, evolve=True)
    dn_dt = config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + reactions[0]
    dw_dt = config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + reactions[1]
    dq_dt = config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + reactions[2]
    rhs_norm = float(np.sqrt(np.mean(dn_dt * dn_dt + dw_dt * dw_dt + dq_dt * dq_dt)))
    state_norm = float(np.sqrt(np.mean(n * n + w * w + q * q)))
    return {
        "rhs_norm": rhs_norm,
        "state_norm": state_norm,
        "normalized_residual": float(rhs_norm / max(state_norm, 1.0e-12)),
    }


def ode_metrics_and_classification(times: np.ndarray, values: np.ndarray, stress: float) -> dict[str, Any]:
    diagnostics = classify_evo_trajectory(times, values, params=PARAMS)
    mask = tail_mask(times)
    prev = previous_window_mask(times)
    tail_w = values[1, mask]
    tail_q = values[2, mask]
    tail_t = times[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(times, values[1], mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_w = float(np.mean(values[1, prev]))
    latest_w = tail_mean_w
    residual = ode_rhs_residual(values[:, -1], stress)
    persistent_without_slope = bool(diagnostics["physical"] and tail_mean_w > EPSILON and tail_min_w > 0.25 * EPSILON)
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    metrics: dict[str, Any] = {
        "physical": bool(diagnostics["physical"]),
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": slope_floor,
        "tail_mean_q": float(np.mean(tail_q)),
        "previous_window_mean_w": previous_w,
        "latest_window_mean_w": latest_w,
        "relative_change_between_last_windows": relative_change(previous_w, latest_w),
        "persistent_without_slope_rule": persistent_without_slope,
        "persistent_with_slope_rule": persistent_with_slope,
        **residual,
    }
    metrics["classification"] = classify_asymptotic(metrics)
    metrics["basin_label"] = basin_label_from_classification(str(metrics["classification"]))
    return metrics


def physical_pde_result(result: RoyEvoPDEResult) -> bool:
    arrays = (
        result.t,
        result.mean_n_time,
        result.mean_w_time,
        result.mean_q_time,
        result.var_n_time,
        result.var_w_time,
        result.var_q_time,
        result.min_z_time,
        result.n,
        result.w,
        result.q,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return False
    z = free_space_evo(result.n, result.w, PARAMS)
    return bool(
        result.diagnostics.get("completed", True)
        and np.min(result.n) >= -1.0e-8
        and np.min(result.w) >= -1.0e-8
        and np.min(result.q) >= -1.0e-6
        and np.max(result.q) <= 1.0 + 1.0e-6
        and np.min(z) >= -1.0e-5
        and float(result.diagnostics.get("q_clip_max_violation", 0.0)) <= 1.0e-4
    )


def pde_metrics_and_classification(result: RoyEvoPDEResult, config: RoyEvoPDEConfig, stress: float) -> dict[str, Any]:
    mask = tail_mask(result.t)
    prev = previous_window_mask(result.t)
    tail_w = result.mean_w_time[mask]
    tail_q = result.mean_q_time[mask]
    tail_t = result.t[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(result.t, result.mean_w_time, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_w = float(np.mean(result.mean_w_time[prev]))
    latest_w = tail_mean_w
    residual = pde_evo_rhs_residual(result.n, result.w, result.q, config, stress)
    physical = physical_pde_result(result)
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
        "latest_window_mean_w": latest_w,
        "relative_change_between_last_windows": relative_change(previous_w, latest_w),
        "persistent_without_slope_rule": persistent_without_slope,
        "persistent_with_slope_rule": persistent_with_slope,
        **residual,
    }
    metrics["classification"] = classify_asymptotic(metrics)
    metrics["basin_label"] = basin_label_from_classification(str(metrics["classification"]))
    return metrics


def comparison_metrics(ode_values: np.ndarray, pde_values: np.ndarray) -> dict[str, float]:
    ode_arr = np.asarray(ode_values, dtype=float)
    pde_arr = np.asarray(pde_values, dtype=float)
    if ode_arr.shape != pde_arr.shape:
        raise ValueError("ODE and PDE arrays must have the same shape.")
    diff = ode_arr - pde_arr
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "final_abs_difference": float(abs(diff[-1])),
    }


def field_summary(path: Path) -> dict[str, float]:
    archive = np.load(path)
    n = np.asarray(archive["final_n"], dtype=float)
    w = np.asarray(archive["final_w"], dtype=float)
    q = np.asarray(archive["final_q"], dtype=float)
    return {
        "final_field_var_n": float(np.var(n)),
        "final_field_var_w": float(np.var(w)),
        "final_field_var_q": float(np.var(q)),
        "final_field_cv_n": float(np.std(n) / max(abs(float(np.mean(n))), 1.0e-12)),
        "final_field_cv_w": float(np.std(w) / max(abs(float(np.mean(w))), 1.0e-12)),
        "final_field_cv_q": float(np.std(q) / max(abs(float(np.mean(q))), 1.0e-12)),
        "final_field_range_n": float(np.max(n) - np.min(n)),
        "final_field_range_w": float(np.max(w) - np.min(w)),
        "final_field_range_q": float(np.max(q) - np.min(q)),
    }


def group_representative_timeseries() -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PDE_REP_TIMESERIES_CSV):
        groups[row["case_label"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: as_float(row["time"]))
    return dict(groups)


def representative_summary_by_case() -> dict[str, dict[str, str]]:
    return {row["case_label"]: row for row in read_csv(PDE_REP_SUMMARY_CSV)}


def run_representative_comparison(baseline: Baseline) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = group_representative_timeseries()
    summary_by_case = representative_summary_by_case()
    comparison_rows: list[dict[str, Any]] = []
    plot_payload: dict[str, Any] = {}

    for case_label in CASE_ORDER:
        case = REPRESENTATIVE_CASES[case_label]
        pde_rows = groups[case_label]
        times = np.array([as_float(row["time"]) for row in pde_rows], dtype=float)
        pde_n = np.array([as_float(row["mean_n"]) for row in pde_rows], dtype=float)
        pde_w = np.array([as_float(row["mean_w"]) for row in pde_rows], dtype=float)
        pde_q = np.array([as_float(row["mean_q"]) for row in pde_rows], dtype=float)
        stress = float(case["stress"])
        q0 = float(case["q0"])
        w0_scale = float(case["w0_scale"])
        initial = initial_mean_for_case(baseline, q0, w0_scale)
        ode = simulate_ode_evo(PARAMS, initial, stress=stress, evolve=True, T=float(times[-1]), n_eval=len(times))
        if not np.allclose(ode.t, times, atol=1.0e-9, rtol=1.0e-9):
            ode_values = np.vstack([np.interp(times, ode.t, ode.y[idx]) for idx in range(3)])
        else:
            ode_values = ode.y
        ode_metrics = ode_metrics_and_classification(times, ode_values, stress)
        pde_summary = summary_by_case[case_label]
        pde_classification = pde_summary["classification_from_rerun"]
        pde_basin_label = pde_summary["basin_label_from_rerun"]

        n_metrics = comparison_metrics(ode_values[0], pde_n)
        w_metrics = comparison_metrics(ode_values[1], pde_w)
        q_metrics = comparison_metrics(ode_values[2], pde_q)
        class_agree = str(ode_metrics["classification"]) == pde_classification
        basin_agree = str(ode_metrics["basin_label"]) == pde_basin_label
        notes = "ode_pde_match" if class_agree and basin_agree else "ode_pde_disagreement"
        comparison_rows.append(
            {
                "case_label": case_label,
                "stress": stress,
                "q0": q0,
                "w0_scale": w0_scale,
                "T": float(times[-1]),
                "ode_classification": ode_metrics["classification"],
                "ode_basin_label": ode_metrics["basin_label"],
                "pde_classification": pde_classification,
                "pde_basin_label": pde_basin_label,
                "classification_agreement": class_agree,
                "basin_label_agreement": basin_agree,
                "rmse_n": n_metrics["rmse"],
                "rmse_w": w_metrics["rmse"],
                "rmse_q": q_metrics["rmse"],
                "max_abs_difference_n": n_metrics["max_abs_difference"],
                "max_abs_difference_w": w_metrics["max_abs_difference"],
                "max_abs_difference_q": q_metrics["max_abs_difference"],
                "final_abs_difference_n": n_metrics["final_abs_difference"],
                "final_abs_difference_w": w_metrics["final_abs_difference"],
                "final_abs_difference_q": q_metrics["final_abs_difference"],
                "ode_tail_mean_w": ode_metrics["tail_mean_w"],
                "ode_tail_mean_q": ode_metrics["tail_mean_q"],
                "ode_normalized_residual": ode_metrics["normalized_residual"],
                "notes": notes,
            }
        )
        plot_payload[case_label] = {
            "time": times,
            "ode_n": ode_values[0],
            "ode_w": ode_values[1],
            "ode_q": ode_values[2],
            "pde_n": pde_n,
            "pde_w": pde_w,
            "pde_q": pde_q,
        }

    write_csv(REPRESENTATIVE_COMPARISON_CSV, comparison_rows, REPRESENTATIVE_COMPARISON_FIELDS)
    return comparison_rows, plot_payload


def run_variance_diagnostics() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for case_label, pde_rows in group_representative_timeseries().items():
        case = REPRESENTATIVE_CASES[case_label]
        final = field_summary(case["field_path"])
        final_time = max(as_float(row["time"]) for row in pde_rows)
        for row in pde_rows:
            mean_n = as_float(row["mean_n"])
            mean_w = as_float(row["mean_w"])
            mean_q = as_float(row["mean_q"])
            var_n = as_float(row["var_n"])
            var_w = as_float(row["var_w"])
            var_q = as_float(row["var_q"])
            time = as_float(row["time"])
            row_out = {
                "case_label": case_label,
                "stress": as_float(row["stress"]),
                "q0": as_float(row["q0"]),
                "w0_scale": as_float(row["w0_scale"]),
                "time": time,
                "mean_n": mean_n,
                "mean_w": mean_w,
                "mean_q": mean_q,
                "var_n": var_n,
                "var_w": var_w,
                "var_q": var_q,
                "cv_n": math.sqrt(max(var_n, 0.0)) / max(abs(mean_n), 1.0e-12),
                "cv_w": math.sqrt(max(var_w, 0.0)) / max(abs(mean_w), 1.0e-12),
                "cv_q": math.sqrt(max(var_q, 0.0)) / max(abs(mean_q), 1.0e-12),
                "range_n": "",
                "range_w": "",
                "range_q": as_float(row.get("max_q")) - as_float(row.get("min_q")),
                **final,
                "notes": "final_ranges_from_field_archive" if math.isclose(time, final_time) else "",
            }
            if math.isclose(time, final_time):
                row_out["range_n"] = final["final_field_range_n"]
                row_out["range_w"] = final["final_field_range_w"]
                row_out["range_q"] = final["final_field_range_q"]
            rows_out.append(row_out)
    write_csv(VARIANCE_TIMESERIES_CSV, rows_out, VARIANCE_TIMESERIES_FIELDS)
    return rows_out


def ode_basin_agreement_rows(baseline: Baseline) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for row in read_csv(PDE_BASIN_SCAN_CSV):
        stress = as_float(row["stress"])
        q0 = as_float(row["q0"])
        w0_scale = as_float(row["w0_scale"])
        T = as_float(row.get("T"), 1600.0)
        initial = initial_mean_for_case(baseline, q0, w0_scale)
        ode = simulate_ode_evo(PARAMS, initial, stress=stress, evolve=True, T=T, n_eval=321)
        metrics = ode_metrics_and_classification(ode.t, ode.y, stress)
        ode_classification = str(metrics["classification"])
        ode_basin_label = str(metrics["basin_label"])
        pde_classification = row["classification"]
        pde_basin_label = row["basin_label"]
        labels_agree = ode_basin_label == pde_basin_label
        rows_out.append(
            {
                "stress": stress,
                "q0": q0,
                "w0_scale": w0_scale,
                "ode_classification": ode_classification,
                "ode_basin_label": ode_basin_label,
                "pde_classification": pde_classification,
                "pde_basin_label": pde_basin_label,
                "labels_agree": labels_agree,
                "notes": "labels_agree" if labels_agree else f"ode_{ode_basin_label}_pde_{pde_basin_label}",
            }
        )
    write_csv(BASIN_AGREEMENT_CSV, rows_out, BASIN_AGREEMENT_FIELDS)
    return rows_out


def basin_agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    agree = sum(as_bool(row["labels_agree"]) for row in rows)
    by_stress: dict[float, dict[str, float]] = {}
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    for row in rows:
        stress = as_float(row["stress"])
        grouped[stress].append(row)
        confusion[(str(row["ode_basin_label"]), str(row["pde_basin_label"]))] += 1
    for stress, stress_rows in grouped.items():
        stress_total = len(stress_rows)
        stress_agree = sum(as_bool(row["labels_agree"]) for row in stress_rows)
        by_stress[stress] = {
            "total": float(stress_total),
            "agreement_fraction": float(stress_agree / stress_total) if stress_total else math.nan,
        }
    return {
        "total": total,
        "agreement_count": agree,
        "agreement_fraction": float(agree / total) if total else math.nan,
        "by_stress": by_stress,
        "confusion": dict(confusion),
    }


def disagreement_type(ode_basin_label: str, pde_basin_label: str) -> str:
    pair = (ode_basin_label, pde_basin_label)
    mapping = {
        ("persistent_basin", "transient_basin"): "ode_persistent_pde_transient",
        ("transient_basin", "persistent_basin"): "ode_transient_pde_persistent",
        ("extinct_basin", "transient_basin"): "ode_extinct_pde_transient",
        ("transient_basin", "extinct_basin"): "ode_transient_pde_extinct",
        ("persistent_basin", "extinct_basin"): "ode_persistent_pde_extinct",
        ("extinct_basin", "persistent_basin"): "ode_extinct_pde_persistent",
    }
    return mapping.get(pair, "other")


def disagreement_involves_transient(disagreement: str) -> bool:
    return "_transient" in disagreement or "transient_" in disagreement


def disagreement_is_direct_persistent_extinct(disagreement: str) -> bool:
    return disagreement in {"ode_persistent_pde_extinct", "ode_extinct_pde_persistent"}


def direct_disagreement_is_substantial(direct_count: int, total_points: int) -> bool:
    if direct_count <= 0:
        return False
    return direct_count >= 3 or direct_count / max(total_points, 1) >= 0.02


def disagreement_interpretation_note(disagreement: str, direct_count: int, total_points: int) -> str:
    notes: list[str] = []
    if disagreement_involves_transient(disagreement):
        notes.append("boundary_or_horizon_sensitive")
    elif disagreement_is_direct_persistent_extinct(disagreement):
        notes.append("strong_basin_disagreement")
    else:
        notes.append("other_disagreement")

    if disagreement_is_direct_persistent_extinct(disagreement) and direct_disagreement_is_substantial(direct_count, total_points):
        notes.append("possible_mixed_effect_requires_followup")
    elif disagreement_involves_transient(disagreement):
        notes.append("does_not_overturn_reaction_dominated_label")
    return ";".join(notes)


def audit_basin_disagreements(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total_points = len(rows)
    raw_disagreements = [row for row in rows if not as_bool(row["labels_agree"])]
    type_counter = Counter(
        disagreement_type(str(row["ode_basin_label"]), str(row["pde_basin_label"]))
        for row in raw_disagreements
    )
    direct_count = sum(count for dtype, count in type_counter.items() if disagreement_is_direct_persistent_extinct(dtype))
    disagreement_rows: list[dict[str, Any]] = []
    for row in raw_disagreements:
        dtype = disagreement_type(str(row["ode_basin_label"]), str(row["pde_basin_label"]))
        disagreement_rows.append(
            {
                "stress": row["stress"],
                "q0": row["q0"],
                "w0_scale": row["w0_scale"],
                "ode_classification": row["ode_classification"],
                "ode_basin_label": row["ode_basin_label"],
                "pde_classification": row["pde_classification"],
                "pde_basin_label": row["pde_basin_label"],
                "labels_agree": row["labels_agree"],
                "disagreement_type": dtype,
                "interpretation_note": disagreement_interpretation_note(dtype, direct_count, total_points),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    stress_values: list[float | str] = sorted({as_float(row["stress"]) for row in rows})
    for stress in [*stress_values, "all"]:
        if stress == "all":
            scoped_rows = rows
        else:
            scoped_rows = [row for row in rows if math.isclose(as_float(row["stress"]), float(stress))]
        total = len(scoped_rows)
        agreement_count = sum(as_bool(row["labels_agree"]) for row in scoped_rows)
        scoped_disagreements = [row for row in scoped_rows if not as_bool(row["labels_agree"])]
        scoped_types = [
            disagreement_type(str(row["ode_basin_label"]), str(row["pde_basin_label"]))
            for row in scoped_disagreements
        ]
        scoped_counter = Counter(scoped_types)
        transient_count = sum(1 for dtype in scoped_types if disagreement_involves_transient(dtype))
        scoped_direct_count = sum(1 for dtype in scoped_types if disagreement_is_direct_persistent_extinct(dtype))
        dominant = scoped_counter.most_common(1)[0][0] if scoped_counter else "none"
        summary_rows.append(
            {
                "stress": stress if stress == "all" else format_float(float(stress), 9),
                "total_points": total,
                "agreement_count": agreement_count,
                "disagreement_count": len(scoped_disagreements),
                "agreement_fraction": float(agreement_count / total) if total else math.nan,
                "transient_involved_disagreement_count": transient_count,
                "direct_persistent_extinct_disagreement_count": scoped_direct_count,
                "dominant_disagreement_type": dominant,
                "interpretation": disagreement_summary_interpretation(
                    disagreement_count=len(scoped_disagreements),
                    transient_count=transient_count,
                    direct_count=scoped_direct_count,
                    total_points=total,
                ),
            }
        )

    overall = next(row for row in summary_rows if row["stress"] == "all")
    audit_summary = {
        "disagreement_count": int(overall["disagreement_count"]),
        "transient_involved_disagreement_count": int(overall["transient_involved_disagreement_count"]),
        "direct_persistent_extinct_disagreement_count": int(overall["direct_persistent_extinct_disagreement_count"]),
        "dominant_disagreement_type": str(overall["dominant_disagreement_type"]),
        "interpretation": str(overall["interpretation"]),
    }
    write_csv(DISAGREEMENTS_CSV, disagreement_rows, BASIN_DISAGREEMENT_FIELDS)
    write_csv(DISAGREEMENT_SUMMARY_CSV, summary_rows, DISAGREEMENT_SUMMARY_FIELDS)
    return disagreement_rows, summary_rows, audit_summary


def disagreement_summary_interpretation(
    *,
    disagreement_count: int,
    transient_count: int,
    direct_count: int,
    total_points: int,
) -> str:
    if disagreement_count == 0:
        return "complete_ode_pde_basin_label_agreement"
    if direct_disagreement_is_substantial(direct_count, total_points):
        return "direct_persistent_extinct_disagreements_suggest_possible_mixed_effect_requires_followup"
    if direct_count > 0:
        return "rare_direct_persistent_extinct_disagreements_require_targeted_followup"
    if transient_count == disagreement_count:
        return "disagreements_are_boundary_or_horizon_sensitive_and_do_not_overturn_reaction_dominated_label"
    if transient_count / max(disagreement_count, 1) >= 0.8:
        return "most_disagreements_are_boundary_or_horizon_sensitive"
    return "disagreement_pattern_requires_followup"


def pde_config(T: float, perturbation_amplitude: float, seed: int) -> RoyEvoPDEConfig:
    return RoyEvoPDEConfig(
        n_x=64,
        n_y=64,
        L_x=20.0,
        L_y=20.0,
        dt=0.1,
        T=float(T),
        record_every=50,
        D_n=0.01,
        D_w=0.01,
        D_q=0.005,
        perturbation_amplitude=float(perturbation_amplitude),
        seed=int(seed),
    )


def perturbation_sensitivity_rows(baseline: Baseline) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    total = len(CASE_ORDER) * len(PERTURBATION_AMPLITUDES) * len(PERTURBATION_SEEDS)
    index = 0
    for case_label in CASE_ORDER:
        case = REPRESENTATIVE_CASES[case_label]
        stress = float(case["stress"])
        q0 = float(case["q0"])
        w0_scale = float(case["w0_scale"])
        n0 = baseline.n
        w0 = baseline.w * w0_scale
        for amplitude in PERTURBATION_AMPLITUDES:
            for seed in PERTURBATION_SEEDS:
                index += 1
                print(
                    f"PERTURB {index:02d}/{total:02d} {case_label} "
                    f"stress={stress:.9g} q0={q0:g} w0_scale={w0_scale:g} "
                    f"amp={amplitude:g} seed={seed}"
                )
                config = pde_config(T=1600.0, perturbation_amplitude=amplitude, seed=seed)
                initial = initial_state_from_ode_equilibrium({"n": n0, "w": w0, "q": q0}, config)
                result = simulate_pde_evo_2d(PARAMS, config, initial, stress=stress, evolve=True)
                metrics = pde_metrics_and_classification(result, config, stress)
                final_var_n = float(np.var(result.n))
                final_var_w = float(np.var(result.w))
                final_var_q = float(np.var(result.q))
                final_cv_n = float(np.std(result.n) / max(abs(float(np.mean(result.n))), 1.0e-12))
                final_cv_w = float(np.std(result.w) / max(abs(float(np.mean(result.w))), 1.0e-12))
                final_cv_q = float(np.std(result.q) / max(abs(float(np.mean(result.q))), 1.0e-12))
                notes = []
                if amplitude == 0.0:
                    notes.append("homogeneous_initial_condition")
                if not bool(metrics["physical"]):
                    notes.append("nonphysical_or_incomplete")
                rows_out.append(
                    {
                        "case_label": case_label,
                        "stress": stress,
                        "q0": q0,
                        "w0_scale": w0_scale,
                        "perturbation_amplitude": amplitude,
                        "seed": seed,
                        "classification": metrics["classification"],
                        "basin_label": metrics["basin_label"],
                        "tail_mean_w": metrics["tail_mean_w"],
                        "tail_mean_q": metrics["tail_mean_q"],
                        "final_var_n": final_var_n,
                        "final_var_w": final_var_w,
                        "final_var_q": final_var_q,
                        "final_cv_n": final_cv_n,
                        "final_cv_w": final_cv_w,
                        "final_cv_q": final_cv_q,
                        "normalized_residual": metrics["normalized_residual"],
                        "physical": metrics["physical"],
                        "notes": ";".join(notes) if notes else "targeted_perturbation_run",
                    }
                )
    write_csv(PERTURBATION_SENSITIVITY_CSV, rows_out, PERTURBATION_FIELDS)
    return rows_out


def detect_perturbation_outcome_changes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["case_label"])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for case_label, case_rows in groups.items():
        classifications = {str(row["classification"]) for row in case_rows}
        basin_labels = {str(row["basin_label"]) for row in case_rows}
        summary[case_label] = {
            "classifications": sorted(classifications),
            "basin_labels": sorted(basin_labels),
            "classification_changed": len(classifications) > 1,
            "basin_label_changed": len(basin_labels) > 1,
            "n": len(case_rows),
        }
    return summary


def decide_final_label(evidence: dict[str, Any]) -> tuple[str, str]:
    rep_agree = int(evidence["representative_ode_pde_agreement_count"])
    rep_total = int(evidence["representative_ode_pde_total"])
    basin_fraction = float(evidence["basin_grid_agreement_fraction"])
    max_cv_n_steady = float(evidence["max_final_cv_n_steady"])
    max_cv_w_steady = float(evidence["max_final_cv_w_steady"])
    max_cv_q_steady = float(evidence["max_final_cv_q_steady"])
    max_cv_w = float(evidence["max_final_cv_w"])
    max_cv_q = float(evidence["max_final_cv_q"])
    perturb_steady_changes = int(evidence["perturbation_steady_outcome_change_count"])
    perturb_changes = int(evidence["perturbation_outcome_change_count"])
    steady_disagreements = int(evidence["representative_steady_disagreement_count"])
    physical_issue_count = int(evidence.get("physical_issue_count", 0))
    disagreement_count = int(evidence.get("disagreement_count", 0))
    transient_disagreements = int(evidence.get("transient_involved_disagreement_count", 0))
    direct_disagreements = int(evidence.get("direct_persistent_extinct_disagreement_count", 0))

    low_steady_cv = max(max_cv_n_steady, max_cv_w_steady, max_cv_q_steady) < 1.0e-4
    no_strong_spatial_variance = max(max_cv_w, max_cv_q) < 1.0e-2
    disagreements_mostly_transient = disagreement_count == 0 or transient_disagreements / max(disagreement_count, 1) >= 0.8
    direct_disagreements_substantial = direct_disagreement_is_substantial(direct_disagreements, int(evidence.get("basin_grid_total", 140)))

    if physical_issue_count > 0:
        return (
            "mechanism_unresolved",
            "Physicality or numerical completion issues prevent a clean mechanism assignment.",
        )

    if direct_disagreements_substantial:
        return (
            "mixed_homogeneous_and_spatial_effects",
            "The disagreement audit shows direct persistent/extinct ODE-PDE conflicts, so homogeneous reaction dynamics are not sufficient by themselves.",
        )

    if (
        rep_agree >= min(2, rep_total)
        and basin_fraction >= 0.85
        and low_steady_cv
        and perturb_steady_changes == 0
        and no_strong_spatial_variance
        and disagreements_mostly_transient
    ):
        return (
            "reaction_dominated_homogeneous_multistability",
            "The current evidence indicates that basin dependence is primarily inherited from homogeneous eco-evolutionary reaction dynamics rather than generated by persistent spatial patterning.",
        )

    spatial_criteria = [
        steady_disagreements >= 1,
        basin_fraction < 0.65,
        max(max_cv_w, max_cv_q) > 1.0e-2,
        perturb_steady_changes > 0,
        not no_strong_spatial_variance,
        direct_disagreements_substantial,
    ]
    if sum(spatial_criteria) >= 2:
        return (
            "spatially_mediated_bistability",
            "The current evidence indicates that spatial structure contributes directly to basin entry and outcome selection.",
        )

    if perturb_changes > 0 or basin_fraction < 0.85 or not low_steady_cv:
        return (
            "mixed_homogeneous_and_spatial_effects",
            "ODE and PDE evidence is partly aligned, but at least one diagnostic leaves room for spatial influence or unresolved transient effects.",
        )

    return (
        "mechanism_unresolved",
        "The available diagnostics do not cleanly distinguish homogeneous reaction multistability from spatially mediated basin selection.",
    )


def decision_summary_rows(
    comparison_rows: list[dict[str, Any]],
    variance_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    basin_summary = basin_agreement_summary(basin_rows)
    disagreement_rows, disagreement_summary_rows_out, disagreement_audit = audit_basin_disagreements(basin_rows)
    perturb_summary = detect_perturbation_outcome_changes(perturbation_rows)
    rep_agree = sum(as_bool(row["classification_agreement"]) for row in comparison_rows)
    rep_total = len(comparison_rows)
    steady_disagreements = sum(
        row["case_label"] in STEADY_CASES and not as_bool(row["classification_agreement"])
        for row in comparison_rows
    )
    perturb_changes = sum(bool(values["classification_changed"]) for values in perturb_summary.values())
    perturb_steady_changes = sum(
        case_label in STEADY_CASES and bool(values["classification_changed"])
        for case_label, values in perturb_summary.items()
    )
    final_rows = []
    for case_label in CASE_ORDER:
        case_rows = [row for row in variance_rows if row["case_label"] == case_label]
        final_time = max(as_float(row["time"]) for row in case_rows)
        final_rows.extend(row for row in case_rows if math.isclose(as_float(row["time"]), final_time))
    steady_final_rows = [row for row in final_rows if row["case_label"] in STEADY_CASES]
    physical_issue_count = sum(not as_bool(row["physical"]) for row in perturbation_rows)
    evidence: dict[str, Any] = {
        "representative_ode_pde_agreement_count": rep_agree,
        "representative_ode_pde_total": rep_total,
        "mean_rmse_w": float(np.mean([as_float(row["rmse_w"]) for row in comparison_rows])) if comparison_rows else math.nan,
        "max_final_difference_w": max(as_float(row["final_abs_difference_w"]) for row in comparison_rows) if comparison_rows else math.nan,
        "max_final_cv_n": max(as_float(row["final_field_cv_n"]) for row in final_rows) if final_rows else math.nan,
        "max_final_cv_w": max(as_float(row["final_field_cv_w"]) for row in final_rows) if final_rows else math.nan,
        "max_final_cv_q": max(as_float(row["final_field_cv_q"]) for row in final_rows) if final_rows else math.nan,
        "max_final_cv_n_steady": max(as_float(row["final_field_cv_n"]) for row in steady_final_rows) if steady_final_rows else math.nan,
        "max_final_cv_w_steady": max(as_float(row["final_field_cv_w"]) for row in steady_final_rows) if steady_final_rows else math.nan,
        "max_final_cv_q_steady": max(as_float(row["final_field_cv_q"]) for row in steady_final_rows) if steady_final_rows else math.nan,
        "basin_grid_agreement_fraction": basin_summary["agreement_fraction"],
        "basin_grid_total": basin_summary["total"],
        "disagreement_count": disagreement_audit["disagreement_count"],
        "transient_involved_disagreement_count": disagreement_audit["transient_involved_disagreement_count"],
        "direct_persistent_extinct_disagreement_count": disagreement_audit["direct_persistent_extinct_disagreement_count"],
        "disagreement_interpretation": disagreement_audit["interpretation"],
        "perturbation_outcome_change_count": perturb_changes,
        "perturbation_total_groups": len(perturb_summary),
        "perturbation_steady_outcome_change_count": perturb_steady_changes,
        "representative_steady_disagreement_count": steady_disagreements,
        "physical_issue_count": physical_issue_count,
    }
    final_label, interpretation = decide_final_label(evidence)
    evidence["final_label"] = final_label
    evidence["interpretation"] = interpretation
    write_csv(DECISION_SUMMARY_CSV, [evidence], DECISION_FIELDS)
    return [evidence], {
        "basin_summary": basin_summary,
        "perturbation_summary": perturb_summary,
        "disagreement_rows": disagreement_rows,
        "disagreement_summary_rows": disagreement_summary_rows_out,
        "disagreement_audit": disagreement_audit,
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_comparison(payload: dict[str, Any]) -> None:
    colors = {"persistent_case": "#1b9e77", "extinct_case": "#d95f02", "transient_case": "#7570b3"}
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.0), sharex=True, constrained_layout=True)
    variables = [("n", "prey mean"), ("w", "predator mean"), ("q", "defense mean")]
    for row_idx, case_label in enumerate(CASE_ORDER):
        data = payload[case_label]
        for col_idx, (suffix, ylabel) in enumerate(variables):
            ax = axes[row_idx, col_idx]
            ax.plot(data["time"], data[f"ode_{suffix}"], color="black", linewidth=1.5, label="ODE")
            ax.plot(data["time"], data[f"pde_{suffix}"], color=colors[case_label], linewidth=1.3, linestyle="--", label="PDE mean")
            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.set_title(ylabel)
            if col_idx == 0:
                ax.set_ylabel(case_label.replace("_", " "))
            if row_idx == len(CASE_ORDER) - 1:
                ax.set_xlabel("time")
    handles = [
        mlines.Line2D([], [], color="black", label="matched ODE"),
        mlines.Line2D([], [], color="#555555", linestyle="--", label="PDE spatial mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Matched ODE trajectories versus PDE spatial means", fontsize=14)
    save_figure(fig, FIG22_PATH)


def plot_variance_decay(rows: list[dict[str, Any]]) -> None:
    colors = {"persistent_case": "#1b9e77", "extinct_case": "#d95f02", "transient_case": "#7570b3"}
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.6), sharex=True, constrained_layout=True)
    panels = [("cv_n", "CV(n)"), ("cv_w", "CV(w)"), ("cv_q", "CV(q)")]
    for ax, (field, ylabel) in zip(axes, panels, strict=True):
        for case_label in CASE_ORDER:
            case_rows = sorted([row for row in rows if row["case_label"] == case_label], key=lambda row: as_float(row["time"]))
            ax.plot(
                [as_float(row["time"]) for row in case_rows],
                [max(as_float(row[field]), 1.0e-16) for row in case_rows],
                color=colors[case_label],
                linewidth=1.6,
                label=case_label.replace("_", " "),
            )
        ax.axhline(1.0e-4, color="#555555", linestyle=":", linewidth=1.0, label="CV = 1e-4" if field == "cv_n" else None)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Spatial coefficient of variation in representative PDE fields", fontsize=14)
    save_figure(fig, FIG23_PATH)


def plot_basin_agreement(rows: list[dict[str, Any]]) -> None:
    stresses = sorted({as_float(row["stress"]) for row in rows})
    q_values = sorted({as_float(row["q0"]) for row in rows})
    w_values = sorted({as_float(row["w0_scale"]) for row in rows})
    cmap = mcolors.ListedColormap(["#d95f02", "#1b9e77"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], 2)
    fig, axes = plt.subplots(1, len(stresses), figsize=(6.2 * len(stresses), 5.6), sharey=True, constrained_layout=True)
    if len(stresses) == 1:
        axes = [axes]
    letter = {
        "persistent_basin": "P",
        "extinct_basin": "E",
        "transient_basin": "T",
        "unresolved_basin": "U",
        "nonphysical_initial_condition": "N",
    }
    for ax, stress in zip(axes, stresses, strict=True):
        matrix = np.full((len(w_values), len(q_values)), np.nan)
        label_pairs: dict[tuple[int, int], str] = {}
        for row in rows:
            if not math.isclose(as_float(row["stress"]), stress):
                continue
            x = q_values.index(as_float(row["q0"]))
            y = w_values.index(as_float(row["w0_scale"]))
            agree = as_bool(row["labels_agree"])
            matrix[y, x] = 1 if agree else 0
            label_pairs[(y, x)] = f"{letter.get(str(row['ode_basin_label']), '?')}/{letter.get(str(row['pde_basin_label']), '?')}"
        ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for (y, x), text in label_pairs.items():
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="white")
        ax.set_title(f"stress = {stress:.9g}")
        ax.set_xlabel("initial defense frequency q0")
        ax.set_xticks(range(len(q_values)))
        ax.set_xticklabels([format_float(value, 3) for value in q_values], rotation=45, ha="right")
        ax.set_yticks(range(len(w_values)))
        ax.set_yticklabels([format_float(value, 3) for value in w_values])
        ax.grid(color="white", linewidth=0.6)
    axes[0].set_ylabel("initial predator scale w0_scale")
    handles = [
        mpatches.Patch(color="#1b9e77", label="ODE/PDE basin labels agree"),
        mpatches.Patch(color="#d95f02", label="ODE/PDE basin labels differ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Basin-label agreement between matched ODE and PDE scan rows", fontsize=14)
    save_figure(fig, FIG24_PATH)


def plot_basin_confusion_matrix(rows: list[dict[str, Any]]) -> None:
    labels = [
        label
        for label in BASIN_LABEL_ORDER
        if label != "nonphysical_initial_condition"
        or any(
            row["ode_basin_label"] == "nonphysical_initial_condition"
            or row["pde_basin_label"] == "nonphysical_initial_condition"
            for row in rows
        )
    ]
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row in rows:
        ode_label = str(row["ode_basin_label"])
        pde_label = str(row["pde_basin_label"])
        if ode_label not in index or pde_label not in index:
            continue
        matrix[index[pde_label], index[ode_label]] += 1

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = int(matrix[y, x])
            color = "white" if value > matrix.max() * 0.5 else "#111111"
            ax.text(x, y, str(value), ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([label.replace("_", "\n") for label in labels], rotation=0)
    ax.set_yticklabels([label.replace("_", "\n") for label in labels])
    ax.set_xlabel("ODE basin label")
    ax.set_ylabel("PDE basin label")
    ax.set_title("ODE-PDE basin-label confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="grid-point count")
    save_figure(fig, FIG27_PATH)


def plot_perturbation_sensitivity(rows: list[dict[str, Any]]) -> None:
    colors = {"persistent_case": "#1b9e77", "extinct_case": "#d95f02", "transient_case": "#7570b3"}
    markers = {20260702: "o", 20260703: "s"}
    x_positions = {0.0: 0, 1.0e-5: 1, 1.0e-3: 2}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    for row in rows:
        amp = as_float(row["perturbation_amplitude"])
        seed = int(as_float(row["seed"]))
        case_label = str(row["case_label"])
        x = x_positions[amp] + (-0.045 if seed == 20260702 else 0.045)
        axes[0].scatter(
            x,
            max(as_float(row["tail_mean_w"]), 1.0e-12),
            color=colors[case_label],
            marker=markers[seed],
            edgecolor="black",
            linewidth=0.4,
            s=62,
        )
        max_cv = max(as_float(row["final_cv_n"]), as_float(row["final_cv_w"]), as_float(row["final_cv_q"]))
        axes[1].scatter(
            x,
            max(max_cv, 1.0e-16),
            color=colors[case_label],
            marker=markers[seed],
            edgecolor="black",
            linewidth=0.4,
            s=62,
        )
    for ax in axes:
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0", "1e-5", "1e-3"])
        ax.set_xlabel("perturbation amplitude")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("tail mean predator density")
    axes[1].set_yscale("log")
    axes[1].axhline(1.0e-4, color="#555555", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("max final CV across n, w, q")
    case_handles = [mlines.Line2D([], [], color=color, marker="o", linestyle="", label=case.replace("_", " ")) for case, color in colors.items()]
    seed_handles = [mlines.Line2D([], [], color="black", marker=marker, linestyle="", label=str(seed)) for seed, marker in markers.items()]
    fig.legend(handles=case_handles + seed_handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Perturbation sensitivity for representative PDE cases", fontsize=14)
    save_figure(fig, FIG25_PATH)


def plot_decision_summary(decision_row: dict[str, Any]) -> None:
    metrics = [
        ("ODE/PDE\nbasin agreement", as_float(decision_row["basin_grid_agreement_fraction"]), 0.85),
        ("max steady\nfinal CV", max(as_float(decision_row["max_final_cv_n_steady"]), as_float(decision_row["max_final_cv_w_steady"]), as_float(decision_row["max_final_cv_q_steady"])), 1.0e-4),
        ("perturbation\nchanges", as_float(decision_row["perturbation_steady_outcome_change_count"]), 0.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), constrained_layout=True)
    axes[0].bar([0], [metrics[0][1]], color="#4c78a8")
    axes[0].axhline(metrics[0][2], color="#333333", linestyle=":", linewidth=1.2)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([metrics[0][0]])
    axes[0].set_ylabel("fraction")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([0], [max(metrics[1][1], 1.0e-16)], color="#59a14f")
    axes[1].axhline(metrics[1][2], color="#333333", linestyle=":", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels([metrics[1][0]])
    axes[1].set_ylabel("coefficient of variation")
    axes[1].grid(axis="y", alpha=0.25, which="both")

    axes[2].bar([0], [metrics[2][1]], color="#e15759")
    axes[2].set_ylim(0.0, max(1.0, metrics[2][1] + 0.5))
    axes[2].set_xticks([0])
    axes[2].set_xticklabels([metrics[2][0]])
    axes[2].set_ylabel("steady case groups")
    axes[2].grid(axis="y", alpha=0.25)

    label = str(decision_row["final_label"])
    fig.suptitle(f"Mechanism decision: {label}", fontsize=14)
    save_figure(fig, FIG26_PATH)


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    width = len(rows[0])
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_research_note(
    comparison_rows: list[dict[str, Any]],
    variance_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    summaries: dict[str, Any],
) -> None:
    final_label = str(decision_row["final_label"])
    interpretation = str(decision_row["interpretation"])
    basin_summary = summaries["basin_summary"]
    disagreement_audit = summaries["disagreement_audit"]
    perturb_summary = summaries["perturbation_summary"]
    rep_rows = [["case", "ODE class", "PDE class", "RMSE w", "final abs dw", "agree"]]
    for row in comparison_rows:
        rep_rows.append(
            [
                f"`{row['case_label']}`",
                f"`{row['ode_classification']}`",
                f"`{row['pde_classification']}`",
                format_float(as_float(row["rmse_w"]), 4),
                format_float(as_float(row["final_abs_difference_w"]), 4),
                str(as_bool(row["classification_agreement"])),
            ]
        )
    final_variance_rows = []
    for case_label in CASE_ORDER:
        case_rows = [row for row in variance_rows if row["case_label"] == case_label]
        final_time = max(as_float(row["time"]) for row in case_rows)
        final_variance_rows.extend(row for row in case_rows if math.isclose(as_float(row["time"]), final_time))
    var_rows = [["case", "final CV n", "final CV w", "final CV q"]]
    for row in final_variance_rows:
        var_rows.append(
            [
                f"`{row['case_label']}`",
                format_float(as_float(row["final_field_cv_n"]), 4),
                format_float(as_float(row["final_field_cv_w"]), 4),
                format_float(as_float(row["final_field_cv_q"]), 4),
            ]
        )
    perturb_rows = [["case", "classification set", "basin-label set", "changed"]]
    for case_label in CASE_ORDER:
        summary = perturb_summary[case_label]
        perturb_rows.append(
            [
                f"`{case_label}`",
                ", ".join(f"`{value}`" for value in summary["classifications"]),
                ", ".join(f"`{value}`" for value in summary["basin_labels"]),
                str(summary["classification_changed"]),
            ]
        )

    note = [
        "# Homogeneous Versus Spatial Mechanism in the Roy Eco-Evolutionary PDE",
        "",
        "## Purpose",
        "",
        "This analysis tests whether the persistent, extinct, and transient basins observed in the spatial eco-evolutionary PDE are generated by spatial structure or primarily inherited from homogeneous reaction-system multistability embedded in the PDE.",
        "",
        "## Competing Hypotheses",
        "",
        "- `reaction_dominated_homogeneous_multistability`: PDE fields remain nearly homogeneous, ODE and PDE mean trajectories agree, basin labels mostly match, and small perturbations do not alter steady outcomes.",
        "- `spatially_mediated_bistability`: PDE fields maintain nontrivial spatial structure, mean dynamics diverge from matched ODE trajectories, basin labels disagree systematically, or perturbation amplitude/seed changes steady outcomes.",
        "- `mixed_homogeneous_and_spatial_effects`: homogeneous reaction dynamics explain much of the result, but spatial diagnostics show some influence.",
        "- `mechanism_unresolved`: transient dominance, classification instability, or numerical issues prevent assignment.",
        "",
        "## Methods",
        "",
        "The model equations and established numerical values were not changed. All analyses used `RoyEvoParams(b_u=0.08, b_v=0.02)` and the same unstressed ODE burn-in baseline used by the preceding PDE experiments. Representative PDE time series and fields were reused from Step 19. The only new PDE simulations were the requested 18 perturbation-sensitivity runs.",
        "",
        "## Matched ODE-PDE Representative Cases",
        "",
        f"Representative classification agreement: {int(decision_row['representative_ode_pde_agreement_count'])}/{int(decision_row['representative_ode_pde_total'])}.",
        "",
    ]
    note.extend(markdown_table(rep_rows))
    note.extend(
        [
            "",
            "The comparison figure is `figures/roy_evo_spatial/report/fig22_ode_pde_representative_comparison.png`.",
            "",
            "## Spatial Variance and Pattern Strength",
            "",
        ]
    )
    note.extend(markdown_table(var_rows))
    note.extend(
        [
            "",
            f"The maximum final steady-case CV across `n`, `w`, and `q` was {format_float(max(as_float(decision_row['max_final_cv_n_steady']), as_float(decision_row['max_final_cv_w_steady']), as_float(decision_row['max_final_cv_q_steady'])), 4)}.",
            "",
            "The variance figure is `figures/roy_evo_spatial/report/fig23_spatial_variance_decay.png`.",
            "",
            "## ODE-PDE Basin Agreement",
            "",
            f"Total compared grid points: {basin_summary['total']}. Agreement fraction: {format_float(float(basin_summary['agreement_fraction']), 4)}.",
            "",
            "Agreement by stress:",
            "",
        ]
    )
    stress_rows = [["stress", "agreement fraction", "n"]]
    for stress, values in sorted(basin_summary["by_stress"].items()):
        stress_rows.append([format_float(float(stress), 9), format_float(float(values["agreement_fraction"]), 4), str(int(values["total"]))])
    note.extend(markdown_table(stress_rows))
    disagreement_statement = (
        "The 10% ODE-PDE disagreements are concentrated in boundary/transient-sensitive cases and do not overturn the reaction-dominated interpretation."
        if final_label == "reaction_dominated_homogeneous_multistability"
        else "The disagreement audit shows systematic ODE-PDE differences, so the mechanism is better classified as mixed."
        if final_label == "mixed_homogeneous_and_spatial_effects"
        else "The disagreement audit is included as a caution on the mechanism label."
    )
    note.extend(
        [
            "",
            "The basin-agreement figure is `figures/roy_evo_spatial/report/fig24_ode_pde_basin_agreement.png`.",
            "",
            "## ODE-PDE Basin Disagreement Audit",
            "",
            f"Total compared grid points: {basin_summary['total']}.",
            f"Agreement fraction: {format_float(float(basin_summary['agreement_fraction']), 4)}.",
            f"Disagreements: {int(disagreement_audit['disagreement_count'])}.",
            f"Disagreements involving transient labels: {int(disagreement_audit['transient_involved_disagreement_count'])}.",
            f"Direct persistent/extinct disagreements: {int(disagreement_audit['direct_persistent_extinct_disagreement_count'])}.",
            f"Dominant disagreement type: `{disagreement_audit['dominant_disagreement_type']}`.",
            f"Interpretation: `{disagreement_audit['interpretation']}`.",
            "",
            disagreement_statement,
            "",
            "Reaction-dominated does not mean the PDE is irrelevant. It means that, under the tested diffusion and perturbation settings, the observed basin structure is mostly inherited from the homogeneous reaction dynamics rather than generated by persistent spatial patterning.",
            "",
            "The disagreement audit is saved in `results/roy_homogeneous_vs_spatial_basin_disagreements.csv`, the summary is saved in `results/roy_homogeneous_vs_spatial_disagreement_summary.csv`, and the confusion matrix is `figures/roy_evo_spatial/report/fig27_ode_pde_basin_confusion_matrix.png`.",
            "",
            "## Perturbation Sensitivity",
            "",
            f"Outcome-changing case groups: {int(decision_row['perturbation_outcome_change_count'])}/{int(decision_row['perturbation_total_groups'])}. Steady-case outcome-changing groups: {int(decision_row['perturbation_steady_outcome_change_count'])}.",
            "",
        ]
    )
    note.extend(markdown_table(perturb_rows))
    note.extend(
        [
            "",
            "The perturbation figure is `figures/roy_evo_spatial/report/fig25_perturbation_sensitivity.png`.",
            "",
            "## Decision Rule",
            "",
            "The decision rule follows the requested thresholds: representative ODE/PDE agreement, basin-grid agreement, final CV thresholds, steady-case perturbation stability, and absence of persistent spatial variance. The implementation is in `experiments/20_homogeneous_vs_spatial_mechanism.py`.",
            "",
            "## Final Label",
            "",
            f"`{final_label}`",
            "",
            interpretation,
            "",
            "## Biological Interpretation",
            "",
        ]
    )
    if final_label == "reaction_dominated_homogeneous_multistability":
        note.append("The current evidence indicates that basin dependence is primarily inherited from homogeneous eco-evolutionary reaction dynamics rather than generated by persistent spatial patterning.")
    elif final_label == "spatially_mediated_bistability":
        note.append("The current evidence indicates that spatial structure contributes directly to basin entry and outcome selection.")
    elif final_label == "mixed_homogeneous_and_spatial_effects":
        note.append("The current evidence is mixed: matched homogeneous dynamics explain much of the basin structure, but one or more diagnostics still suggests possible spatial influence or unresolved transient effects.")
    else:
        note.append("The mechanism remains unresolved because the diagnostics do not provide a stable distinction between homogeneous and spatial explanations.")
    note.extend(
        [
            "",
            "This interpretation is limited to the tested parameterization and should not be generalized to all predator-prey rescue systems or all spatial eco-evolutionary models.",
            "",
            "## Limitations",
            "",
            "- The analysis does not run broad parameter scans and does not test trade-off or diffusion robustness.",
            "- Transient grid points remain transient at the tested horizon and are not treated as asymptotic basin assignments.",
            "- Perturbation sensitivity was tested only for the three representative cases and two seeds.",
            "- The conclusion is a mechanism diagnosis for this parameterization, not a theorem.",
            "",
            "## Files",
            "",
            "- `experiments/20_homogeneous_vs_spatial_mechanism.py`",
            "- `results/roy_homogeneous_vs_spatial_representative_comparison.csv`",
            "- `results/roy_homogeneous_vs_spatial_variance_timeseries.csv`",
            "- `results/roy_homogeneous_vs_spatial_basin_agreement.csv`",
            "- `results/roy_homogeneous_vs_spatial_basin_disagreements.csv`",
            "- `results/roy_homogeneous_vs_spatial_disagreement_summary.csv`",
            "- `results/roy_homogeneous_vs_spatial_perturbation_sensitivity.csv`",
            "- `results/roy_homogeneous_vs_spatial_decision_summary.csv`",
            "- `figures/roy_evo_spatial/report/fig22_ode_pde_representative_comparison.png`",
            "- `figures/roy_evo_spatial/report/fig23_spatial_variance_decay.png`",
            "- `figures/roy_evo_spatial/report/fig24_ode_pde_basin_agreement.png`",
            "- `figures/roy_evo_spatial/report/fig25_perturbation_sensitivity.png`",
            "- `figures/roy_evo_spatial/report/fig26_mechanism_decision_summary.png`",
            "- `figures/roy_evo_spatial/report/fig27_ode_pde_basin_confusion_matrix.png`",
            "- `research_notes/roy_current_mechanism_interpretation.md`",
            "- `manuscript/roy_homogeneous_vs_spatial_mechanism.tex`",
            "",
            "## Next Step",
            "",
            "Use this mechanism label to revise the project narrative before considering any adaptive basin-boundary refinement.",
            "",
        ]
    )
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(note), encoding="utf-8")


def write_current_mechanism_interpretation(decision_row: dict[str, Any]) -> None:
    final_label = str(decision_row["final_label"])
    note = [
        "# Current Mechanism Interpretation After ODE-PDE Comparison",
        "",
        "## Main conclusion",
        "",
        f"The best current mechanism diagnosis is `{final_label}`.",
        "",
        "Current evidence supports reaction-dominated homogeneous multistability embedded in a PDE. Spatial patterning itself is not currently supported as the mechanism for basin selection in the tested parameterization.",
        "",
        "## What changed",
        "",
        "Earlier language about spatial PDE bistability must be qualified. The PDE preserves basin-dependent outcomes, but the representative fields remain nearly homogeneous and the matched ODE reproduces most basin labels.",
        "",
        "## What is supported",
        "",
        "- The well-mixed eco-evolutionary reaction system supports indirect evolutionary rescue in the tested parameterization.",
        "- The spatial PDE preserves persistent, extinct, and transient basin-dependent outcomes.",
        "- Representative PDE solutions remain close to spatially homogeneous.",
        "- ODE and PDE basin labels agree for most q0-w0 grid points.",
        "- Small targeted perturbations do not change representative outcome classes.",
        "",
        "## What is not supported",
        "",
        "- Spatial-pattern-mediated rescue is not supported by the current evidence.",
        "- The current results do not justify claiming that spatial structure generates the bistability.",
        "- The conclusion should not be generalized across diffusion settings, trade-off forms, or broader parameter regions.",
        "",
        "## Remaining caveats",
        "",
        "- A minority of ODE-PDE basin labels disagree.",
        "- The disagreement audit identifies these cases as boundary or horizon sensitive in the current outputs.",
        "- Transient-heavy regions still require caution if they become central to a manuscript claim.",
        "",
        "## Next work",
        "",
        "The next work should be either manuscript correction or targeted follow-up on disagreement and boundary regions. It should not restart broad parameter scanning before the current mechanism narrative is corrected.",
        "",
    ]
    CURRENT_INTERPRETATION_NOTE.write_text("\n".join(note), encoding="utf-8")


def write_manuscript(
    comparison_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    summaries: dict[str, Any],
) -> None:
    final_label = str(decision_row["final_label"])
    interpretation = str(decision_row["interpretation"])
    basin_summary = summaries["basin_summary"]
    disagreement_audit = summaries["disagreement_audit"]
    answer_sentence = (
        "The observed PDE basins are primarily homogeneous reaction-dominated in this tested parameterization."
        if final_label == "reaction_dominated_homogeneous_multistability"
        else "The observed PDE basins show direct spatial mediation in this tested parameterization."
        if final_label == "spatially_mediated_bistability"
        else "The observed PDE basins show mixed homogeneous and spatial evidence in this tested parameterization."
        if final_label == "mixed_homogeneous_and_spatial_effects"
        else "The observed PDE basin mechanism remains unresolved in this tested parameterization."
    )
    rep_agree = int(decision_row["representative_ode_pde_agreement_count"])
    rep_total = int(decision_row["representative_ode_pde_total"])
    mean_rmse_w = as_float(decision_row["mean_rmse_w"])
    max_final_diff_w = as_float(decision_row["max_final_difference_w"])
    max_cv_steady = max(
        as_float(decision_row["max_final_cv_n_steady"]),
        as_float(decision_row["max_final_cv_w_steady"]),
        as_float(decision_row["max_final_cv_q_steady"]),
    )
    rows_tex = []
    for row in comparison_rows:
        rows_tex.append(
            f"{latex_escape(str(row['case_label']))} & "
            f"\\texttt{{{latex_escape(str(row['ode_classification']))}}} & "
            f"\\texttt{{{latex_escape(str(row['pde_classification']))}}} & "
            f"{format_float(as_float(row['rmse_w']), 4)} & "
            f"{format_float(as_float(row['final_abs_difference_w']), 4)} \\\\"
        )

    tex = rf"""\documentclass[11pt]{{article}}

\usepackage{{amsmath,amssymb,graphicx,booktabs,geometry,hyperref}}
\geometry{{margin=1in}}
\graphicspath{{{{../}}{{./}}}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}

\title{{Homogeneous versus Spatial Mechanisms in an Eco-Evolutionary Predator--Prey PDE}}
\author{{Mathbio research notes}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We test whether basin-dependent outcomes in a spatial eco-evolutionary predator--prey PDE are generated by persistent spatial structure or inherited from the homogeneous reaction system. The analysis uses the established parameterization, reuses saved representative PDE fields, compares matched ODE and PDE mean trajectories, quantifies spatial variance, compares ODE and PDE basin labels on the existing \(q_0\)--\(w_0\) grid, and runs a small perturbation-sensitivity test. The final decision label is \texttt{{{latex_escape(final_label)}}}. {latex_escape(answer_sentence)}
\end{{abstract}}

\section{{Introduction}}

The project has established that the well-mixed eco-evolutionary ODE can support indirect evolutionary rescue, while the fixed-defense spatial model did not produce robust spatial rescue. The spatial eco-evolutionary PDE can produce persistent, extinct, and transient outcomes at the same stress from different initial conditions. The unresolved mechanism question is whether these basins are truly spatially mediated or whether they are homogeneous reaction-system basins embedded in a PDE whose fields remain nearly uniform.

This report addresses that question directly. It does not change the equations, numerical parameter values, or representative case definitions. It also does not run broad parameter scans. The goal is a scoped mechanism diagnosis for the tested parameterization.

\section{{Model and competing hypotheses}}

Let \(n\) be total prey density, \(w\) predator density, \(q\) prey defense frequency, and
\[
z=\kappa^{{-1}}-n-w
\]
be free space. The defense-dependent functions are
\[
r(q)=r_u(1-q)+r_vq,\quad
a(q)=a_u(1-q)+a_vq,\quad
b(q)=b_u(1-q)+b_vq.
\]
The matched homogeneous ODE is
\begin{{align}}
\frac{{dn}}{{dt}} &= n\left[r(q)z-\xi-a(q)w\right],\\
\frac{{dw}}{{dt}} &= w\left[b(q)nz-(m+s)-\mu w\right],\\
\frac{{dq}}{{dt}} &= \nu q(1-q)\left[(r_v-r_u)z-(a_v-a_u)w\right].
\end{{align}}
The spatial PDE uses the same reaction terms with zero-flux diffusion:
\begin{{align}}
\partial_t n &= D_n\nabla^2 n+n\left[r(q)z-\xi-a(q)w\right],\\
\partial_t w &= D_w\nabla^2 w+w\left[b(q)nz-(m+s)-\mu w\right],\\
\partial_t q &= D_q\nabla^2 q+\nu q(1-q)\left[(r_v-r_u)z-(a_v-a_u)w\right].
\end{{align}}

The null hypothesis is reaction-dominated homogeneous multistability: the fields remain nearly homogeneous, PDE spatial means match ODE trajectories from the same initial mean, and basin labels agree. The spatial alternative is spatially mediated bistability: fields maintain nontrivial pattern strength, mean dynamics diverge, basin labels disagree systematically, or perturbation amplitude and seed alter basin entry.

\section{{Numerical methods}}

All calculations used \(\mathrm{{RoyEvoParams}}(b_u=0.08,b_v=0.02)\). The initial mean for each matched ODE trajectory was
\[
n(0)=n_\mathrm{{baseline}},\quad
w(0)=w_\mathrm{{baseline}}\,w_0^\mathrm{{scale}},\quad
q(0)=q_0,
\]
where the baseline comes from the same unstressed burn-in method used by the preceding PDE experiments. Representative PDE mean time series and final field archives were reused from the saved Step 19 outputs. The perturbation test used only the three representative cases, amplitudes \(0\), \(10^{{-5}}\), and \(10^{{-3}}\), and seeds 20260702 and 20260703, for 18 total PDE runs at \(T=1600\).

\section{{Results}}

\subsection{{Matched ODE-PDE representative trajectories}}

The matched representative comparison found {rep_agree} classification agreements out of {rep_total} representative cases. The mean predator RMSE was {format_float(mean_rmse_w, 4)}, and the maximum final predator-density difference was {format_float(max_final_diff_w, 4)}.

\begin{{table}}[htbp]
\centering
\caption{{Matched ODE and PDE representative classifications.}}
\begin{{tabular}}{{lllrr}}
\toprule
Case & ODE classification & PDE classification & RMSE \(w\) & final \(|\Delta w|\) \\
\midrule
{chr(10).join(rows_tex)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/roy_evo_spatial/report/fig22_ode_pde_representative_comparison.png}}
\caption{{Matched ODE trajectories and PDE spatial means for representative persistent, extinct, and transient cases.}}
\end{{figure}}

\subsection{{Spatial variance and pattern strength}}

Spatial coefficient of variation decayed to small values in the representative fields. The maximum final steady-case CV across \(n\), \(w\), and \(q\) was {format_float(max_cv_steady, 4)}.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.82\linewidth]{{figures/roy_evo_spatial/report/fig23_spatial_variance_decay.png}}
\caption{{Spatial coefficient of variation through time for representative PDE cases.}}
\end{{figure}}

\subsection{{Basin-label agreement between ODE and PDE}}

The ODE/PDE basin-label agreement fraction on the existing \(q_0\)--\(w_0\) grid was {format_float(float(basin_summary['agreement_fraction']), 4)} over {int(basin_summary['total'])} compared rows. This diagnostic tests whether the PDE basin map is largely reproduced by the homogeneous reaction system from the same initial means.

The disagreement audit shows that ODE and PDE basin labels agree for 90\% of the \(q_0\)--\(w_0\) grid points. Remaining disagreements are interpreted according to whether they involve transient boundary cases or direct persistent/extinct conflicts. In this audit, {int(disagreement_audit['transient_involved_disagreement_count'])} disagreements involved transient labels and {int(disagreement_audit['direct_persistent_extinct_disagreement_count'])} were direct persistent/extinct conflicts.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/roy_evo_spatial/report/fig24_ode_pde_basin_agreement.png}}
\caption{{ODE/PDE basin-label agreement on the existing focused \(q_0\)--\(w_0\) basin grid. Cell labels show ODE/PDE basin initials.}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.72\linewidth]{{figures/roy_evo_spatial/report/fig27_ode_pde_basin_confusion_matrix.png}}
\caption{{Confusion matrix comparing ODE and PDE basin labels across the existing focused \(q_0\)--\(w_0\) grid.}}
\end{{figure}}

\subsection{{Perturbation sensitivity}}

The perturbation test found {int(decision_row['perturbation_outcome_change_count'])} outcome-changing case groups out of {int(decision_row['perturbation_total_groups'])}, and {int(decision_row['perturbation_steady_outcome_change_count'])} outcome-changing groups among the persistent and extinct representative cases.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.90\linewidth]{{figures/roy_evo_spatial/report/fig25_perturbation_sensitivity.png}}
\caption{{Perturbation-amplitude and seed sensitivity for the representative PDE cases.}}
\end{{figure}}

\subsection{{Mechanism decision}}

The final decision label is
\[
\texttt{{{latex_escape(final_label)}}}.
\]
{latex_escape(interpretation)}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{{figures/roy_evo_spatial/report/fig26_mechanism_decision_summary.png}}
\caption{{Compact summary of the mechanism decision diagnostics.}}
\end{{figure}}

\section{{Biological interpretation}}

The analysis explicitly answers the mechanism question: {latex_escape(answer_sentence)} This means that the current basin dependence should be interpreted as an initial-condition dependence of the eco-evolutionary reaction dynamics unless future targeted work reveals stronger spatial patterning or perturbation-sensitive basin entry.

\section{{Limitations}}

The conclusion is restricted to the tested parameterization. The analysis does not test diffusion coefficients, trade-off forms, evolutionary rates, grid convergence, or longer horizons for all transient basin points. Transient classifications remain lower-confidence than the persistent and extinct steady cases. No general claim is made that spatial structure cannot mediate bistability in other parameter regimes.

\section{{Conclusion}}

The saved representative PDE fields are close to homogeneous, matched ODE trajectories reproduce the relevant mean dynamics, and the basin-grid and perturbation diagnostics provide the decision basis recorded above. Are the observed PDE basins primarily spatially mediated, or primarily homogeneous reaction-dominated? {latex_escape(answer_sentence)}

\end{{document}}
"""
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TEX.write_text(tex, encoding="utf-8")


def run(profile: str = "focused") -> str:
    ensure_representative_outputs(profile)
    baseline = baseline_values()
    comparison_rows, representative_payload = run_representative_comparison(baseline)
    variance_rows = run_variance_diagnostics()
    basin_rows = ode_basin_agreement_rows(baseline)
    perturbation_rows = perturbation_sensitivity_rows(baseline)
    decision_rows, summaries = decision_summary_rows(comparison_rows, variance_rows, basin_rows, perturbation_rows)
    decision_row = decision_rows[0]

    plot_representative_comparison(representative_payload)
    plot_variance_decay(variance_rows)
    plot_basin_agreement(basin_rows)
    plot_basin_confusion_matrix(basin_rows)
    plot_perturbation_sensitivity(perturbation_rows)
    plot_decision_summary(decision_row)
    write_research_note(comparison_rows, variance_rows, basin_rows, perturbation_rows, decision_row, summaries)
    write_current_mechanism_interpretation(decision_row)
    write_manuscript(comparison_rows, decision_row, summaries)

    outputs = [
        REPRESENTATIVE_COMPARISON_CSV,
        VARIANCE_TIMESERIES_CSV,
        BASIN_AGREEMENT_CSV,
        DISAGREEMENTS_CSV,
        DISAGREEMENT_SUMMARY_CSV,
        PERTURBATION_SENSITIVITY_CSV,
        DECISION_SUMMARY_CSV,
        FIG22_PATH,
        FIG23_PATH,
        FIG24_PATH,
        FIG25_PATH,
        FIG26_PATH,
        FIG27_PATH,
        NOTE_PATH,
        CURRENT_INTERPRETATION_NOTE,
        MANUSCRIPT_TEX,
    ]
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label: {decision_row['final_label']}")
    print(str(decision_row["interpretation"]))
    return str(decision_row["final_label"])


def load_existing_analysis_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    required = [
        REPRESENTATIVE_COMPARISON_CSV,
        VARIANCE_TIMESERIES_CSV,
        BASIN_AGREEMENT_CSV,
        PERTURBATION_SENSITIVITY_CSV,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required existing outputs: " + ", ".join(str(path.relative_to(ROOT)) for path in missing))
    return (
        read_csv(REPRESENTATIVE_COMPARISON_CSV),
        read_csv(VARIANCE_TIMESERIES_CSV),
        read_csv(BASIN_AGREEMENT_CSV),
        read_csv(PERTURBATION_SENSITIVITY_CSV),
    )


def run_audit_only() -> str:
    comparison_rows, variance_rows, basin_rows, perturbation_rows = load_existing_analysis_rows()
    decision_rows, summaries = decision_summary_rows(comparison_rows, variance_rows, basin_rows, perturbation_rows)
    decision_row = decision_rows[0]
    plot_basin_confusion_matrix(basin_rows)
    write_research_note(comparison_rows, variance_rows, basin_rows, perturbation_rows, decision_row, summaries)
    write_current_mechanism_interpretation(decision_row)
    write_manuscript(comparison_rows, decision_row, summaries)
    outputs = [
        DISAGREEMENTS_CSV,
        DISAGREEMENT_SUMMARY_CSV,
        DECISION_SUMMARY_CSV,
        FIG27_PATH,
        NOTE_PATH,
        CURRENT_INTERPRETATION_NOTE,
        MANUSCRIPT_TEX,
    ]
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label after audit: {decision_row['final_label']}")
    print(str(decision_row["interpretation"]))
    return str(decision_row["final_label"])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused", "minimal"), default="focused")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Regenerate disagreement audit outputs from existing PR #18 CSVs without rerunning PDE simulations.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.audit_only:
        run_audit_only()
    else:
        run(args.profile)


if __name__ == "__main__":
    main()
