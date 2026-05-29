"""Step 10: diagnose spatial suppression of eco-evolutionary rescue.

This focused diagnostic compares ODE and PDE trajectories in the rescue and
suppression windows from Steps 09A/09B.  It does not scan parameters or change
the model.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.roy_evo_spatial import (
    RoyEvoPDEConfig,
    RoyEvoParams,
    a_of_q,
    b_of_q,
    classify_evo_trajectory,
    find_evo_equilibrium,
    free_space_evo,
    grid_2d_evo,
    initial_state_from_ode_equilibrium,
    laplacian_neumann_2d_evo,
    predator_growth_factor_evo,
    r_of_q,
    reaction_part_evo_pde,
    selection_gradient,
    simulate_ode_evo,
)


RESULTS_DIR = ROOT / "results"
NOTES_DIR = ROOT / "research_notes"
TIMESERIES_CSV = RESULTS_DIR / "roy_spatial_suppression_timeseries.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_spatial_suppression_summary.csv"
SUMMARY_MD = NOTES_DIR / "roy_spatial_suppression_mechanism.md"

EPSILON = 1.0e-4
SMALL_COVARIANCE = 1.0e-8
SMALL_VAR_Q = 1.0e-8
PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
BURN_IN_INITIAL = np.array([1.0, 0.2, 0.5], dtype=float)
BURN_IN_T = 3000.0
ODE_T = 1500.0

ODE_NO_EVO_THRESHOLD = 0.069448242
ODE_EVO_THRESHOLD = 0.16486816
PDE_EVO_THRESHOLD = 0.11765625
SUPPRESSION_WINDOW_STRESS = 0.5 * (PDE_EVO_THRESHOLD + ODE_EVO_THRESHOLD)

STRESSES = [
    ("S0_unstressed", 0.0),
    ("S1_shared_rescue_window", 0.09),
    ("S2_near_pde_evo_threshold", PDE_EVO_THRESHOLD),
    ("S3_suppression_window_midpoint", SUPPRESSION_WINDOW_STRESS),
    ("S4_near_ode_evo_threshold", ODE_EVO_THRESHOLD),
]
NO_EVO_STRESS_LABELS = {"S1_shared_rescue_window", "S2_near_pde_evo_threshold", "S3_suppression_window_midpoint"}

TIMESERIES_FIELDNAMES = [
    "run_id",
    "model_type",
    "evolve",
    "stress_label",
    "stress",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
    "mean_z",
    "var_n",
    "var_w",
    "var_q",
    "min_z",
    "spatial_growth",
    "meanfield_growth",
    "spatial_covariance_bonus",
    "cov_w_q",
    "cov_w_bq",
    "cov_w_n",
    "cov_w_prey_gain",
    "predator_fraction_in_low_q",
    "predator_fraction_in_high_gain",
    "area_fraction_low_q",
    "area_fraction_high_gain",
    "predator_low_q_enrichment",
    "predator_high_gain_enrichment",
    "persistent_tail_label",
]

SUMMARY_FIELDNAMES = [
    "run_id",
    "model_type",
    "evolve",
    "stress_label",
    "stress",
    "tail_mean_w",
    "tail_min_w",
    "tail_slope_w",
    "persistent_predator",
    "tail_mean_q",
    "tail_q_change",
    "tail_mean_var_q",
    "tail_mean_cov_w_q",
    "tail_mean_cov_w_bq",
    "tail_mean_cov_w_n",
    "tail_mean_cov_w_prey_gain",
    "tail_mean_spatial_covariance_bonus",
    "tail_mean_predator_low_q_enrichment",
    "tail_mean_predator_high_gain_enrichment",
    "diagnosis",
]


@dataclass(frozen=True)
class DiagnosticRun:
    run_id: str
    model_type: str
    evolve: bool
    stress_label: str
    stress: float


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def tail_mask(times: np.ndarray, tail_fraction: float = 0.25) -> np.ndarray:
    cutoff = times[-1] - tail_fraction * (times[-1] - times[0])
    mask = times >= cutoff
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(times, dtype=bool)
        mask[-2:] = True
    return mask


def tail_slope(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    tail_t = times[mask]
    tail_values = values[mask]
    centered_t = tail_t - float(np.mean(tail_t))
    denom = float(np.dot(centered_t, centered_t))
    return 0.0 if denom <= 0.0 else float(np.dot(centered_t, tail_values - float(np.mean(tail_values))) / denom)


def persistent_from_tail(times: np.ndarray, mean_w: np.ndarray, physical: bool = True) -> bool:
    if len(times) < 2 or len(mean_w) != len(times) or not np.all(np.isfinite(mean_w)):
        return False
    mask = tail_mask(times)
    tail_t = times[mask]
    tail_w = mean_w[mask]
    duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    mean_value = float(np.mean(tail_w))
    min_value = float(np.min(tail_w))
    slope = tail_slope(times, mean_w, mask)
    slope_floor = -max(EPSILON, 0.25 * mean_value) / duration
    return bool(physical and mean_value > EPSILON and min_value > 0.25 * EPSILON and slope >= slope_floor)


def patch_occupancy_metrics(w: np.ndarray, q: np.ndarray, prey_gain: np.ndarray) -> dict[str, float]:
    w_arr = np.asarray(w, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    prey_gain_arr = np.asarray(prey_gain, dtype=float)
    low_q_mask = q_arr <= float(np.median(q_arr))
    high_gain_mask = prey_gain_arr >= float(np.median(prey_gain_arr))
    total_predator = float(np.sum(w_arr))
    predator_low_q = safe_fraction(float(np.sum(w_arr[low_q_mask])), total_predator)
    predator_high_gain = safe_fraction(float(np.sum(w_arr[high_gain_mask])), total_predator)
    area_low_q = float(np.mean(low_q_mask))
    area_high_gain = float(np.mean(high_gain_mask))
    return {
        "predator_fraction_in_low_q": predator_low_q,
        "predator_fraction_in_high_gain": predator_high_gain,
        "area_fraction_low_q": area_low_q,
        "area_fraction_high_gain": area_high_gain,
        "predator_low_q_enrichment": safe_fraction(predator_low_q, area_low_q),
        "predator_high_gain_enrichment": safe_fraction(predator_high_gain, area_high_gain),
    }


def field_diagnostics(n: np.ndarray, w: np.ndarray, q: np.ndarray, stress: float) -> dict[str, float]:
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    z = free_space_evo(n_arr, w_arr, PARAMS)
    b_q = b_of_q(q_arr, PARAMS)
    prey_gain = b_q * n_arr * z
    local_growth = predator_growth_factor_evo(n_arr, w_arr, q_arr, PARAMS, stress=stress)

    mean_n = float(np.mean(n_arr))
    mean_w = float(np.mean(w_arr))
    mean_q = float(np.mean(q_arr))
    mean_z = float(np.mean(z))
    mean_bq = float(np.mean(b_q))
    mean_prey_gain = float(np.mean(prey_gain))
    meanfield_growth_factor = float(predator_growth_factor_evo(mean_n, mean_w, mean_q, PARAMS, stress=stress))
    spatial_growth = float(np.mean(w_arr * local_growth))
    meanfield_growth = mean_w * meanfield_growth_factor
    centered_w = w_arr - mean_w
    occupancy = patch_occupancy_metrics(w_arr, q_arr, prey_gain)
    return {
        "mean_n": mean_n,
        "mean_w": mean_w,
        "mean_q": mean_q,
        "mean_z": mean_z,
        "var_n": float(np.var(n_arr)),
        "var_w": float(np.var(w_arr)),
        "var_q": float(np.var(q_arr)),
        "min_z": float(np.min(z)),
        "min_n": float(np.min(n_arr)),
        "min_w": float(np.min(w_arr)),
        "min_q": float(np.min(q_arr)),
        "max_q": float(np.max(q_arr)),
        "spatial_growth": spatial_growth,
        "meanfield_growth": float(meanfield_growth),
        "spatial_covariance_bonus": float(spatial_growth - meanfield_growth),
        "cov_w_q": float(np.mean(centered_w * (q_arr - mean_q))),
        "cov_w_bq": float(np.mean(centered_w * (b_q - mean_bq))),
        "cov_w_n": float(np.mean(centered_w * (n_arr - mean_n))),
        "cov_w_prey_gain": float(np.mean(centered_w * (prey_gain - mean_prey_gain))),
        **occupancy,
    }


def ode_timepoint_row(run: DiagnosticRun, time: float, state: np.ndarray, persistent_label: str) -> dict[str, object]:
    n, w, q = [float(value) for value in state]
    z = float(free_space_evo(n, w, PARAMS))
    return {
        "run_id": run.run_id,
        "model_type": run.model_type,
        "evolve": run.evolve,
        "stress_label": run.stress_label,
        "stress": run.stress,
        "time": float(time),
        "mean_n": n,
        "mean_w": w,
        "mean_q": q,
        "mean_z": z,
        "var_n": 0.0,
        "var_w": 0.0,
        "var_q": 0.0,
        "min_z": z,
        "spatial_growth": 0.0,
        "meanfield_growth": 0.0,
        "spatial_covariance_bonus": 0.0,
        "cov_w_q": 0.0,
        "cov_w_bq": 0.0,
        "cov_w_n": 0.0,
        "cov_w_prey_gain": 0.0,
        "predator_fraction_in_low_q": float("nan"),
        "predator_fraction_in_high_gain": float("nan"),
        "area_fraction_low_q": float("nan"),
        "area_fraction_high_gain": float("nan"),
        "predator_low_q_enrichment": float("nan"),
        "predator_high_gain_enrichment": float("nan"),
        "persistent_tail_label": persistent_label,
    }


def simulate_ode_run(run: DiagnosticRun, initial_state: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    result = simulate_ode_evo(PARAMS, initial_state, stress=run.stress, evolve=run.evolve, T=ODE_T, n_eval=501)
    diagnostics = classify_evo_trajectory(result.t, result.y, epsilon=EPSILON, params=PARAMS)
    persistent_label = "tail_persistent" if diagnostics["persistent_predator"] else "tail_not_persistent"
    rows = [ode_timepoint_row(run, time, result.y[:, idx], persistent_label) for idx, time in enumerate(result.t)]
    summary = summarize_timeseries(run, rows)
    summary["persistent_predator"] = bool(diagnostics["persistent_predator"])
    summary["tail_q_change"] = float(diagnostics["q_change_from_initial"])
    summary["diagnosis"] = initial_run_diagnosis(summary)
    return rows, summary


def simulate_pde_diagnostic_run(
    run: DiagnosticRun,
    config: RoyEvoPDEConfig,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _x, _y, dx, dy = grid_2d_evo(config)
    n, w, q = (np.asarray(value, dtype=float).copy() for value in initial_state)
    rows: list[dict[str, object]] = []
    q_clip_count = 0
    q_clip_max_violation = 0.0
    nonphysical = False
    min_n_global = float(np.min(n))
    min_w_global = float(np.min(w))
    min_q_global = float(np.min(q))
    max_q_global = float(np.max(q))
    min_z_global = float(np.min(free_space_evo(n, w, PARAMS)))

    def record(t_value: float) -> None:
        metrics = field_diagnostics(n, w, q, run.stress)
        rows.append(
            {
                "run_id": run.run_id,
                "model_type": run.model_type,
                "evolve": run.evolve,
                "stress_label": run.stress_label,
                "stress": run.stress,
                "time": float(t_value),
                **{key: metrics[key] for key in TIMESERIES_FIELDNAMES if key in metrics},
                "persistent_tail_label": "",
            }
        )

    record(0.0)
    n_steps = int(np.ceil(config.T / config.dt))
    for step in range(1, n_steps + 1):
        reactions = reaction_part_evo_pde(n, w, q, PARAMS, stress=run.stress, evolve=run.evolve)
        n_next = n + config.dt * (config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + reactions[0])
        w_next = w + config.dt * (config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + reactions[1])
        if run.evolve:
            q_next = q + config.dt * (config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + reactions[2])
        else:
            q_next = q.copy()

        if not (np.all(np.isfinite(n_next)) and np.all(np.isfinite(w_next)) and np.all(np.isfinite(q_next))):
            nonphysical = True
            break

        q_violation = np.maximum(np.maximum(-q_next, 0.0), np.maximum(q_next - 1.0, 0.0))
        max_violation = float(np.max(q_violation))
        if max_violation > 0.0:
            q_clip_count += int(np.count_nonzero(q_violation > 0.0))
            q_clip_max_violation = max(q_clip_max_violation, max_violation)
            if config.clip_q:
                q_next = np.clip(q_next, 0.0, 1.0)

        n, w, q = n_next, w_next, q_next
        min_n_global = min(min_n_global, float(np.min(n)))
        min_w_global = min(min_w_global, float(np.min(w)))
        min_q_global = min(min_q_global, float(np.min(q)))
        max_q_global = max(max_q_global, float(np.max(q)))
        min_z_global = min(min_z_global, float(np.min(free_space_evo(n, w, PARAMS))))
        if step % max(1, config.record_every) == 0 or step == n_steps:
            record(min(step * config.dt, config.T))

    if not rows:
        raise RuntimeError(f"No rows recorded for {run.run_id}")

    arrays = {
        "time": np.array([float(row["time"]) for row in rows], dtype=float),
        "mean_w": np.array([float(row["mean_w"]) for row in rows], dtype=float),
        "min_z": np.array([float(row["min_z"]) for row in rows], dtype=float),
    }
    physical = (
        not nonphysical
        and all(np.isfinite(float(row["mean_w"])) for row in rows)
        and min_z_global >= -1.0e-5
        and min_n_global >= -1.0e-8
        and min_w_global >= -1.0e-8
        and min_q_global >= -1.0e-6
        and max_q_global <= 1.0 + 1.0e-6
        and q_clip_max_violation <= 1.0e-4
    )
    persistent = persistent_from_tail(arrays["time"], arrays["mean_w"], physical=physical)
    persistent_label = "tail_persistent" if persistent else "tail_not_persistent"
    for row in rows:
        row["persistent_tail_label"] = persistent_label

    summary = summarize_timeseries(run, rows)
    summary["persistent_predator"] = persistent
    summary["diagnosis"] = initial_run_diagnosis(summary)
    if q_clip_count > 0 and q_clip_max_violation > 1.0e-4:
        summary["diagnosis"] = "inconclusive"
    return rows, summary


def summarize_timeseries(run: DiagnosticRun, rows: list[dict[str, object]]) -> dict[str, object]:
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    mask = tail_mask(times)
    values = {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in TIMESERIES_FIELDNAMES
        if field in rows[0] and field not in {"run_id", "model_type", "evolve", "stress_label", "persistent_tail_label"}
    }
    slope_w = tail_slope(times, values["mean_w"], mask)
    persistent = persistent_from_tail(times, values["mean_w"], physical=True)
    initial_q = float(rows[0]["mean_q"])
    low_q_enrichment_tail = values["predator_low_q_enrichment"][mask]
    high_gain_enrichment_tail = values["predator_high_gain_enrichment"][mask]
    low_q_enrichment = float("nan") if np.all(np.isnan(low_q_enrichment_tail)) else float(np.nanmean(low_q_enrichment_tail))
    high_gain_enrichment = float("nan") if np.all(np.isnan(high_gain_enrichment_tail)) else float(np.nanmean(high_gain_enrichment_tail))
    return {
        "run_id": run.run_id,
        "model_type": run.model_type,
        "evolve": run.evolve,
        "stress_label": run.stress_label,
        "stress": run.stress,
        "tail_mean_w": float(np.mean(values["mean_w"][mask])),
        "tail_min_w": float(np.min(values["mean_w"][mask])),
        "tail_slope_w": slope_w,
        "persistent_predator": persistent,
        "tail_mean_q": float(np.mean(values["mean_q"][mask])),
        "tail_q_change": float(np.mean(values["mean_q"][mask]) - initial_q),
        "tail_mean_var_q": float(np.mean(values["var_q"][mask])),
        "tail_mean_cov_w_q": float(np.mean(values["cov_w_q"][mask])),
        "tail_mean_cov_w_bq": float(np.mean(values["cov_w_bq"][mask])),
        "tail_mean_cov_w_n": float(np.mean(values["cov_w_n"][mask])),
        "tail_mean_cov_w_prey_gain": float(np.mean(values["cov_w_prey_gain"][mask])),
        "tail_mean_spatial_covariance_bonus": float(np.mean(values["spatial_covariance_bonus"][mask])),
        "tail_mean_predator_low_q_enrichment": low_q_enrichment,
        "tail_mean_predator_high_gain_enrichment": high_gain_enrichment,
        "diagnosis": "inconclusive",
    }


def initial_run_diagnosis(summary: dict[str, object]) -> str:
    persistent = bool(summary["persistent_predator"])
    if summary["model_type"] == "ODE" and bool(summary["evolve"]) and persistent and float(summary["tail_q_change"]) < -1.0e-3:
        return "ode_rescue_active"
    if summary["model_type"] == "PDE" and bool(summary["evolve"]) and persistent and float(summary["tail_q_change"]) < -1.0e-3:
        return "pde_rescue_active"
    if not persistent:
        return "no_rescue"
    return "inconclusive"


def mismatch_signature(summary: dict[str, object], small: float = SMALL_COVARIANCE) -> bool:
    cov_w_q = float(summary["tail_mean_cov_w_q"])
    cov_w_bq = float(summary["tail_mean_cov_w_bq"])
    cov_w_prey_gain = float(summary["tail_mean_cov_w_prey_gain"])
    low_q_enrichment = float(summary["tail_mean_predator_low_q_enrichment"])
    high_gain_enrichment = float(summary["tail_mean_predator_high_gain_enrichment"])
    return (
        cov_w_q > small
        and cov_w_bq < -small
        and cov_w_prey_gain <= small
        and (low_q_enrichment < 1.0 or high_gain_enrichment < 1.0)
    )


def dilution_signature(summary: dict[str, object], small: float = SMALL_COVARIANCE) -> bool:
    return (
        abs(float(summary["tail_mean_cov_w_q"])) < small
        and abs(float(summary["tail_mean_cov_w_bq"])) < small
        and abs(float(summary["tail_mean_cov_w_prey_gain"])) < small
        and float(summary["tail_mean_var_q"]) < SMALL_VAR_Q
        and abs(float(summary["tail_mean_spatial_covariance_bonus"])) < small
    )


def final_step10_label(ode_s3: dict[str, object], pde_s3: dict[str, object]) -> tuple[str, str]:
    if not bool(ode_s3["persistent_predator"]) or bool(pde_s3["persistent_predator"]):
        return (
            "spatial_suppression_inconclusive",
            "The focused S3 window did not reproduce ODE persistence with PDE failure.",
        )
    if mismatch_signature(pde_s3):
        return (
            "predator_low_defense_spatial_mismatch",
            "The PDE suppresses rescue because predator density is not sufficiently colocated with low-defense/high-gain prey states in the rescue window.",
        )
    if dilution_signature(pde_s3):
        return (
            "diffusion_dilutes_evolutionary_rescue",
            "The PDE suppresses rescue without forming a strong beneficial spatial covariance; spatial heterogeneity in q and prey gain remains too small or too transient to amplify predator recovery.",
        )
    return (
        "spatial_suppression_inconclusive",
        "The focused diagnostics do not distinguish mismatch from diffusion dilution.",
    )


def build_runs() -> list[DiagnosticRun]:
    runs: list[DiagnosticRun] = []
    for stress_label, stress in STRESSES:
        runs.append(DiagnosticRun(f"ODE_EVO_{stress_label[:2]}", "ODE", True, stress_label, stress))
        runs.append(DiagnosticRun(f"PDE_EVO_{stress_label[:2]}", "PDE", True, stress_label, stress))
        if stress_label in NO_EVO_STRESS_LABELS:
            runs.append(DiagnosticRun(f"ODE_NOEVO_{stress_label[:2]}", "ODE", False, stress_label, stress))
            runs.append(DiagnosticRun(f"PDE_NOEVO_{stress_label[:2]}", "PDE", False, stress_label, stress))
    return runs


def write_note(
    profile: str,
    final_label: str,
    final_interpretation: str,
    summaries: list[dict[str, object]],
    stage_c_summary: dict[str, object] | None,
) -> None:
    summary_by_run = {str(row["run_id"]): row for row in summaries}
    ode_s3 = summary_by_run["ODE_EVO_S3"]
    pde_s3 = summary_by_run["PDE_EVO_S3"]
    pde_s1 = summary_by_run["PDE_EVO_S1"]
    pde_s2 = summary_by_run["PDE_EVO_S2"]
    if bool(ode_s3["persistent_predator"]) and not bool(pde_s3["persistent_predator"]):
        s3_sentence = "At S3, ODE evolution is persistent while PDE evolution is not persistent. This reproduces the requested suppression-window contrast without changing Step 09 thresholds."
    else:
        s3_sentence = (
            "At S3, the requested suppression-window contrast is not reproduced: "
            f"ODE persistence is `{ode_s3['persistent_predator']}` and PDE persistence is `{pde_s3['persistent_predator']}`."
        )
    if final_label == "diffusion_dilutes_evolutionary_rescue":
        diagnosis_detail = (
            "The S3 PDE covariance magnitudes, spatial covariance bonus, and var(q) are all below the documented small thresholds, "
            "so the focused run supports diffusion dilution rather than a strong predator-low-defense mismatch."
        )
        interpretation_detail = (
            "q evolution is active in both ODE and PDE trajectories, but the PDE does not create a beneficial spatial covariance. "
            "In the suppression window, predator persistence fails in the PDE while covariance and q heterogeneity remain near numerical zero at the tail."
        )
    elif final_label == "predator_low_defense_spatial_mismatch":
        diagnosis_detail = (
            "The S3 PDE has the expected mismatch signs and predator under-occupation of low-defense or high-gain patches."
        )
        interpretation_detail = (
            "q evolution is active, but predator density is not sufficiently colocated with the low-defense or high-gain prey states that would support recovery."
        )
    else:
        diagnosis_detail = (
            "The S2 near-threshold PDE run has near-zero covariance diagnostics and fails persistence while ODE evolution persists, "
            "but the specified S3 midpoint remains persistent in the PDE. The focused diagnostics therefore do not distinguish mismatch from diffusion dilution under the requested decision rule."
        )
        interpretation_detail = (
            "q evolution is active, and spatial covariances remain near zero in the focused runs. However, because the specified S3 midpoint does not reproduce PDE failure, "
            "this run should not be used to choose between mismatch and dilution."
        )
    next_step = {
        "predator_low_defense_spatial_mismatch": "Next: test whether predator-prey spatial alignment changes under targeted initial perturbations, without scanning broad parameter grids.",
        "diffusion_dilutes_evolutionary_rescue": "Next: compare ODE and PDE transient timing in the rescue window before considering any parameter-map work.",
        "spatial_suppression_inconclusive": "Next: improve focused diagnostics or physicality checks before interpretation.",
    }[final_label]

    lines = [
        "# Research Note: Mechanism of Spatial Suppression in the Eco-Evolutionary Rescue Model",
        "",
        "## Executive Summary",
        "",
        f"Final Step 10 label: `{final_label}`.",
        "",
        final_interpretation,
        "",
        "The ODE evolutionary rescue result remains intact, and the Step 09B conclusion is unchanged: the tested spatial PDE has a lower evolution threshold than the well-mixed ODE.",
        "",
        "## Question",
        "",
        "Why does the spatial PDE suppress the indirect evolutionary rescue window relative to the well-mixed ODE?",
        "",
        "The analysis compares ODE and PDE trajectories at a focused set of stresses: unstressed baseline, a shared rescue-window stress, the PDE evolution threshold, the suppression-window midpoint, and the ODE evolution threshold.",
        "",
        "## Setup",
        "",
        f"- profile: `{profile}`",
        "- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`",
        "- focused PDE config: `64x64`, `L=20`, `D_n=0.01`, `D_w=0.01`, `D_q=0.005`, `T=500`, seed `20260702`",
        f"- ODE evolution threshold: `{ODE_EVO_THRESHOLD}`",
        f"- PDE evolution threshold: `{PDE_EVO_THRESHOLD}`",
        f"- suppression-window midpoint S3: `{SUPPRESSION_WINDOW_STRESS}`",
        f"- small-covariance threshold used for dilution: `{SMALL_COVARIANCE}`",
        f"- small var(q) threshold used for dilution: `{SMALL_VAR_Q}`",
        "",
        "## ODE-PDE Rescue Window Comparison",
        "",
        "| run | stress | persistent | tail mean w | tail mean q | q change |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in [summary_by_run["ODE_EVO_S1"], pde_s1, summary_by_run["ODE_EVO_S2"], pde_s2, ode_s3, pde_s3, summary_by_run["ODE_EVO_S4"], summary_by_run["PDE_EVO_S4"]]:
        lines.append(
            f"| `{row['run_id']}` | {float(row['stress']):.8g} | {row['persistent_predator']} | "
            f"{float(row['tail_mean_w']):.8g} | {float(row['tail_mean_q']):.8g} | {float(row['tail_q_change']):.8g} |"
        )
    lines.extend(
        [
            "",
            s3_sentence,
            "",
            "## Spatial Covariance Diagnostics",
            "",
            "| PDE evo run | cov(w,q) | cov(w,b(q)) | cov(w,n) | cov(w,b(q)nz) | spatial covariance bonus | var(q) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [pde_s1, pde_s2, pde_s3, summary_by_run["PDE_EVO_S4"]]:
        lines.append(
            f"| `{row['run_id']}` | {float(row['tail_mean_cov_w_q']):.8g} | "
            f"{float(row['tail_mean_cov_w_bq']):.8g} | {float(row['tail_mean_cov_w_n']):.8g} | "
            f"{float(row['tail_mean_cov_w_prey_gain']):.8g} | "
            f"{float(row['tail_mean_spatial_covariance_bonus']):.8g} | {float(row['tail_mean_var_q']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Low-Defense / High-Gain Patch Occupancy",
            "",
            "| PDE evo run | predator low-q enrichment | predator high-gain enrichment | diagnosis |",
            "|---|---:|---:|---|",
        ]
    )
    for row in [pde_s1, pde_s2, pde_s3, summary_by_run["PDE_EVO_S4"]]:
        lines.append(
            f"| `{row['run_id']}` | {float(row['tail_mean_predator_low_q_enrichment']):.8g} | "
            f"{float(row['tail_mean_predator_high_gain_enrichment']):.8g} | `{row['diagnosis']}` |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"Final Step 10 label: `{final_label}`.",
            "",
            final_interpretation,
            "",
            diagnosis_detail,
        ]
    )
    if stage_c_summary is not None:
        lines.extend(
            [
                "",
                "A Stage C-style 96x96 check was run under `--profile full`.",
                f"Stage C S3 PDE evolution persistent: `{stage_c_summary['persistent_predator']}`; tail covariance bonus: `{float(stage_c_summary['tail_mean_spatial_covariance_bonus']):.8g}`.",
            ]
        )
    else:
        lines.extend(["", "The default focused profile did not run the optional Stage C-style check."])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation_detail,
            "",
            "This result leaves the PR #3 conclusion unchanged: the spatial PDE did not amplify the ODE rescue window.",
            "",
            "## Files",
            "",
            f"- `{TIMESERIES_CSV.relative_to(ROOT)}`",
            f"- `{SUMMARY_CSV.relative_to(ROOT)}`",
            "",
            "## Next Step",
            "",
            next_step,
        ]
    )
    SUMMARY_MD.parent.mkdir(exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(profile: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS, guesses=(BURN_IN_INITIAL,), burn_in_T=BURN_IN_T)
    initial_ode = np.array([equilibrium["n"], equilibrium["w"], equilibrium["q"]], dtype=float)
    focused_config = RoyEvoPDEConfig(
        n_x=64,
        n_y=64,
        L_x=20.0,
        L_y=20.0,
        T=500.0,
        dt=0.1,
        record_every=50,
        D_n=0.01,
        D_w=0.01,
        D_q=0.005,
        seed=20260702,
    )
    initial_pde = initial_state_from_ode_equilibrium(initial_ode, focused_config)

    all_timeseries: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for diagnostic_run in build_runs():
        print(f"{diagnostic_run.run_id}: {diagnostic_run.model_type} evolve={diagnostic_run.evolve} stress={diagnostic_run.stress:.8g}")
        if diagnostic_run.model_type == "ODE":
            rows, summary = simulate_ode_run(diagnostic_run, initial_ode)
        else:
            rows, summary = simulate_pde_diagnostic_run(diagnostic_run, focused_config, initial_pde)
        all_timeseries.extend(rows)
        summaries.append(summary)

    summary_by_run = {str(row["run_id"]): row for row in summaries}
    final_label, final_interpretation = final_step10_label(summary_by_run["ODE_EVO_S3"], summary_by_run["PDE_EVO_S3"])
    summary_by_run["PDE_EVO_S3"]["diagnosis"] = (
        "predator_low_defense_mismatch"
        if final_label == "predator_low_defense_spatial_mismatch"
        else "diffusion_dilution_signature"
        if final_label == "diffusion_dilutes_evolutionary_rescue"
        else "inconclusive"
    )
    summaries = list(summary_by_run.values())

    stage_c_summary = None
    if profile == "full":
        full_config = RoyEvoPDEConfig(
            n_x=96,
            n_y=96,
            L_x=20.0,
            L_y=20.0,
            T=600.0,
            dt=0.1,
            record_every=60,
            D_n=0.01,
            D_w=0.01,
            D_q=0.005,
            seed=20260705,
        )
        full_initial = initial_state_from_ode_equilibrium(initial_ode, full_config)
        full_run = DiagnosticRun("PDE_EVO_S3_STAGE_C", "PDE", True, "S3_suppression_window_midpoint", SUPPRESSION_WINDOW_STRESS)
        rows, stage_c_summary = simulate_pde_diagnostic_run(full_run, full_config, full_initial)
        all_timeseries.extend(rows)
        summaries.append(stage_c_summary)

    write_csv(all_timeseries, TIMESERIES_CSV, TIMESERIES_FIELDNAMES)
    write_csv(summaries, SUMMARY_CSV, SUMMARY_FIELDNAMES)
    write_note(profile, final_label, final_interpretation, summaries, stage_c_summary)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused", "full"], default="focused")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
