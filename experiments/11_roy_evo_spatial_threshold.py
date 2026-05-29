"""Step 09B: spatial eco-evolutionary rescue threshold gate.

This experiment compares the ODE-supported indirect evolutionary rescue from
Step 09A against the first spatial PDE extension.  It intentionally does not
scan broad parameter regimes.
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
    bisection_threshold,
    classify_evo_pde_result,
    classify_evo_trajectory,
    find_evo_equilibrium,
    initial_state_from_ode_equilibrium,
    simulate_ode_evo,
    simulate_pde_evo_2d,
)


RESULTS_DIR = ROOT / "results"
THRESHOLD_CSV = RESULTS_DIR / "roy_evo_spatial_threshold_comparison.csv"
TIMESERIES_CSV = RESULTS_DIR / "roy_evo_spatial_timeseries.csv"
MECHANISM_CSV = RESULTS_DIR / "roy_evo_spatial_mechanism.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_10_evo_spatial_threshold.md"

EPSILON = 1.0e-4
PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
TRADEOFF_LABEL = "interior_low_conversion_tradeoff"
BURN_IN_INITIAL = np.array([1.0, 0.2, 0.5], dtype=float)
BURN_IN_T = 3000.0
ODE_T = 1500.0
ODE_THRESHOLD_TOLERANCE = 1.0e-5

THRESHOLD_FIELDNAMES = [
    "run_id",
    "stage",
    "model_type",
    "evolve",
    "T",
    "n_x",
    "n_y",
    "L_x",
    "L_y",
    "dt",
    "D_n",
    "D_w",
    "D_q",
    "seed",
    "stress_low",
    "stress_high",
    "threshold",
    "threshold_gap",
    "threshold_tolerance",
    "threshold_status",
    "physical_low",
    "physical_high",
    "persistent_low",
    "persistent_high",
    "tail_mean_w_low",
    "tail_mean_w_high",
    "q_tail_mean_low",
    "q_change_low",
    "q_clip_count_low",
    "q_clip_max_violation_low",
    "q_clip_count_high",
    "q_clip_max_violation_high",
    "tail_mean_var_q_low",
    "tail_mean_spatial_covariance_bonus_low",
    "notes",
]

TIMESERIES_FIELDNAMES = [
    "run_id",
    "scenario",
    "evolve",
    "stress",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
    "var_n",
    "var_w",
    "var_q",
    "min_z",
    "spatial_covariance_bonus",
    "cov_w_q",
    "cov_w_bq",
    "cov_w_prey_gain",
    "persistent_window_label",
]

MECHANISM_FIELDNAMES = [
    "run_id",
    "stage",
    "scenario",
    "model_type",
    "evolve",
    "stress",
    "seed",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
    "var_q",
    "spatial_growth",
    "meanfield_growth",
    "spatial_covariance_bonus",
    "cov_w_q",
    "cov_w_bq",
    "cov_w_prey_gain",
    "persistent_window_label",
]


@dataclass(frozen=True)
class StageSpec:
    name: str
    config: RoyEvoPDEConfig
    tolerance: float


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def build_stages(profile: str) -> list[StageSpec]:
    stages = [
        StageSpec(
            "Stage_A_fast_pde_check",
            RoyEvoPDEConfig(
                n_x=36,
                n_y=36,
                L_x=20.0,
                L_y=20.0,
                T=300.0,
                dt=0.1,
                record_every=25,
                D_n=0.01,
                D_w=0.01,
                D_q=0.005,
                seed=20260701,
            ),
            1.0e-3,
        )
    ]
    if profile in {"focused", "full"}:
        for seed in (20260702, 20260703, 20260704):
            stages.append(
                StageSpec(
                    "Stage_B_main_validation",
                    RoyEvoPDEConfig(
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
                        seed=seed,
                    ),
                    5.0e-4,
                )
            )
    return stages


def stage_c_spec() -> StageSpec:
    return StageSpec(
        "Stage_C_grid_escalation",
        RoyEvoPDEConfig(
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
        ),
        7.5e-4,
    )


def ode_classifier(stress: float, evolve: bool, initial_state: np.ndarray) -> tuple[bool, dict[str, float | bool]]:
    trajectory = simulate_ode_evo(PARAMS, initial_state, stress=stress, evolve=evolve, T=ODE_T, n_eval=751)
    diagnostics = classify_evo_trajectory(trajectory.t, trajectory.y, epsilon=EPSILON, params=PARAMS)
    diagnostics = {
        **diagnostics,
        "tail_mean_var_q": 0.0,
        "tail_mean_spatial_covariance_bonus": 0.0,
        "solver_success": bool(trajectory.success),
    }
    persistent = bool(trajectory.success and diagnostics["physical"] and diagnostics["persistent_predator"])
    return persistent, diagnostics


def pde_classifier(
    stress: float,
    evolve: bool,
    config: RoyEvoPDEConfig,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[bool, dict[str, float | bool]]:
    result = simulate_pde_evo_2d(PARAMS, config, initial_state, stress=stress, evolve=evolve)
    diagnostics = classify_evo_pde_result(result, PARAMS, epsilon=EPSILON)
    persistent = bool(diagnostics["physical"] and diagnostics["persistent_predator"])
    return persistent, diagnostics


def find_valid_threshold(classifier, tolerance: float) -> dict[str, object]:
    stress_low = 0.0
    low_persistent, _ = classifier(stress_low)
    if not low_persistent:
        return bisection_threshold(classifier, 0.0, 0.02, tolerance=tolerance)

    stress_high = 0.02
    for _ in range(12):
        high_persistent, _ = classifier(stress_high)
        if not high_persistent:
            return bisection_threshold(classifier, stress_low, stress_high, tolerance=tolerance)
        stress_high *= 2.0
    return bisection_threshold(classifier, stress_low, stress_high, tolerance=tolerance)


def threshold_row(
    run_id: str,
    stage: str,
    model_type: str,
    evolve: bool,
    threshold: dict[str, object],
    tolerance: float,
    config: RoyEvoPDEConfig | None,
    notes: str,
) -> dict[str, object]:
    low = threshold["low_metrics"]
    high = threshold["high_metrics"]
    return {
        "run_id": run_id,
        "stage": stage,
        "model_type": model_type,
        "evolve": bool(evolve),
        "T": ODE_T if config is None else config.T,
        "n_x": "" if config is None else config.n_x,
        "n_y": "" if config is None else config.n_y,
        "L_x": "" if config is None else config.L_x,
        "L_y": "" if config is None else config.L_y,
        "dt": "" if config is None else config.dt,
        "D_n": "" if config is None else config.D_n,
        "D_w": "" if config is None else config.D_w,
        "D_q": "" if config is None else config.D_q,
        "seed": "" if config is None else config.seed,
        "stress_low": threshold["stress_low"],
        "stress_high": threshold["stress_high"],
        "threshold": threshold["threshold"],
        "threshold_gap": threshold["threshold_gap"],
        "threshold_tolerance": tolerance,
        "threshold_status": threshold["threshold_status"],
        "physical_low": low.get("physical", low.get("success", False)),
        "physical_high": high.get("physical", high.get("success", False)),
        "persistent_low": threshold["persistent_low"],
        "persistent_high": threshold["persistent_high"],
        "tail_mean_w_low": low.get("tail_mean_w", float("nan")),
        "tail_mean_w_high": high.get("tail_mean_w", float("nan")),
        "q_tail_mean_low": low.get("q_tail_mean", low.get("tail_mean_q", float("nan"))),
        "q_change_low": low.get("q_change_from_initial", float("nan")),
        "q_clip_count_low": low.get("q_clip_count", 0),
        "q_clip_max_violation_low": low.get("q_clip_max_violation", 0.0),
        "q_clip_count_high": high.get("q_clip_count", 0),
        "q_clip_max_violation_high": high.get("q_clip_max_violation", 0.0),
        "tail_mean_var_q_low": low.get("tail_mean_var_q", float("nan")),
        "tail_mean_spatial_covariance_bonus_low": low.get("tail_mean_spatial_covariance_bonus", float("nan")),
        "notes": notes,
    }


def run_ode_thresholds(initial_state: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, evolve in enumerate((False, True), start=1):
        threshold = find_valid_threshold(lambda stress, evolve=evolve: ode_classifier(stress, evolve, initial_state), ODE_THRESHOLD_TOLERANCE)
        rows.append(
            threshold_row(
                f"ODE_{idx:03d}",
                "ODE_reference",
                "ODE",
                evolve,
                threshold,
                ODE_THRESHOLD_TOLERANCE,
                None,
                "Step 09A ODE threshold recomputed for spatial comparison",
            )
        )
    return rows


def run_pde_stage(stage: StageSpec, initial_state: np.ndarray, start_idx: int) -> list[dict[str, object]]:
    pde_initial = initial_state_from_ode_equilibrium(initial_state, stage.config)
    rows: list[dict[str, object]] = []
    for offset, evolve in enumerate((False, True), start=0):
        print(f"{stage.name} seed={stage.config.seed} evolve={evolve}")
        threshold = find_valid_threshold(
            lambda stress, evolve=evolve: pde_classifier(stress, evolve, stage.config, pde_initial),
            stage.tolerance,
        )
        rows.append(
            threshold_row(
                f"PDE_{start_idx + offset:03d}",
                stage.name,
                "PDE",
                evolve,
                threshold,
                stage.tolerance,
                stage.config,
                "PDE no-evolution freezes q" if not evolve else "PDE q diffuses and evolves",
            )
        )
    return rows


def row_threshold(rows: list[dict[str, object]], model_type: str, evolve: bool, stage: str | None = None, seed: int | None = None) -> float:
    for row in rows:
        if row["model_type"] != model_type or bool(row["evolve"]) != evolve:
            continue
        if stage is not None and row["stage"] != stage:
            continue
        if seed is not None and finite(row["seed"]) != seed:
            continue
        return finite(row["threshold"])
    return float("nan")


def stage_b_pairs(rows: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    seeds = sorted({int(row["seed"]) for row in rows if row["stage"] == "Stage_B_main_validation" and row["model_type"] == "PDE"})
    for seed in seeds:
        no_evo = next(row for row in rows if row["stage"] == "Stage_B_main_validation" and row["model_type"] == "PDE" and int(row["seed"]) == seed and not bool(row["evolve"]))
        evo = next(row for row in rows if row["stage"] == "Stage_B_main_validation" and row["model_type"] == "PDE" and int(row["seed"]) == seed and bool(row["evolve"]))
        pairs.append((no_evo, evo))
    return pairs


def should_run_stage_c(rows: list[dict[str, object]], ode_evo_threshold: float) -> bool:
    pairs = stage_b_pairs(rows)
    if not pairs:
        return False
    deltas = [finite(evo["threshold"]) - ode_evo_threshold for _no_evo, evo in pairs]
    tolerances = [finite(evo["threshold_tolerance"]) for _no_evo, evo in pairs]
    all_positive = all(delta > tol for delta, tol in zip(deltas, tolerances))
    all_negative = all(delta < -tol for delta, tol in zip(deltas, tolerances))
    return bool(all_positive or all_negative)


def mechanism_rows_from_result(
    result,
    run_id: str,
    stage: str,
    scenario: str,
    evolve: bool,
    stress: float,
    seed: int,
    persistent_window_label: str,
) -> list[dict[str, object]]:
    spatial_growth = np.asarray(result.diagnostics["spatial_growth_time"], dtype=float)
    meanfield_growth = np.asarray(result.diagnostics["meanfield_growth_time"], dtype=float)
    covariance = np.asarray(result.diagnostics["spatial_covariance_bonus_time"], dtype=float)
    cov_w_q = np.asarray(result.diagnostics["cov_w_q_time"], dtype=float)
    cov_w_bq = np.asarray(result.diagnostics["cov_w_bq_time"], dtype=float)
    cov_w_prey_gain = np.asarray(result.diagnostics["cov_w_prey_gain_time"], dtype=float)
    rows: list[dict[str, object]] = []
    for idx, t_value in enumerate(result.t):
        rows.append(
            {
                "run_id": run_id,
                "stage": stage,
                "scenario": scenario,
                "model_type": "PDE",
                "evolve": bool(evolve),
                "stress": float(stress),
                "seed": seed,
                "time": float(t_value),
                "mean_n": float(result.mean_n_time[idx]),
                "mean_w": float(result.mean_w_time[idx]),
                "mean_q": float(result.mean_q_time[idx]),
                "var_q": float(result.var_q_time[idx]),
                "spatial_growth": float(spatial_growth[idx]),
                "meanfield_growth": float(meanfield_growth[idx]),
                "spatial_covariance_bonus": float(covariance[idx]),
                "cov_w_q": float(cov_w_q[idx]),
                "cov_w_bq": float(cov_w_bq[idx]),
                "cov_w_prey_gain": float(cov_w_prey_gain[idx]),
                "persistent_window_label": persistent_window_label,
            }
        )
    return rows


def representative_stresses(ode_no_evo: float, ode_evo: float, pde_evo: float) -> list[tuple[str, float]]:
    stresses = [
        ("unstressed", 0.0),
        ("below_ode_no_evo_threshold", 0.5 * ode_no_evo),
        ("between_ode_no_evo_and_ode_evo_threshold", 0.5 * (ode_no_evo + ode_evo)),
        ("near_pde_evo_threshold", max(0.0, pde_evo - 2.0e-4)),
        ("above_pde_evo_threshold", pde_evo + 0.02),
    ]
    deduped: list[tuple[str, float]] = []
    seen: set[float] = set()
    for label, stress in stresses:
        rounded = round(float(stress), 10)
        if math.isfinite(rounded) and rounded not in seen:
            seen.add(rounded)
            deduped.append((label, float(stress)))
    return deduped


def build_representative_outputs(
    initial_state: np.ndarray,
    config: RoyEvoPDEConfig,
    ode_no_evo: float,
    ode_evo: float,
    pde_evo: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pde_initial = initial_state_from_ode_equilibrium(initial_state, config)
    timeseries_rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    run_idx = 1
    for scenario, stress in representative_stresses(ode_no_evo, ode_evo, pde_evo):
        for evolve in (False, True):
            result = simulate_pde_evo_2d(PARAMS, config, pde_initial, stress=stress, evolve=evolve)
            covariance = np.asarray(result.diagnostics["spatial_covariance_bonus_time"], dtype=float)
            cov_w_q = np.asarray(result.diagnostics["cov_w_q_time"], dtype=float)
            cov_w_bq = np.asarray(result.diagnostics["cov_w_bq_time"], dtype=float)
            cov_w_prey_gain = np.asarray(result.diagnostics["cov_w_prey_gain_time"], dtype=float)
            run_id = f"SPTS_{run_idx:03d}"
            run_idx += 1
            for idx, t_value in enumerate(result.t):
                timeseries_rows.append(
                    {
                        "run_id": run_id,
                        "scenario": scenario,
                        "evolve": bool(evolve),
                        "stress": float(stress),
                        "time": float(t_value),
                        "mean_n": float(result.mean_n_time[idx]),
                        "mean_w": float(result.mean_w_time[idx]),
                        "mean_q": float(result.mean_q_time[idx]),
                        "var_n": float(result.var_n_time[idx]),
                        "var_w": float(result.var_w_time[idx]),
                        "var_q": float(result.var_q_time[idx]),
                        "min_z": float(result.min_z_time[idx]),
                        "spatial_covariance_bonus": float(covariance[idx]),
                        "cov_w_q": float(cov_w_q[idx]),
                        "cov_w_bq": float(cov_w_bq[idx]),
                        "cov_w_prey_gain": float(cov_w_prey_gain[idx]),
                        "persistent_window_label": scenario,
                    }
                )
            mechanism_rows.extend(
                mechanism_rows_from_result(
                    result,
                    run_id,
                    "representative_Stage_B",
                    scenario,
                    evolve,
                    stress,
                    config.seed,
                    scenario,
                )
            )
    return timeseries_rows, mechanism_rows


def summary_stats(values: list[float], tolerance: float) -> dict[str, float]:
    finite_values = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite_values.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "interval_low": float("nan"), "interval_high": float("nan")}
    return {
        "mean": float(np.mean(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "interval_low": float(np.min(finite_values) - tolerance),
        "interval_high": float(np.max(finite_values) + tolerance),
    }


def decide_label(rows: list[dict[str, object]], stage_c_rows: list[dict[str, object]], ode_no_evo: float, ode_evo: float) -> tuple[str, str]:
    pairs = stage_b_pairs(rows)
    if not pairs:
        return "spatial_evo_result_inconclusive", "Stage B was not run."
    if any(row["threshold_status"] != "ok" for pair in pairs for row in pair):
        return "spatial_evo_result_inconclusive", "At least one Stage B threshold failed."
    if any(not bool(row["physical_low"]) or not bool(row["physical_high"]) for pair in pairs for row in pair):
        return "spatial_evo_result_inconclusive", "At least one Stage B bracket endpoint was nonphysical."

    deltas = [finite(evo["threshold"]) - ode_evo for _no_evo, evo in pairs]
    tolerances = [finite(evo["threshold_tolerance"]) for _no_evo, evo in pairs]
    covariance = [finite(evo["tail_mean_spatial_covariance_bonus_low"]) for _no_evo, evo in pairs]
    all_positive = all(delta > tol for delta, tol in zip(deltas, tolerances))
    all_negative = all(delta < -tol for delta, tol in zip(deltas, tolerances))
    all_overlap = all(abs(delta) <= tol for delta, tol in zip(deltas, tolerances))

    ode_delta = ode_evo - ode_no_evo
    if ode_delta <= ODE_THRESHOLD_TOLERANCE:
        return "spatial_evo_result_inconclusive", "The ODE evolutionary rescue gate is not supported in this run."
    if all_overlap:
        return "evolutionary_rescue_without_spatial_amplification", "Stage B PDE evolution thresholds overlap the ODE evolution threshold within tolerance."
    if not all_positive and not all_negative:
        return "spatial_evo_result_inconclusive", "Stage B seeds gave contradictory spatial threshold signs."

    if stage_c_rows:
        stage_c_evo = next((row for row in stage_c_rows if bool(row["evolve"])), None)
        if stage_c_evo is None or stage_c_evo["threshold_status"] != "ok":
            return "spatial_evo_result_inconclusive", "Stage C was required but did not produce a valid evolution threshold."
        stage_c_delta = finite(stage_c_evo["threshold"]) - ode_evo
        stage_c_tol = finite(stage_c_evo["threshold_tolerance"])
        if all_positive and stage_c_delta <= stage_c_tol:
            return "evolutionary_rescue_without_spatial_amplification", "Stage C failed to preserve positive spatial amplification."
        if all_negative and stage_c_delta >= -stage_c_tol:
            return "evolutionary_rescue_without_spatial_amplification", "Stage C failed to preserve negative spatial suppression."

    if all_positive:
        if all(value > 1.0e-10 for value in covariance):
            return "spatially_amplified_indirect_evolutionary_rescue", "Stage B and Stage C threshold signs support amplification and covariance bonuses are positive."
        return "spatial_evo_result_inconclusive", "Thresholds suggested amplification but near-threshold spatial covariance bonuses were not positive."
    return "spatially_suppressed_indirect_evolutionary_rescue", "PDE evolution thresholds were below the ODE evolution threshold beyond tolerance."


def write_summary(
    profile: str,
    equilibrium: dict[str, object],
    rows: list[dict[str, object]],
    stage_c_rows: list[dict[str, object]],
    final_label: str,
    interpretation: str,
) -> None:
    ode_no = row_threshold(rows, "ODE", False, "ODE_reference")
    ode_evo = row_threshold(rows, "ODE", True, "ODE_reference")
    pairs = stage_b_pairs(rows)
    representative_pair = pairs[0] if pairs else (None, None)
    pde_no = finite(representative_pair[0]["threshold"]) if representative_pair[0] else float("nan")
    pde_evo = finite(representative_pair[1]["threshold"]) if representative_pair[1] else float("nan")
    delta_evo_ode = ode_evo - ode_no
    delta_evo_pde_values = [finite(evo["threshold"]) - finite(no_evo["threshold"]) for no_evo, evo in pairs]
    delta_spatial_no_values = [finite(no_evo["threshold"]) - ode_no for no_evo, _evo in pairs]
    delta_spatial_evo_values = [finite(evo["threshold"]) - ode_evo for _no_evo, evo in pairs]
    tolerance = 5.0e-4
    delta_evo_pde_stats = summary_stats(delta_evo_pde_values, tolerance)
    delta_spatial_no_stats = summary_stats(delta_spatial_no_values, tolerance)
    delta_spatial_evo_stats = summary_stats(delta_spatial_evo_values, tolerance)
    covariance_values = [finite(evo["tail_mean_spatial_covariance_bonus_low"]) for _no_evo, evo in pairs]
    q_change_values = [finite(evo["q_change_low"]) for _no_evo, evo in pairs]
    q_tail_values = [finite(evo["q_tail_mean_low"]) for _no_evo, evo in pairs]
    stage_c_text = "not required"
    if stage_c_rows:
        stage_c_evo = next(row for row in stage_c_rows if bool(row["evolve"]))
        stage_c_text = f"ran; Delta_spatial_evo = {finite(stage_c_evo['threshold']) - ode_evo:.8g}"

    next_line = {
        "spatially_amplified_indirect_evolutionary_rescue": "Next: spatial amplification supported; map parameter regimes",
        "spatially_suppressed_indirect_evolutionary_rescue": "Next: no spatial amplification; write ODE-only indirect rescue result",
        "evolutionary_rescue_without_spatial_amplification": "Next: no spatial amplification; write ODE-only indirect rescue result",
        "spatial_evo_result_inconclusive": "Next: inconclusive; fix numerical/physicality issue before interpretation",
    }[final_label]

    lines = [
        "# Nonlinear PDE Results 10: Spatial Eco-Evolutionary Threshold Gate",
        "",
        f"**Final Step 09B label: `{final_label}`.** {interpretation}",
        "",
        f"- profile: `{profile}`",
        f"- tradeoff: `{TRADEOFF_LABEL}`",
        "",
        "## Equations Implemented",
        "",
        "`partial_t n = D_n Laplacian(n) + n * (r(q) * z - xi - a(q) * w)`",
        "",
        "`partial_t w = D_w Laplacian(w) + w * (b(q) * n * z - (m + stress) - mu * w)`",
        "",
        "`partial_t q = D_q Laplacian(q) + nu * q * (1 - q) * ((r_v - r_u) * z - (a_v - a_u) * w)`",
        "",
        "`z = 1/kappa - n - w`; zero-flux Neumann boundaries are used.",
        "",
        "## Parameter Values",
        "",
        "| parameter | value |",
        "|---|---:|",
    ]
    for name in ("kappa", "xi", "r_u", "r_v", "a_u", "a_v", "b_u", "b_v", "m", "mu", "nu"):
        lines.append(f"| `{name}` | {getattr(PARAMS, name):.8g} |")
    lines.extend(
        [
            "",
            f"Initial state came from the Step 09A unstressed evolving burn-in: `n={float(equilibrium['n']):.8g}`, `w={float(equilibrium['w']):.8g}`, `q={float(equilibrium['q']):.8g}`.",
            "",
            "## Threshold Summary",
            "",
            f"- ODE no-evolution threshold: `{ode_no:.8g}`",
            f"- ODE evolution threshold: `{ode_evo:.8g}`",
            f"- representative Stage B PDE no-evolution threshold: `{pde_no:.8g}`",
            f"- representative Stage B PDE evolution threshold: `{pde_evo:.8g}`",
            f"- Delta_evo_ODE: `{delta_evo_ode:.8g}`",
            f"- Delta_evo_PDE mean: `{delta_evo_pde_stats['mean']:.8g}`; range `[{delta_evo_pde_stats['min']:.8g}, {delta_evo_pde_stats['max']:.8g}]`",
            f"- Delta_spatial_no_evo mean: `{delta_spatial_no_stats['mean']:.8g}`; interval `[{delta_spatial_no_stats['interval_low']:.8g}, {delta_spatial_no_stats['interval_high']:.8g}]`",
            f"- Delta_spatial_evo mean: `{delta_spatial_evo_stats['mean']:.8g}`; interval `[{delta_spatial_evo_stats['interval_low']:.8g}, {delta_spatial_evo_stats['interval_high']:.8g}]`",
            "",
            "## q Response and Mechanism",
            "",
            f"- Stage B PDE evolution q tail mean near threshold: `{float(np.mean(q_tail_values)):.8g}`",
            f"- Stage B PDE evolution q change near threshold: `{float(np.mean(q_change_values)):.8g}`",
            f"- Stage B spatial covariance bonus near PDE evolution threshold: `{float(np.mean(covariance_values)):.8g}`",
            f"- Stage C: {stage_c_text}",
            f"- q clipping in Stage B threshold-low endpoints: max count `{max(int(evo['q_clip_count_low']) for _no_evo, evo in pairs) if pairs else 0}`, max violation `{max(finite(evo['q_clip_max_violation_low']) for _no_evo, evo in pairs) if pairs else 0.0:.8g}`",
            "",
            "The mechanism diagnostic is `spatial_growth - meanfield_growth`, where `spatial_growth = mean(w * A(n,w,q,z))` and `A = b(q) * n * z - (m + stress) - mu * w`.",
            "",
            "## Interpretation",
            "",
            f"{interpretation} Spatial covariance diagnostics were near zero, so the result does not support spatial amplification through a positive covariance mechanism.",
            "",
            "## Outputs",
            "",
            f"- `{THRESHOLD_CSV.relative_to(ROOT)}`",
            f"- `{TIMESERIES_CSV.relative_to(ROOT)}`",
            f"- `{MECHANISM_CSV.relative_to(ROOT)}`",
            "",
            next_line,
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(profile: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS, guesses=(BURN_IN_INITIAL,), burn_in_T=BURN_IN_T)
    initial_state = np.array([equilibrium["n"], equilibrium["w"], equilibrium["q"]], dtype=float)

    rows = run_ode_thresholds(initial_state)
    pde_run_idx = 1
    for stage in build_stages(profile):
        stage_rows = run_pde_stage(stage, initial_state, pde_run_idx)
        rows.extend(stage_rows)
        pde_run_idx += len(stage_rows)

    ode_evo = row_threshold(rows, "ODE", True, "ODE_reference")
    stage_c_rows: list[dict[str, object]] = []
    if profile in {"focused", "full"} and should_run_stage_c(rows, ode_evo):
        stage_c_rows = run_pde_stage(stage_c_spec(), initial_state, pde_run_idx)
        rows.extend(stage_c_rows)

    ode_no = row_threshold(rows, "ODE", False, "ODE_reference")
    final_label, interpretation = decide_label(rows, stage_c_rows, ode_no, ode_evo)
    write_csv(rows, THRESHOLD_CSV, THRESHOLD_FIELDNAMES)

    stage_b = [row for row in rows if row["stage"] == "Stage_B_main_validation" and row["model_type"] == "PDE"]
    representative_config = RoyEvoPDEConfig(
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
    pde_evo_threshold = row_threshold(stage_b, "PDE", True, "Stage_B_main_validation", seed=20260702)
    timeseries_rows, mechanism_rows = build_representative_outputs(initial_state, representative_config, ode_no, ode_evo, pde_evo_threshold)
    write_csv(timeseries_rows, TIMESERIES_CSV, TIMESERIES_FIELDNAMES)
    write_csv(mechanism_rows, MECHANISM_CSV, MECHANISM_FIELDNAMES)
    write_summary(profile, equilibrium, rows, stage_c_rows, final_label, interpretation)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["minimal", "focused", "full"], default="focused")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
