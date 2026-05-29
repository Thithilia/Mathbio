"""Step 12: asymptotic and continuation checks for PDE-evo persistence.

This focused experiment tests whether the unresolved PDE-evo boundary is caused
by long transients, continuation/hysteresis effects, nonmonotone response, or
numerical/physical instability. It does not change the model and does not
interpret the spatial suppression mechanism.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (
    RoyEvoPDEConfig,
    RoyEvoPDEResult,
    RoyEvoParams,
    find_evo_equilibrium,
    free_space_evo,
    grid_2d_evo,
    initial_state_from_ode_equilibrium,
    laplacian_neumann_2d_evo,
    reaction_part_evo_pde,
    simulate_pde_evo_2d,
)


RESULTS_DIR = ROOT / "results"
NOTES_DIR = ROOT / "research_notes"
LONG_CSV = RESULTS_DIR / "roy_pde_evo_long_horizon.csv"
CONTINUATION_CSV = RESULTS_DIR / "roy_pde_evo_continuation.csv"
SUMMARY_MD = NOTES_DIR / "roy_pde_evo_asymptotic_continuation.md"

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 1.0e-4
EXTREME_EXTINCTION_W = 1.0e-8
TAIL_FRACTION = 0.25

CONTINUATION_STRESSES = (
    0.09,
    0.105,
    0.11765625,
    0.125,
    0.135,
    0.141262205,
    0.150,
    0.1584375,
    0.16486816,
    0.175,
)
LONG_HORIZON_STRESSES = (
    0.135,
    0.141262205,
    0.150,
    0.1584375,
    0.16486816,
    0.175,
)
FOCUSED_LONG_HORIZONS = (1600.0, 2400.0)
FULL_LONG_HORIZONS = (1600.0, 2400.0, 3200.0)
FOCUSED_CONTINUATION_T = 800.0
FULL_CONTINUATION_T = 1200.0

LONG_FIELDNAMES = [
    "run_id",
    "stress",
    "T",
    "n_x",
    "n_y",
    "seed",
    "physical",
    "classification",
    "tail_mean_w",
    "tail_min_w",
    "tail_slope_w",
    "tail_slope_floor_w",
    "tail_mean_q",
    "tail_q_change",
    "tail_mean_var_q",
    "tail_mean_min_z",
    "previous_window_mean_w",
    "latest_window_mean_w",
    "relative_change_between_last_windows",
    "persistent_with_slope_rule",
    "persistent_without_slope_rule",
    "rhs_norm",
    "state_norm",
    "normalized_residual",
    "max_abs_dn_dt",
    "max_abs_dw_dt",
    "max_abs_dq_dt",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
    "q_clip_count",
    "q_clip_max_violation",
    "notes",
]

CONTINUATION_FIELDNAMES = [
    "run_id",
    "sweep_direction",
    "step_index",
    "stress",
    "T",
    "n_x",
    "n_y",
    "seed",
    "initial_source",
    "physical",
    "classification",
    "tail_mean_w",
    "tail_min_w",
    "tail_slope_w",
    "tail_mean_q",
    "tail_q_change",
    "previous_window_mean_w",
    "latest_window_mean_w",
    "relative_change_between_last_windows",
    "rhs_norm",
    "state_norm",
    "normalized_residual",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
    "q_clip_count",
    "q_clip_max_violation",
    "notes",
]


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pde_config(T: float) -> RoyEvoPDEConfig:
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
        seed=20260702,
    )


def horizons_for_profile(profile: str) -> tuple[float, ...]:
    return FULL_LONG_HORIZONS if profile == "full" else FOCUSED_LONG_HORIZONS


def continuation_T_for_profile(profile: str) -> float:
    return FULL_CONTINUATION_T if profile == "full" else FOCUSED_CONTINUATION_T


def tail_mask(times: np.ndarray, tail_fraction: float = TAIL_FRACTION) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("Need at least two time points for tail classification.")
    cutoff = float(times[-1]) - tail_fraction * float(times[-1] - times[0])
    mask = times >= cutoff
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(times, dtype=bool)
        mask[-2:] = True
    return mask


def previous_window_mask(times: np.ndarray, tail_fraction: float = TAIL_FRACTION) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("Need at least two time points for window diagnostics.")
    total_duration = float(times[-1] - times[0])
    window = tail_fraction * total_duration
    latest_start = float(times[-1]) - window
    previous_start = float(times[-1]) - 2.0 * window
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


def physical_from_result(result: RoyEvoPDEResult) -> bool:
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
    min_n = float(result.diagnostics.get("min_n", np.min(result.n)))
    min_w = float(result.diagnostics.get("min_w", np.min(result.w)))
    min_q = float(result.diagnostics.get("min_q", np.min(result.q)))
    max_q = float(result.diagnostics.get("max_q", np.max(result.q)))
    min_z = float(result.diagnostics.get("min_z", np.min(result.min_z_time)))
    q_clip_max_violation = float(result.diagnostics.get("q_clip_max_violation", 0.0))
    return bool(
        result.diagnostics.get("completed", True)
        and not result.diagnostics.get("nonfinite_detected", False)
        and min_n >= -1.0e-8
        and min_w >= -1.0e-8
        and min_q >= -1.0e-6
        and max_q <= 1.0 + 1.0e-6
        and min_z >= -1.0e-5
        and q_clip_max_violation <= 1.0e-4
    )


def pde_evo_rhs_residual(
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    stress: float,
    evolve: bool = True,
) -> dict[str, float]:
    _x, _y, dx, dy = grid_2d_evo(config)
    reactions = reaction_part_evo_pde(n, w, q, params, stress=stress, evolve=evolve)
    dn_dt = config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + reactions[0]
    dw_dt = config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + reactions[1]
    dq_dt = config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + reactions[2]
    rhs_norm = float(np.sqrt(np.mean(dn_dt * dn_dt + dw_dt * dw_dt + dq_dt * dq_dt)))
    state_norm = float(np.sqrt(np.mean(n * n + w * w + q * q)))
    return {
        "rhs_norm": rhs_norm,
        "state_norm": state_norm,
        "normalized_residual": float(rhs_norm / max(state_norm, 1.0e-12)),
        "max_abs_dn_dt": float(np.max(np.abs(dn_dt))),
        "max_abs_dw_dt": float(np.max(np.abs(dw_dt))),
        "max_abs_dq_dt": float(np.max(np.abs(dq_dt))),
    }


def tail_diagnostics(result: RoyEvoPDEResult) -> dict[str, float | bool]:
    mask = tail_mask(result.t)
    previous_mask = previous_window_mask(result.t)
    tail_t = result.t[mask]
    tail_w = result.mean_w_time[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(result.t, result.mean_w_time, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_mean_w = float(np.mean(result.mean_w_time[previous_mask]))
    latest_mean_w = tail_mean_w
    physical = physical_from_result(result)
    persistent_without_slope = bool(physical and tail_mean_w > EPSILON and tail_min_w > 0.25 * EPSILON)
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    initial_q = float(result.diagnostics.get("initial_mean_q", result.mean_q_time[0]))
    tail_mean_q = float(np.mean(result.mean_q_time[mask]))
    return {
        "physical": physical,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": slope_floor,
        "tail_mean_q": tail_mean_q,
        "tail_q_change": float(tail_mean_q - initial_q),
        "tail_mean_var_q": float(np.mean(result.var_q_time[mask])),
        "tail_mean_min_z": float(np.mean(result.min_z_time[mask])),
        "previous_window_mean_w": previous_mean_w,
        "latest_window_mean_w": latest_mean_w,
        "relative_change_between_last_windows": relative_change(previous_mean_w, latest_mean_w),
        "persistent_with_slope_rule": persistent_with_slope,
        "persistent_without_slope_rule": persistent_without_slope,
    }


def classify_asymptotic_run(metrics: dict[str, object]) -> str:
    physical = bool(metrics["physical"])
    tail_mean_w = float(metrics["tail_mean_w"])
    tail_min_w = float(metrics["tail_min_w"])
    previous_w = float(metrics["previous_window_mean_w"])
    latest_w = float(metrics["latest_window_mean_w"])
    rel_change = float(metrics["relative_change_between_last_windows"])
    normalized_residual = float(metrics["normalized_residual"])
    persistent_with_slope = bool(metrics["persistent_with_slope_rule"])
    persistent_without_slope = bool(metrics["persistent_without_slope_rule"])
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


def result_metrics(result: RoyEvoPDEResult, config: RoyEvoPDEConfig, stress: float) -> dict[str, object]:
    tail = tail_diagnostics(result)
    residual = pde_evo_rhs_residual(result.n, result.w, result.q, PARAMS, config, stress=stress, evolve=True)
    metrics: dict[str, object] = {
        **tail,
        **residual,
        "min_n": float(result.diagnostics.get("min_n", np.min(result.n))),
        "min_w": float(result.diagnostics.get("min_w", np.min(result.w))),
        "min_q": float(result.diagnostics.get("min_q", np.min(result.q))),
        "max_q": float(result.diagnostics.get("max_q", np.max(result.q))),
        "min_z": float(result.diagnostics.get("min_z", np.min(free_space_evo(result.n, result.w, PARAMS)))),
        "q_clip_count": int(result.diagnostics.get("q_clip_count", 0)),
        "q_clip_max_violation": float(result.diagnostics.get("q_clip_max_violation", 0.0)),
    }
    metrics["classification"] = classify_asymptotic_run(metrics)
    return metrics


def long_horizon_row(stress: float, horizon: float, result: RoyEvoPDEResult, config: RoyEvoPDEConfig) -> dict[str, object]:
    metrics = result_metrics(result, config, stress)
    return {
        "run_id": f"LONG_s{stress:.9g}_T{horizon:.0f}".replace(".", "p"),
        "stress": float(stress),
        "T": float(horizon),
        "n_x": int(config.n_x),
        "n_y": int(config.n_y),
        "seed": int(config.seed),
        "physical": bool(metrics["physical"]),
        "classification": str(metrics["classification"]),
        "tail_mean_w": float(metrics["tail_mean_w"]),
        "tail_min_w": float(metrics["tail_min_w"]),
        "tail_slope_w": float(metrics["tail_slope_w"]),
        "tail_slope_floor_w": float(metrics["tail_slope_floor_w"]),
        "tail_mean_q": float(metrics["tail_mean_q"]),
        "tail_q_change": float(metrics["tail_q_change"]),
        "tail_mean_var_q": float(metrics["tail_mean_var_q"]),
        "tail_mean_min_z": float(metrics["tail_mean_min_z"]),
        "previous_window_mean_w": float(metrics["previous_window_mean_w"]),
        "latest_window_mean_w": float(metrics["latest_window_mean_w"]),
        "relative_change_between_last_windows": float(metrics["relative_change_between_last_windows"]),
        "persistent_with_slope_rule": bool(metrics["persistent_with_slope_rule"]),
        "persistent_without_slope_rule": bool(metrics["persistent_without_slope_rule"]),
        "rhs_norm": float(metrics["rhs_norm"]),
        "state_norm": float(metrics["state_norm"]),
        "normalized_residual": float(metrics["normalized_residual"]),
        "max_abs_dn_dt": float(metrics["max_abs_dn_dt"]),
        "max_abs_dw_dt": float(metrics["max_abs_dw_dt"]),
        "max_abs_dq_dt": float(metrics["max_abs_dq_dt"]),
        "min_n": float(metrics["min_n"]),
        "min_w": float(metrics["min_w"]),
        "min_q": float(metrics["min_q"]),
        "max_q": float(metrics["max_q"]),
        "min_z": float(metrics["min_z"]),
        "q_clip_count": int(metrics["q_clip_count"]),
        "q_clip_max_violation": float(metrics["q_clip_max_violation"]),
        "notes": note_for_classification(str(metrics["classification"])),
    }


def continuation_row(
    sweep_direction: str,
    step_index: int,
    stress: float,
    horizon: float,
    result: RoyEvoPDEResult,
    config: RoyEvoPDEConfig,
    initial_source: str,
) -> dict[str, object]:
    metrics = result_metrics(result, config, stress)
    return {
        "run_id": f"{sweep_direction}_step{step_index:02d}_s{stress:.9g}".replace(".", "p"),
        "sweep_direction": sweep_direction,
        "step_index": int(step_index),
        "stress": float(stress),
        "T": float(horizon),
        "n_x": int(config.n_x),
        "n_y": int(config.n_y),
        "seed": int(config.seed),
        "initial_source": initial_source,
        "physical": bool(metrics["physical"]),
        "classification": str(metrics["classification"]),
        "tail_mean_w": float(metrics["tail_mean_w"]),
        "tail_min_w": float(metrics["tail_min_w"]),
        "tail_slope_w": float(metrics["tail_slope_w"]),
        "tail_mean_q": float(metrics["tail_mean_q"]),
        "tail_q_change": float(metrics["tail_q_change"]),
        "previous_window_mean_w": float(metrics["previous_window_mean_w"]),
        "latest_window_mean_w": float(metrics["latest_window_mean_w"]),
        "relative_change_between_last_windows": float(metrics["relative_change_between_last_windows"]),
        "rhs_norm": float(metrics["rhs_norm"]),
        "state_norm": float(metrics["state_norm"]),
        "normalized_residual": float(metrics["normalized_residual"]),
        "min_n": float(metrics["min_n"]),
        "min_w": float(metrics["min_w"]),
        "min_q": float(metrics["min_q"]),
        "max_q": float(metrics["max_q"]),
        "min_z": float(metrics["min_z"]),
        "q_clip_count": int(metrics["q_clip_count"]),
        "q_clip_max_violation": float(metrics["q_clip_max_violation"]),
        "notes": note_for_classification(str(metrics["classification"])),
    }


def note_for_classification(classification: str) -> str:
    if classification.endswith("_steady"):
        return "steady_by_tail_change_and_rhs_residual"
    if classification in {"declining_transient", "recovery_transient"}:
        return "window_change_indicates_transient"
    if classification in {"persistent_transient", "extinct_transient"}:
        return "tail_state_not_steady_by_rhs_residual"
    return classification


def run_long_horizon(equilibrium: dict[str, object], horizons: tuple[float, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stress in LONG_HORIZON_STRESSES:
        for horizon in horizons:
            config = pde_config(horizon)
            initial_state = initial_state_from_ode_equilibrium(equilibrium, config)
            print(f"LONG stress={stress:.9g} T={horizon:.0f}")
            result = simulate_pde_evo_2d(PARAMS, config, initial_state, stress=stress, evolve=True)
            rows.append(long_horizon_row(stress, horizon, result, config))
    return rows


def run_continuation(equilibrium: dict[str, object], horizon: float, direction: str) -> list[dict[str, object]]:
    stresses = CONTINUATION_STRESSES if direction == "upward" else tuple(reversed(CONTINUATION_STRESSES))
    config = pde_config(horizon)
    initial_state = initial_state_from_ode_equilibrium(equilibrium, config)
    rows: list[dict[str, object]] = []
    initial_source = "baseline_burn_in"
    for index, stress in enumerate(stresses):
        print(f"{direction.upper()} step={index:02d} stress={stress:.9g} T={horizon:.0f}")
        result = simulate_pde_evo_2d(PARAMS, config, initial_state, stress=stress, evolve=True)
        rows.append(continuation_row(direction, index, stress, horizon, result, config, initial_source))
        initial_state = (result.n.copy(), result.w.copy(), result.q.copy())
        initial_source = f"{direction}_previous_stress_{stress:.9g}"
    return rows


def detect_hysteresis(continuation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    upward = {float(row["stress"]): str(row["classification"]) for row in continuation_rows if row["sweep_direction"] == "upward"}
    downward = {float(row["stress"]): str(row["classification"]) for row in continuation_rows if row["sweep_direction"] == "downward"}
    mismatches: list[dict[str, object]] = []
    for stress in sorted(set(upward) & set(downward)):
        if upward[stress] != downward[stress]:
            mismatches.append(
                {
                    "stress": stress,
                    "upward": upward[stress],
                    "downward": downward[stress],
                }
            )
    return mismatches


def transient_or_nonsteady(classification: str) -> bool:
    return classification in {
        "persistent_transient",
        "extinct_transient",
        "recovery_transient",
        "declining_transient",
        "indeterminate",
    }


def clean_boundary_stabilized(long_rows: list[dict[str, object]], continuation_rows: list[dict[str, object]], max_horizon: float) -> bool:
    long_latest = [row for row in long_rows if math.isclose(float(row["T"]), max_horizon)]
    long_latest.sort(key=lambda row: float(row["stress"]))
    if not long_latest:
        return False
    classes = [str(row["classification"]) for row in long_latest]
    if any(classification not in {"persistent_steady", "extinct_steady"} for classification in classes):
        return False
    if "persistent_steady" not in classes or "extinct_steady" not in classes:
        return False
    seen_extinct = False
    for classification in classes:
        if classification == "extinct_steady":
            seen_extinct = True
        elif seen_extinct and classification == "persistent_steady":
            return False
    return len(detect_hysteresis(continuation_rows)) == 0


def decide_final_label(
    long_rows: list[dict[str, object]],
    continuation_rows: list[dict[str, object]],
    max_horizon: float,
) -> tuple[str, str, list[dict[str, object]], int]:
    hysteresis = detect_hysteresis(continuation_rows)
    if hysteresis:
        return (
            "pde_evo_hysteresis_detected",
            "Upward and downward continuation sweeps give different classifications at one or more stresses.",
            hysteresis,
            count_long_transients(long_rows, max_horizon),
        )
    transient_count = count_long_transients(long_rows, max_horizon)
    if transient_count >= 2:
        return (
            "pde_evo_long_transients_confirmed",
            "At least two sensitive stresses remain transient or non-steady at the longest horizon.",
            hysteresis,
            transient_count,
        )
    if clean_boundary_stabilized(long_rows, continuation_rows, max_horizon):
        return (
            "pde_evo_boundary_stabilized",
            "Long-horizon and continuation checks give a clean persistent-to-extinct transition.",
            hysteresis,
            transient_count,
        )
    return (
        "pde_evo_asymptotic_unresolved",
        "The asymptotic state remains unresolved under the focused continuation checks.",
        hysteresis,
        transient_count,
    )


def count_long_transients(long_rows: list[dict[str, object]], max_horizon: float) -> int:
    latest = [row for row in long_rows if math.isclose(float(row["T"]), max_horizon)]
    return sum(1 for row in latest if transient_or_nonsteady(str(row["classification"])))


def long_class_by_stress_horizon(long_rows: list[dict[str, object]]) -> dict[tuple[float, float], dict[str, object]]:
    return {(float(row["stress"]), float(row["T"])): row for row in long_rows}


def continuation_class_by_stress(continuation_rows: list[dict[str, object]], direction: str) -> dict[float, str]:
    return {
        float(row["stress"]): str(row["classification"])
        for row in continuation_rows
        if str(row["sweep_direction"]) == direction
    }


def write_note(
    profile: str,
    long_horizons: tuple[float, ...],
    continuation_T: float,
    long_rows: list[dict[str, object]],
    continuation_rows: list[dict[str, object]],
    final_label: str,
    interpretation: str,
    hysteresis: list[dict[str, object]],
    transient_count: int,
) -> None:
    NOTES_DIR.mkdir(exist_ok=True)
    long_lookup = long_class_by_stress_horizon(long_rows)
    upward = continuation_class_by_stress(continuation_rows, "upward")
    downward = continuation_class_by_stress(continuation_rows, "downward")
    max_horizon = max(long_horizons)

    long_table = [
        "| stress | T1600 classification | T2400 classification | T3200 classification | latest tail mean w | normalized residual | conclusion |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for stress in LONG_HORIZON_STRESSES:
        latest = long_lookup.get((float(stress), float(max_horizon)))
        classes = []
        for horizon in (1600.0, 2400.0, 3200.0):
            row = long_lookup.get((float(stress), horizon))
            classes.append(f"`{row['classification']}`" if row else "`not_run`")
        conclusion = str(latest["classification"]) if latest else "not_run"
        long_table.append(
            "| "
            + " | ".join(
                [
                    f"{stress:.9g}",
                    classes[0],
                    classes[1],
                    classes[2],
                    f"{float(latest['tail_mean_w']):.8g}" if latest else "nan",
                    f"{float(latest['normalized_residual']):.8g}" if latest else "nan",
                    f"`{conclusion}`",
                ]
            )
            + " |"
        )

    continuation_table = [
        "| stress | upward classification | downward classification | same? | note |",
        "|---:|---|---|---|---|",
    ]
    for stress in CONTINUATION_STRESSES:
        up = upward.get(float(stress), "missing")
        down = downward.get(float(stress), "missing")
        same = up == down
        note = "same" if same else "direction-dependent"
        continuation_table.append(
            f"| {stress:.9g} | `{up}` | `{down}` | {same} | {note} |"
        )

    if final_label == "pde_evo_boundary_stabilized":
        next_step = "Next: re-run threshold claims using the stabilized asymptotic boundary."
    elif final_label == "pde_evo_hysteresis_detected":
        next_step = "Next: map continuation-dependent regimes before making threshold claims."
    elif final_label == "pde_evo_long_transients_confirmed":
        next_step = "Next: use longer horizons or continuation/steady-state methods before threshold claims."
    else:
        next_step = "Next: inspect numerical stability and physicality before further interpretation."
    mismatch_text = (
        "; ".join(
            f"{item['stress']:.9g}: up={item['upward']}, down={item['downward']}"
            for item in hysteresis
        )
        if hysteresis
        else "none"
    )

    lines = [
        "# Research Note: Asymptotic and Continuation Check for PDE-Evolution Persistence",
        "",
        "## Executive Summary",
        "",
        f"Final Step 12 label: `{final_label}`.",
        "",
        interpretation,
        "",
        "## Why This Was Needed",
        "",
        "PR #5 found the PDE-evo persistence boundary unresolved under multi-horizon classification. Step 12 checks whether the unresolved boundary reflects long transients, hysteresis or multiple attractors, nonmonotone stress response, or unresolved asymptotics.",
        "",
        "This task does not diagnose spatial suppression mechanism and does not change the model equations or previous PR conclusions.",
        "",
        "## Setup",
        "",
        f"- profile run: `{profile}`",
        "- model: PDE-evo only",
        "- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`",
        "- baseline: unstressed Step 09A burn-in via `find_evo_equilibrium`",
        "- grid: `64x64`, `L_x=L_y=20`, seed `20260702`",
        "- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`",
        "- integration: `dt=0.1`, `record_every=50`",
        f"- long-horizon stresses: `{', '.join(f'{stress:.9g}' for stress in LONG_HORIZON_STRESSES)}`",
        f"- long horizons run: `{', '.join(f'{horizon:.0f}' for horizon in long_horizons)}`",
        f"- continuation grid: `{', '.join(f'{stress:.9g}' for stress in CONTINUATION_STRESSES)}`",
        f"- continuation step horizon: `{continuation_T:.0f}`",
        f"- steady criteria: `abs(relative_change) < {STEADY_REL_CHANGE_TOL}` and `normalized_residual < {STEADY_RESIDUAL_TOL}`",
        "",
        "## Long-Horizon Results",
        "",
    ]
    lines.extend(long_table)
    lines.extend(
        [
            "",
            "## Continuation Sweep Results",
            "",
        ]
    )
    lines.extend(continuation_table)
    lines.extend(
        [
            "",
            "## Hysteresis Check",
            "",
            f"- direction-dependent stresses: `{len(hysteresis)}`",
            f"- mismatches: `{mismatch_text}`",
            "",
            "## Boundary Diagnosis",
            "",
            f"- transient or non-steady long-horizon stresses at `T={max_horizon:.0f}`: `{transient_count}` of `{len(LONG_HORIZON_STRESSES)}`",
            f"- final label: `{final_label}`",
            "",
            interpretation,
            "",
            "## Implication for Threshold Claims",
            "",
            "Spatial suppression mechanism should not be interpreted unless the asymptotic or continuation boundary is stable.",
            "",
            "## Files",
            "",
            "- `results/roy_pde_evo_long_horizon.csv`",
            "- `results/roy_pde_evo_continuation.csv`",
            "",
            "## Next Step",
            "",
            next_step,
            "",
            final_label,
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(profile: str = "focused") -> tuple[str, str]:
    long_horizons = horizons_for_profile(profile)
    continuation_T = continuation_T_for_profile(profile)
    RESULTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS)

    long_rows = run_long_horizon(equilibrium, long_horizons)
    continuation_rows = run_continuation(equilibrium, continuation_T, "upward")
    continuation_rows.extend(run_continuation(equilibrium, continuation_T, "downward"))
    final_label, interpretation, hysteresis, transient_count = decide_final_label(
        long_rows,
        continuation_rows,
        max(long_horizons),
    )

    write_csv(long_rows, LONG_CSV, LONG_FIELDNAMES)
    write_csv(continuation_rows, CONTINUATION_CSV, CONTINUATION_FIELDNAMES)
    write_note(
        profile,
        long_horizons,
        continuation_T,
        long_rows,
        continuation_rows,
        final_label,
        interpretation,
        hysteresis,
        transient_count,
    )
    print(SUMMARY_MD.read_text(encoding="utf-8"))
    return final_label, interpretation


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused", "full"], default="focused")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
