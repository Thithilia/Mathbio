"""Step 13: map hysteresis and basin structure in the PDE-evo model.

This experiment treats the PDE-evo response as a stress-regime problem rather
than a single-threshold problem. It reproduces continuation hysteresis and runs
focused initial-condition basin checks without changing the model equations or
diagnosing spatial covariance mechanisms.
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
HYSTERESIS_CSV = RESULTS_DIR / "roy_pde_evo_hysteresis_map.csv"
BASIN_CSV = RESULTS_DIR / "roy_pde_evo_basin_initial_condition_scan.csv"
SUMMARY_MD = NOTES_DIR / "roy_pde_evo_hysteresis_basin_map.md"

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
SENSITIVE_STRESSES = (
    0.125,
    0.135,
    0.141262205,
    0.150,
    0.1584375,
    0.16486816,
    0.175,
)
FOCUSED_BASIN_STRESSES = (
    0.141262205,
    0.150,
    0.1584375,
    0.16486816,
    0.175,
)
FULL_BASIN_STRESSES = (
    0.135,
    0.141262205,
    0.150,
    0.1584375,
    0.16486816,
    0.175,
)
INITIAL_CONDITION_FAMILIES = (
    "baseline_burnin",
    "persistent_branch",
    "extinct_branch",
    "low_predator",
    "low_defense",
    "high_defense",
    "mixed_random",
)

HYSTERESIS_FIELDNAMES = [
    "run_id",
    "sweep_direction",
    "step_index",
    "stress",
    "T",
    "initial_source",
    "classification",
    "tail_mean_w",
    "tail_mean_q",
    "relative_change_between_last_windows",
    "normalized_residual",
    "physical",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
    "notes",
]

BASIN_FIELDNAMES = [
    "run_id",
    "stress",
    "initial_condition_family",
    "T",
    "classification",
    "tail_mean_w",
    "tail_mean_q",
    "q_change",
    "relative_change_between_last_windows",
    "normalized_residual",
    "physical",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
    "basin_label",
    "notes",
]


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pde_config(T: float, perturbation_amplitude: float = 1.0e-5) -> RoyEvoPDEConfig:
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
        seed=20260702,
    )


def profile_settings(profile: str) -> tuple[float, float, tuple[float, ...]]:
    if profile == "full":
        return 2000.0, 2400.0, FULL_BASIN_STRESSES
    return 1200.0, 1600.0, FOCUSED_BASIN_STRESSES


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
    duration = float(times[-1] - times[0])
    window = tail_fraction * duration
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


def physical_arrays(state: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    n, w, q = state
    z = free_space_evo(n, w, PARAMS)
    return bool(
        np.all(np.isfinite(n))
        and np.all(np.isfinite(w))
        and np.all(np.isfinite(q))
        and float(np.min(n)) >= -1.0e-8
        and float(np.min(w)) >= -1.0e-8
        and float(np.min(q)) >= -1.0e-6
        and float(np.max(q)) <= 1.0 + 1.0e-6
        and float(np.min(z)) >= -1.0e-5
    )


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
    return physical_arrays((result.n, result.w, result.q)) and bool(result.diagnostics.get("completed", True))


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
    }


def tail_diagnostics(result: RoyEvoPDEResult) -> dict[str, float | bool]:
    mask = tail_mask(result.t)
    prev_mask = previous_window_mask(result.t)
    tail_t = result.t[mask]
    tail_w = result.mean_w_time[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(result.t, result.mean_w_time, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_mean_w = float(np.mean(result.mean_w_time[prev_mask]))
    initial_q = float(result.diagnostics.get("initial_mean_q", result.mean_q_time[0]))
    tail_mean_q = float(np.mean(result.mean_q_time[mask]))
    physical = physical_from_result(result)
    persistent_without_slope = bool(physical and tail_mean_w > EPSILON and tail_min_w > 0.25 * EPSILON)
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    return {
        "physical": physical,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": slope_floor,
        "tail_mean_q": tail_mean_q,
        "q_change": float(tail_mean_q - initial_q),
        "previous_window_mean_w": previous_mean_w,
        "latest_window_mean_w": tail_mean_w,
        "relative_change_between_last_windows": relative_change(previous_mean_w, tail_mean_w),
        "persistent_with_slope_rule": persistent_with_slope,
        "persistent_without_slope_rule": persistent_without_slope,
    }


def classify_asymptotic(metrics: dict[str, object]) -> str:
    physical = bool(metrics["physical"])
    tail_mean_w = float(metrics["tail_mean_w"])
    tail_min_w = float(metrics.get("tail_min_w", tail_mean_w))
    previous_w = float(metrics["previous_window_mean_w"])
    latest_w = float(metrics["latest_window_mean_w"])
    rel_change = float(metrics["relative_change_between_last_windows"])
    normalized_residual = float(metrics["normalized_residual"])
    persistent_with_slope = bool(metrics.get("persistent_with_slope_rule", False))
    persistent_without_slope = bool(metrics.get("persistent_without_slope_rule", False))
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
    metrics: dict[str, object] = {
        **tail_diagnostics(result),
        **pde_evo_rhs_residual(result.n, result.w, result.q, PARAMS, config, stress=stress, evolve=True),
        "min_n": float(result.diagnostics.get("min_n", np.min(result.n))),
        "min_w": float(result.diagnostics.get("min_w", np.min(result.w))),
        "min_q": float(result.diagnostics.get("min_q", np.min(result.q))),
        "max_q": float(result.diagnostics.get("max_q", np.max(result.q))),
        "min_z": float(result.diagnostics.get("min_z", np.min(free_space_evo(result.n, result.w, PARAMS)))),
        "q_clip_count": int(result.diagnostics.get("q_clip_count", 0)),
        "q_clip_max_violation": float(result.diagnostics.get("q_clip_max_violation", 0.0)),
    }
    metrics["classification"] = classify_asymptotic(metrics)
    return metrics


def basin_label(classification: str) -> str:
    if classification == "persistent_steady":
        return "persistent_basin"
    if classification == "extinct_steady":
        return "extinct_basin"
    if classification in {"persistent_transient", "extinct_transient", "recovery_transient", "declining_transient"}:
        return "transient_basin"
    if classification == "nonphysical":
        return "nonphysical_initial_condition"
    return "unresolved_basin"


def aggregate_stress_regime(basin_labels: Iterable[str]) -> str:
    labels = [label for label in basin_labels if label != "nonphysical_initial_condition"]
    if not labels:
        return "unresolved"
    has_persistent = "persistent_basin" in labels
    has_extinct = "extinct_basin" in labels
    has_transient = "transient_basin" in labels
    has_unresolved = "unresolved_basin" in labels
    if has_persistent and not has_extinct and not has_transient and not has_unresolved:
        return "persistent_only"
    if has_extinct and not has_persistent and not has_transient and not has_unresolved:
        return "extinct_only"
    if has_persistent and has_extinct:
        return "bistable_persistent_extinct"
    if has_persistent and has_transient and not has_extinct:
        return "persistent_transient_mixed"
    return "unresolved"


def detect_direction_dependence(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    upward = {float(row["stress"]): str(row["classification"]) for row in rows if row["sweep_direction"] == "upward"}
    downward = {float(row["stress"]): str(row["classification"]) for row in rows if row["sweep_direction"] == "downward"}
    mismatches: list[dict[str, object]] = []
    for stress in sorted(set(upward) & set(downward)):
        if upward[stress] != downward[stress]:
            mismatches.append({"stress": stress, "upward": upward[stress], "downward": downward[stress]})
    return mismatches


def final_step13_label(regimes: dict[float, str], direction_dependent: bool) -> tuple[str, str]:
    if any(regime == "bistable_persistent_extinct" for regime in regimes.values()):
        return (
            "pde_evo_bistability_mapped",
            "At least one stress admits both persistent and extinct outcomes from different initial conditions.",
        )
    if direction_dependent:
        return (
            "pde_evo_hysteresis_confirmed_but_basins_unresolved",
            "Continuation direction-dependence is reproduced, but the basin scan does not resolve both persistent and extinct basins at one stress.",
        )
    return (
        "pde_evo_basin_mapping_inconclusive",
        "The focused basin scan does not resolve a bistable or direction-dependent regime.",
    )


def row_note(classification: str) -> str:
    if classification.endswith("_steady"):
        return "steady_by_tail_change_and_rhs_residual"
    if classification in {"declining_transient", "recovery_transient"}:
        return "window_change_indicates_transient"
    return classification


def continuation_row(
    direction: str,
    step_index: int,
    stress: float,
    T: float,
    result: RoyEvoPDEResult,
    config: RoyEvoPDEConfig,
    initial_source: str,
) -> dict[str, object]:
    metrics = result_metrics(result, config, stress)
    return {
        "run_id": f"{direction}_step{step_index:02d}_s{stress:.9g}".replace(".", "p"),
        "sweep_direction": direction,
        "step_index": int(step_index),
        "stress": float(stress),
        "T": float(T),
        "initial_source": initial_source,
        "classification": str(metrics["classification"]),
        "tail_mean_w": float(metrics["tail_mean_w"]),
        "tail_mean_q": float(metrics["tail_mean_q"]),
        "relative_change_between_last_windows": float(metrics["relative_change_between_last_windows"]),
        "normalized_residual": float(metrics["normalized_residual"]),
        "physical": bool(metrics["physical"]),
        "min_n": float(metrics["min_n"]),
        "min_w": float(metrics["min_w"]),
        "min_q": float(metrics["min_q"]),
        "max_q": float(metrics["max_q"]),
        "min_z": float(metrics["min_z"]),
        "notes": row_note(str(metrics["classification"])),
    }


def basin_row(
    stress: float,
    family: str,
    T: float,
    result: RoyEvoPDEResult,
    config: RoyEvoPDEConfig,
) -> dict[str, object]:
    metrics = result_metrics(result, config, stress)
    classification = str(metrics["classification"])
    return {
        "run_id": f"BASIN_s{stress:.9g}_{family}".replace(".", "p"),
        "stress": float(stress),
        "initial_condition_family": family,
        "T": float(T),
        "classification": classification,
        "tail_mean_w": float(metrics["tail_mean_w"]),
        "tail_mean_q": float(metrics["tail_mean_q"]),
        "q_change": float(metrics["q_change"]),
        "relative_change_between_last_windows": float(metrics["relative_change_between_last_windows"]),
        "normalized_residual": float(metrics["normalized_residual"]),
        "physical": bool(metrics["physical"]),
        "min_n": float(metrics["min_n"]),
        "min_w": float(metrics["min_w"]),
        "min_q": float(metrics["min_q"]),
        "max_q": float(metrics["max_q"]),
        "min_z": float(metrics["min_z"]),
        "basin_label": basin_label(classification),
        "notes": row_note(classification),
    }


def skipped_basin_row(stress: float, family: str, T: float, reason: str) -> dict[str, object]:
    return {
        "run_id": f"BASIN_s{stress:.9g}_{family}".replace(".", "p"),
        "stress": float(stress),
        "initial_condition_family": family,
        "T": float(T),
        "classification": "nonphysical",
        "tail_mean_w": float("nan"),
        "tail_mean_q": float("nan"),
        "q_change": float("nan"),
        "relative_change_between_last_windows": float("nan"),
        "normalized_residual": float("nan"),
        "physical": False,
        "min_n": float("nan"),
        "min_w": float("nan"),
        "min_q": float("nan"),
        "max_q": float("nan"),
        "min_z": float("nan"),
        "basin_label": "nonphysical_initial_condition",
        "notes": reason,
    }


def run_continuation(
    equilibrium: dict[str, object],
    T: float,
    direction: str,
) -> tuple[list[dict[str, object]], dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    stresses = CONTINUATION_STRESSES if direction == "upward" else tuple(reversed(CONTINUATION_STRESSES))
    config = pde_config(T)
    state = initial_state_from_ode_equilibrium(equilibrium, config)
    rows: list[dict[str, object]] = []
    final_states: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    source = "baseline_burnin"
    for index, stress in enumerate(stresses):
        print(f"{direction.upper()} step={index:02d} stress={stress:.9g} T={T:.0f}")
        result = simulate_pde_evo_2d(PARAMS, config, state, stress=stress, evolve=True)
        rows.append(continuation_row(direction, index, stress, T, result, config, source))
        state = (result.n.copy(), result.w.copy(), result.q.copy())
        final_states[float(stress)] = state
        source = f"{direction}_previous_stress_{stress:.9g}"
    return rows, final_states


def nearest_state_for_stress(
    states: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stress: float,
    prefer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    keys = sorted(states)
    if prefer == "lower":
        candidates = [key for key in keys if key <= stress + 1.0e-12]
        key = max(candidates) if candidates else None
    else:
        candidates = [key for key in keys if key >= stress - 1.0e-12]
        key = min(candidates) if candidates else None
    if key is None:
        return None
    return tuple(array.copy() for array in states[key])


def baseline_state(equilibrium: dict[str, object], config: RoyEvoPDEConfig, perturbation_amplitude: float = 1.0e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_config = pde_config(config.T, perturbation_amplitude=perturbation_amplitude)
    return initial_state_from_ode_equilibrium(equilibrium, local_config)


def initial_condition_state(
    family: str,
    stress: float,
    equilibrium: dict[str, object],
    config: RoyEvoPDEConfig,
    upward_states: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    downward_states: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    base = baseline_state(equilibrium, config)
    if family == "baseline_burnin":
        return base
    if family == "persistent_branch":
        return nearest_state_for_stress(upward_states, stress, "lower")
    if family == "extinct_branch":
        return nearest_state_for_stress(downward_states, stress, "higher")
    n, w, q = (array.copy() for array in base)
    if family == "low_predator":
        return n, 0.05 * w, q
    if family == "low_defense":
        return n, w, np.full_like(q, 0.1)
    if family == "high_defense":
        return n, w, np.full_like(q, 0.9)
    if family == "mixed_random":
        return baseline_state(equilibrium, config, perturbation_amplitude=1.0e-3)
    raise ValueError(f"Unknown initial-condition family {family!r}.")


def run_basin_scan(
    equilibrium: dict[str, object],
    basin_T: float,
    basin_stresses: tuple[float, ...],
    upward_states: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    downward_states: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = pde_config(basin_T)
    for stress in basin_stresses:
        for family in INITIAL_CONDITION_FAMILIES:
            state = initial_condition_state(family, stress, equilibrium, config, upward_states, downward_states)
            if state is None or not physical_arrays(state):
                rows.append(skipped_basin_row(stress, family, basin_T, "skipped_nonphysical_initial_condition"))
                continue
            print(f"BASIN stress={stress:.9g} family={family} T={basin_T:.0f}")
            result = simulate_pde_evo_2d(PARAMS, config, state, stress=stress, evolve=True)
            rows.append(basin_row(stress, family, basin_T, result, config))
    return rows


def regime_counts(basin_rows: list[dict[str, object]], stress: float) -> dict[str, int]:
    rows = [row for row in basin_rows if math.isclose(float(row["stress"]), stress)]
    labels = [str(row["basin_label"]) for row in rows]
    return {
        "persistent": labels.count("persistent_basin"),
        "extinct": labels.count("extinct_basin"),
        "transient_unresolved": labels.count("transient_basin") + labels.count("unresolved_basin"),
        "nonphysical": labels.count("nonphysical_initial_condition"),
    }


def stress_regimes(basin_rows: list[dict[str, object]], stresses: tuple[float, ...]) -> dict[float, str]:
    return {
        stress: aggregate_stress_regime(str(row["basin_label"]) for row in basin_rows if math.isclose(float(row["stress"]), stress))
        for stress in stresses
    }


def continuation_table_rows(hysteresis_rows: list[dict[str, object]], stresses: tuple[float, ...]) -> list[str]:
    upward = {float(row["stress"]): str(row["classification"]) for row in hysteresis_rows if row["sweep_direction"] == "upward"}
    downward = {float(row["stress"]): str(row["classification"]) for row in hysteresis_rows if row["sweep_direction"] == "downward"}
    lines = [
        "| stress | upward classification | downward classification | direction-dependent? |",
        "|---:|---|---|---|",
    ]
    for stress in stresses:
        up = upward.get(float(stress), "missing")
        down = downward.get(float(stress), "missing")
        lines.append(f"| {stress:.9g} | `{up}` | `{down}` | {up != down} |")
    return lines


def basin_table_rows(basin_rows: list[dict[str, object]], regimes: dict[float, str]) -> list[str]:
    lines = [
        "| stress | persistent basin count | extinct basin count | transient/unresolved count | regime label |",
        "|---:|---:|---:|---:|---|",
    ]
    for stress, regime in sorted(regimes.items()):
        counts = regime_counts(basin_rows, stress)
        lines.append(
            f"| {stress:.9g} | {counts['persistent']} | {counts['extinct']} | {counts['transient_unresolved']} | `{regime}` |"
        )
    return lines


def write_note(
    profile: str,
    continuation_T: float,
    basin_T: float,
    basin_stresses: tuple[float, ...],
    hysteresis_rows: list[dict[str, object]],
    basin_rows: list[dict[str, object]],
    regimes: dict[float, str],
    final_label: str,
    interpretation: str,
) -> None:
    NOTES_DIR.mkdir(exist_ok=True)
    direction_mismatches = detect_direction_dependence(hysteresis_rows)
    if final_label == "pde_evo_bistability_mapped":
        next_step = "Next: quantify basin boundaries within the mapped bistable interval."
        scalar_sentence = "A single scalar threshold is inappropriate because at least one stress has both persistent and extinct reachable outcomes."
    elif final_label == "pde_evo_hysteresis_confirmed_but_basins_unresolved":
        next_step = "Next: refine initial-condition families or longer continuation runs to resolve basin boundaries."
        scalar_sentence = "A single scalar threshold remains inappropriate because continuation direction affects outcomes, even though basin structure is not fully resolved."
    else:
        next_step = "Next: improve physicality and runtime diagnostics before basin interpretation."
        scalar_sentence = "A single scalar threshold should remain suspended until basin mapping is reliable."

    lines = [
        "# Research Note: Hysteresis and Basin Structure in the PDE-Evolution Model",
        "",
        "## Executive Summary",
        "",
        f"Final Step 13 label: `{final_label}`.",
        "",
        interpretation,
        "",
        scalar_sentence,
        "",
        "## Why This Was Needed",
        "",
        "PR #6 found `pde_evo_hysteresis_detected`: upward continuation kept a persistent branch to high stress, while downward continuation from high stress could remain extinct or transient at the same stresses.",
        "",
        "Step 13 therefore asks which asymptotic outcomes are reachable at each stress and how they depend on initial condition or continuation path. It does not diagnose spatial covariance mechanism.",
        "",
        "## Setup",
        "",
        f"- profile run: `{profile}`",
        "- model: PDE-evo only",
        "- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`",
        "- grid: `64x64`, `L_x=L_y=20`, seed `20260702`",
        "- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`",
        "- integration: `dt=0.1`, `record_every=50`",
        f"- continuation horizon: `{continuation_T:.0f}`",
        f"- basin scan horizon: `{basin_T:.0f}`",
        f"- basin stresses: `{', '.join(f'{stress:.9g}' for stress in basin_stresses)}`",
        "- initial families: `baseline_burnin`, `persistent_branch`, `extinct_branch`, `low_predator`, `low_defense`, `high_defense`, `mixed_random`",
        f"- steady criteria: `abs(relative_change) < {STEADY_REL_CHANGE_TOL}` and `normalized_residual < {STEADY_RESIDUAL_TOL}`",
        "",
        "## Continuation Hysteresis",
        "",
    ]
    lines.extend(continuation_table_rows(hysteresis_rows, SENSITIVE_STRESSES))
    lines.extend(
        [
            "",
            f"Direction-dependent continuation stresses: `{len(direction_mismatches)}`.",
            "",
            "## Initial-Condition Basin Scan",
            "",
        ]
    )
    lines.extend(basin_table_rows(basin_rows, regimes))
    lines.extend(
        [
            "",
            "## Stress-Regime Map",
            "",
            ", ".join(f"{stress:.9g}: `{regime}`" for stress, regime in sorted(regimes.items())),
            "",
            "## Interpretation",
            "",
            scalar_sentence,
            "",
            "Mechanism interpretation remains out of scope until the reachable outcome regimes are better resolved.",
            "",
            "## Files",
            "",
            "- `results/roy_pde_evo_hysteresis_map.csv`",
            "- `results/roy_pde_evo_basin_initial_condition_scan.csv`",
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
    continuation_T, basin_T, basin_stresses = profile_settings(profile)
    RESULTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS)
    upward_rows, upward_states = run_continuation(equilibrium, continuation_T, "upward")
    downward_rows, downward_states = run_continuation(equilibrium, continuation_T, "downward")
    hysteresis_rows = upward_rows + downward_rows
    basin_rows = run_basin_scan(equilibrium, basin_T, basin_stresses, upward_states, downward_states)
    regimes = stress_regimes(basin_rows, basin_stresses)
    direction_dependent = bool(detect_direction_dependence(hysteresis_rows))
    final_label, interpretation = final_step13_label(regimes, direction_dependent)

    write_csv(hysteresis_rows, HYSTERESIS_CSV, HYSTERESIS_FIELDNAMES)
    write_csv(basin_rows, BASIN_CSV, BASIN_FIELDNAMES)
    write_note(
        profile,
        continuation_T,
        basin_T,
        basin_stresses,
        hysteresis_rows,
        basin_rows,
        regimes,
        final_label,
        interpretation,
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
