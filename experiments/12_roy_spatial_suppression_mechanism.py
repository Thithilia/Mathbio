"""Step 10: check PDE-evolution threshold monotonicity and classifier stability.

This script replaces the premature spatial-suppression mechanism diagnosis with
a focused validation of the PDE-evo predator persistence boundary. It does not
scan parameters or change the model; it tests whether persistence is monotone
and stable near the disputed threshold region.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
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
    initial_state_from_ode_equilibrium,
    simulate_pde_evo_2d,
)


RESULTS_DIR = ROOT / "results"
NOTES_DIR = ROOT / "research_notes"
TIMESERIES_CSV = RESULTS_DIR / "roy_spatial_suppression_timeseries.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_spatial_suppression_summary.csv"
MONOTONICITY_CSV = RESULTS_DIR / "roy_spatial_suppression_monotonicity.csv"
SUMMARY_MD = NOTES_DIR / "roy_spatial_suppression_mechanism.md"

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
TAIL_FRACTIONS = (0.25, 0.35, 0.50)
FOCUSED_HORIZONS = (500.0, 800.0)
MINIMAL_HORIZONS = (500.0,)
STRESS_GRID = (
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

ODE_EVO_THRESHOLD = 0.16486816
PDE_EVO_THRESHOLD_STAGE_B = 0.11765625
PDE_EVO_S3_STRESS = 0.141262205

MONOTONICITY_FIELDNAMES = [
    "run_id",
    "stress",
    "T",
    "n_x",
    "n_y",
    "seed",
    "tail_fraction",
    "physical",
    "persistent_predator",
    "persistent_without_slope_check",
    "tail_mean_w",
    "tail_min_w",
    "tail_slope_w",
    "tail_slope_floor_w",
    "tail_mean_q",
    "tail_q_change",
    "tail_mean_var_q",
    "tail_mean_min_z",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
    "q_clip_count",
    "q_clip_max_violation",
    "classification_note",
]

TIMESERIES_FIELDNAMES = [
    "run_id",
    "stress",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
    "var_n",
    "var_w",
    "var_q",
    "min_z",
]

SUMMARY_FIELDNAMES = [
    "run_id",
    "stress",
    "T",
    "persistent_tail_025",
    "persistent_tail_035",
    "persistent_tail_050",
    "persistent_without_slope_tail_025",
    "tail_mean_w_tail_025",
    "tail_slope_w_tail_025",
    "tail_slope_floor_w_tail_025",
    "tail_mean_q_tail_025",
    "tail_q_change_tail_025",
    "tail_mean_var_q_tail_025",
    "classification_note",
]


@dataclass(frozen=True)
class StabilityFlags:
    physical_failures: int
    tail_fraction_disagreements: int
    horizon_disagreements: int
    slope_rule_disagreements: int
    reentry_sequences: int
    clean_transition_sequences: int
    near_tolerance_rows: int


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bool_text(value: object) -> str:
    return "True" if bool(value) else "False"


def tail_mask(times: np.ndarray, tail_fraction: float) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("Need at least two time points for tail classification.")
    cutoff = float(times[-1]) - tail_fraction * float(times[-1] - times[0])
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
    if denom <= 0.0:
        return 0.0
    return float(np.dot(centered_t, tail_values - float(np.mean(tail_values))) / denom)


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


def classify_tail_series(
    times: np.ndarray,
    mean_w: np.ndarray,
    mean_q: np.ndarray,
    var_q: np.ndarray,
    min_z: np.ndarray,
    initial_q: float,
    physical: bool = True,
    epsilon: float = EPSILON,
    tail_fraction: float = 0.25,
) -> dict[str, object]:
    """Classify a scalar predator time series with and without the slope check."""
    times = np.asarray(times, dtype=float)
    mean_w = np.asarray(mean_w, dtype=float)
    mean_q = np.asarray(mean_q, dtype=float)
    var_q = np.asarray(var_q, dtype=float)
    min_z = np.asarray(min_z, dtype=float)
    if len(times) < 2 or any(len(array) != len(times) for array in (mean_w, mean_q, var_q, min_z)):
        return {
            "physical": False,
            "persistent_predator": False,
            "persistent_without_slope_check": False,
            "tail_mean_w": float("nan"),
            "tail_min_w": float("nan"),
            "tail_slope_w": float("nan"),
            "tail_slope_floor_w": float("nan"),
            "tail_mean_q": float("nan"),
            "tail_q_change": float("nan"),
            "tail_mean_var_q": float("nan"),
            "tail_mean_min_z": float("nan"),
            "classification_note": "invalid_timeseries",
        }
    if any(not np.all(np.isfinite(array)) for array in (times, mean_w, mean_q, var_q, min_z)):
        physical = False

    mask = tail_mask(times, tail_fraction)
    tail_t = times[mask]
    tail_w = mean_w[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(times, mean_w, mask)
    slope_floor = -max(epsilon, 0.25 * tail_mean_w) / tail_duration
    persistent_without_slope = bool(physical and tail_mean_w > epsilon and tail_min_w > 0.25 * epsilon)
    persistent = bool(persistent_without_slope and slope_w >= slope_floor)
    tail_mean_q = float(np.mean(mean_q[mask]))
    near_tolerance = (
        abs(tail_mean_w - epsilon) <= 0.25 * epsilon
        or abs(tail_min_w - 0.25 * epsilon) <= 0.25 * epsilon
        or abs(slope_w - slope_floor) <= max(1.0e-8, 0.05 * abs(slope_floor))
    )
    if not physical:
        note = "nonphysical"
    elif persistent:
        note = "persistent"
    elif persistent_without_slope:
        note = "slope_check_rejects"
    elif near_tolerance:
        note = "near_classifier_tolerance"
    else:
        note = "not_persistent"
    return {
        "physical": bool(physical),
        "persistent_predator": persistent,
        "persistent_without_slope_check": persistent_without_slope,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": slope_floor,
        "tail_mean_q": tail_mean_q,
        "tail_q_change": float(tail_mean_q - initial_q),
        "tail_mean_var_q": float(np.mean(var_q[mask])),
        "tail_mean_min_z": float(np.mean(min_z[mask])),
        "classification_note": note,
    }


def classify_result_at_horizon(
    result: RoyEvoPDEResult,
    config: RoyEvoPDEConfig,
    stress: float,
    horizon: float,
    tail_fraction: float,
) -> dict[str, object]:
    prefix = result.t <= horizon + 1.0e-9
    if np.count_nonzero(prefix) < 2:
        raise ValueError(f"Not enough recorded points for horizon T={horizon:g}.")
    physical = physical_from_result(result)
    initial_q = float(result.diagnostics.get("initial_mean_q", result.mean_q_time[0]))
    metrics = classify_tail_series(
        result.t[prefix],
        result.mean_w_time[prefix],
        result.mean_q_time[prefix],
        result.var_q_time[prefix],
        result.min_z_time[prefix],
        initial_q=initial_q,
        physical=physical,
        tail_fraction=tail_fraction,
    )
    run_id = f"PDE_EVO_s{stress:.9g}_T{horizon:.0f}_tail{tail_fraction:.2f}".replace(".", "p")
    return {
        "run_id": run_id,
        "stress": float(stress),
        "T": float(horizon),
        "n_x": int(config.n_x),
        "n_y": int(config.n_y),
        "seed": int(config.seed),
        "tail_fraction": float(tail_fraction),
        "physical": bool(metrics["physical"]),
        "persistent_predator": bool(metrics["persistent_predator"]),
        "persistent_without_slope_check": bool(metrics["persistent_without_slope_check"]),
        "tail_mean_w": float(metrics["tail_mean_w"]),
        "tail_min_w": float(metrics["tail_min_w"]),
        "tail_slope_w": float(metrics["tail_slope_w"]),
        "tail_slope_floor_w": float(metrics["tail_slope_floor_w"]),
        "tail_mean_q": float(metrics["tail_mean_q"]),
        "tail_q_change": float(metrics["tail_q_change"]),
        "tail_mean_var_q": float(metrics["tail_mean_var_q"]),
        "tail_mean_min_z": float(metrics["tail_mean_min_z"]),
        "min_n": float(result.diagnostics.get("min_n", np.min(result.n))),
        "min_w": float(result.diagnostics.get("min_w", np.min(result.w))),
        "min_q": float(result.diagnostics.get("min_q", np.min(result.q))),
        "max_q": float(result.diagnostics.get("max_q", np.max(result.q))),
        "min_z": float(result.diagnostics.get("min_z", np.min(result.min_z_time))),
        "q_clip_count": int(result.diagnostics.get("q_clip_count", 0)),
        "q_clip_max_violation": float(result.diagnostics.get("q_clip_max_violation", 0.0)),
        "classification_note": str(metrics["classification_note"]),
    }


def timeseries_rows_for_result(result: RoyEvoPDEResult, stress: float) -> list[dict[str, object]]:
    run_id = f"PDE_EVO_s{stress:.9g}".replace(".", "p")
    return [
        {
            "run_id": run_id,
            "stress": float(stress),
            "time": float(t),
            "mean_n": float(mean_n),
            "mean_w": float(mean_w),
            "mean_q": float(mean_q),
            "var_n": float(var_n),
            "var_w": float(var_w),
            "var_q": float(var_q),
            "min_z": float(min_z),
        }
        for t, mean_n, mean_w, mean_q, var_n, var_w, var_q, min_z in zip(
            result.t,
            result.mean_n_time,
            result.mean_w_time,
            result.mean_q_time,
            result.var_n_time,
            result.var_w_time,
            result.var_q_time,
            result.min_z_time,
        )
    ]


def sequence_for(rows: list[dict[str, object]], horizon: float, tail_fraction: float) -> list[bool]:
    selected = [
        row
        for row in rows
        if math.isclose(float(row["T"]), horizon) and math.isclose(float(row["tail_fraction"]), tail_fraction)
    ]
    selected.sort(key=lambda row: float(row["stress"]))
    return [bool(row["persistent_predator"]) for row in selected]


def has_persistence_reentry(sequence: Iterable[bool]) -> bool:
    seen_persistent = False
    seen_loss_after_persistent = False
    for value in sequence:
        if value:
            if seen_loss_after_persistent:
                return True
            seen_persistent = True
        elif seen_persistent:
            seen_loss_after_persistent = True
    return False


def is_clean_monotone_transition(sequence: Iterable[bool]) -> bool:
    values = list(sequence)
    if not values or all(values) or not any(values):
        return False
    return not has_persistence_reentry(values)


def count_tail_fraction_disagreements(rows: list[dict[str, object]]) -> int:
    count = 0
    for stress in sorted({float(row["stress"]) for row in rows}):
        for horizon in sorted({float(row["T"]) for row in rows}):
            values = {
                bool(row["persistent_predator"])
                for row in rows
                if math.isclose(float(row["stress"]), stress) and math.isclose(float(row["T"]), horizon)
            }
            if len(values) > 1:
                count += 1
    return count


def count_horizon_disagreements(rows: list[dict[str, object]]) -> int:
    count = 0
    for stress in sorted({float(row["stress"]) for row in rows}):
        for tail_fraction in sorted({float(row["tail_fraction"]) for row in rows}):
            values = {
                bool(row["persistent_predator"])
                for row in rows
                if math.isclose(float(row["stress"]), stress) and math.isclose(float(row["tail_fraction"]), tail_fraction)
            }
            if len(values) > 1:
                count += 1
    return count


def count_slope_rule_disagreements(rows: list[dict[str, object]]) -> int:
    return sum(
        1
        for row in rows
        if bool(row["persistent_predator"]) != bool(row["persistent_without_slope_check"])
        and bool(row["physical"])
        and float(row["tail_mean_w"]) > EPSILON
        and float(row["tail_min_w"]) > 0.25 * EPSILON
    )


def count_near_tolerance_rows(rows: list[dict[str, object]]) -> int:
    count = 0
    for row in rows:
        tail_mean_w = float(row["tail_mean_w"])
        tail_min_w = float(row["tail_min_w"])
        slope_w = float(row["tail_slope_w"])
        slope_floor = float(row["tail_slope_floor_w"])
        if (
            abs(tail_mean_w - EPSILON) <= 0.25 * EPSILON
            or abs(tail_min_w - 0.25 * EPSILON) <= 0.25 * EPSILON
            or abs(slope_w - slope_floor) <= max(1.0e-8, 0.05 * abs(slope_floor))
        ):
            count += 1
    return count


def stability_flags(rows: list[dict[str, object]], horizons: Iterable[float], tail_fractions: Iterable[float]) -> StabilityFlags:
    reentry_sequences = 0
    clean_transition_sequences = 0
    for horizon in horizons:
        for tail_fraction in tail_fractions:
            sequence = sequence_for(rows, horizon, tail_fraction)
            if has_persistence_reentry(sequence):
                reentry_sequences += 1
            if is_clean_monotone_transition(sequence):
                clean_transition_sequences += 1
    return StabilityFlags(
        physical_failures=sum(1 for row in rows if not bool(row["physical"])),
        tail_fraction_disagreements=count_tail_fraction_disagreements(rows),
        horizon_disagreements=count_horizon_disagreements(rows),
        slope_rule_disagreements=count_slope_rule_disagreements(rows),
        reentry_sequences=reentry_sequences,
        clean_transition_sequences=clean_transition_sequences,
        near_tolerance_rows=count_near_tolerance_rows(rows),
    )


def decide_final_label(rows: list[dict[str, object]], horizons: Iterable[float], tail_fractions: Iterable[float]) -> tuple[str, str, StabilityFlags]:
    horizons_tuple = tuple(horizons)
    tail_fractions_tuple = tuple(tail_fractions)
    flags = stability_flags(rows, horizons_tuple, tail_fractions_tuple)
    total_sequences = len(horizons_tuple) * len(tail_fractions_tuple)
    if flags.physical_failures:
        return (
            "pde_evo_threshold_inconclusive",
            "At least one PDE-evo trajectory failed physicality checks, so threshold interpretation is unreliable.",
            flags,
        )
    if flags.tail_fraction_disagreements or flags.horizon_disagreements or flags.slope_rule_disagreements:
        return (
            "pde_evo_threshold_classifier_sensitive",
            "Persistence classifications change with tail fraction, time horizon, or the tail-slope rule.",
            flags,
        )
    if flags.reentry_sequences:
        return (
            "pde_evo_threshold_nonmonotone",
            "PDE-evo persistence disappears and then reappears at higher stress under fixed classifier settings.",
            flags,
        )
    if flags.clean_transition_sequences == total_sequences and flags.near_tolerance_rows == 0:
        return (
            "pde_evo_threshold_monotone_stable",
            "Increasing stress gives a clean persistent-to-nonpersistent transition across horizons and tail fractions.",
            flags,
        )
    return (
        "pde_evo_threshold_inconclusive",
        "The focused runs do not provide a stable monotone threshold boundary.",
        flags,
    )


def summary_rows(monotonicity_rows: list[dict[str, object]], horizons: Iterable[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stress in STRESS_GRID:
        for horizon in horizons:
            grouped = {
                float(row["tail_fraction"]): row
                for row in monotonicity_rows
                if math.isclose(float(row["stress"]), stress) and math.isclose(float(row["T"]), horizon)
            }
            base = grouped[0.25]
            notes = sorted({str(row["classification_note"]) for row in grouped.values()})
            rows.append(
                {
                    "run_id": f"PDE_EVO_s{stress:.9g}_T{horizon:.0f}".replace(".", "p"),
                    "stress": float(stress),
                    "T": float(horizon),
                    "persistent_tail_025": bool_text(grouped[0.25]["persistent_predator"]),
                    "persistent_tail_035": bool_text(grouped[0.35]["persistent_predator"]),
                    "persistent_tail_050": bool_text(grouped[0.50]["persistent_predator"]),
                    "persistent_without_slope_tail_025": bool_text(base["persistent_without_slope_check"]),
                    "tail_mean_w_tail_025": float(base["tail_mean_w"]),
                    "tail_slope_w_tail_025": float(base["tail_slope_w"]),
                    "tail_slope_floor_w_tail_025": float(base["tail_slope_floor_w"]),
                    "tail_mean_q_tail_025": float(base["tail_mean_q"]),
                    "tail_q_change_tail_025": float(base["tail_q_change"]),
                    "tail_mean_var_q_tail_025": float(base["tail_mean_var_q"]),
                    "classification_note": ";".join(notes),
                }
            )
    return rows


def markdown_table_for_base(rows: list[dict[str, object]], horizon: float) -> list[str]:
    selected = [
        row
        for row in rows
        if math.isclose(float(row["T"]), horizon) and math.isclose(float(row["tail_fraction"]), 0.25)
    ]
    selected.sort(key=lambda row: float(row["stress"]))
    lines = [
        "| stress | persistent | no-slope persistent | tail mean w | tail slope | slope floor | tail mean q | note |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['stress']):.9g}",
                    str(bool(row["persistent_predator"])),
                    str(bool(row["persistent_without_slope_check"])),
                    f"{float(row['tail_mean_w']):.8g}",
                    f"{float(row['tail_slope_w']):.8g}",
                    f"{float(row['tail_slope_floor_w']):.8g}",
                    f"{float(row['tail_mean_q']):.8g}",
                    f"`{row['classification_note']}`",
                ]
            )
            + " |"
        )
    return lines


def markdown_tail_sensitivity_table(summary: list[dict[str, object]]) -> list[str]:
    lines = [
        "| stress | T | tail 0.25 | tail 0.35 | tail 0.50 | no-slope tail 0.25 |",
        "|---:|---:|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['stress']):.9g}",
                    f"{float(row['T']):.0f}",
                    str(row["persistent_tail_025"]),
                    str(row["persistent_tail_035"]),
                    str(row["persistent_tail_050"]),
                    str(row["persistent_without_slope_tail_025"]),
                ]
            )
            + " |"
        )
    return lines


def write_note(
    profile: str,
    horizons: tuple[float, ...],
    monotonicity_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    final_label: str,
    interpretation: str,
    flags: StabilityFlags,
) -> None:
    NOTES_DIR.mkdir(exist_ok=True)
    base_t500 = markdown_table_for_base(monotonicity_rows, 500.0) if 500.0 in horizons else []
    base_t800 = markdown_table_for_base(monotonicity_rows, 800.0) if 800.0 in horizons else []
    tail_table = markdown_tail_sensitivity_table(summary)
    any_persistent_above_stage_b = any(
        float(row["stress"]) > PDE_EVO_THRESHOLD_STAGE_B and bool(row["persistent_predator"])
        for row in monotonicity_rows
        if math.isclose(float(row["tail_fraction"]), 0.25)
    )
    stage_b_warning = (
        "The earlier Stage B threshold should not be treated as a stable monotone boundary without additional checks."
        if any_persistent_above_stage_b
        else "The focused run did not find persistence above the earlier Stage B threshold under the base classifier."
    )
    if final_label == "pde_evo_threshold_classifier_sensitive":
        implication = (
            "The apparent threshold is classifier-sensitive; the next step should stabilize the persistence criterion before interpreting spatial mechanism."
        )
    elif final_label == "pde_evo_threshold_monotone_stable":
        implication = "The mechanism analysis can proceed using the verified threshold region."
    elif final_label == "pde_evo_threshold_nonmonotone":
        implication = "The PDE-evo persistence boundary is not monotone on the focused stress grid."
    else:
        implication = "The threshold boundary remains insufficiently resolved for mechanism interpretation."

    lines = [
        "# Research Note: PDE-Evolution Threshold Monotonicity Check",
        "",
        "## Executive Summary",
        "",
        f"Final Step 10 label: `{final_label}`.",
        "",
        interpretation,
        "",
        stage_b_warning,
        "",
        implication,
        "",
        "## Why This Check Was Needed",
        "",
        "Step 10 originally tried to diagnose the spatial suppression mechanism. But the focused S3 run did not reproduce the expected ODE-persistent/PDE-failed contrast: PDE-evo also remained persistent at S3.",
        "",
        "Therefore mechanism diagnosis is premature. This note tests whether PDE-evo persistence is monotone and stable near the reported PDE and ODE thresholds.",
        "",
        "## Setup",
        "",
        f"- profile run: `{profile}`",
        "- model: PDE-evo only",
        "- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`",
        "- initial state: unstressed Step 09A burn-in via `find_evo_equilibrium`",
        "- grid: `64x64`, `L_x=L_y=20`, seed `20260702`",
        "- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`",
        "- integration: `dt=0.1`, `record_every=50`",
        f"- evaluated horizons: `{', '.join(f'{h:.0f}' for h in horizons)}`",
        "- tail fractions: `0.25`, `0.35`, `0.50`",
        "- base persistence rule: physical trajectory, tail mean predator density above `1e-4`, tail minimum above `2.5e-5`, and tail slope no more negative than the tail-slope floor",
        "- relaxed rule: same density thresholds but without the tail-slope check",
        "",
        "## Stress Grid",
        "",
        "The focused grid was:",
        "",
        "```text",
        ", ".join(f"{stress:.9g}" for stress in STRESS_GRID),
        "```",
        "",
        "This covers the shared rescue region, the reported Stage B PDE threshold, the S3 midpoint, the Stage C/ODE-threshold region, and above the ODE threshold.",
        "",
        "## Persistence Classification Results",
        "",
    ]
    if base_t500:
        lines.extend(["Base classifier results for `T=500`, `tail_fraction=0.25`:", ""])
        lines.extend(base_t500)
        lines.append("")
    if base_t800:
        lines.extend(["Base classifier results for `T=800`, `tail_fraction=0.25`:", ""])
        lines.extend(base_t800)
        lines.append("")
    lines.extend(
        [
            "## Classifier Sensitivity",
            "",
            f"- tail-fraction disagreement groups: `{flags.tail_fraction_disagreements}`",
            f"- horizon disagreement groups: `{flags.horizon_disagreements}`",
            f"- slope-rule disagreement rows: `{flags.slope_rule_disagreements}`",
            f"- near-tolerance rows: `{flags.near_tolerance_rows}`",
            "",
        ]
    )
    lines.extend(tail_table)
    lines.extend(
        [
            "",
            "## Monotonicity Diagnosis",
            "",
            f"- re-entry sequences: `{flags.reentry_sequences}`",
            f"- clean transition sequences: `{flags.clean_transition_sequences}`",
            f"- physical failures: `{flags.physical_failures}`",
            "",
            f"Final Step 10 label: `{final_label}`.",
            "",
            interpretation,
            "",
            "## Implication for Spatial Suppression Mechanism",
            "",
            "The spatial suppression mechanism should not be claimed from the current Step 10 diagnostics.",
            "",
            implication,
            "",
            "This check does not alter the PR #3 conclusions; it only shows that the PDE-evo threshold boundary used for mechanism localization needs a stability audit before mechanistic interpretation.",
            "",
            "## Files",
            "",
            "- `results/roy_spatial_suppression_monotonicity.csv`",
            "- `results/roy_spatial_suppression_summary.csv`",
            "- `results/roy_spatial_suppression_timeseries.csv`",
            "",
            "## Next Step",
            "",
            "Next: stabilize the PDE-evo persistence criterion and threshold boundary before interpreting spatial mechanism.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pde_config(max_horizon: float) -> RoyEvoPDEConfig:
    return RoyEvoPDEConfig(
        n_x=64,
        n_y=64,
        L_x=20.0,
        L_y=20.0,
        dt=0.1,
        T=float(max_horizon),
        record_every=50,
        D_n=0.01,
        D_w=0.01,
        D_q=0.005,
        seed=20260702,
    )


def horizons_for_profile(profile: str) -> tuple[float, ...]:
    if profile == "minimal":
        return MINIMAL_HORIZONS
    return FOCUSED_HORIZONS


def run(profile: str = "focused") -> tuple[str, str]:
    horizons = horizons_for_profile(profile)
    max_horizon = max(horizons)
    config = pde_config(max_horizon)
    RESULTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS)
    initial_state = initial_state_from_ode_equilibrium(equilibrium, config)

    monotonicity_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []
    for stress in STRESS_GRID:
        print(f"PDE_EVO stress={stress:.9g} T={max_horizon:.0f}")
        result = simulate_pde_evo_2d(PARAMS, config, initial_state, stress=stress, evolve=True)
        timeseries_rows.extend(timeseries_rows_for_result(result, stress))
        for horizon in horizons:
            for tail_fraction in TAIL_FRACTIONS:
                monotonicity_rows.append(classify_result_at_horizon(result, config, stress, horizon, tail_fraction))

    summary = summary_rows(monotonicity_rows, horizons)
    final_label, interpretation, flags = decide_final_label(monotonicity_rows, horizons, TAIL_FRACTIONS)
    write_csv(monotonicity_rows, MONOTONICITY_CSV, MONOTONICITY_FIELDNAMES)
    write_csv(summary, SUMMARY_CSV, SUMMARY_FIELDNAMES)
    write_csv(timeseries_rows, TIMESERIES_CSV, TIMESERIES_FIELDNAMES)
    write_note(profile, horizons, monotonicity_rows, summary, final_label, interpretation, flags)
    print(SUMMARY_MD.read_text(encoding="utf-8"))
    return final_label, interpretation


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["minimal", "focused", "full"], default="focused")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
