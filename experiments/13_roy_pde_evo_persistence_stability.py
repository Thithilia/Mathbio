"""Step 11: stabilize PDE-evolution persistence classification.

This experiment tests whether PDE-evo predator persistence near the apparent
threshold region can be classified by a multi-horizon criterion instead of a
single tail-slope decision. It does not change the model equations and does not
diagnose the spatial suppression mechanism.
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
DETAIL_CSV = RESULTS_DIR / "roy_pde_evo_persistence_stability.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_pde_evo_persistence_stability_summary.csv"
SUMMARY_MD = NOTES_DIR / "roy_pde_evo_persistence_criterion.md"

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
TAIL_FRACTIONS = (0.25, 0.35, 0.50)
FOCUSED_HORIZONS = (500.0, 800.0, 1200.0)
FULL_HORIZONS = (500.0, 800.0, 1200.0, 1600.0)
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

DETAIL_FIELDNAMES = [
    "run_id",
    "stress",
    "T",
    "n_x",
    "n_y",
    "seed",
    "tail_fraction",
    "physical",
    "persistent_with_slope_rule",
    "persistent_without_slope_rule",
    "horizon_status",
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
    "trajectory_note",
]

SUMMARY_FIELDNAMES = [
    "stress",
    "final_status",
    "latest_tail_mean_w",
    "previous_tail_mean_w",
    "relative_change_latest",
    "horizon_sequence",
    "tail_fraction_disagreement",
    "slope_rule_disagreement",
    "physical_failure",
    "notes",
]


@dataclass(frozen=True)
class TailMetrics:
    physical: bool
    persistent_with_slope_rule: bool
    persistent_without_slope_rule: bool
    tail_mean_w: float
    tail_min_w: float
    tail_slope_w: float
    tail_slope_floor_w: float
    tail_mean_q: float
    tail_q_change: float
    tail_mean_var_q: float
    tail_mean_min_z: float


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def relative_change(previous: float, latest: float) -> float:
    denom = max(abs(previous), EPSILON)
    return float((latest - previous) / denom)


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


def classify_tail_metrics(
    times: np.ndarray,
    mean_w: np.ndarray,
    mean_q: np.ndarray,
    var_q: np.ndarray,
    min_z: np.ndarray,
    initial_q: float,
    physical: bool = True,
    tail_fraction: float = 0.25,
    epsilon: float = EPSILON,
) -> TailMetrics:
    times = np.asarray(times, dtype=float)
    mean_w = np.asarray(mean_w, dtype=float)
    mean_q = np.asarray(mean_q, dtype=float)
    var_q = np.asarray(var_q, dtype=float)
    min_z = np.asarray(min_z, dtype=float)
    if len(times) < 2 or any(len(array) != len(times) for array in (mean_w, mean_q, var_q, min_z)):
        return TailMetrics(False, False, False, *(float("nan") for _ in range(8)))
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
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    tail_mean_q = float(np.mean(mean_q[mask]))
    return TailMetrics(
        physical=bool(physical),
        persistent_with_slope_rule=persistent_with_slope,
        persistent_without_slope_rule=persistent_without_slope,
        tail_mean_w=tail_mean_w,
        tail_min_w=tail_min_w,
        tail_slope_w=slope_w,
        tail_slope_floor_w=slope_floor,
        tail_mean_q=tail_mean_q,
        tail_q_change=float(tail_mean_q - initial_q),
        tail_mean_var_q=float(np.mean(var_q[mask])),
        tail_mean_min_z=float(np.mean(min_z[mask])),
    )


def classify_horizon_status(metrics: TailMetrics, previous_tail_mean_w: float | None = None) -> str:
    if not metrics.physical:
        return "indeterminate"
    if (
        metrics.tail_mean_w < EPSILON
        and metrics.tail_min_w < EPSILON
        and not metrics.persistent_without_slope_rule
    ):
        return "extinct_like"
    if (
        metrics.persistent_without_slope_rule
        and not metrics.persistent_with_slope_rule
        and metrics.tail_slope_w < metrics.tail_slope_floor_w
    ):
        return "declining_transient"
    if (
        metrics.persistent_with_slope_rule
        and metrics.tail_slope_w > 0.0
        and previous_tail_mean_w is not None
        and relative_change(previous_tail_mean_w, metrics.tail_mean_w) > 0.05
    ):
        return "recovery_transient"
    if (
        metrics.persistent_without_slope_rule
        and metrics.tail_mean_w > EPSILON
        and metrics.tail_min_w > 0.25 * EPSILON
    ):
        return "persistent_like"
    return "indeterminate"


def classify_result_rows(
    result: RoyEvoPDEResult,
    config: RoyEvoPDEConfig,
    stress: float,
    horizons: tuple[float, ...],
    tail_fractions: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    physical = physical_from_result(result)
    initial_q = float(result.diagnostics.get("initial_mean_q", result.mean_q_time[0]))
    for tail_fraction in tail_fractions:
        previous_mean: float | None = None
        for horizon in horizons:
            prefix = result.t <= horizon + 1.0e-9
            if np.count_nonzero(prefix) < 2:
                raise ValueError(f"Not enough recorded points for horizon T={horizon:g}.")
            metrics = classify_tail_metrics(
                result.t[prefix],
                result.mean_w_time[prefix],
                result.mean_q_time[prefix],
                result.var_q_time[prefix],
                result.min_z_time[prefix],
                initial_q=initial_q,
                physical=physical,
                tail_fraction=tail_fraction,
            )
            status = classify_horizon_status(metrics, previous_tail_mean_w=previous_mean)
            if not metrics.physical:
                note = "nonphysical"
            elif status == "declining_transient":
                note = "slope_rule_rejects_positive_tail"
            elif status == "recovery_transient":
                note = "positive_recovery_from_previous_horizon"
            else:
                note = status
            rows.append(
                {
                    "run_id": f"PDE_EVO_s{stress:.9g}_T{horizon:.0f}_tail{tail_fraction:.2f}".replace(".", "p"),
                    "stress": float(stress),
                    "T": float(horizon),
                    "n_x": int(config.n_x),
                    "n_y": int(config.n_y),
                    "seed": int(config.seed),
                    "tail_fraction": float(tail_fraction),
                    "physical": bool(metrics.physical),
                    "persistent_with_slope_rule": bool(metrics.persistent_with_slope_rule),
                    "persistent_without_slope_rule": bool(metrics.persistent_without_slope_rule),
                    "horizon_status": status,
                    "tail_mean_w": float(metrics.tail_mean_w),
                    "tail_min_w": float(metrics.tail_min_w),
                    "tail_slope_w": float(metrics.tail_slope_w),
                    "tail_slope_floor_w": float(metrics.tail_slope_floor_w),
                    "tail_mean_q": float(metrics.tail_mean_q),
                    "tail_q_change": float(metrics.tail_q_change),
                    "tail_mean_var_q": float(metrics.tail_mean_var_q),
                    "tail_mean_min_z": float(metrics.tail_mean_min_z),
                    "min_n": float(result.diagnostics.get("min_n", np.min(result.n))),
                    "min_w": float(result.diagnostics.get("min_w", np.min(result.w))),
                    "min_q": float(result.diagnostics.get("min_q", np.min(result.q))),
                    "max_q": float(result.diagnostics.get("max_q", np.max(result.q))),
                    "min_z": float(result.diagnostics.get("min_z", np.min(result.min_z_time))),
                    "q_clip_count": int(result.diagnostics.get("q_clip_count", 0)),
                    "q_clip_max_violation": float(result.diagnostics.get("q_clip_max_violation", 0.0)),
                    "trajectory_note": note,
                }
            )
            previous_mean = metrics.tail_mean_w
    return rows


def aggregate_tail_fraction_status(rows: list[dict[str, object]], tail_fraction: float) -> str:
    selected = [row for row in rows if math.isclose(float(row["tail_fraction"]), tail_fraction)]
    selected.sort(key=lambda row: float(row["T"]))
    if any(not bool(row["physical"]) for row in selected):
        return "indeterminate"
    if len(selected) < 2:
        return "indeterminate"
    latest = selected[-1]
    previous = selected[-2]
    latest_status = str(latest["horizon_status"])
    previous_status = str(previous["horizon_status"])
    latest_w = float(latest["tail_mean_w"])
    previous_w = float(previous["tail_mean_w"])
    rel = abs(relative_change(previous_w, latest_w))
    statuses = [str(row["horizon_status"]) for row in selected]

    if (
        latest_status == "persistent_like"
        and previous_status in {"persistent_like", "recovery_transient"}
        and rel < 0.05
    ):
        return "persistent_confirmed"
    if latest_status == "extinct_like" and previous_status == "extinct_like" and latest_w < EPSILON:
        return "extinct_confirmed"
    if latest_status == "declining_transient" and latest_w < previous_w:
        return "declining_transient"
    if (
        latest_status == "persistent_like"
        and any(status in {"declining_transient", "extinct_like"} for status in statuses[:-1])
        and relative_change(previous_w, latest_w) > 0.05
    ):
        return "recovery_transient"
    return "indeterminate"


def majority_status(statuses: list[str]) -> str | None:
    counts = {status: statuses.count(status) for status in set(statuses)}
    top_status, top_count = max(counts.items(), key=lambda item: item[1])
    return top_status if top_count > len(statuses) / 2.0 else None


def aggregate_stress_status(rows: list[dict[str, object]], tail_fractions: tuple[float, ...] = TAIL_FRACTIONS) -> dict[str, object]:
    per_tail = {tail: aggregate_tail_fraction_status(rows, tail) for tail in tail_fractions}
    statuses = list(per_tail.values())
    tail_fraction_disagreement = len(set(statuses)) > 1
    final_status = statuses[0] if len(set(statuses)) == 1 else majority_status(statuses) or "indeterminate"
    selected_025 = [row for row in rows if math.isclose(float(row["tail_fraction"]), 0.25)]
    selected_025.sort(key=lambda row: float(row["T"]))
    latest = selected_025[-1]
    previous = selected_025[-2] if len(selected_025) >= 2 else latest
    horizon_sequence = ";".join(f"T{float(row['T']):.0f}:{row['horizon_status']}" for row in selected_025)
    slope_rule_disagreement = any(
        bool(row["persistent_with_slope_rule"]) != bool(row["persistent_without_slope_rule"])
        for row in rows
        if bool(row["physical"])
    )
    physical_failure = any(not bool(row["physical"]) for row in rows)
    notes = ",".join(f"tail{tail:.2f}={status}" for tail, status in per_tail.items())
    return {
        "stress": float(latest["stress"]),
        "final_status": final_status,
        "latest_tail_mean_w": float(latest["tail_mean_w"]),
        "previous_tail_mean_w": float(previous["tail_mean_w"]),
        "relative_change_latest": float(relative_change(float(previous["tail_mean_w"]), float(latest["tail_mean_w"]))),
        "horizon_sequence": horizon_sequence,
        "tail_fraction_disagreement": bool(tail_fraction_disagreement),
        "slope_rule_disagreement": bool(slope_rule_disagreement),
        "physical_failure": bool(physical_failure),
        "notes": notes,
    }


def has_nonmonotone_boundary(statuses: list[str]) -> bool:
    seen_persistent = False
    seen_loss = False
    for status in statuses:
        if status in {"persistent_confirmed", "recovery_transient"}:
            if seen_loss:
                return True
            seen_persistent = True
        elif status in {"extinct_confirmed", "declining_transient"} and seen_persistent:
            seen_loss = True
    return False


def clean_persistent_to_extinct(statuses: list[str]) -> bool:
    if not statuses or any(status not in {"persistent_confirmed", "extinct_confirmed"} for status in statuses):
        return False
    if "persistent_confirmed" not in statuses or "extinct_confirmed" not in statuses:
        return False
    seen_extinct = False
    for status in statuses:
        if status == "extinct_confirmed":
            seen_extinct = True
        elif seen_extinct and status == "persistent_confirmed":
            return False
    return True


def latest_transient_stress_count(detail_rows: list[dict[str, object]], max_horizon: float) -> int:
    count = 0
    for stress in sorted({float(row["stress"]) for row in detail_rows}):
        latest_rows = [
            row
            for row in detail_rows
            if math.isclose(float(row["stress"]), stress) and math.isclose(float(row["T"]), max_horizon)
        ]
        transient_rows = [
            row for row in latest_rows if str(row["horizon_status"]) in {"declining_transient", "recovery_transient"}
        ]
        if len(transient_rows) >= 2:
            count += 1
    return count


def final_step11_label(summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]], max_horizon: float) -> tuple[str, str]:
    statuses = [str(row["final_status"]) for row in sorted(summary_rows, key=lambda row: float(row["stress"]))]
    if any(bool(row["physical_failure"]) for row in summary_rows):
        return (
            "pde_evo_persistence_unresolved",
            "At least one trajectory failed physicality checks, so persistence cannot be resolved.",
        )
    if clean_persistent_to_extinct(statuses):
        return (
            "stable_persistence_criterion_established",
            "The multi-horizon classification gives a clean persistent-to-extinct split.",
        )
    if has_nonmonotone_boundary(statuses):
        return (
            "pde_evo_boundary_nonmonotone",
            "Confirmed persistence reappears at higher stress after extinction or declining-transient behavior.",
        )
    transient_count = latest_transient_stress_count(detail_rows, max_horizon)
    if transient_count >= math.ceil(0.4 * len(summary_rows)):
        return (
            "pde_evo_long_transients_dominate",
            "At least 40 percent of stress values remain in transient classes at the longest horizon.",
        )
    return (
        "pde_evo_persistence_unresolved",
        "The multi-horizon statuses do not form a clean stable boundary.",
    )


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
    if profile == "full":
        return FULL_HORIZONS
    return FOCUSED_HORIZONS


def markdown_status_table(summary_rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| stress | final status | latest tail mean w | horizon sequence | q behavior | note |",
        "|---:|---|---:|---|---|---|",
    ]
    for row in sorted(summary_rows, key=lambda item: float(item["stress"])):
        q_note = "see detail CSV"
        if str(row["final_status"]) in {"declining_transient", "recovery_transient"}:
            q_note = "active during transient"
        elif str(row["final_status"]) == "persistent_confirmed":
            q_note = "stable persistent tail"
        elif str(row["final_status"]) == "extinct_confirmed":
            q_note = "predator tail below epsilon"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['stress']):.9g}",
                    f"`{row['final_status']}`",
                    f"{float(row['latest_tail_mean_w']):.8g}",
                    f"`{row['horizon_sequence']}`",
                    q_note,
                    f"`{row['notes']}`",
                ]
            )
            + " |"
        )
    return lines


def write_note(
    profile: str,
    horizons: tuple[float, ...],
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    final_label: str,
    interpretation: str,
) -> None:
    NOTES_DIR.mkdir(exist_ok=True)
    statuses = [str(row["final_status"]) for row in sorted(summary_rows, key=lambda row: float(row["stress"]))]
    transient_count = latest_transient_stress_count(detail_rows, max(horizons))
    table = markdown_status_table(summary_rows)
    if final_label == "stable_persistence_criterion_established":
        next_step = "Next: re-run the spatial suppression mechanism analysis using the stabilized persistence criterion."
        implication = "A stabilized persistence boundary is available for later mechanism work."
    elif final_label == "pde_evo_boundary_nonmonotone":
        next_step = "Next: replace single-threshold language with stress-response regime mapping."
        implication = "The response cannot be reduced to one monotone threshold."
    elif final_label == "pde_evo_long_transients_dominate":
        next_step = "Next: use longer horizons or continuation/steady-state methods before threshold claims."
        implication = "Spatial suppression mechanism should not be interpreted while long transients dominate the persistence boundary."
    else:
        next_step = "Next: inspect numerical stability and physicality before further interpretation."
        implication = "Spatial suppression mechanism should not be interpreted because the persistence boundary remains unresolved."

    lines = [
        "# Research Note: Stabilizing the PDE-Evolution Persistence Criterion",
        "",
        "## Executive Summary",
        "",
        f"Final Step 11 label: `{final_label}`.",
        "",
        interpretation,
        "",
        implication,
        "",
        "## Why This Was Needed",
        "",
        "PR #4 found that the apparent PDE-evo threshold depended strongly on tail fraction, time horizon, and the tail-slope rule. It also found re-entry-like behavior in the focused stress grid.",
        "",
        "That makes spatial suppression mechanism interpretation premature. Step 11 therefore stabilizes the persistence classification itself before any mechanism diagnosis is attempted.",
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
        f"- stress grid: `{', '.join(f'{stress:.9g}' for stress in STRESS_GRID)}`",
        f"- horizons: `{', '.join(f'{horizon:.0f}' for horizon in horizons)}`",
        "- tail fractions: `0.25`, `0.35`, `0.50`",
        "",
        "## Classification Framework",
        "",
        "For each horizon and tail fraction, Step 11 computes tail mean predator density, tail minimum predator density, tail slope, the tail-slope floor, tail mean q, q change, var(q), minimum free space, physicality, slope-rule persistence, and no-slope persistence.",
        "",
        "The per-horizon classes are:",
        "",
        "- `persistent_like`: positive predator tail that passes the density checks without an active declining-transient rejection.",
        "- `extinct_like`: predator tail remains below epsilon and does not pass the no-slope persistence rule.",
        "- `declining_transient`: predator density is still positive, but the slope rule rejects it because the tail is declining too fast.",
        "- `recovery_transient`: predator density passes the slope rule and increases substantially relative to the previous shorter horizon.",
        "- `indeterminate`: any ambiguous case.",
        "",
        "The cross-horizon classes are `persistent_confirmed`, `extinct_confirmed`, `declining_transient`, `recovery_transient`, and `indeterminate`. This is better than one hard tail-slope threshold because it separates stable tails from long declining or recovery transients and makes horizon sensitivity explicit.",
        "",
        "## Stress-by-Stress Results",
        "",
    ]
    lines.extend(table)
    lines.extend(
        [
            "",
            "## Boundary Diagnosis",
            "",
            f"- final status sequence: `{', '.join(statuses)}`",
            f"- transient stress count at longest horizon: `{transient_count}` of `{len(summary_rows)}`",
            f"- final label: `{final_label}`",
            "",
            interpretation,
            "",
            "## Implication for Spatial Suppression",
            "",
            "Spatial suppression mechanism should not be interpreted until this persistence criterion is stable, unless the final label establishes a stable boundary.",
            "",
            implication,
            "",
            "## Files",
            "",
            "- `results/roy_pde_evo_persistence_stability.csv`",
            "- `results/roy_pde_evo_persistence_stability_summary.csv`",
            "",
            "## Next Step",
            "",
            next_step,
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(profile: str = "focused") -> tuple[str, str]:
    horizons = horizons_for_profile(profile)
    max_horizon = max(horizons)
    config = pde_config(max_horizon)
    RESULTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS)
    initial_state = initial_state_from_ode_equilibrium(equilibrium, config)

    detail_rows: list[dict[str, object]] = []
    for stress in STRESS_GRID:
        print(f"PDE_EVO stress={stress:.9g} T={max_horizon:.0f}")
        result = simulate_pde_evo_2d(PARAMS, config, initial_state, stress=stress, evolve=True)
        detail_rows.extend(classify_result_rows(result, config, stress, horizons, TAIL_FRACTIONS))

    summary_rows: list[dict[str, object]] = []
    for stress in STRESS_GRID:
        stress_rows = [row for row in detail_rows if math.isclose(float(row["stress"]), stress)]
        summary_rows.append(aggregate_stress_status(stress_rows))

    final_label, interpretation = final_step11_label(summary_rows, detail_rows, max_horizon)
    write_csv(detail_rows, DETAIL_CSV, DETAIL_FIELDNAMES)
    write_csv(summary_rows, SUMMARY_CSV, SUMMARY_FIELDNAMES)
    write_note(profile, horizons, detail_rows, summary_rows, final_label, interpretation)
    print(SUMMARY_MD.read_text(encoding="utf-8"))
    return final_label, interpretation


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused", "full"], default="focused")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
