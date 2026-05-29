#!/usr/bin/env python
"""Save and visualize representative PDE solutions for the Roy evo-spatial model.

This is a targeted verification run, not a parameter scan. It selects one
persistent, one extinct, and one transient representative from the existing
basin-boundary scan, reruns those cases with the established PDE settings, and
saves spatial fields, mean time series, residual diagnostics, and figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (  # noqa: E402
    RoyEvoPDEConfig,
    RoyEvoParams,
    find_evo_equilibrium,
    free_space_evo,
    grid_2d_evo,
    initial_state_from_ode_equilibrium,
    laplacian_neumann_2d_evo,
    reaction_part_evo_pde,
)


PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 1.0e-4
TAIL_FRACTION = 0.25
PERTURBATION_AMPLITUDE = 1.0e-5

SCAN_CSV = ROOT / "results" / "roy_pde_evo_basin_boundary_scan.csv"
SUMMARY_CSV = ROOT / "results" / "roy_pde_evo_representative_solution_summary.csv"
TIMESERIES_CSV = ROOT / "results" / "roy_pde_evo_representative_mean_timeseries.csv"
NOTE_PATH = ROOT / "research_notes" / "roy_pde_evo_representative_solutions.md"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
RESULTS_DIR = ROOT / "results"

FIELD_FILES = {
    "persistent_case": RESULTS_DIR / "roy_pde_evo_representative_fields_persistent_case.npz",
    "extinct_case": RESULTS_DIR / "roy_pde_evo_representative_fields_extinct_case.npz",
    "transient_case": RESULTS_DIR / "roy_pde_evo_representative_fields_transient_case.npz",
}

CASE_ORDER = ("persistent_case", "extinct_case", "transient_case")

BASIN_ORDER = [
    "persistent_basin",
    "extinct_basin",
    "transient_basin",
    "unresolved_basin",
    "nonphysical_initial_condition",
]

SUMMARY_FIELDS = [
    "case_label",
    "stress",
    "q0",
    "w0_scale",
    "T",
    "classification_from_prior_scan",
    "classification_from_rerun",
    "basin_label_from_prior_scan",
    "basin_label_from_rerun",
    "tail_mean_w",
    "tail_min_w",
    "tail_slope_w",
    "tail_mean_q",
    "q_change",
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
    "physical",
    "q_clip_count",
    "q_clip_max_violation",
    "notes",
]

TIMESERIES_FIELDS = [
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
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
]


@dataclass
class SnapshotPDEResult:
    """Container for a targeted PDE run with snapshots."""

    times: np.ndarray
    mean_n: np.ndarray
    mean_w: np.ndarray
    mean_q: np.ndarray
    var_n: np.ndarray
    var_w: np.ndarray
    var_q: np.ndarray
    min_n_series: np.ndarray
    min_w_series: np.ndarray
    min_q_series: np.ndarray
    max_q_series: np.ndarray
    min_z_series: np.ndarray
    n_final: np.ndarray
    w_final: np.ndarray
    q_final: np.ndarray
    n_snapshots: np.ndarray
    w_snapshots: np.ndarray
    q_snapshots: np.ndarray
    snapshot_times: np.ndarray
    min_n: float
    min_w: float
    min_q: float
    max_q: float
    min_z: float
    q_clip_count: int
    q_clip_max_violation: float
    completed: bool
    nonfinite_detected: bool


def as_float(value: Any, default: float = math.nan) -> float:
    """Convert CSV values to float with a predictable fallback."""

    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def close_to(value: Any, target: float, tol: float = 1.0e-10) -> bool:
    return abs(as_float(value) - target) <= tol


def read_scan_rows(path: Path = SCAN_CSV) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first_or_none(rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        return row
    return None


def select_representative_cases(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Select one persistent, one extinct, and one transient representative row.

    The selection is deterministic and biased toward the scientific preferences
    specified in the verification task. Fallbacks are used only when a preferred
    row is absent from the existing scan CSV.
    """

    persistent_candidates = [
        row
        for row in rows
        if row.get("basin_label") == "persistent_basin"
        and row.get("classification") == "persistent_steady"
        and close_to(row.get("stress"), 0.1584375)
    ]
    if not persistent_candidates:
        persistent_candidates = [
            row
            for row in rows
            if row.get("basin_label") == "persistent_basin"
            and row.get("classification") == "persistent_steady"
        ]
    persistent_candidates.sort(
        key=lambda row: (
            0 if round(as_float(row.get("q0")), 10) in {0.1, 0.2} else 1,
            abs(as_float(row.get("q0")) - 0.1),
            abs(as_float(row.get("w0_scale")) - 0.1),
            as_float(row.get("stress")),
        )
    )

    extinct_candidates = [
        row
        for row in rows
        if row.get("basin_label") == "extinct_basin"
        and row.get("classification") == "extinct_steady"
        and close_to(row.get("stress"), 0.16486816)
    ]
    if not extinct_candidates:
        extinct_candidates = [
            row
            for row in rows
            if row.get("basin_label") == "extinct_basin"
            and row.get("classification") == "extinct_steady"
        ]
    extinct_candidates.sort(
        key=lambda row: (
            -as_float(row.get("q0")),
            as_float(row.get("w0_scale")),
            abs(as_float(row.get("stress")) - 0.16486816),
        )
    )

    transient_candidates = [
        row
        for row in rows
        if row.get("basin_label") == "transient_basin"
        and row.get("classification") in {"declining_transient", "recovery_transient"}
    ]
    if not transient_candidates:
        transient_candidates = [
            row for row in rows if row.get("basin_label") == "transient_basin"
        ]
    transient_candidates.sort(
        key=lambda row: (
            -abs(as_float(row.get("relative_change_between_last_windows"), 0.0)),
            abs(as_float(row.get("stress")) - 0.1584375),
        )
    )

    selected = {
        "persistent_case": _first_or_none(persistent_candidates),
        "extinct_case": _first_or_none(extinct_candidates),
        "transient_case": _first_or_none(transient_candidates),
    }
    missing = [label for label, row in selected.items() if row is None]
    if missing:
        raise ValueError(f"Could not select representative cases: {missing}")

    return {label: dict(row, case_label=label) for label, row in selected.items() if row is not None}


def snapshot_times_for_horizon(T: float) -> np.ndarray:
    """Return the required snapshot times including initial and final states."""

    return np.array([0.0, 0.25 * T, 0.50 * T, 0.75 * T, T], dtype=float)


def pde_config(T: float) -> RoyEvoPDEConfig:
    return RoyEvoPDEConfig(
        n_x=64,
        n_y=64,
        L_x=20.0,
        L_y=20.0,
        D_n=0.01,
        D_w=0.01,
        D_q=0.005,
        dt=0.1,
        T=float(T),
        record_every=50,
        seed=20260702,
        perturbation_amplitude=PERTURBATION_AMPLITUDE,
    )


def baseline_values() -> tuple[float, float, float]:
    eq = find_evo_equilibrium(PARAMS)
    return float(eq["n"]), float(eq["w"]), float(eq["q"])


def initial_state_for_case(
    *,
    q0: float,
    w0_scale: float,
    config: RoyEvoPDEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_baseline, w_baseline, _q_baseline = baseline_values()
    n0 = n_baseline
    w0 = w_baseline * w0_scale
    z0 = 1.0 / PARAMS.kappa - n0 - w0
    if n0 < 0.0 or w0 < 0.0 or not (0.0 <= q0 <= 1.0) or z0 < 0.0:
        raise ValueError(
            "Nonphysical representative initial condition: "
            f"n0={n0}, w0={w0}, q0={q0}, z0={z0}"
        )
    n, w, q = initial_state_from_ode_equilibrium(
        {"n": n0, "w": w0, "q": q0},
        config,
    )
    return n, w, q, {"n0": n0, "w0": w0, "q0": q0, "z0": z0}


def _record_series(
    result: dict[str, list[float]],
    *,
    time: float,
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
) -> None:
    z = free_space_evo(n, w, PARAMS)
    result["time"].append(float(time))
    result["mean_n"].append(float(np.mean(n)))
    result["mean_w"].append(float(np.mean(w)))
    result["mean_q"].append(float(np.mean(q)))
    result["var_n"].append(float(np.var(n)))
    result["var_w"].append(float(np.var(w)))
    result["var_q"].append(float(np.var(q)))
    result["min_n_series"].append(float(np.min(n)))
    result["min_w_series"].append(float(np.min(w)))
    result["min_q_series"].append(float(np.min(q)))
    result["max_q_series"].append(float(np.max(q)))
    result["min_z_series"].append(float(np.min(z)))


def simulate_pde_evo_with_snapshots(
    *,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    stress: float,
    snapshot_times: np.ndarray,
) -> SnapshotPDEResult:
    """Run the established explicit PDE scheme while saving snapshots."""

    n, w, q = (np.array(field, dtype=float, copy=True) for field in initial_state)
    _x, _y, dx, dy = grid_2d_evo(config)
    n_steps = int(math.ceil(config.T / config.dt))
    record_stride = max(1, int(config.record_every))
    snapshot_steps = {
        int(round(float(snapshot_time) / config.dt)): idx
        for idx, snapshot_time in enumerate(snapshot_times)
    }

    n_snapshots = np.empty((len(snapshot_times), config.n_y, config.n_x), dtype=float)
    w_snapshots = np.empty_like(n_snapshots)
    q_snapshots = np.empty_like(n_snapshots)

    series: dict[str, list[float]] = {
        "time": [],
        "mean_n": [],
        "mean_w": [],
        "mean_q": [],
        "var_n": [],
        "var_w": [],
        "var_q": [],
        "min_n_series": [],
        "min_w_series": [],
        "min_q_series": [],
        "max_q_series": [],
        "min_z_series": [],
    }

    min_n = float(np.min(n))
    min_w = float(np.min(w))
    min_q = float(np.min(q))
    max_q = float(np.max(q))
    min_z = float(np.min(free_space_evo(n, w, params)))
    q_clip_count = 0
    q_clip_max_violation = 0.0
    nonfinite_detected = False
    completed = True

    def capture(step: int) -> None:
        if step in snapshot_steps:
            idx = snapshot_steps[step]
            n_snapshots[idx] = n
            w_snapshots[idx] = w
            q_snapshots[idx] = q

    capture(0)
    _record_series(series, time=0.0, n=n, w=w, q=q)

    for step in range(1, n_steps + 1):
        dn_reaction, dw_reaction, dq_reaction = reaction_part_evo_pde(
            n,
            w,
            q,
            params,
            stress,
            evolve=True,
        )
        n = n + config.dt * (
            config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + dn_reaction
        )
        w = w + config.dt * (
            config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + dw_reaction
        )
        q_raw = q + config.dt * (
            config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + dq_reaction
        )

        lower_violation = np.maximum(0.0, -q_raw)
        upper_violation = np.maximum(0.0, q_raw - 1.0)
        violation = np.maximum(lower_violation, upper_violation)
        q_clip_count += int(np.count_nonzero(violation > 0.0))
        if violation.size:
            q_clip_max_violation = max(q_clip_max_violation, float(np.max(violation)))
        q = np.clip(q_raw, 0.0, 1.0)

        z = free_space_evo(n, w, params)
        min_n = min(min_n, float(np.min(n)))
        min_w = min(min_w, float(np.min(w)))
        min_q = min(min_q, float(np.min(q)))
        max_q = max(max_q, float(np.max(q)))
        min_z = min(min_z, float(np.min(z)))

        if not (
            np.all(np.isfinite(n))
            and np.all(np.isfinite(w))
            and np.all(np.isfinite(q))
            and np.all(np.isfinite(z))
        ):
            nonfinite_detected = True
            completed = False
            break

        if step % record_stride == 0 or step == n_steps or step in snapshot_steps:
            _record_series(series, time=min(step * config.dt, config.T), n=n, w=w, q=q)
        capture(step)

    if n_steps not in snapshot_steps:
        n_snapshots[-1] = n
        w_snapshots[-1] = w
        q_snapshots[-1] = q

    return SnapshotPDEResult(
        times=np.array(series["time"], dtype=float),
        mean_n=np.array(series["mean_n"], dtype=float),
        mean_w=np.array(series["mean_w"], dtype=float),
        mean_q=np.array(series["mean_q"], dtype=float),
        var_n=np.array(series["var_n"], dtype=float),
        var_w=np.array(series["var_w"], dtype=float),
        var_q=np.array(series["var_q"], dtype=float),
        min_n_series=np.array(series["min_n_series"], dtype=float),
        min_w_series=np.array(series["min_w_series"], dtype=float),
        min_q_series=np.array(series["min_q_series"], dtype=float),
        max_q_series=np.array(series["max_q_series"], dtype=float),
        min_z_series=np.array(series["min_z_series"], dtype=float),
        n_final=n,
        w_final=w,
        q_final=q,
        n_snapshots=n_snapshots,
        w_snapshots=w_snapshots,
        q_snapshots=q_snapshots,
        snapshot_times=np.array(snapshot_times, dtype=float),
        min_n=min_n,
        min_w=min_w,
        min_q=min_q,
        max_q=max_q,
        min_z=min_z,
        q_clip_count=q_clip_count,
        q_clip_max_violation=q_clip_max_violation,
        completed=completed,
        nonfinite_detected=nonfinite_detected,
    )


def tail_mask(times: np.ndarray, fraction: float = TAIL_FRACTION) -> np.ndarray:
    start = times[-1] - fraction * (times[-1] - times[0])
    mask = times >= start
    if np.count_nonzero(mask) < 2:
        mask[-2:] = True
    return mask


def previous_window_mask(times: np.ndarray, fraction: float = TAIL_FRACTION) -> np.ndarray:
    span = times[-1] - times[0]
    latest_start = times[-1] - fraction * span
    previous_start = times[-1] - 2.0 * fraction * span
    mask = (times >= previous_start) & (times < latest_start)
    if np.count_nonzero(mask) < 2:
        midpoint = max(1, len(times) // 2)
        mask = np.zeros_like(times, dtype=bool)
        mask[max(0, midpoint - 2) : midpoint] = True
    return mask


def tail_slope(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    x = times[mask]
    y = values[mask]
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, deg=1)[0])


def relative_change(previous: float, latest: float) -> float:
    return float((latest - previous) / max(abs(previous), 1.0e-12))


def physical_arrays(n: np.ndarray, w: np.ndarray, q: np.ndarray, params: RoyEvoParams) -> bool:
    z = free_space_evo(n, w, params)
    return bool(
        np.all(np.isfinite(n))
        and np.all(np.isfinite(w))
        and np.all(np.isfinite(q))
        and np.all(np.isfinite(z))
        and np.min(n) >= -1.0e-8
        and np.min(w) >= -1.0e-8
        and np.min(z) >= -1.0e-8
        and np.min(q) >= -1.0e-8
        and np.max(q) <= 1.0 + 1.0e-8
    )


def pde_evo_rhs_residual(
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    stress: float,
) -> dict[str, float]:
    """Compute the PDE RHS residual at a state using the established RHS."""

    _x, _y, dx, dy = grid_2d_evo(config)
    dn_reaction, dw_reaction, dq_reaction = reaction_part_evo_pde(
        n,
        w,
        q,
        params,
        stress,
        evolve=True,
    )
    dn_dt = config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + dn_reaction
    dw_dt = config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + dw_reaction
    dq_dt = config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + dq_reaction
    rhs_norm = float(
        np.sqrt(np.mean(dn_dt**2) + np.mean(dw_dt**2) + np.mean(dq_dt**2))
    )
    state_norm = float(np.sqrt(np.mean(n**2) + np.mean(w**2) + np.mean(q**2)))
    return {
        "rhs_norm": rhs_norm,
        "state_norm": state_norm,
        "normalized_residual": rhs_norm / max(state_norm, 1.0e-12),
        "max_abs_dn_dt": float(np.max(np.abs(dn_dt))),
        "max_abs_dw_dt": float(np.max(np.abs(dw_dt))),
        "max_abs_dq_dt": float(np.max(np.abs(dq_dt))),
    }


def classify_asymptotic_run(
    *,
    physical: bool,
    tail_mean_w: float,
    tail_min_w: float,
    tail_slope_w: float,
    relative_change_between_last_windows: float,
    normalized_residual: float,
    epsilon: float = EPSILON,
) -> str:
    if not physical:
        return "nonphysical"
    if (
        tail_mean_w > epsilon
        and tail_min_w > 0.25 * epsilon
        and abs(relative_change_between_last_windows) < STEADY_REL_CHANGE_TOL
        and normalized_residual < STEADY_RESIDUAL_TOL
    ):
        return "persistent_steady"
    if (
        tail_mean_w < epsilon
        and tail_min_w < epsilon
        and normalized_residual < STEADY_RESIDUAL_TOL
    ):
        return "extinct_steady"
    if tail_mean_w > epsilon and (
        relative_change_between_last_windows < -STEADY_REL_CHANGE_TOL
        or tail_slope_w < -0.25 * epsilon / max(1.0, 100.0)
    ):
        return "declining_transient"
    if tail_mean_w > epsilon and relative_change_between_last_windows > STEADY_REL_CHANGE_TOL:
        return "recovery_transient"
    if tail_mean_w > epsilon:
        return "persistent_transient"
    if tail_mean_w < epsilon:
        return "extinct_transient"
    return "indeterminate"


def basin_label_from_classification(classification: str) -> str:
    if classification == "persistent_steady":
        return "persistent_basin"
    if classification == "extinct_steady":
        return "extinct_basin"
    if classification in {
        "persistent_transient",
        "extinct_transient",
        "recovery_transient",
        "declining_transient",
    }:
        return "transient_basin"
    if classification == "nonphysical":
        return "nonphysical_initial_condition"
    return "unresolved_basin"


def summarize_result(
    *,
    case_label: str,
    selected_row: dict[str, str],
    result: SnapshotPDEResult,
    config: RoyEvoPDEConfig,
    stress: float,
    q0: float,
    w0_scale: float,
) -> dict[str, Any]:
    tail = tail_mask(result.times)
    previous = previous_window_mask(result.times)
    tail_mean_w = float(np.mean(result.mean_w[tail]))
    tail_min_w = float(np.min(result.mean_w[tail]))
    tail_slope_w = tail_slope(result.times, result.mean_w, tail)
    tail_mean_q = float(np.mean(result.mean_q[tail]))
    q_change = float(result.mean_q[-1] - result.mean_q[0])
    previous_window_mean_w = float(np.mean(result.mean_w[previous]))
    latest_window_mean_w = float(np.mean(result.mean_w[tail]))
    rel_change = relative_change(previous_window_mean_w, latest_window_mean_w)
    residual = pde_evo_rhs_residual(
        result.n_final,
        result.w_final,
        result.q_final,
        PARAMS,
        config,
        stress,
    )
    physical = physical_arrays(result.n_final, result.w_final, result.q_final, PARAMS)
    classification = classify_asymptotic_run(
        physical=physical,
        tail_mean_w=tail_mean_w,
        tail_min_w=tail_min_w,
        tail_slope_w=tail_slope_w,
        relative_change_between_last_windows=rel_change,
        normalized_residual=residual["normalized_residual"],
    )
    basin_label = basin_label_from_classification(classification)
    prior_classification = selected_row.get("classification", "")
    prior_basin = selected_row.get("basin_label", "")
    notes = []
    if classification != prior_classification:
        notes.append(f"classification_differs_from_prior:{prior_classification}")
    else:
        notes.append("classification_matches_prior")
    if basin_label != prior_basin:
        notes.append(f"basin_label_differs_from_prior:{prior_basin}")
    else:
        notes.append("basin_label_matches_prior")
    if not result.completed:
        notes.append("run_did_not_complete")
    if result.nonfinite_detected:
        notes.append("nonfinite_detected")

    return {
        "case_label": case_label,
        "stress": stress,
        "q0": q0,
        "w0_scale": w0_scale,
        "T": config.T,
        "classification_from_prior_scan": prior_classification,
        "classification_from_rerun": classification,
        "basin_label_from_prior_scan": prior_basin,
        "basin_label_from_rerun": basin_label,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": tail_slope_w,
        "tail_mean_q": tail_mean_q,
        "q_change": q_change,
        "previous_window_mean_w": previous_window_mean_w,
        "latest_window_mean_w": latest_window_mean_w,
        "relative_change_between_last_windows": rel_change,
        "rhs_norm": residual["rhs_norm"],
        "state_norm": residual["state_norm"],
        "normalized_residual": residual["normalized_residual"],
        "min_n": result.min_n,
        "min_w": result.min_w,
        "min_q": result.min_q,
        "max_q": result.max_q,
        "min_z": result.min_z,
        "physical": physical,
        "q_clip_count": result.q_clip_count,
        "q_clip_max_violation": result.q_clip_max_violation,
        "notes": ";".join(notes),
    }


def timeseries_rows(
    *,
    case_label: str,
    stress: float,
    q0: float,
    w0_scale: float,
    result: SnapshotPDEResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, time in enumerate(result.times):
        rows.append(
            {
                "case_label": case_label,
                "stress": stress,
                "q0": q0,
                "w0_scale": w0_scale,
                "time": float(time),
                "mean_n": float(result.mean_n[idx]),
                "mean_w": float(result.mean_w[idx]),
                "mean_q": float(result.mean_q[idx]),
                "var_n": float(result.var_n[idx]),
                "var_w": float(result.var_w[idx]),
                "var_q": float(result.var_q[idx]),
                "min_n": float(result.min_n_series[idx]),
                "min_w": float(result.min_w_series[idx]),
                "min_q": float(result.min_q_series[idx]),
                "max_q": float(result.max_q_series[idx]),
                "min_z": float(result.min_z_series[idx]),
            }
        )
    return rows


def save_field_archive(
    path: Path,
    *,
    selected_row: dict[str, str],
    summary_row: dict[str, Any],
    result: SnapshotPDEResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_snapshots=result.n_snapshots,
        w_snapshots=result.w_snapshots,
        q_snapshots=result.q_snapshots,
        snapshot_times=result.snapshot_times,
        final_n=result.n_final,
        final_w=result.w_final,
        final_q=result.q_final,
        case_label=summary_row["case_label"],
        stress=summary_row["stress"],
        q0=summary_row["q0"],
        w0_scale=summary_row["w0_scale"],
        classification_from_prior_scan=selected_row.get("classification", ""),
        classification_from_rerun=summary_row["classification_from_rerun"],
        basin_label_from_prior_scan=selected_row.get("basin_label", ""),
        basin_label_from_rerun=summary_row["basin_label_from_rerun"],
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_solution_snapshots(summary_rows: list[dict[str, Any]]) -> None:
    fig_path = FIG_DIR / "fig19_pde_solution_snapshots.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(CASE_ORDER),
        ncols=3,
        figsize=(11.0, 8.5),
        constrained_layout=True,
    )
    fields = [
        ("n_snapshots", "n(x,y,T)"),
        ("w_snapshots", "w(x,y,T)"),
        ("q_snapshots", "q(x,y,T)"),
    ]
    summary_by_case = {row["case_label"]: row for row in summary_rows}
    for row_idx, case_label in enumerate(CASE_ORDER):
        archive = np.load(FIELD_FILES[case_label])
        summary = summary_by_case[case_label]
        for col_idx, (array_key, title) in enumerate(fields):
            ax = axes[row_idx, col_idx]
            image = archive[array_key][-1]
            im = ax.imshow(image, origin="lower", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=11)
            if col_idx == 0:
                label = case_label.replace("_", " ")
                ax.set_ylabel(
                    f"{label}\n"
                    f"s={float(summary['stress']):.8g}, q0={float(summary['q0']):.2f}, "
                    f"w0x={float(summary['w0_scale']):.2g}\n"
                    f"{summary['classification_from_rerun']}",
                    fontsize=9,
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2e")
    fig.suptitle("Representative PDE final fields", fontsize=14)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def plot_mean_timeseries(timeseries_rows_all: list[dict[str, Any]]) -> None:
    fig_path = FIG_DIR / "fig20_pde_mean_timeseries.png"
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.0, 8.0), sharex=True)
    quantity_info = [
        ("mean_w", "Predator mean, wbar(t)"),
        ("mean_q", "Defense mean, qbar(t)"),
        ("mean_n", "Prey mean, nbar(t)"),
    ]
    colors = {
        "persistent_case": "#2ca02c",
        "extinct_case": "#d62728",
        "transient_case": "#ff7f0e",
    }
    for ax, (field, ylabel) in zip(axes, quantity_info, strict=True):
        for case_label in CASE_ORDER:
            rows = [row for row in timeseries_rows_all if row["case_label"] == case_label]
            ax.plot(
                [float(row["time"]) for row in rows],
                [float(row[field]) for row in rows],
                label=case_label.replace("_", " "),
                color=colors[case_label],
                linewidth=1.8,
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Spatial mean dynamics for representative PDE cases", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def plot_residuals(summary_rows: list[dict[str, Any]]) -> None:
    fig_path = FIG_DIR / "fig21_pde_solution_residuals.png"
    labels = [row["case_label"].replace("_", " ") for row in summary_rows]
    residuals = [float(row["normalized_residual"]) for row in summary_rows]
    rel_changes = [abs(float(row["relative_change_between_last_windows"])) for row in summary_rows]
    tail_w = [float(row["tail_mean_w"]) for row in summary_rows]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11.0, 3.6), constrained_layout=True)
    panels = [
        (residuals, "Normalized RHS residual", True),
        (rel_changes, "|Relative change|\nlast windows", True),
        (tail_w, "Tail mean predator", True),
    ]
    for ax, (values, title, log_scale) in zip(axes, panels, strict=True):
        ax.bar(labels, values, color=["#2ca02c", "#d62728", "#ff7f0e"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Residual and convergence diagnostics", fontsize=14)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def write_research_note(summary_rows: list[dict[str, Any]], profile: str) -> None:
    note = [
        "# Representative PDE Solutions for the Roy Eco-Evolutionary Spatial Model",
        "",
        "## Purpose",
        "",
        "This note addresses a missing evidence requirement from the verification plan: actual PDE solution fields, spatial mean time series, and residual diagnostics for representative outcomes. The run is targeted and reuses existing basin-boundary scan rows; it is not a parameter scan and does not change the model equations.",
        "",
        "## Selected Cases",
        "",
        "| case | stress | q0 | w0_scale | prior classification | rerun classification | prior basin | rerun basin |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in summary_rows:
        note.append(
            f"| `{row['case_label']}` | {float(row['stress']):.8g} | "
            f"{float(row['q0']):.3g} | {float(row['w0_scale']):.3g} | "
            f"`{row['classification_from_prior_scan']}` | "
            f"`{row['classification_from_rerun']}` | "
            f"`{row['basin_label_from_prior_scan']}` | "
            f"`{row['basin_label_from_rerun']}` |"
        )
    note.extend(
        [
            "",
            "## Numerical Setup",
            "",
            f"The script was run with profile `{profile}`. The focused profile uses `T = 1600`, `n_x = n_y = 64`, `L_x = L_y = 20`, `D_n = 0.01`, `D_w = 0.01`, `D_q = 0.005`, `dt = 0.1`, seed `20260702`, and `RoyEvoParams(b_u=0.08, b_v=0.02)`. Initial means are constructed from the Step 09A burn-in baseline using the selected `q0` and `w0_scale` values.",
            "",
            "Snapshots were saved at `t = 0`, `0.25T`, `0.50T`, `0.75T`, and `T` for `n(x,y,t)`, `w(x,y,t)`, and `q(x,y,t)`.",
            "",
            "## Spatial Snapshots",
            "",
            "The final spatial fields are shown in `figures/roy_evo_spatial/report/fig19_pde_solution_snapshots.png`. The full snapshot arrays are saved as compressed `.npz` archives in `results/`.",
            "",
            "## Spatial Mean Dynamics",
            "",
            "The spatial mean time series are saved in `results/roy_pde_evo_representative_mean_timeseries.csv` and plotted in `figures/roy_evo_spatial/report/fig20_pde_mean_timeseries.png`.",
            "",
            "## Residual and Convergence Diagnostics",
            "",
            "| case | tail_mean_w | relative_change_last_windows | normalized_residual | note |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        note.append(
            f"| `{row['case_label']}` | {float(row['tail_mean_w']):.6g} | "
            f"{float(row['relative_change_between_last_windows']):.6g} | "
            f"{float(row['normalized_residual']):.6g} | {row['notes']} |"
        )
    transient_rows = [
        row
        for row in summary_rows
        if row["case_label"] == "transient_case"
        and "transient" in str(row["classification_from_rerun"])
    ]
    transient_sentence = (
        "The representative transient case remains classified as transient in the rerun, so it should not be interpreted as an asymptotic steady outcome."
        if transient_rows
        else "The selected transient case did not remain transient under the rerun; this classification difference is documented in the summary CSV."
    )
    note.extend(
        [
            "",
            "For near-extinct trajectories, the relative change between final windows can be numerically large because both windows have predator abundance close to zero. The extinction interpretation therefore uses tail abundance together with the residual check rather than relative change alone.",
            "",
            "## Interpretation",
            "",
            "The representative trajectories support the interpretation that the spatial PDE can realize different outcome classes from different initial states in the tested parameterization. This is direct PDE solution evidence, but it remains case-specific and does not prove general bistability.",
            "",
            transient_sentence,
            "",
            "If a rerun classification or basin label differs from the prior basin-boundary scan, the difference is retained in the summary table rather than hidden.",
            "",
            "## Limitations",
            "",
            "This analysis reruns only three representative cases from the existing basin-boundary scan. It does not test robustness to grid size, time step, diffusion coefficients, trade-off parameters, perturbation seeds, or longer horizons. Transient-heavy regions still require targeted follow-up.",
            "",
            "## Files",
            "",
            "- `experiments/19_roy_pde_evo_representative_solutions.py`",
            "- `results/roy_pde_evo_representative_solution_summary.csv`",
            "- `results/roy_pde_evo_representative_mean_timeseries.csv`",
            "- `results/roy_pde_evo_representative_fields_persistent_case.npz`",
            "- `results/roy_pde_evo_representative_fields_extinct_case.npz`",
            "- `results/roy_pde_evo_representative_fields_transient_case.npz`",
            "- `figures/roy_evo_spatial/report/fig19_pde_solution_snapshots.png`",
            "- `figures/roy_evo_spatial/report/fig20_pde_mean_timeseries.png`",
            "- `figures/roy_evo_spatial/report/fig21_pde_solution_residuals.png`",
            "",
            "## Next Step",
            "",
            "Use these representative fields to guide adaptive q0-w0 basin-boundary refinement and longer-horizon checks for transient grid points.",
            "",
        ]
    )
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(note), encoding="utf-8")


def run(profile: str) -> None:
    if profile not in {"focused", "minimal"}:
        raise ValueError(f"Unknown profile: {profile}")
    T = 1600.0 if profile == "focused" else 400.0
    config = pde_config(T)
    selected = select_representative_cases(read_scan_rows())
    snapshot_times = snapshot_times_for_horizon(config.T)

    summary_rows: list[dict[str, Any]] = []
    all_timeseries_rows: list[dict[str, Any]] = []

    for case_label in CASE_ORDER:
        row = selected[case_label]
        stress = as_float(row["stress"])
        q0 = as_float(row["q0"])
        w0_scale = as_float(row["w0_scale"])
        n0, w0, q0_field, _initial_means = initial_state_for_case(
            q0=q0,
            w0_scale=w0_scale,
            config=config,
        )
        result = simulate_pde_evo_with_snapshots(
            params=PARAMS,
            config=config,
            initial_state=(n0, w0, q0_field),
            stress=stress,
            snapshot_times=snapshot_times,
        )
        summary = summarize_result(
            case_label=case_label,
            selected_row=row,
            result=result,
            config=config,
            stress=stress,
            q0=q0,
            w0_scale=w0_scale,
        )
        save_field_archive(
            FIELD_FILES[case_label],
            selected_row=row,
            summary_row=summary,
            result=result,
        )
        summary_rows.append(summary)
        all_timeseries_rows.extend(
            timeseries_rows(
                case_label=case_label,
                stress=stress,
                q0=q0,
                w0_scale=w0_scale,
                result=result,
            )
        )

    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    write_csv(TIMESERIES_CSV, all_timeseries_rows, TIMESERIES_FIELDS)
    plot_solution_snapshots(summary_rows)
    plot_mean_timeseries(all_timeseries_rows)
    plot_residuals(summary_rows)
    write_research_note(summary_rows, profile)

    print(f"Wrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Wrote {TIMESERIES_CSV.relative_to(ROOT)}")
    for path in FIELD_FILES.values():
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Wrote {(FIG_DIR / 'fig19_pde_solution_snapshots.png').relative_to(ROOT)}")
    print(f"Wrote {(FIG_DIR / 'fig20_pde_mean_timeseries.png').relative_to(ROOT)}")
    print(f"Wrote {(FIG_DIR / 'fig21_pde_solution_residuals.png').relative_to(ROOT)}")
    print(f"Wrote {NOTE_PATH.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("focused", "minimal"),
        default="focused",
        help="focused uses T=1600; minimal is for fast smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
