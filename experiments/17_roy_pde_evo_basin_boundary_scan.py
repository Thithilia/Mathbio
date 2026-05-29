"""Step 15: quantify PDE-evolution basin boundaries.

This experiment scans initial defense frequency q0 and predator abundance scale
w0_scale inside the bistable stress interval identified in Step 13. It does
not change the model equations and does not run a broad parameter scan.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
FIGURES_DIR = ROOT / "figures" / "roy_evo_spatial"
NOTES_DIR = ROOT / "research_notes"
SCAN_CSV = RESULTS_DIR / "roy_pde_evo_basin_boundary_scan.csv"
HEATMAP_PATH = FIGURES_DIR / "17_basin_boundary_heatmap.png"
NOTE_PATH = NOTES_DIR / "roy_pde_evo_basin_boundary_scan.md"

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
EPSILON = 1.0e-4
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 1.0e-4
EXTREME_EXTINCTION_W = 1.0e-8
TAIL_FRACTION = 0.25

FOCUSED_STRESSES = (0.1584375, 0.16486816)
FULL_STRESSES = (0.1584375, 0.16486816, 0.175)
Q0_VALUES = tuple(round(0.1 * index, 1) for index in range(10))
W0_SCALES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5)

BASIN_LABELS = (
    "persistent_basin",
    "extinct_basin",
    "transient_basin",
    "unresolved_basin",
    "nonphysical_initial_condition",
)
BASIN_COLORS = {
    "persistent_basin": "#1f77b4",
    "extinct_basin": "#ff7f0e",
    "transient_basin": "#2ca02c",
    "unresolved_basin": "#d62728",
    "nonphysical_initial_condition": "#9467bd",
}

FIELDNAMES = [
    "run_id",
    "stress",
    "q0",
    "w0_scale",
    "n0",
    "w0",
    "z0",
    "T",
    "classification",
    "basin_label",
    "tail_mean_w",
    "tail_mean_q",
    "q_change",
    "previous_window_mean_w",
    "latest_window_mean_w",
    "relative_change_between_last_windows",
    "rhs_norm",
    "state_norm",
    "normalized_residual",
    "physical",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
    "q_clip_count",
    "q_clip_max_violation",
    "notes",
]


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


def profile_settings(profile: str) -> tuple[tuple[float, ...], float]:
    if profile == "full":
        return FULL_STRESSES, 2000.0
    return FOCUSED_STRESSES, 1600.0


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def pde_evo_rhs_residual(
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    stress: float,
) -> dict[str, float]:
    _x, _y, dx, dy = grid_2d_evo(config)
    reactions = reaction_part_evo_pde(n, w, q, params, stress=stress, evolve=True)
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


def result_metrics(result: RoyEvoPDEResult, config: RoyEvoPDEConfig, stress: float) -> dict[str, object]:
    metrics: dict[str, object] = {
        **tail_diagnostics(result),
        **pde_evo_rhs_residual(result.n, result.w, result.q, PARAMS, config, stress=stress),
        "min_n": float(result.diagnostics.get("min_n", np.min(result.n))),
        "min_w": float(result.diagnostics.get("min_w", np.min(result.w))),
        "min_q": float(result.diagnostics.get("min_q", np.min(result.q))),
        "max_q": float(result.diagnostics.get("max_q", np.max(result.q))),
        "min_z": float(result.diagnostics.get("min_z", np.min(free_space_evo(result.n, result.w, PARAMS)))),
        "q_clip_count": int(result.diagnostics.get("q_clip_count", 0)),
        "q_clip_max_violation": float(result.diagnostics.get("q_clip_max_violation", 0.0)),
    }
    metrics["classification"] = classify_asymptotic(metrics)
    metrics["basin_label"] = basin_label_from_classification(str(metrics["classification"]))
    return metrics


def stress_regime_from_counts(
    persistent_count: int,
    extinct_count: int,
    transient_count: int,
    unresolved_count: int,
    nonphysical_count: int,
) -> str:
    total = persistent_count + extinct_count + transient_count + unresolved_count + nonphysical_count
    if total == 0:
        return "mostly_transient_or_unresolved"
    if persistent_count > 0 and extinct_count > 0:
        return "bistable_persistent_extinct"
    if persistent_count > 0 and extinct_count == 0 and transient_count == 0 and unresolved_count == 0:
        return "persistent_only"
    if extinct_count > 0 and persistent_count == 0 and transient_count == 0 and unresolved_count == 0:
        return "extinct_only"
    if persistent_count > 0 and extinct_count == 0 and transient_count + unresolved_count > 0:
        return "persistent_transient_mixed"
    if extinct_count > 0 and persistent_count == 0 and transient_count + unresolved_count > 0:
        return "extinct_transient_mixed"
    if transient_count + unresolved_count > total / 2:
        return "mostly_transient_or_unresolved"
    if nonphysical_count > total / 2:
        return "mostly_nonphysical"
    return "mostly_transient_or_unresolved"


def final_step15_label(regime_by_stress: dict[float, str]) -> tuple[str, str]:
    bistable_count = sum(regime == "bistable_persistent_extinct" for regime in regime_by_stress.values())
    clear_count = sum(regime in {"persistent_only", "extinct_only", "bistable_persistent_extinct"} for regime in regime_by_stress.values())
    if bistable_count >= 2 or clear_count >= 2:
        return (
            "basin_boundary_mapped",
            "The q0-w0 grid resolves interpretable basin regions for at least two target stresses.",
        )
    if bistable_count == 1 or clear_count == 1:
        return (
            "basin_boundary_partially_mapped",
            "At least one target stress has an interpretable basin boundary, while other stresses remain mixed or unresolved.",
        )
    return (
        "basin_boundary_unresolved",
        "Most focused outcomes remain transient, unresolved, or nonphysical at the current horizon.",
    )


def row_note(classification: str) -> str:
    if classification.endswith("_steady"):
        return "steady_by_tail_change_and_rhs_residual"
    if classification in {"declining_transient", "recovery_transient"}:
        return "window_change_indicates_transient"
    return classification


def baseline_values() -> tuple[float, float, float]:
    equilibrium = find_evo_equilibrium(PARAMS)
    return float(equilibrium["n"]), float(equilibrium["w"]), float(equilibrium["q"])


def initial_state_for_scan(
    n0: float,
    w0: float,
    q0: float,
    config: RoyEvoPDEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return initial_state_from_ode_equilibrium({"n": n0, "w": w0, "q": q0}, config)


def physical_initial_means(n0: float, w0: float, q0: float) -> tuple[bool, float]:
    z0 = float(free_space_evo(n0, w0, PARAMS))
    physical = bool(np.isfinite([n0, w0, q0, z0]).all() and n0 >= 0.0 and w0 >= 0.0 and 0.0 <= q0 <= 1.0 and z0 >= 0.0)
    return physical, z0


def skipped_row(stress: float, q0: float, w0_scale: float, n0: float, w0: float, z0: float, T: float, reason: str) -> dict[str, object]:
    return {
        "run_id": run_id(stress, q0, w0_scale),
        "stress": float(stress),
        "q0": float(q0),
        "w0_scale": float(w0_scale),
        "n0": float(n0),
        "w0": float(w0),
        "z0": float(z0),
        "T": float(T),
        "classification": "nonphysical",
        "basin_label": "nonphysical_initial_condition",
        "tail_mean_w": float("nan"),
        "tail_mean_q": float("nan"),
        "q_change": float("nan"),
        "previous_window_mean_w": float("nan"),
        "latest_window_mean_w": float("nan"),
        "relative_change_between_last_windows": float("nan"),
        "rhs_norm": float("nan"),
        "state_norm": float("nan"),
        "normalized_residual": float("nan"),
        "physical": False,
        "min_n": float("nan"),
        "min_w": float("nan"),
        "min_q": float("nan"),
        "max_q": float("nan"),
        "min_z": float("nan"),
        "q_clip_count": 0,
        "q_clip_max_violation": 0.0,
        "notes": reason,
    }


def run_id(stress: float, q0: float, w0_scale: float) -> str:
    return f"s{stress:.9g}_q{q0:.1f}_wscale{w0_scale:.3g}".replace(".", "p")


def scan_row(
    stress: float,
    q0: float,
    w0_scale: float,
    n0: float,
    w0: float,
    z0: float,
    T: float,
    result: RoyEvoPDEResult,
    config: RoyEvoPDEConfig,
) -> dict[str, object]:
    metrics = result_metrics(result, config, stress)
    classification = str(metrics["classification"])
    return {
        "run_id": run_id(stress, q0, w0_scale),
        "stress": float(stress),
        "q0": float(q0),
        "w0_scale": float(w0_scale),
        "n0": float(n0),
        "w0": float(w0),
        "z0": float(z0),
        "T": float(T),
        "classification": classification,
        "basin_label": str(metrics["basin_label"]),
        "tail_mean_w": float(metrics["tail_mean_w"]),
        "tail_mean_q": float(metrics["tail_mean_q"]),
        "q_change": float(metrics["q_change"]),
        "previous_window_mean_w": float(metrics["previous_window_mean_w"]),
        "latest_window_mean_w": float(metrics["latest_window_mean_w"]),
        "relative_change_between_last_windows": float(metrics["relative_change_between_last_windows"]),
        "rhs_norm": float(metrics["rhs_norm"]),
        "state_norm": float(metrics["state_norm"]),
        "normalized_residual": float(metrics["normalized_residual"]),
        "physical": bool(metrics["physical"]),
        "min_n": float(metrics["min_n"]),
        "min_w": float(metrics["min_w"]),
        "min_q": float(metrics["min_q"]),
        "max_q": float(metrics["max_q"]),
        "min_z": float(metrics["min_z"]),
        "q_clip_count": int(metrics["q_clip_count"]),
        "q_clip_max_violation": float(metrics["q_clip_max_violation"]),
        "notes": row_note(classification),
    }


def run_scan(profile: str) -> list[dict[str, object]]:
    stresses, T = profile_settings(profile)
    config = pde_config(T)
    n_base, w_base, _q_base = baseline_values()
    rows: list[dict[str, object]] = []
    total_runs = len(stresses) * len(Q0_VALUES) * len(W0_SCALES)
    run_index = 0
    for stress in stresses:
        for q0 in Q0_VALUES:
            for w0_scale in W0_SCALES:
                run_index += 1
                n0 = n_base
                w0 = w_base * w0_scale
                is_physical, z0 = physical_initial_means(n0, w0, q0)
                if not is_physical:
                    rows.append(skipped_row(stress, q0, w0_scale, n0, w0, z0, T, "skipped_nonphysical_initial_condition"))
                    continue
                state = initial_state_for_scan(n0, w0, q0, config)
                if not physical_arrays(state):
                    rows.append(skipped_row(stress, q0, w0_scale, n0, w0, z0, T, "skipped_nonphysical_perturbed_initial_condition"))
                    continue
                print(
                    f"SCAN {run_index:03d}/{total_runs:03d} stress={stress:.9g} "
                    f"q0={q0:.1f} w0_scale={w0_scale:.3g} T={T:.0f}"
                )
                result = simulate_pde_evo_2d(PARAMS, config, state, stress=stress, evolve=True)
                rows.append(scan_row(stress, q0, w0_scale, n0, w0, z0, T, result, config))
    return rows


def count_basin_labels(rows: list[dict[str, object]], stress: float) -> dict[str, int]:
    labels = [str(row["basin_label"]) for row in rows if math.isclose(float(row["stress"]), stress)]
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in BASIN_LABELS}


def stress_regime_summary(rows: list[dict[str, object]], stresses: Iterable[float]) -> dict[float, dict[str, object]]:
    summary: dict[float, dict[str, object]] = {}
    for stress in stresses:
        counts = count_basin_labels(rows, stress)
        regime = stress_regime_from_counts(
            counts["persistent_basin"],
            counts["extinct_basin"],
            counts["transient_basin"],
            counts["unresolved_basin"],
            counts["nonphysical_initial_condition"],
        )
        summary[float(stress)] = {**counts, "regime_label": regime}
    return summary


def plot_heatmap(rows: list[dict[str, object]], stresses: tuple[float, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    category_to_index = {label: index for index, label in enumerate(BASIN_LABELS)}
    colors = [BASIN_COLORS[label] for label in BASIN_LABELS]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(BASIN_LABELS) + 1) - 0.5, len(BASIN_LABELS))

    fig, axes = plt.subplots(1, len(stresses), figsize=(5.2 * len(stresses), 5.6), sharey=True, constrained_layout=True)
    if len(stresses) == 1:
        axes = [axes]

    q_index = {q0: idx for idx, q0 in enumerate(Q0_VALUES)}
    w_index = {w0_scale: idx for idx, w0_scale in enumerate(W0_SCALES)}
    for ax, stress in zip(axes, stresses):
        matrix = np.full((len(W0_SCALES), len(Q0_VALUES)), np.nan)
        for row in rows:
            if not math.isclose(float(row["stress"]), stress):
                continue
            q0 = round(float(row["q0"]), 1)
            w0_scale = float(row["w0_scale"])
            matrix[w_index[w0_scale], q_index[q0]] = category_to_index[str(row["basin_label"])]
        ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"stress = {stress:.9g}")
        ax.set_xlabel("initial defense frequency q0")
        ax.set_xticks(range(len(Q0_VALUES)))
        ax.set_xticklabels([f"{q0:.1f}" for q0 in Q0_VALUES], rotation=45, ha="right")
        ax.set_yticks(range(len(W0_SCALES)))
        ax.set_yticklabels([f"{scale:g}" for scale in W0_SCALES])
        ax.set_ylabel("initial predator scale w0_scale")
        ax.set_xticks(np.arange(-0.5, len(Q0_VALUES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(W0_SCALES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    handles = [mpatches.Patch(color=BASIN_COLORS[label], label=label.replace("_", " ")) for label in BASIN_LABELS]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("PDE-evolution basin boundary scan")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def boundary_interpretation(summary: dict[float, dict[str, object]], rows: list[dict[str, object]]) -> str:
    bistable_stresses = [
        stress for stress, values in sorted(summary.items()) if values["regime_label"] == "bistable_persistent_extinct"
    ]
    if not bistable_stresses:
        return "The focused scan did not resolve a persistent/extinct boundary at the current horizon."

    low_q_pattern = []
    high_w_pattern = []
    for stress in bistable_stresses:
        stress_rows = [row for row in rows if math.isclose(float(row["stress"]), stress)]
        persistent_q = [float(row["q0"]) for row in stress_rows if row["basin_label"] == "persistent_basin"]
        extinct_q = [float(row["q0"]) for row in stress_rows if row["basin_label"] == "extinct_basin"]
        persistent_w = [float(row["w0_scale"]) for row in stress_rows if row["basin_label"] == "persistent_basin"]
        extinct_w = [float(row["w0_scale"]) for row in stress_rows if row["basin_label"] == "extinct_basin"]
        if persistent_q and extinct_q and np.mean(persistent_q) < np.mean(extinct_q):
            low_q_pattern.append(stress)
        if persistent_w and extinct_w and np.mean(persistent_w) > np.mean(extinct_w):
            high_w_pattern.append(stress)
    if low_q_pattern and high_w_pattern:
        return "The basin boundary is two-dimensional in defense-predator initial state: persistent basins concentrate at low-to-intermediate q0, and larger predator initial abundance helps persistence within that range."
    if low_q_pattern:
        return "Initial defense frequency controls basin entry: lower q0 tends to favor predator persistence in the mapped bistable stresses."
    if high_w_pattern:
        return "Predator initial abundance controls basin entry: larger w0_scale tends to favor predator persistence in the mapped bistable stresses."
    return "The scan resolves bistable outcomes, but the boundary is not monotone in q0 or w0_scale at this grid resolution."


def write_note(
    profile: str,
    stresses: tuple[float, ...],
    T: float,
    rows: list[dict[str, object]],
    summary: dict[float, dict[str, object]],
    final_label: str,
    final_interpretation: str,
) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    table_lines = [
        "| stress | persistent | extinct | transient | unresolved | nonphysical | regime label |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stress, values in sorted(summary.items()):
        table_lines.append(
            f"| {stress:.9g} | {values['persistent_basin']} | {values['extinct_basin']} | "
            f"{values['transient_basin']} | {values['unresolved_basin']} | "
            f"{values['nonphysical_initial_condition']} | `{values['regime_label']}` |"
        )

    note = [
        "# Research Note: Basin Boundary Scan for the PDE-Evolution Model",
        "",
        "## Executive Summary",
        "",
        f"Final Step 15 label: `{final_label}`.",
        "",
        final_interpretation,
        "",
        "## Why This Was Needed",
        "",
        "PR #7 mapped bistability qualitatively: at least one stress admitted both predator-persistent and predator-extinct reachable outcomes. Step 15 turns that result into a focused q0-w0 basin-boundary map inside the bistable stress interval.",
        "",
        "This step does not change model equations, does not run a broad parameter scan, and does not diagnose the spatial covariance mechanism.",
        "",
        "## Setup",
        "",
        f"- profile run: `{profile}`",
        "- model: PDE-evo only",
        "- parameters: `RoyEvoParams(b_u=0.08, b_v=0.02)`",
        "- grid: `64x64`, `L_x=L_y=20`, seed `20260702`",
        "- diffusion: `D_n=0.01`, `D_w=0.01`, `D_q=0.005`",
        "- integration: `dt=0.1`, `record_every=50`",
        f"- horizon: `T={T:.0f}`",
        "- perturbation amplitude: `1e-5`",
        f"- target stresses: `{', '.join(f'{stress:.9g}' for stress in stresses)}`",
        f"- steady criteria: `abs(relative_change) < {STEADY_REL_CHANGE_TOL}` and `normalized_residual < {STEADY_RESIDUAL_TOL}`",
        "",
        "## Initial-Condition Grid",
        "",
        "Each run starts from the Step 09A burn-in prey density and scales predator abundance while setting defense frequency:",
        "",
        "```text",
        "n0 = n_baseline",
        "w0 = w_baseline * w0_scale",
        "q0 = q0_value",
        "```",
        "",
        f"`q0` values: `{', '.join(f'{value:.1f}' for value in Q0_VALUES)}`",
        "",
        f"`w0_scale` values: `{', '.join(f'{value:g}' for value in W0_SCALES)}`",
        "",
        "Physicality is checked before simulation. Nonphysical initial states are recorded as `nonphysical_initial_condition` and skipped.",
        "",
        "## Basin Boundary Results",
        "",
        f"Figure: `figures/roy_evo_spatial/17_basin_boundary_heatmap.png`",
        "",
        "![PDE-evolution basin boundary scan](../figures/roy_evo_spatial/17_basin_boundary_heatmap.png)",
        "",
        boundary_interpretation(summary, rows),
        "",
        "## Stress-Level Regime Summary",
        "",
    ]
    note.extend(table_lines)
    note.extend(
        [
            "",
            "## Interpretation",
            "",
            "The scan directly varies initial defense frequency and predator abundance. Bistable stress labels indicate that predator-persistent and predator-extinct basins are both reached on the same q0-w0 grid.",
            "",
            "Transient outcomes remain important: they mark grid points where the current horizon and steady-state residual criteria do not yet justify assigning a persistent or extinct basin.",
            "",
            "## Files",
            "",
            "- `results/roy_pde_evo_basin_boundary_scan.csv`",
            "- `figures/roy_evo_spatial/17_basin_boundary_heatmap.png`",
            "- `experiments/17_roy_pde_evo_basin_boundary_scan.py`",
            "",
            "## Next Step",
            "",
            "Refine the mapped basin boundary with targeted continuation in q0-w0 space before returning to spatial covariance mechanism interpretation.",
            "",
            final_label,
        ]
    )
    NOTE_PATH.write_text("\n".join(note) + "\n", encoding="utf-8")


def run(profile: str = "focused") -> tuple[str, str]:
    stresses, T = profile_settings(profile)
    rows = run_scan(profile)
    summary = stress_regime_summary(rows, stresses)
    regimes = {stress: str(values["regime_label"]) for stress, values in summary.items()}
    final_label, final_interpretation = final_step15_label(regimes)

    write_csv(rows, SCAN_CSV)
    plot_heatmap(rows, stresses, HEATMAP_PATH)
    write_note(profile, stresses, T, rows, summary, final_label, final_interpretation)
    print(NOTE_PATH.read_text(encoding="utf-8"))
    print(HEATMAP_PATH)
    return final_label, final_interpretation


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused", "full"], default="focused")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
