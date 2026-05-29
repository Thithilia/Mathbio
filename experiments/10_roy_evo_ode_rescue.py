"""Step 09A: ODE eco-evolutionary rescue test.

This script tests the well-mixed Roy-Yamamichi model before any spatial PDE
implementation.  It compares predator mortality thresholds with prey defense
frozen at the unstressed baseline and with defense allowed to evolve.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.roy_evo_spatial import (
    RoyEvoParams,
    a_of_q,
    b_of_q,
    bisection_threshold,
    classify_evo_trajectory,
    find_evo_equilibrium,
    free_space_evo,
    r_of_q,
    reaction_ode_evo,
    selection_gradient,
    simulate_ode_evo,
)


RESULTS_DIR = ROOT / "results"
THRESHOLD_CSV = RESULTS_DIR / "roy_evo_ode_threshold_scan.csv"
TIMESERIES_CSV = RESULTS_DIR / "roy_evo_ode_timeseries.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_09_ode_evo_rescue.md"

EPSILON = 1.0e-4
THRESHOLD_TOLERANCE = 1.0e-5
TRADEOFF_LABEL = "interior_low_conversion_tradeoff"
PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
BURN_IN_INITIAL = np.array([1.0, 0.2, 0.5], dtype=float)
BURN_IN_T = 3000.0
STRESS_T = 1500.0

THRESHOLD_FIELDNAMES = [
    "run_id",
    "tradeoff_label",
    "evolve",
    "stress_low",
    "stress_high",
    "threshold",
    "threshold_gap",
    "threshold_status",
    "initial_n",
    "initial_w",
    "initial_q",
    "tail_mean_w_at_low",
    "tail_mean_w_at_high",
    "q_tail_mean_at_threshold_low",
    "q_change_at_threshold_low",
    "persistent_low",
    "persistent_high",
    "notes",
]

TIMESERIES_FIELDNAMES = [
    "run_id",
    "scenario",
    "evolve",
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
]


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def simulate_and_classify(stress: float, evolve: bool, initial_state: np.ndarray) -> tuple[bool, dict[str, float | bool]]:
    trajectory = simulate_ode_evo(PARAMS, initial_state, stress=stress, evolve=evolve, T=STRESS_T, n_eval=751)
    diagnostics = classify_evo_trajectory(trajectory.t, trajectory.y, epsilon=EPSILON, params=PARAMS)
    physical_success = bool(trajectory.success and diagnostics["physical"])
    persistent = bool(physical_success and diagnostics["persistent_predator"])
    diagnostics = {**diagnostics, "solver_success": bool(trajectory.success)}
    return persistent, diagnostics


def find_valid_threshold(initial_state: np.ndarray, evolve: bool) -> dict[str, object]:
    def classifier(stress: float) -> tuple[bool, dict[str, float | bool]]:
        return simulate_and_classify(stress, evolve=evolve, initial_state=initial_state)

    stress_low = 0.0
    low_persistent, _ = classifier(stress_low)
    if not low_persistent:
        return bisection_threshold(classifier, 0.0, 0.02, tolerance=THRESHOLD_TOLERANCE)

    stress_high = 0.02
    for _ in range(12):
        high_persistent, _ = classifier(stress_high)
        if not high_persistent:
            return bisection_threshold(classifier, stress_low, stress_high, tolerance=THRESHOLD_TOLERANCE)
        stress_high *= 2.0
    return bisection_threshold(classifier, stress_low, stress_high, tolerance=THRESHOLD_TOLERANCE)


def threshold_row(run_id: str, evolve: bool, threshold: dict[str, object], initial_state: np.ndarray) -> dict[str, object]:
    low_metrics = threshold["low_metrics"]
    high_metrics = threshold["high_metrics"]
    return {
        "run_id": run_id,
        "tradeoff_label": TRADEOFF_LABEL,
        "evolve": bool(evolve),
        "stress_low": threshold["stress_low"],
        "stress_high": threshold["stress_high"],
        "threshold": threshold["threshold"],
        "threshold_gap": threshold["threshold_gap"],
        "threshold_status": threshold["threshold_status"],
        "initial_n": float(initial_state[0]),
        "initial_w": float(initial_state[1]),
        "initial_q": float(initial_state[2]),
        "tail_mean_w_at_low": low_metrics["tail_mean_w"],
        "tail_mean_w_at_high": high_metrics["tail_mean_w"],
        "q_tail_mean_at_threshold_low": low_metrics["q_tail_mean"],
        "q_change_at_threshold_low": low_metrics["q_change_from_initial"],
        "persistent_low": threshold["persistent_low"],
        "persistent_high": threshold["persistent_high"],
        "notes": "q fixed at baseline" if not evolve else "q evolves after stress",
    }


def finite_threshold(row: dict[str, object]) -> float:
    value = float(row["threshold"])
    return value if math.isfinite(value) else float("nan")


def build_timeseries_stresses(no_evo_threshold: float, evo_threshold: float) -> list[tuple[str, float]]:
    stresses: list[tuple[str, float]] = [("unstressed", 0.0)]
    if math.isfinite(no_evo_threshold):
        stresses.append(("below_no_evo_threshold", max(0.0, 0.5 * no_evo_threshold)))
    if math.isfinite(no_evo_threshold) and math.isfinite(evo_threshold) and evo_threshold - no_evo_threshold > THRESHOLD_TOLERANCE:
        stresses.append(("between_no_evo_and_evo_threshold", 0.5 * (no_evo_threshold + evo_threshold)))
    if math.isfinite(evo_threshold):
        stresses.append(("above_evo_threshold", evo_threshold + max(0.02, 4.0 * THRESHOLD_TOLERANCE)))

    deduped: list[tuple[str, float]] = []
    seen: set[float] = set()
    for scenario, stress in stresses:
        rounded = round(stress, 10)
        if rounded not in seen:
            seen.add(rounded)
            deduped.append((scenario, stress))
    return deduped


def timeseries_rows(initial_state: np.ndarray, no_evo_threshold: float, evo_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_idx = 1
    for scenario, stress in build_timeseries_stresses(no_evo_threshold, evo_threshold):
        for evolve in (False, True):
            trajectory = simulate_ode_evo(PARAMS, initial_state, stress=stress, evolve=evolve, T=STRESS_T, n_eval=501)
            n, w, q = trajectory.y
            z = free_space_evo(n, w, PARAMS)
            r_q = r_of_q(q, PARAMS)
            a_q = a_of_q(q, PARAMS)
            b_q = b_of_q(q, PARAMS)
            gradient = selection_gradient(n, w, q, PARAMS)
            run_id = f"TS_{run_idx:03d}"
            run_idx += 1
            for idx, t_value in enumerate(trajectory.t):
                rows.append(
                    {
                        "run_id": run_id,
                        "scenario": scenario,
                        "evolve": bool(evolve),
                        "stress": float(stress),
                        "time": float(t_value),
                        "n": float(n[idx]),
                        "w": float(w[idx]),
                        "q": float(q[idx]),
                        "z": float(z[idx]),
                        "r_q": float(r_q[idx]),
                        "a_q": float(a_q[idx]),
                        "b_q": float(b_q[idx]),
                        "selection_gradient": float(gradient[idx]),
                    }
                )
    return rows


def classify_final_label(
    no_evo_row: dict[str, object],
    evo_row: dict[str, object],
    window_no_evo_metrics: dict[str, float | bool] | None,
    window_evo_metrics: dict[str, float | bool] | None,
    baseline_q: float,
) -> tuple[str, str]:
    no_evo_threshold = finite_threshold(no_evo_row)
    evo_threshold = finite_threshold(evo_row)
    if not (math.isfinite(no_evo_threshold) and math.isfinite(evo_threshold)):
        return "ODE_inconclusive", "At least one threshold bracket was invalid."
    delta = evo_threshold - no_evo_threshold
    if delta <= THRESHOLD_TOLERANCE:
        return (
            "ODE_no_indirect_rescue_under_tested_tradeoff",
            "Evolution did not increase the predator mortality threshold beyond tolerance.",
        )
    if window_evo_metrics is None:
        return "ODE_inconclusive", "No stress window between no-evolution and evolution thresholds was evaluated."
    if window_no_evo_metrics is None:
        return "ODE_inconclusive", "No no-evolution trajectory was evaluated inside the candidate rescue window."
    q_change = float(window_evo_metrics["q_change_from_initial"])
    if (
        not bool(window_no_evo_metrics["persistent_predator"])
        and bool(window_evo_metrics["persistent_predator"])
        and q_change < -1.0e-3
        and float(window_evo_metrics["q_tail_mean"]) < baseline_q
    ):
        return (
            "ODE_indirect_evolutionary_rescue_supported",
            "Evolution increases the threshold, produces a persistence window, and lowers q after stress.",
        )
    return (
        "ODE_no_indirect_rescue_under_tested_tradeoff",
        "The threshold may shift, but the persistence window or q-decrease criterion failed.",
    )


def write_summary(
    equilibrium: dict[str, object],
    threshold_rows: list[dict[str, object]],
    final_label: str,
    interpretation: str,
    window_stress: float | None,
    window_no_evo_metrics: dict[str, float | bool] | None,
    window_evo_metrics: dict[str, float | bool] | None,
) -> None:
    no_evo = threshold_rows[0]
    evo = threshold_rows[1]
    no_evo_threshold = finite_threshold(no_evo)
    evo_threshold = finite_threshold(evo)
    delta = evo_threshold - no_evo_threshold
    params = PARAMS
    initial_n = float(equilibrium["n"])
    initial_w = float(equilibrium["w"])
    initial_q = float(equilibrium["q"])
    initial_z = float(equilibrium["z"])

    lines = [
        "# Nonlinear PDE Results 09: ODE Eco-Evolutionary Rescue Gate",
        "",
        f"**Final ODE label: `{final_label}`.** {interpretation}",
        "",
        "## Direct Answer",
        "",
        (
            "Prey defense evolution increases the predator mortality threshold in the well-mixed ODE "
            if final_label == "ODE_indirect_evolutionary_rescue_supported"
            else "Prey defense evolution does not satisfy all ODE rescue criteria under the tested tradeoff "
        )
        + f"for `{TRADEOFF_LABEL}`.",
        "",
        "## Equations Implemented",
        "",
        "`z = 1/kappa - n - w`",
        "",
        "`r(q) = r_u * (1 - q) + r_v * q`",
        "",
        "`a(q) = a_u * (1 - q) + a_v * q`",
        "",
        "`b(q) = b_u * (1 - q) + b_v * q`",
        "",
        "`dn/dt = n * (r(q) * z - xi - a(q) * w)`",
        "",
        "`dw/dt = w * (b(q) * n * z - (m + stress) - mu * w)`",
        "",
        "`dq/dt = nu * q * (1 - q) * ((r_v - r_u) * z - (a_v - a_u) * w)`",
        "",
        "When `evolve=False`, `dq/dt = 0` and q is frozen at the baseline value.",
        "",
        "## Parameter Values",
        "",
        "| parameter | value |",
        "|---|---:|",
    ]
    for name in ("kappa", "xi", "r_u", "r_v", "a_u", "a_v", "b_u", "b_v", "m", "mu", "nu"):
        lines.append(f"| `{name}` | {getattr(params, name):.8g} |")
    lines.extend(
        [
            "",
            "The source dataclass keeps the setup-note defaults. This Step 09A run uses a single documented adjusted conversion tradeoff with `b_u=0.08` and `b_v=0.02`. The reason is to place the unstressed evolving baseline at an interior defense frequency, making the ODE gate test a genuine q-response problem rather than a boundary-pinned q=1 control.",
            "",
            "## Initial Condition",
            "",
            f"Burn-in method: integrate the unstressed evolving ODE from `[n0, w0, q0] = [{BURN_IN_INITIAL[0]}, {BURN_IN_INITIAL[1]}, {BURN_IN_INITIAL[2]}]` to `T={BURN_IN_T:g}`.",
            "",
            f"Baseline state used for stress tests: `n={initial_n:.8g}`, `w={initial_w:.8g}`, `q={initial_q:.8g}`, `z={initial_z:.8g}`.",
            "",
            f"Burn-in residual norm: `{float(equilibrium['residual']):.6g}`.",
            "",
            "## Threshold Results",
            "",
            f"- no-evolution threshold: `{no_evo_threshold:.8g}`",
            f"- evolution threshold: `{evo_threshold:.8g}`",
            f"- Delta m_c_evo: `{delta:.8g}`",
            f"- threshold tolerance: `{THRESHOLD_TOLERANCE:g}`",
            "",
            "| condition | bracket low | bracket high | threshold | persistent low | persistent high | q tail at low | q change at low |",
            "|---|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for row in threshold_rows:
        label = "evolution" if bool(row["evolve"]) else "no evolution"
        lines.append(
            f"| {label} | {float(row['stress_low']):.8g} | {float(row['stress_high']):.8g} | "
            f"{float(row['threshold']):.8g} | {row['persistent_low']} | {row['persistent_high']} | "
            f"{float(row['q_tail_mean_at_threshold_low']):.8g} | {float(row['q_change_at_threshold_low']):.8g} |"
        )
    if window_stress is not None and window_no_evo_metrics is not None and window_evo_metrics is not None:
        lines.extend(
            [
                "",
                "## Rescue Window Check",
                "",
                f"Stress between thresholds: `{window_stress:.8g}`.",
                "",
                f"- no-evolution persistent: `{window_no_evo_metrics['persistent_predator']}`; tail mean w = `{float(window_no_evo_metrics['tail_mean_w']):.8g}`",
                f"- evolution persistent: `{window_evo_metrics['persistent_predator']}`; tail mean w = `{float(window_evo_metrics['tail_mean_w']):.8g}`",
                f"- baseline q: `{initial_q:.8g}`",
                f"- evolving q tail mean in window: `{float(window_evo_metrics['q_tail_mean']):.8g}`",
                f"- q change in window: `{float(window_evo_metrics['q_change_from_initial']):.8g}`",
            ]
        )
    else:
        lines.extend(["", "## Rescue Window Check", "", "No valid stress window was available."])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ODE rescue claim is accepted only when the evolution threshold exceeds the no-evolution threshold by more than tolerance, a stress window exists where no-evolution goes extinct but evolution persists, and q decreases relative to the baseline in that window.",
            "",
            interpretation,
            "",
            "## Outputs",
            "",
            f"- `{THRESHOLD_CSV.relative_to(ROOT)}`",
            f"- `{TIMESERIES_CSV.relative_to(ROOT)}`",
        ]
    )
    if final_label == "ODE_indirect_evolutionary_rescue_supported":
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Step 09B should implement the spatial PDE comparison using this ODE-supported tradeoff, with no-flux boundaries and the same threshold and q-response diagnostics.",
            ]
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    equilibrium = find_evo_equilibrium(PARAMS, guesses=(BURN_IN_INITIAL,), burn_in_T=BURN_IN_T)
    initial_state = np.array([equilibrium["n"], equilibrium["w"], equilibrium["q"]], dtype=float)

    no_evo_threshold = find_valid_threshold(initial_state, evolve=False)
    evo_threshold = find_valid_threshold(initial_state, evolve=True)
    threshold_rows = [
        threshold_row("EVO_ODE_001", False, no_evo_threshold, initial_state),
        threshold_row("EVO_ODE_002", True, evo_threshold, initial_state),
    ]
    write_csv(threshold_rows, THRESHOLD_CSV, THRESHOLD_FIELDNAMES)

    no_evo_value = finite_threshold(threshold_rows[0])
    evo_value = finite_threshold(threshold_rows[1])
    window_stress: float | None = None
    window_no_evo_metrics: dict[str, float | bool] | None = None
    window_evo_metrics: dict[str, float | bool] | None = None
    if math.isfinite(no_evo_value) and math.isfinite(evo_value) and evo_value - no_evo_value > THRESHOLD_TOLERANCE:
        window_stress = 0.5 * (no_evo_value + evo_value)
        _, window_no_evo_metrics = simulate_and_classify(window_stress, evolve=False, initial_state=initial_state)
        _, window_evo_metrics = simulate_and_classify(window_stress, evolve=True, initial_state=initial_state)

    final_label, interpretation = classify_final_label(
        threshold_rows[0],
        threshold_rows[1],
        window_no_evo_metrics,
        window_evo_metrics,
        baseline_q=float(initial_state[2]),
    )
    ts_rows = timeseries_rows(initial_state, no_evo_value, evo_value)
    write_csv(ts_rows, TIMESERIES_CSV, TIMESERIES_FIELDNAMES)
    write_summary(
        equilibrium,
        threshold_rows,
        final_label,
        interpretation,
        window_stress,
        window_no_evo_metrics,
        window_evo_metrics,
    )
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    run()
