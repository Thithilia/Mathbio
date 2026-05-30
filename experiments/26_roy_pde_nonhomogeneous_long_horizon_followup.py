#!/usr/bin/env python
"""Long-horizon follow-up for basin-changing non-homogeneous PDE cases.

This experiment extends only the finite-amplitude non-homogeneous cases that
changed basin labels in Step 25. It reuses the Step 25 PDE setup and
classification logic, compares each heterogeneous case with a matched
homogeneous control, and does not run a broad PDE scan.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_step25_module():
    path = Path(__file__).resolve().with_name("25_roy_pde_spatial_stability_and_nonhomogeneous_tests.py")
    spec = importlib.util.spec_from_file_location("step25_pde_spatial_followup_runtime", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load Step 25 module from {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP25 = load_step25_module()

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTE_PATH = ROOT / "research_notes" / "roy_pde_nonhomogeneous_long_horizon_followup.md"

PR23_SUMMARY_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_perturbation_summary.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_summary.csv"
MEAN_TIMESERIES_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_mean_timeseries.csv"
SPATIAL_METRICS_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_spatial_metrics.csv"
DECISION_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_long_horizon_decision.csv"

FIG54_PATH = FIG_DIR / "fig54_long_horizon_mean_timeseries.png"
FIG55_PATH = FIG_DIR / "fig55_long_horizon_spatial_metrics.png"
FIG56_PATH = FIG_DIR / "fig56_long_horizon_final_fields.png"
FIG57_PATH = FIG_DIR / "fig57_long_horizon_decision.png"

FOCUSED_HORIZONS = (1600.0, 3200.0, 6400.0)
FULL_HORIZONS = (1600.0, 3200.0, 6400.0, 10000.0)
FINAL_CV_THRESHOLD = 1.0e-3
MAX_CV_THRESHOLD = 1.0e-2

SUMMARY_FIELDS = [
    "case_label",
    "followup_case_id",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "amplitude",
    "q_amplitude",
    "T",
    "dt",
    "initial_mean_n",
    "initial_mean_w",
    "initial_mean_q",
    "final_mean_n",
    "final_mean_w",
    "final_mean_q",
    "classification",
    "basin_label",
    "homogeneous_control_classification",
    "homogeneous_control_basin_label",
    "basin_changed_relative_to_homogeneous_control",
    "tail_mean_w",
    "tail_mean_q",
    "relative_change_between_last_windows",
    "normalized_residual",
    "initial_cv_n",
    "initial_cv_w",
    "initial_cv_q",
    "final_cv_n",
    "final_cv_w",
    "final_cv_q",
    "max_cv_n",
    "max_cv_w",
    "max_cv_q",
    "spatial_pattern_persisted",
    "resolved_relative_to_pr23",
    "physical",
    "notes",
]

TIMESERIES_FIELDS = [
    "case_label",
    "followup_case_id",
    "run_role",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "T",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
]

SPATIAL_FIELDS = [
    "case_label",
    "followup_case_id",
    "run_role",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "T",
    "time",
    "var_n",
    "var_w",
    "var_q",
    "cv_n",
    "cv_w",
    "cv_q",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
]

DECISION_FIELDS = ["metric", "value", "interpretation"]


@dataclass(frozen=True)
class FollowupCase:
    case_label: str
    stress: float
    baseline_state: str
    perturbation_type: str
    seed: int

    @property
    def followup_case_id(self) -> str:
        return safe_filename(self.case_label)


@dataclass
class RunRecord:
    case: FollowupCase
    T: float
    role: str
    result: Any
    metrics: dict[str, Any]
    amplitude: float
    q_amplitude: float
    initial_fields: tuple[np.ndarray, np.ndarray, np.ndarray]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:140]


def select_followup_cases(rows: list[dict[str, Any]]) -> list[FollowupCase]:
    cases: list[FollowupCase] = []
    for row in rows:
        if not as_bool(row.get("basin_changed_relative_to_homogeneous_control", False)):
            continue
        cases.append(
            FollowupCase(
                case_label=str(row["case_label"]),
                stress=float(row["stress"]),
                baseline_state=str(row["baseline_state"]),
                perturbation_type=str(row["perturbation_type"]),
                seed=int(float(row["seed"])),
            )
        )
    return cases


def coefficient_of_variation(field: np.ndarray, mean_value: float | None = None) -> float:
    array = np.asarray(field, dtype=float)
    mean = float(np.mean(array)) if mean_value is None else float(mean_value)
    return float(np.std(array) / max(abs(mean), 1.0e-12))


def horizons_for_profile(profile: str) -> tuple[float, ...]:
    return FULL_HORIZONS if profile == "full" else FOCUSED_HORIZONS


def snapshot_times_for_horizon(T: float) -> np.ndarray:
    return np.array([fraction * T for fraction in STEP25.SNAPSHOT_FRACTIONS], dtype=float)


def run_single(case: FollowupCase, T: float, role: str, baseline: tuple[float, float, float]) -> RunRecord:
    config = STEP25.pde_config(T)
    perturbation_type = "homogeneous_control" if role == "control" else case.perturbation_type
    mean_state = STEP25.mean_state_for_baseline(case.stress, case.baseline_state, baseline)
    n0, w0, q0, amplitude, q_amplitude = STEP25.make_initial_fields(mean_state, perturbation_type, case.seed, config)
    result = STEP25.simulate_pde_with_snapshots(
        params=STEP25.PARAMS,
        config=config,
        initial_state=(n0, w0, q0),
        stress=case.stress,
        snapshot_times=snapshot_times_for_horizon(T),
    )
    metrics = STEP25.classify_pde_result(result, config, case.stress)
    return RunRecord(
        case=case,
        T=T,
        role=role,
        result=result,
        metrics=metrics,
        amplitude=amplitude,
        q_amplitude=q_amplitude,
        initial_fields=(n0, w0, q0),
    )


def cv_series(record: RunRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = record.result
    cv_n = np.sqrt(result.var_n) / np.maximum(np.abs(result.mean_n), 1.0e-12)
    cv_w = np.sqrt(result.var_w) / np.maximum(np.abs(result.mean_w), 1.0e-12)
    cv_q = np.sqrt(result.var_q) / np.maximum(np.abs(result.mean_q), 1.0e-12)
    return cv_n, cv_w, cv_q


def spatial_pattern_persisted(final_cv_n: float, final_cv_w: float, final_cv_q: float) -> bool:
    return bool(max(final_cv_n, final_cv_w, final_cv_q) > FINAL_CV_THRESHOLD)


def close_to_control(record: RunRecord, control: RunRecord) -> bool:
    result = record.result
    control_result = control.result
    rel_w = abs(float(np.mean(result.w_final)) - float(np.mean(control_result.w_final))) / max(
        abs(float(np.mean(control_result.w_final))), STEP25.EPSILON
    )
    rel_q = abs(float(np.mean(result.q_final)) - float(np.mean(control_result.q_final))) / max(
        abs(float(np.mean(control_result.q_final))), STEP25.EPSILON
    )
    return bool(max(rel_w, rel_q) < 1.0e-3)


def resolution_status(record: RunRecord, control: RunRecord, basin_changed: bool, pattern_persisted: bool) -> str:
    classification = str(record.metrics["classification"])
    if pattern_persisted:
        return "persistent_spatial_pattern"
    if not basin_changed:
        return "resolved_to_control"
    if classification in {"persistent_steady", "extinct_steady"}:
        return "persistent_basin_change"
    if "transient" in classification and close_to_control(record, control):
        return "likely_resolving_but_not_final"
    return "long_transient_unresolved"


def append_timeseries(rows: list[dict[str, Any]], record: RunRecord) -> None:
    case = record.case
    perturbation_type = "homogeneous_control" if record.role == "control" else case.perturbation_type
    for time, mean_n, mean_w, mean_q in zip(
        record.result.times,
        record.result.mean_n,
        record.result.mean_w,
        record.result.mean_q,
        strict=True,
    ):
        rows.append(
            {
                "case_label": case.case_label,
                "followup_case_id": case.followup_case_id,
                "run_role": record.role,
                "stress": case.stress,
                "baseline_state": case.baseline_state,
                "perturbation_type": perturbation_type,
                "seed": case.seed,
                "T": record.T,
                "time": time,
                "mean_n": mean_n,
                "mean_w": mean_w,
                "mean_q": mean_q,
            }
        )


def append_spatial_metrics(rows: list[dict[str, Any]], record: RunRecord) -> None:
    case = record.case
    perturbation_type = "homogeneous_control" if record.role == "control" else case.perturbation_type
    cv_n, cv_w, cv_q = cv_series(record)
    for idx, time in enumerate(record.result.times):
        rows.append(
            {
                "case_label": case.case_label,
                "followup_case_id": case.followup_case_id,
                "run_role": record.role,
                "stress": case.stress,
                "baseline_state": case.baseline_state,
                "perturbation_type": perturbation_type,
                "seed": case.seed,
                "T": record.T,
                "time": time,
                "var_n": record.result.var_n[idx],
                "var_w": record.result.var_w[idx],
                "var_q": record.result.var_q[idx],
                "cv_n": cv_n[idx],
                "cv_w": cv_w[idx],
                "cv_q": cv_q[idx],
                "min_n": record.result.min_n_series[idx],
                "min_w": record.result.min_w_series[idx],
                "min_q": record.result.min_q_series[idx],
                "max_q": record.result.max_q_series[idx],
                "min_z": record.result.min_z_series[idx],
            }
        )


def summary_row(record: RunRecord, control: RunRecord) -> dict[str, Any]:
    n0, w0, q0 = record.initial_fields
    cv_n, cv_w, cv_q = cv_series(record)
    final_cv_n = float(cv_n[-1])
    final_cv_w = float(cv_w[-1])
    final_cv_q = float(cv_q[-1])
    basin_changed = STEP25.basin_changed(str(record.metrics["basin_label"]), str(control.metrics["basin_label"]))
    pattern_persisted = spatial_pattern_persisted(final_cv_n, final_cv_w, final_cv_q)
    status = resolution_status(record, control, basin_changed, pattern_persisted)
    notes = []
    if max(float(np.max(cv_n)), float(np.max(cv_w)), float(np.max(cv_q))) > MAX_CV_THRESHOLD and not pattern_persisted:
        notes.append("large_initial_or_transient_cv_decayed")
    if "transient" in str(record.metrics["classification"]):
        notes.append("transient_classification")
    if close_to_control(record, control):
        notes.append("final_means_close_to_control")
    if not as_bool(record.metrics["physical"]):
        notes.append("nonphysical_or_incomplete")
    notes.append(status)
    return {
        "case_label": record.case.case_label,
        "followup_case_id": record.case.followup_case_id,
        "stress": record.case.stress,
        "baseline_state": record.case.baseline_state,
        "perturbation_type": record.case.perturbation_type,
        "seed": record.case.seed,
        "amplitude": record.amplitude,
        "q_amplitude": record.q_amplitude,
        "T": record.T,
        "dt": STEP25.FOCUSED_DT,
        "initial_mean_n": float(np.mean(n0)),
        "initial_mean_w": float(np.mean(w0)),
        "initial_mean_q": float(np.mean(q0)),
        "final_mean_n": float(np.mean(record.result.n_final)),
        "final_mean_w": float(np.mean(record.result.w_final)),
        "final_mean_q": float(np.mean(record.result.q_final)),
        "classification": record.metrics["classification"],
        "basin_label": record.metrics["basin_label"],
        "homogeneous_control_classification": control.metrics["classification"],
        "homogeneous_control_basin_label": control.metrics["basin_label"],
        "basin_changed_relative_to_homogeneous_control": basin_changed,
        "tail_mean_w": record.metrics["tail_mean_w"],
        "tail_mean_q": record.metrics["tail_mean_q"],
        "relative_change_between_last_windows": record.metrics["relative_change_between_last_windows"],
        "normalized_residual": record.metrics["normalized_residual"],
        "initial_cv_n": coefficient_of_variation(n0),
        "initial_cv_w": coefficient_of_variation(w0),
        "initial_cv_q": coefficient_of_variation(q0),
        "final_cv_n": final_cv_n,
        "final_cv_w": final_cv_w,
        "final_cv_q": final_cv_q,
        "max_cv_n": float(np.max(cv_n)),
        "max_cv_w": float(np.max(cv_w)),
        "max_cv_q": float(np.max(cv_q)),
        "spatial_pattern_persisted": pattern_persisted,
        "resolved_relative_to_pr23": status,
        "physical": record.metrics["physical"],
        "notes": ";".join(notes),
    }


def save_field_archive(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snapshot_times=record.result.snapshot_times,
        n_snapshots=record.result.n_snapshots,
        w_snapshots=record.result.w_snapshots,
        q_snapshots=record.result.q_snapshots,
        stress=record.case.stress,
        baseline_state=record.case.baseline_state,
        perturbation_type=record.case.perturbation_type,
        seed=record.case.seed,
        T=record.T,
        amplitude=record.amplitude,
        q_amplitude=record.q_amplitude,
    )


def field_archive_path(case: FollowupCase) -> Path:
    return RESULTS_DIR / f"roy_pde_nonhomogeneous_long_horizon_fields_{case.followup_case_id}.npz"


def decide_final_label(longest_rows: list[dict[str, Any]]) -> str:
    if not longest_rows:
        return "nonhomogeneous_basin_changes_unresolved"
    if all(str(row["resolved_relative_to_pr23"]) == "resolved_to_control" for row in longest_rows):
        return "nonhomogeneous_basin_changes_resolve_to_homogeneous_control"
    if any(str(row["resolved_relative_to_pr23"]) == "persistent_spatial_pattern" for row in longest_rows):
        return "nonhomogeneous_basin_changes_persist_with_spatial_pattern"
    if any(str(row["resolved_relative_to_pr23"]) == "persistent_basin_change" for row in longest_rows):
        return "nonhomogeneous_basin_changes_persist_without_spatial_pattern"
    return "nonhomogeneous_basin_changes_unresolved"


def write_decision(summary_rows: list[dict[str, Any]], horizons: tuple[float, ...]) -> tuple[list[dict[str, Any]], str]:
    longest = max(horizons)
    longest_rows = [row for row in summary_rows if math.isclose(float(row["T"]), longest)]
    followed_cases = {str(row["followup_case_id"]) for row in summary_rows}
    resolved = sum(str(row["resolved_relative_to_pr23"]) == "resolved_to_control" for row in longest_rows)
    persist_without = sum(str(row["resolved_relative_to_pr23"]) == "persistent_basin_change" for row in longest_rows)
    persist_with = sum(str(row["resolved_relative_to_pr23"]) == "persistent_spatial_pattern" for row in longest_rows)
    unresolved = len(longest_rows) - resolved - persist_without - persist_with
    max_final_cv = max(
        (
            max(float(row["final_cv_n"]), float(row["final_cv_w"]), float(row["final_cv_q"]))
            for row in longest_rows
        ),
        default=math.nan,
    )
    final_label = decide_final_label(longest_rows)
    rows = [
        {
            "metric": "followed_cases_count",
            "value": len(followed_cases),
            "interpretation": "PR #23 basin-changing non-homogeneous cases followed",
        },
        {
            "metric": "horizons_tested",
            "value": ";".join(f"{h:g}" for h in horizons),
            "interpretation": "PDE horizons used for each followed case and matched homogeneous control",
        },
        {
            "metric": "cases_resolved_to_homogeneous_control",
            "value": resolved,
            "interpretation": "followed cases matching the homogeneous-control basin at the longest horizon",
        },
        {
            "metric": "cases_persist_different_basin_without_pattern",
            "value": persist_without,
            "interpretation": "followed cases with a different steady-like basin and no final spatial pattern",
        },
        {
            "metric": "cases_persist_different_basin_with_pattern",
            "value": persist_with,
            "interpretation": "followed cases with different basin and final spatial CV above threshold",
        },
        {
            "metric": "cases_unresolved",
            "value": unresolved,
            "interpretation": "followed cases still transient or near-boundary at the longest horizon",
        },
        {
            "metric": "max_final_cv_at_longest_horizon",
            "value": max_final_cv,
            "interpretation": "maximum final CV across n,w,q among followed cases at the longest horizon",
        },
        {
            "metric": "final_label",
            "value": final_label,
            "interpretation": "allowed final label for the long-horizon follow-up",
        },
    ]
    write_csv(DECISION_CSV, rows, DECISION_FIELDS)
    return rows, final_label


def plot_mean_timeseries(timeseries_rows: list[dict[str, Any]], horizons: tuple[float, ...]) -> None:
    longest = max(horizons)
    selected = [row for row in timeseries_rows if math.isclose(float(row["T"]), longest)]
    case_ids = sorted({str(row["followup_case_id"]) for row in selected})
    fig, axes = plt.subplots(len(case_ids), 2, figsize=(11.0, 4.0 * len(case_ids)), squeeze=False)
    colors = {"heterogeneous": "#d95f02", "control": "#2f6fbb"}
    for row_idx, case_id in enumerate(case_ids):
        for role in ("control", "heterogeneous"):
            rows = sorted(
                [row for row in selected if str(row["followup_case_id"]) == case_id and str(row["run_role"]) == role],
                key=lambda row: float(row["time"]),
            )
            if not rows:
                continue
            time = [float(row["time"]) for row in rows]
            axes[row_idx, 0].plot(time, [float(row["mean_w"]) for row in rows], color=colors[role], label=role)
            axes[row_idx, 1].plot(time, [float(row["mean_q"]) for row in rows], color=colors[role], label=role)
        axes[row_idx, 0].set_title(case_id)
        axes[row_idx, 0].set_ylabel(r"mean predator $\bar w$")
        axes[row_idx, 1].set_ylabel(r"mean defense $\bar q$")
        for ax in axes[row_idx]:
            ax.set_xlabel("time")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
    fig.suptitle(f"Long-horizon mean dynamics at T={longest:g}")
    fig.tight_layout()
    save_figure(fig, FIG54_PATH)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_metrics(spatial_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], horizons: tuple[float, ...]) -> None:
    longest = max(horizons)
    selected = [row for row in spatial_rows if math.isclose(float(row["T"]), longest) and str(row["run_role"]) == "heterogeneous"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["followup_case_id"]), []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for case_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: float(row["time"]))
        max_cv = [max(float(row["cv_n"]), float(row["cv_w"]), float(row["cv_q"])) for row in rows]
        axes[0].plot([float(row["time"]) for row in rows], max_cv, label=case_id)
    axes[0].axhline(FINAL_CV_THRESHOLD, color="black", linestyle=":", linewidth=0.9, label="final CV threshold")
    axes[0].set_yscale("symlog", linthresh=1.0e-8)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("max CV across n,w,q")
    axes[0].set_title(f"CV decay at T={longest:g}")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(alpha=0.25)

    case_ids = sorted({str(row["followup_case_id"]) for row in summary_rows})
    x = np.arange(len(horizons))
    width = 0.8 / max(len(case_ids), 1)
    for idx, case_id in enumerate(case_ids):
        rows = [row for row in summary_rows if str(row["followup_case_id"]) == case_id]
        values = []
        for horizon in horizons:
            match = next(row for row in rows if math.isclose(float(row["T"]), horizon))
            values.append(max(float(match["final_cv_n"]), float(match["final_cv_w"]), float(match["final_cv_q"])))
        axes[1].bar(x + idx * width, values, width=width, label=case_id)
    axes[1].axhline(FINAL_CV_THRESHOLD, color="black", linestyle=":", linewidth=0.9)
    axes[1].set_xticks(x + width * (len(case_ids) - 1) / 2.0, [f"{h:g}" for h in horizons])
    axes[1].set_xlabel("horizon T")
    axes[1].set_ylabel("final max CV")
    axes[1].set_title("Final CV by horizon")
    axes[1].set_yscale("symlog", linthresh=1.0e-8)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Long-horizon spatial heterogeneity diagnostics")
    fig.tight_layout()
    save_figure(fig, FIG55_PATH)


def plot_final_fields(records: list[RunRecord], horizons: tuple[float, ...]) -> None:
    longest = max(horizons)
    selected = [record for record in records if record.role == "heterogeneous" and math.isclose(record.T, longest)]
    fig, axes = plt.subplots(3, len(selected), figsize=(4.0 * len(selected), 8.0), squeeze=False, constrained_layout=True)
    for col, record in enumerate(selected):
        fields = [record.result.n_final, record.result.w_final, record.result.q_final]
        labels = ["n", "w", "q"]
        for row, (field, label) in enumerate(zip(fields, labels, strict=True)):
            im = axes[row, col].imshow(field, origin="lower", cmap="viridis")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f"s={record.case.stress:.8g}\n{record.case.followup_case_id}", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(label)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.suptitle(f"Final heterogeneous fields at T={longest:g}")
    save_figure(fig, FIG56_PATH)


def plot_decision(decision_rows: list[dict[str, Any]]) -> None:
    data = {str(row["metric"]): row["value"] for row in decision_rows}
    labels = ["resolved", "different\nno pattern", "different\nwith pattern", "unresolved"]
    values = [
        float(data["cases_resolved_to_homogeneous_control"]),
        float(data["cases_persist_different_basin_without_pattern"]),
        float(data["cases_persist_different_basin_with_pattern"]),
        float(data["cases_unresolved"]),
    ]
    colors = ["#1b9e77", "#d95f02", "#c23b3b", "#777777"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel("followed cases")
    ax.set_title("Long-horizon basin-change decision")
    final_label = str(data["final_label"])
    ax.text(
        0.5,
        0.88,
        f"Final label:\n{final_label}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef3f8", "edgecolor": "#2f6fbb"},
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, FIG57_PATH)


def interpretation_sentence(final_label: str) -> str:
    if final_label == "nonhomogeneous_basin_changes_resolve_to_homogeneous_control":
        return (
            "The PR #23 basin changes were finite-horizon transient effects near basin-boundary states "
            "rather than persistent spatial-pattern-mediated basin changes."
        )
    if final_label == "nonhomogeneous_basin_changes_persist_without_spatial_pattern":
        return "Finite-amplitude heterogeneity can move the system into a different homogeneous basin without generating persistent spatial patterning."
    if final_label == "nonhomogeneous_basin_changes_persist_with_spatial_pattern":
        return "Finite-amplitude heterogeneity can generate persistent spatially structured outcomes in the tested setup."
    return "The long-horizon follow-up leaves the basin-changing non-homogeneous cases unresolved as long transients or near-boundary outcomes."


def write_note(
    cases: list[FollowupCase],
    summary_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    final_label: str,
    horizons: tuple[float, ...],
) -> None:
    longest = max(horizons)
    longest_rows = [row for row in summary_rows if math.isclose(float(row["T"]), longest)]
    max_final_cv = max(
        (
            max(float(row["final_cv_n"]), float(row["final_cv_w"]), float(row["final_cv_q"]))
            for row in longest_rows
        ),
        default=math.nan,
    )
    case_lines = "\n".join(
        f"- `{case.case_label}`: stress={case.stress:g}, baseline=`{case.baseline_state}`, perturbation=`{case.perturbation_type}`, seed={case.seed}"
        for case in cases
    )
    decision = {str(row["metric"]): row["value"] for row in decision_rows}
    status_lines = "\n".join(
        f"- `{row['followup_case_id']}` at T={float(row['T']):g}: classification=`{row['classification']}`, "
        f"basin=`{row['basin_label']}`, control basin=`{row['homogeneous_control_basin_label']}`, "
        f"status=`{row['resolved_relative_to_pr23']}`"
        for row in longest_rows
    )
    text = f"""# Long-Horizon Follow-Up of Non-Homogeneous PDE Basin-Changing Cases

## Purpose

This note follows only the basin-changing non-homogeneous PDE cases from PR #23. It tests whether localized finite-amplitude defense heterogeneity caused persistent basin changes or only finite-horizon transient label changes.

## Cases Selected from PR #23

The selected cases are exactly rows in `results/roy_pde_nonhomogeneous_perturbation_summary.csv` with `basin_changed_relative_to_homogeneous_control == True`.

{case_lines}

## Methods

The follow-up uses the same model equations, parameterization, diffusion coefficients, grid, timestep, baseline construction, and local-defense-patch construction as PR #23. It does not run broad PDE scans.

Each selected heterogeneous case is compared with a matched homogeneous control at the same stress, baseline mean state, seed convention, and horizon. The focused horizons are `{'; '.join(f'{h:g}' for h in horizons)}`.

## Long-Horizon Mean Dynamics

At the longest focused horizon, the followed cases have the following status:

{status_lines}

Detailed mean time series are saved in `results/roy_pde_nonhomogeneous_long_horizon_mean_timeseries.csv`.

## Spatial Pattern Decay

The maximum final CV across followed cases at the longest horizon is `{max_final_cv:.6g}`. Final spatial pattern persistence is evaluated with the same `1e-3` final-CV threshold used in PR #23.

## Basin Resolution

At the longest focused horizon:

- resolved to homogeneous control: `{decision['cases_resolved_to_homogeneous_control']}`
- persistent different basin without final spatial pattern: `{decision['cases_persist_different_basin_without_pattern']}`
- persistent different basin with final spatial pattern: `{decision['cases_persist_different_basin_with_pattern']}`
- unresolved: `{decision['cases_unresolved']}`

## Final Label

`{final_label}`

## Interpretation

{interpretation_sentence(final_label)}

## Biological Meaning

The follow-up distinguishes finite-amplitude basin entry from persistent spatial patterning. A basin-label change without final spatial CV persistence is not evidence for a spatial-pattern-mediated rescue mechanism; it indicates finite-amplitude sensitivity near the tested basin boundary.

## What This Proves

This targeted follow-up tests whether the specific PR #23 basin-changing local-defense perturbations persist, resolve, or remain unresolved at longer horizons.

## What This Does Not Prove

This is not a broad PDE parameter scan, not a grid-convergence study, and not a proof for all heterogeneous initial conditions or diffusion coefficients.

## Files

- `experiments/26_roy_pde_nonhomogeneous_long_horizon_followup.py`
- `results/roy_pde_nonhomogeneous_long_horizon_summary.csv`
- `results/roy_pde_nonhomogeneous_long_horizon_mean_timeseries.csv`
- `results/roy_pde_nonhomogeneous_long_horizon_spatial_metrics.csv`
- `results/roy_pde_nonhomogeneous_long_horizon_decision.csv`
- `figures/roy_evo_spatial/report/fig54_long_horizon_mean_timeseries.png`
- `figures/roy_evo_spatial/report/fig55_long_horizon_spatial_metrics.png`
- `figures/roy_evo_spatial/report/fig56_long_horizon_final_fields.png`
- `figures/roy_evo_spatial/report/fig57_long_horizon_decision.png`

## Next Step

Use this result to decide whether any later work should refine local basin-boundary diagnostics rather than broaden PDE scanning.

{final_label}
"""
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(text, encoding="utf-8")


def run_followup(profile: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    pr23_rows = read_csv(PR23_SUMMARY_CSV)
    cases = select_followup_cases(pr23_rows)
    if not cases:
        NOTE_PATH.write_text(
            "# Long-Horizon Follow-Up of Non-Homogeneous PDE Basin-Changing Cases\n\nNo PR #23 basin-changing cases were found, so no follow-up simulations were run.\n\nnonhomogeneous_basin_changes_unresolved\n",
            encoding="utf-8",
        )
        return [], [], [], [], "nonhomogeneous_basin_changes_unresolved"

    horizons = horizons_for_profile(profile)
    baseline = STEP25.baseline_burnin_state()
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    all_records: list[RunRecord] = []
    longest = max(horizons)

    for case in cases:
        for horizon in horizons:
            print(f"Running matched control for {case.case_label} at T={horizon:g}")
            control = run_single(case, horizon, "control", baseline)
            print(f"Running heterogeneous follow-up for {case.case_label} at T={horizon:g}")
            heterogeneous = run_single(case, horizon, "heterogeneous", baseline)
            all_records.extend([control, heterogeneous])
            append_timeseries(timeseries_rows, control)
            append_timeseries(timeseries_rows, heterogeneous)
            append_spatial_metrics(spatial_rows, control)
            append_spatial_metrics(spatial_rows, heterogeneous)
            row = summary_row(heterogeneous, control)
            summary_rows.append(row)
            if math.isclose(horizon, longest):
                save_field_archive(heterogeneous, field_archive_path(case))

    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    write_csv(MEAN_TIMESERIES_CSV, timeseries_rows, TIMESERIES_FIELDS)
    write_csv(SPATIAL_METRICS_CSV, spatial_rows, SPATIAL_FIELDS)
    decision_rows, final_label = write_decision(summary_rows, horizons)
    plot_mean_timeseries(timeseries_rows, horizons)
    plot_spatial_metrics(spatial_rows, summary_rows, horizons)
    plot_final_fields(all_records, horizons)
    plot_decision(decision_rows)
    write_note(cases, summary_rows, decision_rows, final_label, horizons)
    return summary_rows, timeseries_rows, spatial_rows, decision_rows, final_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused", "full"), default="focused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows, _timeseries_rows, _spatial_rows, decision_rows, final_label = run_followup(args.profile)
    for path in [
        SUMMARY_CSV,
        MEAN_TIMESERIES_CSV,
        SPATIAL_METRICS_CSV,
        DECISION_CSV,
        FIG54_PATH,
        FIG55_PATH,
        FIG56_PATH,
        FIG57_PATH,
        NOTE_PATH,
    ]:
        if path.exists():
            print(f"Wrote {path.relative_to(ROOT)}")
    for case_id in sorted({str(row["followup_case_id"]) for row in summary_rows}):
        path = RESULTS_DIR / f"roy_pde_nonhomogeneous_long_horizon_fields_{case_id}.npz"
        if path.exists():
            print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label: {final_label}")
    if decision_rows:
        print("Decision metrics:")
        for row in decision_rows:
            print(f"  {row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
