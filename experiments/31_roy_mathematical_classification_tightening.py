#!/usr/bin/env python
"""Mathematical and classification tightening diagnostics.

This script adds review-facing diagnostics without changing model equations,
baseline parameters, or existing classifications:

1. A dense Routh-Hurwitz/eigenvalue scan along the analytic linear branch.
2. A q-bounds audit of saved PDE diagnostic field files.

It reads existing model helpers and saved outputs; it does not run PDE
simulations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import RoyEvoParams, free_space_evo  # noqa: E402


def load_step23_module():
    path = Path(__file__).resolve().with_name("23_roy_ode_compensation_conditions.py")
    spec = importlib.util.spec_from_file_location("step23_compensation_conditions_runtime", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP23 = load_step23_module()

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"

DENSE_STABILITY_CSV = RESULTS_DIR / "roy_ode_compensation_dense_stability_scan.csv"
DENSE_STABILITY_SUMMARY_CSV = RESULTS_DIR / "roy_ode_compensation_dense_stability_summary.csv"
Q_BOUNDS_AUDIT_CSV = RESULTS_DIR / "roy_pde_q_bounds_audit.csv"
Q_BOUNDS_SUMMARY_CSV = RESULTS_DIR / "roy_pde_q_bounds_audit_summary.csv"
FIG71_PATH = FIG_DIR / "fig71_ode_dense_branch_stability.png"

DENSE_GRID_POINTS = 1001
ENDPOINT_EPS = 1.0e-8
STABILITY_TOL = 1.0e-10
Q_BOUNDS_TOL = 1.0e-10

DENSE_FIELDS = [
    "stress",
    "q_star",
    "n_star",
    "w_star",
    "z_star",
    "A1",
    "A2",
    "A3",
    "A1A2_minus_A3",
    "minimum_rh_margin",
    "routh_hurwitz_stable",
    "max_real_eigenvalue",
    "eigenvalues_real",
    "eigenvalues_imag",
    "eigenvalue_stable",
]

Q_BOUNDS_FIELDS = [
    "source_file",
    "source_group",
    "snapshot_count",
    "min_q",
    "max_q",
    "q_within_unit_interval",
    "q_bounds_tolerance",
    "q_lower_violation",
    "q_upper_violation",
    "notes",
]

SUMMARY_FIELDS = ["metric", "value", "interpretation"]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_float(value: float, digits: int = 8) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def eigen_text(values: np.ndarray) -> tuple[str, str]:
    return (
        ";".join(format_float(float(value), 10) for value in np.real(values)),
        ";".join(format_float(float(value), 10) for value in np.imag(values)),
    )


def characteristic_coefficients(jacobian: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(jacobian, dtype=float)
    trace = float(np.trace(matrix))
    trace_j2 = float(np.trace(matrix @ matrix))
    determinant = float(np.linalg.det(matrix))
    return {
        "A1": -trace,
        "A2": 0.5 * (trace * trace - trace_j2),
        "A3": -determinant,
    }


def branch_interval() -> tuple[float, float]:
    interval = STEP23.stress_interval_for_q_in_unit_interval(PARAMS)
    low = float(interval["interior_stress_interval_low"])
    high = float(interval["interior_stress_interval_high"])
    return low, high


def run_dense_stability_scan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    low, high = branch_interval()
    stresses = np.linspace(low + ENDPOINT_EPS, high - ENDPOINT_EPS, DENSE_GRID_POINTS)
    rows: list[dict[str, Any]] = []
    for stress in stresses:
        n, w, q = STEP23.branch_state(PARAMS, float(stress))
        z = float(free_space_evo(n, w, PARAMS))
        jacobian = STEP23.ode_rhs_jacobian(n, w, q, float(stress), PARAMS)
        coeffs = characteristic_coefficients(jacobian)
        margin = coeffs["A1"] * coeffs["A2"] - coeffs["A3"]
        min_margin = min(coeffs["A1"], coeffs["A2"], coeffs["A3"], margin)
        eigenvalues = np.linalg.eigvals(jacobian)
        max_real = float(np.max(np.real(eigenvalues)))
        real_text, imag_text = eigen_text(eigenvalues)
        rh_stable = bool(
            coeffs["A1"] > STABILITY_TOL
            and coeffs["A2"] > STABILITY_TOL
            and coeffs["A3"] > STABILITY_TOL
            and margin > STABILITY_TOL
        )
        rows.append(
            {
                "stress": float(stress),
                "q_star": float(q),
                "n_star": float(n),
                "w_star": float(w),
                "z_star": z,
                "A1": coeffs["A1"],
                "A2": coeffs["A2"],
                "A3": coeffs["A3"],
                "A1A2_minus_A3": margin,
                "minimum_rh_margin": min_margin,
                "routh_hurwitz_stable": rh_stable,
                "max_real_eigenvalue": max_real,
                "eigenvalues_real": real_text,
                "eigenvalues_imag": imag_text,
                "eigenvalue_stable": bool(max_real < -STABILITY_TOL),
            }
        )

    stable_rows = [row for row in rows if bool(row["routh_hurwitz_stable"]) and bool(row["eigenvalue_stable"])]
    summary_rows = [
        {
            "metric": "dense_grid_points",
            "value": len(rows),
            "interpretation": "stress values sampled across the open interior branch interval",
        },
        {
            "metric": "dense_stress_low",
            "value": float(stresses[0]),
            "interpretation": "first sampled stress, excluding the q=1 endpoint by a small epsilon",
        },
        {
            "metric": "dense_stress_high",
            "value": float(stresses[-1]),
            "interpretation": "last sampled stress, excluding the q=0 endpoint by a small epsilon",
        },
        {
            "metric": "all_dense_points_rh_stable",
            "value": len(stable_rows) == len(rows),
            "interpretation": "True if all dense branch samples satisfy RH inequalities and eigenvalue stability",
        },
        {
            "metric": "minimum_dense_rh_margin",
            "value": min(float(row["minimum_rh_margin"]) for row in rows),
            "interpretation": "smallest value among A1, A2, A3, and A1A2-A3 across the dense scan",
        },
        {
            "metric": "maximum_dense_eigenvalue_real_part",
            "value": max(float(row["max_real_eigenvalue"]) for row in rows),
            "interpretation": "largest real eigenvalue part across the dense scan",
        },
    ]
    write_csv(DENSE_STABILITY_CSV, rows, DENSE_FIELDS)
    write_csv(DENSE_STABILITY_SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    make_dense_stability_figure(rows)
    return rows, summary_rows


def source_group(path: Path) -> str:
    name = path.name
    if name.startswith("roy_pde_evo_representative"):
        return "representative_pde_fields"
    if name.startswith("roy_pde_nonhomogeneous_long_horizon"):
        return "long_horizon_nonhomogeneous_fields"
    if name.startswith("roy_pde_nonhomogeneous"):
        return "nonhomogeneous_pde_fields"
    if name.startswith("roy_nonlinear_tradeoff"):
        return "nonlinear_tradeoff_pde_fields"
    return "other_pde_fields"


def q_values_from_npz(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        arrays: list[np.ndarray] = []
        for key in ("q_snapshots", "final_q", "q_final"):
            if key in data.files:
                arrays.append(np.asarray(data[key], dtype=float).ravel())
        if not arrays:
            return None
        return np.concatenate(arrays)


def run_q_bounds_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*fields*.npz")):
        q_values = q_values_from_npz(path)
        if q_values is None or q_values.size == 0:
            continue
        min_q = float(np.nanmin(q_values))
        max_q = float(np.nanmax(q_values))
        lower_violation = max(0.0, -min_q)
        upper_violation = max(0.0, max_q - 1.0)
        within = bool(lower_violation <= Q_BOUNDS_TOL and upper_violation <= Q_BOUNDS_TOL)
        rows.append(
            {
                "source_file": path.name,
                "source_group": source_group(path),
                "snapshot_count": int(q_values.size),
                "min_q": min_q,
                "max_q": max_q,
                "q_within_unit_interval": within,
                "q_bounds_tolerance": Q_BOUNDS_TOL,
                "q_lower_violation": lower_violation,
                "q_upper_violation": upper_violation,
                "notes": "saved_field_snapshot_bounds_audit",
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_group"])].append(row)
    group_text = "; ".join(f"{group}:{len(group_rows)}" for group, group_rows in sorted(groups.items()))
    summary_rows = [
        {
            "metric": "pde_field_files_audited",
            "value": len(rows),
            "interpretation": "saved NPZ field files with q arrays included in the bounds audit",
        },
        {
            "metric": "pde_field_files_with_q_in_bounds",
            "value": sum(bool(row["q_within_unit_interval"]) for row in rows),
            "interpretation": f"files with min_q>=-tol and max_q<=1+tol, tol={Q_BOUNDS_TOL:g}",
        },
        {
            "metric": "global_min_q_saved_fields",
            "value": min(float(row["min_q"]) for row in rows) if rows else math.nan,
            "interpretation": "minimum q value across saved PDE field snapshots",
        },
        {
            "metric": "global_max_q_saved_fields",
            "value": max(float(row["max_q"]) for row in rows) if rows else math.nan,
            "interpretation": "maximum q value across saved PDE field snapshots",
        },
        {
            "metric": "source_group_file_counts",
            "value": group_text,
            "interpretation": "number of audited field files by diagnostic group",
        },
        {
            "metric": "pde_solver_clip_q_default",
            "value": True,
            "interpretation": "RoyEvoPDEConfig defaults to clip_q=True; saved fields report post-clipping q values",
        },
    ]
    write_csv(Q_BOUNDS_AUDIT_CSV, rows, Q_BOUNDS_FIELDS)
    write_csv(Q_BOUNDS_SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    return rows, summary_rows


def make_dense_stability_figure(rows: list[dict[str, Any]]) -> None:
    stresses = np.asarray([float(row["stress"]) for row in rows])
    min_margin = np.asarray([float(row["minimum_rh_margin"]) for row in rows])
    max_real = np.asarray([float(row["max_real_eigenvalue"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    ax_margin, ax_eig = axes
    ax_margin.plot(stresses, min_margin, color="#2f6fbb", lw=2.0)
    ax_margin.axhline(0.0, color="#111111", lw=1.0, ls="--")
    ax_margin.set_title("A. Dense RH margin")
    ax_margin.set_xlabel("stress s")
    ax_margin.set_ylabel("minimum RH margin")
    ax_margin.grid(alpha=0.25)

    ax_eig.plot(stresses, max_real, color="#9b4a97", lw=2.0)
    ax_eig.axhline(0.0, color="#111111", lw=1.0, ls="--")
    ax_eig.set_title("B. Dense eigenvalue check")
    ax_eig.set_xlabel("stress s")
    ax_eig.set_ylabel("max Re eigenvalue")
    ax_eig.grid(alpha=0.25)

    fig.suptitle("Dense local-stability scan across the open compensation branch", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    FIG71_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG71_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused",), default="focused")
    return parser.parse_args()


def main() -> None:
    parse_args()
    dense_rows, _dense_summary = run_dense_stability_scan()
    q_rows, _q_summary = run_q_bounds_audit()
    group_counts = Counter(str(row["source_group"]) for row in q_rows)
    print(f"Wrote {DENSE_STABILITY_CSV.relative_to(ROOT)} with {len(dense_rows)} rows.")
    print(f"Wrote {DENSE_STABILITY_SUMMARY_CSV.relative_to(ROOT)}.")
    print(f"Wrote {Q_BOUNDS_AUDIT_CSV.relative_to(ROOT)} with {len(q_rows)} audited field files.")
    print(f"Wrote {Q_BOUNDS_SUMMARY_CSV.relative_to(ROOT)} ({dict(group_counts)}).")
    print(f"Wrote {FIG71_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
