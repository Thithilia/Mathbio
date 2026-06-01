#!/usr/bin/env python
"""Spatial and nonlinear robustness clarification diagnostics.

This script adds semi-analytic modal stability evidence for the manuscript
without changing model equations or running PDE simulations.  It evaluates the
Routh-Hurwitz coefficients of the modal matrix

    J(lambda) = J_F(U*) - lambda D

over a dense sampled lambda grid for the current homogeneous compensation
branch states.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import RoyEvoParams  # noqa: E402


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
TARGET_STRESSES = (0.11765625, 0.1584375, 0.16486816, 0.175)

D_N = 0.01
D_W = 0.01
D_Q = 0.005
L_X = 20.0
L_Y = 20.0
MODE_MAX_TESTED = 64
LAMBDA_GRID_POINTS = 2001
STABILITY_TOL = 1.0e-10

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"

MODAL_RH_SCAN_CSV = RESULTS_DIR / "roy_pde_compensation_modal_routh_hurwitz_scan.csv"
MODAL_RH_SUMMARY_CSV = RESULTS_DIR / "roy_pde_compensation_modal_routh_hurwitz_summary.csv"
FIG72_PATH = FIG_DIR / "fig72_pde_modal_routh_hurwitz_margins.png"

SCAN_FIELDS = [
    "stress",
    "lambda_value",
    "n_star",
    "w_star",
    "q_star",
    "A1",
    "A2",
    "A3",
    "A1A2_minus_A3",
    "minimum_rh_margin",
    "routh_hurwitz_stable",
    "max_real_growth",
    "eigenvalues_real",
    "eigenvalues_imag",
    "eigenvalue_stable",
    "lambda_range_note",
]

SUMMARY_FIELDS = [
    "stress",
    "lambda_min",
    "lambda_max",
    "lambda_points",
    "mode_range_i_j",
    "min_A1",
    "min_A2",
    "min_A3",
    "min_A1A2_minus_A3",
    "min_modal_rh_margin",
    "max_real_growth",
    "lambda_at_max_growth",
    "all_lambda_rh_stable",
    "all_lambda_eigenvalue_stable",
    "rh_eigenvalue_disagreement_count",
    "notes",
]


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
    """Coefficients of det(lambda I - J)=lambda^3+A1 lambda^2+A2 lambda+A3."""
    matrix = np.asarray(jacobian, dtype=float)
    trace = float(np.trace(matrix))
    trace_j2 = float(np.trace(matrix @ matrix))
    determinant = float(np.linalg.det(matrix))
    return {
        "A1": -trace,
        "A2": 0.5 * (trace * trace - trace_j2),
        "A3": -determinant,
    }


def lambda_max_for_mode_range() -> float:
    return (MODE_MAX_TESTED * math.pi / L_X) ** 2 + (MODE_MAX_TESTED * math.pi / L_Y) ** 2


def branch_state_at(stress: float, params: RoyEvoParams = PARAMS) -> tuple[float, float, float]:
    n, w, q = STEP23.branch_state(params, stress)
    return float(n), float(w), float(q)


def modal_matrix(stress: float, lambda_value: float, params: RoyEvoParams = PARAMS) -> tuple[np.ndarray, tuple[float, float, float]]:
    n, w, q = branch_state_at(stress, params)
    jacobian = np.asarray(STEP23.ode_rhs_jacobian(n, w, q, stress, params), dtype=float)
    diffusion = np.diag([D_N, D_W, D_Q])
    return jacobian - float(lambda_value) * diffusion, (n, w, q)


def modal_rh_row(stress: float, lambda_value: float) -> dict[str, Any]:
    matrix, state = modal_matrix(stress, lambda_value)
    n, w, q = state
    coeffs = characteristic_coefficients(matrix)
    margin = coeffs["A1"] * coeffs["A2"] - coeffs["A3"]
    min_margin = min(coeffs["A1"], coeffs["A2"], coeffs["A3"], margin)
    eigenvalues = np.linalg.eigvals(matrix)
    max_real = float(np.max(np.real(eigenvalues)))
    real_text, imag_text = eigen_text(eigenvalues)
    rh_stable = bool(
        coeffs["A1"] > STABILITY_TOL
        and coeffs["A2"] > STABILITY_TOL
        and coeffs["A3"] > STABILITY_TOL
        and margin > STABILITY_TOL
    )
    eigen_stable = bool(max_real < -STABILITY_TOL)
    return {
        "stress": stress,
        "lambda_value": float(lambda_value),
        "n_star": n,
        "w_star": w,
        "q_star": q,
        "A1": coeffs["A1"],
        "A2": coeffs["A2"],
        "A3": coeffs["A3"],
        "A1A2_minus_A3": margin,
        "minimum_rh_margin": min_margin,
        "routh_hurwitz_stable": rh_stable,
        "max_real_growth": max_real,
        "eigenvalues_real": real_text,
        "eigenvalues_imag": imag_text,
        "eigenvalue_stable": eigen_stable,
        "lambda_range_note": f"dense sampled lambda grid over mode-equivalent range i,j=0..{MODE_MAX_TESTED}",
    }


def run_modal_routh_hurwitz_scan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lambda_values = np.linspace(0.0, lambda_max_for_mode_range(), LAMBDA_GRID_POINTS)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for stress in TARGET_STRESSES:
        stress_rows = [modal_rh_row(stress, lambda_value) for lambda_value in lambda_values]
        rows.extend(stress_rows)

        max_growth_row = max(stress_rows, key=lambda row: float(row["max_real_growth"]))
        disagreement_count = sum(
            bool(row["routh_hurwitz_stable"]) != bool(row["eigenvalue_stable"])
            for row in stress_rows
        )
        summary_rows.append(
            {
                "stress": stress,
                "lambda_min": float(lambda_values[0]),
                "lambda_max": float(lambda_values[-1]),
                "lambda_points": len(lambda_values),
                "mode_range_i_j": f"0..{MODE_MAX_TESTED}",
                "min_A1": min(float(row["A1"]) for row in stress_rows),
                "min_A2": min(float(row["A2"]) for row in stress_rows),
                "min_A3": min(float(row["A3"]) for row in stress_rows),
                "min_A1A2_minus_A3": min(float(row["A1A2_minus_A3"]) for row in stress_rows),
                "min_modal_rh_margin": min(float(row["minimum_rh_margin"]) for row in stress_rows),
                "max_real_growth": float(max_growth_row["max_real_growth"]),
                "lambda_at_max_growth": float(max_growth_row["lambda_value"]),
                "all_lambda_rh_stable": all(bool(row["routh_hurwitz_stable"]) for row in stress_rows),
                "all_lambda_eigenvalue_stable": all(bool(row["eigenvalue_stable"]) for row in stress_rows),
                "rh_eigenvalue_disagreement_count": disagreement_count,
                "notes": "sampled modal RH margins support stability over tested lambda range",
            }
        )

    write_csv(MODAL_RH_SCAN_CSV, rows, SCAN_FIELDS)
    write_csv(MODAL_RH_SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    make_modal_rh_figure(rows, summary_rows)
    return rows, summary_rows


def make_modal_rh_figure(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    colors = {
        0.11765625: "#4c78a8",
        0.1584375: "#f58518",
        0.16486816: "#54a24b",
        0.175: "#b279a2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True)
    ax_margin, ax_growth = axes
    for stress in TARGET_STRESSES:
        subset = [row for row in rows if abs(float(row["stress"]) - stress) < 1.0e-12]
        x = np.asarray([float(row["lambda_value"]) for row in subset])
        margin = np.asarray([float(row["minimum_rh_margin"]) for row in subset])
        growth = np.asarray([float(row["max_real_growth"]) for row in subset])
        label = f"s={stress:g}"
        ax_margin.plot(x, margin, lw=1.8, label=label, color=colors[stress])
        ax_growth.plot(x, growth, lw=1.8, label=label, color=colors[stress])

    for ax in axes:
        ax.axhline(0.0, color="#111111", lw=1.0, ls="--")
        ax.grid(alpha=0.24)
        ax.set_xlabel(r"continuous modal value $\lambda$")
    ax_margin.set_title("A. Minimum modal Routh-Hurwitz margin", loc="left", fontweight="bold")
    ax_margin.set_ylabel(r"min$(A_1,A_2,A_3,A_1A_2-A_3)$")
    ax_margin.set_yscale("symlog", linthresh=1.0e-6)
    ax_growth.set_title("B. Maximum real modal eigenvalue", loc="left", fontweight="bold")
    ax_growth.set_ylabel(r"max Re eig$(J_F(U^*)-\lambda D)$")
    ax_growth.legend(frameon=False, fontsize=8, loc="best")

    min_margin = min(float(row["min_modal_rh_margin"]) for row in summary_rows)
    max_growth = max(float(row["max_real_growth"]) for row in summary_rows)
    fig.suptitle(
        f"Dense sampled modal stability margins: min RH margin={min_margin:.3g}, max growth={max_growth:.3g}",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    FIG72_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG72_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused"], default="focused")
    parser.parse_args()

    rows, summary_rows = run_modal_routh_hurwitz_scan()
    stable_count = sum(str(row["all_lambda_rh_stable"]).lower() == "true" for row in summary_rows)
    print(f"Wrote {MODAL_RH_SCAN_CSV.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"Wrote {MODAL_RH_SUMMARY_CSV.relative_to(ROOT)} ({len(summary_rows)} rows; stable stresses={stable_count})")
    print(f"Wrote {FIG72_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
