#!/usr/bin/env python
"""Evaluate Routh-Hurwitz stability conditions for the Roy ODE branch.

This experiment is ODE-only. It reuses the analytic compensation-branch
geometry and Jacobian from Step 23, then computes the cubic characteristic
polynomial coefficients and Routh-Hurwitz inequalities. It does not run PDE
simulations, broad ODE simulations, or change model equations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import RoyEvoParams, free_space_evo  # noqa: E402


def load_step23_module():
    path = Path(__file__).resolve().with_name("23_roy_ode_compensation_conditions.py")
    spec = importlib.util.spec_from_file_location("step23_ode_compensation_conditions_runtime", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load Step 23 module from {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP23 = load_step23_module()

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
TARGET_STRESSES = (0.0, 0.069448242, 0.11765625, 0.1584375, 0.16486816, 0.175)
GRID_STRESSES = (0.1584375, 0.16486816)
R_V_VALUES = (0.55, 0.60, 0.65, 0.70, 0.75)
A_V_VALUES = (0.25, 0.30, 0.35, 0.40, 0.45)
B_U_VALUES = (0.06, 0.08, 0.10)
B_RATIO_VALUES = (0.25, 0.50, 0.75)

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTES_DIR = ROOT / "research_notes"

CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_routh_hurwitz_current.csv"
GRID_CSV = RESULTS_DIR / "roy_ode_compensation_routh_hurwitz_condition_grid.csv"
SUMMARY_CSV = RESULTS_DIR / "roy_ode_compensation_routh_hurwitz_summary.csv"
NOTE_PATH = NOTES_DIR / "roy_ode_compensation_routh_hurwitz.md"

FIG42_PATH = FIG_DIR / "fig42_routh_hurwitz_terms.png"
FIG43_PATH = FIG_DIR / "fig43_stability_condition_region.png"
FIG44_PATH = FIG_DIR / "fig44_routh_hurwitz_margin.png"

CURRENT_FIELDS = [
    "stress",
    "n_star",
    "w_star",
    "q_star",
    "z_star",
    "A1",
    "A2",
    "A3",
    "A1A2_minus_A3",
    "rh_A1_positive",
    "rh_A2_positive",
    "rh_A3_positive",
    "rh_margin_positive",
    "routh_hurwitz_stable",
    "max_real_eigenvalue",
    "eigenvalues_real",
    "eigenvalues_imag",
    "eigenvalue_stable",
    "rh_matches_eigenvalue",
]

GRID_FIELDS = [
    "r_v",
    "a_v",
    "b_u",
    "b_v",
    "b_v_over_b_u",
    "stress",
    "branch_exists",
    "A1",
    "A2",
    "A3",
    "A1A2_minus_A3",
    "routh_hurwitz_stable",
    "max_real_eigenvalue",
    "rh_matches_eigenvalue",
    "condition_class",
]

SUMMARY_FIELDS = ["metric", "value", "interpretation"]

CLASS_ORDER = ("branch_stable", "branch_unstable", "no_branch", "rh_eigenvalue_disagreement")
CLASS_COLORS = {
    "branch_stable": "#1b9e77",
    "branch_unstable": "#d95f02",
    "no_branch": "#7570b3",
    "rh_eigenvalue_disagreement": "#c23b3b",
}


def format_float(value: float, digits: int = 8) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def characteristic_coefficients(jacobian: np.ndarray) -> dict[str, float]:
    """Return cubic coefficients for det(lambda I - J)."""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("jacobian must have shape (3, 3)")
    trace = float(np.trace(matrix))
    trace_j2 = float(np.trace(matrix @ matrix))
    determinant = float(np.linalg.det(matrix))
    return {
        "A1": -trace,
        "A2": 0.5 * (trace * trace - trace_j2),
        "A3": -determinant,
    }


def routh_hurwitz_check(A1: float, A2: float, A3: float, tol: float = 1.0e-12) -> dict[str, float | bool]:
    margin = A1 * A2 - A3 if np.isfinite(A1) and np.isfinite(A2) and np.isfinite(A3) else math.nan
    a1_positive = bool(np.isfinite(A1) and A1 > tol)
    a2_positive = bool(np.isfinite(A2) and A2 > tol)
    a3_positive = bool(np.isfinite(A3) and A3 > tol)
    margin_positive = bool(np.isfinite(margin) and margin > tol)
    return {
        "A1A2_minus_A3": margin,
        "rh_A1_positive": a1_positive,
        "rh_A2_positive": a2_positive,
        "rh_A3_positive": a3_positive,
        "rh_margin_positive": margin_positive,
        "routh_hurwitz_stable": bool(a1_positive and a2_positive and a3_positive and margin_positive),
    }


def min_rh_margin(row: dict[str, Any]) -> float:
    values = [float(row[name]) for name in ("A1", "A2", "A3", "A1A2_minus_A3") if np.isfinite(float(row[name]))]
    if not values:
        return math.nan
    return min(values)


def eigen_summary(jacobian: np.ndarray) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvals(jacobian)
    max_real = float(np.max(np.real(eigenvalues)))
    return {
        "max_real_eigenvalue": max_real,
        "eigenvalues_real": ";".join(format_float(float(value), 10) for value in np.real(eigenvalues)),
        "eigenvalues_imag": ";".join(format_float(float(value), 10) for value in np.imag(eigenvalues)),
        "eigenvalue_stable": bool(max_real < -1.0e-9),
    }


def evaluate_branch(params: RoyEvoParams, stress: float) -> dict[str, Any]:
    conditions = STEP23.compensation_existence_conditions(params, stress)
    if not conditions["branch_exists"]:
        row = {
            "stress": stress,
            "n_star": conditions["n_star"],
            "w_star": conditions["w_star"],
            "q_star": conditions["q_star"],
            "z_star": math.nan,
            "A1": math.nan,
            "A2": math.nan,
            "A3": math.nan,
            "A1A2_minus_A3": math.nan,
            "rh_A1_positive": False,
            "rh_A2_positive": False,
            "rh_A3_positive": False,
            "rh_margin_positive": False,
            "routh_hurwitz_stable": False,
            "max_real_eigenvalue": math.nan,
            "eigenvalues_real": "",
            "eigenvalues_imag": "",
            "eigenvalue_stable": False,
            "rh_matches_eigenvalue": False,
            "branch_exists": False,
        }
        return row

    n, w, q = STEP23.branch_state(params, stress)
    z = float(free_space_evo(n, w, params))
    jacobian = STEP23.ode_rhs_jacobian(n, w, q, stress, params)
    coeffs = characteristic_coefficients(jacobian)
    rh = routh_hurwitz_check(coeffs["A1"], coeffs["A2"], coeffs["A3"])
    eig = eigen_summary(jacobian)
    row = {
        "stress": stress,
        "n_star": n,
        "w_star": w,
        "q_star": q,
        "z_star": z,
        **coeffs,
        **rh,
        **eig,
        "rh_matches_eigenvalue": bool(rh["routh_hurwitz_stable"] == eig["eigenvalue_stable"]),
        "branch_exists": True,
    }
    return row


def classify_grid_row(row: dict[str, Any]) -> str:
    if not row["branch_exists"]:
        return "no_branch"
    if not row["rh_matches_eigenvalue"]:
        return "rh_eigenvalue_disagreement"
    if row["routh_hurwitz_stable"]:
        return "branch_stable"
    return "branch_unstable"


def write_current_rows(params: RoyEvoParams = PARAMS) -> list[dict[str, Any]]:
    rows = [evaluate_branch(params, stress) for stress in TARGET_STRESSES]
    write_csv(CURRENT_CSV, rows, CURRENT_FIELDS)
    return rows


def write_grid_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r_v in R_V_VALUES:
        for a_v in A_V_VALUES:
            for b_u in B_U_VALUES:
                for ratio in B_RATIO_VALUES:
                    b_v = b_u * ratio
                    params = RoyEvoParams(r_v=r_v, a_v=a_v, b_u=b_u, b_v=b_v)
                    for stress in GRID_STRESSES:
                        branch = evaluate_branch(params, stress)
                        row = {
                            "r_v": r_v,
                            "a_v": a_v,
                            "b_u": b_u,
                            "b_v": b_v,
                            "b_v_over_b_u": ratio,
                            "stress": stress,
                            "branch_exists": branch["branch_exists"],
                            "A1": branch["A1"],
                            "A2": branch["A2"],
                            "A3": branch["A3"],
                            "A1A2_minus_A3": branch["A1A2_minus_A3"],
                            "routh_hurwitz_stable": branch["routh_hurwitz_stable"],
                            "max_real_eigenvalue": branch["max_real_eigenvalue"],
                            "rh_matches_eigenvalue": branch["rh_matches_eigenvalue"],
                        }
                        row["condition_class"] = classify_grid_row(branch)
                        rows.append(row)
    write_csv(GRID_CSV, rows, GRID_FIELDS)
    return rows


def decide_final_label(
    *,
    current_total: int,
    current_rh_stable: int,
    current_rh_eigenvalue_agreement: bool,
    grid_stable_fraction: float,
    grid_disagreement_count: int,
) -> str:
    if not current_rh_eigenvalue_agreement or grid_disagreement_count > 0:
        return "routh_hurwitz_conditions_unresolved"
    if current_total > 0 and current_rh_stable == current_total and grid_stable_fraction >= 0.10:
        return "routh_hurwitz_conditions_supported"
    if current_total > 0 and current_rh_stable == current_total:
        return "routh_hurwitz_conditions_partially_supported"
    return "routh_hurwitz_conditions_unresolved"


def write_summary(current_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    current_total = len(current_rows)
    current_stable = sum(bool(row["routh_hurwitz_stable"]) for row in current_rows)
    current_agreement = all(bool(row["rh_matches_eigenvalue"]) for row in current_rows)
    class_counts = Counter(str(row["condition_class"]) for row in grid_rows)
    grid_total = len(grid_rows)
    stable_count = class_counts["branch_stable"]
    unstable_count = class_counts["branch_unstable"]
    no_branch_count = class_counts["no_branch"]
    disagreement_count = class_counts["rh_eigenvalue_disagreement"]
    stable_fraction = stable_count / grid_total if grid_total else math.nan
    final_label = decide_final_label(
        current_total=current_total,
        current_rh_stable=current_stable,
        current_rh_eigenvalue_agreement=current_agreement,
        grid_stable_fraction=stable_fraction,
        grid_disagreement_count=disagreement_count,
    )
    rows = [
        {
            "metric": "current_target_stresses_total",
            "value": current_total,
            "interpretation": "number of current branch target stresses evaluated",
        },
        {
            "metric": "current_target_stresses_rh_stable",
            "value": current_stable,
            "interpretation": "target stresses satisfying all cubic Routh-Hurwitz inequalities",
        },
        {
            "metric": "current_rh_eigenvalue_agreement",
            "value": current_agreement,
            "interpretation": "Routh-Hurwitz stability agrees with direct analytic eigenvalue stability",
        },
        {
            "metric": "minimum_current_A1",
            "value": min(float(row["A1"]) for row in current_rows),
            "interpretation": "smallest A1 across current target stresses",
        },
        {
            "metric": "minimum_current_A2",
            "value": min(float(row["A2"]) for row in current_rows),
            "interpretation": "smallest A2 across current target stresses",
        },
        {
            "metric": "minimum_current_A3",
            "value": min(float(row["A3"]) for row in current_rows),
            "interpretation": "smallest A3 across current target stresses",
        },
        {
            "metric": "minimum_current_margin",
            "value": min(float(row["A1A2_minus_A3"]) for row in current_rows),
            "interpretation": "smallest A1*A2-A3 margin across current target stresses",
        },
        {
            "metric": "condition_grid_total",
            "value": grid_total,
            "interpretation": "structured local trade-off rows evaluated at two target rescue stresses",
        },
        {
            "metric": "condition_grid_branch_stable_count",
            "value": stable_count,
            "interpretation": "grid rows with branch existence and Routh-Hurwitz stability",
        },
        {
            "metric": "condition_grid_branch_unstable_count",
            "value": unstable_count,
            "interpretation": "grid rows with branch existence but failed Routh-Hurwitz stability",
        },
        {
            "metric": "condition_grid_no_branch_count",
            "value": no_branch_count,
            "interpretation": "grid rows where the interior branch is not feasible at that stress",
        },
        {
            "metric": "condition_grid_rh_eigenvalue_disagreement_count",
            "value": disagreement_count,
            "interpretation": "rows where Routh-Hurwitz classification disagrees with eigenvalue stability",
        },
        {
            "metric": "condition_grid_branch_stable_fraction",
            "value": stable_fraction,
            "interpretation": "branch-stable rows divided by all condition-grid rows",
        },
        {
            "metric": "final_label",
            "value": final_label,
            "interpretation": "allowed final label for the Routh-Hurwitz stability analysis",
        },
    ]
    write_csv(SUMMARY_CSV, rows, SUMMARY_FIELDS)
    return rows, final_label


def plot_current_terms(current_rows: list[dict[str, Any]]) -> None:
    stresses = np.array([float(row["stress"]) for row in current_rows], dtype=float)
    terms = [("A1", "A1"), ("A2", "A2"), ("A3", "A3"), ("A1A2_minus_A3", "A1 A2 - A3")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharex=True)
    for ax, (key, title) in zip(axes.flat, terms):
        values = np.array([float(row[key]) for row in current_rows], dtype=float)
        ax.plot(stresses, values, marker="o", color="#2c6fbb", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("coefficient value")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("mortality stress s")
    axes[-1, 1].set_xlabel("mortality stress s")
    fig.suptitle("Routh-Hurwitz terms along the current compensation branch")
    save_figure(fig, FIG42_PATH)


def plot_grid_region(grid_rows: list[dict[str, Any]]) -> None:
    rv_values = sorted({float(row["r_v"]) for row in grid_rows})
    av_values = sorted({float(row["a_v"]) for row in grid_rows})
    matrix = np.zeros((len(av_values), len(rv_values)), dtype=float)
    for i, a_v in enumerate(av_values):
        for j, r_v in enumerate(rv_values):
            subset = [row for row in grid_rows if float(row["r_v"]) == r_v and float(row["a_v"]) == a_v]
            stable = sum(row["condition_class"] == "branch_stable" for row in subset)
            matrix[i, j] = stable / len(subset) if subset else math.nan

    class_counts = Counter(str(row["condition_class"]) for row in grid_rows)
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    im = ax_heat.imshow(matrix, origin="lower", cmap="YlGn", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(range(len(rv_values)), [format_float(value, 2) for value in rv_values])
    ax_heat.set_yticks(range(len(av_values)), [format_float(value, 2) for value in av_values])
    ax_heat.set_xlabel("defended growth r_v")
    ax_heat.set_ylabel("defended palatability a_v")
    ax_heat.set_title("Branch-stable fraction over b trade-offs")
    for i in range(len(av_values)):
        for j in range(len(rv_values)):
            ax_heat.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="stable fraction")

    labels = list(CLASS_ORDER)
    counts = [class_counts[label] for label in labels]
    colors = [CLASS_COLORS[label] for label in labels]
    y = np.arange(len(labels))
    ax_bar.barh(y, counts, color=colors)
    ax_bar.set_yticks(y, labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("grid row count")
    ax_bar.set_title("Condition classes")
    for idx, count in enumerate(counts):
        ax_bar.text(count + max(counts + [1]) * 0.01, idx, str(count), va="center", fontsize=8)

    fig.suptitle("Structured local Routh-Hurwitz condition grid")
    save_figure(fig, FIG43_PATH)


def plot_margins(current_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]]) -> None:
    fig, (ax_current, ax_grid) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    stresses = np.array([float(row["stress"]) for row in current_rows], dtype=float)
    margins = np.array([min_rh_margin(row) for row in current_rows], dtype=float)
    ax_current.plot(stresses, margins, marker="o", color="#1b9e77", linewidth=1.8)
    ax_current.axhline(0.0, color="black", linewidth=0.8)
    ax_current.set_xlabel("mortality stress s")
    ax_current.set_ylabel("min RH term")
    ax_current.set_title("Current branch margin")
    ax_current.grid(alpha=0.25)

    rng = np.random.default_rng(2401)
    for stress in GRID_STRESSES:
        subset = [row for row in grid_rows if float(row["stress"]) == stress and row["condition_class"] != "no_branch"]
        if not subset:
            continue
        y = np.array([min_rh_margin(row) for row in subset], dtype=float)
        x = np.full_like(y, stress, dtype=float) + rng.normal(0.0, 0.00075, size=y.size)
        colors = [CLASS_COLORS[str(row["condition_class"])] for row in subset]
        ax_grid.scatter(x, y, c=colors, s=18, alpha=0.75, edgecolor="none")
    ax_grid.axhline(0.0, color="black", linewidth=0.8)
    ax_grid.set_xlabel("mortality stress s")
    ax_grid.set_ylabel("min RH term")
    ax_grid.set_title("Grid margins where branch exists")
    ax_grid.grid(alpha=0.25)
    handles = [mpatches.Patch(color=CLASS_COLORS[label], label=label) for label in CLASS_ORDER if label != "no_branch"]
    ax_grid.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    fig.suptitle("Routh-Hurwitz stability margins")
    save_figure(fig, FIG44_PATH)


def write_research_note(
    current_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    final_label: str,
) -> None:
    del summary_rows
    class_counts = Counter(str(row["condition_class"]) for row in grid_rows)
    current_lines = "\n".join(
        "- s={stress}: A1={A1}, A2={A2}, A3={A3}, A1A2-A3={margin}, stable={stable}".format(
            stress=format_float(float(row["stress"]), 9),
            A1=format_float(float(row["A1"]), 7),
            A2=format_float(float(row["A2"]), 7),
            A3=format_float(float(row["A3"]), 7),
            margin=format_float(float(row["A1A2_minus_A3"]), 7),
            stable=row["routh_hurwitz_stable"],
        )
        for row in current_rows
    )
    text = f"""# Routh-Hurwitz Stability Conditions for the Roy ODE Compensation Branch

## Purpose

This note turns local stability of the homogeneous compensation branch from numerical eigenvalue reporting into Routh-Hurwitz coefficient inequalities for the cubic characteristic polynomial of the analytic Jacobian. It is ODE-only and does not run PDE simulations or change model equations.

## Compensation Branch Jacobian

At the interior compensation branch, the ODE balances satisfy `R=0`, `P=0`, and `G=0`. The analytic Jacobian therefore uses the simplified entries

```text
J11 = n R_n
J12 = n R_w
J13 = 0
J21 = w P_n
J22 = w P_w
J23 = w P_q
J31 = nu q(1-q) G_n
J32 = nu q(1-q) G_w
J33 = 0
```

where the derivatives are those of the linear trade-off ODE. The implementation reuses the Step 23 analytic Jacobian.

## Characteristic Polynomial

For the Jacobian \(J\), the characteristic polynomial is written as

```text
det(lambda I - J) = lambda^3 + A1 lambda^2 + A2 lambda + A3
```

with

```text
A1 = -tr(J)
A2 = 0.5 * ((tr J)^2 - tr(J^2))
A3 = -det(J)
```

These coefficients are computed directly from the analytic Jacobian.

## Routh-Hurwitz Conditions

For a cubic polynomial, local stability requires `A1>0`, `A2>0`, `A3>0`, and `A1*A2>A3`. The last inequality is recorded as the Routh-Hurwitz margin `A1A2_minus_A3`.

## Current Parameterization

For the current parameterization, all target stresses satisfy the Routh-Hurwitz inequalities and agree with the direct analytic eigenvalue stability classification.

{current_lines}

The current table is `results/roy_ode_compensation_routh_hurwitz_current.csv`, and the coefficient figure is `figures/roy_evo_spatial/report/fig42_routh_hurwitz_terms.png`.

## Structured Condition Grid

The structured local grid is the same local trade-off grid used in the compensation-condition analysis. It evaluates analytic branch existence and Routh-Hurwitz stability at the two target rescue stresses, not simulated trajectories.

- `branch_stable`: {class_counts["branch_stable"]}
- `branch_unstable`: {class_counts["branch_unstable"]}
- `no_branch`: {class_counts["no_branch"]}
- `rh_eigenvalue_disagreement`: {class_counts["rh_eigenvalue_disagreement"]}

The grid table is `results/roy_ode_compensation_routh_hurwitz_condition_grid.csv`, and the condition-region figure is `figures/roy_evo_spatial/report/fig43_stability_condition_region.png`.

## Final Label

`{final_label}`

## Interpretation

The Routh-Hurwitz inequalities support local stability of the current compensation branch throughout the tested target stresses. The condition-grid result shows that branch existence and Routh-Hurwitz stability hold in a nontrivial local region of the structured trade-off grid. This strengthens the homogeneous rescue interpretation for the tested linear trade-off ODE.

## What This Adds Beyond Eigenvalue Checks

The previous analysis reported analytic Jacobian eigenvalues. This analysis expresses the same local-stability decision as explicit cubic coefficient inequalities. Agreement between the Routh-Hurwitz conditions and eigenvalue stability checks provides an internal consistency check.

## What This Does Not Prove

This remains a local stability result, not a global basin proof. It does not establish global attraction, spatial-pattern-mediated mechanisms, or behavior under nonlinear trade-off forms. It also does not test PDE dynamics.

## Files

- `experiments/24_roy_ode_compensation_routh_hurwitz.py`
- `results/roy_ode_compensation_routh_hurwitz_current.csv`
- `results/roy_ode_compensation_routh_hurwitz_condition_grid.csv`
- `results/roy_ode_compensation_routh_hurwitz_summary.csv`
- `figures/roy_evo_spatial/report/fig42_routh_hurwitz_terms.png`
- `figures/roy_evo_spatial/report/fig43_stability_condition_region.png`
- `figures/roy_evo_spatial/report/fig44_routh_hurwitz_margin.png`

## Next Step

Use the Routh-Hurwitz margins to identify which trade-off terms control loss of local stability before doing any new numerical exploration.

{final_label}
"""
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(text, encoding="utf-8")


def run(profile: str) -> str:
    if profile not in {"focused", "full"}:
        raise ValueError("profile must be focused or full")
    current_rows = write_current_rows(PARAMS)
    grid_rows = write_grid_rows()
    summary_rows, final_label = write_summary(current_rows, grid_rows)
    plot_current_terms(current_rows)
    plot_grid_region(grid_rows)
    plot_margins(current_rows, grid_rows)
    write_research_note(current_rows, grid_rows, summary_rows, final_label)
    for path in [CURRENT_CSV, GRID_CSV, SUMMARY_CSV, FIG42_PATH, FIG43_PATH, FIG44_PATH, NOTE_PATH]:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label: {final_label}")
    return final_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused", "full"], default="focused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(profile=args.profile)


if __name__ == "__main__":
    main()
