#!/usr/bin/env python
"""Audit manuscript-reported values and reference DOI fields.

This script is intentionally read-only with respect to model outputs: it reads
existing CSV and BibTeX files, writes audit CSVs, and exits nonzero if a key
reported numerical value no longer matches the source outputs within the
documented tolerance. It does not run simulations or alter model equations.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MANUSCRIPT_DIR = ROOT / "manuscript"

VALUE_AUDIT_CSV = RESULTS_DIR / "roy_compensation_branch_value_audit.csv"
DOI_AUDIT_CSV = RESULTS_DIR / "roy_compensation_branch_reference_doi_audit.csv"
NONLINEAR_FAILURE_MODES_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_failure_modes.csv"

BRANCH_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_branch_current.csv"
COUNTERFACTUAL_SUMMARY_CSV = RESULTS_DIR / "roy_ode_no_evolution_counterfactual_summary.csv"
INTERVAL_CURRENT_CSV = RESULTS_DIR / "roy_ode_compensation_interval_current.csv"
NONLINEAR_SHAPE_SUMMARY_CSV = RESULTS_DIR / "roy_nonlinear_tradeoff_shape_summary.csv"
LAMBDA_SUMMARY_CSV = RESULTS_DIR / "roy_pde_compensation_lambda_scan_summary.csv"
MODAL_RH_SUMMARY_CSV = RESULTS_DIR / "roy_pde_compensation_modal_routh_hurwitz_summary.csv"
SPATIAL_MODES_CSV = RESULTS_DIR / "roy_pde_compensation_spatial_modes_current.csv"
REFERENCES_BIB = MANUSCRIPT_DIR / "references.bib"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def metric_dict(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in read_csv(path)}


def pass_numeric(value: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(value) and abs(value - expected) <= tolerance


def audit_row(item: str, value: Any, expected: str, tolerance: str, status: bool, source: str, notes: str = "") -> dict[str, Any]:
    return {
        "audit_item": item,
        "reported_value": value,
        "expected_or_target": expected,
        "tolerance": tolerance,
        "status": "PASS" if status else "FAIL",
        "source": source,
        "notes": notes,
    }


def run_value_audit() -> list[dict[str, Any]]:
    branch_rows = read_csv(BRANCH_CURRENT_CSV)
    branch_s0 = min(branch_rows, key=lambda row: abs(float(row["stress"])))
    summary = metric_dict(COUNTERFACTUAL_SUMMARY_CSV)
    interval_rows = read_csv(INTERVAL_CURRENT_CSV)
    interval = interval_rows[0]
    shape_counts = Counter(row["shape_class"] for row in read_csv(NONLINEAR_SHAPE_SUMMARY_CSV))
    lambda_rows = read_csv(LAMBDA_SUMMARY_CSV)
    modal_rh_rows = read_csv(MODAL_RH_SUMMARY_CSV) if MODAL_RH_SUMMARY_CSV.exists() else []
    mode_rows = read_csv(SPATIAL_MODES_CSV)

    rows: list[dict[str, Any]] = []
    checks = [
        (
            "q_star_at_s0",
            float(branch_s0["q_star_analytic"]),
            0.6726,
            5.0e-5,
            "results/roy_ode_compensation_branch_current.csv",
            "Manuscript reports q*(0) approximately 0.6726.",
        ),
        (
            "n_star_branch",
            float(branch_s0["n_star_analytic"]),
            4.8333,
            5.0e-5,
            "results/roy_ode_compensation_branch_current.csv",
            "Manuscript reports n* approximately 4.8333.",
        ),
        (
            "w_star_branch",
            float(branch_s0["w_star_analytic"]),
            0.6417,
            5.0e-5,
            "results/roy_ode_compensation_branch_current.csv",
            "Manuscript reports w* approximately 0.6417.",
        ),
        (
            "fixed_q_invasion_threshold",
            float(summary["fixed_q_invasion_threshold"]),
            0.0696,
            5.0e-5,
            "results/roy_ode_no_evolution_counterfactual_summary.csv",
            "Manuscript reports s_fixed approximately 0.0696.",
        ),
        (
            "rescue_window_low_grid",
            float(summary["rescue_window_low_grid"]),
            0.0726,
            5.0e-5,
            "results/roy_ode_no_evolution_counterfactual_summary.csv",
            "Manuscript reports verified rescue window lower endpoint approximately 0.0726.",
        ),
        (
            "rescue_window_high_grid",
            float(summary["rescue_window_high_grid"]),
            0.1656,
            5.0e-5,
            "results/roy_ode_no_evolution_counterfactual_summary.csv",
            "Manuscript reports verified rescue window upper endpoint approximately 0.1656.",
        ),
        (
            "branch_interval_low",
            float(interval["s_at_q_equals_1"]),
            -0.1131,
            5.0e-5,
            "results/roy_ode_compensation_interval_current.csv",
            "Manuscript reports branch interval lower endpoint approximately -0.1131.",
        ),
        (
            "branch_interval_high",
            float(interval["s_at_q_equals_0"]),
            0.2324,
            5.0e-5,
            "results/roy_ode_compensation_interval_current.csv",
            "Manuscript reports branch interval upper endpoint approximately 0.2324.",
        ),
    ]
    for item, value, expected, tolerance, source, notes in checks:
        rows.append(audit_row(item, value, f"approximately {expected:g}", tolerance, pass_numeric(value, expected, tolerance), source, notes))

    count_targets = {
        "robust_compensation_shape": 11,
        "partial_compensation_shape": 11,
        "no_compensation_shape": 4,
        "unresolved_shape": 1,
    }
    for shape_class, expected_count in count_targets.items():
        value = int(shape_counts.get(shape_class, 0))
        rows.append(
            audit_row(
                f"nonlinear_count_{shape_class}",
                value,
                str(expected_count),
                "exact",
                value == expected_count,
                "results/roy_nonlinear_tradeoff_shape_summary.csv",
                "Nonlinear shape-grid class count reported in manuscript.",
            )
        )

    lambda_points = {int(float(row["lambda_points"])) for row in lambda_rows}
    mode_ranges = {row["mode_range_i_j"] for row in lambda_rows}
    max_m = max(int(float(row["m"])) for row in mode_rows)
    max_n = max(int(float(row["n"])) for row in mode_rows)
    rows.append(
        audit_row(
            "lambda_scan_points",
            ";".join(str(value) for value in sorted(lambda_points)),
            "600",
            "exact",
            lambda_points == {600},
            "results/roy_pde_compensation_lambda_scan_summary.csv",
            "Sampled continuous-lambda scan point count.",
        )
    )
    rows.append(
        audit_row(
            "lambda_scan_mode_range",
            ";".join(sorted(mode_ranges)),
            "0..64",
            "exact",
            mode_ranges == {"0..64"},
            "results/roy_pde_compensation_lambda_scan_summary.csv",
            "Mode-equivalent lambda range used by sampled continuous-lambda scan.",
        )
    )
    rows.append(
        audit_row(
            "discrete_neumann_mode_range",
            f"m=0..{max_m};n=0..{max_n}",
            "m=0..64;n=0..64",
            "exact",
            max_m == 64 and max_n == 64,
            "results/roy_pde_compensation_spatial_modes_current.csv",
            "Discrete Neumann mode range reported in manuscript.",
        )
    )
    if modal_rh_rows:
        modal_lambda_points = {int(float(row["lambda_points"])) for row in modal_rh_rows}
        modal_mode_ranges = {row["mode_range_i_j"] for row in modal_rh_rows}
        all_modal_rh_stable = all(str(row["all_lambda_rh_stable"]).strip().lower() == "true" for row in modal_rh_rows)
        all_modal_eig_stable = all(str(row["all_lambda_eigenvalue_stable"]).strip().lower() == "true" for row in modal_rh_rows)
        modal_disagreements = sum(int(float(row["rh_eigenvalue_disagreement_count"])) for row in modal_rh_rows)
        min_modal_margin = min(float(row["min_modal_rh_margin"]) for row in modal_rh_rows)
        rows.append(
            audit_row(
                "modal_routh_hurwitz_lambda_points",
                ";".join(str(value) for value in sorted(modal_lambda_points)),
                "2001",
                "exact",
                modal_lambda_points == {2001},
                "results/roy_pde_compensation_modal_routh_hurwitz_summary.csv",
                "Dense modal Routh-Hurwitz lambda grid size reported in supplement.",
            )
        )
        rows.append(
            audit_row(
                "modal_routh_hurwitz_mode_range",
                ";".join(sorted(modal_mode_ranges)),
                "0..64",
                "exact",
                modal_mode_ranges == {"0..64"},
                "results/roy_pde_compensation_modal_routh_hurwitz_summary.csv",
                "Mode-equivalent lambda range used by dense modal Routh-Hurwitz diagnostic.",
            )
        )
        rows.append(
            audit_row(
                "modal_routh_hurwitz_stability",
                f"all_rh={all_modal_rh_stable};all_eig={all_modal_eig_stable};disagreements={modal_disagreements};min_margin={min_modal_margin:.6g}",
                "all stable, zero RH/eigen disagreements, positive margin",
                "logical",
                all_modal_rh_stable and all_modal_eig_stable and modal_disagreements == 0 and min_modal_margin > 0.0,
                "results/roy_pde_compensation_modal_routh_hurwitz_summary.csv",
                "Dense sampled modal Routh-Hurwitz diagnostic reported as sampled evidence, not proof.",
            )
        )
    write_csv(
        VALUE_AUDIT_CSV,
        rows,
        ["audit_item", "reported_value", "expected_or_target", "tolerance", "status", "source", "notes"],
    )
    return rows


def classify_nonlinear_failure_mode(row: dict[str, str]) -> tuple[str, str]:
    shape_class = row["shape_class"]
    branches = int(float(row["target_stresses_with_branch"]))
    stable = int(float(row["target_stresses_stable"]))
    interval_low = float(row["stress_interval_low_estimate"])
    interval_high = float(row["stress_interval_high_estimate"])

    if shape_class == "no_compensation_shape" and branches == 0:
        return (
            "no_target_stress_feasible_branch",
            "No feasible branch was found at the four target stresses; the estimated interval does not cover the tested rescue stresses.",
        )
    if branches > 0 and stable == 0:
        return (
            "branch_found_but_not_stable_at_targets",
            "A branch was detected for some target stresses, but no target stress was classified as stable in the existing diagnostics.",
        )
    if interval_high <= interval_low or not (math.isfinite(interval_low) and math.isfinite(interval_high)):
        return (
            "infeasible_or_unresolved_interval",
            "The existing interval estimate is non-finite or not ordered, so compensation cannot be assigned from the current diagnostics.",
        )
    return (
        "unresolved_classification",
        "Existing shape-grid diagnostics do not cleanly separate feasibility, stability, and basin outcome for this case.",
    )


def run_nonlinear_failure_mode_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(NONLINEAR_SHAPE_SUMMARY_CSV):
        if row["shape_class"] not in {"no_compensation_shape", "unresolved_shape"}:
            continue
        reason, interpretation = classify_nonlinear_failure_mode(row)
        rows.append(
            {
                "gamma_r": row["gamma_r"],
                "gamma_a": row["gamma_a"],
                "gamma_b": row["gamma_b"],
                "shape_class": row["shape_class"],
                "target_stresses_total": row["target_stresses_total"],
                "target_stresses_with_branch": row["target_stresses_with_branch"],
                "target_stresses_stable": row["target_stresses_stable"],
                "stress_interval_low_estimate": row["stress_interval_low_estimate"],
                "stress_interval_high_estimate": row["stress_interval_high_estimate"],
                "failure_or_unresolved_reason": reason,
                "interpretation": interpretation,
            }
        )
    write_csv(
        NONLINEAR_FAILURE_MODES_CSV,
        rows,
        [
            "gamma_r",
            "gamma_a",
            "gamma_b",
            "shape_class",
            "target_stresses_total",
            "target_stresses_with_branch",
            "target_stresses_stable",
            "stress_interval_low_estimate",
            "stress_interval_high_estimate",
            "failure_or_unresolved_reason",
            "interpretation",
        ],
    )
    return rows


def parse_bib_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for match in re.finditer(r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,]+),(?P<body>.*?)(?=\n@|\Z)", text, re.DOTALL):
        body = match.group("body")
        doi_match = re.search(r"\bdoi\s*=\s*\{(?P<doi>[^}]+)\}", body, re.IGNORECASE)
        entries.append(
            {
                "entry_type": match.group("type"),
                "key": match.group("key").strip(),
                "doi": doi_match.group("doi").strip() if doi_match else "",
            }
        )
    return entries


def run_doi_audit() -> list[dict[str, Any]]:
    text = REFERENCES_BIB.read_text(encoding="utf-8")
    entries = parse_bib_entries(text)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        doi_present = bool(entry["doi"])
        if entry["key"] == "RoyEtAl2026" and entry["doi"] == "10.1038/s42005-025-02434-1":
            resolver_status = "manual_article_record_check"
            manual_check_required = False
            notes = (
                "Manual article PDF/page record check recorded: Communications Physics 9, Article 4 (2026), "
                "DOI 10.1038/s42005-025-02434-1. The script checks fields and does not perform live resolver validation."
            )
        elif doi_present:
            resolver_status = "doi_field_present_not_resolved_by_script"
            manual_check_required = True
            notes = "DOI field present; resolver validation should still be checked before journal submission."
        else:
            resolver_status = "missing_doi"
            manual_check_required = True
            notes = "Missing DOI field; verify manually before journal submission."
        rows.append(
            {
                "key": entry["key"],
                "entry_type": entry["entry_type"],
                "doi_present": doi_present,
                "doi": entry["doi"],
                "resolver_status": resolver_status,
                "manual_check_required": manual_check_required,
                "notes": notes,
            }
        )
    write_csv(DOI_AUDIT_CSV, rows, ["key", "entry_type", "doi_present", "doi", "resolver_status", "manual_check_required", "notes"])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focused",), default="focused")
    return parser.parse_args()


def main() -> None:
    parse_args()
    value_rows = run_value_audit()
    failure_rows = run_nonlinear_failure_mode_audit()
    doi_rows = run_doi_audit()
    failed = [row for row in value_rows if row["status"] != "PASS"]
    missing_doi = [row for row in doi_rows if not row["doi_present"]]
    print(f"Wrote {VALUE_AUDIT_CSV.relative_to(ROOT)} with {len(value_rows)} checks.")
    print(f"Wrote {NONLINEAR_FAILURE_MODES_CSV.relative_to(ROOT)} with {len(failure_rows)} nonlinear failure/unresolved rows.")
    print(f"Wrote {DOI_AUDIT_CSV.relative_to(ROOT)} with {len(doi_rows)} references.")
    if missing_doi:
        print(f"References missing DOI fields: {len(missing_doi)}")
    if failed:
        print("Failed value checks:")
        for row in failed:
            print(f"  - {row['audit_item']}: {row['reported_value']} expected {row['expected_or_target']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
