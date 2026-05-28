"""Compute nonlinear thresholds for Turing-unstable candidates.

The script is conservative about interpretation.  A positive mean-density-normalized
Delta m_c is not called pattern-mediated rescue unless the candidate is Turing-unstable
and the nonlinear PDE simulation develops measurable spatial variance.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate_pde_1d import (
    compute_mc_ode,
    compute_mc_pde,
    default_threshold_initial_state,
    homogeneous_profile,
    simulate_pde,
)
from src.turing_rescue_model import CoexistenceEquilibrium, RescueParams


RESULTS_DIR = ROOT / "results"
CANDIDATES_CSV = RESULTS_DIR / "turing_candidates.csv"
OUTPUT_CSV = RESULTS_DIR / "threshold_scan.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_02.md"


FIELDNAMES = [
    "candidate_id",
    "m_candidate",
    "dominant_mode",
    "dominant_growth",
    "m_c_ode",
    "m_c_pde_total",
    "m_c_pde_mean",
    "delta_m_total",
    "delta_m_mean",
    "var_U",
    "var_D",
    "var_P",
    "O_PU",
    "mean_edible",
    "B_P",
    "mean_P",
    "persistent_total",
    "persistent_mean",
    "negative_detected",
    "pattern_measurable",
    "conservative_classification",
    "status",
]


def float_from(row: dict[str, str], key: str) -> float:
    return float(row[key])


def params_from_row(row: dict[str, str]) -> RescueParams:
    return RescueParams(
        r_U=float_from(row, "r_U"),
        r_D=float_from(row, "r_D"),
        a_U=float_from(row, "a_U"),
        a_D=float_from(row, "a_D"),
        K=float_from(row, "K"),
        e=float_from(row, "e"),
        m=float_from(row, "m"),
        mu_UD=float_from(row, "mu_UD"),
        mu_DU=float_from(row, "mu_DU"),
        delta_U=float_from(row, "delta_U"),
        delta_D=float_from(row, "delta_D"),
        delta_P=float_from(row, "delta_P"),
        L=float_from(row, "L"),
        n_x=int(float_from(row, "n_x")),
    )


def equilibrium_from_row(row: dict[str, str]) -> CoexistenceEquilibrium:
    return CoexistenceEquilibrium(
        q=float_from(row, "q_star"),
        N=float_from(row, "N_star"),
        U=float_from(row, "U_star"),
        D=float_from(row, "D_star"),
        P=float_from(row, "P_star"),
        residual=float_from(row, "equilibrium_residual"),
    )


def read_candidates(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def initial_states_for_threshold(
    params: RescueParams,
    fallback_eq: CoexistenceEquilibrium,
    m_low: float,
    perturbation: float,
    seed: int,
):
    try:
        return default_threshold_initial_state(params, m_low, perturbation, seed)
    except ValueError:
        y0_ode = np.array([fallback_eq.U, fallback_eq.D, fallback_eq.P], dtype=float)
        y0_pde = homogeneous_profile(params, fallback_eq, perturbation, seed)
        return y0_ode, y0_pde


def conservative_classification(delta_mean: float, pattern_measurable: bool, tol: float) -> str:
    if not pattern_measurable:
        return "Turing candidate but no measurable nonlinear pattern in diagnostic run"
    if delta_mean > tol:
        return "Turing candidate with Delta m_c > 0 under mean-density criterion"
    if delta_mean < -tol:
        return "Turing candidate with Delta m_c < 0 under mean-density criterion"
    return "Turing candidate with no resolvable mean-density threshold difference"


def evaluate_candidate(
    row: dict[str, str],
    T: float,
    epsilon: float,
    tol_m: float,
    max_iter: int,
    perturbation: float,
    seed: int,
    pattern_tol: float,
) -> dict[str, object]:
    params = params_from_row(row)
    eq = equilibrium_from_row(row)

    m_candidate = params.m
    m_low = max(1.0e-4, 0.10 * m_candidate)
    m_high = max(1.25 * params.e * params.K * params.a_U, 2.5 * m_candidate, m_candidate + 0.2)
    y0_ode, y0_pde = initial_states_for_threshold(params, eq, m_low, perturbation, seed)

    base = {
        "candidate_id": int(float(row["candidate_id"])),
        "m_candidate": m_candidate,
        "dominant_mode": int(float(row["dominant_mode"])),
        "dominant_growth": float_from(row, "dominant_growth"),
    }

    try:
        ode = compute_mc_ode(
            params,
            m_low=m_low,
            m_high=m_high,
            epsilon=epsilon,
            T=T,
            y0=y0_ode,
            tol_m=tol_m,
            max_iter=max_iter,
        )
        pde_total = compute_mc_pde(
            params,
            m_low=m_low,
            m_high=m_high,
            epsilon=epsilon,
            T=T,
            initial_state=y0_pde,
            tol_m=tol_m,
            max_iter=max_iter,
            persistence_criterion="total",
            n_time=100,
        )
        pde_mean = compute_mc_pde(
            params,
            m_low=m_low,
            m_high=m_high,
            epsilon=epsilon,
            T=T,
            initial_state=y0_pde,
            tol_m=tol_m,
            max_iter=max_iter,
            persistence_criterion="mean",
            n_time=100,
        )

        diagnostic = simulate_pde(
            params,
            T=T,
            initial_state=homogeneous_profile(params, eq, perturbation, seed),
            epsilon=epsilon,
            n_time=120,
        )
        d = diagnostic.diagnostics
        pattern_measurable = bool(max(d.var_U, d.var_D, d.var_P) > pattern_tol)
        delta_total = pde_total.threshold - ode.threshold
        delta_mean = pde_mean.threshold - ode.threshold

        return {
            **base,
            "m_c_ode": ode.threshold,
            "m_c_pde_total": pde_total.threshold,
            "m_c_pde_mean": pde_mean.threshold,
            "delta_m_total": delta_total,
            "delta_m_mean": delta_mean,
            "var_U": d.var_U,
            "var_D": d.var_D,
            "var_P": d.var_P,
            "O_PU": d.O_PU,
            "mean_edible": d.mean_edible,
            "B_P": d.B_P,
            "mean_P": d.mean_P,
            "persistent_total": d.persistent_total,
            "persistent_mean": d.persistent_mean,
            "negative_detected": d.negative_detected,
            "pattern_measurable": pattern_measurable,
            "conservative_classification": conservative_classification(delta_mean, pattern_measurable, tol_m),
            "status": "ok",
        }
    except Exception as exc:
        return {
            **base,
            "m_c_ode": np.nan,
            "m_c_pde_total": np.nan,
            "m_c_pde_mean": np.nan,
            "delta_m_total": np.nan,
            "delta_m_mean": np.nan,
            "var_U": np.nan,
            "var_D": np.nan,
            "var_P": np.nan,
            "O_PU": np.nan,
            "mean_edible": np.nan,
            "B_P": np.nan,
            "mean_P": np.nan,
            "persistent_total": False,
            "persistent_mean": False,
            "negative_detected": False,
            "pattern_measurable": False,
            "conservative_classification": "not classified",
            "status": f"failed: {exc}",
        }


def write_summary(candidate_rows: list[dict[str, str]], threshold_rows: list[dict[str, object]], path: Path) -> None:
    ok_rows = [row for row in threshold_rows if row["status"] == "ok"]
    if not candidate_rows:
        state = "no Turing candidates"
    elif not ok_rows:
        state = "Turing candidates found, but nonlinear threshold computations failed"
    else:
        signs = []
        for row in ok_rows:
            delta = float(row["delta_m_mean"])
            if bool(row["pattern_measurable"]):
                signs.append(np.sign(delta))
        if not signs:
            state = "Turing candidates but no measurable nonlinear pattern in diagnostic runs"
        elif all(s > 0 for s in signs):
            state = "Turing candidates with Delta m_c > 0 under the mean-density criterion"
        elif all(s < 0 for s in signs):
            state = "Turing candidates with Delta m_c < 0 under the mean-density criterion"
        elif any(s > 0 for s in signs) and any(s < 0 for s in signs):
            state = "parameter-dependent sign changes"
        else:
            state = "Turing candidates but no resolvable rescue effect"

    lines = [
        "# Nonlinear PDE Results 02",
        "",
        f"Conservative scan outcome: **{state}**.",
        "",
        "Interpretation rule: pattern-mediated rescue is not claimed unless the parameter set is Turing-unstable, the nonlinear PDE develops measurable spatial patterning, and `Delta m_c` remains positive under the density-normalized mean predator criterion.",
        "",
        f"- Turing candidates loaded: `{len(candidate_rows)}`",
        f"- threshold rows computed: `{len(threshold_rows)}`",
        f"- successful threshold rows: `{len(ok_rows)}`",
        "",
        "Outputs:",
        "",
        f"- `{CANDIDATES_CSV.relative_to(ROOT)}`",
        f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
    ]

    if ok_rows:
        lines.extend(
            [
                "Best computed rows:",
                "",
                "| candidate | Delta total | Delta mean | pattern measurable | classification |",
                "|---:|---:|---:|:---:|---|",
            ]
        )
        for row in ok_rows[:10]:
            lines.append(
                f"| {row['candidate_id']} | {float(row['delta_m_total']):.6g} | "
                f"{float(row['delta_m_mean']):.6g} | {row['pattern_measurable']} | "
                f"{row['conservative_classification']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--summary", type=Path, default=SUMMARY_MD)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--T", type=float, default=60.0)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument("--tol-m", type=float, default=2.0e-2)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--perturbation", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--pattern-tol", type=float, default=1.0e-6)
    args = parser.parse_args()

    candidates = read_candidates(args.candidates)
    candidates.sort(key=lambda row: float(row.get("dominant_growth", "-inf")), reverse=True)
    selected = candidates[: args.max_candidates]

    rows = [
        evaluate_candidate(
            row,
            T=args.T,
            epsilon=args.epsilon,
            tol_m=args.tol_m,
            max_iter=args.max_iter,
            perturbation=args.perturbation,
            seed=args.seed,
            pattern_tol=args.pattern_tol,
        )
        for row in selected
    ]
    write_rows(rows, args.output)
    write_summary(candidates, rows, args.summary)

    print(f"loaded {len(candidates)} Turing candidates")
    print(f"computed thresholds for {len(rows)} candidates")
    print(f"saved threshold scan to {args.output}")
    print(f"saved summary to {args.summary}")


if __name__ == "__main__":
    main()
