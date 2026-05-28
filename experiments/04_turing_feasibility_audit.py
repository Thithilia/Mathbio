"""Targeted Turing feasibility audit for the current rescue models.

This script optimizes the spatial growth objective instead of relying only on random
sampling.  It first audits the minimal mass-action model.  If no minimal-model
continuous Turing candidate is found, it repeats the audit for a separated Holling-II
variant.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.turing_rescue_holling2 import (
    HollingIIParams,
    continuous_turing_scan_holling2,
    solve_coexistence_equilibria_holling2,
    turing_scan_holling2,
)
from src.turing_rescue_model import RescueParams, continuous_turing_scan, solve_coexistence_equilibria, turing_scan


RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "turing_feasibility_audit.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_03.md"


FIELDNAMES = [
    "model",
    "restart",
    "rank",
    "objective_value",
    "failure_reason",
    "continuous_turing_candidate",
    "discrete_turing_candidate",
    "r_U",
    "r_D",
    "defense_cost",
    "a_U",
    "a_D",
    "vulnerability_contrast",
    "K",
    "e",
    "m",
    "mu_UD",
    "mu_DU",
    "delta_U",
    "delta_D",
    "delta_P",
    "delta_D_over_delta_U",
    "delta_P_over_delta_U",
    "h",
    "L",
    "n_x",
    "q_star",
    "N_star",
    "U_star",
    "D_star",
    "P_star",
    "equilibrium_residual",
    "ode_max_real",
    "continuous_max_spatial_growth",
    "best_k",
    "continuous_unstable_windows",
    "discrete_max_spatial_growth",
    "dominant_discrete_mode",
    "dominant_discrete_k",
]


MINIMAL_BOUNDS = [
    (0.05, 2.0),  # r_D
    (-3.0, math.log10(5.0)),  # log10(r_U-r_D)
    (-2.0, math.log10(2.0)),  # log10(a_D)
    (-3.0, math.log10(8.0)),  # log10(a_U-a_D)
    (-6.0, 0.0),  # log10(mu_UD)
    (-6.0, 0.0),  # log10(mu_DU)
    (0.05, 2.0),  # e
    (0.02, 0.98),  # m as fraction of prey-only invasion threshold
    (-7.0, 1.0),  # log10(delta_U)
    (-5.0, 6.0),  # log10(delta_D/delta_U)
    (-5.0, 6.0),  # log10(delta_P/delta_U)
]
HOLLING_BOUNDS = [*MINIMAL_BOUNDS, (-3.0, math.log10(20.0))]  # log10(handling time)


@dataclass(frozen=True)
class AuditEvaluation:
    model: str
    params: RescueParams | HollingIIParams
    objective_value: float
    failure_reason: str
    continuous_turing_candidate: bool
    discrete_turing_candidate: bool
    q_star: float = math.nan
    N_star: float = math.nan
    U_star: float = math.nan
    D_star: float = math.nan
    P_star: float = math.nan
    equilibrium_residual: float = math.nan
    ode_max_real: float = math.nan
    continuous_max_spatial_growth: float = math.nan
    best_k: float = math.nan
    continuous_unstable_windows: str = ""
    discrete_max_spatial_growth: float = math.nan
    dominant_discrete_mode: int = -1
    dominant_discrete_k: float = math.nan


def format_windows(windows: list[tuple[float, float]]) -> str:
    return ";".join(f"{left:.6g}:{right:.6g}" for left, right in windows)


def decode_minimal(x: np.ndarray, L: float, n_x: int) -> RescueParams:
    r_D = float(x[0])
    r_U = r_D + 10.0 ** float(x[1])
    a_D = 10.0 ** float(x[2])
    a_U = a_D + 10.0 ** float(x[3])
    mu_UD = 10.0 ** float(x[4])
    mu_DU = 10.0 ** float(x[5])
    e = float(x[6])
    q0 = mu_DU / (mu_UD + mu_DU)
    m_inv = e * (a_D + (a_U - a_D) * q0)
    m = float(x[7]) * m_inv
    delta_U = 10.0 ** float(x[8])
    delta_D = delta_U * 10.0 ** float(x[9])
    delta_P = delta_U * 10.0 ** float(x[10])
    return RescueParams(
        r_U=r_U,
        r_D=r_D,
        a_U=a_U,
        a_D=a_D,
        K=1.0,
        e=e,
        m=m,
        mu_UD=mu_UD,
        mu_DU=mu_DU,
        delta_U=delta_U,
        delta_D=delta_D,
        delta_P=delta_P,
        L=L,
        n_x=n_x,
    )


def decode_holling2(x: np.ndarray, L: float, n_x: int) -> HollingIIParams:
    r_D = float(x[0])
    r_U = r_D + 10.0 ** float(x[1])
    a_D = 10.0 ** float(x[2])
    a_U = a_D + 10.0 ** float(x[3])
    mu_UD = 10.0 ** float(x[4])
    mu_DU = 10.0 ** float(x[5])
    e = float(x[6])
    h = 10.0 ** float(x[11])
    q0 = mu_DU / (mu_UD + mu_DU)
    prey_only_edible = a_D + (a_U - a_D) * q0
    m_inv = e * prey_only_edible / (1.0 + h * prey_only_edible)
    m = float(x[7]) * m_inv
    delta_U = 10.0 ** float(x[8])
    delta_D = delta_U * 10.0 ** float(x[9])
    delta_P = delta_U * 10.0 ** float(x[10])
    return HollingIIParams(
        r_U=r_U,
        r_D=r_D,
        a_U=a_U,
        a_D=a_D,
        K=1.0,
        e=e,
        m=m,
        mu_UD=mu_UD,
        mu_DU=mu_DU,
        delta_U=delta_U,
        delta_D=delta_D,
        delta_P=delta_P,
        h=h,
        L=L,
        n_x=n_x,
    )


def classify_failure(ode_stable: bool, continuous_growth: float, tol: float) -> str:
    if not ode_stable:
        return "ode_instability"
    if continuous_growth <= tol:
        return "no_spatial_instability"
    return "turing_candidate"


class FeasibilityEvaluator:
    def __init__(
        self,
        model: str,
        L: float,
        n_x: int,
        eq_grid: int,
        k_min: float,
        k_max: float,
        n_k: int,
        n_max: int,
        tol: float,
    ) -> None:
        self.model = model
        self.L = L
        self.n_x = n_x
        self.eq_grid = eq_grid
        self.k_min = k_min
        self.k_max = k_max
        self.n_k = n_k
        self.n_max = n_max
        self.tol = tol
        self.cache: dict[tuple[float, ...], AuditEvaluation] = {}

    def decode(self, x: np.ndarray) -> RescueParams | HollingIIParams:
        if self.model == "minimal":
            return decode_minimal(x, self.L, self.n_x)
        if self.model == "holling2":
            return decode_holling2(x, self.L, self.n_x)
        raise ValueError(f"Unknown model: {self.model}")

    def equilibria(self, params: RescueParams | HollingIIParams):
        if self.model == "minimal":
            return solve_coexistence_equilibria(params, n_grid=self.eq_grid)
        return solve_coexistence_equilibria_holling2(params, n_grid=self.eq_grid)

    def continuous_scan(self, params: RescueParams | HollingIIParams, eq):
        if self.model == "minimal":
            return continuous_turing_scan(params, eq, self.k_min, self.k_max, self.n_k, tol=self.tol)
        return continuous_turing_scan_holling2(params, eq, self.k_min, self.k_max, self.n_k, tol=self.tol)

    def discrete_scan(self, params: RescueParams | HollingIIParams, eq):
        if self.model == "minimal":
            return turing_scan(params, eq, n_max=self.n_max, tol=self.tol)
        return turing_scan_holling2(params, eq, n_max=self.n_max, tol=self.tol)

    def evaluate(self, x: np.ndarray) -> AuditEvaluation:
        key = tuple(np.round(x, 8))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        params = self.decode(x)
        if params.r_U <= params.r_D or params.a_U <= params.a_D:
            evaluation = AuditEvaluation(self.model, params, 200.0, "invalid_biological_ordering", False, False)
            self.cache[key] = evaluation
            return evaluation

        equilibria = self.equilibria(params)
        if not equilibria:
            evaluation = AuditEvaluation(self.model, params, 100.0, "no_coexistence_equilibrium", False, False)
            self.cache[key] = evaluation
            return evaluation

        best: AuditEvaluation | None = None
        for eq in equilibria:
            continuous = self.continuous_scan(params, eq)
            discrete = self.discrete_scan(params, eq)
            ode_max = float(np.max(continuous.ode_eigenvalues.real))
            failure = classify_failure(continuous.ode_stable, continuous.maximum_spatial_growth, self.tol)
            if continuous.ode_stable:
                objective = -continuous.maximum_spatial_growth
            else:
                objective = 10.0 + max(ode_max, 0.0) * 1000.0 - min(continuous.maximum_spatial_growth, 0.0)

            evaluation = AuditEvaluation(
                model=self.model,
                params=params,
                objective_value=float(objective),
                failure_reason=failure,
                continuous_turing_candidate=bool(continuous.turing_unstable),
                discrete_turing_candidate=bool(discrete.turing_unstable),
                q_star=eq.q,
                N_star=eq.N,
                U_star=eq.U,
                D_star=eq.D,
                P_star=eq.P,
                equilibrium_residual=eq.residual,
                ode_max_real=ode_max,
                continuous_max_spatial_growth=continuous.maximum_spatial_growth,
                best_k=continuous.k_at_maximum_growth,
                continuous_unstable_windows=format_windows(continuous.unstable_windows),
                discrete_max_spatial_growth=discrete.dominant_growth,
                dominant_discrete_mode=discrete.dominant_mode or -1,
                dominant_discrete_k=discrete.dominant_k or math.nan,
            )
            if best is None or evaluation.objective_value < best.objective_value:
                best = evaluation

        assert best is not None
        self.cache[key] = best
        return best

    def objective(self, x: np.ndarray) -> float:
        return self.evaluate(np.asarray(x, dtype=float)).objective_value


def evaluation_to_row(evaluation: AuditEvaluation, restart: int, rank: int) -> dict[str, object]:
    params = evaluation.params
    h = getattr(params, "h", math.nan)
    return {
        "model": evaluation.model,
        "restart": restart,
        "rank": rank,
        "objective_value": evaluation.objective_value,
        "failure_reason": evaluation.failure_reason,
        "continuous_turing_candidate": evaluation.continuous_turing_candidate,
        "discrete_turing_candidate": evaluation.discrete_turing_candidate,
        "r_U": params.r_U,
        "r_D": params.r_D,
        "defense_cost": params.r_U - params.r_D,
        "a_U": params.a_U,
        "a_D": params.a_D,
        "vulnerability_contrast": params.a_U - params.a_D,
        "K": params.K,
        "e": params.e,
        "m": params.m,
        "mu_UD": params.mu_UD,
        "mu_DU": params.mu_DU,
        "delta_U": params.delta_U,
        "delta_D": params.delta_D,
        "delta_P": params.delta_P,
        "delta_D_over_delta_U": params.delta_D / params.delta_U,
        "delta_P_over_delta_U": params.delta_P / params.delta_U,
        "h": h,
        "L": params.L,
        "n_x": params.n_x,
        "q_star": evaluation.q_star,
        "N_star": evaluation.N_star,
        "U_star": evaluation.U_star,
        "D_star": evaluation.D_star,
        "P_star": evaluation.P_star,
        "equilibrium_residual": evaluation.equilibrium_residual,
        "ode_max_real": evaluation.ode_max_real,
        "continuous_max_spatial_growth": evaluation.continuous_max_spatial_growth,
        "best_k": evaluation.best_k,
        "continuous_unstable_windows": evaluation.continuous_unstable_windows,
        "discrete_max_spatial_growth": evaluation.discrete_max_spatial_growth,
        "dominant_discrete_mode": evaluation.dominant_discrete_mode,
        "dominant_discrete_k": evaluation.dominant_discrete_k,
    }


def run_model_audit(
    model: str,
    bounds: list[tuple[float, float]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for restart in range(args.restarts):
        evaluator = FeasibilityEvaluator(
            model=model,
            L=args.L,
            n_x=args.n_x,
            eq_grid=args.eq_grid,
            k_min=args.k_min,
            k_max=args.k_max,
            n_k=args.continuous_k,
            n_max=args.n_max,
            tol=args.tol,
        )
        result = differential_evolution(
            evaluator.objective,
            bounds=bounds,
            seed=args.seed + restart,
            maxiter=args.maxiter,
            popsize=args.popsize,
            tol=args.de_tol,
            polish=args.polish,
            updating="immediate",
            workers=1,
        )

        vectors = [np.asarray(result.x, dtype=float), *[np.asarray(vec, dtype=float) for vec in result.population]]
        unique: dict[tuple[float, ...], AuditEvaluation] = {}
        for vector in vectors:
            key = tuple(np.round(vector, 7))
            unique[key] = evaluator.evaluate(vector)

        ranked = sorted(unique.values(), key=lambda item: item.objective_value)
        for rank, evaluation in enumerate(ranked[: args.rows_per_restart], start=1):
            rows.append(evaluation_to_row(evaluation, restart=restart, rank=rank))

        best = ranked[0]
        print(
            f"{model} restart {restart}: best failure={best.failure_reason}, "
            f"continuous_growth={best.continuous_max_spatial_growth:.6g}, "
            f"ode_max={best.ode_max_real:.6g}, objective={best.objective_value:.6g}"
        )
    return rows


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize_model(rows: list[dict[str, object]], model: str) -> dict[str, object]:
    model_rows = [row for row in rows if row["model"] == model]
    if not model_rows:
        return {"rows": 0, "continuous": 0, "discrete": 0, "best": None, "failures": {}}
    failures: dict[str, int] = {}
    for row in model_rows:
        failure = str(row["failure_reason"])
        failures[failure] = failures.get(failure, 0) + 1
    return {
        "rows": len(model_rows),
        "continuous": sum(bool(row["continuous_turing_candidate"]) for row in model_rows),
        "discrete": sum(bool(row["discrete_turing_candidate"]) for row in model_rows),
        "best": min(model_rows, key=lambda row: float(row["objective_value"])),
        "failures": failures,
    }


def write_summary(rows: list[dict[str, object]], args: argparse.Namespace, output: Path) -> None:
    minimal = summarize_model(rows, "minimal")
    holling = summarize_model(rows, "holling2")

    if int(minimal["continuous"]) > 0:
        minimal_answer = "Yes, within this targeted optimization run the minimal model admitted at least one continuous-k Turing candidate."
    else:
        minimal_answer = "No, this targeted optimization run did not find a minimal-model continuous-k Turing candidate."

    if int(holling["rows"]) == 0:
        holling_answer = "The Holling-II audit was not run because a minimal-model candidate was already found."
    elif int(holling["continuous"]) > 0:
        holling_answer = "Yes, the Holling-II variant admitted at least one continuous-k Turing candidate in this audit."
    else:
        holling_answer = "No, the Holling-II variant did not admit a continuous-k Turing candidate in this audit."

    if int(holling["discrete"]) > 0:
        next_model = "Use the Holling-II variant for the next nonlinear threshold computation, prioritizing rows that are also discrete-mode unstable on the finite domain."
    elif int(minimal["discrete"]) > 0:
        next_model = "Use the minimal model candidate that is discrete-mode unstable for the next nonlinear threshold computation."
    elif int(holling["continuous"]) > 0:
        next_model = "Inspect the Holling-II continuous candidate against domain size and Neumann modes before nonlinear threshold computation."
    else:
        next_model = "Do not proceed to nonlinear threshold computation yet; broaden or analytically refine the feasibility audit first."

    lines = [
        "# Nonlinear PDE Results 03",
        "",
        "This audit optimized the linear Turing growth objective with `scipy.optimize.differential_evolution`; it did not run nonlinear PDE threshold simulations and does not claim pattern-mediated rescue.",
        "",
        "## Audit settings",
        "",
        f"- restarts: `{args.restarts}`",
        f"- maxiter: `{args.maxiter}`",
        f"- popsize: `{args.popsize}`",
        f"- continuous k range: `[{args.k_min}, {args.k_max}]` with `{args.continuous_k}` grid points",
        f"- discrete Neumann modes: `1..{args.n_max}`",
        f"- equilibrium grid: `{args.eq_grid}`",
        "- parameter ranges: `r_D in [0.05,2]`, `r_U-r_D in [1e-3,5]`, `a_D in [1e-2,2]`, `a_U-a_D in [1e-3,8]`, `mu in [1e-6,1]`, `delta_U in [1e-7,10]`, diffusion ratios in `[1e-5,1e6]`",
        "- Holling-II handling-time range, when run: `h in [1e-3,20]`",
        "",
        "## Minimal model",
        "",
        minimal_answer,
        "",
        f"- best rows saved: `{minimal['rows']}`",
        f"- continuous-k candidates among saved rows: `{minimal['continuous']}`",
        f"- discrete-domain candidates among saved rows: `{minimal['discrete']}`",
        f"- failure counts among saved rows: `{minimal['failures']}`",
        "",
        "The negative result is only robust within the parameter bounds and optimizer budget used here; it is stronger than the previous random scan because the objective was directly optimized, but it is not a mathematical non-existence proof.",
        "",
        "## Holling-II variant",
        "",
        holling_answer,
        "",
        f"- best rows saved: `{holling['rows']}`",
        f"- continuous-k candidates among saved rows: `{holling['continuous']}`",
        f"- discrete-domain candidates among saved rows: `{holling['discrete']}`",
        f"- failure counts among saved rows: `{holling['failures']}`",
        "",
        "## Next model for nonlinear thresholds",
        "",
        next_model,
        "",
        "Output:",
        "",
        f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--summary", type=Path, default=SUMMARY_MD)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--rows-per-restart", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--de-tol", type=float, default=1.0e-3)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--eq-grid", type=int, default=350)
    parser.add_argument("--k-min", type=float, default=1.0e-4)
    parser.add_argument("--k-max", type=float, default=20.0)
    parser.add_argument("--continuous-k", type=int, default=240)
    parser.add_argument("--n-max", type=int, default=120)
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--n-x", type=int, default=50)
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--always-run-holling", action="store_true")
    parser.add_argument("--minimal-only", action="store_true")
    args = parser.parse_args()

    rows = run_model_audit("minimal", MINIMAL_BOUNDS, args)
    minimal_has_candidate = any(bool(row["continuous_turing_candidate"]) for row in rows if row["model"] == "minimal")
    if not args.minimal_only and (args.always_run_holling or not minimal_has_candidate):
        rows.extend(run_model_audit("holling2", HOLLING_BOUNDS, args))

    model_order = {"minimal": 0, "holling2": 1}
    rows.sort(key=lambda row: (model_order.get(str(row["model"]), 99), int(row["restart"]), int(row["rank"])))
    write_rows(rows, args.output)
    write_summary(rows, args, args.summary)

    print(f"saved feasibility audit to {args.output}")
    print(f"saved summary to {args.summary}")


if __name__ == "__main__":
    main()
