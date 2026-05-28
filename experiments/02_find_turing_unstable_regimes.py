"""Search for Turing-unstable parameter regimes in the current model.

The scan is deliberately conservative: a candidate is saved only if it has a positive
coexistence equilibrium, the ODE equilibrium is locally stable, and at least one
Neumann spatial mode is unstable.
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

from src.turing_rescue_model import RescueParams, solve_coexistence_equilibria, turing_scan


RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "turing_candidates.csv"


FIELDNAMES = [
    "candidate_id",
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
    "L",
    "n_x",
    "q_star",
    "N_star",
    "U_star",
    "D_star",
    "P_star",
    "equilibrium_residual",
    "ode_max_real",
    "dominant_mode",
    "dominant_k",
    "dominant_growth",
    "n_unstable_modes",
]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_params(rng: np.random.Generator, n_x: int, L: float) -> RescueParams:
    r_D = rng.uniform(0.2, 1.2)
    defense_cost = rng.uniform(0.03, 2.0)
    r_U = r_D + defense_cost

    a_D = rng.uniform(0.03, 0.9)
    vulnerability_contrast = rng.uniform(0.03, 4.0)
    a_U = a_D + vulnerability_contrast

    mu_UD = log_uniform(rng, 1.0e-4, 3.0e-1)
    mu_DU = log_uniform(rng, 1.0e-4, 3.0e-1)
    e = rng.uniform(0.15, 1.3)
    K = 1.0

    q0 = mu_DU / (mu_UD + mu_DU)
    prey_only_attack = a_D + (a_U - a_D) * q0
    m_inv = e * K * prey_only_attack
    m = rng.uniform(0.05, 0.98) * m_inv

    delta_U = log_uniform(rng, 1.0e-5, 1.0e-1)
    delta_D_ratio = log_uniform(rng, 1.0e-2, 1.0e4)
    delta_P_ratio = log_uniform(rng, 1.0e-2, 1.0e5)

    return RescueParams(
        r_U=r_U,
        r_D=r_D,
        a_U=a_U,
        a_D=a_D,
        K=K,
        e=e,
        m=m,
        mu_UD=mu_UD,
        mu_DU=mu_DU,
        delta_U=delta_U,
        delta_D=delta_U * delta_D_ratio,
        delta_P=delta_U * delta_P_ratio,
        L=L,
        n_x=n_x,
    )


def candidate_row(candidate_id: int, params: RescueParams, eq, scan) -> dict[str, float | int]:
    ode_max_real = float(np.max(scan.ode_eigenvalues.real))
    n_unstable_modes = int(np.count_nonzero(scan.max_real_by_mode > 0.0))
    return {
        "candidate_id": candidate_id,
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
        "L": params.L,
        "n_x": params.n_x,
        "q_star": eq.q,
        "N_star": eq.N,
        "U_star": eq.U,
        "D_star": eq.D,
        "P_star": eq.P,
        "equilibrium_residual": eq.residual,
        "ode_max_real": ode_max_real,
        "dominant_mode": scan.dominant_mode or -1,
        "dominant_k": scan.dominant_k or float("nan"),
        "dominant_growth": scan.dominant_growth,
        "n_unstable_modes": n_unstable_modes,
    }


def run_search(
    n_samples: int,
    seed: int,
    n_max: int,
    n_x: int,
    L: float,
    max_candidates: int | None,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    candidate_id = 0

    for _ in range(n_samples):
        params = sample_params(rng, n_x=n_x, L=L)
        equilibria = solve_coexistence_equilibria(params, n_grid=500)
        if not equilibria:
            continue

        for eq in equilibria:
            scan = turing_scan(params, eq, n_max=n_max, tol=1.0e-8)
            if not scan.turing_unstable:
                continue
            rows.append(candidate_row(candidate_id, params, eq, scan))
            candidate_id += 1
            if max_candidates is not None and len(rows) >= max_candidates:
                return rows

    return rows


def write_candidates(rows: list[dict[str, float | int]], output: Path) -> None:
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--n-x", type=int, default=50)
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    rows = run_search(
        n_samples=args.samples,
        seed=args.seed,
        n_max=args.n_max,
        n_x=args.n_x,
        L=args.L,
        max_candidates=args.max_candidates,
    )
    rows.sort(key=lambda row: float(row["dominant_growth"]), reverse=True)
    write_candidates(rows, args.output)

    print(f"scanned {args.samples} parameter sets")
    print(f"saved {len(rows)} Turing candidates to {args.output}")
    if rows:
        best = rows[0]
        print(
            "best candidate: "
            f"id={best['candidate_id']}, m={best['m']:.6g}, "
            f"dominant_growth={best['dominant_growth']:.6g}, "
            f"mode={best['dominant_mode']}"
        )


if __name__ == "__main__":
    main()
