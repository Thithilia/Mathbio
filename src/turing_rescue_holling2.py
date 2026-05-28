"""Holling type II variant for Turing feasibility checks.

This module is intentionally separate from ``turing_rescue_model.py``.  It keeps the
same U, D, P state variables, conversion terms, and diffusion coefficients, but
replaces mass-action predation by a saturating multi-prey functional response.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from scipy.optimize import brentq

from .turing_rescue_model import CoexistenceEquilibrium, ContinuousTuringScanResult, TuringScanResult, ode_local_stability


@dataclass(frozen=True)
class HollingIIParams:
    r_U: float = 1.2
    r_D: float = 0.6
    a_U: float = 1.4
    a_D: float = 0.35
    K: float = 1.0
    e: float = 0.55
    m: float = 0.22
    mu_UD: float = 0.04
    mu_DU: float = 0.04
    delta_U: float = 1.0e-3
    delta_D: float = 5.0e-2
    delta_P: float = 5.0e-1
    h: float = 0.8
    L: float = 20.0
    n_x: int = 80

    def with_updates(self, **kwargs: float) -> "HollingIIParams":
        return replace(self, **kwargs)


def attack_weight(q: float | np.ndarray, params: HollingIIParams) -> float | np.ndarray:
    return params.a_D + (params.a_U - params.a_D) * q


def growth_weight(q: float | np.ndarray, params: HollingIIParams) -> float | np.ndarray:
    return params.r_D + (params.r_U - params.r_D) * q


def reaction_part_holling2(
    U: np.ndarray | float,
    D: np.ndarray | float,
    P: np.ndarray | float,
    params: HollingIIParams,
) -> np.ndarray:
    U_arr = np.asarray(U, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    P_arr = np.asarray(P, dtype=float)
    total_prey = U_arr + D_arr
    resource = 1.0 - total_prey / params.K
    edible = params.a_U * U_arr + params.a_D * D_arr
    denominator = 1.0 + params.h * edible

    pred_U = params.a_U * U_arr * P_arr / denominator
    pred_D = params.a_D * D_arr * P_arr / denominator
    predator_gain = params.e * edible * P_arr / denominator

    dU = params.r_U * U_arr * resource - pred_U + params.mu_DU * D_arr - params.mu_UD * U_arr
    dD = params.r_D * D_arr * resource - pred_D + params.mu_UD * U_arr - params.mu_DU * D_arr
    dP = predator_gain - params.m * P_arr
    return np.stack([dU, dD, dP], axis=0)


def reaction_ode_holling2(_t: float, y: Iterable[float], params: HollingIIParams) -> np.ndarray:
    U, D, P = np.asarray(y, dtype=float)
    return reaction_part_holling2(U, D, P, params)


def coexistence_scalar_G_holling2(q: float, params: HollingIIParams, m: float | None = None) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1).")

    current_m = params.m if m is None else m
    if params.h < 0.0:
        raise ValueError("handling time h must be non-negative.")
    if params.e - params.h * current_m <= 0.0:
        return float("nan")

    edible_total = current_m / (params.e - params.h * current_m)
    A = attack_weight(q, params)
    R = growth_weight(q, params)
    N = edible_total / A
    resource = 1.0 - N / params.K

    return resource * ((params.r_U - params.r_D) - (params.a_U - params.a_D) * R / A) + params.mu_DU / q - params.mu_UD / (1.0 - q)


def reconstruct_equilibrium_holling2(
    q: float,
    params: HollingIIParams,
    m: float | None = None,
) -> CoexistenceEquilibrium:
    current_m = params.m if m is None else m
    if params.e - params.h * current_m <= 0.0:
        raise ValueError("m must be less than e/h for a positive Holling-II predator equilibrium.")

    edible_total = current_m / (params.e - params.h * current_m)
    A = attack_weight(q, params)
    R = growth_weight(q, params)
    N = edible_total / A
    resource = 1.0 - N / params.K
    functional_response = edible_total / (1.0 + params.h * edible_total)
    P = R * N * resource / functional_response
    U = q * N
    D = (1.0 - q) * N
    residual = float(np.linalg.norm(reaction_part_holling2(U, D, P, params.with_updates(m=current_m))))
    return CoexistenceEquilibrium(q=q, N=N, U=U, D=D, P=P, residual=residual)


def solve_coexistence_equilibria_holling2(
    params: HollingIIParams,
    m: float | None = None,
    n_grid: int = 1200,
    q_eps: float = 1.0e-6,
    residual_tol: float = 1.0e-7,
) -> list[CoexistenceEquilibrium]:
    current_m = params.m if m is None else m
    qs = np.linspace(q_eps, 1.0 - q_eps, n_grid)
    values = np.empty_like(qs)
    for i, q in enumerate(qs):
        try:
            values[i] = coexistence_scalar_G_holling2(float(q), params, current_m)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            values[i] = np.nan

    roots: list[float] = []
    for left, right, f_left, f_right in zip(qs[:-1], qs[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            roots.append(float(left))
            continue
        if f_left * f_right > 0.0:
            continue
        try:
            root = brentq(
                lambda z: coexistence_scalar_G_holling2(z, params, current_m),
                float(left),
                float(right),
                maxiter=100,
            )
            roots.append(float(root))
        except ValueError:
            continue

    unique_roots: list[float] = []
    for root in sorted(roots):
        if not unique_roots or abs(root - unique_roots[-1]) > 1.0e-5:
            unique_roots.append(root)

    equilibria: list[CoexistenceEquilibrium] = []
    for root in unique_roots:
        eq = reconstruct_equilibrium_holling2(root, params, current_m)
        if eq.U > 0.0 and eq.D > 0.0 and eq.P > 0.0 and eq.N > 0.0 and eq.residual < residual_tol:
            equilibria.append(eq)
    return equilibria


def jacobian_reaction_holling2(U: float, D: float, P: float, params: HollingIIParams) -> np.ndarray:
    edible = params.a_U * U + params.a_D * D
    denominator = 1.0 + params.h * edible
    denominator2 = denominator * denominator

    dfU_dU = params.a_U / denominator - params.a_U * U * params.h * params.a_U / denominator2
    dfU_dD = -params.a_U * U * params.h * params.a_D / denominator2
    dfD_dU = -params.a_D * D * params.h * params.a_U / denominator2
    dfD_dD = params.a_D / denominator - params.a_D * D * params.h * params.a_D / denominator2
    dF_dU = params.a_U / denominator2
    dF_dD = params.a_D / denominator2
    fU = params.a_U * U / denominator
    fD = params.a_D * D / denominator
    F = edible / denominator

    return np.array(
        [
            [
                params.r_U * (1.0 - (2.0 * U + D) / params.K) - P * dfU_dU - params.mu_UD,
                -params.r_U * U / params.K - P * dfU_dD + params.mu_DU,
                -fU,
            ],
            [
                -params.r_D * D / params.K - P * dfD_dU + params.mu_UD,
                params.r_D * (1.0 - (U + 2.0 * D) / params.K) - P * dfD_dD - params.mu_DU,
                -fD,
            ],
            [
                params.e * P * dF_dU,
                params.e * P * dF_dD,
                params.e * F - params.m,
            ],
        ],
        dtype=float,
    )


def _unstable_windows_from_grid(ks: np.ndarray, max_real: np.ndarray, tol: float) -> list[tuple[float, float]]:
    unstable = max_real > tol
    windows: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_unstable in enumerate(unstable):
        if is_unstable and start_idx is None:
            start_idx = idx
        elif not is_unstable and start_idx is not None:
            windows.append((float(ks[start_idx]), float(ks[idx - 1])))
            start_idx = None
    if start_idx is not None:
        windows.append((float(ks[start_idx]), float(ks[-1])))
    return windows


def continuous_turing_scan_holling2(
    params: HollingIIParams,
    equilibrium: CoexistenceEquilibrium,
    k_min: float,
    k_max: float,
    n_k: int,
    tol: float = 1.0e-9,
) -> ContinuousTuringScanResult:
    if k_min < 0.0:
        raise ValueError("k_min must be non-negative.")
    if k_max <= k_min:
        raise ValueError("k_max must be greater than k_min.")
    if n_k < 2:
        raise ValueError("n_k must be at least 2.")

    J = jacobian_reaction_holling2(equilibrium.U, equilibrium.D, equilibrium.P, params)
    ode_stable, ode_eigs = ode_local_stability(J, tol=tol)
    diffusion_diag = np.diag([params.delta_U, params.delta_D, params.delta_P])
    ks = np.linspace(k_min, k_max, n_k)
    eigvals = np.empty((n_k, 3), dtype=complex)
    max_real = np.empty(n_k, dtype=float)
    for idx, k in enumerate(ks):
        shifted = J - (k * k) * diffusion_diag
        eigvals[idx] = np.linalg.eigvals(shifted)
        max_real[idx] = np.max(eigvals[idx].real)

    dominant_idx = int(np.argmax(max_real))
    has_unstable_spatial_mode = bool(np.any(max_real > tol))
    return ContinuousTuringScanResult(
        ks=ks,
        eigenvalues=eigvals,
        max_real_by_k=max_real,
        maximum_spatial_growth=float(max_real[dominant_idx]),
        k_at_maximum_growth=float(ks[dominant_idx]),
        unstable_windows=_unstable_windows_from_grid(ks, max_real, tol),
        ode_eigenvalues=ode_eigs,
        ode_stable=ode_stable,
        has_unstable_spatial_mode=has_unstable_spatial_mode,
        turing_unstable=bool(ode_stable and has_unstable_spatial_mode),
    )


def turing_scan_holling2(
    params: HollingIIParams,
    equilibrium: CoexistenceEquilibrium,
    n_max: int = 80,
    tol: float = 1.0e-9,
) -> TuringScanResult:
    J = jacobian_reaction_holling2(equilibrium.U, equilibrium.D, equilibrium.P, params)
    ode_stable, ode_eigs = ode_local_stability(J, tol=tol)
    diffusion_diag = np.diag([params.delta_U, params.delta_D, params.delta_P])
    modes = np.arange(1, n_max + 1)
    ks = modes * np.pi / params.L
    eigvals = np.empty((n_max, 3), dtype=complex)
    max_real = np.empty(n_max, dtype=float)
    for idx, k in enumerate(ks):
        shifted = J - (k * k) * diffusion_diag
        eigvals[idx] = np.linalg.eigvals(shifted)
        max_real[idx] = np.max(eigvals[idx].real)

    dominant_idx = int(np.argmax(max_real)) if n_max else None
    dominant_growth = float(max_real[dominant_idx]) if dominant_idx is not None else float("-inf")
    has_unstable_spatial_mode = bool(np.any(max_real > tol))
    return TuringScanResult(
        ks=ks,
        eigenvalues=eigvals,
        max_real_by_mode=max_real,
        ode_eigenvalues=ode_eigs,
        ode_stable=ode_stable,
        has_unstable_spatial_mode=has_unstable_spatial_mode,
        turing_unstable=bool(ode_stable and has_unstable_spatial_mode),
        dominant_mode=(dominant_idx + 1) if dominant_idx is not None else None,
        dominant_k=float(ks[dominant_idx]) if dominant_idx is not None else None,
        dominant_growth=dominant_growth,
    )
