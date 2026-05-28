"""Reaction terms and linear analysis for the defended-prey rescue model.

This module deliberately separates the analytic linear invasion calculation from the
nonlinear persistence thresholds.  The invasion threshold is a local prey-only result;
the rescue thresholds m_c^ODE and m_c^PDE are computed by time integration in
``simulate_pde_1d.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class RescueParams:
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
    L: float = 20.0
    n_x: int = 80

    def with_updates(self, **kwargs: float) -> "RescueParams":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class CoexistenceEquilibrium:
    q: float
    N: float
    U: float
    D: float
    P: float
    residual: float


@dataclass(frozen=True)
class TuringScanResult:
    ks: np.ndarray
    eigenvalues: np.ndarray
    max_real_by_mode: np.ndarray
    ode_eigenvalues: np.ndarray
    ode_stable: bool
    has_unstable_spatial_mode: bool
    turing_unstable: bool
    dominant_mode: int | None
    dominant_k: float | None
    dominant_growth: float


def attack_weight(q: float | np.ndarray, params: RescueParams) -> float | np.ndarray:
    """Return a_D + (a_U-a_D)q, the mean attack rate at prey fraction q."""
    return params.a_D + (params.a_U - params.a_D) * q


def growth_weight(q: float | np.ndarray, params: RescueParams) -> float | np.ndarray:
    """Return r_D + (r_U-r_D)q, the mean prey growth rate at prey fraction q."""
    return params.r_D + (params.r_U - params.r_D) * q


def reaction_ode(t: float, y: Iterable[float], params: RescueParams) -> np.ndarray:
    """Reaction terms for the spatially homogeneous ODE model."""
    U, D, P = np.asarray(y, dtype=float)
    return reaction_part(U, D, P, params)


def reaction_part(
    U: np.ndarray | float,
    D: np.ndarray | float,
    P: np.ndarray | float,
    params: RescueParams,
) -> np.ndarray:
    """Reaction terms shared by the ODE and PDE models.

    Inputs may be scalars or arrays of matching shape.  The returned array has leading
    axis 0 corresponding to (dU, dD, dP).
    """
    U_arr = np.asarray(U, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    P_arr = np.asarray(P, dtype=float)
    total_prey = U_arr + D_arr
    resource = 1.0 - total_prey / params.K

    dU = (
        params.r_U * U_arr * resource
        - params.a_U * U_arr * P_arr
        + params.mu_DU * D_arr
        - params.mu_UD * U_arr
    )
    dD = (
        params.r_D * D_arr * resource
        - params.a_D * D_arr * P_arr
        + params.mu_UD * U_arr
        - params.mu_DU * D_arr
    )
    dP = params.e * (params.a_U * U_arr + params.a_D * D_arr) * P_arr - params.m * P_arr
    return np.stack([dU, dD, dP], axis=0)


def prey_only_equilibrium(params: RescueParams) -> tuple[float, float, float]:
    """Return the homogeneous prey-only equilibrium (U0, D0, P0=0)."""
    denom = params.mu_UD + params.mu_DU
    if denom <= 0:
        raise ValueError("mu_UD + mu_DU must be positive for the prey-only equilibrium.")
    q = params.mu_DU / denom
    U = params.K * q
    D = params.K * (1.0 - q)
    return U, D, 0.0


def prey_only_invasion_threshold(params: RescueParams) -> float:
    """Linear predator invasion threshold into the homogeneous prey-only state.

    This is not the nonlinear rescue threshold.  It only checks whether a rare
    predator can initially increase in the prey-only equilibrium.
    """
    denom = params.mu_UD + params.mu_DU
    if denom <= 0:
        raise ValueError("mu_UD + mu_DU must be positive for m_inv^ODE.")
    q_prey_only = params.mu_DU / denom
    return params.e * params.K * attack_weight(q_prey_only, params)


def coexistence_scalar_G(q: float, params: RescueParams, m: float | None = None) -> float:
    """Scalar coexistence equation G(q;m)=0 for positive equilibria.

    Given q=U/(U+D), the predator equation fixes
    N=m/[e(a_D+(a_U-a_D)q)].  The total-prey equation then fixes P.  Substituting
    these into the prey-type ratio equation gives G(q;m)=0.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1).")

    current_m = params.m if m is None else m
    A = attack_weight(q, params)
    R = growth_weight(q, params)
    N = current_m / (params.e * A)
    resource = 1.0 - N / params.K
    P = R * resource / A

    return (
        (params.r_U - params.r_D) * resource
        - (params.a_U - params.a_D) * P
        + params.mu_DU / q
        - params.mu_UD / (1.0 - q)
    )


def reconstruct_equilibrium(
    q: float,
    params: RescueParams,
    m: float | None = None,
) -> CoexistenceEquilibrium:
    """Reconstruct (N, U, D, P) from a root q of G(q;m)."""
    current_m = params.m if m is None else m
    A = attack_weight(q, params)
    R = growth_weight(q, params)
    N = current_m / (params.e * A)
    resource = 1.0 - N / params.K
    P = R * resource / A
    U = q * N
    D = (1.0 - q) * N
    residual = float(np.linalg.norm(reaction_part(U, D, P, params.with_updates(m=current_m))))
    return CoexistenceEquilibrium(q=q, N=N, U=U, D=D, P=P, residual=residual)


def solve_coexistence_equilibria(
    params: RescueParams,
    m: float | None = None,
    n_grid: int = 1200,
    q_eps: float = 1.0e-6,
    residual_tol: float = 1.0e-7,
) -> list[CoexistenceEquilibrium]:
    """Find positive coexistence equilibria by scanning q in (0,1)."""
    current_m = params.m if m is None else m
    qs = np.linspace(q_eps, 1.0 - q_eps, n_grid)

    values = np.empty_like(qs)
    for i, q in enumerate(qs):
        try:
            values[i] = coexistence_scalar_G(float(q), params, current_m)
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
                lambda z: coexistence_scalar_G(z, params, current_m),
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
        eq = reconstruct_equilibrium(root, params, current_m)
        if eq.U > 0.0 and eq.D > 0.0 and eq.P > 0.0 and eq.N > 0.0 and eq.residual < residual_tol:
            equilibria.append(eq)
    return equilibria


def jacobian_reaction(U: float, D: float, P: float, params: RescueParams) -> np.ndarray:
    """Analytic Jacobian of the reaction terms at (U,D,P)."""
    return np.array(
        [
            [
                params.r_U * (1.0 - (2.0 * U + D) / params.K) - params.a_U * P - params.mu_UD,
                -params.r_U * U / params.K + params.mu_DU,
                -params.a_U * U,
            ],
            [
                -params.r_D * D / params.K + params.mu_UD,
                params.r_D * (1.0 - (U + 2.0 * D) / params.K) - params.a_D * P - params.mu_DU,
                -params.a_D * D,
            ],
            [
                params.e * params.a_U * P,
                params.e * params.a_D * P,
                params.e * (params.a_U * U + params.a_D * D) - params.m,
            ],
        ],
        dtype=float,
    )


def ode_local_stability(J: np.ndarray, tol: float = 1.0e-9) -> tuple[bool, np.ndarray]:
    """Return whether all ODE eigenvalues have real part below -tol."""
    eigvals = np.linalg.eigvals(J)
    return bool(np.max(eigvals.real) < -tol), eigvals


def turing_scan(
    params: RescueParams,
    equilibrium: CoexistenceEquilibrium,
    n_max: int = 80,
    tol: float = 1.0e-9,
) -> TuringScanResult:
    """Scan Neumann modes k_n=n*pi/L for diffusion-driven instability."""
    J = jacobian_reaction(equilibrium.U, equilibrium.D, equilibrium.P, params)
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
    turing_unstable = bool(ode_stable and has_unstable_spatial_mode)

    return TuringScanResult(
        ks=ks,
        eigenvalues=eigvals,
        max_real_by_mode=max_real,
        ode_eigenvalues=ode_eigs,
        ode_stable=ode_stable,
        has_unstable_spatial_mode=has_unstable_spatial_mode,
        turing_unstable=turing_unstable,
        dominant_mode=(dominant_idx + 1) if dominant_idx is not None else None,
        dominant_k=float(ks[dominant_idx]) if dominant_idx is not None else None,
        dominant_growth=dominant_growth,
    )


def require_single_equilibrium(params: RescueParams, m: float | None = None) -> CoexistenceEquilibrium:
    """Return the first positive coexistence equilibrium or raise a useful error."""
    equilibria = solve_coexistence_equilibria(params, m=m)
    if not equilibria:
        current_m = params.m if m is None else m
        raise ValueError(f"No positive coexistence equilibrium found for m={current_m:.6g}.")
    return equilibria[0]

