"""Nonlinear ODE/PDE simulations for the 1D defended-prey rescue model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .turing_rescue_model import (
    CoexistenceEquilibrium,
    RescueParams,
    prey_only_invasion_threshold,
    reaction_ode,
    reaction_part,
    require_single_equilibrium,
)


@dataclass(frozen=True)
class Diagnostics:
    B_P: float
    B_U: float
    B_D: float
    O_PU: float
    mean_edible: float
    var_U: float
    var_D: float
    var_P: float
    persistent: bool
    min_value: float
    negative_detected: bool


@dataclass(frozen=True)
class ODEResult:
    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    persistent: bool
    min_value: float
    negative_detected: bool


@dataclass(frozen=True)
class PDEResult:
    t: np.ndarray
    x: np.ndarray
    U: np.ndarray
    D: np.ndarray
    P: np.ndarray
    success: bool
    message: str
    diagnostics: Diagnostics
    predator_biomass_time: np.ndarray


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    m_low: float
    m_high: float
    iterations: int
    history: list[tuple[float, bool, float]]
    epsilon: float
    T: float


def grid_1d(params: RescueParams) -> np.ndarray:
    if params.n_x < 3:
        raise ValueError("n_x must be at least 3 for a 1D finite-difference grid.")
    return np.linspace(0.0, params.L, params.n_x)


def laplacian_neumann(values: np.ndarray, dx: float) -> np.ndarray:
    """Second-order finite-difference Laplacian with no-flux boundaries."""
    lap = np.empty_like(values)
    inv_dx2 = 1.0 / (dx * dx)
    lap[1:-1] = (values[:-2] - 2.0 * values[1:-1] + values[2:]) * inv_dx2
    lap[0] = 2.0 * (values[1] - values[0]) * inv_dx2
    lap[-1] = 2.0 * (values[-2] - values[-1]) * inv_dx2
    return lap


def pack_state(U: np.ndarray, D: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.concatenate([U, D, P]).astype(float)


def unpack_state(y: np.ndarray, n_x: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U = y[:n_x]
    D = y[n_x : 2 * n_x]
    P = y[2 * n_x : 3 * n_x]
    return U, D, P


def homogeneous_profile(
    params: RescueParams,
    equilibrium: CoexistenceEquilibrium,
    perturbation_amplitude: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create homogeneous profiles, optionally with small random perturbations."""
    rng = np.random.default_rng(seed)
    shape = params.n_x
    U = np.full(shape, equilibrium.U, dtype=float)
    D = np.full(shape, equilibrium.D, dtype=float)
    P = np.full(shape, equilibrium.P, dtype=float)
    if perturbation_amplitude > 0.0:
        for arr in (U, D, P):
            arr *= 1.0 + perturbation_amplitude * rng.standard_normal(shape)
            np.maximum(arr, 1.0e-12, out=arr)
    return U, D, P


def default_threshold_initial_state(
    params: RescueParams,
    m_reference: float,
    perturbation_amplitude: float = 1.0e-3,
    seed: int | None = 1,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Initial ODE/PDE states from a low-stress coexistence equilibrium."""
    eq = require_single_equilibrium(params.with_updates(m=m_reference))
    y0_ode = np.array([eq.U, eq.D, eq.P], dtype=float)
    y0_pde = homogeneous_profile(params, eq, perturbation_amplitude, seed)
    return y0_ode, y0_pde


def pde_rhs_factory(
    params: RescueParams,
    clip_negative_rhs: bool = False,
) -> tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[], float]]:
    x = grid_1d(params)
    dx = x[1] - x[0]
    min_seen = {"value": float("inf")}

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        U, D, P = unpack_state(y, params.n_x)
        current_min = float(np.min(y))
        if current_min < min_seen["value"]:
            min_seen["value"] = current_min
        if clip_negative_rhs:
            U = np.maximum(U, 0.0)
            D = np.maximum(D, 0.0)
            P = np.maximum(P, 0.0)

        reactions = reaction_part(U, D, P, params)
        dU = params.delta_U * laplacian_neumann(U, dx) + reactions[0]
        dD = params.delta_D * laplacian_neumann(D, dx) + reactions[1]
        dP = params.delta_P * laplacian_neumann(P, dx) + reactions[2]
        return pack_state(dU, dD, dP)

    def get_min_seen() -> float:
        return min_seen["value"]

    return rhs, get_min_seen


def compute_diagnostics(
    x: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
    P: np.ndarray,
    params: RescueParams,
    epsilon: float,
    min_seen: float | None = None,
    negative_tol: float = -1.0e-8,
) -> Diagnostics:
    B_U = float(np.trapezoid(U, x))
    B_D = float(np.trapezoid(D, x))
    B_P = float(np.trapezoid(P, x))
    denom = B_P * B_U
    O_PU = float(np.trapezoid(P * U, x) / denom) if denom > 0.0 else float("nan")
    mean_edible = float(np.trapezoid(params.a_U * U + params.a_D * D, x) / params.L)
    min_value = float(np.min([np.min(U), np.min(D), np.min(P), min_seen if min_seen is not None else np.inf]))
    return Diagnostics(
        B_P=B_P,
        B_U=B_U,
        B_D=B_D,
        O_PU=O_PU,
        mean_edible=mean_edible,
        var_U=float(np.var(U)),
        var_D=float(np.var(D)),
        var_P=float(np.var(P)),
        persistent=bool(B_P > epsilon),
        min_value=min_value,
        negative_detected=bool(min_value < negative_tol),
    )


def simulate_pde(
    params: RescueParams,
    T: float,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    equilibrium: CoexistenceEquilibrium | None = None,
    perturbation_amplitude: float = 1.0e-3,
    seed: int | None = 1,
    epsilon: float = 1.0e-5,
    method: str = "BDF",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    n_time: int = 200,
    clip_negative_rhs: bool = False,
    max_step: float | None = None,
) -> PDEResult:
    """Simulate the 1D PDE with Neumann boundary conditions."""
    x = grid_1d(params)
    if initial_state is None:
        eq = equilibrium or require_single_equilibrium(params)
        initial_state = homogeneous_profile(params, eq, perturbation_amplitude, seed)

    U0, D0, P0 = [np.asarray(arr, dtype=float).copy() for arr in initial_state]
    if any(arr.shape != (params.n_x,) for arr in (U0, D0, P0)):
        raise ValueError("Initial PDE arrays must all have shape (n_x,).")

    y0 = pack_state(U0, D0, P0)
    rhs, get_min_seen = pde_rhs_factory(params, clip_negative_rhs=clip_negative_rhs)
    t_eval = np.linspace(0.0, T, n_time)
    solve_kwargs = {}
    if max_step is not None:
        solve_kwargs["max_step"] = max_step

    sol = solve_ivp(
        rhs,
        (0.0, T),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **solve_kwargs,
    )

    if sol.y.size == 0:
        raise RuntimeError(f"PDE solver returned no solution: {sol.message}")

    U_T, D_T, P_T = unpack_state(sol.y[:, -1], params.n_x)
    diagnostics = compute_diagnostics(x, U_T, D_T, P_T, params, epsilon, get_min_seen())

    predator_biomass_time = np.empty(sol.y.shape[1], dtype=float)
    for idx in range(sol.y.shape[1]):
        _, _, P_i = unpack_state(sol.y[:, idx], params.n_x)
        predator_biomass_time[idx] = np.trapezoid(P_i, x)

    return PDEResult(
        t=sol.t,
        x=x,
        U=U_T,
        D=D_T,
        P=P_T,
        success=bool(sol.success),
        message=sol.message,
        diagnostics=diagnostics,
        predator_biomass_time=predator_biomass_time,
    )


def simulate_ode(
    params: RescueParams,
    T: float,
    y0: np.ndarray | None = None,
    epsilon: float = 1.0e-5,
    method: str = "BDF",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    n_time: int = 400,
) -> ODEResult:
    """Simulate the spatially homogeneous ODE model."""
    if y0 is None:
        eq = require_single_equilibrium(params)
        y0 = np.array([eq.U, eq.D, eq.P], dtype=float)
    y0 = np.asarray(y0, dtype=float)
    t_eval = np.linspace(0.0, T, n_time)
    sol = solve_ivp(
        lambda t, y: reaction_ode(t, y, params),
        (0.0, T),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    min_value = float(np.min(sol.y)) if sol.y.size else float("nan")
    final_P = float(sol.y[2, -1]) if sol.y.size else float("nan")
    return ODEResult(
        t=sol.t,
        y=sol.y,
        success=bool(sol.success),
        message=sol.message,
        persistent=bool(final_P > epsilon),
        min_value=min_value,
        negative_detected=bool(min_value < -1.0e-8),
    )


def compute_mc_ode(
    params: RescueParams,
    m_low: float,
    m_high: float,
    epsilon: float,
    T: float,
    y0: np.ndarray | None = None,
    tol_m: float = 1.0e-3,
    max_iter: int = 24,
) -> ThresholdResult:
    """Compute nonlinear ODE persistence threshold by bisection."""
    if y0 is None:
        y0, _ = default_threshold_initial_state(params, m_low, perturbation_amplitude=0.0)

    def classify(m_value: float) -> tuple[bool, float]:
        result = simulate_ode(params.with_updates(m=m_value), T=T, y0=y0, epsilon=epsilon)
        if not result.success:
            return False, float("nan")
        return result.persistent, float(result.y[2, -1])

    low_persistent, low_measure = classify(m_low)
    high_persistent, high_measure = classify(m_high)
    if not low_persistent or high_persistent:
        raise ValueError(
            "ODE bisection bracket must satisfy persistent(m_low)=True and "
            "persistent(m_high)=False. "
            f"Got low={low_persistent} ({low_measure:.3e}), "
            f"high={high_persistent} ({high_measure:.3e})."
        )

    history: list[tuple[float, bool, float]] = [(m_low, low_persistent, low_measure), (m_high, high_persistent, high_measure)]
    lo, hi = float(m_low), float(m_high)
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        persistent, measure = classify(mid)
        history.append((mid, persistent, measure))
        if persistent:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol_m:
            break
    return ThresholdResult(
        threshold=lo,
        m_low=m_low,
        m_high=m_high,
        iterations=iteration,
        history=history,
        epsilon=epsilon,
        T=T,
    )


def compute_mc_pde(
    params: RescueParams,
    m_low: float,
    m_high: float,
    epsilon: float,
    T: float,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    perturbation_amplitude: float = 1.0e-3,
    seed: int | None = 1,
    tol_m: float = 1.0e-3,
    max_iter: int = 18,
    method: str = "BDF",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    n_time: int = 120,
) -> ThresholdResult:
    """Compute nonlinear PDE persistence threshold by bisection."""
    if initial_state is None:
        _, initial_state = default_threshold_initial_state(params, m_low, perturbation_amplitude, seed)

    def classify(m_value: float) -> tuple[bool, float]:
        result = simulate_pde(
            params.with_updates(m=m_value),
            T=T,
            initial_state=initial_state,
            epsilon=epsilon,
            method=method,
            rtol=rtol,
            atol=atol,
            n_time=n_time,
        )
        if not result.success:
            return False, float("nan")
        return result.diagnostics.persistent, result.diagnostics.B_P

    low_persistent, low_measure = classify(m_low)
    high_persistent, high_measure = classify(m_high)
    if not low_persistent or high_persistent:
        raise ValueError(
            "PDE bisection bracket must satisfy persistent(m_low)=True and "
            "persistent(m_high)=False. "
            f"Got low={low_persistent} ({low_measure:.3e}), "
            f"high={high_persistent} ({high_measure:.3e})."
        )

    history: list[tuple[float, bool, float]] = [(m_low, low_persistent, low_measure), (m_high, high_persistent, high_measure)]
    lo, hi = float(m_low), float(m_high)
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        persistent, measure = classify(mid)
        history.append((mid, persistent, measure))
        if persistent:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol_m:
            break
    return ThresholdResult(
        threshold=lo,
        m_low=m_low,
        m_high=m_high,
        iterations=iteration,
        history=history,
        epsilon=epsilon,
        T=T,
    )


def classify_delta(delta_mc: float, tol: float = 1.0e-3) -> str:
    if delta_mc > tol:
        return "m_c^PDE > m_c^ODE"
    if delta_mc < -tol:
        return "m_c^PDE < m_c^ODE"
    return "no resolvable threshold difference at this tolerance"


def invasion_sanity_report(params: RescueParams) -> dict[str, float]:
    """Return the analytic prey-only invasion threshold for reporting only."""
    m_inv = prey_only_invasion_threshold(params)
    U0, D0, _ = prey_only_equilibrium_for_report(params)
    return {"m_inv_ode": float(m_inv), "prey_only_U": float(U0), "prey_only_D": float(D0)}


def prey_only_equilibrium_for_report(params: RescueParams) -> tuple[float, float, float]:
    from .turing_rescue_model import prey_only_equilibrium

    return prey_only_equilibrium(params)
