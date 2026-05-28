"""Roy-style defended/undefended prey reaction-diffusion model.

This module implements the dimensionless model structure used by Roy et al.
for undefended prey ``u``, defended prey ``v``, predator ``w``, and free
ecological space ``z = 1/kappa - u - v - w``.  It is separate from the
minimal mass-action models, which are now treated as negative controls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RoyParams:
    kappa: float = 0.15
    xi: float = 0.55
    gamma: float = 3.73
    rho: float = 1.76
    eta: float = 0.005
    beta1: float = 0.5
    delta: float = 0.1
    mu: float = 0.8
    D_u: float = 0.01
    D_v: float = 0.01
    D_w: float = 1.0
    L: float = 20.0
    n_x: int = 200

    def with_updates(self, **kwargs: float) -> "RoyParams":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RoyEquilibrium:
    u: float
    v: float
    w: float
    z: float
    residual: float


@dataclass(frozen=True)
class RoyTuringScan:
    ks: np.ndarray
    max_real_by_k: np.ndarray
    eigenvalues: np.ndarray
    ode_eigenvalues: np.ndarray
    ode_stable: bool
    max_spatial_growth: float
    best_k: float
    unstable_windows: list[tuple[float, float]]
    turing_unstable: bool
    dominant_mode: int | None = None


@dataclass(frozen=True)
class RoyPDEResult:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    success: bool
    message: str
    mean_w: float
    var_u: float
    var_v: float
    var_w: float
    min_value: float
    negative_detected: bool


def validate_params(params: RoyParams) -> None:
    if params.kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    if not 0.0 <= params.eta < params.kappa:
        raise ValueError("eta must satisfy 0 <= eta < kappa.")
    if params.rho <= 0.0 or params.beta1 <= 0.0:
        raise ValueError("rho and beta1 must be positive.")
    if min(params.D_u, params.D_v, params.D_w) < 0.0:
        raise ValueError("diffusion coefficients must be non-negative.")
    if params.n_x < 3:
        raise ValueError("n_x must be at least 3.")


def theta(params: RoyParams) -> float:
    validate_params(params)
    return params.kappa / (params.kappa - params.eta)


def beta2_sharp(params: RoyParams) -> float:
    """Sharp-switch limit for the defended-prey conversion coefficient.

    Roy et al. define beta2 from a bounded tanh transition.  In the sharp limit
    this equals beta1/theta for eta < kappa/2, zero at eta = kappa/2, and
    -beta1/theta for eta > kappa/2.
    """
    th = theta(params)
    center = 0.5 * params.kappa
    if np.isclose(params.eta, center):
        return 0.0
    return params.beta1 / th if params.eta < center else -params.beta1 / th


def free_space(u: np.ndarray | float, v: np.ndarray | float, w: np.ndarray | float, params: RoyParams) -> np.ndarray | float:
    return 1.0 / params.kappa - u - v - w


def reaction_part(
    u: np.ndarray | float,
    v: np.ndarray | float,
    w: np.ndarray | float,
    params: RoyParams,
) -> np.ndarray:
    th = theta(params)
    beta2 = beta2_sharp(params)
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    z = free_space(u_arr, v_arr, w_arr, params)
    prey_sum = 1.0 + u_arr + v_arr

    du = u_arr * (z / (params.rho + params.kappa * u_arr) - params.xi - params.gamma * w_arr / prey_sum)
    dv = v_arr * (z / (params.rho + params.kappa * th * v_arr) - params.xi - params.gamma * w_arr / (th * prey_sum))
    dw = w_arr * (((params.beta1 * u_arr + beta2 * v_arr) / prey_sum) * z - params.delta - params.mu * w_arr)
    return np.stack([du, dv, dw], axis=0)


def reaction_ode(_t: float, y: np.ndarray, params: RoyParams) -> np.ndarray:
    u, v, w = np.asarray(y, dtype=float)
    return reaction_part(u, v, w, params)


def reaction_jacobian(u: float, v: float, w: float, params: RoyParams) -> np.ndarray:
    th = theta(params)
    beta2 = beta2_sharp(params)
    z = free_space(u, v, w, params)
    prey_sum = 1.0 + u + v
    A_u = params.rho + params.kappa * u
    A_v = params.rho + params.kappa * th * v

    f = z / A_u - params.xi - params.gamma * w / prey_sum
    df_du = (-A_u - params.kappa * z) / (A_u * A_u) + params.gamma * w / (prey_sum * prey_sum)
    df_dv = -1.0 / A_u + params.gamma * w / (prey_sum * prey_sum)
    df_dw = -1.0 / A_u - params.gamma / prey_sum

    g = z / A_v - params.xi - params.gamma * w / (th * prey_sum)
    dg_du = -1.0 / A_v + params.gamma * w / (th * prey_sum * prey_sum)
    dg_dv = (-A_v - params.kappa * th * z) / (A_v * A_v) + params.gamma * w / (th * prey_sum * prey_sum)
    dg_dw = -1.0 / A_v - params.gamma / (th * prey_sum)

    numerator = params.beta1 * u + beta2 * v
    edible = numerator / prey_sum
    d_edible_du = (params.beta1 * prey_sum - numerator) / (prey_sum * prey_sum)
    d_edible_dv = (beta2 * prey_sum - numerator) / (prey_sum * prey_sum)

    h = edible * z - params.delta - params.mu * w
    dh_du = d_edible_du * z - edible
    dh_dv = d_edible_dv * z - edible
    dh_dw = -edible - params.mu

    return np.array(
        [
            [f + u * df_du, u * df_dv, u * df_dw],
            [v * dg_du, g + v * dg_dv, v * dg_dw],
            [w * dh_du, w * dh_dv, h + w * dh_dw],
        ],
        dtype=float,
    )


def solve_homogeneous_equilibria(
    params: RoyParams,
    n_seeds: int = 5,
    residual_tol: float = 1.0e-8,
) -> list[RoyEquilibrium]:
    """Find positive homogeneous coexistence equilibria by constrained multistart."""
    validate_params(params)
    capacity = 1.0 / params.kappa
    values = np.linspace(0.03 * capacity, 0.85 * capacity, n_seeds)
    w_values = np.linspace(0.02 * capacity, 0.60 * capacity, max(3, n_seeds - 1))
    seeds: list[np.ndarray] = []
    for u0 in values:
        for v0 in values:
            for w0 in w_values:
                if u0 + v0 + w0 < 0.95 * capacity:
                    seeds.append(np.array([u0, v0, w0], dtype=float))

    lower = np.full(3, 1.0e-12)
    upper = np.full(3, capacity - 1.0e-12)
    equilibria: list[RoyEquilibrium] = []
    for seed in seeds:
        fit = least_squares(
            lambda y: reaction_part(y[0], y[1], y[2], params),
            seed,
            bounds=(lower, upper),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=5000,
        )
        u_star, v_star, w_star = fit.x
        z_star = float(free_space(u_star, v_star, w_star, params))
        residual = float(np.linalg.norm(reaction_part(u_star, v_star, w_star, params)))
        if residual > residual_tol or min(u_star, v_star, w_star, z_star) <= 1.0e-9:
            continue
        eq = RoyEquilibrium(float(u_star), float(v_star), float(w_star), z_star, residual)
        if not any(np.linalg.norm(np.array([eq.u - old.u, eq.v - old.v, eq.w - old.w])) < 1.0e-6 for old in equilibria):
            equilibria.append(eq)
    return sorted(equilibria, key=lambda item: (item.u, item.v, item.w))


def require_positive_equilibrium(params: RoyParams) -> RoyEquilibrium:
    equilibria = solve_homogeneous_equilibria(params)
    if not equilibria:
        raise ValueError("No positive Roy-style coexistence equilibrium found.")
    return equilibria[0]


def _unstable_windows(ks: np.ndarray, growth: np.ndarray, tol: float) -> list[tuple[float, float]]:
    unstable = growth > tol
    windows: list[tuple[float, float]] = []
    start: int | None = None
    for idx, is_unstable in enumerate(unstable):
        if is_unstable and start is None:
            start = idx
        elif not is_unstable and start is not None:
            windows.append((float(ks[start]), float(ks[idx - 1])))
            start = None
    if start is not None:
        windows.append((float(ks[start]), float(ks[-1])))
    return windows


def _scan_at_ks(params: RoyParams, equilibrium: RoyEquilibrium, ks: np.ndarray, tol: float) -> RoyTuringScan:
    J = reaction_jacobian(equilibrium.u, equilibrium.v, equilibrium.w, params)
    ode_eigs = np.linalg.eigvals(J)
    ode_stable = bool(np.max(ode_eigs.real) < -tol)
    diffusion = np.diag([params.D_u, params.D_v, params.D_w])
    eigvals = np.empty((len(ks), 3), dtype=complex)
    growth = np.empty(len(ks), dtype=float)
    for idx, k in enumerate(ks):
        eigvals[idx] = np.linalg.eigvals(J - (k * k) * diffusion)
        growth[idx] = float(np.max(eigvals[idx].real))
    best_idx = int(np.argmax(growth))
    return RoyTuringScan(
        ks=ks,
        max_real_by_k=growth,
        eigenvalues=eigvals,
        ode_eigenvalues=ode_eigs,
        ode_stable=ode_stable,
        max_spatial_growth=float(growth[best_idx]),
        best_k=float(ks[best_idx]),
        unstable_windows=_unstable_windows(ks, growth, tol),
        turing_unstable=bool(ode_stable and np.any(growth > tol)),
    )


def continuous_turing_scan(
    params: RoyParams,
    equilibrium: RoyEquilibrium,
    k_min: float = 1.0e-4,
    k_max: float = 20.0,
    n_k: int = 600,
    tol: float = 1.0e-9,
) -> RoyTuringScan:
    if k_min < 0.0 or k_max <= k_min or n_k < 2:
        raise ValueError("Require 0 <= k_min < k_max and n_k >= 2.")
    return _scan_at_ks(params, equilibrium, np.linspace(k_min, k_max, n_k), tol)


def neumann_turing_scan(
    params: RoyParams,
    equilibrium: RoyEquilibrium,
    n_max: int = 160,
    tol: float = 1.0e-9,
) -> RoyTuringScan:
    modes = np.arange(1, n_max + 1)
    scan = _scan_at_ks(params, equilibrium, modes * np.pi / params.L, tol)
    best_idx = int(np.argmax(scan.max_real_by_k))
    return RoyTuringScan(
        ks=scan.ks,
        max_real_by_k=scan.max_real_by_k,
        eigenvalues=scan.eigenvalues,
        ode_eigenvalues=scan.ode_eigenvalues,
        ode_stable=scan.ode_stable,
        max_spatial_growth=scan.max_spatial_growth,
        best_k=scan.best_k,
        unstable_windows=scan.unstable_windows,
        turing_unstable=scan.turing_unstable,
        dominant_mode=int(modes[best_idx]),
    )


def grid_1d(params: RoyParams) -> np.ndarray:
    return np.linspace(0.0, params.L, params.n_x)


def laplacian_neumann(values: np.ndarray, dx: float) -> np.ndarray:
    lap = np.empty_like(values)
    inv_dx2 = 1.0 / (dx * dx)
    lap[1:-1] = (values[:-2] - 2.0 * values[1:-1] + values[2:]) * inv_dx2
    lap[0] = 2.0 * (values[1] - values[0]) * inv_dx2
    lap[-1] = 2.0 * (values[-2] - values[-1]) * inv_dx2
    return lap


def pack_state(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([u, v, w]).astype(float)


def unpack_state(y: np.ndarray, n_x: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return y[:n_x], y[n_x : 2 * n_x], y[2 * n_x : 3 * n_x]


def perturbed_equilibrium_profile(
    params: RoyParams,
    equilibrium: RoyEquilibrium,
    amplitude: float = 1.0e-5,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = params.n_x
    u = np.full(shape, equilibrium.u, dtype=float)
    v = np.full(shape, equilibrium.v, dtype=float)
    w = np.full(shape, equilibrium.w, dtype=float)
    if amplitude > 0.0:
        for arr in (u, v, w):
            arr *= 1.0 + amplitude * rng.standard_normal(shape)
            np.maximum(arr, 1.0e-12, out=arr)
    return u, v, w


def pde_rhs_factory(params: RoyParams) -> tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[], float]]:
    x = grid_1d(params)
    dx = x[1] - x[0]
    min_seen = {"value": float("inf")}

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u, v, w = unpack_state(y, params.n_x)
        min_seen["value"] = min(min_seen["value"], float(np.min(y)))
        reactions = reaction_part(u, v, w, params)
        du = params.D_u * laplacian_neumann(u, dx) + reactions[0]
        dv = params.D_v * laplacian_neumann(v, dx) + reactions[1]
        dw = params.D_w * laplacian_neumann(w, dx) + reactions[2]
        return pack_state(du, dv, dw)

    def get_min_seen() -> float:
        return min_seen["value"]

    return rhs, get_min_seen


def simulate_pde_1d(
    params: RoyParams,
    equilibrium: RoyEquilibrium,
    T: float = 80.0,
    perturbation_amplitude: float = 1.0e-5,
    seed: int = 1,
    method: str = "BDF",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    n_time: int = 240,
) -> RoyPDEResult:
    x = grid_1d(params)
    initial = perturbed_equilibrium_profile(params, equilibrium, perturbation_amplitude, seed)
    rhs, get_min_seen = pde_rhs_factory(params)
    sol = solve_ivp(
        rhs,
        (0.0, T),
        pack_state(*initial),
        method=method,
        t_eval=np.linspace(0.0, T, n_time),
        rtol=rtol,
        atol=atol,
    )
    if sol.y.size == 0:
        raise RuntimeError(f"Roy PDE solver returned no solution: {sol.message}")
    u_T, v_T, w_T = unpack_state(sol.y[:, -1], params.n_x)
    min_value = float(min(np.min(u_T), np.min(v_T), np.min(w_T), get_min_seen()))
    return RoyPDEResult(
        t=sol.t,
        x=x,
        u=u_T,
        v=v_T,
        w=w_T,
        success=bool(sol.success),
        message=sol.message,
        mean_w=float(np.trapezoid(w_T, x) / params.L),
        var_u=float(np.var(u_T)),
        var_v=float(np.var(v_T)),
        var_w=float(np.var(w_T)),
        min_value=min_value,
        negative_detected=bool(min_value < -1.0e-8),
    )
