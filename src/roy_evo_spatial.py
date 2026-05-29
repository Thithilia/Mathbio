"""Roy-Yamamichi eco-evolutionary ODE helpers.

This module implements the well-mixed part of the proposed spatial
eco-evolutionary model.  No PDE solver is included here; Step 09A uses this
ODE gate to decide whether a spatial extension is scientifically warranted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class RoyEvoParams:
    kappa: float = 0.15
    xi: float = 0.55
    r_u: float = 1.0
    r_v: float = 0.65
    a_u: float = 1.0
    a_v: float = 0.35
    b_u: float = 0.55
    b_v: float = 0.30
    m: float = 0.10
    mu: float = 0.20
    nu: float = 0.05

    def with_updates(self, **kwargs: float) -> "RoyEvoParams":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RoyEvoTrajectory:
    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    stress: float
    evolve: bool


@dataclass(frozen=True)
class RoyEvoPDEConfig:
    n_x: int = 64
    n_y: int = 64
    L_x: float = 20.0
    L_y: float = 20.0
    dt: float = 0.01
    T: float = 300.0
    record_every: int = 100
    D_n: float = 0.01
    D_w: float = 0.01
    D_q: float = 0.005
    perturbation_amplitude: float = 1.0e-5
    seed: int = 1
    clip_q: bool = True


@dataclass(frozen=True)
class RoyEvoPDEResult:
    t: np.ndarray
    mean_n_time: np.ndarray
    mean_w_time: np.ndarray
    mean_q_time: np.ndarray
    var_n_time: np.ndarray
    var_w_time: np.ndarray
    var_q_time: np.ndarray
    min_z_time: np.ndarray
    n: np.ndarray
    w: np.ndarray
    q: np.ndarray
    diagnostics: dict[str, object]


def validate_evo_params(params: RoyEvoParams) -> None:
    if params.kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    if min(params.r_u, params.r_v, params.a_u, params.a_v, params.b_u, params.b_v) < 0.0:
        raise ValueError("tradeoff coefficients must be non-negative.")
    if min(params.m, params.mu, params.nu) < 0.0:
        raise ValueError("mortality and evolution-rate parameters must be non-negative.")
    if params.r_v >= params.r_u:
        raise ValueError("defended prey must have lower growth: r_v < r_u.")
    if params.a_v >= params.a_u:
        raise ValueError("defended prey must have lower attack pressure: a_v < a_u.")


def free_space_evo(n: np.ndarray | float, w: np.ndarray | float, params: RoyEvoParams) -> np.ndarray | float:
    return 1.0 / params.kappa - np.asarray(n) - np.asarray(w)


def r_of_q(q: np.ndarray | float, params: RoyEvoParams) -> np.ndarray | float:
    q_arr = np.asarray(q)
    return params.r_u * (1.0 - q_arr) + params.r_v * q_arr


def a_of_q(q: np.ndarray | float, params: RoyEvoParams) -> np.ndarray | float:
    q_arr = np.asarray(q)
    return params.a_u * (1.0 - q_arr) + params.a_v * q_arr


def b_of_q(q: np.ndarray | float, params: RoyEvoParams) -> np.ndarray | float:
    q_arr = np.asarray(q)
    return params.b_u * (1.0 - q_arr) + params.b_v * q_arr


def selection_gradient(
    n: np.ndarray | float,
    w: np.ndarray | float,
    q: np.ndarray | float,
    params: RoyEvoParams,
) -> np.ndarray | float:
    del q
    z = free_space_evo(n, w, params)
    return (params.r_v - params.r_u) * z - (params.a_v - params.a_u) * np.asarray(w)


def reaction_ode_evo(
    _t: float,
    y: Iterable[float],
    params: RoyEvoParams,
    stress: float = 0.0,
    evolve: bool = True,
) -> np.ndarray:
    validate_evo_params(params)
    n, w, q = np.asarray(y, dtype=float)
    z = free_space_evo(n, w, params)
    r_q = r_of_q(q, params)
    a_q = a_of_q(q, params)
    b_q = b_of_q(q, params)
    m_eff = params.m + stress
    gradient = selection_gradient(n, w, q, params)

    dn = n * (r_q * z - params.xi - a_q * w)
    dw = w * (b_q * n * z - m_eff - params.mu * w)
    dq = params.nu * q * (1.0 - q) * gradient if evolve else 0.0
    return np.array([dn, dw, dq], dtype=float)


def grid_2d_evo(config: RoyEvoPDEConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    if config.n_x < 3 or config.n_y < 3:
        raise ValueError("n_x and n_y must be at least 3.")
    if min(config.L_x, config.L_y, config.dt, config.T) <= 0.0:
        raise ValueError("L_x, L_y, dt, and T must be positive.")
    x = np.linspace(0.0, config.L_x, config.n_x)
    y = np.linspace(0.0, config.L_y, config.n_y)
    return x, y, float(x[1] - x[0]), float(y[1] - y[0])


def laplacian_neumann_2d_evo(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a 2D array.")
    left = np.empty_like(array)
    right = np.empty_like(array)
    down = np.empty_like(array)
    up = np.empty_like(array)

    left[:, 1:] = array[:, :-1]
    left[:, 0] = array[:, 1]
    right[:, :-1] = array[:, 1:]
    right[:, -1] = array[:, -2]
    down[1:, :] = array[:-1, :]
    down[0, :] = array[1, :]
    up[:-1, :] = array[1:, :]
    up[-1, :] = array[-2, :]
    return (left - 2.0 * array + right) / (dx * dx) + (down - 2.0 * array + up) / (dy * dy)


def check_evo_pde_cfl(config: RoyEvoPDEConfig, safety: float = 0.22) -> None:
    _, _, dx, dy = grid_2d_evo(config)
    max_diffusion = max(config.D_n, config.D_w, config.D_q)
    if max_diffusion < 0.0:
        raise ValueError("diffusion coefficients must be non-negative.")
    if max_diffusion == 0.0:
        return
    stable_dt = safety / (max_diffusion * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if config.dt > stable_dt:
        raise ValueError(f"dt={config.dt:g} exceeds explicit diffusion stability estimate {stable_dt:g}.")


def initial_state_from_ode_equilibrium(
    equilibrium: dict[str, object] | Iterable[float],
    config: RoyEvoPDEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(equilibrium, dict):
        n0 = float(equilibrium["n"])
        w0 = float(equilibrium["w"])
        q0 = float(equilibrium["q"])
    else:
        n0, w0, q0 = [float(value) for value in equilibrium]
    rng = np.random.default_rng(config.seed)
    shape = (config.n_y, config.n_x)

    def mean_zero_noise() -> np.ndarray:
        noise = rng.standard_normal(shape)
        return noise - float(np.mean(noise))

    amp = config.perturbation_amplitude
    n = n0 * (1.0 + amp * mean_zero_noise())
    w = w0 * (1.0 + amp * mean_zero_noise())
    q = np.clip(q0 + amp * mean_zero_noise(), 0.0, 1.0)
    return n.astype(float), w.astype(float), q.astype(float)


def reaction_part_evo_pde(
    n: np.ndarray | float,
    w: np.ndarray | float,
    q: np.ndarray | float,
    params: RoyEvoParams,
    stress: float = 0.0,
    evolve: bool = True,
) -> np.ndarray:
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    z = free_space_evo(n_arr, w_arr, params)
    r_q = r_of_q(q_arr, params)
    a_q = a_of_q(q_arr, params)
    b_q = b_of_q(q_arr, params)
    gradient = selection_gradient(n_arr, w_arr, q_arr, params)
    dn = n_arr * (r_q * z - params.xi - a_q * w_arr)
    dw = w_arr * (b_q * n_arr * z - (params.m + stress) - params.mu * w_arr)
    dq = params.nu * q_arr * (1.0 - q_arr) * gradient if evolve else np.zeros_like(q_arr)
    return np.stack([dn, dw, dq], axis=0)


def predator_growth_factor_evo(
    n: np.ndarray | float,
    w: np.ndarray | float,
    q: np.ndarray | float,
    params: RoyEvoParams,
    stress: float = 0.0,
) -> np.ndarray | float:
    z = free_space_evo(n, w, params)
    return b_of_q(q, params) * np.asarray(n) * z - (params.m + stress) - params.mu * np.asarray(w)


def spatial_mechanism_diagnostics(
    n: np.ndarray | float,
    w: np.ndarray | float,
    q: np.ndarray | float,
    params: RoyEvoParams,
    stress: float = 0.0,
) -> dict[str, float]:
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    z = free_space_evo(n_arr, w_arr, params)
    local_growth = predator_growth_factor_evo(n_arr, w_arr, q_arr, params, stress=stress)
    b_q = b_of_q(q_arr, params)
    prey_gain = b_q * n_arr * z

    mean_n = float(np.mean(n_arr))
    mean_w = float(np.mean(w_arr))
    mean_q = float(np.mean(q_arr))
    mean_z = float(np.mean(z))
    mean_bq = float(np.mean(b_q))
    mean_prey_gain = float(np.mean(prey_gain))
    meanfield_growth_factor = float(predator_growth_factor_evo(mean_n, mean_w, mean_q, params, stress=stress))
    spatial_growth = float(np.mean(w_arr * local_growth))
    meanfield_growth = mean_w * meanfield_growth_factor
    centered_w = w_arr - mean_w
    return {
        "spatial_growth": spatial_growth,
        "meanfield_growth": float(meanfield_growth),
        "spatial_covariance_bonus": float(spatial_growth - meanfield_growth),
        "mean_q": mean_q,
        "var_q": float(np.var(q_arr)),
        "cov_w_q": float(np.mean(centered_w * (q_arr - mean_q))),
        "cov_w_bq": float(np.mean(centered_w * (b_q - mean_bq))),
        "cov_w_prey_gain": float(np.mean(centered_w * (prey_gain - mean_prey_gain))),
    }


def simulate_pde_evo_2d(
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    stress: float = 0.0,
    evolve: bool = True,
) -> RoyEvoPDEResult:
    validate_evo_params(params)
    check_evo_pde_cfl(config)
    _x, _y, dx, dy = grid_2d_evo(config)
    n, w, q = (np.asarray(value, dtype=float).copy() for value in initial_state)
    expected_shape = (config.n_y, config.n_x)
    if n.shape != expected_shape or w.shape != expected_shape or q.shape != expected_shape:
        raise ValueError(f"initial_state arrays must have shape {expected_shape}.")

    n_steps = int(np.ceil(config.T / config.dt))
    record_every = max(1, int(config.record_every))
    times: list[float] = []
    mean_n_time: list[float] = []
    mean_w_time: list[float] = []
    mean_q_time: list[float] = []
    var_n_time: list[float] = []
    var_w_time: list[float] = []
    var_q_time: list[float] = []
    min_z_time: list[float] = []
    spatial_growth_time: list[float] = []
    meanfield_growth_time: list[float] = []
    covariance_bonus_time: list[float] = []
    cov_w_q_time: list[float] = []
    cov_w_bq_time: list[float] = []
    cov_w_prey_gain_time: list[float] = []

    q_clip_count = 0
    q_clip_max_violation = 0.0
    nonfinite_detected = False
    completed = True
    initial_mean_q = float(np.mean(q))
    min_n_global = float(np.min(n))
    min_w_global = float(np.min(w))
    min_q_global = float(np.min(q))
    max_q_global = float(np.max(q))
    min_z_global = float(np.min(free_space_evo(n, w, params)))

    def update_global_extrema() -> None:
        nonlocal min_n_global, min_w_global, min_q_global, max_q_global, min_z_global
        min_n_global = min(min_n_global, float(np.min(n)))
        min_w_global = min(min_w_global, float(np.min(w)))
        min_q_global = min(min_q_global, float(np.min(q)))
        max_q_global = max(max_q_global, float(np.max(q)))
        min_z_global = min(min_z_global, float(np.min(free_space_evo(n, w, params))))

    def record(t_value: float) -> None:
        z = free_space_evo(n, w, params)
        mechanism = spatial_mechanism_diagnostics(n, w, q, params, stress=stress)
        times.append(float(t_value))
        mean_n_time.append(float(np.mean(n)))
        mean_w_time.append(float(np.mean(w)))
        mean_q_time.append(float(np.mean(q)))
        var_n_time.append(float(np.var(n)))
        var_w_time.append(float(np.var(w)))
        var_q_time.append(float(np.var(q)))
        min_z_time.append(float(np.min(z)))
        spatial_growth_time.append(mechanism["spatial_growth"])
        meanfield_growth_time.append(mechanism["meanfield_growth"])
        covariance_bonus_time.append(mechanism["spatial_covariance_bonus"])
        cov_w_q_time.append(mechanism["cov_w_q"])
        cov_w_bq_time.append(mechanism["cov_w_bq"])
        cov_w_prey_gain_time.append(mechanism["cov_w_prey_gain"])

    record(0.0)
    for step in range(1, n_steps + 1):
        reactions = reaction_part_evo_pde(n, w, q, params, stress=stress, evolve=evolve)
        n_next = n + config.dt * (config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + reactions[0])
        w_next = w + config.dt * (config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + reactions[1])
        if evolve:
            q_next = q + config.dt * (config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + reactions[2])
        else:
            q_next = q.copy()

        if not (np.all(np.isfinite(n_next)) and np.all(np.isfinite(w_next)) and np.all(np.isfinite(q_next))):
            nonfinite_detected = True
            completed = False
            break

        low_violation = np.maximum(-q_next, 0.0)
        high_violation = np.maximum(q_next - 1.0, 0.0)
        violation = np.maximum(low_violation, high_violation)
        max_violation = float(np.max(violation))
        if max_violation > 0.0:
            q_clip_count += int(np.count_nonzero(violation > 0.0))
            q_clip_max_violation = max(q_clip_max_violation, max_violation)
            if config.clip_q:
                q_next = np.clip(q_next, 0.0, 1.0)

        n, w, q = n_next, w_next, q_next
        update_global_extrema()
        if step % record_every == 0 or step == n_steps:
            record(min(step * config.dt, config.T))

    if nonfinite_detected and (not times or times[-1] < min(step * config.dt, config.T)):
        record(min(step * config.dt, config.T))

    diagnostics: dict[str, object] = {
        "completed": bool(completed),
        "nonfinite_detected": bool(nonfinite_detected),
        "initial_mean_q": initial_mean_q,
        "min_n": min_n_global,
        "min_w": min_w_global,
        "min_q": min_q_global,
        "max_q": max_q_global,
        "min_z": min_z_global,
        "q_clip_count": int(q_clip_count),
        "q_clip_max_violation": float(q_clip_max_violation),
        "spatial_growth_time": np.asarray(spatial_growth_time, dtype=float),
        "meanfield_growth_time": np.asarray(meanfield_growth_time, dtype=float),
        "spatial_covariance_bonus_time": np.asarray(covariance_bonus_time, dtype=float),
        "cov_w_q_time": np.asarray(cov_w_q_time, dtype=float),
        "cov_w_bq_time": np.asarray(cov_w_bq_time, dtype=float),
        "cov_w_prey_gain_time": np.asarray(cov_w_prey_gain_time, dtype=float),
    }
    return RoyEvoPDEResult(
        t=np.asarray(times, dtype=float),
        mean_n_time=np.asarray(mean_n_time, dtype=float),
        mean_w_time=np.asarray(mean_w_time, dtype=float),
        mean_q_time=np.asarray(mean_q_time, dtype=float),
        var_n_time=np.asarray(var_n_time, dtype=float),
        var_w_time=np.asarray(var_w_time, dtype=float),
        var_q_time=np.asarray(var_q_time, dtype=float),
        min_z_time=np.asarray(min_z_time, dtype=float),
        n=n,
        w=w,
        q=q,
        diagnostics=diagnostics,
    )


def _tail_mean_from_time(t: np.ndarray, values: np.ndarray, tail_fraction: float) -> float:
    if len(t) == 0 or len(values) != len(t):
        return float("nan")
    cutoff = t[-1] - tail_fraction * (t[-1] - t[0])
    mask = t >= cutoff
    if np.count_nonzero(mask) == 0:
        mask = np.zeros_like(t, dtype=bool)
        mask[-1] = True
    return float(np.mean(values[mask]))


def classify_evo_pde_result(
    result: RoyEvoPDEResult,
    params: RoyEvoParams,
    epsilon: float = 1.0e-4,
    tail_fraction: float = 0.25,
) -> dict[str, float | bool]:
    del params
    arrays = (
        result.t,
        result.mean_n_time,
        result.mean_w_time,
        result.mean_q_time,
        result.var_n_time,
        result.var_w_time,
        result.var_q_time,
        result.min_z_time,
    )
    nonfinite_detected = bool(result.diagnostics.get("nonfinite_detected", False))
    if len(result.t) < 2 or any(len(array) != len(result.t) for array in arrays):
        nonfinite_detected = True
    if any(not np.all(np.isfinite(array)) for array in arrays):
        nonfinite_detected = True
    if not (np.all(np.isfinite(result.n)) and np.all(np.isfinite(result.w)) and np.all(np.isfinite(result.q))):
        nonfinite_detected = True

    if nonfinite_detected:
        return {
            "physical": False,
            "persistent_predator": False,
            "tail_mean_n": float("nan"),
            "tail_mean_w": float("nan"),
            "tail_min_w": float("nan"),
            "tail_slope_w": float("nan"),
            "tail_mean_q": float("nan"),
            "q_change_from_initial": float("nan"),
            "tail_mean_var_q": float("nan"),
            "tail_mean_pattern_strength": float("nan"),
            "tail_mean_min_z": float("nan"),
            "tail_mean_spatial_covariance_bonus": float("nan"),
            "min_n": float("nan"),
            "min_w": float("nan"),
            "min_q": float("nan"),
            "max_q": float("nan"),
            "min_z": float("nan"),
            "nonfinite_detected": True,
            "q_clip_count": int(result.diagnostics.get("q_clip_count", 0)),
            "q_clip_max_violation": float(result.diagnostics.get("q_clip_max_violation", 0.0)),
        }

    t = result.t
    cutoff = t[-1] - tail_fraction * (t[-1] - t[0])
    mask = t >= cutoff
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(t, dtype=bool)
        mask[-2:] = True
    tail_t = t[mask]
    tail_w = result.mean_w_time[mask]
    centered_t = tail_t - float(np.mean(tail_t))
    denom = float(np.dot(centered_t, centered_t))
    slope_w = 0.0 if denom <= 0.0 else float(np.dot(centered_t, tail_w - float(np.mean(tail_w))) / denom)
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_floor = -max(epsilon, 0.25 * tail_mean_w) / tail_duration

    pattern_strength = np.maximum.reduce([result.var_n_time, result.var_w_time, result.var_q_time])
    q_clip_count = int(result.diagnostics.get("q_clip_count", 0))
    q_clip_max_violation = float(result.diagnostics.get("q_clip_max_violation", 0.0))
    min_n = float(result.diagnostics.get("min_n", np.min(result.n)))
    min_w = float(result.diagnostics.get("min_w", np.min(result.w)))
    min_q = float(result.diagnostics.get("min_q", np.min(result.q)))
    max_q = float(result.diagnostics.get("max_q", np.max(result.q)))
    min_z = float(result.diagnostics.get("min_z", np.min(result.min_z_time)))
    physical = (
        bool(result.diagnostics.get("completed", True))
        and min_n >= -1.0e-8
        and min_w >= -1.0e-8
        and min_q >= -1.0e-6
        and max_q <= 1.0 + 1.0e-6
        and min_z >= -1.0e-5
        and q_clip_max_violation <= 1.0e-4
    )
    persistent = physical and tail_mean_w > epsilon and tail_min_w > 0.25 * epsilon and slope_w >= slope_floor
    covariance_time = np.asarray(result.diagnostics.get("spatial_covariance_bonus_time", np.full_like(t, np.nan)), dtype=float)
    return {
        "physical": bool(physical),
        "persistent_predator": bool(persistent),
        "tail_mean_n": float(np.mean(result.mean_n_time[mask])),
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_mean_q": float(np.mean(result.mean_q_time[mask])),
        "q_change_from_initial": float(np.mean(result.mean_q_time[mask]) - float(result.diagnostics.get("initial_mean_q", result.mean_q_time[0]))),
        "tail_mean_var_q": float(np.mean(result.var_q_time[mask])),
        "tail_mean_pattern_strength": float(np.mean(pattern_strength[mask])),
        "tail_mean_min_z": float(np.mean(result.min_z_time[mask])),
        "tail_mean_spatial_covariance_bonus": _tail_mean_from_time(t, covariance_time, tail_fraction),
        "min_n": min_n,
        "min_w": min_w,
        "min_q": min_q,
        "max_q": max_q,
        "min_z": min_z,
        "nonfinite_detected": False,
        "q_clip_count": q_clip_count,
        "q_clip_max_violation": q_clip_max_violation,
    }


def _invalid_classifier() -> dict[str, float | bool]:
    return {
        "success": False,
        "physical": False,
        "tail_mean_n": float("nan"),
        "tail_mean_w": float("nan"),
        "tail_mean_q": float("nan"),
        "tail_min_w": float("nan"),
        "tail_slope_w": float("nan"),
        "tail_slope_floor_w": float("nan"),
        "persistent_predator": False,
        "q_final": float("nan"),
        "q_tail_mean": float("nan"),
        "q_change_from_initial": float("nan"),
        "min_z": float("nan"),
        "min_n": float("nan"),
        "min_w": float("nan"),
        "min_q": float("nan"),
        "max_q": float("nan"),
    }


def classify_evo_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1.0e-4,
    tail_fraction: float = 0.25,
    params: RoyEvoParams | None = None,
) -> dict[str, float | bool]:
    """Classify physicality and predator persistence from the final tail."""
    params = RoyEvoParams() if params is None else params
    try:
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape[0] != 3 and y_arr.shape[-1] == 3:
            y_arr = y_arr.T
        if t_arr.ndim != 1 or y_arr.ndim != 2 or y_arr.shape[0] != 3 or y_arr.shape[1] != len(t_arr):
            return _invalid_classifier()
        if len(t_arr) < 2 or not 0.0 < tail_fraction <= 1.0:
            return _invalid_classifier()
        if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(y_arr)):
            return _invalid_classifier()
    except (TypeError, ValueError):
        return _invalid_classifier()

    n, w, q = y_arr
    z = free_space_evo(n, w, params)
    physical = (
        np.all(n >= -1.0e-8)
        and np.all(w >= -1.0e-8)
        and np.all(q >= -1.0e-6)
        and np.all(q <= 1.0 + 1.0e-6)
        and np.all(z >= -1.0e-5)
    )

    tail_start = t_arr[-1] - tail_fraction * (t_arr[-1] - t_arr[0])
    mask = t_arr >= tail_start
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(t_arr, dtype=bool)
        mask[-2:] = True
    tail_t = t_arr[mask]
    tail_n = n[mask]
    tail_w = w[mask]
    tail_q = q[mask]
    centered_t = tail_t - float(np.mean(tail_t))
    denom = float(np.dot(centered_t, centered_t))
    slope_w = 0.0 if denom <= 0.0 else float(np.dot(centered_t, tail_w - float(np.mean(tail_w))) / denom)
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    slope_floor = -max(epsilon, 0.25 * tail_mean_w) / tail_duration
    persistent = (
        physical
        and tail_mean_w > epsilon
        and float(np.min(tail_w)) > 0.25 * epsilon
        and slope_w >= slope_floor
    )
    q_tail_mean = float(np.mean(tail_q))
    return {
        "success": bool(physical),
        "physical": bool(physical),
        "tail_mean_n": float(np.mean(tail_n)),
        "tail_mean_w": tail_mean_w,
        "tail_mean_q": q_tail_mean,
        "tail_min_w": float(np.min(tail_w)),
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": float(slope_floor),
        "persistent_predator": bool(persistent),
        "q_final": float(q[-1]),
        "q_tail_mean": q_tail_mean,
        "q_change_from_initial": float(q_tail_mean - q[0]),
        "min_z": float(np.min(z)),
        "min_n": float(np.min(n)),
        "min_w": float(np.min(w)),
        "min_q": float(np.min(q)),
        "max_q": float(np.max(q)),
    }


def simulate_ode_evo(
    params: RoyEvoParams,
    initial_state: Iterable[float],
    stress: float = 0.0,
    evolve: bool = True,
    T: float = 1000.0,
    n_eval: int = 501,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> RoyEvoTrajectory:
    validate_evo_params(params)
    if T <= 0.0 or n_eval < 2:
        raise ValueError("T must be positive and n_eval must be at least 2.")
    t_eval = np.linspace(0.0, T, n_eval)
    y0 = np.asarray(initial_state, dtype=float)
    if y0.shape != (3,):
        raise ValueError("initial_state must contain [n, w, q].")
    result = solve_ivp(
        lambda t, y: reaction_ode_evo(t, y, params, stress=stress, evolve=evolve),
        (0.0, T),
        y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    return RoyEvoTrajectory(
        t=result.t,
        y=result.y,
        success=bool(result.success),
        message=str(result.message),
        stress=float(stress),
        evolve=bool(evolve),
    )


def find_evo_equilibrium(
    params: RoyEvoParams,
    guesses: Iterable[Iterable[float]] | None = None,
    burn_in_T: float = 3000.0,
) -> dict[str, object]:
    """Approximate a positive unstressed evolving equilibrium by long burn-in."""
    validate_evo_params(params)
    if guesses is None:
        guesses = (
            (1.0, 0.2, 0.5),
            (2.0, 0.5, 0.3),
            (4.0, 0.5, 0.7),
            (2.5, 1.0, 0.5),
        )
    candidates: list[dict[str, object]] = []
    for guess in guesses:
        trajectory = simulate_ode_evo(params, guess, stress=0.0, evolve=True, T=burn_in_T, n_eval=801)
        diagnostics = classify_evo_trajectory(trajectory.t, trajectory.y, params=params)
        final = trajectory.y[:, -1]
        residual = float(np.linalg.norm(reaction_ode_evo(0.0, final, params, stress=0.0, evolve=True)))
        if trajectory.success and diagnostics["physical"] and diagnostics["persistent_predator"]:
            candidates.append(
                {
                    "n": float(final[0]),
                    "w": float(final[1]),
                    "q": float(final[2]),
                    "z": float(free_space_evo(final[0], final[1], params)),
                    "residual": residual,
                    "method": "unstressed_evolving_burn_in",
                    "initial_guess": tuple(float(value) for value in guess),
                    "trajectory_success": trajectory.success,
                    "message": trajectory.message,
                    "diagnostics": diagnostics,
                }
            )
    if not candidates:
        raise ValueError("No positive persistent evolving ODE baseline was found.")
    return min(candidates, key=lambda item: float(item["residual"]))


def bisection_threshold(
    classify_stress: Callable[[float], tuple[bool, dict[str, float | bool]]],
    stress_low: float,
    stress_high: float,
    tolerance: float = 1.0e-5,
    max_iter: int = 28,
) -> dict[str, object]:
    """Find the largest persistent stress in a monotone persistence bracket."""
    low_persistent, low_metrics = classify_stress(stress_low)
    high_persistent, high_metrics = classify_stress(stress_high)
    history: list[tuple[float, bool]] = [(float(stress_low), bool(low_persistent)), (float(stress_high), bool(high_persistent))]
    if not low_persistent:
        return {
            "threshold": float("nan"),
            "stress_low": float(stress_low),
            "stress_high": float(stress_high),
            "threshold_gap": float(stress_high - stress_low),
            "threshold_status": "invalid_bracket_low_not_persistent",
            "iterations": 0,
            "persistent_low": bool(low_persistent),
            "persistent_high": bool(high_persistent),
            "low_metrics": low_metrics,
            "high_metrics": high_metrics,
            "history": history,
        }
    if high_persistent:
        return {
            "threshold": float("nan"),
            "stress_low": float(stress_low),
            "stress_high": float(stress_high),
            "threshold_gap": float(stress_high - stress_low),
            "threshold_status": "invalid_bracket_high_persistent",
            "iterations": 0,
            "persistent_low": bool(low_persistent),
            "persistent_high": bool(high_persistent),
            "low_metrics": low_metrics,
            "high_metrics": high_metrics,
            "history": history,
        }

    low = float(stress_low)
    high = float(stress_high)
    last_low_metrics = low_metrics
    last_high_metrics = high_metrics
    iterations = 0
    while iterations < max_iter and high - low > tolerance:
        mid = 0.5 * (low + high)
        mid_persistent, mid_metrics = classify_stress(mid)
        history.append((float(mid), bool(mid_persistent)))
        iterations += 1
        if mid_persistent:
            low = mid
            last_low_metrics = mid_metrics
        else:
            high = mid
            last_high_metrics = mid_metrics

    return {
        "threshold": 0.5 * (low + high),
        "stress_low": low,
        "stress_high": high,
        "threshold_gap": high - low,
        "threshold_status": "ok",
        "iterations": iterations,
        "persistent_low": True,
        "persistent_high": False,
        "low_metrics": last_low_metrics,
        "high_metrics": last_high_metrics,
        "history": history,
    }
