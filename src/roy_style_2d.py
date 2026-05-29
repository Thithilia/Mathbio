"""2D Roy-style reaction-diffusion simulations and threshold diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .roy_style_model import (
    RoyEquilibrium,
    RoyParams,
    free_space,
    reaction_ode,
    reaction_part,
    require_positive_equilibrium,
)


@dataclass(frozen=True)
class Roy2DConfig:
    n_x: int = 64
    n_y: int = 64
    L_x: float = 20.0
    L_y: float = 20.0
    dt: float = 0.01
    T: float = 80.0
    record_every: int = 20
    perturbation_amplitude: float = 1.0e-5
    seed: int = 1
    clip_negative: bool = False
    record_fourier: bool = False


@dataclass(frozen=True)
class Roy2DDiagnostics:
    mean_u: float
    mean_v: float
    mean_w: float
    var_u: float
    var_v: float
    var_w: float
    min_value: float
    min_z: float
    negative_detected: bool
    z_negative_detected: bool
    dominant_k: float
    dominant_wavelength: float
    dominant_power: float


@dataclass(frozen=True)
class Roy2DResult:
    t: np.ndarray
    mean_u_time: np.ndarray
    mean_v_time: np.ndarray
    mean_w_time: np.ndarray
    var_u_time: np.ndarray
    var_v_time: np.ndarray
    var_w_time: np.ndarray
    min_z_time: np.ndarray
    dominant_wavelength_time: np.ndarray
    dominant_power_time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    diagnostics: Roy2DDiagnostics


@dataclass(frozen=True)
class RoyODEResult:
    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    final_w: float
    persistent: bool


@dataclass(frozen=True)
class StressThreshold:
    threshold: float
    s_low: float
    s_high: float
    iterations: int
    history: list[tuple[float, bool, float]]
    epsilon: float
    T: float


def tail_metrics(t: np.ndarray, y: np.ndarray, tail_fraction: float = 0.25) -> dict[str, float]:
    """Return summary metrics over the final fraction of a time series."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.ndim != 1 or y_arr.ndim != 1 or len(t_arr) != len(y_arr):
        raise ValueError("t and y must be one-dimensional arrays with matching lengths.")
    if len(t_arr) < 2:
        raise ValueError("At least two time points are required.")
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("t and y must contain only finite values.")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 1.")

    tail_start_target = t_arr[-1] - tail_fraction * (t_arr[-1] - t_arr[0])
    mask = t_arr >= tail_start_target
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(t_arr, dtype=bool)
        mask[-2:] = True
    tail_t = t_arr[mask]
    tail_y = y_arr[mask]
    centered_t = tail_t - float(np.mean(tail_t))
    denom = float(np.dot(centered_t, centered_t))
    slope = 0.0 if denom <= 0.0 else float(np.dot(centered_t, tail_y - float(np.mean(tail_y))) / denom)
    return {
        "tail_mean": float(np.mean(tail_y)),
        "tail_min": float(np.min(tail_y)),
        "tail_start": float(tail_t[0]),
        "tail_end": float(tail_t[-1]),
        "tail_slope": slope,
        "tail_duration": float(max(tail_t[-1] - tail_t[0], 0.0)),
    }


def _invalid_tail_metrics() -> dict[str, float]:
    return {
        "tail_mean": float("nan"),
        "tail_min": float("nan"),
        "tail_start": float("nan"),
        "tail_end": float("nan"),
        "tail_slope": float("nan"),
        "tail_duration": float("nan"),
        "tail_slope_floor": float("nan"),
    }


def is_persistent_tail(
    t: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    tail_fraction: float = 0.25,
) -> tuple[bool, dict[str, float]]:
    """Classify persistence from tail behavior rather than a final value.

    A trajectory is persistent when the final-tail mean exceeds ``epsilon``,
    the tail minimum remains above one quarter of ``epsilon``, and the fitted
    tail slope is not negative enough to indicate transient survival.
    """
    try:
        metrics = tail_metrics(t, y, tail_fraction=tail_fraction)
    except ValueError:
        return False, _invalid_tail_metrics()
    tail_duration = max(metrics["tail_duration"], 1.0e-12)
    slope_floor = -max(epsilon, 0.25 * metrics["tail_mean"]) / tail_duration
    persistent = (
        metrics["tail_mean"] > epsilon
        and metrics["tail_min"] > 0.25 * epsilon
        and metrics["tail_slope"] >= slope_floor
    )
    metrics["tail_slope_floor"] = float(slope_floor)
    return bool(persistent), metrics


def ode_tail_persistence(result: RoyODEResult, epsilon: float, tail_fraction: float = 0.25) -> tuple[bool, dict[str, float]]:
    """Classify ODE persistence, rejecting failed or nonfinite integrations."""
    if not result.success or result.y.size == 0:
        return False, _invalid_tail_metrics()
    if result.y.ndim != 2 or result.y.shape[0] < 3:
        return False, _invalid_tail_metrics()
    if not np.all(np.isfinite(result.t)) or not np.all(np.isfinite(result.y)):
        return False, _invalid_tail_metrics()
    return is_persistent_tail(result.t, result.y[2], epsilon, tail_fraction=tail_fraction)


def pde_tail_persistence(result: Roy2DResult, epsilon: float, tail_fraction: float = 0.25) -> tuple[bool, dict[str, float]]:
    """Classify PDE persistence, rejecting nonphysical or nonfinite trajectories."""
    arrays = (
        result.t,
        result.mean_u_time,
        result.mean_v_time,
        result.mean_w_time,
        result.var_u_time,
        result.var_v_time,
        result.var_w_time,
        result.min_z_time,
        result.u,
        result.v,
        result.w,
    )
    if result.diagnostics.negative_detected or result.diagnostics.z_negative_detected:
        return False, _invalid_tail_metrics()
    if any(not np.all(np.isfinite(arr)) for arr in arrays):
        return False, _invalid_tail_metrics()
    return is_persistent_tail(result.t, result.mean_w_time, epsilon, tail_fraction=tail_fraction)


def make_refined_stress_bracket(
    ode_threshold: float,
    pde_threshold: float,
    margin: float = 0.04,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, float]:
    """Build a bracket centered on prior ODE/PDE threshold estimates."""
    if not np.isfinite(ode_threshold) or not np.isfinite(pde_threshold):
        return lower_bound, upper_bound
    low = max(lower_bound, min(ode_threshold, pde_threshold) - margin)
    high = min(upper_bound, max(ode_threshold, pde_threshold) + margin)
    if low >= high:
        return lower_bound, upper_bound
    return float(low), float(high)


def _classifier_bool(value: object) -> bool:
    if isinstance(value, tuple):
        return bool(value[0])
    return bool(value)


def ensure_valid_threshold_bracket(
    classify_fn: Callable[[float], object],
    low: float,
    high: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    expansion: float = 0.03,
    max_expansions: int = 10,
) -> tuple[float, float, str]:
    """Expand a stress bracket until low persists and high is extinct."""
    lo = float(max(lower_bound, low))
    hi = float(min(upper_bound, high))
    if lo >= hi:
        return lo, hi, "invalid_bracket_order"

    status = "invalid_bracket"
    for _ in range(max_expansions + 1):
        low_persistent = _classifier_bool(classify_fn(lo))
        high_persistent = _classifier_bool(classify_fn(hi))
        if low_persistent and not high_persistent:
            return lo, hi, "ok"

        moved = False
        if not low_persistent and lo > lower_bound:
            new_lo = max(lower_bound, lo - expansion)
            moved = moved or new_lo != lo
            lo = new_lo
        if high_persistent and hi < upper_bound:
            new_hi = min(upper_bound, hi + expansion)
            moved = moved or new_hi != hi
            hi = new_hi
        if not moved:
            if not low_persistent and high_persistent:
                status = "invalid_bracket_low_not_persistent_high_persistent"
            elif not low_persistent:
                status = "invalid_bracket_low_not_persistent"
            else:
                status = "invalid_bracket_high_persistent"
            break
    else:
        low_persistent = _classifier_bool(classify_fn(lo))
        high_persistent = _classifier_bool(classify_fn(hi))
        if not low_persistent and high_persistent:
            status = "invalid_bracket_low_not_persistent_high_persistent"
        elif not low_persistent:
            status = "invalid_bracket_low_not_persistent"
        elif high_persistent:
            status = "invalid_bracket_high_persistent"
    return lo, hi, status


def summarize_delta_group(deltas: np.ndarray, tolerances: np.ndarray) -> dict[str, float | str]:
    """Summarize seed-level Delta m_c intervals for a validation group."""
    delta_arr = np.asarray(deltas, dtype=float)
    tol_arr = np.asarray(tolerances, dtype=float)
    if delta_arr.ndim != 1 or tol_arr.ndim != 1 or len(delta_arr) != len(tol_arr) or len(delta_arr) == 0:
        return {
            "delta_min": float("nan"),
            "delta_max": float("nan"),
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "interval_low": float("nan"),
            "interval_high": float("nan"),
            "conclusion": "invalid",
        }
    if not np.all(np.isfinite(delta_arr)) or not np.all(np.isfinite(tol_arr)):
        return {
            "delta_min": float("nan"),
            "delta_max": float("nan"),
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "interval_low": float("nan"),
            "interval_high": float("nan"),
            "conclusion": "invalid",
        }

    lows = delta_arr - tol_arr
    highs = delta_arr + tol_arr
    interval_low = float(np.min(lows))
    interval_high = float(np.max(highs))
    if interval_low > 0.0:
        conclusion = "rescue_supported"
    elif interval_high < 0.0:
        conclusion = "inhibition_supported"
    else:
        conclusion = "no_measurable_effect"
    return {
        "delta_min": float(np.min(delta_arr)),
        "delta_max": float(np.max(delta_arr)),
        "delta_mean": float(np.mean(delta_arr)),
        "delta_std": float(np.std(delta_arr)),
        "interval_low": interval_low,
        "interval_high": interval_high,
        "conclusion": conclusion,
    }


def with_stress(params: RoyParams, stress: float) -> RoyParams:
    return params.with_updates(delta=params.delta + stress)


def grid_2d(config: Roy2DConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    if config.n_x < 3 or config.n_y < 3:
        raise ValueError("n_x and n_y must be at least 3.")
    x = np.linspace(0.0, config.L_x, config.n_x)
    y = np.linspace(0.0, config.L_y, config.n_y)
    return x, y, x[1] - x[0], y[1] - y[0]


def laplacian_neumann_2d(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Second-order finite-difference Laplacian with no-flux boundaries."""
    lap = np.empty_like(values)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    center = values
    left = np.empty_like(values)
    right = np.empty_like(values)
    down = np.empty_like(values)
    up = np.empty_like(values)

    left[:, 1:] = values[:, :-1]
    left[:, 0] = values[:, 1]
    right[:, :-1] = values[:, 1:]
    right[:, -1] = values[:, -2]
    down[1:, :] = values[:-1, :]
    down[0, :] = values[1, :]
    up[:-1, :] = values[1:, :]
    up[-1, :] = values[-2, :]

    lap[:, :] = (left - 2.0 * center + right) * inv_dx2 + (down - 2.0 * center + up) * inv_dy2
    return lap


def check_explicit_diffusion_cfl(params: RoyParams, config: Roy2DConfig, safety: float = 0.24) -> None:
    _, _, dx, dy = grid_2d(config)
    max_diffusion = max(params.D_u, params.D_v, params.D_w)
    if max_diffusion <= 0.0:
        return
    stable_dt = safety / (max_diffusion * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if config.dt > stable_dt:
        raise ValueError(f"dt={config.dt:g} exceeds explicit diffusion stability estimate {stable_dt:g}.")


def perturbed_equilibrium_2d(
    equilibrium: RoyEquilibrium,
    config: Roy2DConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    shape = (config.n_y, config.n_x)
    u = np.full(shape, equilibrium.u, dtype=float)
    v = np.full(shape, equilibrium.v, dtype=float)
    w = np.full(shape, equilibrium.w, dtype=float)
    if config.perturbation_amplitude > 0.0:
        for arr in (u, v, w):
            arr *= 1.0 + config.perturbation_amplitude * rng.standard_normal(shape)
            np.maximum(arr, 1.0e-12, out=arr)
    return u, v, w


def spatial_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def dominant_fourier_mode(values: np.ndarray, L_x: float, L_y: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft2(centered)) ** 2
    ky = np.fft.fftfreq(values.shape[0], d=L_y / values.shape[0])
    kx = np.fft.rfftfreq(values.shape[1], d=L_x / values.shape[1])
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    radial_k = 2.0 * np.pi * np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
    spectrum[0, 0] = 0.0
    idx = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    dominant_k = float(radial_k[idx])
    wavelength = float(2.0 * np.pi / dominant_k) if dominant_k > 0.0 else float("inf")
    return dominant_k, wavelength, float(spectrum[idx]), radial_k, spectrum


def compute_diagnostics(u: np.ndarray, v: np.ndarray, w: np.ndarray, params: RoyParams, config: Roy2DConfig) -> Roy2DDiagnostics:
    z = free_space(u, v, w, params)
    min_value = float(min(np.min(u), np.min(v), np.min(w)))
    min_z = float(np.min(z))
    dominant_k, wavelength, power, _, _ = dominant_fourier_mode(u, config.L_x, config.L_y)
    return Roy2DDiagnostics(
        mean_u=spatial_mean(u),
        mean_v=spatial_mean(v),
        mean_w=spatial_mean(w),
        var_u=float(np.var(u)),
        var_v=float(np.var(v)),
        var_w=float(np.var(w)),
        min_value=min_value,
        min_z=min_z,
        negative_detected=bool(min_value < -1.0e-8),
        z_negative_detected=bool(min_z < -1.0e-8),
        dominant_k=dominant_k,
        dominant_wavelength=wavelength,
        dominant_power=power,
    )


def snapshot_diagnostics(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    params: RoyParams,
    config: Roy2DConfig,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    z = free_space(u, v, w, params)
    wavelength = float("nan")
    power = float("nan")
    if config.record_fourier:
        _, wavelength, power, _, _ = dominant_fourier_mode(u, config.L_x, config.L_y)
    return (
        spatial_mean(u),
        spatial_mean(v),
        spatial_mean(w),
        float(np.var(u)),
        float(np.var(v)),
        float(np.var(w)),
        float(np.min(z)),
        wavelength,
        power,
    )


def simulate_pde_2d(
    params: RoyParams,
    config: Roy2DConfig,
    stress: float = 0.0,
    equilibrium: RoyEquilibrium | None = None,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Roy2DResult:
    """Simulate the 2D Roy-style PDE by explicit finite differences."""
    stressed = with_stress(params, stress)
    check_explicit_diffusion_cfl(stressed, config)
    x, y, dx, dy = grid_2d(config)
    if initial_state is None:
        eq = equilibrium or require_positive_equilibrium(params)
        u, v, w = perturbed_equilibrium_2d(eq, config)
    else:
        u, v, w = [np.asarray(arr, dtype=float).copy() for arr in initial_state]
    if u.shape != (config.n_y, config.n_x) or v.shape != u.shape or w.shape != u.shape:
        raise ValueError("Initial 2D arrays must have shape (n_y, n_x).")

    n_steps = int(np.ceil(config.T / config.dt))
    record_every = max(1, config.record_every)
    times: list[float] = [0.0]
    mean_u: list[float] = []
    mean_v: list[float] = []
    mean_w: list[float] = []
    var_u: list[float] = []
    var_v: list[float] = []
    var_w: list[float] = []
    min_z: list[float] = []
    dominant_wavelength: list[float] = []
    dominant_power: list[float] = []

    def record() -> None:
        values = snapshot_diagnostics(u, v, w, stressed, config)
        mean_u.append(values[0])
        mean_v.append(values[1])
        mean_w.append(values[2])
        var_u.append(values[3])
        var_v.append(values[4])
        var_w.append(values[5])
        min_z.append(values[6])
        dominant_wavelength.append(values[7])
        dominant_power.append(values[8])

    record()

    for step in range(1, n_steps + 1):
        reactions = reaction_part(u, v, w, stressed)
        u_next = u + config.dt * (stressed.D_u * laplacian_neumann_2d(u, dx, dy) + reactions[0])
        v_next = v + config.dt * (stressed.D_v * laplacian_neumann_2d(v, dx, dy) + reactions[1])
        w_next = w + config.dt * (stressed.D_w * laplacian_neumann_2d(w, dx, dy) + reactions[2])
        if config.clip_negative:
            np.maximum(u_next, 0.0, out=u_next)
            np.maximum(v_next, 0.0, out=v_next)
            np.maximum(w_next, 0.0, out=w_next)
        u, v, w = u_next, v_next, w_next
        if step % record_every == 0 or step == n_steps:
            times.append(min(step * config.dt, config.T))
            record()

    diagnostics = compute_diagnostics(u, v, w, stressed, config)
    return Roy2DResult(
        t=np.asarray(times, dtype=float),
        mean_u_time=np.asarray(mean_u, dtype=float),
        mean_v_time=np.asarray(mean_v, dtype=float),
        mean_w_time=np.asarray(mean_w, dtype=float),
        var_u_time=np.asarray(var_u, dtype=float),
        var_v_time=np.asarray(var_v, dtype=float),
        var_w_time=np.asarray(var_w, dtype=float),
        min_z_time=np.asarray(min_z, dtype=float),
        dominant_wavelength_time=np.asarray(dominant_wavelength, dtype=float),
        dominant_power_time=np.asarray(dominant_power, dtype=float),
        x=x,
        y=y,
        u=u,
        v=v,
        w=w,
        diagnostics=diagnostics,
    )


def simulate_ode_stress(
    params: RoyParams,
    stress: float,
    T: float,
    epsilon: float,
    y0: np.ndarray | None = None,
    n_time: int = 300,
) -> RoyODEResult:
    stressed = with_stress(params, stress)
    if y0 is None:
        eq = require_positive_equilibrium(params)
        y0 = np.array([eq.u, eq.v, eq.w], dtype=float)
    sol = solve_ivp(
        lambda t, y: reaction_ode(t, y, stressed),
        (0.0, T),
        np.asarray(y0, dtype=float),
        method="BDF",
        t_eval=np.linspace(0.0, T, n_time),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    final_w = float(sol.y[2, -1]) if sol.y.size else float("nan")
    result = RoyODEResult(sol.t, sol.y, bool(sol.success), sol.message, final_w, False)
    persistent, _ = ode_tail_persistence(result, epsilon)
    return RoyODEResult(sol.t, sol.y, bool(sol.success), sol.message, final_w, bool(persistent))


def _threshold_search(
    classify: Callable[[float], tuple[bool, float, dict[str, float]]],
    s_low: float,
    s_high: float,
    max_iter: int,
) -> dict[str, object]:
    low_persistent, low_measure, low_metrics = classify(s_low)
    high_persistent, high_measure, high_metrics = classify(s_high)
    history: list[dict[str, object]] = [
        {"stress": float(s_low), "persistent": low_persistent, "measure": low_measure, **low_metrics},
        {"stress": float(s_high), "persistent": high_persistent, "measure": high_measure, **high_metrics},
    ]
    if not low_persistent or high_persistent:
        return {
            "threshold": float("nan"),
            "s_low": float(s_low),
            "s_high": float(s_high),
            "iterations": 0,
            "persistent_low": bool(low_persistent),
            "persistent_high": bool(high_persistent),
            "history": history,
            "status": "invalid_bracket",
        }

    lo = float(s_low)
    hi = float(s_high)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        persistent, measure, metrics = classify(mid)
        history.append({"stress": mid, "persistent": persistent, "measure": measure, **metrics})
        if persistent:
            lo = mid
        else:
            hi = mid
    return {
        "threshold": lo,
        "s_low": lo,
        "s_high": hi,
        "iterations": iterations,
        "persistent_low": True,
        "persistent_high": False,
        "history": history,
        "status": "ok",
    }


def find_ode_threshold(
    params: RoyParams,
    s_low: float,
    s_high: float,
    T: float,
    epsilon: float,
    max_iter: int = 16,
) -> dict[str, object]:
    """Find the well-mixed predator-mortality stress threshold by bisection."""
    eq = require_positive_equilibrium(params)
    y0 = np.array([eq.u, eq.v, eq.w], dtype=float)
    cache: dict[float, tuple[bool, float, dict[str, float]]] = {}

    def classify(stress: float) -> tuple[bool, float, dict[str, float]]:
        key = float(stress)
        if key in cache:
            return cache[key]
        result = simulate_ode_stress(params, stress, T, epsilon, y0=y0)
        persistent, metrics = ode_tail_persistence(result, epsilon)
        value = (bool(result.success and persistent), float(metrics["tail_mean"]), metrics)
        cache[key] = value
        return value

    bracket_low, bracket_high, bracket_status = ensure_valid_threshold_bracket(
        lambda stress: classify(stress)[0],
        s_low,
        s_high,
    )
    if bracket_status != "ok":
        low_persistent, low_measure, low_metrics = classify(bracket_low)
        high_persistent, high_measure, high_metrics = classify(bracket_high)
        return {
            "threshold": float("nan"),
            "s_low": float(bracket_low),
            "s_high": float(bracket_high),
            "iterations": 0,
            "persistent_low": bool(low_persistent),
            "persistent_high": bool(high_persistent),
            "history": [
                {"stress": float(bracket_low), "persistent": low_persistent, "measure": low_measure, **low_metrics},
                {"stress": float(bracket_high), "persistent": high_persistent, "measure": high_measure, **high_metrics},
            ],
            "status": bracket_status,
        }

    return _threshold_search(classify, bracket_low, bracket_high, max_iter=max_iter)


def find_pde_threshold(
    params: RoyParams,
    config: Roy2DConfig,
    s_low: float,
    s_high: float,
    epsilon: float,
    max_iter: int = 12,
) -> dict[str, object]:
    """Find the 2D PDE predator-mortality stress threshold by bisection."""
    eq = require_positive_equilibrium(params)
    cache: dict[float, tuple[bool, float, dict[str, float]]] = {}

    def classify(stress: float) -> tuple[bool, float, dict[str, float]]:
        key = float(stress)
        if key in cache:
            return cache[key]
        result = simulate_pde_2d(params, config, stress=stress, equilibrium=eq)
        persistent, metrics = pde_tail_persistence(result, epsilon)
        value = (bool(persistent), float(metrics["tail_mean"]), metrics)
        cache[key] = value
        return value

    bracket_low, bracket_high, bracket_status = ensure_valid_threshold_bracket(
        lambda stress: classify(stress)[0],
        s_low,
        s_high,
    )
    if bracket_status != "ok":
        low_persistent, low_measure, low_metrics = classify(bracket_low)
        high_persistent, high_measure, high_metrics = classify(bracket_high)
        return {
            "threshold": float("nan"),
            "s_low": float(bracket_low),
            "s_high": float(bracket_high),
            "iterations": 0,
            "persistent_low": bool(low_persistent),
            "persistent_high": bool(high_persistent),
            "history": [
                {"stress": float(bracket_low), "persistent": low_persistent, "measure": low_measure, **low_metrics},
                {"stress": float(bracket_high), "persistent": high_persistent, "measure": high_measure, **high_metrics},
            ],
            "status": bracket_status,
        }

    return _threshold_search(classify, bracket_low, bracket_high, max_iter=max_iter)


def compute_stress_threshold_ode(
    params: RoyParams,
    s_low: float,
    s_high: float,
    epsilon: float,
    T: float,
    tol_s: float = 1.0e-3,
    max_iter: int = 20,
) -> StressThreshold:
    threshold = find_ode_threshold(params, s_low, s_high, T, epsilon, max_iter=max_iter)
    if threshold["status"] != "ok":
        raise ValueError(
            "ODE stress bracket must satisfy persistent(s_low)=True and persistent(s_high)=False. "
            f"Got low={threshold['persistent_low']}, high={threshold['persistent_high']}."
        )
    history = [
        (float(item["stress"]), bool(item["persistent"]), float(item["measure"]))
        for item in threshold["history"]
    ]
    return StressThreshold(
        float(threshold["threshold"]),
        float(threshold["s_low"]),
        float(threshold["s_high"]),
        int(threshold["iterations"]),
        history,
        epsilon,
        T,
    )


def compute_stress_threshold_pde(
    params: RoyParams,
    config: Roy2DConfig,
    s_low: float,
    s_high: float,
    epsilon: float,
    tol_s: float = 1.0e-3,
    max_iter: int = 14,
) -> StressThreshold:
    threshold = find_pde_threshold(params, config, s_low, s_high, epsilon, max_iter=max_iter)
    if threshold["status"] != "ok":
        raise ValueError(
            "PDE stress bracket must satisfy persistent(s_low)=True and persistent(s_high)=False. "
            f"Got low={threshold['persistent_low']}, high={threshold['persistent_high']}."
        )
    history = [
        (float(item["stress"]), bool(item["persistent"]), float(item["measure"]))
        for item in threshold["history"]
    ]
    return StressThreshold(
        float(threshold["threshold"]),
        float(threshold["s_low"]),
        float(threshold["s_high"]),
        int(threshold["iterations"]),
        history,
        epsilon,
        config.T,
    )
