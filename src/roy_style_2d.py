"""2D Roy-style reaction-diffusion simulations and threshold diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

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
    return RoyODEResult(sol.t, sol.y, bool(sol.success), sol.message, final_w, bool(final_w > epsilon))


def compute_stress_threshold_ode(
    params: RoyParams,
    s_low: float,
    s_high: float,
    epsilon: float,
    T: float,
    tol_s: float = 1.0e-3,
    max_iter: int = 20,
) -> StressThreshold:
    eq = require_positive_equilibrium(params)
    y0 = np.array([eq.u, eq.v, eq.w], dtype=float)

    def classify(stress: float) -> tuple[bool, float]:
        result = simulate_ode_stress(params, stress, T, epsilon, y0=y0)
        return result.persistent and result.success, result.final_w

    low_persistent, low_measure = classify(s_low)
    high_persistent, high_measure = classify(s_high)
    if not low_persistent or high_persistent:
        raise ValueError(
            "ODE stress bracket must satisfy persistent(s_low)=True and persistent(s_high)=False. "
            f"Got low={low_persistent} ({low_measure:.3e}), high={high_persistent} ({high_measure:.3e})."
        )

    history = [(s_low, low_persistent, low_measure), (s_high, high_persistent, high_measure)]
    lo, hi = float(s_low), float(s_high)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        persistent, measure = classify(mid)
        history.append((mid, persistent, measure))
        if persistent:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol_s:
            break
    return StressThreshold(lo, s_low, s_high, iteration, history, epsilon, T)


def compute_stress_threshold_pde(
    params: RoyParams,
    config: Roy2DConfig,
    s_low: float,
    s_high: float,
    epsilon: float,
    tol_s: float = 1.0e-3,
    max_iter: int = 14,
) -> StressThreshold:
    eq = require_positive_equilibrium(params)
    initial = perturbed_equilibrium_2d(eq, config)

    def classify(stress: float) -> tuple[bool, float]:
        result = simulate_pde_2d(params, config, stress=stress, equilibrium=eq, initial_state=initial)
        mean_w = result.diagnostics.mean_w
        return bool(mean_w > epsilon), mean_w

    low_persistent, low_measure = classify(s_low)
    high_persistent, high_measure = classify(s_high)
    if not low_persistent or high_persistent:
        raise ValueError(
            "PDE stress bracket must satisfy persistent(s_low)=True and persistent(s_high)=False. "
            f"Got low={low_persistent} ({low_measure:.3e}), high={high_persistent} ({high_measure:.3e})."
        )

    history = [(s_low, low_persistent, low_measure), (s_high, high_persistent, high_measure)]
    lo, hi = float(s_low), float(s_high)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        persistent, measure = classify(mid)
        history.append((mid, persistent, measure))
        if persistent:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol_s:
            break
    return StressThreshold(lo, s_low, s_high, iteration, history, epsilon, config.T)
