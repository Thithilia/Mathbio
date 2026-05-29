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
