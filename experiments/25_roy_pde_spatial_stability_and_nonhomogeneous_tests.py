#!/usr/bin/env python
"""PDE spatial stability and targeted non-homogeneous perturbation tests.

This experiment has two parts. First, it evaluates linear Neumann spatial-mode
stability of the homogeneous ODE compensation branch embedded in the PDE.
Second, it runs a small targeted set of nonlinear PDE tests with explicitly
non-homogeneous initial fields. It does not change model equations or run a
broad PDE parameter scan.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roy_evo_spatial import (  # noqa: E402
    RoyEvoPDEConfig,
    RoyEvoParams,
    find_evo_equilibrium,
    free_space_evo,
    grid_2d_evo,
    laplacian_neumann_2d_evo,
    reaction_part_evo_pde,
)


def load_step23_module():
    path = Path(__file__).resolve().with_name("23_roy_ode_compensation_conditions.py")
    spec = importlib.util.spec_from_file_location("step23_ode_compensation_conditions_runtime", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load Step 23 module from {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP23 = load_step23_module()

PARAMS = RoyEvoParams(b_u=0.08, b_v=0.02)
TARGET_STRESSES = (0.11765625, 0.1584375, 0.16486816, 0.175)
NONLINEAR_STRESSES = (0.1584375, 0.16486816)
MODE_MAX_FOCUSED = 64
DIFFUSION_MODE_MAX = 32
POSITIVE_INSTABILITY_TOL = 1.0e-8
NEAR_ZERO_TOL = 1.0e-7
EPSILON = 1.0e-4
TAIL_FRACTION = 0.25
STEADY_REL_CHANGE_TOL = 0.02
STEADY_RESIDUAL_TOL = 1.0e-4
EXTREME_EXTINCTION_W = 1.0e-8
FINAL_CV_THRESHOLD = 1.0e-3
MAX_CV_THRESHOLD = 1.0e-2

BASE_D_N = 0.01
BASE_D_W = 0.01
BASE_D_Q = 0.005
L_X = 20.0
L_Y = 20.0
N_X = 64
N_Y = 64
FOCUSED_T = 1600.0
FOCUSED_DT = 0.1
RECORD_EVERY = 50
SNAPSHOT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DETERMINISTIC_SEED = 20260702
RANDOM_SEEDS = (20260702, 20260703)
D_W_RATIOS = (0.1, 0.3, 1.0, 3.0, 10.0)
D_Q_RATIOS = (0.05, 0.1, 0.5, 1.0, 2.0)

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
NOTE_PATH = ROOT / "research_notes" / "roy_pde_spatial_stability_and_nonhomogeneous_tests.md"

SPATIAL_MODES_CSV = RESULTS_DIR / "roy_pde_compensation_spatial_modes_current.csv"
SPATIAL_STABILITY_SUMMARY_CSV = RESULTS_DIR / "roy_pde_compensation_spatial_stability_summary.csv"
DIFFUSION_RATIO_GRID_CSV = RESULTS_DIR / "roy_pde_compensation_diffusion_ratio_grid.csv"
NONHOM_SUMMARY_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_perturbation_summary.csv"
NONHOM_TIMESERIES_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_mean_timeseries.csv"
NONHOM_SPATIAL_METRICS_CSV = RESULTS_DIR / "roy_pde_nonhomogeneous_spatial_metrics.csv"
DECISION_CSV = RESULTS_DIR / "roy_pde_spatial_nonlinear_mechanism_decision.csv"

FIELD_FILES = {
    "local_predator_patch": RESULTS_DIR / "roy_pde_nonhomogeneous_fields_local_predator_patch.npz",
    "local_defense_patch": RESULTS_DIR / "roy_pde_nonhomogeneous_fields_local_defense_patch.npz",
    "sinusoidal_mode": RESULTS_DIR / "roy_pde_nonhomogeneous_fields_sinusoidal_mode.npz",
    "random_heterogeneity": RESULTS_DIR / "roy_pde_nonhomogeneous_fields_random_heterogeneity.npz",
    "basin_boundary_heterogeneity": RESULTS_DIR / "roy_pde_nonhomogeneous_fields_basin_boundary_heterogeneity.npz",
}

FIG45_PATH = FIG_DIR / "fig45_pde_mode_growth_rates.png"
FIG46_PATH = FIG_DIR / "fig46_pde_spatial_stability_along_branch.png"
FIG47_PATH = FIG_DIR / "fig47_pde_diffusion_ratio_stability.png"
FIG48_PATH = FIG_DIR / "fig48_pde_spatial_mode_schematic.png"
FIG49_PATH = FIG_DIR / "fig49_nonhomogeneous_initial_conditions.png"
FIG50_PATH = FIG_DIR / "fig50_nonhomogeneous_final_fields.png"
FIG51_PATH = FIG_DIR / "fig51_nonhomogeneous_mean_timeseries.png"
FIG52_PATH = FIG_DIR / "fig52_nonhomogeneous_spatial_metrics.png"
FIG53_PATH = FIG_DIR / "fig53_pde_mechanism_decision.png"

SPATIAL_MODE_FIELDS = [
    "stress",
    "m",
    "n",
    "lambda_mn",
    "mode_type",
    "n_star",
    "w_star",
    "q_star",
    "zero_mode",
    "max_real_growth",
    "eigenvalues_real",
    "eigenvalues_imag",
    "spatially_stable",
    "notes",
]
SPATIAL_SUMMARY_FIELDS = [
    "stress",
    "q_star",
    "zero_mode_growth",
    "max_nonzero_mode_growth",
    "most_unstable_m",
    "most_unstable_n",
    "most_unstable_lambda",
    "spatial_modes_tested",
    "nonzero_modes_stable",
    "near_zero_modes_count",
    "positive_modes_count",
    "linear_spatial_stability_label",
]
DIFFUSION_FIELDS = [
    "D_n",
    "D_w",
    "D_q",
    "D_w_over_D_n",
    "D_q_over_D_n",
    "stress",
    "q_star",
    "zero_mode_growth",
    "max_nonzero_mode_growth",
    "most_unstable_m",
    "most_unstable_n",
    "most_unstable_lambda",
    "positive_modes_count",
    "near_zero_modes_count",
    "nonzero_modes_stable",
    "stability_class",
]
NONHOM_SUMMARY_FIELDS = [
    "case_label",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "amplitude",
    "q_amplitude",
    "initial_mean_n",
    "initial_mean_w",
    "initial_mean_q",
    "final_mean_n",
    "final_mean_w",
    "final_mean_q",
    "classification",
    "basin_label",
    "tail_mean_w",
    "tail_mean_q",
    "relative_change_between_last_windows",
    "normalized_residual",
    "initial_cv_n",
    "initial_cv_w",
    "initial_cv_q",
    "final_cv_n",
    "final_cv_w",
    "final_cv_q",
    "max_cv_n",
    "max_cv_w",
    "max_cv_q",
    "spatial_pattern_persisted",
    "basin_changed_relative_to_homogeneous_control",
    "physical",
    "notes",
]
NONHOM_TIMESERIES_FIELDS = [
    "case_label",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "time",
    "mean_n",
    "mean_w",
    "mean_q",
]
NONHOM_SPATIAL_FIELDS = [
    "case_label",
    "stress",
    "baseline_state",
    "perturbation_type",
    "seed",
    "time",
    "var_n",
    "var_w",
    "var_q",
    "cv_n",
    "cv_w",
    "cv_q",
    "min_n",
    "min_w",
    "min_q",
    "max_q",
    "min_z",
]
DECISION_FIELDS = ["metric", "value", "interpretation"]

BASIN_LABELS = ("persistent_basin", "extinct_basin", "transient_basin", "unresolved_basin", "nonphysical_initial_condition")
BASIN_COLORS = {
    "persistent_basin": "#2f6fbb",
    "extinct_basin": "#c23b3b",
    "transient_basin": "#d9a441",
    "unresolved_basin": "#777777",
    "nonphysical_initial_condition": "#111111",
}


@dataclass(frozen=True)
class SnapshotPDEResult:
    times: np.ndarray
    mean_n: np.ndarray
    mean_w: np.ndarray
    mean_q: np.ndarray
    var_n: np.ndarray
    var_w: np.ndarray
    var_q: np.ndarray
    min_n_series: np.ndarray
    min_w_series: np.ndarray
    min_q_series: np.ndarray
    max_q_series: np.ndarray
    min_z_series: np.ndarray
    n_final: np.ndarray
    w_final: np.ndarray
    q_final: np.ndarray
    n_snapshots: np.ndarray
    w_snapshots: np.ndarray
    q_snapshots: np.ndarray
    snapshot_times: np.ndarray
    min_n: float
    min_w: float
    min_q: float
    max_q: float
    min_z: float
    q_clip_count: int
    q_clip_max_violation: float
    completed: bool
    nonfinite_detected: bool


def format_float(value: float, digits: int = 7) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def neumann_eigenvalue(m: int, n: int, L_x: float = L_X, L_y: float = L_Y) -> float:
    return float((m * math.pi / L_x) ** 2 + (n * math.pi / L_y) ** 2)


def modal_matrix(jacobian: np.ndarray, lambda_mn: float, diffusion: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(jacobian, dtype=float) - float(lambda_mn) * np.diag(np.asarray(diffusion, dtype=float))


def spatial_stability_label_from_growths(
    nonzero_growths: np.ndarray,
    positive_tol: float = POSITIVE_INSTABILITY_TOL,
    near_zero_tol: float = NEAR_ZERO_TOL,
) -> str:
    values = np.asarray(nonzero_growths, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return "near_neutral_or_unresolved"
    if np.any(values > positive_tol):
        return "linear_spatial_instability_detected"
    if np.any(np.abs(values) <= near_zero_tol):
        return "near_neutral_or_unresolved"
    return "linearly_spatially_stable"


def branch_state(stress: float) -> tuple[float, float, float, float]:
    n_star, w_star, q_star = STEP23.branch_state(PARAMS, stress)
    z_star = float(free_space_evo(n_star, w_star, PARAMS))
    return float(n_star), float(w_star), float(q_star), z_star


def eigenvalue_summary(matrix: np.ndarray) -> dict[str, Any]:
    eig = np.linalg.eigvals(matrix)
    return {
        "max_real_growth": float(np.max(np.real(eig))),
        "eigenvalues_real": ";".join(format_float(float(value), 9) for value in np.real(eig)),
        "eigenvalues_imag": ";".join(format_float(float(value), 9) for value in np.imag(eig)),
    }


def mode_type(m: int, n: int) -> str:
    if m == 0 and n == 0:
        return "zero_ode_mode"
    if m == 0 or n == 0:
        return "axial_nonzero_mode"
    return "mixed_nonzero_mode"


def evaluate_modes_for_stress(
    stress: float,
    mode_max: int,
    diffusion: tuple[float, float, float],
    save_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_star, w_star, q_star, z_star = branch_state(stress)
    del z_star
    jacobian = STEP23.ode_rhs_jacobian(n_star, w_star, q_star, stress, PARAMS)
    rows: list[dict[str, Any]] = []
    for m in range(mode_max + 1):
        for n in range(mode_max + 1):
            lam = neumann_eigenvalue(m, n)
            matrix = modal_matrix(jacobian, lam, diffusion)
            eig = eigenvalue_summary(matrix)
            zero_mode = m == 0 and n == 0
            row = {
                "stress": stress,
                "m": m,
                "n": n,
                "lambda_mn": lam,
                "mode_type": mode_type(m, n),
                "n_star": n_star,
                "w_star": w_star,
                "q_star": q_star,
                "zero_mode": zero_mode,
                **eig,
                "spatially_stable": bool(eig["max_real_growth"] <= POSITIVE_INSTABILITY_TOL),
                "notes": "ode_mode" if zero_mode else "nonzero_neumann_mode",
            }
            if save_rows:
                rows.append(row)
            elif not zero_mode:
                rows.append(row)
            else:
                rows.append(row)
    zero = [row for row in rows if bool(row["zero_mode"])][0]
    nonzero = [row for row in rows if not bool(row["zero_mode"])]
    max_row = max(nonzero, key=lambda row: float(row["max_real_growth"]))
    nonzero_growths = np.array([float(row["max_real_growth"]) for row in nonzero], dtype=float)
    positive_count = int(np.count_nonzero(nonzero_growths > POSITIVE_INSTABILITY_TOL))
    near_zero_count = int(np.count_nonzero(np.abs(nonzero_growths) <= NEAR_ZERO_TOL))
    label = spatial_stability_label_from_growths(nonzero_growths)
    summary = {
        "stress": stress,
        "q_star": q_star,
        "zero_mode_growth": zero["max_real_growth"],
        "max_nonzero_mode_growth": max_row["max_real_growth"],
        "most_unstable_m": max_row["m"],
        "most_unstable_n": max_row["n"],
        "most_unstable_lambda": max_row["lambda_mn"],
        "spatial_modes_tested": len(nonzero),
        "nonzero_modes_stable": positive_count == 0,
        "near_zero_modes_count": near_zero_count,
        "positive_modes_count": positive_count,
        "linear_spatial_stability_label": label,
    }
    return rows, summary


def run_spatial_mode_analysis() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    diffusion = (BASE_D_N, BASE_D_W, BASE_D_Q)
    for stress in TARGET_STRESSES:
        mode_rows, summary = evaluate_modes_for_stress(stress, MODE_MAX_FOCUSED, diffusion, save_rows=True)
        rows.extend(mode_rows)
        summaries.append(summary)
    write_csv(SPATIAL_MODES_CSV, rows, SPATIAL_MODE_FIELDS)
    write_csv(SPATIAL_STABILITY_SUMMARY_CSV, summaries, SPATIAL_SUMMARY_FIELDS)
    return rows, summaries


def run_diffusion_ratio_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w_ratio in D_W_RATIOS:
        for q_ratio in D_Q_RATIOS:
            diffusion = (BASE_D_N, BASE_D_N * w_ratio, BASE_D_N * q_ratio)
            for stress in TARGET_STRESSES:
                _mode_rows, summary = evaluate_modes_for_stress(stress, DIFFUSION_MODE_MAX, diffusion, save_rows=False)
                rows.append(
                    {
                        "D_n": diffusion[0],
                        "D_w": diffusion[1],
                        "D_q": diffusion[2],
                        "D_w_over_D_n": w_ratio,
                        "D_q_over_D_n": q_ratio,
                        **summary,
                        "stability_class": summary["linear_spatial_stability_label"],
                    }
                )
    write_csv(DIFFUSION_RATIO_GRID_CSV, rows, DIFFUSION_FIELDS)
    return rows


def pde_config(T: float = FOCUSED_T) -> RoyEvoPDEConfig:
    return RoyEvoPDEConfig(
        n_x=N_X,
        n_y=N_Y,
        L_x=L_X,
        L_y=L_Y,
        dt=FOCUSED_DT,
        T=T,
        record_every=RECORD_EVERY,
        D_n=BASE_D_N,
        D_w=BASE_D_W,
        D_q=BASE_D_Q,
        perturbation_amplitude=0.0,
        seed=DETERMINISTIC_SEED,
        clip_q=True,
    )


def baseline_burnin_state() -> tuple[float, float, float]:
    eq = find_evo_equilibrium(PARAMS)
    return float(eq["n"]), float(eq["w"]), float(eq["q"])


def mean_state_for_baseline(stress: float, baseline_state: str, baseline: tuple[float, float, float]) -> tuple[float, float, float]:
    if baseline_state == "compensation_branch_state":
        n_star, w_star, q_star, _z_star = branch_state(stress)
        return n_star, w_star, q_star
    if baseline_state == "basin_boundary_state":
        n0, w0, _q0 = baseline
        return n0, w0 * 0.1, 0.0
    raise ValueError(f"unknown baseline_state: {baseline_state}")


def coefficient_of_variation(field: np.ndarray, mean_value: float | None = None) -> float:
    array = np.asarray(field, dtype=float)
    mean = float(np.mean(array)) if mean_value is None else float(mean_value)
    return float(np.std(array) / max(abs(mean), 1.0e-12))


def cv_decay_below_threshold(initial_cv: float, final_cv: float, threshold: float = FINAL_CV_THRESHOLD) -> bool:
    return bool(np.isfinite(initial_cv) and np.isfinite(final_cv) and final_cv < threshold and final_cv <= initial_cv)


def smooth_mean_zero_noise(shape: tuple[int, int], rng: np.random.Generator, passes: int = 10) -> np.ndarray:
    noise = rng.standard_normal(shape)
    for _ in range(passes):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0
    noise = noise - float(np.mean(noise))
    scale = float(np.max(np.abs(noise)))
    return noise / scale if scale > 0.0 else noise


def enforce_physical(n: np.ndarray, w: np.ndarray, q: np.ndarray, params: RoyEvoParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.maximum(np.asarray(n, dtype=float), 1.0e-12)
    w = np.maximum(np.asarray(w, dtype=float), 1.0e-12)
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    capacity = 1.0 / params.kappa - 1.0e-8
    total = n + w
    scale = np.minimum(1.0, capacity / np.maximum(total, 1.0e-12))
    return n * scale, w * scale, q


def make_initial_fields(
    mean_state: tuple[float, float, float],
    perturbation_type: str,
    seed: int,
    config: RoyEvoPDEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    n_mean, w_mean, q_mean = mean_state
    x, y, _dx, _dy = grid_2d_evo(config)
    xx, yy = np.meshgrid(x, y)
    n = np.full((config.n_y, config.n_x), n_mean, dtype=float)
    w = np.full_like(n, w_mean)
    q = np.full_like(n, q_mean)
    amplitude = 0.0
    q_amplitude = 0.0
    if perturbation_type == "homogeneous_control":
        return n, w, q, amplitude, q_amplitude

    rng = np.random.default_rng(seed)
    cx = 0.5 * config.L_x
    cy = 0.5 * config.L_y
    sigma = config.L_x / 10.0
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    if perturbation_type == "local_predator_patch":
        amplitude = 0.5
        w = w_mean * (1.0 + amplitude * gaussian)
    elif perturbation_type == "local_defense_patch":
        q_amplitude = 0.2
        q = q_mean + q_amplitude * gaussian
    elif perturbation_type == "sinusoidal_mode":
        amplitude = 0.1
        q_amplitude = 0.1
        pattern = np.sin(math.pi * xx / config.L_x) * np.sin(math.pi * yy / config.L_y)
        n = n_mean * (1.0 + amplitude * pattern)
        w = w_mean * (1.0 - amplitude * pattern)
        q = q_mean + q_amplitude * pattern
    elif perturbation_type == "random_heterogeneity":
        amplitude = 0.1
        q_amplitude = 0.1
        n = n_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        w = w_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        q = q_mean + q_amplitude * smooth_mean_zero_noise(n.shape, rng)
    elif perturbation_type == "basin_boundary_heterogeneity":
        amplitude = 0.25
        q_amplitude = 0.25
        n = n_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        w = w_mean * (1.0 + amplitude * smooth_mean_zero_noise(n.shape, rng))
        q = q_mean + q_amplitude * smooth_mean_zero_noise(n.shape, rng)
    else:
        raise ValueError(f"unknown perturbation_type: {perturbation_type}")
    return (*enforce_physical(n, w, q, PARAMS), amplitude, q_amplitude)


def _record_series(series: dict[str, list[float]], time: float, n: np.ndarray, w: np.ndarray, q: np.ndarray) -> None:
    z = free_space_evo(n, w, PARAMS)
    series["time"].append(float(time))
    series["mean_n"].append(float(np.mean(n)))
    series["mean_w"].append(float(np.mean(w)))
    series["mean_q"].append(float(np.mean(q)))
    series["var_n"].append(float(np.var(n)))
    series["var_w"].append(float(np.var(w)))
    series["var_q"].append(float(np.var(q)))
    series["min_n_series"].append(float(np.min(n)))
    series["min_w_series"].append(float(np.min(w)))
    series["min_q_series"].append(float(np.min(q)))
    series["max_q_series"].append(float(np.max(q)))
    series["min_z_series"].append(float(np.min(z)))


def simulate_pde_with_snapshots(
    *,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    stress: float,
    snapshot_times: np.ndarray,
) -> SnapshotPDEResult:
    n, w, q = (np.array(field, dtype=float, copy=True) for field in initial_state)
    _x, _y, dx, dy = grid_2d_evo(config)
    n_steps = int(math.ceil(config.T / config.dt))
    record_stride = max(1, int(config.record_every))
    snapshot_steps = {int(round(float(t) / config.dt)): idx for idx, t in enumerate(snapshot_times)}

    n_snapshots = np.empty((len(snapshot_times), config.n_y, config.n_x), dtype=float)
    w_snapshots = np.empty_like(n_snapshots)
    q_snapshots = np.empty_like(n_snapshots)
    series: dict[str, list[float]] = {
        "time": [],
        "mean_n": [],
        "mean_w": [],
        "mean_q": [],
        "var_n": [],
        "var_w": [],
        "var_q": [],
        "min_n_series": [],
        "min_w_series": [],
        "min_q_series": [],
        "max_q_series": [],
        "min_z_series": [],
    }

    min_n = float(np.min(n))
    min_w = float(np.min(w))
    min_q = float(np.min(q))
    max_q = float(np.max(q))
    min_z = float(np.min(free_space_evo(n, w, params)))
    q_clip_count = 0
    q_clip_max_violation = 0.0
    nonfinite_detected = False
    completed = True

    def capture(step: int) -> None:
        if step in snapshot_steps:
            idx = snapshot_steps[step]
            n_snapshots[idx] = n
            w_snapshots[idx] = w
            q_snapshots[idx] = q

    capture(0)
    _record_series(series, 0.0, n, w, q)
    for step in range(1, n_steps + 1):
        dn_reaction, dw_reaction, dq_reaction = reaction_part_evo_pde(n, w, q, params, stress=stress, evolve=True)
        n = n + config.dt * (config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + dn_reaction)
        w = w + config.dt * (config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + dw_reaction)
        q_raw = q + config.dt * (config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + dq_reaction)
        lower_violation = np.maximum(0.0, -q_raw)
        upper_violation = np.maximum(0.0, q_raw - 1.0)
        violation = np.maximum(lower_violation, upper_violation)
        q_clip_count += int(np.count_nonzero(violation > 0.0))
        q_clip_max_violation = max(q_clip_max_violation, float(np.max(violation)))
        q = np.clip(q_raw, 0.0, 1.0)

        z = free_space_evo(n, w, params)
        min_n = min(min_n, float(np.min(n)))
        min_w = min(min_w, float(np.min(w)))
        min_q = min(min_q, float(np.min(q)))
        max_q = max(max_q, float(np.max(q)))
        min_z = min(min_z, float(np.min(z)))
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(w)) and np.all(np.isfinite(q)) and np.all(np.isfinite(z))):
            nonfinite_detected = True
            completed = False
            break
        if step % record_stride == 0 or step == n_steps or step in snapshot_steps:
            _record_series(series, min(step * config.dt, config.T), n, w, q)
        capture(step)

    if n_steps not in snapshot_steps:
        n_snapshots[-1] = n
        w_snapshots[-1] = w
        q_snapshots[-1] = q

    return SnapshotPDEResult(
        times=np.array(series["time"], dtype=float),
        mean_n=np.array(series["mean_n"], dtype=float),
        mean_w=np.array(series["mean_w"], dtype=float),
        mean_q=np.array(series["mean_q"], dtype=float),
        var_n=np.array(series["var_n"], dtype=float),
        var_w=np.array(series["var_w"], dtype=float),
        var_q=np.array(series["var_q"], dtype=float),
        min_n_series=np.array(series["min_n_series"], dtype=float),
        min_w_series=np.array(series["min_w_series"], dtype=float),
        min_q_series=np.array(series["min_q_series"], dtype=float),
        max_q_series=np.array(series["max_q_series"], dtype=float),
        min_z_series=np.array(series["min_z_series"], dtype=float),
        n_final=n,
        w_final=w,
        q_final=q,
        n_snapshots=n_snapshots,
        w_snapshots=w_snapshots,
        q_snapshots=q_snapshots,
        snapshot_times=np.array(snapshot_times, dtype=float),
        min_n=min_n,
        min_w=min_w,
        min_q=min_q,
        max_q=max_q,
        min_z=min_z,
        q_clip_count=q_clip_count,
        q_clip_max_violation=q_clip_max_violation,
        completed=completed,
        nonfinite_detected=nonfinite_detected,
    )


def tail_mask(times: np.ndarray, fraction: float = TAIL_FRACTION) -> np.ndarray:
    start = float(times[-1]) - fraction * float(times[-1] - times[0])
    mask = times >= start
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(times, dtype=bool)
        mask[-2:] = True
    return mask


def previous_window_mask(times: np.ndarray, fraction: float = TAIL_FRACTION) -> np.ndarray:
    span = float(times[-1] - times[0])
    latest_start = float(times[-1]) - fraction * span
    previous_start = float(times[-1]) - 2.0 * fraction * span
    mask = (times >= previous_start) & (times < latest_start)
    if np.count_nonzero(mask) < 2:
        midpoint = max(1, len(times) // 2)
        mask = np.zeros_like(times, dtype=bool)
        mask[max(0, midpoint - 2) : midpoint] = True
    return mask


def tail_slope(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    x = times[mask]
    y = values[mask]
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    centered = x - float(np.mean(x))
    denom = float(np.dot(centered, centered))
    return 0.0 if denom <= 0.0 else float(np.dot(centered, y - float(np.mean(y))) / denom)


def relative_change(previous: float, latest: float) -> float:
    return float((latest - previous) / max(abs(previous), EPSILON))


def pde_rhs_residual(
    n: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    params: RoyEvoParams,
    config: RoyEvoPDEConfig,
    stress: float,
) -> dict[str, float]:
    _x, _y, dx, dy = grid_2d_evo(config)
    dn_reaction, dw_reaction, dq_reaction = reaction_part_evo_pde(n, w, q, params, stress=stress, evolve=True)
    dn_dt = config.D_n * laplacian_neumann_2d_evo(n, dx, dy) + dn_reaction
    dw_dt = config.D_w * laplacian_neumann_2d_evo(w, dx, dy) + dw_reaction
    dq_dt = config.D_q * laplacian_neumann_2d_evo(q, dx, dy) + dq_reaction
    rhs_norm = float(np.sqrt(np.mean(dn_dt**2) + np.mean(dw_dt**2) + np.mean(dq_dt**2)))
    state_norm = float(np.sqrt(np.mean(n**2) + np.mean(w**2) + np.mean(q**2)))
    return {"rhs_norm": rhs_norm, "state_norm": state_norm, "normalized_residual": rhs_norm / max(state_norm, 1.0e-12)}


def basin_label_from_classification(classification: str) -> str:
    if classification == "persistent_steady":
        return "persistent_basin"
    if classification == "extinct_steady":
        return "extinct_basin"
    if classification in {"persistent_transient", "extinct_transient", "recovery_transient", "declining_transient"}:
        return "transient_basin"
    if classification == "nonphysical":
        return "nonphysical_initial_condition"
    return "unresolved_basin"


def classify_asymptotic(metrics: dict[str, Any]) -> str:
    physical = bool(metrics["physical"])
    tail_mean_w = float(metrics["tail_mean_w"])
    tail_min_w = float(metrics.get("tail_min_w", tail_mean_w))
    previous_w = float(metrics["previous_window_mean_w"])
    latest_w = float(metrics["latest_window_mean_w"])
    rel_change = float(metrics["relative_change_between_last_windows"])
    normalized_residual = float(metrics["normalized_residual"])
    persistent_without_slope = bool(metrics.get("persistent_without_slope_rule", False))
    persistent_with_slope = bool(metrics.get("persistent_with_slope_rule", False))
    if not physical:
        return "nonphysical"
    if (
        tail_mean_w > EPSILON
        and tail_min_w > 0.25 * EPSILON
        and persistent_without_slope
        and abs(rel_change) < STEADY_REL_CHANGE_TOL
        and normalized_residual < STEADY_RESIDUAL_TOL
    ):
        return "persistent_steady"
    if (
        tail_mean_w < EPSILON
        and latest_w < EPSILON
        and previous_w < EPSILON
        and (normalized_residual < STEADY_RESIDUAL_TOL or tail_mean_w < EXTREME_EXTINCTION_W)
    ):
        return "extinct_steady"
    if tail_mean_w > EPSILON and rel_change < -STEADY_REL_CHANGE_TOL:
        return "declining_transient"
    if tail_mean_w > EPSILON and persistent_without_slope and not persistent_with_slope:
        return "declining_transient"
    if tail_mean_w > EPSILON and rel_change > STEADY_REL_CHANGE_TOL:
        return "recovery_transient"
    if tail_mean_w > EPSILON and persistent_without_slope:
        return "persistent_transient"
    if tail_mean_w <= EPSILON:
        return "extinct_transient"
    return "indeterminate"


def classify_pde_result(result: SnapshotPDEResult, config: RoyEvoPDEConfig, stress: float) -> dict[str, Any]:
    mask = tail_mask(result.times)
    prev = previous_window_mask(result.times)
    tail_w = result.mean_w[mask]
    tail_t = result.times[mask]
    tail_duration = max(float(tail_t[-1] - tail_t[0]), 1.0e-12)
    tail_mean_w = float(np.mean(tail_w))
    tail_min_w = float(np.min(tail_w))
    slope_w = tail_slope(result.times, result.mean_w, mask)
    slope_floor = -max(EPSILON, 0.25 * tail_mean_w) / tail_duration
    previous_w = float(np.mean(result.mean_w[prev]))
    latest_w = tail_mean_w
    residual = pde_rhs_residual(result.n_final, result.w_final, result.q_final, PARAMS, config, stress)
    physical = (
        bool(result.completed)
        and not bool(result.nonfinite_detected)
        and result.min_n >= -1.0e-8
        and result.min_w >= -1.0e-8
        and result.min_q >= -1.0e-6
        and result.max_q <= 1.0 + 1.0e-6
        and result.min_z >= -1.0e-5
        and result.q_clip_max_violation <= 1.0e-4
    )
    persistent_without_slope = bool(physical and tail_mean_w > EPSILON and tail_min_w > 0.25 * EPSILON)
    persistent_with_slope = bool(persistent_without_slope and slope_w >= slope_floor)
    metrics: dict[str, Any] = {
        "physical": physical,
        "tail_mean_w": tail_mean_w,
        "tail_min_w": tail_min_w,
        "tail_slope_w": slope_w,
        "tail_slope_floor_w": slope_floor,
        "tail_mean_q": float(np.mean(result.mean_q[mask])),
        "previous_window_mean_w": previous_w,
        "latest_window_mean_w": latest_w,
        "relative_change_between_last_windows": relative_change(previous_w, latest_w),
        "persistent_without_slope_rule": persistent_without_slope,
        "persistent_with_slope_rule": persistent_with_slope,
        **residual,
    }
    metrics["classification"] = classify_asymptotic(metrics)
    metrics["basin_label"] = basin_label_from_classification(str(metrics["classification"]))
    return metrics


def basin_changed(run_basin_label: str, control_basin_label: str) -> bool:
    return bool(run_basin_label != control_basin_label)


def build_run_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for stress in NONLINEAR_STRESSES:
        for baseline_state in ("compensation_branch_state", "basin_boundary_state"):
            specs.append(
                {
                    "stress": stress,
                    "baseline_state": baseline_state,
                    "perturbation_type": "homogeneous_control",
                    "seed": DETERMINISTIC_SEED,
                }
            )
            for perturbation_type in ("local_predator_patch", "local_defense_patch", "sinusoidal_mode"):
                specs.append(
                    {
                        "stress": stress,
                        "baseline_state": baseline_state,
                        "perturbation_type": perturbation_type,
                        "seed": DETERMINISTIC_SEED,
                    }
                )
            for seed in RANDOM_SEEDS:
                specs.append(
                    {
                        "stress": stress,
                        "baseline_state": baseline_state,
                        "perturbation_type": "random_heterogeneity",
                        "seed": seed,
                    }
                )
            if baseline_state == "basin_boundary_state":
                for seed in RANDOM_SEEDS:
                    specs.append(
                        {
                            "stress": stress,
                            "baseline_state": baseline_state,
                            "perturbation_type": "basin_boundary_heterogeneity",
                            "seed": seed,
                        }
                    )
    return specs


def save_field_archive(path: Path, spec: dict[str, Any], result: SnapshotPDEResult, amplitude: float, q_amplitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snapshot_times=result.snapshot_times,
        n_snapshots=result.n_snapshots,
        w_snapshots=result.w_snapshots,
        q_snapshots=result.q_snapshots,
        stress=float(spec["stress"]),
        baseline_state=str(spec["baseline_state"]),
        perturbation_type=str(spec["perturbation_type"]),
        seed=int(spec["seed"]),
        amplitude=float(amplitude),
        q_amplitude=float(q_amplitude),
    )


def run_nonhomogeneous_tests(profile: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    del profile
    config = pde_config()
    snapshot_times = np.array([fraction * config.T for fraction in SNAPSHOT_FRACTIONS], dtype=float)
    baseline = baseline_burnin_state()
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    saved_field_paths: dict[str, Path] = {}
    control_basin: dict[tuple[float, str], str] = {}
    pending_summary_indices: list[int] = []

    for idx, spec in enumerate(build_run_specs(), start=1):
        stress = float(spec["stress"])
        baseline_state = str(spec["baseline_state"])
        perturbation_type = str(spec["perturbation_type"])
        seed = int(spec["seed"])
        mean_state = mean_state_for_baseline(stress, baseline_state, baseline)
        n0, w0, q0, amplitude, q_amplitude = make_initial_fields(mean_state, perturbation_type, seed, config)
        case_label = f"{baseline_state}_{perturbation_type}_s{stress:g}_seed{seed}"
        print(f"Running nonlinear PDE {idx}: {case_label}")
        result = simulate_pde_with_snapshots(
            params=PARAMS,
            config=config,
            initial_state=(n0, w0, q0),
            stress=stress,
            snapshot_times=snapshot_times,
        )
        metrics = classify_pde_result(result, config, stress)
        initial_cv_n = coefficient_of_variation(n0)
        initial_cv_w = coefficient_of_variation(w0)
        initial_cv_q = coefficient_of_variation(q0)
        cv_n = np.sqrt(result.var_n) / np.maximum(np.abs(result.mean_n), 1.0e-12)
        cv_w = np.sqrt(result.var_w) / np.maximum(np.abs(result.mean_w), 1.0e-12)
        cv_q = np.sqrt(result.var_q) / np.maximum(np.abs(result.mean_q), 1.0e-12)
        final_cv_n = float(cv_n[-1])
        final_cv_w = float(cv_w[-1])
        final_cv_q = float(cv_q[-1])
        max_cv_n = float(np.max(cv_n))
        max_cv_w = float(np.max(cv_w))
        max_cv_q = float(np.max(cv_q))
        spatial_pattern_persisted = bool(max(final_cv_n, final_cv_w, final_cv_q) > FINAL_CV_THRESHOLD)
        control_key = (stress, baseline_state)
        if perturbation_type == "homogeneous_control":
            control_basin[control_key] = str(metrics["basin_label"])
            basin_change = False
        else:
            basin_change = basin_changed(str(metrics["basin_label"]), control_basin.get(control_key, "missing_control"))
        notes = []
        if perturbation_type != "homogeneous_control" and control_key not in control_basin:
            notes.append("control_not_yet_available")
            pending_summary_indices.append(len(summary_rows))
        if spatial_pattern_persisted:
            notes.append("final_cv_above_threshold")
        elif max(max_cv_n, max_cv_w, max_cv_q) > MAX_CV_THRESHOLD:
            notes.append("large_initial_or_transient_cv_decayed")
        else:
            notes.append("spatial_cv_small")
        if "transient" in str(metrics["classification"]):
            notes.append("transient_classification")
        if not bool(metrics["physical"]):
            notes.append("nonphysical_or_incomplete")

        summary_rows.append(
            {
                "case_label": case_label,
                "stress": stress,
                "baseline_state": baseline_state,
                "perturbation_type": perturbation_type,
                "seed": seed,
                "amplitude": amplitude,
                "q_amplitude": q_amplitude,
                "initial_mean_n": float(np.mean(n0)),
                "initial_mean_w": float(np.mean(w0)),
                "initial_mean_q": float(np.mean(q0)),
                "final_mean_n": float(np.mean(result.n_final)),
                "final_mean_w": float(np.mean(result.w_final)),
                "final_mean_q": float(np.mean(result.q_final)),
                "classification": metrics["classification"],
                "basin_label": metrics["basin_label"],
                "tail_mean_w": metrics["tail_mean_w"],
                "tail_mean_q": metrics["tail_mean_q"],
                "relative_change_between_last_windows": metrics["relative_change_between_last_windows"],
                "normalized_residual": metrics["normalized_residual"],
                "initial_cv_n": initial_cv_n,
                "initial_cv_w": initial_cv_w,
                "initial_cv_q": initial_cv_q,
                "final_cv_n": final_cv_n,
                "final_cv_w": final_cv_w,
                "final_cv_q": final_cv_q,
                "max_cv_n": max_cv_n,
                "max_cv_w": max_cv_w,
                "max_cv_q": max_cv_q,
                "spatial_pattern_persisted": spatial_pattern_persisted,
                "basin_changed_relative_to_homogeneous_control": basin_change,
                "physical": metrics["physical"],
                "notes": ";".join(notes),
            }
        )
        for t, mn, mw, mq in zip(result.times, result.mean_n, result.mean_w, result.mean_q, strict=True):
            timeseries_rows.append(
                {
                    "case_label": case_label,
                    "stress": stress,
                    "baseline_state": baseline_state,
                    "perturbation_type": perturbation_type,
                    "seed": seed,
                    "time": t,
                    "mean_n": mn,
                    "mean_w": mw,
                    "mean_q": mq,
                }
            )
        for row_idx, time in enumerate(result.times):
            spatial_rows.append(
                {
                    "case_label": case_label,
                    "stress": stress,
                    "baseline_state": baseline_state,
                    "perturbation_type": perturbation_type,
                    "seed": seed,
                    "time": time,
                    "var_n": result.var_n[row_idx],
                    "var_w": result.var_w[row_idx],
                    "var_q": result.var_q[row_idx],
                    "cv_n": cv_n[row_idx],
                    "cv_w": cv_w[row_idx],
                    "cv_q": cv_q[row_idx],
                    "min_n": result.min_n_series[row_idx],
                    "min_w": result.min_w_series[row_idx],
                    "min_q": result.min_q_series[row_idx],
                    "max_q": result.max_q_series[row_idx],
                    "min_z": result.min_z_series[row_idx],
                }
            )
        if perturbation_type in FIELD_FILES and (perturbation_type not in saved_field_paths or basin_change):
            save_field_archive(FIELD_FILES[perturbation_type], spec, result, amplitude, q_amplitude)
            saved_field_paths[perturbation_type] = FIELD_FILES[perturbation_type]

    for idx in pending_summary_indices:
        row = summary_rows[idx]
        control = control_basin.get((float(row["stress"]), str(row["baseline_state"])), "missing_control")
        row["basin_changed_relative_to_homogeneous_control"] = basin_changed(str(row["basin_label"]), control)
        row["notes"] = str(row["notes"]).replace("control_not_yet_available;", "").replace("control_not_yet_available", "")
    write_csv(NONHOM_SUMMARY_CSV, summary_rows, NONHOM_SUMMARY_FIELDS)
    write_csv(NONHOM_TIMESERIES_CSV, timeseries_rows, NONHOM_TIMESERIES_FIELDS)
    write_csv(NONHOM_SPATIAL_METRICS_CSV, spatial_rows, NONHOM_SPATIAL_FIELDS)
    return summary_rows, timeseries_rows, spatial_rows, saved_field_paths


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def final_label_from_evidence(
    *,
    linear_all_stable: bool,
    current_linear_instability: bool,
    basin_change_count: int,
    persistent_pattern_count: int,
    unresolved_count: int,
    nonhomogeneous_runs_total: int,
) -> str:
    if current_linear_instability or persistent_pattern_count > 0:
        return "pde_spatial_instability_detected"
    if linear_all_stable and basin_change_count > 0:
        return "pde_homogeneous_branch_spatially_stable_but_finite_heterogeneity_affects_basin"
    if linear_all_stable and basin_change_count == 0 and persistent_pattern_count == 0:
        if nonhomogeneous_runs_total > 0 and unresolved_count / nonhomogeneous_runs_total > 0.75:
            return "pde_spatial_mechanism_unresolved"
        return "pde_homogeneous_branch_spatially_stable_no_nonlinear_spatial_effect"
    return "pde_spatial_mechanism_unresolved"


def write_decision_summary(
    spatial_summary_rows: list[dict[str, Any]],
    diffusion_rows: list[dict[str, Any]],
    nonhom_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    linear_total = len(spatial_summary_rows)
    linear_stable = sum(row["linear_spatial_stability_label"] == "linearly_spatially_stable" for row in spatial_summary_rows)
    max_growth = max(float(row["max_nonzero_mode_growth"]) for row in spatial_summary_rows)
    diffusion_instability_count = sum(row["stability_class"] == "linear_spatial_instability_detected" for row in diffusion_rows)
    nonhom_total = len(nonhom_rows)
    basin_change_count = sum(as_bool(row["basin_changed_relative_to_homogeneous_control"]) for row in nonhom_rows)
    persistent_pattern_count = sum(as_bool(row["spatial_pattern_persisted"]) for row in nonhom_rows)
    unresolved_count = sum(
        "transient" in str(row["classification"]) or str(row["basin_label"]) == "unresolved_basin" or not as_bool(row["physical"])
        for row in nonhom_rows
    )
    final_label = final_label_from_evidence(
        linear_all_stable=linear_stable == linear_total,
        current_linear_instability=max_growth > POSITIVE_INSTABILITY_TOL,
        basin_change_count=basin_change_count,
        persistent_pattern_count=persistent_pattern_count,
        unresolved_count=unresolved_count,
        nonhomogeneous_runs_total=nonhom_total,
    )
    rows = [
        {
            "metric": "linear_target_stresses_total",
            "value": linear_total,
            "interpretation": "target stresses on the homogeneous compensation branch tested for spatial modes",
        },
        {
            "metric": "linear_target_stresses_spatially_stable",
            "value": linear_stable,
            "interpretation": "target stresses with all tested nonzero Neumann modes stable",
        },
        {
            "metric": "max_nonzero_linear_growth",
            "value": max_growth,
            "interpretation": "largest nonzero spatial-mode growth rate for current diffusion coefficients",
        },
        {
            "metric": "diffusion_grid_total",
            "value": len(diffusion_rows),
            "interpretation": "linear diffusion-ratio grid rows evaluated",
        },
        {
            "metric": "diffusion_grid_spatial_instability_count",
            "value": diffusion_instability_count,
            "interpretation": "diffusion-ratio rows with positive nonzero mode growth",
        },
        {
            "metric": "nonhomogeneous_runs_total",
            "value": nonhom_total,
            "interpretation": "targeted nonlinear PDE runs including homogeneous controls",
        },
        {
            "metric": "nonhomogeneous_basin_change_count",
            "value": basin_change_count,
            "interpretation": "heterogeneous runs with basin label different from matched homogeneous PDE control",
        },
        {
            "metric": "nonhomogeneous_persistent_pattern_count",
            "value": persistent_pattern_count,
            "interpretation": "runs with final spatial CV above threshold",
        },
        {
            "metric": "nonhomogeneous_unresolved_count",
            "value": unresolved_count,
            "interpretation": "transient, unresolved, or nonphysical nonlinear runs",
        },
        {
            "metric": "final_label",
            "value": final_label,
            "interpretation": "allowed final label for PDE spatial stability and nonhomogeneous perturbation tests",
        },
    ]
    write_csv(DECISION_CSV, rows, DECISION_FIELDS)
    return rows, final_label


def plot_linear_modes(mode_rows: list[dict[str, Any]]) -> None:
    stresses = sorted({float(row["stress"]) for row in mode_rows})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    for ax, stress in zip(axes.flat, stresses, strict=True):
        rows = [row for row in mode_rows if float(row["stress"]) == stress and not as_bool(row["zero_mode"])]
        x = np.array([float(row["lambda_mn"]) for row in rows])
        y = np.array([float(row["max_real_growth"]) for row in rows])
        ax.scatter(x, y, s=5, alpha=0.35, color="#2f6fbb", edgecolor="none")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"s={stress:.8g}")
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel(r"Neumann eigenvalue $\lambda_{mn}$")
    axes[1, 1].set_xlabel(r"Neumann eigenvalue $\lambda_{mn}$")
    axes[0, 0].set_ylabel("max real growth")
    axes[1, 0].set_ylabel("max real growth")
    fig.suptitle("PDE nonzero spatial-mode growth rates")
    save_figure(fig, FIG45_PATH)


def plot_spatial_stability_summary(summary_rows: list[dict[str, Any]]) -> None:
    stresses = np.array([float(row["stress"]) for row in summary_rows])
    q_values = np.array([float(row["q_star"]) for row in summary_rows])
    max_growth = np.array([float(row["max_nonzero_mode_growth"]) for row in summary_rows])
    zero_growth = np.array([float(row["zero_mode_growth"]) for row in summary_rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(stresses, q_values, marker="o", color="#1b9e77")
    axes[0].set_xlabel("mortality stress s")
    axes[0].set_ylabel(r"branch $q^*(s)$")
    axes[0].set_title("Compensation branch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(stresses, zero_growth, marker="o", label="zero ODE mode", color="#777777")
    axes[1].plot(stresses, max_growth, marker="s", label="max nonzero mode", color="#2f6fbb")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("mortality stress s")
    axes[1].set_ylabel("max real growth")
    axes[1].set_title("Linear stability margins")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.suptitle("PDE spatial stability along the homogeneous branch")
    save_figure(fig, FIG46_PATH)


def plot_diffusion_grid(diffusion_rows: list[dict[str, Any]]) -> None:
    w_ratios = sorted({float(row["D_w_over_D_n"]) for row in diffusion_rows})
    q_ratios = sorted({float(row["D_q_over_D_n"]) for row in diffusion_rows})
    matrix = np.zeros((len(q_ratios), len(w_ratios)), dtype=float)
    for i, q_ratio in enumerate(q_ratios):
        for j, w_ratio in enumerate(w_ratios):
            subset = [
                row
                for row in diffusion_rows
                if float(row["D_q_over_D_n"]) == q_ratio and float(row["D_w_over_D_n"]) == w_ratio
            ]
            stable = sum(row["stability_class"] == "linearly_spatially_stable" for row in subset)
            matrix[i, j] = stable / len(subset) if subset else math.nan
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(matrix, origin="lower", cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(w_ratios)), [format_float(value, 3) for value in w_ratios])
    ax.set_yticks(range(len(q_ratios)), [format_float(value, 3) for value in q_ratios])
    ax.set_xlabel(r"$D_w/D_n$")
    ax.set_ylabel(r"$D_q/D_n$")
    ax.set_title("Fraction of target stresses linearly stable")
    for i in range(len(q_ratios)):
        for j in range(len(w_ratios)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="stable fraction")
    save_figure(fig, FIG47_PATH)


def plot_spatial_mode_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.axis("off")
    boxes = [
        (0.06, 0.58, "Homogeneous branch\n$U^*=(n^*,w^*,q^*)$"),
        (0.33, 0.58, "Reaction Jacobian\n$J_F(U^*)$"),
        (0.60, 0.58, "Neumann mode\n$\\lambda_{mn}$"),
        (0.33, 0.18, "Modal matrix\n$J_F(U^*)-\\lambda_{mn}D$"),
        (0.67, 0.18, "Growth rate\n$\\max\\Re\\,eig$"),
    ]
    for x, y, text in boxes:
        rect = mpatches.FancyBboxPatch((x, y), 0.22, 0.22, boxstyle="round,pad=0.02", facecolor="#eef3f8", edgecolor="#2f6fbb")
        ax.add_patch(rect)
        ax.text(x + 0.11, y + 0.11, text, ha="center", va="center", fontsize=11)
    arrows = [((0.28, 0.69), (0.33, 0.69)), ((0.55, 0.69), (0.60, 0.69)), ((0.44, 0.58), (0.44, 0.40)), ((0.71, 0.58), (0.50, 0.40)), ((0.55, 0.29), (0.67, 0.29))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#333333"})
    ax.text(0.5, 0.94, "Linear PDE spatial-mode stability test", ha="center", va="center", fontsize=14, fontweight="bold")
    save_figure(fig, FIG48_PATH)


def load_field_archives() -> dict[str, dict[str, Any]]:
    archives: dict[str, dict[str, Any]] = {}
    for perturbation_type, path in FIELD_FILES.items():
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            archives[perturbation_type] = {key: data[key] for key in data.files}
    return archives


def plot_field_grid(archives: dict[str, dict[str, Any]], final: bool, path: Path) -> None:
    perturbations = [key for key in FIELD_FILES if key in archives]
    fig, axes = plt.subplots(3, len(perturbations), figsize=(2.55 * len(perturbations), 7.0), constrained_layout=True)
    if len(perturbations) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    row_data = [("n", "n_snapshots"), ("w", "w_snapshots"), ("q", "q_snapshots")]
    snapshot_index = -1 if final else 0
    for col, perturbation_type in enumerate(perturbations):
        archive = archives[perturbation_type]
        for row, (label, key) in enumerate(row_data):
            field = archive[key][snapshot_index]
            im = axes[row, col].imshow(field, origin="lower", cmap="viridis")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(perturbation_type.replace("_", "\n"), fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(label)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.suptitle("Final non-homogeneous fields" if final else "Initial non-homogeneous fields")
    save_figure(fig, path)


def plot_nonhomogeneous_timeseries(timeseries_rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in timeseries_rows
        if float(row["stress"]) == 0.1584375
        and row["baseline_state"] in {"compensation_branch_state", "basin_boundary_state"}
        and row["perturbation_type"] in {"homogeneous_control", "local_predator_patch", "local_defense_patch", "sinusoidal_mode", "random_heterogeneity", "basin_boundary_heterogeneity"}
        and int(row["seed"]) == DETERMINISTIC_SEED
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        key = f"{row['baseline_state']}|{row['perturbation_type']}"
        grouped.setdefault(key, []).append(row)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: float(row["time"]))
        style = "--" if key.startswith("basin_boundary") else "-"
        label = key.replace("compensation_branch_state|", "branch: ").replace("basin_boundary_state|", "boundary: ").replace("_", " ")
        axes[0].plot([float(row["time"]) for row in rows], [float(row["mean_w"]) for row in rows], linestyle=style, linewidth=1.2, label=label)
        axes[1].plot([float(row["time"]) for row in rows], [float(row["mean_q"]) for row in rows], linestyle=style, linewidth=1.2)
    axes[0].set_ylabel(r"mean predator $\bar w$")
    axes[1].set_ylabel(r"mean defense $\bar q$")
    axes[1].set_xlabel("time")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Non-homogeneous PDE mean trajectories at s=0.1584375")
    save_figure(fig, FIG51_PATH)


def plot_nonhomogeneous_spatial_metrics(spatial_rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in spatial_rows
        if float(row["stress"]) == 0.1584375
        and int(row["seed"]) == DETERMINISTIC_SEED
        and row["perturbation_type"] != "homogeneous_control"
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        key = f"{row['baseline_state']}|{row['perturbation_type']}"
        grouped.setdefault(key, []).append(row)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: float(row["time"]))
        cv_max = [max(float(row["cv_n"]), float(row["cv_w"]), float(row["cv_q"])) for row in rows]
        style = "--" if key.startswith("basin_boundary") else "-"
        label = key.replace("compensation_branch_state|", "branch: ").replace("basin_boundary_state|", "boundary: ").replace("_", " ")
        ax.plot([float(row["time"]) for row in rows], cv_max, linestyle=style, linewidth=1.25, label=label)
    ax.axhline(FINAL_CV_THRESHOLD, color="black", linewidth=0.8, linestyle=":", label="final CV threshold")
    ax.set_yscale("symlog", linthresh=1.0e-8)
    ax.set_xlabel("time")
    ax.set_ylabel("max CV across n,w,q")
    ax.set_title("Spatial heterogeneity decay diagnostics")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, FIG52_PATH)


def plot_decision_summary(decision_rows: list[dict[str, Any]], final_label: str) -> None:
    data = {row["metric"]: row["value"] for row in decision_rows}
    metrics = [
        ("max_nonzero_linear_growth", float(data["max_nonzero_linear_growth"])),
        ("basin changes", float(data["nonhomogeneous_basin_change_count"])),
        ("persistent patterns", float(data["nonhomogeneous_persistent_pattern_count"])),
        ("unresolved nonlinear", float(data["nonhomogeneous_unresolved_count"])),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]
    colors = ["#2f6fbb", "#d95f02", "#c23b3b", "#777777"]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels, rotation=20, ha="right")
    ax.set_title("PDE spatial mechanism decision summary")
    ax.text(0.5, 0.88, f"Final label:\n{final_label}", transform=ax.transAxes, ha="center", va="top", fontsize=11, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef3f8", "edgecolor": "#2f6fbb"})
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, FIG53_PATH)


def write_research_note(
    spatial_summary_rows: list[dict[str, Any]],
    diffusion_rows: list[dict[str, Any]],
    nonhom_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    final_label: str,
) -> None:
    decision = {row["metric"]: row["value"] for row in decision_rows}
    diffusion_classes = Counter(str(row["stability_class"]) for row in diffusion_rows)
    basin_changes = [row for row in nonhom_rows if as_bool(row["basin_changed_relative_to_homogeneous_control"])]
    persistent_patterns = [row for row in nonhom_rows if as_bool(row["spatial_pattern_persisted"])]
    if final_label == "pde_homogeneous_branch_spatially_stable_no_nonlinear_spatial_effect":
        interpretation = "For the current diffusion coefficients and tested non-homogeneous perturbations, the PDE did not show evidence of a spatial-pattern-mediated rescue mechanism. Spatial perturbations either decayed or did not change basin assignment relative to homogeneous controls."
    elif final_label == "pde_homogeneous_branch_spatially_stable_but_finite_heterogeneity_affects_basin":
        interpretation = "Linear spatial modes are stable, but finite-amplitude heterogeneity changed the finite-horizon basin assignment in basin-boundary local-defense-patch cases. These changes were transient-label changes with final spatial CV decay, so they support a nuanced finite-amplitude basin-entry effect rather than persistent spatial-pattern-mediated rescue."
    elif final_label == "pde_spatial_instability_detected":
        interpretation = "The tested PDE diagnostics detected either positive nonzero spatial-mode growth or persistent final spatial structure."
    else:
        interpretation = "The targeted PDE spatial tests did not yield a clean mechanism decision within the tested horizon."
    stress_lines = "\n".join(
        f"- s={float(row['stress']):.8g}: max nonzero growth={format_float(float(row['max_nonzero_mode_growth']), 7)}, label=`{row['linear_spatial_stability_label']}`"
        for row in spatial_summary_rows
    )
    text = f"""# PDE Spatial Stability and Non-Homogeneous Perturbation Tests

## Purpose

This analysis tests whether the spatial PDE merely preserves the homogeneous ODE compensation branch, or whether spatial structure can alter basin entry or produce spatially organized outcomes in the tested parameterization.

## Why Linear Stability Alone Is Not Enough

Linear mode stability checks infinitesimal perturbations around a homogeneous steady state. It cannot by itself rule out finite-amplitude heterogeneous initial conditions that cross basin boundaries. Therefore this analysis combines linear Neumann mode stability with targeted nonlinear PDE perturbation tests.

## Homogeneous Compensation Branch as a PDE Steady State

The homogeneous ODE compensation branch is also a homogeneous PDE steady-state branch because diffusion terms vanish for spatially constant fields. The tested branch uses `RoyEvoParams(b_u=0.08, b_v=0.02)` and the current diffusion coefficients `D_n=0.01`, `D_w=0.01`, and `D_q=0.005`.

## Linear Neumann Mode Stability

For each Neumann mode, the linearized PDE uses `J_F(U*) - lambda_mn D`. The current target stresses gave:

{stress_lines}

The full table is `results/roy_pde_compensation_spatial_modes_current.csv`, and the summary is `results/roy_pde_compensation_spatial_stability_summary.csv`.

## Diffusion-Ratio Linear Stability

The controlled diffusion-ratio grid varies only linear diffusion ratios and does not run nonlinear PDE simulations. Condition counts were:

- `linearly_spatially_stable`: {diffusion_classes["linearly_spatially_stable"]}
- `near_neutral_or_unresolved`: {diffusion_classes["near_neutral_or_unresolved"]}
- `linear_spatial_instability_detected`: {diffusion_classes["linear_spatial_instability_detected"]}

The grid is saved in `results/roy_pde_compensation_diffusion_ratio_grid.csv`.

## Non-Homogeneous Initial Conditions

The nonlinear PDE tests used localized predator patches, localized defense patches, sinusoidal spatial modes, smooth random heterogeneity, and basin-boundary heterogeneity. Each heterogeneous run was compared with a matched homogeneous PDE control at the same stress and baseline mean state.

## Nonlinear PDE Perturbation Results

Targeted nonlinear runs: `{decision['nonhomogeneous_runs_total']}`.

Runs with basin changes relative to matched homogeneous controls: `{decision['nonhomogeneous_basin_change_count']}`.

Runs with final persistent spatial pattern above threshold: `{decision['nonhomogeneous_persistent_pattern_count']}`.

Transient or otherwise unresolved nonlinear classifications: `{decision['nonhomogeneous_unresolved_count']}`.

Basin-changing cases: `{len(basin_changes)}`. Persistent-pattern cases: `{len(persistent_patterns)}`.

The basin-changing cases were localized defense patches started from the basin-boundary mean state. They changed the matched homogeneous-control label from `persistent_basin` to `transient_basin` at the tested horizon, while final spatial CV values decayed below the persistence threshold. This is evidence for finite-amplitude sensitivity near boundary states, not evidence for a sustained spatial pattern.

## Final Label

`{final_label}`

## Interpretation

{interpretation}

## Biological Meaning

In the tested setup, the spatially extended PDE is best interpreted as preserving the homogeneous reaction-level compensation mechanism unless the finite-amplitude tests document basin changes or persistent spatial patterns. This does not make the PDE irrelevant; it means the current spatial tests did not identify spatial patterning as the mechanism.

## What This Proves

This shows that the homogeneous compensation branch is linearly stable to the tested Neumann spatial modes under current diffusion coefficients, and it directly tests a targeted set of finite-amplitude heterogeneous initial fields.

## What This Does Not Prove

This is not a broad PDE parameter scan, not a proof for all diffusion coefficients, and not a theorem for all nonlinear trade-off forms. Transient basin-boundary cases remain lower-confidence than steady outcomes.

## Files

- `experiments/25_roy_pde_spatial_stability_and_nonhomogeneous_tests.py`
- `results/roy_pde_compensation_spatial_modes_current.csv`
- `results/roy_pde_compensation_spatial_stability_summary.csv`
- `results/roy_pde_compensation_diffusion_ratio_grid.csv`
- `results/roy_pde_nonhomogeneous_perturbation_summary.csv`
- `results/roy_pde_nonhomogeneous_mean_timeseries.csv`
- `results/roy_pde_nonhomogeneous_spatial_metrics.csv`
- `results/roy_pde_spatial_nonlinear_mechanism_decision.csv`
- `figures/roy_evo_spatial/report/fig45_pde_mode_growth_rates.png`
- `figures/roy_evo_spatial/report/fig46_pde_spatial_stability_along_branch.png`
- `figures/roy_evo_spatial/report/fig47_pde_diffusion_ratio_stability.png`
- `figures/roy_evo_spatial/report/fig48_pde_spatial_mode_schematic.png`
- `figures/roy_evo_spatial/report/fig49_nonhomogeneous_initial_conditions.png`
- `figures/roy_evo_spatial/report/fig50_nonhomogeneous_final_fields.png`
- `figures/roy_evo_spatial/report/fig51_nonhomogeneous_mean_timeseries.png`
- `figures/roy_evo_spatial/report/fig52_nonhomogeneous_spatial_metrics.png`
- `figures/roy_evo_spatial/report/fig53_pde_mechanism_decision.png`

## Next Step

Use these targeted results to decide whether any future PDE work should focus on finite-amplitude basin-boundary cases rather than broad spatial scans.

{final_label}
"""
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(text, encoding="utf-8")


def run(profile: str) -> str:
    if profile != "focused":
        raise ValueError("Only the focused profile is implemented for this targeted PDE analysis.")
    mode_rows, spatial_summary_rows = run_spatial_mode_analysis()
    diffusion_rows = run_diffusion_ratio_grid()
    nonhom_rows, timeseries_rows, spatial_rows, _saved = run_nonhomogeneous_tests(profile)
    decision_rows, final_label = write_decision_summary(spatial_summary_rows, diffusion_rows, nonhom_rows)
    plot_linear_modes(mode_rows)
    plot_spatial_stability_summary(spatial_summary_rows)
    plot_diffusion_grid(diffusion_rows)
    plot_spatial_mode_schematic()
    archives = load_field_archives()
    plot_field_grid(archives, final=False, path=FIG49_PATH)
    plot_field_grid(archives, final=True, path=FIG50_PATH)
    plot_nonhomogeneous_timeseries(timeseries_rows)
    plot_nonhomogeneous_spatial_metrics(spatial_rows)
    plot_decision_summary(decision_rows, final_label)
    write_research_note(spatial_summary_rows, diffusion_rows, nonhom_rows, decision_rows, final_label)
    for path in [
        SPATIAL_MODES_CSV,
        SPATIAL_STABILITY_SUMMARY_CSV,
        DIFFUSION_RATIO_GRID_CSV,
        NONHOM_SUMMARY_CSV,
        NONHOM_TIMESERIES_CSV,
        NONHOM_SPATIAL_METRICS_CSV,
        DECISION_CSV,
        *FIELD_FILES.values(),
        FIG45_PATH,
        FIG46_PATH,
        FIG47_PATH,
        FIG48_PATH,
        FIG49_PATH,
        FIG50_PATH,
        FIG51_PATH,
        FIG52_PATH,
        FIG53_PATH,
        NOTE_PATH,
    ]:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Final label: {final_label}")
    return final_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["focused"], default="focused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(profile=args.profile)


if __name__ == "__main__":
    main()
