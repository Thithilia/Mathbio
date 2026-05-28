"""Threshold-focused robustness checks for Roy-style 2D predator persistence."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.roy_style_2d import (
    Roy2DConfig,
    find_ode_threshold,
    find_pde_threshold,
    is_persistent_tail,
    simulate_ode_stress,
    simulate_pde_2d,
)
from src.roy_style_model import RoyParams, continuous_turing_scan, require_positive_equilibrium


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_2d_longtime"
LONGTIME_CSV = RESULTS_DIR / "roy_2d_longtime_pattern_scan.csv"
FINE_CSV = RESULTS_DIR / "roy_2d_fine_threshold_scan.csv"
TIMESERIES_CSV = RESULTS_DIR / "roy_2d_pattern_timeseries.csv"
THRESHOLD_CSV = RESULTS_DIR / "roy_2d_threshold_comparison.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_07.md"


PATTERN_TOL = 1.0e-5
EPSILON = 1.0e-4
FIXED_THRESHOLD_TOLERANCE = 1.0e-3


CLASS_TO_INT = {
    "ODE persistent, PDE persistent": 0,
    "ODE extinct, PDE persistent": 1,
    "ODE persistent, PDE extinct": 2,
    "ODE extinct, PDE extinct": 3,
    "failed": 4,
}
CLASS_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#7f7f7f", "#d62728"]
EQ_CACHE: dict[tuple[float, ...], object] = {}
TURING_CACHE: dict[tuple[float, ...], tuple[bool, float]] = {}


SCAN_FIELDNAMES = [
    "run_id",
    "scan",
    "axis",
    "mu",
    "D_w_over_D_u",
    "stress",
    "T",
    "n_x",
    "n_y",
    "dt",
    "seed",
    "epsilon",
    "baseline_turing_unstable",
    "baseline_growth",
    "ode_persistent",
    "pde_persistent",
    "classification",
    "mean_w_ode",
    "mean_w_pde",
    "ode_tail_mean_w",
    "ode_tail_min_w",
    "ode_tail_slope_w",
    "pde_tail_mean_w",
    "pde_tail_min_w",
    "pde_tail_slope_w",
    "var_u_final",
    "var_v_final",
    "var_w_final",
    "final_pattern_strength",
    "final_pattern_measurable",
    "final_quarter_mean_pattern_strength",
    "persistent_pattern_measurable",
    "dominant_wavelength_final",
    "dominant_power_final",
    "min_z_final",
    "negative_detected",
    "z_negative_detected",
    "status",
]


TIMESERIES_FIELDNAMES = [
    "run_id",
    "scan",
    "axis",
    "mu",
    "D_w_over_D_u",
    "stress",
    "T",
    "n_x",
    "n_y",
    "dt",
    "seed",
    "time",
    "mean_w",
    "var_u",
    "var_v",
    "var_w",
    "pattern_strength",
    "dominant_wavelength",
    "dominant_power",
    "min_z",
]


THRESHOLD_FIELDNAMES = [
    "run_id",
    "stage",
    "axis",
    "varied_parameter",
    "varied_value",
    "mu",
    "D_w_over_D_u",
    "eta",
    "gamma",
    "beta1",
    "T",
    "n_x",
    "n_y",
    "dt",
    "seed",
    "epsilon",
    "baseline_turing_unstable",
    "baseline_growth",
    "ode_threshold",
    "pde_threshold",
    "delta_threshold",
    "threshold_gap_width",
    "threshold_tolerance",
    "candidate_spatial_rescue",
    "candidate_spatial_inhibition",
    "validated",
    "validation_seed_count",
    "ode_status",
    "pde_status",
    "ode_iterations",
    "pde_iterations",
    "ode_s_low",
    "ode_s_high",
    "pde_s_low",
    "pde_s_high",
    "status",
]


def baseline_params(mu: float = 0.85, D_w_ratio: float = 100.0, **updates: float) -> RoyParams:
    D_u = 0.01
    params = RoyParams(
        kappa=0.15,
        xi=0.55,
        gamma=3.73,
        rho=1.76,
        eta=0.005,
        beta1=0.5,
        delta=0.1,
        mu=mu,
        D_u=D_u,
        D_v=D_u,
        D_w=D_u * D_w_ratio,
    )
    return params.with_updates(**updates) if updates else params


def param_key(params: RoyParams) -> tuple[float, ...]:
    return (
        params.kappa,
        params.xi,
        params.gamma,
        params.rho,
        params.eta,
        params.beta1,
        params.delta,
        params.mu,
        params.D_u,
        params.D_v,
        params.D_w,
    )


def stable_config_for(params: RoyParams, config: Roy2DConfig) -> Roy2DConfig:
    dx = config.L_x / (config.n_x - 1)
    dy = config.L_y / (config.n_y - 1)
    max_diffusion = max(params.D_u, params.D_v, params.D_w)
    stable_dt = 0.22 / (max_diffusion * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if config.dt <= stable_dt:
        return config
    return replace(config, dt=0.9 * stable_dt)


def classify(ode_persistent: bool, pde_persistent: bool) -> str:
    if ode_persistent and pde_persistent:
        return "ODE persistent, PDE persistent"
    if (not ode_persistent) and pde_persistent:
        return "ODE extinct, PDE persistent"
    if ode_persistent and (not pde_persistent):
        return "ODE persistent, PDE extinct"
    return "ODE extinct, PDE extinct"


def pattern_strength(result) -> np.ndarray:
    return np.maximum.reduce([result.var_u_time, result.var_v_time, result.var_w_time])


def final_quarter_pattern_strength(result) -> float:
    strength = pattern_strength(result)
    mask = result.t >= 0.75 * float(result.t[-1])
    if not np.any(mask):
        return float(strength[-1])
    return float(np.mean(strength[mask]))


def equilibrium_for(params: RoyParams):
    key = param_key(params)
    cached = EQ_CACHE.get(key)
    if cached is not None:
        return cached
    eq = require_positive_equilibrium(params)
    EQ_CACHE[key] = eq
    return eq


def baseline_turing_status(params: RoyParams) -> tuple[bool, float]:
    key = param_key(params)
    cached = TURING_CACHE.get(key)
    if cached is not None:
        return cached
    eq = equilibrium_for(params)
    scan = continuous_turing_scan(params, eq, k_min=1.0e-4, k_max=12.0, n_k=500, tol=1.0e-8)
    result = (scan.turing_unstable, scan.max_spatial_growth)
    TURING_CACHE[key] = result
    return result


def run_point(
    run_id: str,
    scan_name: str,
    axis: str,
    params: RoyParams,
    stress: float,
    config: Roy2DConfig,
    record_timeseries: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    run_config = stable_config_for(params, config)
    try:
        eq = equilibrium_for(params)
        baseline_turing, baseline_growth = baseline_turing_status(params)
        ode = simulate_ode_stress(params, stress, run_config.T, EPSILON, y0=np.array([eq.u, eq.v, eq.w]))
        pde = simulate_pde_2d(params, run_config, stress=stress, equilibrium=eq)
        ode_persistent, ode_tail = is_persistent_tail(ode.t, ode.y[2], EPSILON)
        pde_persistent, pde_tail = is_persistent_tail(pde.t, pde.mean_w_time, EPSILON)
        ode_persistent = bool(ode.success and ode_persistent)
        label = classify(ode_persistent, pde_persistent)
        strength = pattern_strength(pde)
        final_strength = float(strength[-1])
        final_quarter_strength = final_quarter_pattern_strength(pde)
        row = {
            "run_id": run_id,
            "scan": scan_name,
            "axis": axis,
            "mu": params.mu,
            "D_w_over_D_u": params.D_w / params.D_u,
            "stress": stress,
            "T": run_config.T,
            "n_x": run_config.n_x,
            "n_y": run_config.n_y,
            "dt": run_config.dt,
            "seed": run_config.seed,
            "epsilon": EPSILON,
            "baseline_turing_unstable": baseline_turing,
            "baseline_growth": baseline_growth,
            "ode_persistent": ode_persistent,
            "pde_persistent": pde_persistent,
            "classification": label,
            "mean_w_ode": ode.final_w,
            "mean_w_pde": pde.diagnostics.mean_w,
            "ode_tail_mean_w": ode_tail["tail_mean"],
            "ode_tail_min_w": ode_tail["tail_min"],
            "ode_tail_slope_w": ode_tail["tail_slope"],
            "pde_tail_mean_w": pde_tail["tail_mean"],
            "pde_tail_min_w": pde_tail["tail_min"],
            "pde_tail_slope_w": pde_tail["tail_slope"],
            "var_u_final": pde.diagnostics.var_u,
            "var_v_final": pde.diagnostics.var_v,
            "var_w_final": pde.diagnostics.var_w,
            "final_pattern_strength": final_strength,
            "final_pattern_measurable": bool(final_strength > PATTERN_TOL),
            "final_quarter_mean_pattern_strength": final_quarter_strength,
            "persistent_pattern_measurable": bool(final_quarter_strength > PATTERN_TOL),
            "dominant_wavelength_final": pde.diagnostics.dominant_wavelength,
            "dominant_power_final": pde.diagnostics.dominant_power,
            "min_z_final": pde.diagnostics.min_z,
            "negative_detected": pde.diagnostics.negative_detected,
            "z_negative_detected": pde.diagnostics.z_negative_detected,
            "status": "ok",
        }
        series: list[dict[str, object]] = []
        if record_timeseries:
            for idx, t_value in enumerate(pde.t):
                series.append(
                    {
                        "run_id": run_id,
                        "scan": scan_name,
                        "axis": axis,
                        "mu": params.mu,
                        "D_w_over_D_u": params.D_w / params.D_u,
                        "stress": stress,
                        "T": run_config.T,
                        "n_x": run_config.n_x,
                        "n_y": run_config.n_y,
                        "dt": run_config.dt,
                        "seed": run_config.seed,
                        "time": float(t_value),
                        "mean_w": float(pde.mean_w_time[idx]),
                        "var_u": float(pde.var_u_time[idx]),
                        "var_v": float(pde.var_v_time[idx]),
                        "var_w": float(pde.var_w_time[idx]),
                        "pattern_strength": float(strength[idx]),
                        "dominant_wavelength": float(pde.dominant_wavelength_time[idx]),
                        "dominant_power": float(pde.dominant_power_time[idx]),
                        "min_z": float(pde.min_z_time[idx]),
                    }
                )
        return row, series
    except Exception as exc:
        row = {
            field: np.nan for field in SCAN_FIELDNAMES
        }
        row.update(
            {
                "run_id": run_id,
                "scan": scan_name,
                "axis": axis,
                "mu": params.mu,
                "D_w_over_D_u": params.D_w / params.D_u,
                "stress": stress,
                "T": run_config.T,
                "n_x": run_config.n_x,
                "n_y": run_config.n_y,
                "dt": run_config.dt,
                "seed": run_config.seed,
                "epsilon": EPSILON,
                "baseline_turing_unstable": False,
                "ode_persistent": False,
                "pde_persistent": False,
                "classification": "failed",
                "status": f"failed: {exc}",
            }
        )
        return row, []


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def threshold_gap(result: dict[str, object]) -> float:
    low = finite_float(result.get("s_low"))
    high = finite_float(result.get("s_high"))
    if low is None or high is None:
        return float("nan")
    return max(0.0, high - low)


def threshold_row(
    run_id: str,
    stage: str,
    axis: str,
    varied_parameter: str,
    varied_value: float,
    params: RoyParams,
    config: Roy2DConfig,
    seed_count: int,
    max_iter_ode: int,
    max_iter_pde: int,
    validated: bool,
    s_low: float = 0.0,
    s_high: float = 1.0,
) -> dict[str, object]:
    run_config = stable_config_for(params, config)
    try:
        baseline_turing, baseline_growth = baseline_turing_status(params)
        ode_threshold = find_ode_threshold(params, s_low, s_high, run_config.T, EPSILON, max_iter=max_iter_ode)
        pde_threshold = find_pde_threshold(params, run_config, s_low, s_high, EPSILON, max_iter=max_iter_pde)
        ode_value = finite_float(ode_threshold["threshold"])
        pde_value = finite_float(pde_threshold["threshold"])
        ode_gap = threshold_gap(ode_threshold)
        pde_gap = threshold_gap(pde_threshold)
        gap_width = float(np.nanmax([ode_gap, pde_gap]))
        tolerance = max(FIXED_THRESHOLD_TOLERANCE, gap_width)
        if ode_threshold["status"] == "ok" and pde_threshold["status"] == "ok" and ode_value is not None and pde_value is not None:
            delta = pde_value - ode_value
            rescue = bool(delta > tolerance)
            inhibition = bool(delta < -tolerance)
            status = "ok"
        else:
            delta = float("nan")
            rescue = False
            inhibition = False
            status = f"threshold_failed: ode={ode_threshold['status']}; pde={pde_threshold['status']}"
        return {
            "run_id": run_id,
            "stage": stage,
            "axis": axis,
            "varied_parameter": varied_parameter,
            "varied_value": varied_value,
            "mu": params.mu,
            "D_w_over_D_u": params.D_w / params.D_u,
            "eta": params.eta,
            "gamma": params.gamma,
            "beta1": params.beta1,
            "T": run_config.T,
            "n_x": run_config.n_x,
            "n_y": run_config.n_y,
            "dt": run_config.dt,
            "seed": run_config.seed,
            "epsilon": EPSILON,
            "baseline_turing_unstable": baseline_turing,
            "baseline_growth": baseline_growth,
            "ode_threshold": ode_value if ode_value is not None else np.nan,
            "pde_threshold": pde_value if pde_value is not None else np.nan,
            "delta_threshold": delta,
            "threshold_gap_width": gap_width,
            "threshold_tolerance": tolerance,
            "candidate_spatial_rescue": rescue,
            "candidate_spatial_inhibition": inhibition,
            "validated": validated,
            "validation_seed_count": seed_count,
            "ode_status": ode_threshold["status"],
            "pde_status": pde_threshold["status"],
            "ode_iterations": ode_threshold["iterations"],
            "pde_iterations": pde_threshold["iterations"],
            "ode_s_low": ode_threshold["s_low"],
            "ode_s_high": ode_threshold["s_high"],
            "pde_s_low": pde_threshold["s_low"],
            "pde_s_high": pde_threshold["s_high"],
            "status": status,
        }
    except Exception as exc:
        row = {field: np.nan for field in THRESHOLD_FIELDNAMES}
        row.update(
            {
                "run_id": run_id,
                "stage": stage,
                "axis": axis,
                "varied_parameter": varied_parameter,
                "varied_value": varied_value,
                "mu": params.mu,
                "D_w_over_D_u": params.D_w / params.D_u,
                "eta": params.eta,
                "gamma": params.gamma,
                "beta1": params.beta1,
                "T": config.T,
                "n_x": config.n_x,
                "n_y": config.n_y,
                "dt": config.dt,
                "seed": config.seed,
                "epsilon": EPSILON,
                "candidate_spatial_rescue": False,
                "candidate_spatial_inhibition": False,
                "validated": validated,
                "validation_seed_count": seed_count,
                "status": f"failed: {exc}",
            }
        )
        return row


def stage_a_regimes() -> list[tuple[str, str, float, RoyParams]]:
    regimes: list[tuple[str, str, float, RoyParams]] = [
        ("baseline", "none", 0.0, baseline_params(mu=0.85, D_w_ratio=100.0)),
    ]
    for mu in [0.60, 0.72, 0.80, 0.85, 0.89, 0.95]:
        regimes.append(("mu", "mu", mu, baseline_params(mu=mu, D_w_ratio=100.0)))
    for ratio in [40.0, 70.0, 100.0, 150.0, 250.0, 400.0]:
        regimes.append(("diffusion_ratio", "D_w_over_D_u", ratio, baseline_params(mu=0.85, D_w_ratio=ratio)))
    for eta in [0.0025, 0.0075]:
        regimes.append(("eta_exploratory", "eta", eta, baseline_params(mu=0.85, D_w_ratio=100.0, eta=eta)))

    unique: list[tuple[str, str, float, RoyParams]] = []
    seen: set[tuple[float, ...]] = set()
    for item in regimes:
        key = param_key(item[3])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def stage_b_regimes(stage_a_rows: list[dict[str, object]]) -> list[tuple[str, str, float, RoyParams]]:
    ok_rows = [row for row in stage_a_rows if row["status"] == "ok"]
    positive = [row for row in ok_rows if truthy(row["candidate_spatial_rescue"])]
    selected = positive[:3]
    if not selected:
        baseline = [row for row in ok_rows if row["axis"] == "baseline"]
        selected.extend(baseline[:1])
        finite_delta = [
            row
            for row in ok_rows
            if finite_float(row["delta_threshold"]) is not None and row not in selected
        ]
        if finite_delta:
            selected.append(min(finite_delta, key=lambda row: abs(float(row["delta_threshold"]))))

    regimes: list[tuple[str, str, float, RoyParams]] = []
    for row in selected:
        axis = str(row["axis"])
        varied_parameter = str(row["varied_parameter"])
        varied_value = float(row["varied_value"])
        regimes.append(
            (
                axis,
                varied_parameter,
                varied_value,
                baseline_params(
                    mu=float(row["mu"]),
                    D_w_ratio=float(row["D_w_over_D_u"]),
                    eta=float(row["eta"]),
                    gamma=float(row["gamma"]),
                    beta1=float(row["beta1"]),
                ),
            )
        )
    return regimes


def plot_threshold_delta(rows: list[dict[str, object]], path: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return
    labels = [str(row["run_id"]) for row in ok_rows]
    deltas = [float(row["delta_threshold"]) for row in ok_rows]
    colors = ["#2ca02c" if value > float(row["threshold_tolerance"]) else "#d62728" if value < -float(row["threshold_tolerance"]) else "#7f7f7f" for value, row in zip(deltas, ok_rows)]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(np.arange(len(deltas)), deltas, color=colors)
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Delta m_c = m_c_PDE - m_c_ODE")
    ax.set_title("Roy-style 2D predator-mortality threshold comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_longtime(long_rows: list[dict[str, object]], path: Path) -> None:
    subset = [row for row in long_rows if row["status"] == "ok"]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for stress in sorted({float(row["stress"]) for row in subset}):
        rows = sorted([row for row in subset if np.isclose(float(row["stress"]), stress)], key=lambda item: float(item["T"]))
        ax.plot(
            [float(row["T"]) for row in rows],
            [float(row["final_quarter_mean_pattern_strength"]) for row in rows],
            marker="o",
            label=f"s={stress:g}",
        )
    ax.axhline(PATTERN_TOL, color="0.35", ls="--", lw=1)
    ax.set_xlabel("final time T")
    ax.set_ylabel("final-quarter mean pattern strength")
    ax.set_title("Exploratory pattern persistence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase(rows: list[dict[str, object]], axis: str, path: Path) -> None:
    subset = [row for row in rows if row["axis"] == axis and row["status"] == "ok"]
    if not subset:
        return
    stresses = sorted({float(row["stress"]) for row in subset})
    if axis == "mu":
        x_values = sorted({float(row["mu"]) for row in subset})
        xlabel = "mu"
        title = "Tail persistence over stress and mu"
    else:
        x_values = sorted({float(row["D_w_over_D_u"]) for row in subset})
        xlabel = "D_w / D_u"
        title = "Tail persistence over stress and diffusion ratio"
    matrix = np.full((len(stresses), len(x_values)), CLASS_TO_INT["failed"], dtype=float)
    for row in subset:
        i = stresses.index(float(row["stress"]))
        x_key = float(row["mu"]) if axis == "mu" else float(row["D_w_over_D_u"])
        j = x_values.index(x_key)
        matrix[i, j] = CLASS_TO_INT[str(row["classification"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=4, cmap=ListedColormap(CLASS_COLORS))
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:g}" for value in x_values], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(stresses)))
    ax.set_yticklabels([f"{value:g}" for value in stresses])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("stress s")
    ax.set_title(title)
    handles = [Patch(facecolor=CLASS_COLORS[value], label=label) for label, value in CLASS_TO_INT.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(
    threshold_rows: list[dict[str, object]],
    long_rows: list[dict[str, object]],
    fine_rows: list[dict[str, object]],
    path: Path,
    figure_paths: dict[str, Path],
) -> None:
    ok_thresholds = [row for row in threshold_rows if row["status"] == "ok"]
    stage_a = [row for row in ok_thresholds if row["stage"] == "A_quick_threshold"]
    stage_b = [row for row in ok_thresholds if row["stage"] == "B_validation"]
    baseline_rows = [
        row
        for row in ok_thresholds
        if row["axis"] == "baseline" and np.isclose(float(row["mu"]), 0.85) and np.isclose(float(row["D_w_over_D_u"]), 100.0)
    ]
    baseline_stage_a = next((row for row in baseline_rows if row["stage"] == "A_quick_threshold"), None)
    baseline_validation = [row for row in baseline_rows if row["stage"] == "B_validation"]
    baseline_validation_row = baseline_validation[0] if baseline_validation else None
    rescue = [row for row in ok_thresholds if truthy(row["candidate_spatial_rescue"])]
    stage_a_rescue = [row for row in stage_a if truthy(row["candidate_spatial_rescue"])]
    stage_b_rescue = [row for row in stage_b if truthy(row["candidate_spatial_rescue"])]
    inhibition = [row for row in ok_thresholds if truthy(row["candidate_spatial_inhibition"])]
    closest = None
    if stage_a:
        closest = min(stage_a, key=lambda row: abs(float(row["delta_threshold"])))

    if stage_b_rescue:
        conclusion = "spatial rescue is supported only for the validated positive-threshold rows listed below."
    elif inhibition:
        conclusion = "the tested regimes lean toward spatial inhibition where threshold differences exceed tolerance."
    else:
        conclusion = "no measurable spatial-rescue effect was found; threshold differences are within tolerance or negative."

    def threshold_line(row: dict[str, object] | None) -> list[str]:
        if row is None:
            return ["- unavailable"]
        return [
            f"- `m_c^ODE = {float(row['ode_threshold']):.6g}`",
            f"- `m_c^PDE = {float(row['pde_threshold']):.6g}`",
            f"- `Delta m_c = {float(row['delta_threshold']):.6g}`",
            f"- tolerance used for this row: `{float(row['threshold_tolerance']):.6g}`",
            f"- row status: `{row['status']}`",
        ]

    validation_summary = []
    for row in stage_b:
        validation_summary.append(
            f"- `{row['run_id']}`: axis `{row['axis']}`, seed `{row['seed']}`, "
            f"T `{float(row['T']):g}`, grid `{int(row['n_x'])}x{int(row['n_y'])}`, "
            f"Delta `{float(row['delta_threshold']):.6g}`, status `{row['status']}`"
        )
    if not validation_summary:
        validation_summary = ["- no validation rows completed"]

    if stage_a_rescue:
        validation_intro = (
            f"Stage A produced `{len(stage_a_rescue)}` positive `Delta m_c` candidate rows at the quick "
            "`T=70`, `36x36` setting. Stage B therefore validated the baseline and the first positive "
            "candidate regimes using longer `T=200`, `64x64` grids, and three perturbation seeds."
        )
    else:
        validation_intro = (
            "Stage A produced no positive `Delta m_c` candidates. Stage B therefore validated the baseline "
            "and the closest-to-zero Stage A regime using longer `T=200`, `64x64` grids, and three perturbation seeds."
        )

    closest_lines = threshold_line(closest)
    if closest is not None:
        closest_lines.insert(
            0,
            f"- closest Stage A regime: `{closest['axis']}` with `{closest['varied_parameter']}={closest['varied_value']}`",
        )

    lines = [
        "# Nonlinear PDE Results 07",
        "",
        "This revision makes the predator-mortality threshold comparison the primary scientific output. Pattern morphology remains exploratory and is not used as the rescue criterion.",
        "",
        "Persistence is classified from the final 25% of the predator-density time series. A trajectory is persistent when the tail mean is above `epsilon`, the tail minimum stays above `0.25 epsilon`, and the least-squares tail slope is not strongly negative:",
        "",
        "`tail_slope >= -max(epsilon, 0.25 tail_mean) / tail_duration`.",
        "",
        "## Core Threshold Question",
        "",
        "The decisive quantity is",
        "",
        "`Delta m_c = m_c_PDE - m_c_ODE`.",
        "",
        "Positive values indicate that spatial structure expands the predator-persistence range under mortality stress; negative values indicate that spatial structure shrinks it; values within tolerance are treated as no measurable threshold effect.",
        "",
        "## Baseline Regime",
        "",
        "Baseline parameters use `mu=0.85` and `D_w/D_u=100`.",
        "",
        "Quick Stage A estimate:",
        "",
        *threshold_line(baseline_stage_a),
        "",
        "Validated `T=200`, `64x64` estimate from the first baseline seed:",
        "",
        *threshold_line(baseline_validation_row),
        "",
        "## Stage A: Quick Threshold Finder",
        "",
        f"- completed threshold rows: `{len(stage_a)}`",
        f"- candidate spatial-rescue rows: `{len(stage_a_rescue)}`",
        f"- candidate spatial-inhibition rows: `{len([row for row in stage_a if truthy(row['candidate_spatial_inhibition'])])}`",
        "- scanned axes: `mu`, `D_w/D_u`, and a small exploratory `eta` axis.",
        "- Stage A is a candidate finder only; its positive rows are not interpreted as rescue without Stage B validation.",
        "",
        "## Closest Regime To A Rescue Candidate",
        "",
        *closest_lines,
        "",
        "## Stage B: Validation",
        "",
        validation_intro,
        "",
        f"- validated positive-threshold rows after Stage B: `{len(stage_b_rescue)}`",
        "",
        *validation_summary,
        "",
        "## Interpretation",
        "",
        conclusion,
        "",
        "Spatial rescue is not claimed unless a positive threshold difference survives validation. In this run, the conclusion is framed entirely around the sign and robustness of `Delta m_c`, not around the mere existence of spatial patterning.",
        "",
        "## Secondary Pattern Diagnostics",
        "",
        f"- representative long-time rows: `{len([row for row in long_rows if row['status'] == 'ok'])}`",
        f"- representative fine-stress rows: `{len([row for row in fine_rows if row['status'] == 'ok'])}`",
        "- dominant wavelength and Fourier power remain exploratory diagnostics only.",
        "",
        "Outputs:",
        "",
        f"- `{THRESHOLD_CSV.relative_to(ROOT)}`",
        f"- `{LONGTIME_CSV.relative_to(ROOT)}`",
        f"- `{FINE_CSV.relative_to(ROOT)}`",
        f"- `{TIMESERIES_CSV.relative_to(ROOT)}`",
        f"- `{figure_paths['threshold'].relative_to(ROOT)}`",
        f"- `{figure_paths['longtime'].relative_to(ROOT)}`",
        f"- `{figure_paths['mu'].relative_to(ROOT)}`",
        f"- `{figure_paths['ratio'].relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def representative_scans(threshold_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    long_rows: list[dict[str, object]] = []
    fine_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []
    baseline = baseline_params(mu=0.85, D_w_ratio=100.0)
    baseline_threshold = next(
        (
            row
            for row in threshold_rows
            if row["stage"] == "A_quick_threshold" and row["axis"] == "baseline" and row["status"] == "ok"
        ),
        None,
    )
    stress_near = 0.5
    if baseline_threshold is not None and finite_float(baseline_threshold["ode_threshold"]) is not None:
        stress_near = float(baseline_threshold["ode_threshold"])

    run_counter = 0
    for T in [35.0, 70.0, 120.0, 200.0]:
        for stress in [0.0, stress_near]:
            run_counter += 1
            config = Roy2DConfig(
                n_x=64,
                n_y=64,
                L_x=20.0,
                L_y=20.0,
                T=T,
                dt=0.01,
                record_every=100,
                seed=20260630,
                record_fourier=stress == stress_near,
            )
            row, series = run_point(
                f"long_{run_counter:03d}",
                "longtime",
                "baseline_threshold",
                baseline,
                stress,
                config,
                record_timeseries=T in {70.0, 200.0},
            )
            long_rows.append(row)
            timeseries_rows.extend(series)
            print(f"representative long {row['run_id']}: T={T}, s={stress:.6g}, class={row['classification']}")

    stress_grid = [0.45, 0.50, 0.525, 0.55, 0.575, 0.60]
    fine_config = Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=70.0, dt=0.025, record_every=80, seed=20260631)
    for stress in stress_grid:
        run_counter += 1
        row, _ = run_point(
            f"fine_{run_counter:03d}",
            "fine_threshold",
            "baseline_threshold",
            baseline,
            stress,
            fine_config,
            False,
        )
        fine_rows.append(row)
        print(f"representative fine {row['run_id']}: s={stress:.6g}, class={row['classification']}")

    for mu in [0.72, 0.85, 0.95]:
        for stress in [0.50, 0.55, 0.60]:
            run_counter += 1
            params = baseline_params(mu=mu, D_w_ratio=100.0)
            row, _ = run_point(f"fine_{run_counter:03d}", "fine_threshold", "mu", params, stress, fine_config, False)
            fine_rows.append(row)
    for ratio in [70.0, 100.0, 250.0]:
        for stress in [0.50, 0.55, 0.60]:
            run_counter += 1
            params = baseline_params(mu=0.85, D_w_ratio=ratio)
            row, _ = run_point(f"fine_{run_counter:03d}", "fine_threshold", "diffusion_ratio", params, stress, fine_config, False)
            fine_rows.append(row)
    return long_rows, fine_rows, timeseries_rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    threshold_rows: list[dict[str, object]] = []
    stage_a_config = Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=70.0, dt=0.025, record_every=80, seed=20260620)
    for idx, (axis, varied_parameter, varied_value, params) in enumerate(stage_a_regimes(), start=1):
        row = threshold_row(
            f"A_{idx:03d}",
            "A_quick_threshold",
            axis,
            varied_parameter,
            varied_value,
            params,
            stage_a_config,
            seed_count=1,
            max_iter_ode=12,
            max_iter_pde=10,
            validated=False,
        )
        threshold_rows.append(row)
        print(
            f"Stage A {row['run_id']}: axis={axis}, Delta={row['delta_threshold']}, "
            f"status={row['status']}"
        )

    validation_regimes = stage_b_regimes(threshold_rows)
    validation_config = Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=200.0, dt=0.01, record_every=200)
    validation_counter = 0
    for axis, varied_parameter, varied_value, params in validation_regimes:
        for seed in [20260621, 20260622, 20260623]:
            validation_counter += 1
            row = threshold_row(
                f"B_{validation_counter:03d}",
                "B_validation",
                axis,
                varied_parameter,
                varied_value,
                params,
                replace(validation_config, seed=seed),
                seed_count=3,
                max_iter_ode=12,
                max_iter_pde=8,
                validated=True,
            )
            threshold_rows.append(row)
            print(
                f"Stage B {row['run_id']}: axis={axis}, seed={seed}, "
                f"Delta={row['delta_threshold']}, status={row['status']}"
            )

    long_rows, fine_rows, timeseries_rows = representative_scans(threshold_rows)

    write_csv(threshold_rows, THRESHOLD_CSV, THRESHOLD_FIELDNAMES)
    write_csv(long_rows, LONGTIME_CSV, SCAN_FIELDNAMES)
    write_csv(fine_rows, FINE_CSV, SCAN_FIELDNAMES)
    write_csv(timeseries_rows, TIMESERIES_CSV, TIMESERIES_FIELDNAMES)

    figure_paths = {
        "threshold": FIG_DIR / "07_threshold_delta.png",
        "longtime": FIG_DIR / "07_longtime_pattern_strength.png",
        "mu": FIG_DIR / "07_fine_phase_stress_mu.png",
        "ratio": FIG_DIR / "07_fine_phase_stress_diffusion_ratio.png",
    }
    plot_threshold_delta(threshold_rows, figure_paths["threshold"])
    plot_longtime(long_rows, figure_paths["longtime"])
    plot_phase(fine_rows, "mu", figure_paths["mu"])
    plot_phase(fine_rows, "diffusion_ratio", figure_paths["ratio"])
    summarize(threshold_rows, long_rows, fine_rows, SUMMARY_MD, figure_paths)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
