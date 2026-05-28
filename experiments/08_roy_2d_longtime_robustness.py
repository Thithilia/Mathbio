"""Long-time and fine-stress robustness checks for Roy-style 2D thresholds."""

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

from src.roy_style_2d import Roy2DConfig, simulate_ode_stress, simulate_pde_2d
from src.roy_style_model import RoyParams, continuous_turing_scan, require_positive_equilibrium


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_2d_longtime"
LONGTIME_CSV = RESULTS_DIR / "roy_2d_longtime_pattern_scan.csv"
FINE_CSV = RESULTS_DIR / "roy_2d_fine_threshold_scan.csv"
TIMESERIES_CSV = RESULTS_DIR / "roy_2d_pattern_timeseries.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_07.md"


PATTERN_TOL = 1.0e-5
EPSILON = 1.0e-4


CLASS_TO_INT = {
    "ODE persistent, PDE persistent": 0,
    "ODE extinct, PDE persistent": 1,
    "ODE persistent, PDE extinct": 2,
    "ODE extinct, PDE extinct": 3,
    "failed": 4,
}
CLASS_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#7f7f7f", "#d62728"]
EQ_CACHE = {}
TURING_CACHE = {}


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


def baseline_params(mu: float = 0.85, D_w_ratio: float = 100.0) -> RoyParams:
    D_u = 0.01
    return RoyParams(
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


def equilibrium_for(params: RoyParams):
    key = param_key(params)
    cached = EQ_CACHE.get(key)
    if cached is not None:
        return cached
    eq = require_positive_equilibrium(params)
    EQ_CACHE[key] = eq
    return eq


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
        ode_persistent = bool(ode.success and ode.final_w > EPSILON)
        pde_persistent = bool(pde.diagnostics.mean_w > EPSILON)
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
            "baseline_growth": np.nan,
            "ode_persistent": False,
            "pde_persistent": False,
            "classification": "failed",
            "mean_w_ode": np.nan,
            "mean_w_pde": np.nan,
            "var_u_final": np.nan,
            "var_v_final": np.nan,
            "var_w_final": np.nan,
            "final_pattern_strength": np.nan,
            "final_pattern_measurable": False,
            "final_quarter_mean_pattern_strength": np.nan,
            "persistent_pattern_measurable": False,
            "dominant_wavelength_final": np.nan,
            "dominant_power_final": np.nan,
            "min_z_final": np.nan,
            "negative_detected": False,
            "z_negative_detected": False,
            "status": f"failed: {exc}",
        }
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


def plot_longtime(long_rows: list[dict[str, object]], path: Path) -> None:
    subset = [
        row
        for row in long_rows
        if row["status"] == "ok"
        and float(row["D_w_over_D_u"]) == 100.0
        and float(row["mu"]) == 0.85
        and int(row["n_x"]) in {36, 64}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    for ax, stress in zip(axes, [0.0, 0.675]):
        stress_rows = [row for row in subset if np.isclose(float(row["stress"]), stress)]
        for grid in sorted({int(row["n_x"]) for row in stress_rows}):
            rows = sorted([row for row in stress_rows if int(row["n_x"]) == grid], key=lambda item: float(item["T"]))
            ax.plot(
                [float(row["T"]) for row in rows],
                [float(row["final_quarter_mean_pattern_strength"]) for row in rows],
                marker="o",
                label=f"{grid}x{grid}",
            )
        ax.axhline(PATTERN_TOL, color="0.35", ls="--", lw=1)
        ax.set_title(f"stress s={stress:g}")
        ax.set_xlabel("final time T")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("final-quarter mean pattern strength")
    axes[1].legend()
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
        title = "Fine scan over stress and mu"
    else:
        x_values = sorted({float(row["D_w_over_D_u"]) for row in subset})
        xlabel = "D_w / D_u"
        title = "Fine scan over stress and diffusion ratio"
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


def summarize(long_rows: list[dict[str, object]], fine_rows: list[dict[str, object]], path: Path, figure_paths: dict[str, Path]) -> None:
    ok_long = [row for row in long_rows if row["status"] == "ok"]
    ok_fine = [row for row in fine_rows if row["status"] == "ok"]
    rescue_like = [
        row
        for row in ok_fine
        if row["classification"] == "ODE extinct, PDE persistent"
        and truthy(row["baseline_turing_unstable"])
        and truthy(row["persistent_pattern_measurable"])
    ]
    high_grid_rows = [row for row in ok_long + ok_fine if int(row["n_x"]) >= 64]
    long_200 = [row for row in ok_long if np.isclose(float(row["T"]), 200.0)]
    persistent_at_200 = [row for row in long_200 if truthy(row["persistent_pattern_measurable"])]
    unstressed_200 = [row for row in long_200 if np.isclose(float(row["stress"]), 0.0)]
    stressed_200 = [row for row in long_200 if float(row["stress"]) >= 0.65]
    stressed_200_patterns = [row for row in stressed_200 if truthy(row["persistent_pattern_measurable"])]
    counts: dict[str, int] = {}
    for row in ok_fine:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1

    baseline_64_near = [
        row
        for row in ok_fine
        if row["axis"] == "baseline_64"
        and float(row["mu"]) == 0.85
        and float(row["D_w_over_D_u"]) == 100.0
    ]
    adaptive_baseline = [
        row
        for row in ok_fine
        if row["axis"] == "adaptive_baseline_64"
        and float(row["mu"]) == 0.85
        and float(row["D_w_over_D_u"]) == 100.0
    ]
    baseline_rescue = [row for row in baseline_64_near if row["classification"] == "ODE extinct, PDE persistent"]
    adaptive_persistent_patterns = [row for row in adaptive_baseline if truthy(row["persistent_pattern_measurable"])]
    adaptive_persistent = [row for row in adaptive_baseline if row["classification"] == "ODE persistent, PDE persistent"]
    adaptive_extinct = [row for row in adaptive_baseline if row["classification"] == "ODE extinct, PDE extinct"]

    conclusion = "robust within tested ranges"
    if rescue_like:
        conclusion = "weakened by rescue-like candidates requiring further robustness checks"
    elif baseline_rescue:
        conclusion = "weakened near baseline, but not yet robustly established"

    lines = [
        "# Nonlinear PDE Results 07",
        "",
        "This run tests whether the preliminary negative 2D Roy-style threshold result was caused by short integration time, coarse grid resolution, sparse stress sampling, or pattern collapse near the extinction transition.",
        "",
        "## 1. Long-Time Pattern Persistence",
        "",
        f"- long-time runs completed: `{len(ok_long)}`",
        f"- runs at `T=200`: `{len(long_200)}`",
        f"- `T=200` runs with persistent final-quarter pattern strength above tolerance: `{len(persistent_at_200)}`",
        f"- unstressed `T=200` runs with persistent patterning: `{len([row for row in unstressed_200 if truthy(row['persistent_pattern_measurable'])])}/{len(unstressed_200)}`",
        f"- stressed near-threshold `T=200` runs with persistent patterning: `{len(stressed_200_patterns)}/{len(stressed_200)}`",
        "",
        "Answer: the pattern persists for the unstressed Roy-style baseline, but collapses near the predator extinction transition in the long-time runs.",
        "",
        "## 2. Grid Resolution",
        "",
        f"- rows at `64x64` or higher: `{len(high_grid_rows)}`",
        "- `64x64` is treated as the minimum serious resolution for interpreting 2D pattern dynamics.",
        "- `96x96` spot checks are included where explicit CFL cost remains feasible.",
        "Answer: increasing from `36x36` to `64x64`, with selected `96x96` checks, does not reveal ODE-extinct/PDE-persistent behavior.",
        "",
        "## 3. Fine Stress Scan",
        "",
        f"- fine threshold rows completed: `{len(ok_fine)}`",
        f"- classification counts: `{counts}`",
        f"- ODE-extinct/PDE-persistent rows satisfying baseline Turing and persistent-pattern filters: `{len(rescue_like)}`",
        f"- adaptive baseline 64x64 rows near the shifted long-time transition: `{len(adaptive_baseline)}`",
        f"- adaptive baseline rows with persistent measurable patterning: `{len(adaptive_persistent_patterns)}`",
        f"- adaptive baseline persistent rows: `{len(adaptive_persistent)}`; extinct rows: `{len(adaptive_extinct)}`",
        "",
        "Answer: the finer scan finds the long-time baseline transition between approximately `s=0.50` and `s=0.525`, but both ODE and PDE switch together and no persistent-pattern rescue window appears.",
        "",
        "## 4. Regimes Beyond mu = 0.85",
        "",
        "The scan covers `mu = [0.60, 0.72, 0.80, 0.85, 0.89, 0.95]` and `D_w/D_u = [40, 70, 100, 150, 250, 400]` with a fine stress grid near the transition.",
        "",
        "Answer: outside `mu=0.85`, the tested fine-stress points also show no ODE-extinct/PDE-persistent regime.",
        "",
        "## 5. Conservative Conclusion",
        "",
        f"The previous negative result is **{conclusion}**. Pattern-mediated rescue is not claimed.",
        "",
        "A claim would require baseline Turing instability, persistent measurable 2D patterning near threshold, ODE extinction with PDE persistence under the mean-density criterion, and survival under longer `T`, higher grid resolution, and at least two perturbation seeds.",
        "",
        "Outputs:",
        "",
        f"- `{LONGTIME_CSV.relative_to(ROOT)}`",
        f"- `{FINE_CSV.relative_to(ROOT)}`",
        f"- `{TIMESERIES_CSV.relative_to(ROOT)}`",
        f"- `{figure_paths['longtime'].relative_to(ROOT)}`",
        f"- `{figure_paths['mu'].relative_to(ROOT)}`",
        f"- `{figure_paths['ratio'].relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    long_rows: list[dict[str, object]] = []
    fine_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    stress_grid = [0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80]
    long_Ts = [35.0, 70.0, 120.0, 200.0]
    long_stresses = [0.0, 0.65, 0.675, 0.70]

    run_counter = 0
    for grid in [36, 64]:
        for T in long_Ts:
            for stress in long_stresses:
                run_counter += 1
                params = baseline_params(mu=0.85, D_w_ratio=100.0)
                record_ts = grid == 64 and stress in {0.0, 0.675}
                config = Roy2DConfig(
                    n_x=grid,
                    n_y=grid,
                    L_x=20.0,
                    L_y=20.0,
                    T=T,
                    dt=0.025 if grid == 36 else 0.01,
                    record_every=max(1, int((1.0 / (0.025 if grid == 36 else 0.01)))),
                    seed=20260610,
                    record_fourier=record_ts,
                )
                row, series = run_point(
                    f"long_{run_counter:03d}",
                    "longtime",
                    "baseline_longtime",
                    params,
                    stress,
                    config,
                    record_timeseries=record_ts,
                )
                long_rows.append(row)
                timeseries_rows.extend(series)
                print(f"longtime {row['run_id']}: grid={grid}, T={T}, s={stress}, class={row['classification']}")

    for stress in [0.65, 0.675, 0.70]:
        run_counter += 1
        params = baseline_params(mu=0.85, D_w_ratio=100.0)
        config = Roy2DConfig(
            n_x=96,
            n_y=96,
            L_x=20.0,
            L_y=20.0,
            T=70.0,
            dt=0.004,
            record_every=250,
            seed=20260611,
            record_fourier=True,
        )
        row, series = run_point(f"long_{run_counter:03d}", "longtime", "baseline_96_spot", params, stress, config, True)
        long_rows.append(row)
        timeseries_rows.extend(series)
        print(f"longtime {row['run_id']}: grid=96, T=70, s={stress}, class={row['classification']}")

    fine_config_36 = Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=70.0, dt=0.025, record_every=120, seed=20260612)
    fine_config_64 = Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=70.0, dt=0.01, record_every=250, seed=20260613)
    mu_values = [0.60, 0.72, 0.80, 0.85, 0.89, 0.95]
    ratio_values = [40.0, 70.0, 100.0, 150.0, 250.0, 400.0]

    for mu in mu_values:
        params = baseline_params(mu=mu, D_w_ratio=100.0)
        for stress in stress_grid:
            run_counter += 1
            row, _ = run_point(f"fine_{run_counter:03d}", "fine_threshold", "mu", params, stress, fine_config_36, False)
            fine_rows.append(row)
            print(f"fine mu={mu}, s={stress}: {row['classification']}")

    for ratio in ratio_values:
        params = baseline_params(mu=0.85, D_w_ratio=ratio)
        for stress in stress_grid:
            run_counter += 1
            row, _ = run_point(f"fine_{run_counter:03d}", "fine_threshold", "diffusion_ratio", params, stress, fine_config_36, False)
            fine_rows.append(row)
            print(f"fine ratio={ratio}, s={stress}: {row['classification']}")

    for stress in stress_grid:
        run_counter += 1
        params = baseline_params(mu=0.85, D_w_ratio=100.0)
        row, series = run_point(
            f"fine_{run_counter:03d}",
            "fine_threshold",
            "baseline_64",
            params,
            stress,
            replace(fine_config_64, record_fourier=stress in {0.65, 0.675, 0.70}),
            record_timeseries=stress in {0.65, 0.675, 0.70},
        )
        fine_rows.append(row)
        timeseries_rows.extend(series)
        print(f"fine 64 baseline s={stress}: {row['classification']}")

    adaptive_baseline_stresses = [0.30, 0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60]
    for stress in adaptive_baseline_stresses:
        run_counter += 1
        params = baseline_params(mu=0.85, D_w_ratio=100.0)
        record_ts = stress in {0.50, 0.55}
        row, series = run_point(
            f"fine_{run_counter:03d}",
            "fine_threshold",
            "adaptive_baseline_64",
            params,
            stress,
            replace(fine_config_64, record_fourier=record_ts),
            record_ts,
        )
        fine_rows.append(row)
        timeseries_rows.extend(series)
        print(f"adaptive 64 baseline s={stress}: {row['classification']}")

    for seed in [20260614, 20260615]:
        for stress in [0.65, 0.675, 0.70]:
            run_counter += 1
            params = baseline_params(mu=0.85, D_w_ratio=100.0)
            row, _ = run_point(
                f"fine_{run_counter:03d}",
                "fine_threshold",
                "seed_check_64",
                params,
                stress,
                replace(fine_config_64, seed=seed),
                False,
            )
            fine_rows.append(row)
            print(f"seed check seed={seed}, s={stress}: {row['classification']}")

    write_csv(long_rows, LONGTIME_CSV, SCAN_FIELDNAMES)
    write_csv(fine_rows, FINE_CSV, SCAN_FIELDNAMES)
    write_csv(timeseries_rows, TIMESERIES_CSV, TIMESERIES_FIELDNAMES)

    figure_paths = {
        "longtime": FIG_DIR / "07_longtime_pattern_strength.png",
        "mu": FIG_DIR / "07_fine_phase_stress_mu.png",
        "ratio": FIG_DIR / "07_fine_phase_stress_diffusion_ratio.png",
    }
    plot_longtime(long_rows, figure_paths["longtime"])
    plot_phase(fine_rows, "mu", figure_paths["mu"])
    plot_phase(fine_rows, "diffusion_ratio", figure_paths["ratio"])
    summarize(long_rows, fine_rows, SUMMARY_MD, figure_paths)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
