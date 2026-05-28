"""Reproduce a 2D Roy-style Turing pattern for the validated parameter line."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.roy_style_2d import Roy2DConfig, dominant_fourier_mode, simulate_pde_2d
from src.roy_style_model import RoyParams, continuous_turing_scan, neumann_turing_scan, require_positive_equilibrium


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_2d"
OUTPUT_CSV = RESULTS_DIR / "roy_2d_pattern_reproduction.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_05.md"


FIELDNAMES = [
    "kappa",
    "eta",
    "mu",
    "delta",
    "D_u",
    "D_v",
    "D_w",
    "n_x",
    "n_y",
    "L_x",
    "L_y",
    "T",
    "dt",
    "seed",
    "u_star",
    "v_star",
    "w_star",
    "z_star",
    "ode_stable",
    "continuous_max_spatial_growth",
    "continuous_best_k",
    "discrete_max_spatial_growth",
    "dominant_discrete_mode",
    "mean_w_final",
    "var_u_final",
    "var_v_final",
    "var_w_final",
    "dominant_k",
    "dominant_wavelength",
    "dominant_power",
    "min_value",
    "min_z",
    "negative_detected",
    "z_negative_detected",
    "pattern_measurable",
]


def baseline_params() -> RoyParams:
    return RoyParams(kappa=0.15, xi=0.55, gamma=3.73, rho=1.76, eta=0.005, beta1=0.5, delta=0.1, mu=0.85)


def plot_maps(result, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    fields = [(result.u, "undefended prey u"), (result.v, "defended prey v"), (result.w, "predator w")]
    for ax, (field, title) in zip(axes, fields):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[result.x[0], result.x[-1], result.y[0], result.y[-1]],
            cmap="viridis",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mean_w(result, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(result.t, result.mean_w_time, lw=1.9)
    ax.set_xlabel("time")
    ax.set_ylabel("mean predator density")
    ax.set_title("Roy-style 2D mean w(t)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_power_spectrum(result, path: Path) -> None:
    dominant_k, _, _, radial_k, spectrum = dominant_fourier_mode(result.u, result.x[-1] - result.x[0], result.y[-1] - result.y[0])
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    positive = radial_k > 0.0
    ax.scatter(radial_k[positive], spectrum[positive], s=8, alpha=0.45)
    ax.axvline(dominant_k, color="tab:red", lw=1.5, label=f"dominant k={dominant_k:.3g}")
    ax.set_xlabel("radial wavenumber")
    ax.set_ylabel("Fourier power of u")
    ax.set_yscale("symlog", linthresh=1.0e-10)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def write_summary(row: dict[str, object], paths: dict[str, Path], path: Path) -> None:
    lines = [
        "# Nonlinear PDE Results 05",
        "",
        "This run ends the 1D validation phase and starts the Roy-style 2D phase. It only verifies nonlinear 2D pattern formation; it does not compute predator mortality thresholds.",
        "",
        "## 2D Pattern Reproduction",
        "",
        f"- parameter line: `kappa={row['kappa']}`, `eta={row['eta']}`, `mu={row['mu']}`, `delta={row['delta']}`",
        f"- grid: `{row['n_x']} x {row['n_y']}`, `T={row['T']}`, `dt={row['dt']}`, seed `{row['seed']}`",
        f"- ODE stable at homogeneous equilibrium: `{row['ode_stable']}`",
        f"- continuous max spatial growth: `{float(row['continuous_max_spatial_growth']):.6g}` at `k={float(row['continuous_best_k']):.6g}`",
        f"- discrete max spatial growth: `{float(row['discrete_max_spatial_growth']):.6g}` at mode `{row['dominant_discrete_mode']}`",
        f"- final mean predator density: `{float(row['mean_w_final']):.6g}`",
        f"- final spatial variances `(u, v, w)`: `({float(row['var_u_final']):.6g}, {float(row['var_v_final']):.6g}, {float(row['var_w_final']):.6g})`",
        f"- dominant Fourier wavelength: `{float(row['dominant_wavelength']):.6g}`",
        f"- pattern measurable: `{row['pattern_measurable']}`",
        f"- negative population values detected: `{row['negative_detected']}`",
        f"- negative free-space values detected: `{row['z_negative_detected']}`",
        "",
        "Interpretation: this confirms that the validated Roy-style parameter line can produce a measurable 2D nonlinear spatial pattern under no-flux finite-difference simulation. It is not a rescue-threshold result.",
        "",
        "Outputs:",
        "",
        f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
        f"- `{paths['maps'].relative_to(ROOT)}`",
        f"- `{paths['mean_w'].relative_to(ROOT)}`",
        f"- `{paths['power'].relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    params = baseline_params()
    config = Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=80.0, dt=0.01, record_every=20, seed=20260601)
    eq = require_positive_equilibrium(params)
    continuous = continuous_turing_scan(params, eq, k_min=1.0e-4, k_max=12.0, n_k=700, tol=1.0e-8)
    discrete = neumann_turing_scan(params.with_updates(L=config.L_x), eq, n_max=160, tol=1.0e-8)
    result = simulate_pde_2d(params, config, equilibrium=eq)
    d = result.diagnostics

    row = {
        "kappa": params.kappa,
        "eta": params.eta,
        "mu": params.mu,
        "delta": params.delta,
        "D_u": params.D_u,
        "D_v": params.D_v,
        "D_w": params.D_w,
        "n_x": config.n_x,
        "n_y": config.n_y,
        "L_x": config.L_x,
        "L_y": config.L_y,
        "T": config.T,
        "dt": config.dt,
        "seed": config.seed,
        "u_star": eq.u,
        "v_star": eq.v,
        "w_star": eq.w,
        "z_star": eq.z,
        "ode_stable": continuous.ode_stable,
        "continuous_max_spatial_growth": continuous.max_spatial_growth,
        "continuous_best_k": continuous.best_k,
        "discrete_max_spatial_growth": discrete.max_spatial_growth,
        "dominant_discrete_mode": discrete.dominant_mode,
        "mean_w_final": d.mean_w,
        "var_u_final": d.var_u,
        "var_v_final": d.var_v,
        "var_w_final": d.var_w,
        "dominant_k": d.dominant_k,
        "dominant_wavelength": d.dominant_wavelength,
        "dominant_power": d.dominant_power,
        "min_value": d.min_value,
        "min_z": d.min_z,
        "negative_detected": d.negative_detected,
        "z_negative_detected": d.z_negative_detected,
        "pattern_measurable": bool(max(d.var_u, d.var_v, d.var_w) > 1.0e-5),
    }

    maps_path = FIG_DIR / "05_final_maps_mu_0p85.png"
    mean_w_path = FIG_DIR / "05_mean_w_timeseries_mu_0p85.png"
    power_path = FIG_DIR / "05_power_spectrum_mu_0p85.png"
    plot_maps(result, maps_path)
    plot_mean_w(result, mean_w_path)
    plot_power_spectrum(result, power_path)
    write_csv(row, OUTPUT_CSV)
    write_summary(row, {"maps": maps_path, "mean_w": mean_w_path, "power": power_path}, SUMMARY_MD)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
