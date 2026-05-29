"""Final Roy-style 2D threshold pipeline for spatial-rescue testing.

The primary output is the predator-mortality threshold difference

    Delta m_c = m_c_PDE - m_c_ODE.

Pattern morphology diagnostics are intentionally secondary in this script.
"""

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

from src.roy_style_2d import (
    Roy2DConfig,
    find_ode_threshold,
    find_pde_threshold,
    make_refined_stress_bracket,
    summarize_delta_group,
)
from src.roy_style_model import RoyParams, continuous_turing_scan, require_positive_equilibrium


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_2d_longtime"
THRESHOLD_CSV = RESULTS_DIR / "roy_2d_threshold_comparison.csv"
GROUP_SUMMARY_CSV = RESULTS_DIR / "roy_2d_threshold_group_summary.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_07.md"
THRESHOLD_FIG = FIG_DIR / "07_threshold_delta.png"


EPSILON = 1.0e-4
STAGE_A_FIXED_THRESHOLD_TOLERANCE = 1.0e-3
VALIDATION_FIXED_THRESHOLD_TOLERANCE = 5.0e-4
STAGE_D_FIXED_THRESHOLD_TOLERANCE = 7.5e-4


EQ_CACHE: dict[tuple[float, ...], object] = {}
TURING_CACHE: dict[tuple[float, ...], tuple[bool, float]] = {}


THRESHOLD_FIELDNAMES = [
    "run_id",
    "stage",
    "group_id",
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
    "delta_interval_low",
    "delta_interval_high",
    "threshold_precision_ok",
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


GROUP_FIELDNAMES = [
    "group_id",
    "mu",
    "D_w_over_D_u",
    "eta",
    "gamma",
    "beta1",
    "stage_c_seed_count",
    "stage_c_delta_min",
    "stage_c_delta_max",
    "stage_c_delta_mean",
    "stage_c_delta_std",
    "stage_c_interval_low",
    "stage_c_interval_high",
    "stage_c_group_conclusion",
    "stage_d_delta",
    "stage_d_tolerance",
    "stage_d_status",
    "final_group_conclusion",
    "final_reason",
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


def group_id_for(params: RoyParams) -> str:
    key = param_key(params)
    return (
        f"mu={key[7]:.4g}|DwDu={key[10] / key[8]:.4g}|eta={key[4]:.4g}|"
        f"gamma={key[2]:.4g}|beta1={key[5]:.4g}"
    )


def stable_config_for(params: RoyParams, config: Roy2DConfig) -> Roy2DConfig:
    dx = config.L_x / (config.n_x - 1)
    dy = config.L_y / (config.n_y - 1)
    max_diffusion = max(params.D_u, params.D_v, params.D_w)
    stable_dt = 0.22 / (max_diffusion * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if config.dt <= stable_dt:
        return config
    return replace(config, dt=0.9 * stable_dt)


def equilibrium_for(params: RoyParams):
    key = param_key(params)
    if key not in EQ_CACHE:
        EQ_CACHE[key] = require_positive_equilibrium(params)
    return EQ_CACHE[key]


def baseline_turing_status(params: RoyParams) -> tuple[bool, float]:
    key = param_key(params)
    if key not in TURING_CACHE:
        eq = equilibrium_for(params)
        scan = continuous_turing_scan(params, eq, k_min=1.0e-4, k_max=12.0, n_k=500, tol=1.0e-8)
        TURING_CACHE[key] = (scan.turing_unstable, scan.max_spatial_growth)
    return TURING_CACHE[key]


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def threshold_gap(result: dict[str, object]) -> float:
    low = finite_float(result.get("s_low"))
    high = finite_float(result.get("s_high"))
    if low is None or high is None:
        return float("nan")
    return max(0.0, high - low)


def status_for_delta(stage: str, delta: float, tolerance: float) -> str:
    if stage == "A_quick_threshold":
        if delta > tolerance:
            return "stage_a_positive_candidate"
        if delta < -tolerance:
            return "stage_a_negative_candidate"
        return "stage_a_no_measurable_difference"
    return "ok"


def threshold_row(
    run_id: str,
    stage: str,
    group_id: str,
    axis: str,
    varied_parameter: str,
    varied_value: float,
    params: RoyParams,
    config: Roy2DConfig,
    fixed_tolerance: float,
    seed_count: int,
    max_iter_ode: int,
    max_iter_pde: int,
    validated: bool,
    s_low: float,
    s_high: float,
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
        tolerance = max(fixed_tolerance, gap_width)
        precision_ok = bool(np.isfinite(gap_width) and gap_width <= fixed_tolerance)
        thresholds_ok = (
            ode_threshold["status"] == "ok"
            and pde_threshold["status"] == "ok"
            and ode_value is not None
            and pde_value is not None
        )
        if thresholds_ok:
            delta = pde_value - ode_value
            delta_low = delta - tolerance
            delta_high = delta + tolerance
            rescue = bool(precision_ok and delta > tolerance)
            inhibition = bool(precision_ok and delta < -tolerance)
            status = status_for_delta(stage, delta, tolerance) if precision_ok else "insufficient_precision"
        else:
            delta = float("nan")
            delta_low = float("nan")
            delta_high = float("nan")
            rescue = False
            inhibition = False
            precision_ok = False
            status = f"threshold_failed: ode={ode_threshold['status']}; pde={pde_threshold['status']}"
        return {
            "run_id": run_id,
            "stage": stage,
            "group_id": group_id,
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
            "delta_interval_low": delta_low,
            "delta_interval_high": delta_high,
            "threshold_precision_ok": precision_ok,
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
                "group_id": group_id,
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
                "threshold_precision_ok": False,
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
        ("baseline", "none", 0.0, baseline_params(mu=0.85, D_w_ratio=100.0, eta=0.005)),
    ]
    for mu in [0.60, 0.72, 0.80, 0.85, 0.89, 0.95]:
        regimes.append(("mu", "mu", mu, baseline_params(mu=mu, D_w_ratio=100.0, eta=0.005)))
    for ratio in [40.0, 70.0, 100.0, 150.0, 250.0, 400.0]:
        regimes.append(("diffusion_ratio", "D_w_over_D_u", ratio, baseline_params(mu=0.85, D_w_ratio=ratio, eta=0.005)))
    for eta in [0.0025, 0.005, 0.0075]:
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


def row_is_valid_threshold(row: dict[str, object]) -> bool:
    return str(row["status"]) in {
        "ok",
        "stage_a_positive_candidate",
        "stage_a_negative_candidate",
        "stage_a_no_measurable_difference",
    }


def select_stage_b_rows(stage_a_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in stage_a_rows if row_is_valid_threshold(row)]
    selected: list[dict[str, object]] = []
    selected.extend(row for row in valid if row["axis"] == "baseline")
    selected.extend(row for row in valid if truthy(row["candidate_spatial_rescue"]))
    selected.extend(row for row in valid if truthy(row["candidate_spatial_inhibition"]))
    finite = [row for row in valid if finite_float(row["delta_threshold"]) is not None]
    if finite:
        selected.append(min(finite, key=lambda row: abs(float(row["delta_threshold"]))))
    return dedupe_rows_by_group(selected)


def select_stage_c_rows(stage_b_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in stage_b_rows if row["status"] == "ok"]
    selected: list[dict[str, object]] = []
    selected.extend(row for row in valid if row["axis"] == "baseline")
    selected.extend(row for row in valid if truthy(row["candidate_spatial_rescue"]))
    selected.extend(row for row in valid if truthy(row["candidate_spatial_inhibition"]))
    if not any(truthy(row["candidate_spatial_rescue"]) or truthy(row["candidate_spatial_inhibition"]) for row in valid):
        finite = [row for row in valid if finite_float(row["delta_threshold"]) is not None]
        if finite:
            selected.append(min(finite, key=lambda row: abs(float(row["delta_threshold"]))))
    return dedupe_rows_by_group(selected)


def dedupe_rows_by_group(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in seen:
            continue
        seen.add(group_id)
        selected.append(row)
    return selected


def params_from_row(row: dict[str, object]) -> RoyParams:
    return baseline_params(
        mu=float(row["mu"]),
        D_w_ratio=float(row["D_w_over_D_u"]),
        eta=float(row["eta"]),
        gamma=float(row["gamma"]),
        beta1=float(row["beta1"]),
    )


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_group_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stage_c_rows = [row for row in rows if row["stage"] == "C_seed_validation"]
    group_ids = sorted({str(row["group_id"]) for row in stage_c_rows})
    summaries: list[dict[str, object]] = []
    for group_id in group_ids:
        group_rows = [row for row in stage_c_rows if row["group_id"] == group_id]
        valid_rows = [row for row in group_rows if row["status"] == "ok" and truthy(row["threshold_precision_ok"])]
        representative = group_rows[0]
        if len(valid_rows) >= 3:
            deltas = np.array([float(row["delta_threshold"]) for row in valid_rows], dtype=float)
            tolerances = np.array([float(row["threshold_tolerance"]) for row in valid_rows], dtype=float)
            summary = summarize_delta_group(deltas, tolerances)
            stage_c_conclusion = str(summary["conclusion"])
        else:
            summary = summarize_delta_group(np.array([], dtype=float), np.array([], dtype=float))
            stage_c_conclusion = "invalid"

        stage_d_rows = [row for row in rows if row["stage"] == "D_grid_escalation" and row["group_id"] == group_id]
        stage_d = stage_d_rows[0] if stage_d_rows else None
        stage_d_delta = finite_float(stage_d["delta_threshold"]) if stage_d is not None else None
        stage_d_tolerance = finite_float(stage_d["threshold_tolerance"]) if stage_d is not None else None
        stage_d_status = str(stage_d["status"]) if stage_d is not None else "not_required"

        if stage_c_conclusion == "invalid":
            final = "invalid"
            reason = "Stage C did not produce three valid high-precision seed rows."
        elif stage_c_conclusion == "no_measurable_effect":
            final = "no_measurable_effect"
            reason = "Stage C validation intervals overlap zero."
        else:
            if stage_d is None:
                final = "inconclusive_candidate"
                reason = "Stage C suggested an effect, but Stage D was not run."
            elif stage_d["status"] != "ok" or not truthy(stage_d["threshold_precision_ok"]):
                final = "inconclusive_candidate"
                reason = "Stage D did not produce a valid high-precision row."
            elif stage_c_conclusion == "rescue_supported" and stage_d_delta is not None and stage_d_tolerance is not None and stage_d_delta > stage_d_tolerance:
                final = "rescue_supported"
                reason = "Stage C intervals were positive and Stage D retained positive Delta m_c."
            elif stage_c_conclusion == "inhibition_supported" and stage_d_delta is not None and stage_d_tolerance is not None and stage_d_delta < -stage_d_tolerance:
                final = "inhibition_supported"
                reason = "Stage C intervals were negative and Stage D retained negative Delta m_c."
            else:
                final = "inconclusive_candidate"
                reason = "Stage D failed to preserve the Stage C sign beyond tolerance."

        summaries.append(
            {
                "group_id": group_id,
                "mu": representative["mu"],
                "D_w_over_D_u": representative["D_w_over_D_u"],
                "eta": representative["eta"],
                "gamma": representative["gamma"],
                "beta1": representative["beta1"],
                "stage_c_seed_count": len(valid_rows),
                "stage_c_delta_min": summary["delta_min"],
                "stage_c_delta_max": summary["delta_max"],
                "stage_c_delta_mean": summary["delta_mean"],
                "stage_c_delta_std": summary["delta_std"],
                "stage_c_interval_low": summary["interval_low"],
                "stage_c_interval_high": summary["interval_high"],
                "stage_c_group_conclusion": stage_c_conclusion,
                "stage_d_delta": stage_d_delta if stage_d_delta is not None else np.nan,
                "stage_d_tolerance": stage_d_tolerance if stage_d_tolerance is not None else np.nan,
                "stage_d_status": stage_d_status,
                "final_group_conclusion": final,
                "final_reason": reason,
            }
        )
    return summaries


def final_label(group_summaries: list[dict[str, object]]) -> str:
    finals = {str(row["final_group_conclusion"]) for row in group_summaries}
    if "rescue_supported" in finals and "inhibition_supported" not in finals:
        return "Final conclusion: spatial rescue supported"
    if "inhibition_supported" in finals and "rescue_supported" not in finals:
        return "Final conclusion: spatial inhibition supported"
    if "rescue_supported" in finals and "inhibition_supported" in finals:
        return "Final conclusion: inconclusive candidate, not claimed"
    if "inconclusive_candidate" in finals:
        return "Final conclusion: inconclusive candidate, not claimed"
    return "Final conclusion: no measurable spatial-rescue effect in tested regimes"


def plot_threshold_delta(group_summaries: list[dict[str, object]], path: Path) -> None:
    if not group_summaries:
        return
    labels = [str(row["group_id"]) for row in group_summaries]
    deltas = np.array([float(row["stage_c_delta_mean"]) for row in group_summaries], dtype=float)
    low = np.array([float(row["stage_c_interval_low"]) for row in group_summaries], dtype=float)
    high = np.array([float(row["stage_c_interval_high"]) for row in group_summaries], dtype=float)
    yerr = np.vstack([np.maximum(deltas - low, 0.0), np.maximum(high - deltas, 0.0)])
    colors_by_result = {
        "rescue_supported": "#2ca02c",
        "inhibition_supported": "#d62728",
        "no_measurable_effect": "#7f7f7f",
        "inconclusive_candidate": "#ff7f0e",
        "invalid": "#9467bd",
    }
    colors = [colors_by_result.get(str(row["final_group_conclusion"]), "#7f7f7f") for row in group_summaries]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(group_summaries))
    ax.bar(x, deltas, yerr=yerr, capsize=4, color=colors)
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Stage C mean Delta m_c")
    ax.set_title("Validated Roy-style 2D threshold comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_float(value: object, digits: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "nan"
    return f"{number:.{digits}g}"


def write_summary(
    stage_a_rows: list[dict[str, object]],
    stage_b_rows: list[dict[str, object]],
    stage_c_rows: list[dict[str, object]],
    stage_d_rows: list[dict[str, object]],
    group_summaries: list[dict[str, object]],
) -> None:
    label = final_label(group_summaries)
    valid_a = [row for row in stage_a_rows if row_is_valid_threshold(row)]
    positive_a = [row for row in valid_a if truthy(row["candidate_spatial_rescue"])]
    negative_a = [row for row in valid_a if truthy(row["candidate_spatial_inhibition"])]
    finite_a = [row for row in valid_a if finite_float(row["delta_threshold"]) is not None]
    largest_positive = max([float(row["delta_threshold"]) for row in positive_a], default=float("nan"))
    largest_negative = min([float(row["delta_threshold"]) for row in negative_a], default=float("nan"))
    closest = min(finite_a, key=lambda row: abs(float(row["delta_threshold"]))) if finite_a else None

    if label == "Final conclusion: spatial rescue supported":
        answer = "Spatial rescue is supported in the tested Roy-style 2D regimes because at least one Stage C group had strictly positive validation intervals and retained a positive Delta m_c in Stage D."
    elif label == "Final conclusion: spatial inhibition supported":
        answer = "Spatial inhibition is supported in the tested Roy-style 2D regimes because at least one Stage C group had strictly negative validation intervals and retained a negative Delta m_c in Stage D."
    elif label == "Final conclusion: inconclusive candidate, not claimed":
        answer = "A threshold-shift candidate appeared, but it did not survive the full validation/escalation rule; no rescue or inhibition claim is made."
    else:
        answer = "No measurable spatial-rescue effect is found in the tested Roy-style 2D regimes: Stage A candidates do not survive group-level validation as positive Delta m_c effects."

    table_lines = [
        "| group_id | mu | D_w/D_u | eta | Stage C seeds | mean Delta | Delta range | interval range | group conclusion | Stage D result | final conclusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in group_summaries:
        group_label = str(row["group_id"]).replace("|", "\\|")
        table_lines.append(
            "| "
            + " | ".join(
                [
                    group_label,
                    format_float(row["mu"], 4),
                    format_float(row["D_w_over_D_u"], 4),
                    format_float(row["eta"], 4),
                    str(row["stage_c_seed_count"]),
                    format_float(row["stage_c_delta_mean"]),
                    f"[{format_float(row['stage_c_delta_min'])}, {format_float(row['stage_c_delta_max'])}]",
                    f"[{format_float(row['stage_c_interval_low'])}, {format_float(row['stage_c_interval_high'])}]",
                    str(row["stage_c_group_conclusion"]),
                    f"{format_float(row['stage_d_delta'])} ({row['stage_d_status']})",
                    str(row["final_group_conclusion"]),
                ]
            )
            + " |"
        )

    lines = [
        "# Nonlinear PDE Results 07",
        "",
        f"**{label}.** {answer}",
        "",
        "## Core Criterion",
        "",
        "The decisive quantity is `Delta m_c = m_c_PDE - m_c_ODE`, where `m_c_ODE` is the mortality-stress threshold for predator persistence in the well-mixed ODE and `m_c_PDE` is the corresponding threshold in the spatial PDE.",
        "",
        "- `Delta m_c > 0`: spatial structure expands the predator-persistence / indirect-rescue range.",
        "- `Delta m_c < 0`: spatial structure shrinks that range.",
        "- values within the row tolerance are treated as no measurable threshold effect.",
        "",
        "## Persistence Rule",
        "",
        "Persistence is evaluated on the final 25% of the predator-density trajectory. A trajectory is persistent only if all three conditions hold:",
        "",
        "- `tail_mean > epsilon`",
        "- `tail_min > 0.25 * epsilon`",
        "- `tail_slope >= -max(epsilon, 0.25 * tail_mean) / max(tail_duration, 1e-12)`",
        "",
        "PDE runs are also rejected as nonpersistent if negative state values, negative free space `z`, or nonfinite diagnostic time series are detected. ODE runs are rejected if integration fails or produces nonfinite output.",
        "",
        "## Stage A Results",
        "",
        f"- Stage A rows: `{len(valid_a)}`",
        f"- positive candidates: `{len(positive_a)}`",
        f"- negative candidates: `{len(negative_a)}`",
        f"- largest positive Stage A `Delta m_c`: `{format_float(largest_positive)}`",
        f"- largest negative Stage A `Delta m_c`: `{format_float(largest_negative)}`",
        f"- closest-to-zero row: `{closest['group_id'] if closest else 'none'}` with `Delta m_c = {format_float(closest['delta_threshold'] if closest else np.nan)}`",
        "",
        "Stage A is candidate discovery only; it is not used as evidence for rescue or inhibition.",
        "",
        "## Stage B/C Validation",
        "",
        f"- Stage B precision-screen rows: `{len(stage_b_rows)}`",
        f"- Stage C seed-validation rows: `{len(stage_c_rows)}`",
        f"- Stage D grid-escalation rows: `{len(stage_d_rows)}`",
        "",
        *table_lines,
        "",
        "## Final Conclusion",
        "",
        f"**{label}.** The final classification is based on the group summary intervals in `results/roy_2d_threshold_group_summary.csv`, not on pattern morphology or sparse stress classification counts.",
        "",
        "## Secondary Diagnostics",
        "",
        "Pattern morphology, Fourier power, and dominant wavelength remain exploratory diagnostics. They are not part of the rescue criterion in this PR.",
        "",
        "Outputs:",
        "",
        f"- `{THRESHOLD_CSV.relative_to(ROOT)}`",
        f"- `{GROUP_SUMMARY_CSV.relative_to(ROOT)}`",
        f"- `{THRESHOLD_FIG.relative_to(ROOT)}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage_a() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = Roy2DConfig(n_x=36, n_y=36, L_x=20.0, L_y=20.0, T=70.0, dt=0.025, record_every=200, seed=20260620)
    for idx, (axis, varied_parameter, varied_value, params) in enumerate(stage_a_regimes(), start=1):
        gid = group_id_for(params)
        row = threshold_row(
            f"A_{idx:03d}",
            "A_quick_threshold",
            gid,
            axis,
            varied_parameter,
            varied_value,
            params,
            config,
            STAGE_A_FIXED_THRESHOLD_TOLERANCE,
            seed_count=1,
            max_iter_ode=13,
            max_iter_pde=12,
            validated=False,
            s_low=0.0,
            s_high=1.0,
        )
        rows.append(row)
        print(f"Stage A {row['run_id']}: {gid}, Delta={row['delta_threshold']}, status={row['status']}")
    return rows


def run_stage_b(stage_a_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=200.0, dt=0.01, record_every=500, seed=20260621)
    for idx, source in enumerate(select_stage_b_rows(stage_a_rows), start=1):
        params = params_from_row(source)
        low, high = make_refined_stress_bracket(float(source["ode_threshold"]), float(source["pde_threshold"]), margin=0.04)
        row = threshold_row(
            f"B_{idx:03d}",
            "B_precision_screen",
            str(source["group_id"]),
            str(source["axis"]),
            str(source["varied_parameter"]),
            float(source["varied_value"]),
            params,
            config,
            VALIDATION_FIXED_THRESHOLD_TOLERANCE,
            seed_count=1,
            max_iter_ode=14,
            max_iter_pde=12,
            validated=True,
            s_low=low,
            s_high=high,
        )
        rows.append(row)
        print(f"Stage B {row['run_id']}: {row['group_id']}, Delta={row['delta_threshold']}, status={row['status']}")
    return rows


def run_stage_c(stage_b_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_config = Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=200.0, dt=0.01, record_every=500)
    run_idx = 0
    for source in select_stage_c_rows(stage_b_rows):
        params = params_from_row(source)
        low, high = make_refined_stress_bracket(float(source["ode_threshold"]), float(source["pde_threshold"]), margin=0.025)
        for seed in [20260621, 20260622, 20260623]:
            run_idx += 1
            row = threshold_row(
                f"C_{run_idx:03d}",
                "C_seed_validation",
                str(source["group_id"]),
                str(source["axis"]),
                str(source["varied_parameter"]),
                float(source["varied_value"]),
                params,
                replace(base_config, seed=seed),
                VALIDATION_FIXED_THRESHOLD_TOLERANCE,
                seed_count=3,
                max_iter_ode=14,
                max_iter_pde=12,
                validated=True,
                s_low=low,
                s_high=high,
            )
            rows.append(row)
            print(f"Stage C {row['run_id']}: {row['group_id']}, seed={seed}, Delta={row['delta_threshold']}, status={row['status']}")
    return rows


def run_stage_d(rows: list[dict[str, object]], group_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    stage_d_rows: list[dict[str, object]] = []
    base_config = Roy2DConfig(n_x=96, n_y=96, L_x=20.0, L_y=20.0, T=250.0, dt=0.01, record_every=800, seed=20260624)
    effect_groups = [
        row
        for row in group_summaries
        if row["stage_c_group_conclusion"] in {"rescue_supported", "inhibition_supported"}
    ]
    for idx, summary in enumerate(effect_groups, start=1):
        group_rows = [row for row in rows if row["stage"] == "C_seed_validation" and row["group_id"] == summary["group_id"] and row["status"] == "ok"]
        if not group_rows:
            continue
        representative = group_rows[0]
        params = params_from_row(representative)
        ode_mean = float(np.mean([float(row["ode_threshold"]) for row in group_rows]))
        pde_mean = float(np.mean([float(row["pde_threshold"]) for row in group_rows]))
        low, high = make_refined_stress_bracket(ode_mean, pde_mean, margin=0.025)
        row = threshold_row(
            f"D_{idx:03d}",
            "D_grid_escalation",
            str(summary["group_id"]),
            str(representative["axis"]),
            str(representative["varied_parameter"]),
            float(representative["varied_value"]),
            params,
            base_config,
            STAGE_D_FIXED_THRESHOLD_TOLERANCE,
            seed_count=1,
            max_iter_ode=14,
            max_iter_pde=11,
            validated=True,
            s_low=low,
            s_high=high,
        )
        stage_d_rows.append(row)
        print(f"Stage D {row['run_id']}: {row['group_id']}, Delta={row['delta_threshold']}, status={row['status']}")
    return stage_d_rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    stage_a_rows = run_stage_a()
    stage_b_rows = run_stage_b(stage_a_rows)
    stage_c_rows = run_stage_c(stage_b_rows)
    rows = stage_a_rows + stage_b_rows + stage_c_rows
    preliminary_group_summaries = build_group_summaries(rows)
    stage_d_rows = run_stage_d(rows, preliminary_group_summaries)
    rows.extend(stage_d_rows)
    group_summaries = build_group_summaries(rows)

    write_csv(rows, THRESHOLD_CSV, THRESHOLD_FIELDNAMES)
    write_csv(group_summaries, GROUP_SUMMARY_CSV, GROUP_FIELDNAMES)
    plot_threshold_delta(group_summaries, THRESHOLD_FIG)
    write_summary(stage_a_rows, stage_b_rows, stage_c_rows, stage_d_rows, group_summaries)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
