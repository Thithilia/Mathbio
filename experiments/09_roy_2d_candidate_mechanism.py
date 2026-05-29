"""Mechanism and convergence analysis for the Roy-style 2D candidate.

This experiment follows PR #3's threshold-validation result.  The previous
pipeline found one weak candidate at ``D_w / D_u = 150``: Stage C gave a
positive threshold shift, but Stage D did not preserve that sign beyond
validation tolerance.  This script asks *why* by decomposing the spatial PDE
predator growth term into a mean-field contribution and a spatial covariance
bonus.

Primary question
----------------
Is the ``D_w / D_u = 150`` signal a real spatial-growth mechanism, a transient
spatial effect, or a numerical/grid/domain artifact?

Generated outputs
-----------------
- results/roy_2d_candidate_mechanism.csv
- results/roy_2d_candidate_convergence.csv
- nonlinear_pde_results_08_mechanism.md

The script intentionally does not use Fourier/wavelength morphology as the
criterion.  Pattern diagnostics are recorded only to contextualize the growth
covariance signal.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.roy_style_2d import (
    Roy2DConfig,
    check_explicit_diffusion_cfl,
    grid_2d,
    laplacian_neumann_2d,
    perturbed_equilibrium_2d,
    pde_tail_persistence,
    simulate_pde_2d,
    with_stress,
)
from src.roy_style_model import (
    RoyParams,
    beta2_sharp,
    continuous_turing_scan,
    free_space,
    reaction_part,
    require_positive_equilibrium,
)


RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "roy_2d_longtime"
THRESHOLD_CSV = RESULTS_DIR / "roy_2d_threshold_comparison.csv"
GROUP_SUMMARY_CSV = RESULTS_DIR / "roy_2d_threshold_group_summary.csv"
MECHANISM_CSV = RESULTS_DIR / "roy_2d_candidate_mechanism.csv"
CONVERGENCE_CSV = RESULTS_DIR / "roy_2d_candidate_convergence.csv"
SUMMARY_MD = ROOT / "nonlinear_pde_results_08_mechanism.md"

EPSILON = 1.0e-4
TAIL_FRACTION = 0.25
CANDIDATE_RATIO = 150.0
BASELINE_RATIO = 100.0


MECHANISM_FIELDNAMES = [
    "run_id",
    "label",
    "group_id",
    "mu",
    "D_w_over_D_u",
    "eta",
    "stress",
    "T",
    "n_x",
    "n_y",
    "L_x",
    "L_y",
    "dt",
    "seed",
    "time",
    "mean_u",
    "mean_v",
    "mean_w",
    "mean_z",
    "var_u",
    "var_v",
    "var_w",
    "pattern_strength",
    "min_z",
    "spatial_growth_term",
    "meanfield_growth_term",
    "spatial_covariance_bonus",
    "relative_covariance_bonus",
    "local_growth_mean",
    "local_growth_at_means",
    "cov_w_local_growth",
    "edible_mean",
    "edible_at_means",
    "success",
]


CONVERGENCE_FIELDNAMES = [
    "run_id",
    "label",
    "group_id",
    "mu",
    "D_w_over_D_u",
    "eta",
    "stress",
    "T",
    "n_x",
    "n_y",
    "L_x",
    "L_y",
    "dt",
    "seed",
    "source_threshold_stage",
    "source_delta_threshold",
    "source_threshold_tolerance",
    "source_final_group_conclusion",
    "tail_mean_w",
    "tail_min_w",
    "tail_mean_pattern_strength",
    "tail_mean_min_z",
    "tail_mean_spatial_growth_term",
    "tail_mean_meanfield_growth_term",
    "tail_mean_spatial_covariance_bonus",
    "tail_mean_relative_covariance_bonus",
    "tail_mean_cov_w_local_growth",
    "tail_fraction_positive_covariance",
    "tail_persistent",
    "turing_available",
    "turing_growth_at_stress",
    "turing_best_k_at_stress",
    "turing_unstable_at_stress",
    "diagnosis",
]


@dataclass(frozen=True)
class ThresholdReference:
    group_id: str
    stage: str
    stress: float
    ode_threshold: float
    pde_threshold: float
    delta_threshold: float
    threshold_tolerance: float
    final_group_conclusion: str


@dataclass(frozen=True)
class MechanismRunSpec:
    run_id: str
    label: str
    params: RoyParams
    stress: float
    config: Roy2DConfig
    threshold_reference: ThresholdReference | None


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


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_threshold_references() -> dict[str, ThresholdReference]:
    """Read the highest-priority threshold rows from PR #3 outputs.

    Priority is D > C > B > A, because later stages are higher precision and/or
    higher grid/time validation.  The selected stress is the PDE threshold for
    that group, since this is the mortality region where a spatial advantage
    would matter most.
    """

    rows = read_csv_dicts(THRESHOLD_CSV)
    group_rows = read_csv_dicts(GROUP_SUMMARY_CSV)
    final_by_group = {
        row["group_id"]: row.get("final_group_conclusion", "unknown") for row in group_rows
    }
    priority = {
        "D_grid_escalation": 4,
        "C_seed_validation": 3,
        "B_precision_screen": 2,
        "A_quick_threshold": 1,
    }
    best: dict[str, tuple[int, ThresholdReference]] = {}
    for row in rows:
        group_id = row.get("group_id", "")
        stage = row.get("stage", "")
        pde_threshold = finite_float(row.get("pde_threshold"))
        ode_threshold = finite_float(row.get("ode_threshold"))
        delta = finite_float(row.get("delta_threshold"))
        tolerance = finite_float(row.get("threshold_tolerance"))
        if not group_id or pde_threshold is None or ode_threshold is None or delta is None or tolerance is None:
            continue
        rank = priority.get(stage, 0)
        reference = ThresholdReference(
            group_id=group_id,
            stage=stage,
            stress=pde_threshold,
            ode_threshold=ode_threshold,
            pde_threshold=pde_threshold,
            delta_threshold=delta,
            threshold_tolerance=tolerance,
            final_group_conclusion=final_by_group.get(group_id, "unknown"),
        )
        if group_id not in best or rank > best[group_id][0]:
            best[group_id] = (rank, reference)
    return {group_id: item[1] for group_id, item in best.items()}


def edible_fraction(u: np.ndarray | float, v: np.ndarray | float, params: RoyParams) -> np.ndarray | float:
    beta2 = beta2_sharp(params)
    return (params.beta1 * u + beta2 * v) / (1.0 + u + v)


def local_predator_growth_factor(
    u: np.ndarray | float,
    v: np.ndarray | float,
    w: np.ndarray | float,
    params: RoyParams,
) -> np.ndarray | float:
    z = free_space(u, v, w, params)
    return edible_fraction(u, v, params) * z - params.delta - params.mu * w


def growth_decomposition(u: np.ndarray, v: np.ndarray, w: np.ndarray, params: RoyParams) -> dict[str, float]:
    """Compare spatial predator growth with the mean-field growth term.

    The PDE mean predator equation contains

        mean(w * local_growth_factor(u, v, w)).

    The comparable mean-field term is

        mean(w) * local_growth_factor(mean(u), mean(v), mean(w)).

    Their difference is the spatial covariance bonus.  A robust positive value
    near threshold is necessary for a mechanistic spatial-rescue interpretation.
    """

    z = free_space(u, v, w, params)
    local_growth = local_predator_growth_factor(u, v, w, params)
    edible = edible_fraction(u, v, params)
    mean_u = float(np.mean(u))
    mean_v = float(np.mean(v))
    mean_w = float(np.mean(w))
    mean_z = float(np.mean(z))
    local_at_means = float(local_predator_growth_factor(mean_u, mean_v, mean_w, params))
    edible_at_means = float(edible_fraction(mean_u, mean_v, params))
    spatial_growth = float(np.mean(w * local_growth))
    meanfield_growth = mean_w * local_at_means
    covariance_bonus = spatial_growth - meanfield_growth
    scale = max(abs(meanfield_growth), abs(spatial_growth), 1.0e-12)
    centered_w = w - mean_w
    centered_local_growth = local_growth - float(np.mean(local_growth))
    return {
        "mean_u": mean_u,
        "mean_v": mean_v,
        "mean_w": mean_w,
        "mean_z": mean_z,
        "var_u": float(np.var(u)),
        "var_v": float(np.var(v)),
        "var_w": float(np.var(w)),
        "pattern_strength": float(max(np.var(u), np.var(v), np.var(w))),
        "min_z": float(np.min(z)),
        "spatial_growth_term": spatial_growth,
        "meanfield_growth_term": float(meanfield_growth),
        "spatial_covariance_bonus": float(covariance_bonus),
        "relative_covariance_bonus": float(covariance_bonus / scale),
        "local_growth_mean": float(np.mean(local_growth)),
        "local_growth_at_means": local_at_means,
        "cov_w_local_growth": float(np.mean(centered_w * centered_local_growth)),
        "edible_mean": float(np.mean(edible)),
        "edible_at_means": edible_at_means,
    }


def simulate_pde_mechanism(spec: MechanismRunSpec) -> list[dict[str, object]]:
    params = spec.params
    config = stable_config_for(params, spec.config)
    stressed = with_stress(params, spec.stress)
    check_explicit_diffusion_cfl(stressed, config)
    x, y, dx, dy = grid_2d(config)
    equilibrium = require_positive_equilibrium(params)
    u, v, w = perturbed_equilibrium_2d(equilibrium, config)
    n_steps = int(np.ceil(config.T / config.dt))
    record_every = max(1, config.record_every)
    rows: list[dict[str, object]] = []
    group_id = group_id_for(params)

    def record(t_value: float, success: bool) -> None:
        values = growth_decomposition(u, v, w, stressed)
        rows.append(
            {
                "run_id": spec.run_id,
                "label": spec.label,
                "group_id": group_id,
                "mu": params.mu,
                "D_w_over_D_u": params.D_w / params.D_u,
                "eta": params.eta,
                "stress": spec.stress,
                "T": config.T,
                "n_x": config.n_x,
                "n_y": config.n_y,
                "L_x": config.L_x,
                "L_y": config.L_y,
                "dt": config.dt,
                "seed": config.seed,
                "time": float(t_value),
                **values,
                "success": bool(success),
            }
        )

    record(0.0, True)
    success = True
    for step in range(1, n_steps + 1):
        reactions = reaction_part(u, v, w, stressed)
        u_next = u + config.dt * (stressed.D_u * laplacian_neumann_2d(u, dx, dy) + reactions[0])
        v_next = v + config.dt * (stressed.D_v * laplacian_neumann_2d(v, dx, dy) + reactions[1])
        w_next = w + config.dt * (stressed.D_w * laplacian_neumann_2d(w, dx, dy) + reactions[2])
        if not (np.all(np.isfinite(u_next)) and np.all(np.isfinite(v_next)) and np.all(np.isfinite(w_next))):
            success = False
            break
        u, v, w = u_next, v_next, w_next
        if step % record_every == 0 or step == n_steps:
            record(min(step * config.dt, config.T), success)
    if not success:
        record(min(step * config.dt, config.T), success)
    return rows


def tail_mean(rows: list[dict[str, object]], key: str, tail_fraction: float = TAIL_FRACTION) -> float:
    if not rows:
        return float("nan")
    final_t = max(float(row["time"]) for row in rows)
    initial_t = min(float(row["time"]) for row in rows)
    cutoff = final_t - tail_fraction * (final_t - initial_t)
    values = [float(row[key]) for row in rows if float(row["time"]) >= cutoff]
    return float(np.mean(values)) if values else float("nan")


def tail_fraction_positive(rows: list[dict[str, object]], key: str, tail_fraction: float = TAIL_FRACTION) -> float:
    if not rows:
        return float("nan")
    final_t = max(float(row["time"]) for row in rows)
    initial_t = min(float(row["time"]) for row in rows)
    cutoff = final_t - tail_fraction * (final_t - initial_t)
    values = [float(row[key]) for row in rows if float(row["time"]) >= cutoff]
    return float(np.mean([value > 0.0 for value in values])) if values else float("nan")


def turing_at_stress(params: RoyParams, stress: float) -> dict[str, object]:
    stressed = with_stress(params, stress)
    try:
        eq = require_positive_equilibrium(stressed)
        scan = continuous_turing_scan(stressed, eq, k_min=1.0e-4, k_max=12.0, n_k=500, tol=1.0e-8)
    except Exception:
        return {
            "turing_available": False,
            "turing_growth_at_stress": float("nan"),
            "turing_best_k_at_stress": float("nan"),
            "turing_unstable_at_stress": False,
        }
    return {
        "turing_available": True,
        "turing_growth_at_stress": scan.max_spatial_growth,
        "turing_best_k_at_stress": scan.best_k,
        "turing_unstable_at_stress": scan.turing_unstable,
    }


def convergence_row(spec: MechanismRunSpec, rows: list[dict[str, object]]) -> dict[str, object]:
    params = spec.params
    config = stable_config_for(params, spec.config)
    reference = spec.threshold_reference
    pde_result = simulate_pde_2d(params, config, stress=spec.stress, equilibrium=require_positive_equilibrium(params))
    persistent, _ = pde_tail_persistence(pde_result, EPSILON)
    turing = turing_at_stress(params, spec.stress)
    tail_bonus = tail_mean(rows, "spatial_covariance_bonus")
    tail_relative = tail_mean(rows, "relative_covariance_bonus")
    tail_pattern = tail_mean(rows, "pattern_strength")
    source_delta = reference.delta_threshold if reference is not None else float("nan")
    source_tol = reference.threshold_tolerance if reference is not None else float("nan")
    source_final = reference.final_group_conclusion if reference is not None else "none"
    if not bool(persistent):
        diagnosis = "tail_predator_not_persistent"
    elif tail_bonus > 0.0 and np.isfinite(source_delta) and source_delta > source_tol:
        diagnosis = "positive_covariance_and_positive_threshold_reference"
    elif tail_bonus > 0.0:
        diagnosis = "positive_covariance_without_threshold_support"
    elif tail_bonus < 0.0:
        diagnosis = "negative_covariance"
    else:
        diagnosis = "near_zero_covariance"

    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "group_id": group_id_for(params),
        "mu": params.mu,
        "D_w_over_D_u": params.D_w / params.D_u,
        "eta": params.eta,
        "stress": spec.stress,
        "T": config.T,
        "n_x": config.n_x,
        "n_y": config.n_y,
        "L_x": config.L_x,
        "L_y": config.L_y,
        "dt": config.dt,
        "seed": config.seed,
        "source_threshold_stage": reference.stage if reference is not None else "none",
        "source_delta_threshold": source_delta,
        "source_threshold_tolerance": source_tol,
        "source_final_group_conclusion": source_final,
        "tail_mean_w": tail_mean(rows, "mean_w"),
        "tail_min_w": min(float(row["mean_w"]) for row in rows[-max(1, len(rows) // 4) :]) if rows else float("nan"),
        "tail_mean_pattern_strength": tail_pattern,
        "tail_mean_min_z": tail_mean(rows, "min_z"),
        "tail_mean_spatial_growth_term": tail_mean(rows, "spatial_growth_term"),
        "tail_mean_meanfield_growth_term": tail_mean(rows, "meanfield_growth_term"),
        "tail_mean_spatial_covariance_bonus": tail_bonus,
        "tail_mean_relative_covariance_bonus": tail_relative,
        "tail_mean_cov_w_local_growth": tail_mean(rows, "cov_w_local_growth"),
        "tail_fraction_positive_covariance": tail_fraction_positive(rows, "spatial_covariance_bonus"),
        "tail_persistent": bool(persistent),
        **turing,
        "diagnosis": diagnosis,
    }


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_reference(references: dict[str, ThresholdReference], params: RoyParams) -> ThresholdReference | None:
    return references.get(group_id_for(params))


def default_reference_stress(reference: ThresholdReference | None, fallback: float = 0.396) -> float:
    return reference.stress if reference is not None and np.isfinite(reference.stress) else fallback


def build_specs(profile: str) -> list[MechanismRunSpec]:
    references = read_threshold_references()
    baseline = baseline_params(mu=0.85, D_w_ratio=BASELINE_RATIO)
    candidate = baseline_params(mu=0.85, D_w_ratio=CANDIDATE_RATIO)
    baseline_ref = select_reference(references, baseline)
    candidate_ref = select_reference(references, candidate)

    specs = [
        MechanismRunSpec(
            "M_001",
            "baseline_64_near_threshold",
            baseline,
            default_reference_stress(baseline_ref, 0.39576),
            Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=200.0, dt=0.01, record_every=200, seed=20260621),
            baseline_ref,
        ),
        MechanismRunSpec(
            "M_002",
            "candidate_64_near_threshold",
            candidate,
            default_reference_stress(candidate_ref, 0.39664),
            Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=200.0, dt=0.0067, record_every=300, seed=20260621),
            candidate_ref,
        ),
        MechanismRunSpec(
            "M_003",
            "candidate_96_grid_escalation",
            candidate,
            default_reference_stress(candidate_ref, 0.39662),
            Roy2DConfig(n_x=96, n_y=96, L_x=20.0, L_y=20.0, T=250.0, dt=0.00293, record_every=700, seed=20260624),
            candidate_ref,
        ),
    ]

    if profile in {"focused", "full"}:
        for idx, ratio in enumerate([120.0, 140.0, 160.0, 180.0], start=4):
            params = baseline_params(mu=0.85, D_w_ratio=ratio)
            ref = select_reference(references, params)
            specs.append(
                MechanismRunSpec(
                    f"M_{idx:03d}",
                    f"ratio_{ratio:g}_64_same_stress_window",
                    params,
                    default_reference_stress(candidate_ref, 0.39664),
                    Roy2DConfig(n_x=64, n_y=64, L_x=20.0, L_y=20.0, T=200.0, dt=0.008, record_every=300, seed=20260625),
                    ref,
                )
            )

    if profile == "full":
        specs.append(
            MechanismRunSpec(
                "M_008",
                "candidate_96_larger_domain",
                candidate,
                default_reference_stress(candidate_ref, 0.39662),
                Roy2DConfig(n_x=96, n_y=96, L_x=30.0, L_y=30.0, T=250.0, dt=0.0066, record_every=450, seed=20260626),
                candidate_ref,
            )
        )

    return specs


def interpret(convergence_rows: list[dict[str, object]]) -> tuple[str, str]:
    candidate_rows = [row for row in convergence_rows if float(row["D_w_over_D_u"]) == CANDIDATE_RATIO]
    high_rows = [row for row in candidate_rows if int(row["n_x"]) >= 96]
    candidate_cov = [float(row["tail_mean_spatial_covariance_bonus"]) for row in candidate_rows]
    high_cov = [float(row["tail_mean_spatial_covariance_bonus"]) for row in high_rows]
    candidate_reference = next((row for row in candidate_rows if row["source_final_group_conclusion"] != "none"), None)
    final_reference = candidate_reference["source_final_group_conclusion"] if candidate_reference else "unknown"

    if final_reference == "rescue_supported":
        return (
            "robust_spatial_rescue_mechanism",
            "The threshold pipeline already classified the candidate as rescue_supported, and this mechanism run should be used to document the covariance pathway.",
        )
    if final_reference == "inconclusive_candidate":
        if high_cov and np.mean(high_cov) > 0.0:
            return (
                "positive_covariance_but_threshold_not_robust",
                "The candidate retains a positive spatial-growth covariance signal in high-grid diagnostics, but the Stage D threshold shift remains within tolerance; treat it as a weak mechanism, not a rescue claim.",
            )
        return (
            "transient_or_numerical_candidate",
            "The candidate did not retain a positive high-grid covariance signal and Stage D already failed to preserve the threshold sign beyond tolerance.",
        )
    if candidate_cov and np.mean(candidate_cov) > 0.0:
        return (
            "positive_covariance_without_threshold_support",
            "Spatial structure can increase the instantaneous predator growth term, but the validated threshold outputs do not support a rescue claim.",
        )
    return (
        "no_mechanistic_spatial_rescue_signal",
        "Neither threshold validation nor growth-covariance diagnostics support spatial rescue in the candidate regime.",
    )


def write_summary(convergence_rows: list[dict[str, object]], profile: str) -> None:
    conclusion, explanation = interpret(convergence_rows)
    lines = [
        "# Nonlinear PDE Results 08: Candidate Mechanism",
        "",
        f"**Mechanism conclusion: `{conclusion}`.** {explanation}",
        "",
        "## Question",
        "",
        "PR #3 found an inconclusive threshold-shift candidate at `D_w/D_u=150`: Stage C was positive, but Stage D did not preserve the sign beyond tolerance. This analysis asks whether that signal has a real spatial-growth mechanism.",
        "",
        "## Diagnostic quantity",
        "",
        "For each PDE snapshot, the predator growth term is decomposed as:",
        "",
        "`spatial_growth = mean(w * A(u, v, w, z))`",
        "",
        "`meanfield_growth = mean(w) * A(mean(u), mean(v), mean(w), mean(z))`",
        "",
        "`spatial_covariance_bonus = spatial_growth - meanfield_growth`",
        "",
        "where `A = ((beta1*u + beta2*v)/(1+u+v))*z - delta - mu*w` under the stressed parameters.",
        "",
        "A robust spatial-rescue mechanism requires a positive tail covariance bonus that also survives threshold validation. Positive covariance alone is not enough.",
        "",
            "## Runs",
            "",
            f"- profile: `{profile}`",
            f"- run summaries: `{len(convergence_rows)}`",
            "",
        "| run_id | label | D_w/D_u | grid | L | stress | tail mean w | tail covariance bonus | threshold reference | diagnosis |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in convergence_rows:
        lines.append(
            "| {run_id} | {label} | {ratio:.4g} | {nx}x{ny} | {lx:.4g} | {stress:.6g} | {mean_w:.6g} | {bonus:.6g} | {ref} | {diagnosis} |".format(
                run_id=row["run_id"],
                label=row["label"],
                ratio=float(row["D_w_over_D_u"]),
                nx=int(row["n_x"]),
                ny=int(row["n_y"]),
                lx=float(row["L_x"]),
                stress=float(row["stress"]),
                mean_w=float(row["tail_mean_w"]),
                bonus=float(row["tail_mean_spatial_covariance_bonus"]),
                ref=row["source_final_group_conclusion"],
                diagnosis=row["diagnosis"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- If the covariance bonus is positive but the threshold pipeline remains `inconclusive_candidate`, the result is a weak spatial mechanism, not rescue.",
            "- If covariance vanishes or changes sign under higher grid/domain checks, the candidate is interpreted as transient or numerical.",
            "- Only a positive covariance signal plus validated positive `Delta m_c` would support a spatial-rescue mechanism.",
            "",
            "## Outputs",
            "",
            f"- `{MECHANISM_CSV.relative_to(ROOT)}`",
            f"- `{CONVERGENCE_CSV.relative_to(ROOT)}`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(profile: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    specs = build_specs(profile)
    mechanism_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    for spec in specs:
        print(f"mechanism {spec.run_id}: {spec.label}, stress={spec.stress:.6g}")
        rows = simulate_pde_mechanism(spec)
        mechanism_rows.extend(rows)
        convergence_rows.append(convergence_row(spec, rows))
    write_csv(mechanism_rows, MECHANISM_CSV, MECHANISM_FIELDNAMES)
    write_csv(convergence_rows, CONVERGENCE_CSV, CONVERGENCE_FIELDNAMES)
    write_summary(convergence_rows, profile)
    print(SUMMARY_MD.read_text(encoding="utf-8"))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["minimal", "focused", "full"],
        default="focused",
        help="minimal runs baseline/candidate/high-grid only; focused adds nearby ratios; full adds a larger-domain check.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.profile)
