import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.simulate_pde_1d import compute_diagnostics, grid_1d, pack_state, simulate_ode, simulate_pde, unpack_state
from src.roy_style_model import (
    RoyParams,
    continuous_turing_scan as roy_continuous_turing_scan,
    reaction_jacobian as roy_reaction_jacobian,
    reaction_part as roy_reaction_part,
    require_positive_equilibrium as roy_require_positive_equilibrium,
)
from src.roy_style_2d import (
    Roy2DDiagnostics,
    Roy2DResult,
    Roy2DConfig,
    _threshold_search,
    ensure_valid_threshold_bracket,
    is_persistent_tail,
    laplacian_neumann_2d,
    make_refined_stress_bracket,
    pde_tail_persistence,
    simulate_pde_2d,
    summarize_delta_group,
    tail_metrics,
)
from src.roy_evo_spatial import (
    RoyEvoPDEConfig,
    RoyEvoPDEResult,
    RoyEvoParams,
    a_of_q,
    b_of_q,
    bisection_threshold,
    classify_evo_pde_result,
    classify_evo_trajectory,
    initial_state_from_ode_equilibrium,
    r_of_q,
    reaction_ode_evo,
    simulate_pde_evo_2d,
    spatial_mechanism_diagnostics,
    selection_gradient,
    simulate_ode_evo,
)
from src.turing_rescue_holling2 import HollingIIParams, continuous_turing_scan_holling2, solve_coexistence_equilibria_holling2
from src.turing_rescue_model import (
    RescueParams,
    continuous_turing_scan,
    prey_only_equilibrium,
    prey_only_invasion_threshold,
    require_single_equilibrium,
    turing_scan,
)


def test_zero_diffusion_pde_matches_independent_odes():
    params = RescueParams(n_x=6, L=1.0, delta_U=0.0, delta_D=0.0, delta_P=0.0, m=0.22)
    rng = np.random.default_rng(123)
    U0 = 0.15 + 0.05 * rng.random(params.n_x)
    D0 = 0.45 + 0.05 * rng.random(params.n_x)
    P0 = 0.10 + 0.03 * rng.random(params.n_x)

    pde = simulate_pde(params, T=2.0, initial_state=(U0, D0, P0), n_time=5)
    pde_final = pack_state(pde.U, pde.D, pde.P)

    ode_final = []
    for idx in range(params.n_x):
        ode = simulate_ode(params, T=2.0, y0=np.array([U0[idx], D0[idx], P0[idx]]), n_time=5)
        ode_final.append(ode.y[:, -1])
    ode_final = np.asarray(ode_final)
    ode_packed = pack_state(ode_final[:, 0], ode_final[:, 1], ode_final[:, 2])

    assert np.allclose(pde_final, ode_packed, atol=2.0e-5, rtol=2.0e-5)


def test_homogeneous_initial_condition_remains_homogeneous_with_equal_diffusion():
    params = RescueParams(n_x=20, L=2.0, delta_U=0.05, delta_D=0.05, delta_P=0.05)
    eq = require_single_equilibrium(params)
    pde = simulate_pde(params, T=3.0, equilibrium=eq, perturbation_amplitude=0.0, n_time=8)

    assert pde.diagnostics.var_U < 1.0e-12
    assert pde.diagnostics.var_D < 1.0e-12
    assert pde.diagnostics.var_P < 1.0e-12


def test_density_normalized_persistence_is_separate_from_total_biomass():
    params = RescueParams(n_x=11, L=10.0)
    x = grid_1d(params)
    U = np.full(params.n_x, 0.2)
    D = np.full(params.n_x, 0.3)
    P = np.full(params.n_x, 5.0e-5)

    diagnostics = compute_diagnostics(x, U, D, P, params, epsilon=1.0e-4)

    assert np.isclose(diagnostics.B_P, 5.0e-4)
    assert np.isclose(diagnostics.mean_P, 5.0e-5)
    assert diagnostics.persistent_total
    assert not diagnostics.persistent_mean
    assert diagnostics.persistent == diagnostics.persistent_total


def test_prey_only_invasion_threshold_matches_direct_ode_invasion():
    params = RescueParams()
    m_inv = prey_only_invasion_threshold(params)
    U0, D0, _ = prey_only_equilibrium(params)
    p0 = 1.0e-7

    below = simulate_ode(params.with_updates(m=0.95 * m_inv), T=20.0, y0=np.array([U0, D0, p0]))
    above = simulate_ode(params.with_updates(m=1.05 * m_inv), T=20.0, y0=np.array([U0, D0, p0]))

    assert below.y[2, -1] > p0
    assert above.y[2, -1] < p0


def test_turing_flag_requires_ode_stability_and_spatial_instability():
    params = RescueParams()
    eq = require_single_equilibrium(params)
    scan = turing_scan(params, eq, n_max=20)

    assert scan.turing_unstable == (scan.ode_stable and scan.has_unstable_spatial_mode)


def test_continuous_scan_matches_discrete_scan_on_same_k_grid():
    params = RescueParams()
    eq = require_single_equilibrium(params)
    n_max = 20
    discrete = turing_scan(params, eq, n_max=n_max)
    continuous = continuous_turing_scan(
        params,
        eq,
        k_min=np.pi / params.L,
        k_max=n_max * np.pi / params.L,
        n_k=n_max,
    )

    assert continuous.ode_stable == discrete.ode_stable
    assert np.allclose(continuous.ks, discrete.ks)
    assert np.allclose(continuous.max_real_by_k, discrete.max_real_by_mode)
    assert np.isclose(continuous.maximum_spatial_growth, discrete.dominant_growth)


def test_holling2_equilibrium_and_continuous_scan_are_finite():
    params = HollingIIParams(m=0.10, h=0.50)
    equilibria = solve_coexistence_equilibria_holling2(params)

    assert equilibria
    assert equilibria[0].residual < 1.0e-10

    scan = continuous_turing_scan_holling2(params, equilibria[0], k_min=1.0e-4, k_max=2.0, n_k=20)

    assert np.isfinite(scan.maximum_spatial_growth)
    assert np.isfinite(scan.k_at_maximum_growth)


def test_roy_style_jacobian_matches_finite_difference():
    params = RoyParams(mu=0.8)
    eq = roy_require_positive_equilibrium(params)
    y = np.array([eq.u, eq.v, eq.w], dtype=float)
    analytic = roy_reaction_jacobian(eq.u, eq.v, eq.w, params)
    numerical = np.empty_like(analytic)
    step = 1.0e-6
    for idx in range(3):
        perturbation = np.zeros(3)
        perturbation[idx] = step
        plus = roy_reaction_part(*(y + perturbation), params)
        minus = roy_reaction_part(*(y - perturbation), params)
        numerical[:, idx] = ((plus - minus) / (2.0 * step)).ravel()

    assert np.allclose(analytic, numerical, atol=1.0e-6, rtol=1.0e-5)


def test_roy_style_table_parameter_set_is_turing_unstable():
    params = RoyParams(mu=0.8)
    eq = roy_require_positive_equilibrium(params)
    scan = roy_continuous_turing_scan(params, eq, k_min=1.0e-4, k_max=8.0, n_k=240)

    assert scan.ode_stable
    assert scan.max_spatial_growth > 0.0
    assert scan.turing_unstable


def test_roy_2d_neumann_laplacian_constant_is_zero():
    values = np.full((8, 9), 3.2)
    lap = laplacian_neumann_2d(values, dx=0.5, dy=0.4)

    assert np.allclose(lap, 0.0)


def test_roy_2d_homogeneous_initial_state_stays_homogeneous_short_run():
    params = RoyParams(mu=0.85)
    eq = roy_require_positive_equilibrium(params)
    config = Roy2DConfig(n_x=12, n_y=10, L_x=3.0, L_y=2.5, T=0.1, dt=0.005, perturbation_amplitude=0.0)
    result = simulate_pde_2d(params, config, equilibrium=eq)

    assert result.diagnostics.var_u < 1.0e-12
    assert result.diagnostics.var_v < 1.0e-12
    assert result.diagnostics.var_w < 1.0e-12
    assert len(result.t) == len(result.mean_w_time)
    assert len(result.t) == len(result.var_u_time)
    assert len(result.t) == len(result.mean_u_time)
    assert len(result.t) == len(result.mean_v_time)
    assert len(result.t) == len(result.var_v_time)
    assert len(result.t) == len(result.var_w_time)
    assert len(result.t) == len(result.min_z_time)
    assert len(result.t) == len(result.dominant_wavelength_time)
    assert len(result.t) == len(result.dominant_power_time)
    assert np.all(np.isfinite(result.min_z_time))


def test_tail_metrics_constant_positive_series_is_persistent():
    t = np.linspace(0.0, 10.0, 21)
    y = np.full_like(t, 2.0e-3)

    metrics = tail_metrics(t, y)
    persistent, persistent_metrics = is_persistent_tail(t, y, epsilon=1.0e-4)

    assert np.isclose(metrics["tail_mean"], 2.0e-3)
    assert np.isclose(metrics["tail_slope"], 0.0)
    assert persistent
    assert persistent_metrics["tail_min"] > 1.0e-4


def test_tail_metrics_decaying_series_has_negative_slope():
    t = np.linspace(0.0, 10.0, 51)
    y = 1.0 - 0.05 * t

    metrics = tail_metrics(t, y)

    assert metrics["tail_slope"] < 0.0
    assert np.isclose(metrics["tail_slope"], -0.05)


def test_tail_persistence_rejects_declining_transient_survivor():
    t = np.linspace(0.0, 10.0, 101)
    y = np.linspace(1.0e-3, 1.1e-4, len(t))

    persistent, metrics = is_persistent_tail(t, y, epsilon=1.0e-4)

    assert y[-1] > 1.0e-4
    assert metrics["tail_mean"] > 1.0e-4
    assert metrics["tail_slope"] < metrics["tail_slope_floor"]
    assert not persistent


def test_tail_persistence_rejects_nonfinite_series():
    t = np.linspace(0.0, 10.0, 11)
    y = np.full_like(t, 1.0e-3)
    y[-2] = np.nan

    persistent, metrics = is_persistent_tail(t, y, epsilon=1.0e-4)

    assert not persistent
    assert np.isnan(metrics["tail_mean"])


def test_pde_tail_persistence_rejects_nonphysical_result():
    t = np.linspace(0.0, 1.0, 5)
    series = np.full_like(t, 1.0e-3)
    field = np.full((3, 3), 1.0e-3)
    diagnostics = Roy2DDiagnostics(
        mean_u=1.0e-3,
        mean_v=1.0e-3,
        mean_w=1.0e-3,
        var_u=0.0,
        var_v=0.0,
        var_w=0.0,
        min_value=-1.0e-6,
        min_z=1.0,
        negative_detected=True,
        z_negative_detected=False,
        dominant_k=0.0,
        dominant_wavelength=np.inf,
        dominant_power=0.0,
    )
    result = Roy2DResult(
        t=t,
        mean_u_time=series,
        mean_v_time=series,
        mean_w_time=series,
        var_u_time=np.zeros_like(t),
        var_v_time=np.zeros_like(t),
        var_w_time=np.zeros_like(t),
        min_z_time=np.ones_like(t),
        dominant_wavelength_time=np.full_like(t, np.nan),
        dominant_power_time=np.full_like(t, np.nan),
        x=np.arange(3),
        y=np.arange(3),
        u=field,
        v=field,
        w=field,
        diagnostics=diagnostics,
    )

    persistent, metrics = pde_tail_persistence(result, epsilon=1.0e-4)

    assert not persistent
    assert np.isnan(metrics["tail_mean"])


def test_threshold_search_synthetic_monotonic_classifier():
    def classify(stress: float):
        persistent = stress <= 0.42
        metrics = {
            "tail_mean": 1.0 - stress,
            "tail_min": 1.0 - stress,
            "tail_start": 0.0,
            "tail_end": 1.0,
            "tail_slope": 0.0,
            "tail_duration": 1.0,
        }
        return persistent, 1.0 - stress, metrics

    result = _threshold_search(classify, 0.0, 1.0, max_iter=12)

    assert result["status"] == "ok"
    assert result["threshold"] <= 0.42
    assert result["s_high"] - result["s_low"] <= 1.0 / 2.0**12


def test_threshold_bracket_helpers_validate_and_reject_invalid_brackets():
    assert np.allclose(make_refined_stress_bracket(0.40, 0.45, margin=0.05), (0.35, 0.5))

    valid = ensure_valid_threshold_bracket(lambda stress: stress < 0.5, 0.4, 0.6)
    invalid_low = ensure_valid_threshold_bracket(lambda _stress: False, 0.4, 0.6, max_expansions=1)
    invalid_high = ensure_valid_threshold_bracket(lambda _stress: True, 0.4, 0.6, max_expansions=1)

    assert valid[2] == "ok"
    assert invalid_low[2] == "invalid_bracket_low_not_persistent"
    assert invalid_high[2] == "invalid_bracket_high_persistent"


def test_group_summary_conclusion_rule():
    rescue = summarize_delta_group(np.array([0.0020, 0.0018, 0.0022]), np.array([0.0005, 0.0005, 0.0005]))
    inhibition = summarize_delta_group(np.array([-0.0020, -0.0018, -0.0022]), np.array([0.0005, 0.0005, 0.0005]))
    none = summarize_delta_group(np.array([0.0002, -0.0001, 0.0003]), np.array([0.0005, 0.0005, 0.0005]))

    assert rescue["conclusion"] == "rescue_supported"
    assert inhibition["conclusion"] == "inhibition_supported"
    assert none["conclusion"] == "no_measurable_effect"


def test_roy_evo_tradeoff_functions_are_monotone():
    params = RoyEvoParams()
    q_values = np.array([0.0, 0.5, 1.0])

    r_values = r_of_q(q_values, params)
    a_values = a_of_q(q_values, params)
    b_values = b_of_q(q_values, params)

    assert np.all(np.diff(r_values) < 0.0)
    assert np.all(np.diff(a_values) < 0.0)
    assert np.all(np.diff(b_values) < 0.0)


def test_roy_evo_selection_gradient_changes_with_predator_pressure():
    params = RoyEvoParams()

    low_w_gradient = selection_gradient(n=1.0, w=0.01, q=0.5, params=params)
    high_w_gradient = selection_gradient(n=1.0, w=3.0, q=0.5, params=params)

    assert low_w_gradient < 0.0
    assert high_w_gradient > 0.0


def test_roy_evo_q_remains_bounded_in_short_physical_run():
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    result = simulate_ode_evo(params, initial_state=np.array([4.8, 0.64, 0.67]), T=5.0, n_eval=50)

    assert result.success
    q = result.y[2]
    assert np.min(q) >= -1.0e-8
    assert np.max(q) <= 1.0 + 1.0e-8


def test_roy_evo_no_evolution_mode_freezes_q_derivative():
    params = RoyEvoParams()
    dydt = reaction_ode_evo(0.0, np.array([1.0, 0.5, 0.4]), params, stress=0.1, evolve=False)

    assert dydt[2] == 0.0


def test_roy_evo_classifier_detects_persistence_and_extinction():
    params = RoyEvoParams()
    t = np.linspace(0.0, 10.0, 51)
    persistent_y = np.vstack(
        [
            np.full_like(t, 1.0),
            np.full_like(t, 2.0e-3),
            np.full_like(t, 0.5),
        ]
    )
    extinct_y = np.vstack(
        [
            np.full_like(t, 1.0),
            np.linspace(2.0e-4, 1.0e-6, len(t)),
            np.full_like(t, 0.5),
        ]
    )

    persistent = classify_evo_trajectory(t, persistent_y, params=params)
    extinct = classify_evo_trajectory(t, extinct_y, params=params)

    assert persistent["persistent_predator"]
    assert not extinct["persistent_predator"]


def test_roy_evo_bisection_threshold_synthetic_monotonic_case():
    def classify(stress: float):
        return stress <= 0.37, {"tail_mean_w": 1.0 - stress}

    result = bisection_threshold(classify, 0.0, 1.0, tolerance=1.0e-4, max_iter=20)

    assert result["threshold_status"] == "ok"
    assert result["threshold"] <= 0.3701
    assert result["threshold_gap"] <= 1.0e-4


def test_roy_evo_homogeneous_pde_short_run_matches_ode():
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    y0 = np.array([4.8, 0.64, 0.67])
    config = RoyEvoPDEConfig(n_x=8, n_y=7, L_x=2.0, L_y=2.0, T=0.05, dt=0.001, record_every=10, perturbation_amplitude=0.0)
    initial = initial_state_from_ode_equilibrium(y0, config)

    pde = simulate_pde_evo_2d(params, config, initial, stress=0.02, evolve=True)
    ode = simulate_ode_evo(params, y0, stress=0.02, evolve=True, T=config.T, n_eval=2)

    assert np.allclose([pde.mean_n_time[-1], pde.mean_w_time[-1], pde.mean_q_time[-1]], ode.y[:, -1], atol=2.0e-4)


def test_roy_evo_pde_q_bounds_are_recorded_for_short_run():
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    y0 = np.array([4.8, 0.64, 0.67])
    config = RoyEvoPDEConfig(n_x=10, n_y=10, L_x=3.0, L_y=3.0, T=1.0, dt=0.02, record_every=10, seed=7)
    result = simulate_pde_evo_2d(params, config, initial_state_from_ode_equilibrium(y0, config), stress=0.02, evolve=True)
    diagnostics = classify_evo_pde_result(result, params)

    assert diagnostics["physical"]
    assert diagnostics["min_q"] >= -1.0e-6
    assert diagnostics["max_q"] <= 1.0 + 1.0e-6
    assert diagnostics["q_clip_max_violation"] <= 1.0e-4


def test_roy_evo_pde_no_evolution_keeps_mean_q_constant():
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    y0 = np.array([4.8, 0.64, 0.67])
    config = RoyEvoPDEConfig(n_x=10, n_y=10, L_x=3.0, L_y=3.0, T=1.0, dt=0.02, record_every=10, seed=8)
    result = simulate_pde_evo_2d(params, config, initial_state_from_ode_equilibrium(y0, config), stress=0.02, evolve=False)

    assert np.isclose(result.mean_q_time[0], result.mean_q_time[-1], atol=1.0e-12)


def test_roy_evo_pde_classifier_rejects_nonfinite_arrays():
    params = RoyEvoParams()
    t = np.array([0.0, 1.0])
    values = np.array([1.0, np.nan])
    result = RoyEvoPDEResult(
        t=t,
        mean_n_time=values,
        mean_w_time=np.ones_like(t),
        mean_q_time=np.full_like(t, 0.5),
        var_n_time=np.zeros_like(t),
        var_w_time=np.zeros_like(t),
        var_q_time=np.zeros_like(t),
        min_z_time=np.ones_like(t),
        n=np.ones((2, 2)),
        w=np.ones((2, 2)),
        q=np.full((2, 2), 0.5),
        diagnostics={"nonfinite_detected": True, "q_clip_count": 0, "q_clip_max_violation": 0.0},
    )

    diagnostics = classify_evo_pde_result(result, params)

    assert diagnostics["nonfinite_detected"]
    assert not diagnostics["persistent_predator"]


def test_roy_evo_pde_classifier_rejects_negative_free_space():
    params = RoyEvoParams()
    t = np.linspace(0.0, 1.0, 3)
    result = RoyEvoPDEResult(
        t=t,
        mean_n_time=np.full_like(t, 10.0),
        mean_w_time=np.full_like(t, 1.0e-3),
        mean_q_time=np.full_like(t, 0.5),
        var_n_time=np.zeros_like(t),
        var_w_time=np.zeros_like(t),
        var_q_time=np.zeros_like(t),
        min_z_time=np.full_like(t, -0.1),
        n=np.full((2, 2), 10.0),
        w=np.full((2, 2), 1.0e-3),
        q=np.full((2, 2), 0.5),
        diagnostics={
            "completed": True,
            "nonfinite_detected": False,
            "initial_mean_q": 0.5,
            "min_n": 10.0,
            "min_w": 1.0e-3,
            "min_q": 0.5,
            "max_q": 0.5,
            "min_z": -0.1,
            "q_clip_count": 0,
            "q_clip_max_violation": 0.0,
            "spatial_covariance_bonus_time": np.zeros_like(t),
        },
    )

    diagnostics = classify_evo_pde_result(result, params)

    assert not diagnostics["physical"]
    assert not diagnostics["persistent_predator"]


def test_roy_evo_spatial_mechanism_is_zero_for_homogeneous_fields():
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    n = np.full((4, 5), 4.8)
    w = np.full((4, 5), 0.64)
    q = np.full((4, 5), 0.67)

    diagnostics = spatial_mechanism_diagnostics(n, w, q, params, stress=0.02)

    assert abs(diagnostics["spatial_covariance_bonus"]) < 1.0e-14
    assert abs(diagnostics["cov_w_q"]) < 1.0e-14


def load_step12_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "12_roy_spatial_suppression_mechanism.py"
    spec = importlib.util.spec_from_file_location("step12_suppression", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step12_monotonicity_detects_clean_transition():
    step12 = load_step12_module()

    assert step12.is_clean_monotone_transition([True, True, False, False])
    assert not step12.is_clean_monotone_transition([True, False, True, False])
    assert not step12.is_clean_monotone_transition([True, True, True])


def test_step12_monotonicity_detects_reentry():
    step12 = load_step12_module()

    assert step12.has_persistence_reentry([True, False, True])
    assert step12.has_persistence_reentry([False, True, False, True])
    assert not step12.has_persistence_reentry([True, True, False, False])


def test_step12_relaxed_classifier_differs_for_slow_decline():
    step12 = load_step12_module()
    times = np.linspace(0.0, 100.0, 101)
    mean_w = np.linspace(1.0, 0.2, 101)
    mean_q = np.full_like(times, 0.5)
    var_q = np.zeros_like(times)
    min_z = np.ones_like(times)

    diagnostics = step12.classify_tail_series(
        times,
        mean_w,
        mean_q,
        var_q,
        min_z,
        initial_q=0.5,
        physical=True,
        tail_fraction=0.25,
    )

    assert diagnostics["persistent_without_slope_check"]
    assert not diagnostics["persistent_predator"]
    assert diagnostics["classification_note"] == "slope_check_rejects"


def test_step12_final_label_detects_tail_fraction_sensitivity():
    step12 = load_step12_module()
    rows = []
    for tail_fraction, persistent in [(0.25, True), (0.35, False), (0.50, False)]:
        rows.append(
            {
                "stress": 0.1,
                "T": 500.0,
                "tail_fraction": tail_fraction,
                "physical": True,
                "persistent_predator": persistent,
                "persistent_without_slope_check": persistent,
                "tail_mean_w": 1.0,
                "tail_min_w": 0.9,
                "tail_slope_w": 0.0,
                "tail_slope_floor_w": -0.01,
            }
        )

    label, _interpretation, flags = step12.decide_final_label(rows, (500.0,), (0.25, 0.35, 0.50))

    assert label == "pde_evo_threshold_classifier_sensitive"
    assert flags.tail_fraction_disagreements == 1


def test_step12_final_label_detects_nonmonotone_reentry():
    step12 = load_step12_module()
    rows = []
    for stress, persistent in [(0.1, True), (0.2, False), (0.3, True)]:
        rows.append(
            {
                "stress": stress,
                "T": 500.0,
                "tail_fraction": 0.25,
                "physical": True,
                "persistent_predator": persistent,
                "persistent_without_slope_check": persistent,
                "tail_mean_w": 1.0,
                "tail_min_w": 0.9,
                "tail_slope_w": 0.0,
                "tail_slope_floor_w": -0.01,
            }
        )

    label, _interpretation, flags = step12.decide_final_label(rows, (500.0,), (0.25,))

    assert label == "pde_evo_threshold_nonmonotone"
    assert flags.reentry_sequences == 1


def load_step13_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "13_roy_pde_evo_persistence_stability.py"
    spec = importlib.util.spec_from_file_location("step13_persistence", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step13_horizon_status_persistent_like_for_stable_tail():
    step13 = load_step13_module()
    metrics = step13.TailMetrics(
        physical=True,
        persistent_with_slope_rule=True,
        persistent_without_slope_rule=True,
        tail_mean_w=0.5,
        tail_min_w=0.49,
        tail_slope_w=0.0,
        tail_slope_floor_w=-0.01,
        tail_mean_q=0.4,
        tail_q_change=-0.2,
        tail_mean_var_q=0.0,
        tail_mean_min_z=1.0,
    )

    assert step13.classify_horizon_status(metrics) == "persistent_like"


def test_step13_horizon_status_extinct_like_for_near_zero_tail():
    step13 = load_step13_module()
    metrics = step13.TailMetrics(
        physical=True,
        persistent_with_slope_rule=False,
        persistent_without_slope_rule=False,
        tail_mean_w=1.0e-6,
        tail_min_w=5.0e-7,
        tail_slope_w=0.0,
        tail_slope_floor_w=-1.0e-6,
        tail_mean_q=0.1,
        tail_q_change=-0.5,
        tail_mean_var_q=0.0,
        tail_mean_min_z=1.0,
    )

    assert step13.classify_horizon_status(metrics) == "extinct_like"


def test_step13_horizon_status_declining_transient_for_slope_rejection():
    step13 = load_step13_module()
    metrics = step13.TailMetrics(
        physical=True,
        persistent_with_slope_rule=False,
        persistent_without_slope_rule=True,
        tail_mean_w=0.5,
        tail_min_w=0.4,
        tail_slope_w=-0.02,
        tail_slope_floor_w=-0.01,
        tail_mean_q=0.3,
        tail_q_change=-0.3,
        tail_mean_var_q=0.0,
        tail_mean_min_z=1.0,
    )

    assert step13.classify_horizon_status(metrics) == "declining_transient"


def step13_row(stress, horizon, tail_fraction, status, tail_mean_w, physical=True):
    return {
        "stress": stress,
        "T": horizon,
        "tail_fraction": tail_fraction,
        "physical": physical,
        "horizon_status": status,
        "tail_mean_w": tail_mean_w,
        "persistent_with_slope_rule": status in {"persistent_like", "recovery_transient"},
        "persistent_without_slope_rule": status not in {"extinct_like", "indeterminate"},
    }


def step13_rows_for_all_tails(stress, statuses, means):
    rows = []
    for tail_fraction in (0.25, 0.35, 0.50):
        for horizon, status, mean_w in zip((500.0, 800.0, 1200.0), statuses, means):
            rows.append(step13_row(stress, horizon, tail_fraction, status, mean_w))
    return rows


def test_step13_aggregate_status_persistent_confirmed_for_stable_latest_horizons():
    step13 = load_step13_module()
    rows = step13_rows_for_all_tails(
        0.09,
        ["persistent_like", "persistent_like", "persistent_like"],
        [0.63, 0.64, 0.641],
    )

    summary = step13.aggregate_stress_status(rows)

    assert summary["final_status"] == "persistent_confirmed"


def test_step13_aggregate_status_extinct_confirmed_for_two_extinct_latest_horizons():
    step13 = load_step13_module()
    rows = step13_rows_for_all_tails(
        0.175,
        ["declining_transient", "extinct_like", "extinct_like"],
        [0.1, 1.0e-6, 5.0e-7],
    )

    summary = step13.aggregate_stress_status(rows)

    assert summary["final_status"] == "extinct_confirmed"


def test_step13_aggregate_status_recovery_transient_after_earlier_decline():
    step13 = load_step13_module()
    rows = step13_rows_for_all_tails(
        0.15,
        ["extinct_like", "declining_transient", "persistent_like"],
        [1.0e-6, 0.10, 0.20],
    )

    summary = step13.aggregate_stress_status(rows)

    assert summary["final_status"] == "recovery_transient"


def test_step13_final_label_long_transients_dominate():
    step13 = load_step13_module()
    summary_rows = []
    detail_rows = []
    for idx, stress in enumerate(np.linspace(0.09, 0.18, 10)):
        status = "declining_transient" if idx < 4 else "indeterminate"
        summary_rows.append(
            {
                "stress": float(stress),
                "final_status": status,
                "physical_failure": False,
            }
        )
        latest_statuses = ["declining_transient", "declining_transient", "persistent_like"] if idx < 4 else ["persistent_like"] * 3
        for tail_fraction, horizon_status in zip((0.25, 0.35, 0.50), latest_statuses):
            detail_rows.append(
                {
                    "stress": float(stress),
                    "T": 1200.0,
                    "tail_fraction": tail_fraction,
                    "horizon_status": horizon_status,
                }
            )

    label, _interpretation = step13.final_step11_label(summary_rows, detail_rows, 1200.0)

    assert label == "pde_evo_long_transients_dominate"


def test_step13_final_label_boundary_nonmonotone():
    step13 = load_step13_module()
    summary_rows = [
        {"stress": 0.1, "final_status": "persistent_confirmed", "physical_failure": False},
        {"stress": 0.2, "final_status": "extinct_confirmed", "physical_failure": False},
        {"stress": 0.3, "final_status": "persistent_confirmed", "physical_failure": False},
    ]

    label, _interpretation = step13.final_step11_label(summary_rows, [], 1200.0)

    assert label == "pde_evo_boundary_nonmonotone"


def load_step15_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "15_roy_pde_evo_hysteresis_basins.py"
    spec = importlib.util.spec_from_file_location("step15_basins", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step15_regime_persistent_only():
    step15 = load_step15_module()

    regime = step15.aggregate_stress_regime(["persistent_basin", "persistent_basin"])

    assert regime == "persistent_only"


def test_step15_regime_extinct_only():
    step15 = load_step15_module()

    regime = step15.aggregate_stress_regime(["extinct_basin", "extinct_basin"])

    assert regime == "extinct_only"


def test_step15_regime_bistable_persistent_extinct():
    step15 = load_step15_module()

    regime = step15.aggregate_stress_regime(["persistent_basin", "extinct_basin", "transient_basin"])

    assert regime == "bistable_persistent_extinct"


def test_step15_regime_persistent_transient_mixed():
    step15 = load_step15_module()

    regime = step15.aggregate_stress_regime(["persistent_basin", "transient_basin", "unresolved_basin"])

    assert regime == "persistent_transient_mixed"


def test_step15_final_label_bistability_mapped():
    step15 = load_step15_module()

    label, _interpretation = step15.final_step13_label(
        {0.15: "bistable_persistent_extinct"},
        direction_dependent=True,
    )

    assert label == "pde_evo_bistability_mapped"


def test_step15_final_label_hysteresis_unresolved():
    step15 = load_step15_module()

    label, _interpretation = step15.final_step13_label(
        {0.15: "persistent_transient_mixed"},
        direction_dependent=True,
    )

    assert label == "pde_evo_hysteresis_confirmed_but_basins_unresolved"
