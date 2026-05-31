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


def load_step17_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "17_roy_pde_evo_basin_boundary_scan.py"
    spec = importlib.util.spec_from_file_location("step17_boundary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step17_basin_label_persistent_steady():
    step17 = load_step17_module()

    assert step17.basin_label_from_classification("persistent_steady") == "persistent_basin"


def test_step17_basin_label_extinct_steady():
    step17 = load_step17_module()

    assert step17.basin_label_from_classification("extinct_steady") == "extinct_basin"


def test_step17_regime_bistable_when_persistent_and_extinct_positive():
    step17 = load_step17_module()

    regime = step17.stress_regime_from_counts(
        persistent_count=2,
        extinct_count=1,
        transient_count=4,
        unresolved_count=0,
        nonphysical_count=0,
    )

    assert regime == "bistable_persistent_extinct"


def test_step17_regime_persistent_transient_mixed_without_extinct():
    step17 = load_step17_module()

    regime = step17.stress_regime_from_counts(
        persistent_count=2,
        extinct_count=0,
        transient_count=5,
        unresolved_count=0,
        nonphysical_count=0,
    )

    assert regime == "persistent_transient_mixed"


def test_step17_final_label_mapped_for_two_bistable_stresses():
    step17 = load_step17_module()

    label, _interpretation = step17.final_step15_label(
        {
            0.1584375: "bistable_persistent_extinct",
            0.16486816: "bistable_persistent_extinct",
        }
    )

    assert label == "basin_boundary_mapped"


def test_step17_final_label_partially_mapped_for_one_bistable_stress():
    step17 = load_step17_module()

    label, _interpretation = step17.final_step15_label(
        {
            0.1584375: "bistable_persistent_extinct",
            0.16486816: "mostly_transient_or_unresolved",
        }
    )

    assert label == "basin_boundary_partially_mapped"


def load_step19_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "19_roy_pde_evo_representative_solutions.py"
    spec = importlib.util.spec_from_file_location("step19_representatives", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step19_select_representative_cases_from_synthetic_rows():
    step19 = load_step19_module()
    rows = [
        {
            "stress": "0.1584375",
            "q0": "0.1",
            "w0_scale": "0.1",
            "classification": "persistent_steady",
            "basin_label": "persistent_basin",
            "relative_change_between_last_windows": "0.0",
        },
        {
            "stress": "0.16486816",
            "q0": "0.9",
            "w0_scale": "0.05",
            "classification": "extinct_steady",
            "basin_label": "extinct_basin",
            "relative_change_between_last_windows": "0.0",
        },
        {
            "stress": "0.1584375",
            "q0": "0.7",
            "w0_scale": "0.02",
            "classification": "recovery_transient",
            "basin_label": "transient_basin",
            "relative_change_between_last_windows": "10.0",
        },
    ]

    selected = step19.select_representative_cases(rows)

    assert selected["persistent_case"]["basin_label"] == "persistent_basin"
    assert selected["extinct_case"]["basin_label"] == "extinct_basin"
    assert selected["transient_case"]["basin_label"] == "transient_basin"


def test_step19_snapshot_times_include_first_and_final():
    step19 = load_step19_module()

    times = step19.snapshot_times_for_horizon(1600.0)

    assert np.isclose(times[0], 0.0)
    assert np.isclose(times[-1], 1600.0)
    assert len(times) == 5


def test_step19_residual_helper_returns_finite_values_for_homogeneous_state():
    step19 = load_step19_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    config = RoyEvoPDEConfig(n_x=4, n_y=3, L_x=2.0, L_y=2.0, D_n=0.01, D_w=0.01, D_q=0.005)
    n = np.full((config.n_y, config.n_x), 4.8)
    w = np.full((config.n_y, config.n_x), 0.64)
    q = np.full((config.n_y, config.n_x), 0.67)

    residual = step19.pde_evo_rhs_residual(n, w, q, params, config, stress=0.02)

    assert np.isfinite(residual["rhs_norm"])
    assert np.isfinite(residual["state_norm"])
    assert np.isfinite(residual["normalized_residual"])


def test_step19_basin_label_mapping():
    step19 = load_step19_module()

    assert step19.basin_label_from_classification("persistent_steady") == "persistent_basin"
    assert step19.basin_label_from_classification("extinct_steady") == "extinct_basin"
    assert step19.basin_label_from_classification("declining_transient") == "transient_basin"


def load_step20_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "20_homogeneous_vs_spatial_mechanism.py"
    spec = importlib.util.spec_from_file_location("step20_homogeneous_vs_spatial", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step20_comparison_metrics_are_zero_for_identical_series():
    step20 = load_step20_module()
    values = np.array([0.0, 1.0, 2.0, 3.0])

    metrics = step20.comparison_metrics(values, values.copy())

    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_difference"] == 0.0
    assert metrics["final_abs_difference"] == 0.0


def test_step20_basin_agreement_summary_fraction_on_synthetic_rows():
    step20 = load_step20_module()
    rows = [
        {"stress": 0.1, "ode_basin_label": "persistent_basin", "pde_basin_label": "persistent_basin", "labels_agree": True},
        {"stress": 0.1, "ode_basin_label": "extinct_basin", "pde_basin_label": "persistent_basin", "labels_agree": False},
        {"stress": 0.2, "ode_basin_label": "transient_basin", "pde_basin_label": "transient_basin", "labels_agree": True},
    ]

    summary = step20.basin_agreement_summary(rows)

    assert summary["total"] == 3
    assert summary["agreement_count"] == 2
    assert np.isclose(summary["agreement_fraction"], 2.0 / 3.0)
    assert np.isclose(summary["by_stress"][0.1]["agreement_fraction"], 0.5)


def test_step20_disagreement_type_detects_persistent_extinct_conflict():
    step20 = load_step20_module()

    dtype = step20.disagreement_type("persistent_basin", "extinct_basin")

    assert dtype == "ode_persistent_pde_extinct"
    assert step20.disagreement_is_direct_persistent_extinct(dtype)


def test_step20_disagreement_type_detects_transient_involved_case():
    step20 = load_step20_module()

    dtype = step20.disagreement_type("persistent_basin", "transient_basin")

    assert dtype == "ode_persistent_pde_transient"
    assert step20.disagreement_involves_transient(dtype)


def test_step20_decision_rule_returns_reaction_dominated_for_high_agreement_low_cv():
    step20 = load_step20_module()
    evidence = {
        "representative_ode_pde_agreement_count": 3,
        "representative_ode_pde_total": 3,
        "basin_grid_agreement_fraction": 0.95,
        "max_final_cv_n_steady": 1.0e-7,
        "max_final_cv_w_steady": 1.0e-7,
        "max_final_cv_q_steady": 1.0e-7,
        "max_final_cv_w": 1.0e-6,
        "max_final_cv_q": 1.0e-6,
        "perturbation_steady_outcome_change_count": 0,
        "perturbation_outcome_change_count": 0,
        "representative_steady_disagreement_count": 0,
        "physical_issue_count": 0,
        "disagreement_count": 10,
        "transient_involved_disagreement_count": 10,
        "direct_persistent_extinct_disagreement_count": 0,
    }

    label, _interpretation = step20.decide_final_label(evidence)

    assert label == "reaction_dominated_homogeneous_multistability"


def test_step20_decision_rule_returns_spatial_for_low_agreement_high_cv():
    step20 = load_step20_module()
    evidence = {
        "representative_ode_pde_agreement_count": 1,
        "representative_ode_pde_total": 3,
        "basin_grid_agreement_fraction": 0.4,
        "max_final_cv_n_steady": 5.0e-3,
        "max_final_cv_w_steady": 2.0e-2,
        "max_final_cv_q_steady": 2.0e-2,
        "max_final_cv_w": 2.0e-2,
        "max_final_cv_q": 2.0e-2,
        "perturbation_steady_outcome_change_count": 1,
        "perturbation_outcome_change_count": 1,
        "representative_steady_disagreement_count": 1,
        "physical_issue_count": 0,
        "disagreement_count": 20,
        "transient_involved_disagreement_count": 20,
        "direct_persistent_extinct_disagreement_count": 0,
    }

    label, _interpretation = step20.decide_final_label(evidence)

    assert label == "spatially_mediated_bistability"


def test_step20_decision_rule_returns_mixed_for_many_direct_basin_conflicts():
    step20 = load_step20_module()
    evidence = {
        "representative_ode_pde_agreement_count": 3,
        "representative_ode_pde_total": 3,
        "basin_grid_agreement_fraction": 0.9,
        "max_final_cv_n_steady": 1.0e-7,
        "max_final_cv_w_steady": 1.0e-7,
        "max_final_cv_q_steady": 1.0e-7,
        "max_final_cv_w": 1.0e-6,
        "max_final_cv_q": 1.0e-6,
        "perturbation_steady_outcome_change_count": 0,
        "perturbation_outcome_change_count": 0,
        "representative_steady_disagreement_count": 0,
        "physical_issue_count": 0,
        "disagreement_count": 8,
        "transient_involved_disagreement_count": 0,
        "direct_persistent_extinct_disagreement_count": 4,
        "basin_grid_total": 140,
    }

    label, _interpretation = step20.decide_final_label(evidence)

    assert label == "mixed_homogeneous_and_spatial_effects"


def test_step20_decision_rule_keeps_reaction_label_for_transient_disagreements():
    step20 = load_step20_module()
    evidence = {
        "representative_ode_pde_agreement_count": 3,
        "representative_ode_pde_total": 3,
        "basin_grid_agreement_fraction": 0.9,
        "max_final_cv_n_steady": 1.0e-7,
        "max_final_cv_w_steady": 1.0e-7,
        "max_final_cv_q_steady": 1.0e-7,
        "max_final_cv_w": 1.0e-6,
        "max_final_cv_q": 1.0e-6,
        "perturbation_steady_outcome_change_count": 0,
        "perturbation_outcome_change_count": 0,
        "representative_steady_disagreement_count": 0,
        "physical_issue_count": 0,
        "disagreement_count": 14,
        "transient_involved_disagreement_count": 14,
        "direct_persistent_extinct_disagreement_count": 0,
        "basin_grid_total": 140,
    }

    label, _interpretation = step20.decide_final_label(evidence)

    assert label == "reaction_dominated_homogeneous_multistability"


def test_step20_perturbation_grouping_detects_outcome_changes():
    step20 = load_step20_module()
    rows = [
        {"case_label": "persistent_case", "classification": "persistent_steady", "basin_label": "persistent_basin"},
        {"case_label": "persistent_case", "classification": "persistent_steady", "basin_label": "persistent_basin"},
        {"case_label": "extinct_case", "classification": "extinct_steady", "basin_label": "extinct_basin"},
        {"case_label": "extinct_case", "classification": "recovery_transient", "basin_label": "transient_basin"},
    ]

    summary = step20.detect_perturbation_outcome_changes(rows)

    assert not summary["persistent_case"]["classification_changed"]
    assert summary["extinct_case"]["classification_changed"]
    assert summary["extinct_case"]["basin_label_changed"]


def load_step21_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "21_roy_ode_homogeneous_mechanism.py"
    spec = importlib.util.spec_from_file_location("step21_ode_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step21_finite_difference_jacobian_shape():
    step21 = load_step21_module()

    def func(x):
        return np.array([x[0] + x[1], x[0] * x[1], x[2] ** 2], dtype=float)

    jacobian = step21.finite_difference_jacobian(func, np.array([1.0, 2.0, 3.0]))

    assert jacobian.shape == (3, 3)


def test_step21_equilibrium_deduplication_merges_close_equilibria():
    step21 = load_step21_module()
    equilibria = [
        {"n_star": 1.0, "w_star": 0.2, "q_star": 0.3, "residual_norm": 1.0e-8},
        {"n_star": 1.0 + 1.0e-7, "w_star": 0.2 - 1.0e-7, "q_star": 0.3, "residual_norm": 1.0e-7},
        {"n_star": 2.0, "w_star": 0.1, "q_star": 0.8, "residual_norm": 1.0e-8},
    ]

    unique = step21.deduplicate_equilibria(equilibria, tol=1.0e-5)

    assert len(unique) == 2


def test_step21_basin_label_mapping():
    step21 = load_step21_module()

    assert step21.basin_label_from_classification("persistent_steady") == "persistent_basin"
    assert step21.basin_label_from_classification("extinct_steady") == "extinct_basin"
    assert step21.basin_label_from_classification("recovery_transient") == "transient_basin"
    assert step21.basin_label_from_classification("declining_transient") == "transient_basin"


def test_step21_mechanism_summary_label_supported_for_stable_basin_structure():
    step21 = load_step21_module()
    basin_counts = {
        0.1584375: {
            "persistent_basin": 3,
            "extinct_basin": 2,
            "transient_basin": 1,
        }
    }
    representative_classes = {"persistent_steady", "extinct_steady", "recovery_transient"}

    label = step21.mechanism_label_from_inputs(
        basin_counts=basin_counts,
        ode_pde_agreement_fraction=0.9,
        representative_classes=representative_classes,
        stable_persistent_equilibria=1,
        stable_extinct_equilibria=1,
    )

    assert label == "ode_homogeneous_basin_structure_supported"


def test_step21_selection_gradient_helper_matches_formula():
    step21 = load_step21_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    n = 1.0
    w = 0.5
    q = 0.4
    z = 1.0 / params.kappa - n - w
    expected = (params.r_v - params.r_u) * z - (params.a_v - params.a_u) * w

    gradient = step21.selection_gradient_value(n=n, w=w, q=q, params=params)

    assert np.isclose(gradient, expected)


def load_step22_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "22_roy_ode_compensation_robustness.py"
    spec = importlib.util.spec_from_file_location("step22_ode_compensation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step22_analytic_branch_current_matches_known_stress_zero_values():
    step22 = load_step22_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)

    branch = step22.analytic_compensation_branch(params, 0.0)

    assert branch.interior_exists
    assert np.isclose(branch.n_star, 4.833333333333334)
    assert np.isclose(branch.w_star, 0.6416666666666666)
    assert np.isclose(branch.q_star, 0.672614741580259)


def test_step22_branch_existence_rejects_invalid_selection_ratio():
    step22 = load_step22_module()
    params = RoyEvoParams(r_v=1.1, a_v=0.35, b_u=0.08, b_v=0.02)

    branch = step22.analytic_compensation_branch(params, 0.0)

    assert not branch.interior_exists
    assert branch.existence_failure_reason == "invalid_selection_ratio"


def test_step22_interval_is_positive_for_current_parameterization():
    step22 = load_step22_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)

    interval = step22.compensation_interval(params)

    assert interval.valid
    assert interval.interior_stress_interval_length > 0.0
    assert interval.nonnegative_interval_length > 0.0


def test_step22_branch_comparison_detects_small_delta_for_current_numerical_equilibria():
    step22 = load_step22_module()
    rows = step22.read_csv(step22.NUMERICAL_EQUILIBRIA_CSV)
    branch = step22.analytic_compensation_branch(step22.PARAMS, 0.1584375)

    numerical = step22.current_numerical_stable_branch(rows, 0.1584375, branch)

    assert numerical is not None
    assert abs(branch.q_star - float(numerical["q_star"])) < 1.0e-10


def test_step22_final_label_rule_synthetic_inputs():
    step22 = load_step22_module()

    supported = step22.decide_final_label(
        analytic_matches_current=True,
        max_abs_delta_q=1.0e-8,
        branch_present_fraction=0.5,
        basin_maps_completed=True,
        equilibria_completed=True,
    )
    sensitive = step22.decide_final_label(
        analytic_matches_current=True,
        max_abs_delta_q=1.0e-8,
        branch_present_fraction=0.2,
        basin_maps_completed=True,
        equilibria_completed=True,
    )
    unresolved = step22.decide_final_label(
        analytic_matches_current=False,
        max_abs_delta_q=1.0e-8,
        branch_present_fraction=0.5,
        basin_maps_completed=True,
        equilibria_completed=True,
    )

    assert supported == "ode_compensation_branch_supported"
    assert sensitive == "ode_compensation_branch_parameter_sensitive"
    assert unresolved == "ode_compensation_branch_unresolved"


def load_step23_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "23_roy_ode_compensation_conditions.py"
    spec = importlib.util.spec_from_file_location("step23_ode_compensation_conditions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step23_current_geometry_matches_known_values():
    step23 = load_step23_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)

    geometry = step23.analytic_compensation_geometry(params)

    assert geometry["valid_geometry"]
    assert np.isclose(geometry["z_star"], 1.1916666666666667)
    assert np.isclose(geometry["w_star"], 0.6416666666666666)
    assert np.isclose(geometry["n_star"], 4.833333333333334)


def test_step23_current_q_star_stress_zero_matches_known_value():
    step23 = load_step23_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)

    q_star = step23.analytic_q_star(params, 0.0)

    assert np.isclose(q_star, 0.672614741580259)


def test_step23_current_q_star_decreases_with_stress():
    step23 = load_step23_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)

    q_low = step23.analytic_q_star(params, 0.0)
    q_high = step23.analytic_q_star(params, 0.175)

    assert q_high < q_low


def test_step23_current_stress_interval_contains_target_stresses():
    step23 = load_step23_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)

    interval = step23.stress_interval_for_q_in_unit_interval(params)

    assert interval["valid_interval"]
    assert interval["interior_stress_interval_low"] < 0.1584375 < interval["interior_stress_interval_high"]
    assert interval["interior_stress_interval_low"] < 0.16486816 < interval["interior_stress_interval_high"]


def test_step23_analytic_jacobian_shape_and_finite_entries():
    step23 = load_step23_module()
    params = RoyEvoParams(b_u=0.08, b_v=0.02)
    conditions = step23.compensation_existence_conditions(params, 0.1584375)

    jacobian = step23.ode_rhs_jacobian(
        conditions["n_star"],
        conditions["w_star"],
        conditions["q_star"],
        0.1584375,
        params,
    )

    assert jacobian.shape == (3, 3)
    assert np.all(np.isfinite(jacobian))


def test_step23_final_label_supported_for_satisfied_conditions():
    step23 = load_step23_module()

    label = step23.decide_final_label(
        conditions_all_satisfied=True,
        analytic_matches_current=True,
        stable_all_targets=True,
        valid_stable_grid_count=1,
    )

    assert label == "compensation_conditions_derived_and_supported"


def load_step24_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "24_roy_ode_compensation_routh_hurwitz.py"
    spec = importlib.util.spec_from_file_location("step24_ode_routh_hurwitz", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step24_characteristic_coefficients_for_stable_diagonal_matrix():
    step24 = load_step24_module()
    jacobian = np.diag([-1.0, -2.0, -3.0])

    coeffs = step24.characteristic_coefficients(jacobian)

    assert np.isclose(coeffs["A1"], 6.0)
    assert np.isclose(coeffs["A2"], 11.0)
    assert np.isclose(coeffs["A3"], 6.0)


def test_step24_routh_hurwitz_check_accepts_stable_cubic():
    step24 = load_step24_module()

    check = step24.routh_hurwitz_check(A1=6.0, A2=11.0, A3=6.0)

    assert check["routh_hurwitz_stable"]


def test_step24_routh_hurwitz_check_rejects_negative_a3():
    step24 = load_step24_module()

    check = step24.routh_hurwitz_check(A1=6.0, A2=11.0, A3=-6.0)

    assert not check["routh_hurwitz_stable"]
    assert not check["rh_A3_positive"]


def test_step24_coefficients_agree_with_numpy_poly():
    step24 = load_step24_module()
    jacobian = np.array(
        [
            [-1.0, 0.2, 0.0],
            [0.1, -2.0, 0.3],
            [0.0, -0.4, -3.0],
        ],
        dtype=float,
    )

    coeffs = step24.characteristic_coefficients(jacobian)
    poly = np.poly(jacobian)

    assert np.allclose([1.0, coeffs["A1"], coeffs["A2"], coeffs["A3"]], poly)


def test_step24_final_label_supported_for_stable_current_and_no_disagreements():
    step24 = load_step24_module()

    label = step24.decide_final_label(
        current_total=6,
        current_rh_stable=6,
        current_rh_eigenvalue_agreement=True,
        grid_stable_fraction=0.25,
        grid_disagreement_count=0,
    )

    assert label == "routh_hurwitz_conditions_supported"


def load_step25_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "25_roy_pde_spatial_stability_and_nonhomogeneous_tests.py"
    spec = importlib.util.spec_from_file_location("step25_pde_spatial_stability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step25_neumann_eigenvalue_zero_mode_is_zero():
    step25 = load_step25_module()

    assert np.isclose(step25.neumann_eigenvalue(0, 0, L_x=20.0, L_y=20.0), 0.0)


def test_step25_neumann_eigenvalue_first_x_mode_matches_formula():
    step25 = load_step25_module()
    L_x = 20.0

    assert np.isclose(step25.neumann_eigenvalue(1, 0, L_x=L_x, L_y=20.0), (np.pi / L_x) ** 2)


def test_step25_modal_matrix_zero_mode_equals_ode_jacobian():
    step25 = load_step25_module()
    jacobian = np.array(
        [
            [-1.0, 0.2, 0.0],
            [0.1, -2.0, 0.3],
            [0.0, -0.4, -3.0],
        ],
        dtype=float,
    )

    modal = step25.modal_matrix(jacobian, lambda_mn=0.0, diffusion=(0.01, 0.02, 0.03))

    assert np.allclose(modal, jacobian)


def test_step25_modal_matrix_zero_diffusion_equals_ode_jacobian_for_any_mode():
    step25 = load_step25_module()
    jacobian = np.array(
        [
            [-1.0, 0.2, 0.0],
            [0.1, -2.0, 0.3],
            [0.0, -0.4, -3.0],
        ],
        dtype=float,
    )

    modal = step25.modal_matrix(jacobian, lambda_mn=5.0, diffusion=(0.0, 0.0, 0.0))

    assert np.allclose(modal, jacobian)


def test_step25_spatial_stability_label_detects_stable_growths():
    step25 = load_step25_module()

    label = step25.spatial_stability_label_from_growths(np.array([-0.1, -0.01, -1.0e-5]))

    assert label == "linearly_spatially_stable"


def test_step25_spatial_stability_label_detects_positive_growth():
    step25 = load_step25_module()

    label = step25.spatial_stability_label_from_growths(np.array([-0.1, 1.0e-4]))

    assert label == "linear_spatial_instability_detected"


def test_step25_basin_changed_detects_nonhomogeneous_basin_switch():
    step25 = load_step25_module()

    assert step25.basin_changed("transient_basin", "persistent_basin")
    assert not step25.basin_changed("persistent_basin", "persistent_basin")


def test_step25_cv_helper_returns_finite_value_and_detects_decay():
    step25 = load_step25_module()
    field = np.array([[1.0, 1.1], [0.9, 1.0]], dtype=float)

    cv = step25.coefficient_of_variation(field)

    assert np.isfinite(cv)
    assert cv > 0.0
    assert step25.cv_decay_below_threshold(initial_cv=0.2, final_cv=5.0e-4, threshold=1.0e-3)


def load_step26_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "26_roy_pde_nonhomogeneous_long_horizon_followup.py"
    spec = importlib.util.spec_from_file_location("step26_pde_long_horizon_followup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step26_select_followup_cases_uses_only_basin_changes():
    step26 = load_step26_module()
    rows = [
        {
            "case_label": "changed",
            "stress": "0.1584375",
            "baseline_state": "basin_boundary_state",
            "perturbation_type": "local_defense_patch",
            "seed": "20260702",
            "basin_changed_relative_to_homogeneous_control": "True",
        },
        {
            "case_label": "unchanged",
            "stress": "0.1584375",
            "baseline_state": "basin_boundary_state",
            "perturbation_type": "local_predator_patch",
            "seed": "20260702",
            "basin_changed_relative_to_homogeneous_control": "False",
        },
    ]

    cases = step26.select_followup_cases(rows)

    assert [case.case_label for case in cases] == ["changed"]


def test_step26_decision_resolves_when_all_longest_rows_match_control():
    step26 = load_step26_module()
    rows = [
        {"T": 6400.0, "resolved_relative_to_pr23": "resolved_to_control"},
        {"T": 6400.0, "resolved_relative_to_pr23": "resolved_to_control"},
    ]

    assert step26.decide_final_label(rows) == "nonhomogeneous_basin_changes_resolve_to_homogeneous_control"


def test_step26_decision_detects_persistent_difference_without_pattern():
    step26 = load_step26_module()
    rows = [
        {"T": 6400.0, "resolved_relative_to_pr23": "resolved_to_control"},
        {"T": 6400.0, "resolved_relative_to_pr23": "persistent_basin_change"},
    ]

    assert step26.decide_final_label(rows) == "nonhomogeneous_basin_changes_persist_without_spatial_pattern"


def test_step26_decision_detects_persistent_difference_with_pattern():
    step26 = load_step26_module()
    rows = [
        {"T": 6400.0, "resolved_relative_to_pr23": "persistent_spatial_pattern"},
    ]

    assert step26.decide_final_label(rows) == "nonhomogeneous_basin_changes_persist_with_spatial_pattern"


def test_step26_cv_helper_handles_homogeneous_fields():
    step26 = load_step26_module()
    field = np.full((4, 5), 2.0)

    cv = step26.coefficient_of_variation(field)

    assert np.isfinite(cv)
    assert np.isclose(cv, 0.0)


def load_step27_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "27_roy_nonlinear_tradeoff_compensation.py"
    spec = importlib.util.spec_from_file_location("step27_nonlinear_tradeoff", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step27_shape_function_linear_returns_q():
    step27 = load_step27_module()
    q = np.array([0.1, 0.5, 0.9])

    assert np.allclose(step27.shape_function(q, 1.0), q)


def test_step27_generalized_linear_branch_recovers_reference():
    step27 = load_step27_module()
    stress = 0.1584375

    generalized = step27.find_branch_state_at_stress(step27.PARAMS, 1.0, 1.0, 1.0, stress)
    reference = step27.linear_reference_branch(step27.PARAMS, stress)

    assert generalized["branch_found"]
    assert np.isclose(generalized["q_star"], reference["q"], atol=1.0e-5)
    assert np.isclose(generalized["n_star"], reference["n"], atol=1.0e-5)
    assert np.isclose(generalized["w_star"], reference["w"], atol=1.0e-5)


def test_step27_branch_state_finder_matches_linear_q_at_rescue_stress():
    step27 = load_step27_module()
    stress = 0.1584375

    branch = step27.find_branch_state_at_stress(step27.PARAMS, 1.0, 1.0, 1.0, stress)

    assert branch["branch_found"]
    assert np.isclose(branch["q_star"], 0.21421875, atol=1.0e-4)


def test_step27_finite_difference_jacobian_is_finite_3x3():
    step27 = load_step27_module()
    state = np.array([4.8333333, 0.6416667, 0.21421875])

    jacobian = step27.generalized_jacobian_fd(state, step27.PARAMS, 1.0, 1.0, 1.0, 0.1584375)

    assert jacobian.shape == (3, 3)
    assert np.all(np.isfinite(jacobian))


def test_step27_basin_label_mapping_for_synthetic_classifications():
    step27 = load_step27_module()

    assert step27.basin_label_from_classification("persistent_steady") == "persistent_basin"
    assert step27.basin_label_from_classification("extinct_steady") == "extinct_basin"
    assert step27.basin_label_from_classification("declining_transient") == "transient_basin"


def test_step27_final_decision_synthetic_labels():
    step27 = load_step27_module()

    supported = step27.decide_final_label(
        linear_branch_recovered=True,
        has_concave_stable=True,
        has_convex_stable=True,
        pde_spatial_instability_count=0,
        persistent_pattern_count=0,
    )
    sensitive = step27.decide_final_label(
        linear_branch_recovered=True,
        has_concave_stable=True,
        has_convex_stable=False,
        pde_spatial_instability_count=0,
        persistent_pattern_count=0,
    )
    unresolved = step27.decide_final_label(
        linear_branch_recovered=False,
        has_concave_stable=True,
        has_convex_stable=True,
        pde_spatial_instability_count=0,
        persistent_pattern_count=0,
    )

    assert supported == "nonlinear_tradeoff_compensation_supported"
    assert sensitive == "nonlinear_tradeoff_compensation_parameter_sensitive"
    assert unresolved == "nonlinear_tradeoff_compensation_unresolved"


def test_step27_pde_spatial_helper_detects_synthetic_instability():
    step27 = load_step27_module()

    assert step27.pde_spatial_stability_detects_instability(np.array([-0.1, 1.0e-4]))
    assert not step27.pde_spatial_stability_detects_instability(np.array([-0.1, -1.0e-4]))


def test_step27_cv_helper_handles_homogeneous_fields():
    step27 = load_step27_module()
    field = np.full((3, 4), 5.0)

    cv = step27.coefficient_of_variation(field)

    assert np.isfinite(cv)
    assert np.isclose(cv, 0.0)
