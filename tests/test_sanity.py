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
