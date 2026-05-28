import numpy as np

from src.simulate_pde_1d import pack_state, simulate_ode, simulate_pde, unpack_state
from src.turing_rescue_model import (
    RescueParams,
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

