"""Contract tests for deformable spatial templates (no recording I/O)."""

import numpy as np

from Spatial.algorithms.deformable_spatial_template import (
    DeformableSpatialTemplate,
    StateEnables,
    closed_form_scales,
    dyn_penalty,
    event_peak_channel,
    geometry_k_table,
    gather_posneg,
    interpolation_matrix,
    RestState,
    shift_query_xy,
    spatial_observables,
)


def _line_geom(n: int = 8, pitch: float = 20.0) -> np.ndarray:
    g = np.zeros((n, 2), dtype=np.float64)
    g[:, 1] = np.arange(n) * pitch
    return g


def test_geometry_table_is_physical_not_amplitude():
    g = _line_geom(8)
    table, rel = geometry_k_table(g, k=5)
    # Home 3: five nearest are channels 1..5, ordered by y offset.
    np.testing.assert_array_equal(table[3], [1, 2, 3, 4, 5])
    np.testing.assert_allclose(rel[3, 2], 0.0)


def test_k16_on_16_site_probe_uses_every_electrode():
    g = _line_geom(16)
    table, _ = geometry_k_table(g, k=16)
    assert table.shape == (16, 16)
    np.testing.assert_array_equal(np.sort(table[0]), np.arange(16))


def test_shift_zero_is_identity():
    xy = np.array([[0.0, i * 20.0] for i in range(5)])
    live = np.ones(5, dtype=bool)
    a = interpolation_matrix(xy, xy, live)
    np.testing.assert_allclose(a, np.eye(5), atol=1e-8)


def test_fractional_shift_interpolates_between_neighbors():
    xy = np.array([[0.0, i * 20.0] for i in range(5)])
    live = np.ones(5, dtype=bool)
    dst = shift_query_xy(xy, delta_pitch=0.5, pitch_um=20.0, axis=1)
    a = interpolation_matrix(xy, dst, live)
    mu = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    out = a @ mu
    # Sampling the field 0.5 pitch toward +y: slot 2 looks toward slot 1? 
    # query = xy - 0.5*pitch*e_y, so slot 2 at y=40 queries y=30, between slots 1 (20) and 2 (40).
    assert out[2] > 4.0
    assert out[2] < 10.0


def test_closed_form_gain_recovers_global_scale():
    mu_p = np.array([1.0, 2.0, 0.0])
    mu_n = np.array([0.5, 0.0, 1.0])
    live = np.array([True, True, True])
    ap, an = closed_form_scales(1.5 * mu_p, 1.5 * mu_n, mu_p, mu_n, live, gain=True, polarity=False)
    np.testing.assert_allclose(ap, 1.5, rtol=1e-6)
    np.testing.assert_allclose(an, 1.5, rtol=1e-6)
    ap, an = closed_form_scales(2.0 * mu_p, 0.5 * mu_n, mu_p, mu_n, live, gain=True, polarity=True)
    np.testing.assert_allclose(ap, 2.0, rtol=1e-6)
    np.testing.assert_allclose(an, 0.5, rtol=1e-6)


def test_dyn_penalty_grows_with_offset_jump():
    rest = RestState(delta=0.2, sig_delta=0.2)
    small = dyn_penalty(0.0, 0.24, 0.0, np.zeros(0), rest)
    large = dyn_penalty(0.0, 0.9, 0.0, np.zeros(0), rest)
    assert large > small


def test_candidate_home_gather_keeps_shifted_mass():
    # 5 channels, mass moved from ch2 to a mix of ch2/ch3.
    vmax = np.array([0.0, 0.0, 4.0, 2.0, 0.0])
    vmin = np.zeros(5)
    table, _ = geometry_k_table(_line_geom(5), k=3)
    pos_u2, _, _ = gather_posneg(vmax, vmin, table[2])
    pos_u3, _, _ = gather_posneg(vmax, vmin, table[3])
    # Home 2 sees ch 1,2,3; home 3 sees ch 2,3,4 — same mass, different slots.
    assert np.max(pos_u2) == np.max(pos_u3) == 4.0
    assert not np.allclose(pos_u2, pos_u3)


def test_event_peak_channel_is_argmax_of_pos_or_neg():
    vmax = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    vmin = np.array([[0.0, 0.0, 0.0], [0.0, -9.0, 0.0]])
    np.testing.assert_array_equal(event_peak_channel(vmax, vmin), [0, 1])


def test_fit_mean_is_identity_and_gain_arm_fits_scale():
    g = _line_geom(6)
    rng = np.random.default_rng(0)
    n = 40
    vmax = np.zeros((n, 6))
    vmin = np.zeros((n, 6))
    # Unit 0 lives at ch 2 with a slow gain.
    for i in range(20):
        a = 1.0 + 0.01 * i
        vmax[i, 2] = 8.0 * a
        vmin[i, 2] = -2.0 * a
    for i in range(20, 40):
        vmax[i, 4] = 7.0
        vmin[i, 4] = -3.0
    labels = np.array([0] * 20 + [1] * 20)
    homes = event_peak_channel(vmax, vmin)
    model = DeformableSpatialTemplate(g, k=5, enables=StateEnables(gain=True)).fit(
        vmax, vmin, labels, homes
    )
    rec = model.scores_for_event(vmax[19], vmin[19])
    assert int(model.units_[np.argmin(rec["score"])]) == 0
    batch = model.score_matrix(vmax[18:20], vmin[18:20], homes[18:20], frame="candidate_home")
    np.testing.assert_allclose(batch["score"][1], rec["score"], rtol=1e-5, atol=1e-5)


def test_spatial_observables_polarity_and_centroid():
    pos = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
    neg = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
    g = _line_geom(5)
    _, rel = geometry_k_table(g, 5)
    live = np.ones(5, dtype=bool)
    obs = spatial_observables(pos, neg, rel[2], live, axis=1)
    assert obs["c_pos"] > obs["c_neg"]
    assert obs["rho"] > 0.0
