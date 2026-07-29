"""No-I/O tests for forward-only SOS preprocessing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.causal_preprocessing import (
    CausalBandpassFilter,
    causal_bandpass_filter,
    causal_event_guard_mask,
)


def test_full_and_arbitrary_chunk_partitions_are_numerically_identical():
    rng = np.random.default_rng(23)
    signal = rng.normal(size=(3, 257))
    full = causal_bandpass_filter(signal, 30_000)
    stream = CausalBandpassFilter(30_000, n_channels=3)
    partition = [1, 17, 1, 64, 3, 91, 80]
    pieces = []
    start = 0
    for width in partition:
        pieces.append(stream.process(signal[:, start : start + width]))
        start += width
    chunked = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(chunked, full, rtol=0.0, atol=0.0)
    assert stream.state is not None and stream.state.shape == (stream.sos.shape[0], 3, 2)


def test_future_perturbation_cannot_change_already_emitted_samples():
    rng = np.random.default_rng(7)
    prefix = rng.normal(size=(2, 80))
    future_a = rng.normal(size=(2, 120))
    future_b = future_a.copy()
    future_b[:, :] += rng.normal(scale=100.0, size=future_b.shape)
    out_a = causal_bandpass_filter(np.concatenate([prefix, future_a], axis=1), 30_000)
    out_b = causal_bandpass_filter(np.concatenate([prefix, future_b], axis=1), 30_000)
    np.testing.assert_array_equal(out_a[:, : prefix.shape[1]], out_b[:, : prefix.shape[1]])


def test_reset_restores_zero_state_and_repeats_initial_response():
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(2, 64))
    stream = CausalBandpassFilter(30_000, n_channels=2)
    first = stream.process(signal)
    stream.process(rng.normal(size=(2, 12)))
    stream.reset()
    again = stream.process(signal)
    np.testing.assert_array_equal(first, again)
    stream.reset(3)
    assert stream.n_channels == 3 and stream.state is not None and stream.state.shape[1] == 3


def test_constant_and_impulse_outputs_remain_finite():
    constant = np.ones((2, 1000))
    impulse = np.zeros((2, 1000))
    impulse[0, 0] = 1.0
    for values in (constant, impulse):
        result = causal_bandpass_filter(values, 30_000)
        assert result.shape == values.shape
        assert np.all(np.isfinite(result))


def test_event_guard_boundaries_are_explicit():
    times = np.array([9, 10, 94, 95, 99, 100], dtype=np.int64)
    got = causal_event_guard_mask(times, warmup_samples=10, start_sample=0, stop_sample=100, post_guard_samples=5)
    np.testing.assert_array_equal(got, [False, True, True, False, False, False])
    np.testing.assert_array_equal(causal_event_guard_mask(np.array([5, 6]), warmup_samples=5, start_sample=1), [False, True])


def test_invalid_filter_and_channel_inputs_raise_clear_errors():
    for kwargs in (
        {"low": 0.0}, {"high": 15_000.0}, {"low": 6000.0, "high": 300.0}, {"order": 0},
    ):
        with np.testing.assert_raises(ValueError):
            CausalBandpassFilter(30_000, **kwargs)
    with np.testing.assert_raises(ValueError):
        CausalBandpassFilter(30_000, n_channels=0)
    stream = CausalBandpassFilter(30_000, n_channels=2)
    with np.testing.assert_raises(ValueError):
        stream.process(np.ones(10))
    with np.testing.assert_raises(ValueError):
        stream.process(np.ones((3, 10)))
    with np.testing.assert_raises(ValueError):
        stream.process(np.array([[np.nan]]))
    with np.testing.assert_raises(ValueError):
        causal_event_guard_mask(np.array([1.0]))
    with np.testing.assert_raises(ValueError):
        causal_event_guard_mask(np.array([1], dtype=np.int64), warmup_samples=-1)
