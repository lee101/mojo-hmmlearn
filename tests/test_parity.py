"""Numerical and behavioural parity with hmmlearn 0.3.3."""

from itertools import product

import numpy as np
import pytest
from hmmlearn import _hmmc as upstream

from mojo_hmmlearn import _hmmc
from mojo_hmmlearn._lib import addr, f64


def model(n_samples=37, n_components=5, seed=0):
    rng = np.random.default_rng(seed)
    startprob = rng.dirichlet(np.ones(n_components))
    transmat = rng.dirichlet(np.ones(n_components), size=n_components)
    frameprob = rng.uniform(0.001, 1.0, size=(n_samples, n_components))
    return startprob, transmat, frameprob


@pytest.mark.parametrize(
    "n_samples,n_components", [(1, 1), (1, 7), (2, 2), (37, 5), (103, 9)]
)
def test_forward_log_matches_upstream(n_samples, n_components):
    start, trans, frames = model(n_samples, n_components, seed=n_components)
    expected = upstream.forward_log(start, trans, np.log(frames))
    actual = _hmmc.forward_log(start, trans, np.log(frames))
    assert actual[0] == pytest.approx(expected[0], abs=2e-8)
    assert actual[1] == pytest.approx(expected[1], abs=2e-8)


@pytest.mark.parametrize(
    "n_samples,n_components", [(1, 1), (1, 7), (2, 2), (37, 5), (103, 9)]
)
def test_forward_scaling_matches_upstream(n_samples, n_components):
    start, trans, frames = model(n_samples, n_components, seed=10 + n_components)
    expected = upstream.forward_scaling(start, trans, frames)
    actual = _hmmc.forward_scaling(start, trans, frames)
    assert actual[0] == pytest.approx(expected[0], abs=2e-8)
    assert actual[1] == pytest.approx(expected[1], abs=2e-13)
    assert actual[2] == pytest.approx(expected[2], abs=2e-13)


@pytest.mark.parametrize("n_components", [4, 8])
def test_backward_log_matches_upstream(n_components):
    start, trans, frames = model(83, n_components)
    log_frames = np.log(frames)
    assert _hmmc.backward_log(start, trans, log_frames) == pytest.approx(
        upstream.backward_log(start, trans, log_frames), abs=2e-8
    )


def test_backward_scaling_matches_upstream():
    start, trans, frames = model(83, 8)
    _, _, scaling = upstream.forward_scaling(start, trans, frames)
    assert _hmmc.backward_scaling(start, trans, frames, scaling) == pytest.approx(
        upstream.backward_scaling(start, trans, frames, scaling), abs=2e-13
    )


@pytest.mark.parametrize("n_components", [3, 6])
def test_simd_tail_paths_match_upstream(n_components):
    start, trans, frames = model(41, n_components, seed=100 + n_components)
    log_frames = np.log(frames)

    expected_log_prob, expected_fwd_log = upstream.forward_log(
        start, trans, log_frames
    )
    log_prob, fwd_log = _hmmc.forward_log(start, trans, log_frames)
    assert log_prob == pytest.approx(expected_log_prob, abs=2e-8)
    assert fwd_log == pytest.approx(expected_fwd_log, abs=2e-8)
    assert _hmmc.backward_log(start, trans, log_frames) == pytest.approx(
        upstream.backward_log(start, trans, log_frames), abs=2e-8
    )

    expected_scaling = upstream.forward_scaling(start, trans, frames)
    scaling = _hmmc.forward_scaling(start, trans, frames)
    assert scaling[0] == pytest.approx(expected_scaling[0], abs=2e-8)
    assert scaling[1] == pytest.approx(expected_scaling[1], abs=2e-13)
    assert scaling[2] == pytest.approx(expected_scaling[2], abs=2e-13)
    assert _hmmc.backward_scaling(start, trans, frames, scaling[2]) == pytest.approx(
        upstream.backward_scaling(start, trans, frames, expected_scaling[2]),
        abs=2e-13,
    )

    expected_score, expected_path = upstream.viterbi(start, trans, log_frames)
    score, path = _hmmc.viterbi(start, trans, log_frames)
    assert score == pytest.approx(expected_score, abs=2e-8)
    assert np.array_equal(path, expected_path)


def test_log_xi_sum_matches_upstream():
    start, trans, frames = model(61, 7)
    log_frames = np.log(frames)
    _, ours_fwd = _hmmc.forward_log(start, trans, log_frames)
    ours_bwd = _hmmc.backward_log(start, trans, log_frames)
    _, ref_fwd = upstream.forward_log(start, trans, log_frames)
    ref_bwd = upstream.backward_log(start, trans, log_frames)
    assert _hmmc.compute_log_xi_sum(
        ours_fwd, trans, ours_bwd, log_frames
    ) == pytest.approx(
        upstream.compute_log_xi_sum(ref_fwd, trans, ref_bwd, log_frames),
        abs=3e-8,
    )


def test_scaling_xi_sum_matches_upstream():
    start, trans, frames = model(61, 7)
    _, ours_fwd, scaling = _hmmc.forward_scaling(start, trans, frames)
    ours_bwd = _hmmc.backward_scaling(start, trans, frames, scaling)
    _, ref_fwd, ref_scaling = upstream.forward_scaling(start, trans, frames)
    ref_bwd = upstream.backward_scaling(start, trans, frames, ref_scaling)
    assert _hmmc.compute_scaling_xi_sum(
        ours_fwd, trans, ours_bwd, frames
    ) == pytest.approx(
        upstream.compute_scaling_xi_sum(ref_fwd, trans, ref_bwd, frames),
        abs=3e-12,
    )


@pytest.mark.parametrize("n_samples,n_components", [(1, 1), (1, 6), (2, 3), (91, 8)])
def test_viterbi_matches_upstream(n_samples, n_components):
    start, trans, frames = model(n_samples, n_components, seed=21 + n_components)
    expected_score, expected_path = upstream.viterbi(start, trans, np.log(frames))
    score, path = _hmmc.viterbi(start, trans, np.log(frames))
    assert score == pytest.approx(expected_score, abs=2e-8)
    assert np.array_equal(path, expected_path)
    assert path.dtype == np.int64


def test_zero_probabilities_match_upstream():
    start = np.array([1.0, 0.0, 0.0])
    trans = np.array([[0.7, 0.3, 0.0], [0.0, 0.6, 0.4], [0.2, 0.0, 0.8]])
    frames = np.array(
        [[0.9, 0.0, 0.2], [0.1, 0.8, 0.0], [0.0, 0.2, 0.9], [0.3, 0.1, 0.7]]
    )
    with np.errstate(divide="ignore"):
        log_frames = np.log(frames)
    for name in ("forward_log", "backward_log", "viterbi"):
        actual = getattr(_hmmc, name)(start, trans, log_frames)
        expected = getattr(upstream, name)(start, trans, log_frames)
        if isinstance(actual, tuple):
            assert actual[0] == pytest.approx(expected[0], abs=2e-8)
            if name == "viterbi":
                assert np.array_equal(actual[1], expected[1])
            else:
                assert actual[1] == pytest.approx(expected[1], abs=2e-8)
        else:
            assert actual == pytest.approx(expected, abs=2e-8)


def test_viterbi_tie_breaking_matches_upstream():
    start = np.array([0.5, 0.5])
    trans = np.full((2, 2), 0.5)
    log_frames = np.zeros((5, 2))
    expected = upstream.viterbi(start, trans, log_frames)
    actual = _hmmc.viterbi(start, trans, log_frames)
    assert actual[0] == pytest.approx(expected[0], abs=2e-8)
    assert np.array_equal(actual[1], expected[1])


def test_classic_weather_example_and_brute_force():
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    frames = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
    score, path = _hmmc.viterbi(start, trans, np.log(frames))

    candidates = []
    for states in product(range(2), repeat=3):
        probability = start[states[0]] * frames[0, states[0]]
        for t in range(1, 3):
            probability *= trans[states[t - 1], states[t]] * frames[t, states[t]]
        candidates.append((probability, states))
    probability, states = max(candidates)
    assert score == pytest.approx(np.log(probability), abs=2e-8)
    assert path.tolist() == list(states) == [0, 0, 1]

    total_score, _ = _hmmc.forward_log(start, trans, np.log(frames))
    assert np.exp(total_score) == pytest.approx(sum(value for value, _ in candidates))
    assert np.exp(total_score) == pytest.approx(0.03628)


def test_log_and_scaling_posteriors_agree():
    start, trans, frames = model(50, 6)
    log_score, log_fwd = _hmmc.forward_log(start, trans, np.log(frames))
    log_bwd = _hmmc.backward_log(start, trans, np.log(frames))
    scaling_score, scaling_fwd, scaling = _hmmc.forward_scaling(start, trans, frames)
    scaling_bwd = _hmmc.backward_scaling(start, trans, frames, scaling)

    log_posterior = np.exp(log_fwd + log_bwd - log_score)
    scaled_posterior = scaling_fwd * scaling_bwd
    scaled_posterior /= scaled_posterior.sum(axis=1, keepdims=True)
    assert log_score == pytest.approx(scaling_score, abs=2e-8)
    assert log_posterior == pytest.approx(scaled_posterior, abs=2e-8)
    assert log_posterior.sum(axis=1) == pytest.approx(np.ones(50), abs=2e-8)


def test_array_like_float32_and_strided_inputs():
    start, trans, frames = model(30, 4)
    padded = np.zeros((30, 8), dtype=np.float32)
    padded[:, ::2] = frames.astype(np.float32)
    strided = padded[:, ::2]
    actual = _hmmc.forward_scaling(
        start.astype(np.float32).tolist(), trans.astype(np.float32), strided
    )
    expected = upstream.forward_scaling(
        start.astype(np.float32).tolist(), trans.astype(np.float32), strided
    )
    for left, right in zip(actual, expected):
        assert left == pytest.approx(right, abs=2e-8)


def test_ffi_conversion_makes_native_contiguous_float64_buffers():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    converted = f64(source)
    assert converted.dtype == np.float64
    assert converted.flags.c_contiguous
    assert converted.shape == source.shape
    assert addr(converted) == converted.ctypes.data != 0


@pytest.mark.parametrize(
    "value,error",
    [
        (np.empty(0, dtype=np.float64), ValueError),
        (np.arange(6, dtype=np.float64)[::2], ValueError),
        (np.ones(1, dtype=np.float32), TypeError),
    ],
)
def test_ffi_rejects_buffers_that_violate_abi(value, error):
    with pytest.raises(error):
        addr(value)


def test_single_frame_xi_sums_have_upstream_identity_values():
    start, trans, frames = model(1, 4)
    _, fwd_log = _hmmc.forward_log(start, trans, np.log(frames))
    bwd_log = _hmmc.backward_log(start, trans, np.log(frames))
    _, fwd_scaling, scaling = _hmmc.forward_scaling(start, trans, frames)
    bwd_scaling = _hmmc.backward_scaling(start, trans, frames, scaling)
    assert np.all(
        np.isneginf(_hmmc.compute_log_xi_sum(fwd_log, trans, bwd_log, np.log(frames)))
    )
    assert np.array_equal(
        _hmmc.compute_scaling_xi_sum(fwd_scaling, trans, bwd_scaling, frames),
        np.zeros((4, 4)),
    )


def test_scaling_underflow_raises_upstream_compatible_error():
    start = np.array([0.5, 0.5])
    trans = np.full((2, 2), 0.5)
    frames = np.full((3, 2), 1e-310)
    with pytest.raises(ValueError, match="forward pass failed with underflow"):
        _hmmc.forward_scaling(start, trans, frames)
    with pytest.raises(ValueError, match="forward pass failed with underflow"):
        upstream.forward_scaling(start, trans, frames)


@pytest.mark.parametrize(
    "function,args",
    [
        (_hmmc.forward_log, ([1.0], [[1.0]], np.empty((0, 1)))),
        (_hmmc.forward_scaling, ([1.0, 0.0], [[1.0]], [[1.0, 1.0]])),
        (_hmmc.backward_log, ([1.0], [[1.0]], [0.0])),
        (
            _hmmc.compute_log_xi_sum,
            (np.ones((2, 2)), np.eye(2), np.ones((3, 2)), np.ones((2, 2))),
        ),
    ],
)
def test_invalid_shapes_raise_value_error(function, args):
    with pytest.raises(ValueError):
        function(*args)
