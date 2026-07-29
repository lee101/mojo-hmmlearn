"""Drop-in implementations of hmmlearn._hmmc's inference functions."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def _inputs(startprob, transmat, lattice):
    start = f64(startprob)
    trans = f64(transmat)
    frames = f64(lattice)
    if start.ndim != 1 or trans.ndim != 2 or frames.ndim != 2:
        raise ValueError("startprob must be 1D; transmat and frame probabilities must be 2D")
    n_samples, n_components = frames.shape
    if n_samples == 0 or n_components == 0:
        raise ValueError("frame probabilities must have non-zero dimensions")
    if start.shape != (n_components,) or trans.shape != (n_components, n_components):
        raise ValueError("shape mismatch")
    return start, trans, frames, n_samples, n_components


def _lattices(fwdlattice, transmat, bwdlattice, frameprob):
    fwd = f64(fwdlattice)
    trans = f64(transmat)
    bwd = f64(bwdlattice)
    frames = f64(frameprob)
    if fwd.ndim != 2 or trans.ndim != 2 or bwd.ndim != 2 or frames.ndim != 2:
        raise ValueError("all arguments must be 2D")
    n_samples, n_components = frames.shape
    if n_samples == 0 or n_components == 0:
        raise ValueError("frame probabilities must have non-zero dimensions")
    if (
        fwd.shape != frames.shape
        or bwd.shape != frames.shape
        or trans.shape != (n_components, n_components)
    ):
        raise ValueError("shape mismatch")
    return fwd, trans, bwd, frames, n_samples, n_components


def forward_scaling(startprob, transmat, frameprob):
    start, trans, frames, n_samples, n_components = _inputs(
        startprob, transmat, frameprob
    )
    fwd = np.empty_like(frames)
    scaling = np.empty(n_samples, dtype=np.float64)
    log_prob = np.empty(1, dtype=np.float64)
    ok = lib().mhl_forward_scaling(
        addr(start),
        addr(trans),
        addr(frames),
        addr(fwd),
        addr(scaling),
        addr(log_prob),
        n_samples,
        n_components,
    )
    if not ok:
        raise ValueError(
            "forward pass failed with underflow; "
            "consider using implementation='log' instead"
        )
    return float(log_prob[0]), fwd, scaling


def forward_log(startprob, transmat, log_frameprob):
    start, trans, frames, n_samples, n_components = _inputs(
        startprob, transmat, log_frameprob
    )
    fwd = np.empty_like(frames)
    log_trans = np.empty_like(trans)
    log_prob = lib().mhl_forward_log(
        addr(start),
        addr(trans),
        addr(frames),
        addr(fwd),
        addr(log_trans),
        n_samples,
        n_components,
    )
    return float(log_prob), fwd


def backward_scaling(startprob, transmat, frameprob, scaling_factors):
    start, trans, frames, n_samples, n_components = _inputs(
        startprob, transmat, frameprob
    )
    del start
    scaling = f64(scaling_factors)
    if scaling.ndim != 1 or scaling.shape != (n_samples,):
        raise ValueError("shape mismatch")
    bwd = np.empty_like(frames)
    lib().mhl_backward_scaling(
        addr(trans),
        addr(frames),
        addr(scaling),
        addr(bwd),
        n_samples,
        n_components,
    )
    return bwd


def backward_log(startprob, transmat, log_frameprob):
    start, trans, frames, n_samples, n_components = _inputs(
        startprob, transmat, log_frameprob
    )
    del start
    bwd = np.empty_like(frames)
    log_trans = np.empty_like(trans)
    lib().mhl_backward_log(
        addr(trans),
        addr(frames),
        addr(bwd),
        addr(log_trans),
        n_samples,
        n_components,
    )
    return bwd


def compute_scaling_xi_sum(fwdlattice, transmat, bwdlattice, frameprob):
    fwd, trans, bwd, frames, n_samples, n_components = _lattices(
        fwdlattice, transmat, bwdlattice, frameprob
    )
    xi = np.empty_like(trans)
    lib().mhl_compute_scaling_xi_sum(
        addr(fwd),
        addr(trans),
        addr(bwd),
        addr(frames),
        addr(xi),
        n_samples,
        n_components,
    )
    return xi


def compute_log_xi_sum(fwdlattice, transmat, bwdlattice, log_frameprob):
    fwd, trans, bwd, frames, n_samples, n_components = _lattices(
        fwdlattice, transmat, bwdlattice, log_frameprob
    )
    xi = np.empty_like(trans)
    log_trans = np.empty_like(trans)
    lib().mhl_compute_log_xi_sum(
        addr(fwd),
        addr(trans),
        addr(bwd),
        addr(frames),
        addr(xi),
        addr(log_trans),
        n_samples,
        n_components,
    )
    return xi


def viterbi(startprob, transmat, log_frameprob):
    start, trans, frames, n_samples, n_components = _inputs(
        startprob, transmat, log_frameprob
    )
    state_sequence = np.empty(n_samples, dtype=np.int64)
    lattice = np.empty_like(frames)
    log_trans = np.empty_like(trans)
    log_prob = lib().mhl_viterbi(
        addr(start),
        addr(trans),
        addr(frames),
        addr(state_sequence),
        addr(lattice),
        addr(log_trans),
        n_samples,
        n_components,
    )
    return float(log_prob), state_sequence


__all__ = [
    "backward_log",
    "backward_scaling",
    "compute_log_xi_sum",
    "compute_scaling_xi_sum",
    "forward_log",
    "forward_scaling",
    "viterbi",
]
