"""Mojo implementations of hmmlearn's inference kernels."""

from . import _hmmc
from ._hmmc import (
    backward_log,
    backward_scaling,
    compute_log_xi_sum,
    compute_scaling_xi_sum,
    forward_log,
    forward_scaling,
    viterbi,
)

__version__ = "0.1.0"

__all__ = [
    "_hmmc",
    "backward_log",
    "backward_scaling",
    "compute_log_xi_sum",
    "compute_scaling_xi_sum",
    "forward_log",
    "forward_scaling",
    "viterbi",
]
