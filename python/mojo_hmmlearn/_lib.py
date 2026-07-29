"""ctypes bindings for the Mojo HMM kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_HMMLEARN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-hmmlearn.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mhl_forward_scaling": ([I] * 8, I),
    "mhl_forward_log": ([I] * 7, F),
    "mhl_backward_scaling": ([I] * 6, None),
    "mhl_backward_log": ([I] * 6, None),
    "mhl_compute_scaling_xi_sum": ([I] * 7, None),
    "mhl_compute_log_xi_sum": ([I] * 8, None),
    "mhl_viterbi": ([I] * 8, F),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "hmmlearn.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run through pixi or set MOJO_HMMLEARN_LIB")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(value) -> np.ndarray:
    """Return an owning-or-borrowed buffer satisfying the Mojo C ABI."""
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    """Return a checked address for a synchronous FFI call.

    Callers retain ``value`` in a local variable for the duration of the ctypes
    call, so NumPy continues to own the allocation while Mojo borrows it.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype != np.dtype(np.float64) and value.dtype != np.dtype(np.int64):
        raise TypeError("FFI buffers must use float64 or int64 storage")
    if not value.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if value.size == 0 or value.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(value.ctypes.data)
