"""Benchmark Mojo kernels against hmmlearn 0.3.3 on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from hmmlearn import _hmmc as upstream  # noqa: E402
from mojo_hmmlearn import _hmmc as mojo  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def inputs(n_samples, n_components, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.dirichlet(np.ones(n_components))
    trans = rng.dirichlet(np.ones(n_components), size=n_components)
    frames = rng.uniform(0.01, 1.0, size=(n_samples, n_components))
    return start, trans, frames, np.log(frames)


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def forward_backward_scaling(module, start, trans, frames):
    log_prob, fwd, scaling = module.forward_scaling(start, trans, frames)
    bwd = module.backward_scaling(start, trans, frames, scaling)
    return log_prob, fwd, bwd


def forward_backward_log(module, start, trans, log_frames):
    log_prob, fwd = module.forward_log(start, trans, log_frames)
    bwd = module.backward_log(start, trans, log_frames)
    return log_prob, fwd, bwd


def main():
    small = inputs(250_000, 4)
    wide = inputs(25_000, 16, seed=1)

    cases = [
        (
            "forward scaling, T=250k K=4",
            lambda: mojo.forward_scaling(*small[:3]),
            lambda: upstream.forward_scaling(*small[:3]),
        ),
        (
            "forward log, T=250k K=4",
            lambda: mojo.forward_log(small[0], small[1], small[3]),
            lambda: upstream.forward_log(small[0], small[1], small[3]),
        ),
        (
            "Viterbi, T=250k K=4",
            lambda: mojo.viterbi(small[0], small[1], small[3]),
            lambda: upstream.viterbi(small[0], small[1], small[3]),
        ),
        (
            "forward-backward scaling, T=250k K=4",
            lambda: forward_backward_scaling(mojo, *small[:3]),
            lambda: forward_backward_scaling(upstream, *small[:3]),
        ),
        (
            "forward-backward log, T=250k K=4",
            lambda: forward_backward_log(mojo, small[0], small[1], small[3]),
            lambda: forward_backward_log(upstream, small[0], small[1], small[3]),
        ),
        (
            "forward scaling, T=25k K=16",
            lambda: mojo.forward_scaling(*wide[:3]),
            lambda: upstream.forward_scaling(*wide[:3]),
        ),
        (
            "forward log, T=25k K=16",
            lambda: mojo.forward_log(wide[0], wide[1], wide[3]),
            lambda: upstream.forward_log(wide[0], wide[1], wide[3]),
        ),
        (
            "Viterbi, T=25k K=16",
            lambda: mojo.viterbi(wide[0], wide[1], wide[3]),
            lambda: upstream.viterbi(wide[0], wide[1], wide[3]),
        ),
        (
            "forward-backward scaling, T=25k K=16",
            lambda: forward_backward_scaling(mojo, *wide[:3]),
            lambda: forward_backward_scaling(upstream, *wide[:3]),
        ),
        (
            "forward-backward log, T=25k K=16",
            lambda: forward_backward_log(mojo, wide[0], wide[1], wide[3]),
            lambda: forward_backward_log(upstream, wide[0], wide[1], wide[3]),
        ),
    ]

    cases[0][1]()
    cases[0][2]()
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| Kernel | Mojo | hmmlearn | Speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        mojo_time = timeit(ours)
        upstream_time = timeit(theirs)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
