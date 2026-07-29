# mojo-hmmlearn

`mojo-hmmlearn` is a standalone Mojo port of the compute-bound inference
kernels in [hmmlearn](https://github.com/hmmlearn/hmmlearn). It provides the
same function names, positional signatures, return values, shapes, and NumPy
dtypes as `hmmlearn._hmmc` for the covered subset.

The Python package is named `mojo_hmmlearn`, so it can be installed beside the
real `hmmlearn` package for parity testing or gradual adoption:

```python
from mojo_hmmlearn import _hmmc
```

## Coverage

The complete forward-backward and Viterbi surface from `hmmlearn._hmmc` is
implemented:

| Function | Purpose |
|---|---|
| `forward_log(startprob, transmat, log_frameprob)` | Log-domain forward pass |
| `backward_log(startprob, transmat, log_frameprob)` | Log-domain backward pass |
| `compute_log_xi_sum(fwdlattice, transmat, bwdlattice, log_frameprob)` | Log transition posteriors |
| `forward_scaling(startprob, transmat, frameprob)` | Scaled-probability forward pass |
| `backward_scaling(startprob, transmat, frameprob, scaling_factors)` | Scaled-probability backward pass |
| `compute_scaling_xi_sum(fwdlattice, transmat, bwdlattice, frameprob)` | Scaled transition posteriors |
| `viterbi(startprob, transmat, log_frameprob)` | Most likely state sequence |

Both implementations accept NumPy arrays, array-like inputs, non-contiguous
views, and values convertible to `float64`. Viterbi returns an `int64` state
sequence, matching upstream.

This project does not port hmmlearn's model and emission classes
(`GaussianHMM`, `GMMHMM`, `CategoricalHMM`, `MultinomialHMM`, and
`PoissonHMM`), Baum-Welch control loop, parameter initialization and updates,
sampling, or variational inference. Applications compute frame likelihoods as
usual and call these inference kernels.

## Install

The reproducible development environment includes Mojo, NumPy, pytest, and
the real `hmmlearn==0.3.3`:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-hmmlearn.so`. Run the parity suite with:

```bash
pixi run test
```

For a conventional editable Python install after building:

```bash
pixi run python -m pip install -e .
```

## Usage

This is a complete forward-backward pass for a two-state HMM:

```python
import numpy as np
from mojo_hmmlearn import _hmmc

startprob = np.array([0.6, 0.4])
transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
frameprob = np.array([
    [0.5, 0.1],
    [0.4, 0.3],
    [0.1, 0.6],
])

log_prob, fwd = _hmmc.forward_log(
    startprob, transmat, np.log(frameprob)
)
bwd = _hmmc.backward_log(startprob, transmat, np.log(frameprob))
posteriors = np.exp(fwd + bwd - log_prob)

path_log_prob, states = _hmmc.viterbi(
    startprob, transmat, np.log(frameprob)
)

print(round(log_prob, 6))       # -3.316489
print(states.tolist())          # [0, 0, 1]
print(posteriors.sum(axis=1))   # [1. 1. 1.]
```

Use the scaling functions when frame probabilities can be represented safely
in ordinary probability space. They avoid the repeated `exp` and `log`
operations of log-sum-exp.

## Benchmarks

These are real best-of-five wall-clock results from `pixi run bench`. Input
construction, shared-library loading, and the warm-up call are outside the
timed region. A speedup below `1.00x` means Mojo is slower.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux 6.8.0-136-generic.

| Kernel | Mojo | hmmlearn | Speedup |
|---|---:|---:|---:|
| forward scaling, T=250k K=4 | 8.70 ms | 10.67 ms | 1.23x |
| forward log, T=250k K=4 | 64.99 ms | 66.02 ms | 1.02x |
| Viterbi, T=250k K=4 | 6.63 ms | 8.12 ms | 1.23x |
| forward-backward scaling, T=250k K=4 | 37.12 ms | 34.52 ms | 0.93x |
| forward-backward log, T=250k K=4 | 165.47 ms | 172.78 ms | 1.04x |
| forward scaling, T=25k K=16 | 4.91 ms | 15.15 ms | 3.08x |
| forward log, T=25k K=16 | 74.39 ms | 140.65 ms | 1.89x |
| Viterbi, T=25k K=16 | 7.76 ms | 9.42 ms | 1.21x |
| forward-backward scaling, T=25k K=16 | 7.56 ms | 23.47 ms | 3.10x |
| forward-backward log, T=25k K=16 | 129.16 ms | 162.88 ms | 1.26x |

The four-state combined scaling pass is slower than hmmlearn in this run; the
other measured cases are faster. Timings vary with machine load; the table is
the verbatim output of the final locked benchmark run.

Run the benchmark on your machine with:

```bash
pixi run bench
```

## How it works

All kernels are in one Mojo compilation unit and compiled to a shared library.
The Python layer uses `ctypes`; each NumPy buffer crosses the C ABI as a 64-bit
integer address. Mojo reconstructs a mutable `UnsafePointer` inside each
non-parametric exported function. There are no Python objects or allocations
inside the kernels.

Matrices use C-contiguous row-major `float64` storage:
`frameprob[T, K]`, `transmat[K, K]`, and forward/backward lattices `[T, K]`.
NumPy owns every input, output, and scratch buffer, so ownership never crosses
the FFI boundary. The wrapper copies only when dtype or contiguity requires it.
Transition logarithms and Viterbi's dynamic-programming lattice are explicit
scratch arrays allocated once per call.

Forward-log transitions are transposed once into scratch storage so each
log-sum-exp reduction reads contiguous memory. Forward and backward reductions
use the native `float64` SIMD width, with scalar tails for component counts
that are not a multiple of that width.

There is no threaded or GPU path. Each time step depends on the preceding
step, so the implementation keeps execution and all buffers on the CPU.

Numerical behavior follows hmmlearn 0.3.3: log-domain routines use stable
log-sum-exp, zero probabilities become negative infinity, scaling rejects a
forward row sum below `1e-300`, and Viterbi reproduces upstream's tie-breaking.

## License

MIT. See [LICENSE](LICENSE).
