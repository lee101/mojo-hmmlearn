"""Forward-backward and Viterbi kernels for finite-state HMMs."""

from std.math import exp, log
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def infinity() -> Float64:
    var zero = 0.0
    return 1.0 / zero


def safe_log(value: Float64) -> Float64:
    if value == 0.0:
        return -infinity()
    return log(value)


def logsum_transition(
    previous: FPtr, log_transmat: FPtr, state: Int, n_components: Int
) -> Float64:
    var transition = log_transmat + state * n_components
    var maximum = -infinity()
    if n_components == 4:
        var value0 = previous[0] + transition[0]
        var value1 = previous[1] + transition[1]
        var value2 = previous[2] + transition[2]
        var value3 = previous[3] + transition[3]
        maximum = max(max(value0, value1), max(value2, value3))
        if maximum == -infinity() or maximum == infinity():
            return maximum
        var total = (
            exp(value0 - maximum)
            + exp(value1 - maximum)
            + exp(value2 - maximum)
            + exp(value3 - maximum)
        )
        return maximum + log(total)
    var vector_end = n_components - n_components % W
    for source in range(0, vector_end, W):
        var chunk_max = (
            previous.load[width=W](source)
            + transition.load[width=W](source)
        ).reduce_max()
        if chunk_max > maximum:
            maximum = chunk_max
    for source in range(vector_end, n_components):
        var value = previous[source] + transition[source]
        if value > maximum:
            maximum = value
    if maximum == -infinity() or maximum == infinity():
        return maximum
    var total = 0.0
    for source in range(0, vector_end, W):
        total += exp(
            previous.load[width=W](source)
            + transition.load[width=W](source)
            - maximum
        ).reduce_add()
    for source in range(vector_end, n_components):
        total += exp(
            previous[source]
            + transition[source]
            - maximum
        )
    return maximum + log(total)


def logsum_row(row: FPtr, n: Int) -> Float64:
    var maximum = -infinity()
    var vector_end = n - n % W
    for i in range(0, vector_end, W):
        var chunk_max = row.load[width=W](i).reduce_max()
        if chunk_max > maximum:
            maximum = chunk_max
    for i in range(vector_end, n):
        if row[i] > maximum:
            maximum = row[i]
    if maximum == -infinity() or maximum == infinity():
        return maximum
    var total = 0.0
    for i in range(0, vector_end, W):
        total += exp(row.load[width=W](i) - maximum).reduce_add()
    for i in range(vector_end, n):
        total += exp(row[i] - maximum)
    return maximum + log(total)


def logaddexp(a: Float64, b: Float64) -> Float64:
    if a == -infinity():
        return b
    if b == -infinity():
        return a
    var maximum = max(a, b)
    return maximum + log(1.0 + exp(min(a, b) - maximum))


def fill_log_transmat(
    transmat: FPtr, log_transmat: FPtr, n_components: Int, transpose: Bool = False
):
    if transpose:
        for source in range(n_components):
            for target in range(n_components):
                var index = source * n_components + target
                log_transmat[target * n_components + source] = safe_log(
                    transmat[index]
                )
    else:
        for i in range(n_components * n_components):
            log_transmat[i] = safe_log(transmat[i])


@export("mhl_forward_scaling")
def mhl_forward_scaling(
    startprob_addr: Int,
    transmat_addr: Int,
    frameprob_addr: Int,
    fwd_addr: Int,
    scaling_addr: Int,
    log_prob_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C") -> Int:
    var startprob = fp(startprob_addr)
    var transmat = fp(transmat_addr)
    var frameprob = fp(frameprob_addr)
    var fwd = fp(fwd_addr)
    var scaling = fp(scaling_addr)
    var log_prob_ptr = fp(log_prob_addr)

    var total = 0.0
    var vector_end = n_components - n_components % W
    for state in range(0, vector_end, W):
        var values = (
            startprob.load[width=W](state) * frameprob.load[width=W](state)
        )
        fwd.store(state, values)
        total += values.reduce_add()
    for state in range(vector_end, n_components):
        var value = startprob[state] * frameprob[state]
        fwd[state] = value
        total += value
    if total < 1.0e-300:
        return 0
    var scale = 1.0 / total
    scaling[0] = scale
    var log_prob = -log(scale)
    for state in range(0, vector_end, W):
        fwd.store(state, fwd.load[width=W](state) * scale)
    for state in range(vector_end, n_components):
        fwd[state] *= scale

    for t in range(1, n_samples):
        total = 0.0
        var previous = fwd + (t - 1) * n_components
        var current = fwd + t * n_components
        var frame = frameprob + t * n_components
        for state in range(0, vector_end, W):
            var values = (
                transmat.load[width=W](state) * previous[0]
            )
            for source in range(1, n_components):
                values += (
                    transmat.load[width=W](source * n_components + state)
                    * previous[source]
                )
            values *= frame.load[width=W](state)
            current.store(state, values)
            total += values.reduce_add()
        for state in range(vector_end, n_components):
            var value = 0.0
            for source in range(n_components):
                value += (
                    previous[source]
                    * transmat[source * n_components + state]
                )
            value *= frame[state]
            current[state] = value
            total += value
        if total < 1.0e-300:
            return 0
        scale = 1.0 / total
        scaling[t] = scale
        log_prob -= log(scale)
        for state in range(0, vector_end, W):
            current.store(state, current.load[width=W](state) * scale)
        for state in range(vector_end, n_components):
            current[state] *= scale

    log_prob_ptr[0] = log_prob
    return 1


@export("mhl_forward_log")
def mhl_forward_log(
    startprob_addr: Int,
    transmat_addr: Int,
    log_frameprob_addr: Int,
    fwd_addr: Int,
    log_transmat_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C") -> Float64:
    var startprob = fp(startprob_addr)
    var transmat = fp(transmat_addr)
    var log_frameprob = fp(log_frameprob_addr)
    var fwd = fp(fwd_addr)
    var log_transmat = fp(log_transmat_addr)
    fill_log_transmat(transmat, log_transmat, n_components, True)

    for state in range(n_components):
        fwd[state] = safe_log(startprob[state]) + log_frameprob[state]
    for t in range(1, n_samples):
        var previous = fwd + (t - 1) * n_components
        for state in range(n_components):
            fwd[t * n_components + state] = (
                logsum_transition(previous, log_transmat, state, n_components)
                + log_frameprob[t * n_components + state]
            )
    return logsum_row(fwd + (n_samples - 1) * n_components, n_components)


@export("mhl_backward_scaling")
def mhl_backward_scaling(
    transmat_addr: Int,
    frameprob_addr: Int,
    scaling_addr: Int,
    bwd_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C"):
    var transmat = fp(transmat_addr)
    var frameprob = fp(frameprob_addr)
    var scaling = fp(scaling_addr)
    var bwd = fp(bwd_addr)

    for state in range(n_components):
        bwd[(n_samples - 1) * n_components + state] = scaling[n_samples - 1]
    for reverse_t in range(n_samples - 1):
        var t = n_samples - 2 - reverse_t
        var next_frame = frameprob + (t + 1) * n_components
        var next_bwd = bwd + (t + 1) * n_components
        for state in range(n_components):
            var value = 0.0
            var transition = transmat + state * n_components
            var vector_end = n_components - n_components % W
            for target in range(0, vector_end, W):
                value += (
                    transition.load[width=W](target)
                    * next_frame.load[width=W](target)
                    * next_bwd.load[width=W](target)
                ).reduce_add()
            for target in range(vector_end, n_components):
                value += (
                    transition[target]
                    * next_frame[target]
                    * next_bwd[target]
                )
            bwd[t * n_components + state] = value * scaling[t]


@export("mhl_backward_log")
def mhl_backward_log(
    transmat_addr: Int,
    log_frameprob_addr: Int,
    bwd_addr: Int,
    log_transmat_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C"):
    var transmat = fp(transmat_addr)
    var log_frameprob = fp(log_frameprob_addr)
    var bwd = fp(bwd_addr)
    var log_transmat = fp(log_transmat_addr)
    fill_log_transmat(transmat, log_transmat, n_components)

    for state in range(n_components):
        bwd[(n_samples - 1) * n_components + state] = 0.0
    for reverse_t in range(n_samples - 1):
        var t = n_samples - 2 - reverse_t
        var next_frame = log_frameprob + (t + 1) * n_components
        var next_bwd = bwd + (t + 1) * n_components
        for state in range(n_components):
            var transition = log_transmat + state * n_components
            var maximum = -infinity()
            if n_components == 4:
                var value0 = transition[0] + next_frame[0] + next_bwd[0]
                var value1 = transition[1] + next_frame[1] + next_bwd[1]
                var value2 = transition[2] + next_frame[2] + next_bwd[2]
                var value3 = transition[3] + next_frame[3] + next_bwd[3]
                maximum = max(max(value0, value1), max(value2, value3))
                if maximum == -infinity() or maximum == infinity():
                    bwd[t * n_components + state] = maximum
                    continue
                var total = (
                    exp(value0 - maximum)
                    + exp(value1 - maximum)
                    + exp(value2 - maximum)
                    + exp(value3 - maximum)
                )
                bwd[t * n_components + state] = maximum + log(total)
                continue
            var vector_end = n_components - n_components % W
            for target in range(0, vector_end, W):
                var chunk_max = (
                    transition.load[width=W](target)
                    + next_frame.load[width=W](target)
                    + next_bwd.load[width=W](target)
                ).reduce_max()
                if chunk_max > maximum:
                    maximum = chunk_max
            for target in range(vector_end, n_components):
                var value = (
                    transition[target]
                    + next_frame[target]
                    + next_bwd[target]
                )
                if value > maximum:
                    maximum = value
            if maximum == -infinity() or maximum == infinity():
                bwd[t * n_components + state] = maximum
                continue
            var total = 0.0
            for target in range(0, vector_end, W):
                total += exp(
                    transition.load[width=W](target)
                    + next_frame.load[width=W](target)
                    + next_bwd.load[width=W](target)
                    - maximum
                ).reduce_add()
            for target in range(vector_end, n_components):
                total += exp(
                    transition[target]
                    + next_frame[target]
                    + next_bwd[target]
                    - maximum
                )
            bwd[t * n_components + state] = maximum + log(total)


@export("mhl_compute_scaling_xi_sum")
def mhl_compute_scaling_xi_sum(
    fwd_addr: Int,
    transmat_addr: Int,
    bwd_addr: Int,
    frameprob_addr: Int,
    xi_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C"):
    var fwd = fp(fwd_addr)
    var transmat = fp(transmat_addr)
    var bwd = fp(bwd_addr)
    var frameprob = fp(frameprob_addr)
    var xi = fp(xi_addr)
    for i in range(n_components * n_components):
        xi[i] = 0.0
    for t in range(n_samples - 1):
        for source in range(n_components):
            for target in range(n_components):
                xi[source * n_components + target] += (
                    fwd[t * n_components + source]
                    * transmat[source * n_components + target]
                    * frameprob[(t + 1) * n_components + target]
                    * bwd[(t + 1) * n_components + target]
                )


@export("mhl_compute_log_xi_sum")
def mhl_compute_log_xi_sum(
    fwd_addr: Int,
    transmat_addr: Int,
    bwd_addr: Int,
    log_frameprob_addr: Int,
    xi_addr: Int,
    log_transmat_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C"):
    var fwd = fp(fwd_addr)
    var transmat = fp(transmat_addr)
    var bwd = fp(bwd_addr)
    var log_frameprob = fp(log_frameprob_addr)
    var xi = fp(xi_addr)
    var log_transmat = fp(log_transmat_addr)
    fill_log_transmat(transmat, log_transmat, n_components)

    var log_prob = logsum_row(
        fwd + (n_samples - 1) * n_components, n_components
    )
    for i in range(n_components * n_components):
        xi[i] = -infinity()
    for t in range(n_samples - 1):
        for source in range(n_components):
            for target in range(n_components):
                var index = source * n_components + target
                var log_xi = (
                    fwd[t * n_components + source]
                    + log_transmat[index]
                    + log_frameprob[(t + 1) * n_components + target]
                    + bwd[(t + 1) * n_components + target]
                    - log_prob
                )
                xi[index] = logaddexp(xi[index], log_xi)


@export("mhl_viterbi")
def mhl_viterbi(
    startprob_addr: Int,
    transmat_addr: Int,
    log_frameprob_addr: Int,
    state_sequence_addr: Int,
    lattice_addr: Int,
    log_transmat_addr: Int,
    n_samples: Int,
    n_components: Int,
) abi("C") -> Float64:
    var startprob = fp(startprob_addr)
    var transmat = fp(transmat_addr)
    var log_frameprob = fp(log_frameprob_addr)
    var state_sequence = ip(state_sequence_addr)
    var lattice = fp(lattice_addr)
    var log_transmat = fp(log_transmat_addr)
    fill_log_transmat(transmat, log_transmat, n_components)

    for state in range(n_components):
        lattice[state] = safe_log(startprob[state]) + log_frameprob[state]
    for t in range(1, n_samples):
        for state in range(n_components):
            var best = -infinity()
            for source in range(n_components):
                var candidate = (
                    lattice[(t - 1) * n_components + source]
                    + log_transmat[source * n_components + state]
                )
                if candidate > best:
                    best = candidate
            lattice[t * n_components + state] = (
                best + log_frameprob[t * n_components + state]
            )

    var last_state = 0
    var best_score = lattice[(n_samples - 1) * n_components]
    for state in range(1, n_components):
        var candidate = lattice[(n_samples - 1) * n_components + state]
        if candidate > best_score:
            best_score = candidate
            last_state = state
    state_sequence[n_samples - 1] = Int64(last_state)

    var next_state = last_state
    for reverse_t in range(n_samples - 1):
        var t = n_samples - 2 - reverse_t
        var best_state = 0
        var best = (
            lattice[t * n_components]
            + log_transmat[next_state]
        )
        for state in range(1, n_components):
            var candidate = (
                lattice[t * n_components + state]
                + log_transmat[state * n_components + next_state]
            )
            if candidate >= best:
                best = candidate
                best_state = state
        state_sequence[t] = Int64(best_state)
        next_state = best_state
    return best_score
