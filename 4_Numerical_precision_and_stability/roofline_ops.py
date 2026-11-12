import time
import numpy as np

DTYPES = {"float32": np.float32, "float64": np.float64}


def matvec_model(n, s):
    # y = A x: n*n multiplications + n*(n-1) additions ~ 2 n^2 flops.
    # Traffic: read A (n^2), read x (n), write y (n); s = bytes per element.
    return 2 * n * n, (n * n + 2 * n) * s


def matmat_model(n, s):
    # C = A B: ~2 n^3 flops.
    # Traffic if every element moves once: read A and B, write C (3 n^2).
    return 2 * n ** 3, 3 * n * n * s


def make_matvec(n, dtype):
    A = np.full((n, n), 1.0, dtype=dtype)
    x = np.full(n, 3.0, dtype=dtype)
    y = np.empty(n, dtype=dtype)
    return lambda: np.matmul(A, x, out=y)


def make_matmat(n, dtype):
    A = np.full((n, n), 1.0, dtype=dtype)
    B = np.full((n, n), 2.0, dtype=dtype)
    C = np.empty((n, n), dtype=dtype)
    return lambda: np.matmul(A, B, out=C)


def time_kernel(fn, repeats=7, min_time=0.2, warmup=2):
    """Return `repeats` timings (seconds per call) of fn().
    Fast kernels are called several times per sample so each sample lasts about min_time."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    fn()
    one_call = time.perf_counter() - t0
    inner = max(1, int(min_time / max(one_call, 1e-9)))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples
