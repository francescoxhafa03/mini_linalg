import argparse, csv, os, platform
import numpy as np
from roofline_ops import (DTYPES, matvec_model, matmat_model,
                          make_matvec, make_matmat, time_kernel)

MATVEC_N = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
MATMAT_N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def stream_bandwidth(dtype, n=50_000_000):
    """Memory bandwidth estimate in GB/s with a = b + 2*c on arrays much bigger than the cache.
    NumPy does it in two passes (tmp = 2*c, then a = b + tmp), so the traffic is
    5 arrays: read c, write tmp, read b, read tmp, write a."""
    b = np.ones(n, dtype=dtype)
    c = np.ones(n, dtype=dtype)
    a = np.empty(n, dtype=dtype)
    tmp = np.empty(n, dtype=dtype)

    def kernel():
        np.multiply(c, dtype(2.0), out=tmp)
        np.add(b, tmp, out=a)

    t = min(time_kernel(kernel, min_time=0.5))
    return 5 * n * np.dtype(dtype).itemsize / t / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--out", default="results")
    p.add_argument("--max-matvec", type=int, default=MATVEC_N[-1])
    p.add_argument("--max-matmat", type=int, default=MATMAT_N[-1])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = [("matvec", MATVEC_N, args.max_matvec, matvec_model, make_matvec),
             ("matmat", MATMAT_N, args.max_matmat, matmat_model, make_matmat)]
    rows = []
    for op, sizes, nmax, model, make in cases:
        for dname, dtype in DTYPES.items():
            for n in (n for n in sizes if n <= nmax):
                samples = time_kernel(make(n, dtype), repeats=args.repeats)
                flops, nbytes = model(n, np.dtype(dtype).itemsize)
                t_min, t_med = min(samples), float(np.median(samples))
                row = dict(op=op, dtype=dname, N=n, t_min=t_min, t_median=t_med,
                           gflops_med=flops / t_med / 1e9, gflops_max=flops / t_min / 1e9,
                           gbs_med=nbytes / t_med / 1e9, ai=flops / nbytes)
                rows.append(row)
                print(f"{op} {dname} N={n}: {t_med:.3e} s, "
                      f"{row['gflops_med']:.1f} GFLOP/s, {row['gbs_med']:.1f} GB/s", flush=True)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    bw = {name: stream_bandwidth(dt) for name, dt in DTYPES.items()}
    with open(os.path.join(args.out, "machine.txt"), "w") as f:
        f.write(f"platform: {platform.platform()}\nprocessor: {platform.processor()}\n"
                f"numpy: {np.__version__}\nstream_like_GBs: {bw}\n")
    print("bandwidth estimate (GB/s):", bw)


if __name__ == "__main__":
    main()
