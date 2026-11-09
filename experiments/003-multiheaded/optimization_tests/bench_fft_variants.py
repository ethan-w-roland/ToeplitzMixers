"""
Throughput benchmark for the two FFTToeplitzCausalLinear implementations in
toep_mixer_multiheaded.py:

  - optimized=True  (default): axis-last convolution, zero-padded to N = next_fast_len(2S-1)
  - optimized=False (legacy):  axis-first circulant matmul, FFTs at length S, transposes
  - dense:                     materialized Toeplitz matrix multiply (O(S^2)) reference

Run:  CUDA_VISIBLE_DEVICES=1 python bench_fft_variants.py [--dim 32768 --rows 256]
"""
import argparse
import importlib.util
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# profile.py in this directory shadows the stdlib `profile` module, which breaks
# `import transformers` (via torch._dynamo -> cProfile) for any script run from here.
# Drop this directory from sys.path; the model file below is loaded explicitly by path.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
_spec = importlib.util.spec_from_file_location(
    "toep_mixer_multiheaded",
    os.path.join(os.path.dirname(HERE), "toep_mixer_multiheaded.py"),
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # load by path: running scripts from this dir shadows stdlib `profile`


def _time(fn, iters=20, warmup=5):
    import time

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters


def bench(dim, rows, device):
    torch.manual_seed(0)
    dense = m.ToeplitzCausalLinear(dim).to(device)
    with torch.no_grad():
        dense.bias.normal_(0, 0.1)  # nonzero bias shared by every variant
    opt = m.FFTToeplitzCausalLinear(dim, optimized=True).to(device)
    legacy = m.FFTToeplitzCausalLinear(dim, optimized=False).to(device)
    for f in (opt, legacy):
        f.load_state_dict(dense.state_dict())

    x = torch.randn(1, rows, dim, device=device, requires_grad=True)  # layers take (B, E, S)
    g = torch.randn(1, rows, dim, device=device)

    # correctness parity first (dense reference only when the S x S matrix fits comfortably)
    use_dense = dim <= 8192
    with torch.no_grad():
        oo, ol = opt(x), legacy(x)
    if use_dense:
        od = dense(x)
        assert torch.allclose(od, oo, atol=1e-3), "optimized != dense"
        assert torch.allclose(od, ol, atol=1e-3), "legacy != dense"
    else:
        assert torch.allclose(oo, ol, atol=1e-3), "optimized != legacy"

    print(f"\ndim={dim} rows={rows} device={device}")
    print(f"{'variant':28s} {'fwd (ms)':>10s} {'fwd GiB/s':>10s} {'fwd+bwd (ms)':>13s}")
    results = {}
    variants = [("optimized FFT", opt), ("legacy FFT", legacy)]
    if use_dense:
        variants.append(("dense matmul", dense))
    for name, layer in variants:
        def fwd():
            with torch.no_grad():
                layer(x)

        def fb():
            layer.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            layer(x).backward(g)

        try:
            tf = _time(fwd)
            tb = _time(fb)
        except Exception as ex:
            print(f"{name:28s} skipped: {repr(ex)[:100]}")
            continue
        gibs = 2 * x.numel() * 4 / (tf * 1e-3) / 2**30
        results[name] = (tf, tb)
        print(f"{name:28s} {tf:10.2f} {gibs:10.1f} {tb:13.2f}")

    if "optimized FFT" in results and "legacy FFT" in results:
        fo, lo = results["optimized FFT"], results["legacy FFT"]
        print(f"-> fwd speedup   optimized/legacy: {lo[0] / fo[0]:.2f}x")
        print(f"-> fwd+bwd speedup optimized/legacy: {lo[1] / fo[1]:.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32768)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="benchmark several dims at once")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in (args.dims or [args.dim]):
        bench(d, args.rows, dev)
