"""
Memory-lean causal Toeplitz (shared-filter) FFT convolution.

    y[..., i] = sum_{j<=i} c[i-j] * x[..., j]        (x: (..., S), c: (S,))

Custom autograd.Function that saves only (x, c) for backward and recomputes the
spectra there, instead of letting autograd keep the ~(N/2+1)-complex spectrum of x alive
for every layer.

Backward (g = grad_y, F = rfft padded to N >= 2S-1, so there is no circular aliasing):
    grad_x[..., j] = sum_{i>=j} c[i-j] g[..., i]  = irfft(conj(F_c) * F_g)[:S]
    grad_c[d]      = sum_{rows} sum_j x[j] g[j+d]  = irfft(sum_rows conj(F_x) * F_g)[:S]

The 1/N of the inverse transform is folded into the (tiny) filter spectrum, and the inverse
is run with norm="forward" (= unscaled), which removes one full elementwise pass.

Usage:
    python toeplitz_autograd.py --dim 32768 --B 1 --E 512
"""
import argparse
import math
import torch
import torch.nn as nn
from scipy.fft import next_fast_len


def _compute_dtype(x, c):
    return torch.float64 if (x.dtype == torch.float64 or c.dtype == torch.float64) else torch.float32


class ToeplitzCausalConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, chunk_rows=None):
        S = x.shape[-1]
        N = next_fast_len(2 * S - 1, real=True)
        cdt = _compute_dtype(x, c)
        f_c = torch.fft.rfft(c.to(cdt), n=N) / N                      # fold 1/N into the filter
        f_x = torch.fft.rfft(x.to(cdt), n=N, dim=-1)                  # padding handled inside
        y = torch.fft.irfft(f_c * f_x, n=N, dim=-1, norm="forward")[..., :S]
        ctx.save_for_backward(x, c)                                   # NOT the spectra
        ctx.N, ctx.chunk_rows = N, chunk_rows
        # y is a slice view of the N-length buffer; the caller's next op (bias add) materializes it.
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, c = ctx.saved_tensors
        N, S = ctx.N, x.shape[-1]
        cdt = _compute_dtype(x, c)
        need_x, need_c = ctx.needs_input_grad[0], ctx.needs_input_grad[1]

        R = x.numel() // S
        xf = x.reshape(R, S)
        gf = g.reshape(R, S)
        chunk = ctx.chunk_rows or R

        grad_x = torch.empty_like(xf) if need_x else None
        acc = None
        f_c = torch.fft.rfft(c.to(cdt), n=N) / N if need_x else None

        for s in range(0, R, chunk):
            f_g = torch.fft.rfft(gf[s:s + chunk].to(cdt), n=N, dim=-1)
            if need_x:
                gx = torch.fft.irfft(f_c.conj() * f_g, n=N, dim=-1, norm="forward")[..., :S]
                grad_x[s:s + chunk] = gx.to(x.dtype)                  # copy into a compact buffer
            if need_c:
                f_x = torch.fft.rfft(xf[s:s + chunk].to(cdt), n=N, dim=-1)   # recompute
                part = (f_x.conj() * f_g).sum(0)
                acc = part if acc is None else acc + part
            del f_g

        grad_c = None
        if need_c:
            grad_c = torch.fft.irfft(acc / N, n=N, norm="forward")[:S].to(c.dtype)
        return (grad_x.view(x.shape) if need_x else None), grad_c, None


class FFTToeplitzCausalLinearAG(nn.Module):
    """Same parameters as FFTToeplitzCausalLinear (weight: (1, dim), bias: (dim,))."""
    def __init__(self, dim: int, chunk_rows=None):
        super().__init__()
        self.dim = dim
        self.chunk_rows = chunk_rows
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):                                              # x: (..., dim)
        y = ToeplitzCausalConvFn.apply(x, self.weight.squeeze(0), self.chunk_rows)
        return y + self.bias


class FFTToeplitzCausalLinearBaseline(nn.Module):
    """Plain autograd, padded fp32 FFT (your current optimized path)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.N = next_fast_len(2 * dim - 1, real=True)
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        S, N = self.dim, self.N
        f_c = torch.fft.rfft(self.weight.squeeze(0).float(), n=N) / N
        f_x = torch.fft.rfft(x.float(), n=N, dim=-1)
        y = torch.fft.irfft(f_c * f_x, n=N, dim=-1, norm="forward")[..., :S]
        return y.to(x.dtype) + self.bias


# ----------------------------- half-precision (fp16 FFT) variant -----------------------------
# cuFFT half precision needs power-of-2 lengths and has no bf16 FFT, so bf16/fp32 inputs are cast to fp16 for the
# transforms. To keep fp16 in range: transforms use norm="ortho" (values stay O(1)), x/c/g are divided by their
# per-tensor amax first (rescale=True), and the scales and sqrt(N) are applied afterwards in fp32:
#     y      = sx*sc*sqrt(N) * irfft_ortho(O_c * O_x)
#     grad_x = sc*sg*sqrt(N) * irfft_ortho(conj(O_c) * O_g)
#     grad_c = sx*sg*sqrt(N) * irfft_ortho(sum_rows conj(O_x) * O_g)     (row sum accumulated in complex64)
_NATIVE_CHALF = None


def _native_chalf_ok():
    """True if complex32 multiply (and conj-multiply) work in this PyTorch build; probed once."""
    global _NATIVE_CHALF
    if _NATIVE_CHALF is None:
        try:
            z = torch.fft.rfft(torch.randn(8, device="cuda").half(), norm="ortho")
            _ = z * z
            _ = z.conj() * z
            _NATIVE_CHALF = True
        except Exception:
            _NATIVE_CHALF = False
    return _NATIVE_CHALF


def _cmul(a, b, conj_a=False):
    """(conj(a) if conj_a else a) * b for complex32 spectra; falls back to real/imag halves."""
    if _native_chalf_ok():
        return (a.conj() if conj_a else a) * b
    ar, br = torch.view_as_real(a), torch.view_as_real(b)
    a0, a1, b0, b1 = ar[..., 0], ar[..., 1], br[..., 0], br[..., 1]
    if conj_a:
        re, im = a0 * b0 + a1 * b1, a0 * b1 - a1 * b0
    else:
        re, im = a0 * b0 - a1 * b1, a0 * b1 + a1 * b0
    return torch.view_as_complex(torch.stack((re, im), dim=-1))


def _conj_mul_sum_rows(a, b, sub=64):
    """sum over rows of conj(a)*b for (R, K) complex32 inputs, accumulated in complex64, `sub` rows at a time."""
    acc = None
    for s in range(0, a.shape[0], sub):
        part = (a[s:s + sub].to(torch.complex64).conj() * b[s:s + sub].to(torch.complex64)).sum(0)
        acc = part if acc is None else acc + part
    return acc


def _scale(t, rescale):
    if not rescale:
        return t.new_ones((), dtype=torch.float32)
    return t.abs().amax().float().clamp_min(1e-20)


class ToeplitzCausalConvHalfFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, chunk_rows=None, rescale=True):
        S = x.shape[-1]
        N = 1 << (2 * S - 2).bit_length()                         # power of 2 >= 2S-1 (required for half)
        sx, sc = _scale(x, rescale), _scale(c, rescale)
        xh = (x * (1.0 / sx)).half()
        ch = (c * (1.0 / sc)).half()
        f_x = torch.fft.rfft(xh, n=N, dim=-1, norm="ortho")
        f_c = torch.fft.rfft(ch, n=N, norm="ortho")
        y = torch.fft.irfft(_cmul(f_c, f_x), n=N, dim=-1, norm="ortho")[..., :S]
        out = y.to(x.dtype).contiguous()                          # compact copy; frees the N-length buffer
        out.mul_(sx * sc * math.sqrt(N))
        ctx.save_for_backward(x, c, sx, sc)
        ctx.N, ctx.chunk_rows, ctx.rescale = N, chunk_rows, rescale
        return out

    @staticmethod
    def backward(ctx, g):
        x, c, sx, sc = ctx.saved_tensors
        N, S = ctx.N, x.shape[-1]
        need_x, need_c = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        R = x.numel() // S
        xf, gf = x.reshape(R, S), g.reshape(R, S)
        sg = _scale(gf, ctx.rescale)
        chunk = ctx.chunk_rows or R

        f_c = torch.fft.rfft((c * (1.0 / sc)).half(), n=N, norm="ortho") if need_x else None
        grad_x = torch.empty_like(xf) if need_x else None
        acc = None
        for s in range(0, R, chunk):
            f_g = torch.fft.rfft((gf[s:s + chunk] * (1.0 / sg)).half(), n=N, dim=-1, norm="ortho")
            if need_x:
                gx = torch.fft.irfft(_cmul(f_c, f_g, conj_a=True), n=N, dim=-1, norm="ortho")[..., :S]
                grad_x[s:s + chunk] = gx.to(x.dtype)
            if need_c:
                f_x = torch.fft.rfft((xf[s:s + chunk] * (1.0 / sx)).half(), n=N, dim=-1, norm="ortho")  # recompute
                part = _conj_mul_sum_rows(f_x, f_g)
                acc = part if acc is None else acc + part
            del f_g

        grad_c = None
        if need_x:
            grad_x.mul_(sc * sg * math.sqrt(N))
        if need_c:
            grad_c = (torch.fft.irfft(acc, n=N, norm="ortho")[:S] * (sx * sg * math.sqrt(N))).to(c.dtype)
        return (grad_x.view(x.shape) if need_x else None), grad_c, None, None


class FFTToeplitzCausalLinearHalf(nn.Module):
    """Same parameters as the other layers (weight: (1, dim), bias: (dim,)); fp16 FFTs, recompute backward."""
    def __init__(self, dim: int, chunk_rows=None, rescale: bool = True):
        super().__init__()
        self.dim, self.chunk_rows, self.rescale = dim, chunk_rows, rescale
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        y = ToeplitzCausalConvHalfFn.apply(x, self.weight.squeeze(0), self.chunk_rows, self.rescale)
        return y + self.bias


def check_half(device="cuda"):
    """Half variant vs the fp64 fp-exact Function on a small problem (forward + both grads)."""
    torch.manual_seed(0)
    S = 2048
    c = torch.randn(S, device=device, requires_grad=True)
    x = torch.randn(2, 3, S, device=device, requires_grad=True)
    g = torch.randn(2, 3, S, device=device)
    out = ToeplitzCausalConvHalfFn.apply(x, c, None, True)
    gx, gc = torch.autograd.grad(out, [x, c], g)
    xd, cd = x.detach().double().requires_grad_(), c.detach().double().requires_grad_()
    ref = ToeplitzCausalConvFn.apply(xd, cd, None)
    rgx, rgc = torch.autograd.grad(ref, [xd, cd], g.double())
    rel = lambda a, b: ((a.double() - b).norm() / b.norm()).item()
    print(f"half variant (native complex32 mul: {_native_chalf_ok()}): "
          f"fwd {rel(out, ref):.1e}  gx {rel(gx, rgx):.1e}  gw {rel(gc, rgc):.1e}   (fp16-level, ~1e-3 expected)")


# ----------------------------- tests / benchmark -----------------------------
def dense_reference(c, x):
    S = c.shape[0]
    i = torch.arange(S, device=c.device)
    d = i[:, None] - i[None, :]
    T = torch.where(d >= 0, c[d.clamp(min=0)], torch.zeros((), dtype=c.dtype, device=c.device))
    return x @ T.t()


def check_small(device="cpu"):
    """float64 forward vs dense T, plus gradcheck of the custom backward."""
    torch.manual_seed(0)
    S = 24                                            # not a power of 2 on purpose
    c = torch.randn(S, dtype=torch.float64, device=device, requires_grad=True)
    x = torch.randn(2, 3, S, dtype=torch.float64, device=device, requires_grad=True)
    torch.testing.assert_close(ToeplitzCausalConvFn.apply(x, c, None), dense_reference(c, x))
    assert torch.autograd.gradcheck(lambda a, b: ToeplitzCausalConvFn.apply(a, b, None), (x, c))
    assert torch.autograd.gradcheck(lambda a, b: ToeplitzCausalConvFn.apply(a, b, 2), (x, c))  # chunked path
    print("small-size forward + gradcheck (incl. chunked backward): OK")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _mem_profile(layer, x):
    """Returns (MiB held by autograd after forward excluding the output, peak MiB over fwd+bwd)."""
    layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = layer(x)
    torch.cuda.synchronize()
    out_bytes = out.numel() * out.element_size()
    held = torch.cuda.memory_allocated() - base - out_bytes
    out.sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return held / 2**20, peak / 2**20


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32768)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--chunk_rows", type=int, default=128, help="rows per backward chunk for the chunked variant")
    args = ap.parse_args()
    dev = "cuda"

    check_small("cpu")
    check_small(dev)
    check_half(dev)

    torch.manual_seed(0)
    D = args.dim
    base = FFTToeplitzCausalLinearBaseline(D).to(dev)
    variants = {
        "baseline autograd (fp32)": base,
        "recompute fn": FFTToeplitzCausalLinearAG(D).to(dev),
        f"recompute fn, chunk={args.chunk_rows}": FFTToeplitzCausalLinearAG(D, args.chunk_rows).to(dev),
        "recompute fn, fp16 FFT": FFTToeplitzCausalLinearHalf(D).to(dev),
        f"recompute fn, fp16 FFT, chunk={args.chunk_rows}": FFTToeplitzCausalLinearHalf(D, args.chunk_rows).to(dev),
    }

    with torch.no_grad():                             # identical (nonzero) weights and bias everywhere
        base.bias.normal_(0, 0.1)
    for v in variants.values():
        v.load_state_dict(base.state_dict())

    x = torch.randn(args.B, args.E, D, device=dev, requires_grad=True)
    g = torch.randn(args.B, args.E, D, device=dev)

    for name in list(variants):                        # drop variants the kernels reject (e.g. unsupported H)
        try:
            variants[name](x).backward(g)
        except Exception as ex:
            print(f"[skip {name}: {repr(ex)[:160]}]")
            del variants[name]
    x.grad = None

    # ---- correctness vs baseline (forward, grad x, grad weight) ----
    ref_out = base(x)
    ref_gx, ref_gw = torch.autograd.grad(ref_out, [x, base.weight], g)
    print(f"\n{'variant':34s} {'fwd err':>9s} {'gx err':>9s} {'gw err':>9s}")
    for name, v in variants.items():
        out = v(x)
        gx, gw = torch.autograd.grad(out, [x, v.weight], g)
        print(f"{name:34s} {_rel(out, ref_out):9.1e} {_rel(gx, ref_gx):9.1e} {_rel(gw, ref_gw):9.1e}")

    # ---- memory ----
    print(f"\n{'variant':34s} {'held after fwd (MiB)':>22s} {'peak fwd+bwd (MiB)':>20s}")
    for name, v in variants.items():
        held, peak = _mem_profile(v, x)
        print(f"{name:34s} {held:22.0f} {peak:20.0f}")

    # ---- speed ----
    print(f"\n{'variant':34s} {'fwd (ms)':>10s} {'fwd+bwd (ms)':>14s}")
    for name, v in variants.items():
        def fwd():
            with torch.no_grad():
                v(x)
        def fb():
            v.zero_grad(set_to_none=True)
            x.grad = None
            v(x).backward(g)
        print(f"{name:34s} {_time(fwd):10.2f} {_time(fb):14.2f}")
