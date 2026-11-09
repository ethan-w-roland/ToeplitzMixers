"""
Whole-model (MLPMixer) throughput benchmark: optimized (default) vs legacy FFT path.

Both models are built with identical weights, use_FFT=True, and benchmarked with
batch size 1 at long context lengths, reporting forward and forward+backward time,
tokens/sec, peak memory, and the share of forward time spent in token mixing.

Run:
  CUDA_VISIBLE_DEVICES=1 python bench_model_variants.py --seq 16384 --hidden 256 --blocks 8
"""
import argparse
import gc
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# profile.py in this directory shadows the stdlib `profile` module, which breaks
# `import transformers` (via torch._dynamo -> cProfile) for any script run from here.
# Drop this directory from sys.path; the model file below is loaded explicitly by path.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
_spec = importlib.util.spec_from_file_location(
    "toep_mixer_multiheaded", os.path.join(HERE, "toep_mixer_multiheaded.py")
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _time(fn, iters, warmup=3):
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


def bench(seq_len, hidden, heads, blocks, vocab, device, fp16, iters, expanded=False):
    torch.manual_seed(0)
    models = {}
    for name, optimized in (("optimized FFT", True), ("legacy FFT", False)):
        models[name] = m.MLPMixer(
            vocab_size=vocab,
            hidden_dim=hidden,
            seq_len=seq_len,
            num_blocks=blocks,
            heads=heads,
            expanded_convs=expanded,
            use_FFT=True,
            optimized=optimized,
        ).to(device)
    models["legacy FFT"].load_state_dict(models["optimized FFT"].state_dict())

    ids = torch.randint(0, vocab, (1, seq_len), device=device)
    labels = ids  # MLPMixer.forward needs labels (labels=None raises); measure the training-style path
    autocast = torch.autocast("cuda", dtype=torch.float16) if fp16 else None

    print(f"\nseq={seq_len} hidden={hidden} heads={heads} blocks={blocks} vocab={vocab} "
          f"batch=1 fp16={fp16} expanded_convs={expanded} device={device}")

    # forward parity of the two variants (same weights)
    try:
        with torch.no_grad():
            outs = {}
            for name in models:
                if fp16:
                    with autocast:
                        outs[name] = models[name](ids, labels=labels)[1]
                else:
                    outs[name] = models[name](ids, labels=labels)[1]
        maxdiff = (outs["optimized FFT"] - outs["legacy FFT"]).abs().max().item()
        print(f"logit parity maxdiff: {maxdiff:.2e}")
        del outs
    except RuntimeError as ex:
        print(f"logit parity skipped: {repr(ex)[:90]}")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"{'variant':14s} {'fwd (ms)':>10s} {'fwd tok/s':>10s} {'f+b (ms)':>10s} "
          f"{'f+b tok/s':>10s} {'peak MiB':>9s}")
    results = {}
    for name, model in list(models.items()):
        def fwd():
            with torch.no_grad():
                if fp16:
                    with autocast:
                        model(ids, labels=labels)
                else:
                    model(ids, labels=labels)

        def fb():
            model.zero_grad(set_to_none=True)
            if fp16:
                with autocast:
                    loss, _ = model(ids, labels=labels)
            else:
                loss, _ = model(ids, labels=labels)
            loss.backward()

        try:
            fwd_ms = _time(fwd, iters)
        except RuntimeError as ex:  # e.g. legacy cuFFT cannot do half precision at 2S-1
            print(f"{name:14s} skipped: {repr(ex)[:90]}")
            model.zero_grad(set_to_none=True)
            models.pop(name)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        bwd_ms = _time(fb, max(2, iters // 2), warmup=2)
        peak = torch.cuda.max_memory_allocated() / 2**20
        results[name] = (fwd_ms, bwd_ms)
        print(f"{name:14s} {fwd_ms:10.1f} {seq_len / (fwd_ms * 1e-3):10.0f} "
              f"{bwd_ms:10.1f} {seq_len / (bwd_ms * 1e-3):10.0f} {peak:9.0f}")
        model.zero_grad(set_to_none=True)
        models.pop(name)
        gc.collect()
        torch.cuda.empty_cache()

    if len(results) == 2:
        fo, lo = results["optimized FFT"], results["legacy FFT"]
        print(f"-> fwd speedup     optimized/legacy: {lo[0] / fo[0]:.2f}x")
        print(f"-> fwd+bwd speedup optimized/legacy: {lo[1] / fo[1]:.2f}x")

    # forward-pass breakdown: which part of the model the FFT speedup can touch
    parts = {}
    stacks = {}
    for opt_name, optimized in (("optimized", True), ("legacy", False)):
        model = m.MLPMixer(
            vocab_size=vocab, hidden_dim=hidden, seq_len=seq_len, num_blocks=blocks,
            heads=heads, expanded_convs=expanded, use_FFT=True, optimized=optimized,
        ).to(device)
        stacks[opt_name] = model

    model = stacks["optimized"]
    x = torch.randn(1, hidden, seq_len, device=device)      # (B, E, S): token-mixing layout
    xs = torch.randn(1, seq_len, hidden, device=device)     # (B, S, E): channel layout

    def embed():
        with torch.no_grad():
            model.input_layer(ids)

    def chan_mix():
        with torch.no_grad():
            for block in model.mixer_blocks:
                block.channel_mixing_layer(block.channel_norm(xs))

    def token_mix(which="optimized"):
        with torch.no_grad():
            for block in stacks[which].mixer_blocks:
                block.token_mixing_layer(xs.transpose(1, 2))

    def head():
        with torch.no_grad():
            logits = model.output_layer(xs)
            model.loss_fn(logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1))

    def full_fwd():
        with torch.no_grad():
            model(ids, labels=labels)

    parts["embedding"] = _time(embed, iters)
    parts["token mixing (mixers)"] = _time(lambda: token_mix("optimized"), iters)
    legacy_tm = _time(lambda: token_mix("legacy"), iters)
    parts["channel mixing (MLP)"] = _time(chan_mix, iters)
    parts["lm head + loss"] = _time(head, iters)
    ff = _time(full_fwd, iters)
    for name, ms in parts.items():
        print(f"  {name:26s} {ms:8.1f} ms  {100 * ms / ff:5.1f}% of forward")
    print(f"  {'full forward':26s} {ff:8.1f} ms")
    print(f"  in-model token-mixing speedup: {legacy_tm / parts['token mixing (mixers)']:.2f}x "
          f"({legacy_tm:.1f} ms legacy -> {parts['token mixing (mixers)']:.1f} ms optimized)")
    stacks.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--fp16", action="store_true", help="run forward under autocast(fp16)")
    ap.add_argument("--expanded", action="store_true", help="expanded_convs=True (two Toeplitz convs per head)")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        bench(args.seq, args.hidden, args.heads, args.blocks, args.vocab, device, args.fp16, args.iters, args.expanded)
    except torch.cuda.OutOfMemoryError as ex:
        print(f"OOM: {ex}")
        print("try smaller --blocks / --hidden / --vocab")
