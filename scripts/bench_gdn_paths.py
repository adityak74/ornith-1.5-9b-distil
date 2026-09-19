"""One-layer gated-delta benchmark: sequential loop (mlx-lm), checkpointed
sequential loop (mlx-lm-lora's recurrent_patch), chunkwise-parallel (mlx-lm
PR #1870 / odistil.gdn_chunkwise). Forward + backward w.r.t. q, k, v.

    uv run python scripts/bench_gdn_paths.py [--seqs 512,1024,2048,4096]

Hk=16, Hv=32, Dk=Dv=128, batch 1, bf16 inputs, float32 state -- the PR's setup.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx_lm.models.gated_delta as gd

from odistil.gdn_chunkwise import gated_delta_chunkwise

ap = argparse.ArgumentParser()
ap.add_argument("--seqs", default="512,1024,2048,4096")
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

Hk, Hv, D = 16, 32, 128
step = gd._gated_delta_step_ops


def checkpointed_ops(q, k, v, g, beta, state=None, mask=None):
    """mlx-lm-lora recurrent_patch.checkpointed_gated_delta, verbatim in logic."""
    def chunk_fn(q, k, v, g, beta, state, mask):
        outs = []
        for t in range(q.shape[1]):
            o, state = step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state, None)
            outs.append(o)
        return mx.stack(outs, axis=1), state

    ck = mx.checkpoint(chunk_fn)
    B, T, _, Dk = q.shape
    if state is None:
        state = mx.zeros((B, Hv, D, Dk), dtype=mx.float32)
    rep = Hv // Hk
    if rep > 1:
        q = mx.repeat(q, rep, axis=-2); k = mx.repeat(k, rep, axis=-2)
    outs = []
    for s in range(0, T, a.chunk):
        e = min(s + a.chunk, T)
        o, state = ck(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], state, None)
        outs.append(o)
    return mx.concatenate(outs, axis=1), state


PATHS = {
    "sequential (mlx-lm)": gd.gated_delta_ops,
    "checkpointed sequential (mlx-lm-lora)": checkpointed_ops,
    "chunkwise (PR #1870)": lambda q, k, v, g, b, s=None, m=None: gated_delta_chunkwise(q, k, v, g, b, s, chunk=a.chunk),
}


def bench(fn, T):
    mx.random.seed(0)
    q = mx.random.normal((1, T, Hk, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, T, Hk, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, T, Hv, D)).astype(mx.bfloat16)
    g = -mx.abs(mx.random.normal((1, T, Hv))) * 0.1
    beta = mx.sigmoid(mx.random.normal((1, T, Hv)))

    def loss(q, k, v):
        y, _ = fn(q, k, v, g, beta)
        return y.astype(mx.float32).sum()

    vg = mx.value_and_grad(loss, argnums=(0, 1, 2))
    mx.reset_peak_memory()
    out = vg(q, k, v); mx.eval(out)                      # warm-up (also compiles)
    peak = mx.get_peak_memory() / 1e9
    t0 = time.time()
    for _ in range(a.reps):
        out = vg(q, k, v); mx.eval(out)
    return peak, (time.time() - t0) / a.reps


print(f"{'path':<40} {'T':>5} {'peak GB':>8} {'s / fwd+bwd':>12}")
for T in map(int, a.seqs.split(",")):
    for name, fn in PATHS.items():
        if "sequential" in name and T > 2048 and "checkpointed" not in name:
            print(f"{name:<40} {T:>5} {'skip':>8} {'(108 GB in the PR table)':>12}"); continue
        try:
            peak, dt = bench(fn, T)
            print(f"{name:<40} {T:>5} {peak:>8.2f} {dt:>12.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<40} {T:>5} {'fail':>8} {type(e).__name__}")
        mx.clear_cache()
