"""Chunkwise-parallel training path for the gated delta rule.

mlx-lm trains these layers through a sequential loop over every timestep -- its
own docstring calls it a "reference implementation for prompt prefill" -- because
the fast Metal kernel used at inference is not differentiable. That costs about
8 MB of autograd state per token per layer, which is what pins training on a
64 GB machine to 1024-token sequences.

The gated delta rule admits a chunkwise parallel form (Yang et al.,
*Parallelizing Linear Transformers with the Delta Rule over Sequence Length*,
arXiv:2406.06484; Yang, Kautz & Hatamizadeh, *Gated Delta Networks*,
arXiv:2412.06464). Splitting the sequence into chunks of C turns O(T)
materialised states into O(T/C) and replaces the per-step loop with matmuls.

The recurrence, matching `mlx_lm.models.gated_delta`:

    S_t = g_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T
    y_t = S_t q_t

Writing S_t = g_t * S_{t-1} + u_t k_t^T with u_t = b_t (v_t - g_t S_{t-1} k_t)
telescopes within a chunk, so with S_0 the state entering the chunk:

    S_j = G_j S_0 + sum_{i<=j} (G_j / G_i) u_i k_i^T,   G_j = prod_{m<=j} g_m

and the u_i follow a triangular system solved once per chunk. Everything is
expressed in *ratios* G_j / G_i (bounded by 1 for i <= j) rather than in G_j
itself, which would underflow and send the intermediate writes to infinity on
heads with strong decay.
"""

from __future__ import annotations

import mlx.core as mx

NEG_INF = -1e30


def unit_tri_inv(m: mx.array) -> mx.array:
    """Inverse of a unit lower-triangular matrix, via 2x2 block recursion.

    `mx.linalg.tri_inv` is CPU-only and has no VJP, so it cannot appear in a
    training graph. This builds the inverse from matmuls instead, in log2(C)
    levels, which autodiff handles natively:

        [A 0; C B]^-1 = [A^-1 0; -B^-1 C A^-1, B^-1]
    """
    n = m.shape[-1]
    if n == 1:
        return mx.ones_like(m)
    h = n // 2
    a, c, b = m[..., :h, :h], m[..., h:, :h], m[..., h:, h:]
    ai, bi = unit_tri_inv(a), unit_tri_inv(b)
    lower_left = -(bi @ c @ ai)
    top = mx.concatenate([ai, mx.zeros_like(mx.swapaxes(lower_left, -1, -2))], axis=-1)
    bottom = mx.concatenate([lower_left, bi], axis=-1)
    return mx.concatenate([top, bottom], axis=-2)


def _pad_to_chunk(x: mx.array, chunk: int, axis: int, value: float) -> mx.array:
    t = x.shape[axis]
    rem = (-t) % chunk
    if rem == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, rem)
    return mx.pad(x, pad, constant_values=value)


def gated_delta_chunkwise(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    chunk: int = 64,
) -> tuple[mx.array, mx.array]:
    """Drop-in replacement for `gated_delta_ops`, chunk-parallel and differentiable.

    Shapes match the reference:
      q, k: [B, T, Hk, Dk] | v: [B, T, Hv, Dv] | g, beta: [B, T, Hv]
      state: [B, Hv, Dv, Dk]
    """
    if mask is not None:
        raise NotImplementedError("masked sequences still take the sequential path")
    if g.ndim != 3:
        raise NotImplementedError("vectorised gating (g.ndim == 4) not supported yet")

    b_sz, t_len, hk, dk = q.shape
    hv, dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((b_sz, hv, dv, dk), dtype=mx.float32)

    if (repeat := hv // hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)

    # [B, T, H, D] -> [B, H, T, D], accumulate in fp32
    to_bhtd = lambda x: mx.swapaxes(x, 1, 2).astype(mx.float32)
    q, k, v = to_bhtd(q), to_bhtd(k), to_bhtd(v)
    g, beta = mx.swapaxes(g, 1, 2).astype(mx.float32), mx.swapaxes(beta, 1, 2).astype(mx.float32)

    n_pad = (-t_len) % chunk
    if n_pad:
        # pad writes to nothing: g = 1 (no decay), beta = 0 (no write)
        q, k, v = (_pad_to_chunk(x, chunk, 2, 0.0) for x in (q, k, v))
        g = _pad_to_chunk(g, chunk, 2, 1.0)
        beta = _pad_to_chunk(beta, chunk, 2, 0.0)

    t_pad = q.shape[2]
    n_chunks = t_pad // chunk
    split = lambda x: x.reshape(b_sz, hv, n_chunks, chunk, -1)
    q, k, v = split(q), split(k), split(v)
    g = g.reshape(b_sz, hv, n_chunks, chunk)
    beta = beta.reshape(b_sz, hv, n_chunks, chunk)

    log_g = mx.log(mx.maximum(g, 1e-30))
    cum = mx.cumsum(log_g, axis=-1)                       # log G_j within chunk

    # ratio[j, i] = G_j / G_i for i <= j, else 0. Bounded by 1, so no overflow.
    diff = cum[..., :, None] - cum[..., None, :]
    causal = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    ratio = mx.where(causal, mx.exp(mx.where(causal, diff, NEG_INF)), 0.0)
    eye = mx.eye(chunk, dtype=mx.float32)
    strict = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), -1)
    gamma = mx.exp(cum)                                   # G_j, only used scaled by S_0

    ys, s = [], state.astype(mx.float32)
    for c in range(n_chunks):
        qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]           # [B,H,C,D]
        bc, gam, rat = beta[:, :, c], gamma[:, :, c], ratio[:, :, c]

        kkt = kc @ mx.swapaxes(kc, -1, -2)                        # [B,H,C,C]
        # system matrix: I + b_j (k_i . k_j)(G_j/G_i) for i < j
        sys = eye + mx.where(strict, bc[..., :, None] * kkt * rat, 0.0)
        inv = unit_tri_inv(sys)

        ks0 = kc @ mx.swapaxes(s, -1, -2)                         # [B,H,C,Dv]
        rhs = bc[..., None] * (vc - gam[..., None] * ks0)
        u = inv @ rhs                                             # scaled pseudo-values

        qk = mx.where(causal, (qc @ mx.swapaxes(kc, -1, -2)) * rat, 0.0)
        ys.append(gam[..., None] * (qc @ mx.swapaxes(s, -1, -2)) + qk @ u)

        tail = rat[..., -1, :]                                    # G_C / G_i
        s = gam[..., -1, None, None] * s + mx.swapaxes(tail[..., None] * u, -1, -2) @ kc

    y = mx.stack(ys, axis=2).reshape(b_sz, hv, t_pad, dv)
    if n_pad:
        y = y[:, :, :t_len]
    return mx.swapaxes(y, 1, 2).astype(v.dtype), s


def install(chunk: int = 64) -> bool:
    """Route mlx-lm's gated-delta training path through the chunkwise form.

    Inference is untouched: that path uses a Metal kernel and never calls
    `gated_delta_ops`. Anything the chunkwise form does not cover -- a padding
    mask, vectorised gating -- falls back to the original implementation, so
    installing this can change speed and memory but not results.
    """
    import mlx_lm.models.gated_delta as gd

    if getattr(gd, "_odistil_chunkwise", False):
        return False
    original = gd.gated_delta_ops

    def dispatch(q, k, v, g, beta, state=None, mask=None):
        if mask is None and g.ndim == 3:
            return gated_delta_chunkwise(q, k, v, g, beta, state, chunk=chunk)
        return original(q, k, v, g, beta, state, mask)

    gd.gated_delta_ops = dispatch
    gd._odistil_chunkwise = True
    gd._odistil_original_ops = original
    return True


def uninstall() -> bool:
    import mlx_lm.models.gated_delta as gd

    if not getattr(gd, "_odistil_chunkwise", False):
        return False
    gd.gated_delta_ops = gd._odistil_original_ops
    gd._odistil_chunkwise = False
    return True
