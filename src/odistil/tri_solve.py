"""Fused unit-triangular solve for the chunkwise gated delta rule.

The chunkwise form needs ``u = (I + L)^-1 rhs`` once per chunk. Forming the
inverse and multiplying costs two steps and, in the matmul formulation, six
levels of block recursion -- profiled at 5.07 ms of the ~9.3 ms spent per chunk
at T=2048. Solving directly avoids ever materialising the inverse.

Both directions are needed: forward substitution for the value, and a
transposed (back) substitution for the gradient, since

    u = A^-1 b   =>   b_bar = A^-T u_bar,  A_bar = -b_bar u^T  (strictly lower)

which is why this is wrapped in `mx.custom_function` rather than left to
autodiff -- there is no autodiff path through a hand-written Metal kernel.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    uint m = thread_position_in_grid.z;      // which matrix
    uint c = thread_position_in_grid.x;      // which right-hand-side column
    if (c >= N) return;

    const device T* a = A + m * C * C;
    const device T* b = Bm + m * C * N;
    device T* x = X + m * C * N;

    if (UPPER) {
        // A^T is unit upper triangular: back substitution
        for (int j = C - 1; j >= 0; --j) {
            float acc = static_cast<float>(b[j * N + c]);
            for (int i = j + 1; i < C; ++i) {
                acc -= static_cast<float>(a[i * C + j]) * static_cast<float>(x[i * N + c]);
            }
            x[j * N + c] = static_cast<T>(acc);
        }
    } else {
        // A is unit lower triangular: forward substitution
        for (int j = 0; j < C; ++j) {
            float acc = static_cast<float>(b[j * N + c]);
            for (int i = 0; i < j; ++i) {
                acc -= static_cast<float>(a[j * C + i]) * static_cast<float>(x[i * N + c]);
            }
            x[j * N + c] = static_cast<T>(acc);
        }
    }
"""


def _make_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="unit_tri_solve",
        input_names=["A", "Bm"],
        output_names=["X"],
        source=_SOURCE,
    )


_KERNEL = _make_kernel()


def _solve_kernel(a: mx.array, b: mx.array, upper: bool) -> mx.array:
    """Solve A x = b (or A^T x = b) for unit-triangular A, batched."""
    *batch, c_dim, n_dim = b.shape
    n_mat = 1
    for d in batch:
        n_mat *= d
    a_flat = a.reshape(n_mat, c_dim, c_dim)
    b_flat = b.reshape(n_mat, c_dim, n_dim)
    threads = min(n_dim, 256)
    (out,) = _KERNEL(
        inputs=[a_flat, b_flat],
        template=[("T", b.dtype), ("C", c_dim), ("N", n_dim), ("UPPER", upper)],
        grid=(((n_dim + threads - 1) // threads) * threads, 1, n_mat),
        threadgroup=(threads, 1, 1),
        output_shapes=[b_flat.shape],
        output_dtypes=[b.dtype],
        init_value=0,
    )
    return out.reshape(*batch, c_dim, n_dim)


def _solve_ops(a: mx.array, b: mx.array, upper: bool) -> mx.array:
    """Portable fallback: substitution written with array ops."""
    c_dim = a.shape[-1]
    rows = []
    order = range(c_dim - 1, -1, -1) if upper else range(c_dim)
    acc: dict[int, mx.array] = {}
    for j in order:
        x = b[..., j, :]
        if upper:
            for i in range(j + 1, c_dim):
                x = x - a[..., i, j][..., None] * acc[i]
        else:
            for i in range(j):
                x = x - a[..., j, i][..., None] * acc[i]
        acc[j] = x
    rows = [acc[j] for j in range(c_dim)]
    return mx.stack(rows, axis=-2)


@mx.custom_function
def unit_tri_solve(a: mx.array, b: mx.array) -> mx.array:
    """Solve ``(I + strictly_lower(a)) x = b``.

    ``a`` is [..., C, C] and assumed unit lower triangular; only its strictly
    lower entries are read. ``b`` is [..., C, N].
    """
    if _KERNEL is not None and mx.default_device() == mx.gpu:
        return _solve_kernel(a, b, upper=False)
    return _solve_ops(a, b, upper=False)


@unit_tri_solve.vjp
def _unit_tri_solve_vjp(primals, cotangent, output):
    a, _ = primals
    x = output
    # b_bar solves A^T y = cotangent
    if _KERNEL is not None and mx.default_device() == mx.gpu:
        b_bar = _solve_kernel(a, cotangent, upper=True)
    else:
        b_bar = _solve_ops(a, cotangent, upper=True)
    # A_bar = -b_bar x^T, and only the strictly lower part of A was used
    a_bar = -(b_bar @ mx.swapaxes(x, -1, -2))
    a_bar = mx.tril(a_bar, -1)
    return a_bar, b_bar
