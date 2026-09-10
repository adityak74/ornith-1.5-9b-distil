Builds on #1 — please review that first; this branch contains its commit.

## Problem

With the chunkwise path from #1, the per-chunk unit lower-triangular inverse dominates. Profiled at T=2048, C=64, Hv=32:

| step | time |
|---|---:|
| `k @ k^T` | 1.39 ms |
| build the triangular system | 1.37 ms |
| **unit-triangular inverse** | **5.07 ms** |
| `inv @ rhs` | 0.62 ms |
| masked `(q @ k^T) * ratio` | 0.83 ms |

The block recursion is six levels of batched matmuls over 64x64 matrices, and the inverse is used exactly once, as `inv @ rhs`. Solving directly avoids materializing it.

## Change

`_unit_tri_solve` runs substitution in a single Metal kernel: one thread per right-hand-side column per matrix, parallel across the batch. `mx.custom_function` supplies the gradient, which autodiff cannot derive through a hand-written kernel:

    x = A^-1 b   =>   b_bar = A^-T x_bar,   A_bar = tril(-b_bar x^T, -1)

so the backward pass is the transposed substitution plus an outer product — the same kernel with `UPPER` flipped. Without Metal it falls back to the block-recursion inverse from #1, so CPU behaviour is unchanged.

## Result

One GDN layer (Hk=16, Hv=32, Dk=Dv=128), forward and backward, M4 Pro:

| seq | #1 (block-recursion inverse) | this PR (fused solve) | speedup |
|---:|---:|---:|---:|
| 1024 | 0.97 GB / 0.09 s | 1.00 GB / 0.03 s | 2.7x |
| 2048 | 1.74 GB / 0.23 s | 1.75 GB / 0.09 s | 2.5x |
| 4096 | 4.73 GB / 0.60 s | 4.60 GB / 0.29 s | 2.1x |

Cumulative against the sequential loop on current `main`:

| seq | `main` | #1 + this | total |
|---:|---:|---:|---|
| 1024 | 10.18 GB / 1.16 s | 1.00 GB / 0.03 s | 10x mem, 39x speed |
| 2048 | 31.44 GB / 4.20 s | 1.75 GB / 0.09 s | 18x mem, 47x speed |
| 4096 | 108.33 GB / 39.02 s | 4.60 GB / 0.29 s | **24x mem, 135x speed** |

## Testing

Adds `test_fused_solve_matches_matmul_inverse`: the kernel against the portable path, the residual `A x - b`, and both gradients against autodiff through the block-recursion inverse. All agree to ~1e-5 or better. The chunkwise tests from #1 and the existing kernel tests pass unchanged (12 passed, 32 subtests).
