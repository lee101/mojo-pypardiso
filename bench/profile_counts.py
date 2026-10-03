"""Count the work the factorisation's step 5 actually does.

The step 5 inner loop runs once per (multiplier, pivot-row-entry) pair, so its
cost is `sum over steps of na(k) * nu(k)`. Printing that product against the
measured time says whether the loop is latency bound (few iterations, each
expensive) or throughput bound (many cheap ones), which decides whether SIMD
can help at all.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_pypardiso as mp  # noqa: E402


def dense_spd(n, seed=0):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix(np.eye(n) + 0.2 * rng.standard_normal((n, n)))


def laplacian_2d(nx, ny):
    n = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            rows.append(k); cols.append(k); vals.append(4.0)
            if i > 0: rows.append(k); cols.append(k - ny); vals.append(-1.0)
            if i < nx - 1: rows.append(k); cols.append(k + ny); vals.append(-1.0)
            if j > 0: rows.append(k); cols.append(k - 1); vals.append(-1.0)
            if j < ny - 1: rows.append(k); cols.append(k + 1); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sort_indices()
    return A


def report(label, A):
    n = A.shape[0]
    f = mp.PyPardisoSolver()._analyze(A)
    ll = (f["l_ptr"][1:] - f["l_ptr"][:-1]).astype(np.int64)
    ul = (f["u_ptr"][1:] - f["u_ptr"][:-1]).astype(np.int64)
    # step k: na = column k live count ~= ll[k] (the multipliers), nu = ul[k]
    # (the U row streamed once per multiplier). The inner loop runs na*nu.
    inner = int((ll * ul).sum())
    # the stamp loop runs once per multiplier over its whole row: ~ sum ll * len(row)
    print(f"{label:20s} n={n:5d} nnz(L+U)={int(ll.sum()+ul.sum()):8d}  "
          f"mean L row={ll.mean():6.1f} mean U row={ul.mean():6.1f}  "
          f"step-5 inner iters={inner:12,d}")


if __name__ == "__main__":
    report("dense 200", dense_spd(200, 0))
    report("dense 400", dense_spd(400, 1))
    report("dense 800", dense_spd(800, 2))
    report("laplacian 80x80", laplacian_2d(80, 80))