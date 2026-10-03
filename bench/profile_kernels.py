"""Kernel-level profile: where does the time inside one `solve` go?

Not part of the benchmark table. Run with `pixi run python bench/profile_kernels.py`.
Times the individual stages of the pipeline at the sizes bench/bench.py uses, so
the optimisation target is chosen from measurement rather than from the table.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_pypardiso as mp  # noqa: E402
from mojo_pypardiso import _lib  # noqa: E402


def best(fn, repeat=3):
    out = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


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


def banded(n):
    main = 4.0 + np.random.default_rng(0).uniform(0, 1, n)
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def stats(f, n):
    lp, ll = f["l_ptr"], f["l_ptr"][1:] - f["l_ptr"][:-1]
    up, ul = f["u_ptr"], f["u_ptr"][1:] - f["u_ptr"][:-1]
    total = int(ll.sum() + ul.sum())
    return total, float(ll.mean()), float(ul.mean()), int(max(ll.max(), ul.max()))


def report(label, A, nrhs=1):
    n = A.shape[0]
    b = np.random.default_rng(1).standard_normal((n, nrhs)) if nrhs > 1 \
        else np.random.default_rng(1).standard_normal(n)
    solver = mp.PyPardisoSolver()
    lib = solver._lib

    whole = best(lambda: mp.PyPardisoSolver().solve(A, b))
    t_check = best(lambda: solver._check_A(A))
    t_fact = best(lambda: solver._analyze(A, symbolic=False))
    t_sym = best(lambda: solver._analyze(A, symbolic=True))
    f = solver._analyze(A)
    solver._factors = f
    nnz, ml, mu, mx = stats(f, n)

    bf = np.ascontiguousarray(b, dtype=np.float64).reshape(-1, order="F")
    perm = f["perm"]
    pb = np.zeros_like(bf)
    xf = np.zeros_like(pb)
    t_perm = best(lambda: lib.mpp_permute_rhs(n, nrhs, _lib.addr(perm), _lib.addr(bf), _lib.addr(pb)))
    args = solver._factor_args()
    t_solve = best(lambda: lib.mpp_lu_solve(n, nrhs, *args, _lib.addr(pb), _lib.addr(xf), _lib.nworkers()))
    t_solve1 = best(lambda: lib.mpp_lu_solve(n, nrhs, *args, _lib.addr(pb), _lib.addr(xf), 1))

    # python-only cost of the factorise() front end (alloc + copies, no kernel)
    def alloc_only():
        nnzv = A.nnz
        [np.zeros(max(2 * nnzv, n + 16, 1024), dtype=np.float64) for _ in range(3)]
        [np.zeros(max(2 * nnzv, n + 16, 1024), dtype=np.int32) for _ in range(7)]
        [np.zeros(n + max(2 * nnzv, n + 16, 1024) * 2, dtype=np.int32)]
    t_alloc = best(alloc_only, 20)

    print(f"{label:34s} n={n:6d} nnz(A)={A.nnz:8d} nnz(L+U)={nnz:9d} "
          f"len L={ml:6.1f} len U={mu:6.1f} max={mx:5d}")
    print(f"  {'whole solve()':26s} {whole*1e3:10.3f} ms")
    print(f"  {'_check_A (python+kernel)':26s} {t_check*1e3:10.3f} ms")
    print(f"  {'_analyze numeric (full)':26s} {t_fact*1e3:10.3f} ms")
    print(f"  {'_analyze symbolic (full)':26s} {t_sym*1e3:10.3f} ms")
    print(f"  {'  of which np.zeros alloc':26s} {t_alloc*1e3:10.3f} ms")
    print(f"  {'mpp_permute_rhs':26s} {t_perm*1e3:10.3f} ms")
    print(f"  {'mpp_lu_solve':26s} {t_solve*1e3:10.3f} ms  (1 worker: {t_solve1*1e3:.3f} ms)")
    print()


if __name__ == "__main__":
    report("dense 200", dense_spd(200, 0))
    report("dense 400", dense_spd(400, 1))
    report("dense 800", dense_spd(800, 2))
    report("dense 500 x 32 rhs", dense_spd(500, 3), nrhs=32)
    report("laplacian 80x80", laplacian_2d(80, 80))
    report("banded 20000", banded(20000))
