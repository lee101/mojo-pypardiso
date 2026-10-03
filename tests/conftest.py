"""Shared fixtures and matrix generators for the parity tests."""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="session")
def upstream():
    """The real upstream `pypardiso` solver, or skip if MKL is unavailable."""
    pypardiso = pytest.importorskip("pypardiso")
    try:
        return pypardiso.PyPardisoSolver()
    except Exception as exc:  # pragma: no cover - depends on the host MKL
        pytest.skip(f"upstream pypardiso unavailable: {exc}")


@pytest.fixture(scope="session")
def ours():
    from mojo_pypardiso import PyPardisoSolver
    return PyPardisoSolver


def dense_spd(n, seed=0):
    """Dense diagonally dominant matrix, so it is nonsingular and well scaled."""
    rng = np.random.default_rng(seed)
    return np.eye(n) + 0.2 * rng.standard_normal((n, n))


def dense_nonsym(n, seed=0):
    """Dense matrix with a weak diagonal, so partial pivoting is exercised."""
    rng = np.random.default_rng(seed)
    return np.eye(n) * 0.01 + rng.standard_normal((n, n))


def banded(n, seed=0):
    """Tridiagonal matrix: no fill in the factorisation."""
    main = 4.0 + np.random.default_rng(seed).uniform(0, 1, n)
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def random_sparse(n, density=0.01, seed=0, shift=5.0):
    """Random sparse matrix shifted so it is comfortably nonsingular."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csr", random_state=seed,
                  data_rvs=lambda k: rng.standard_normal(k))
    A = (A + sp.eye(n, format="csr") * shift).tocsr()
    A.sort_indices()
    return A


def laplacian_2d(nx, ny):
    """5-point Laplacian on an nx by ny grid; sparse, mildly ill conditioned."""
    n = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            rows.append(k); cols.append(k); vals.append(4.0)
            if i > 0:
                rows.append(k); cols.append(k - ny); vals.append(-1.0)
            if i < nx - 1:
                rows.append(k); cols.append(k + ny); vals.append(-1.0)
            if j > 0:
                rows.append(k); cols.append(k - 1); vals.append(-1.0)
            if j < ny - 1:
                rows.append(k); cols.append(k + 1); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sort_indices()
    return A


def csr(A):
    A = sp.csr_matrix(A)
    A.sort_indices()
    return A