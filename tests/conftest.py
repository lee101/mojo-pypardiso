"""Shared fixtures and matrix generators for the parity tests."""

import os

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="session")
def compiled_lib():
    """The compiled Mojo library, or fail loudly if it cannot be produced.
    A missing toolchain skips, because the kernel assertions cannot run without
    it. A toolchain that IS present and then fails the build is a different
    thing: that is a broken source, and skipping it would let `pixi run test`
    exit 0 with every kernel assertion unrun.

    Every assertion below goes through this handle, so a test can only pass by
    actually calling into `dist/libmojo-pypardiso.so`. The path is asserted to be
    the build output and not something else that happens to be on the loader
    path, and `MOJO_PYPARDISO_LIB` is cleared for the duration of the session so
    an env var cannot redirect the tests at a stale library.
    """
    saved = os.environ.pop("MOJO_PYPARDISO_LIB", None)
    try:
        from mojo_pypardiso import _lib

        try:
            path = _lib.build(force=True)
        except Exception as exc:
            # includes a missing mojo binary (OSError), a compiler error
            # (BuildError), and a timeout. Only the first of those is a
            # missing toolchain; anything else is a real build failure.
            if not _toolchain_present():
                pytest.skip(f"could not build the Mojo library: {exc!r}")
            pytest.fail(f"could not build the Mojo library: {exc!r}")
        if not os.path.exists(path):
            pytest.fail(f"compiled Mojo library missing at {path}")
        try:
            lib = _lib.lib()
        except OSError as exc:
            pytest.fail(f"compiled Mojo library would not load: {exc}")
        loaded = os.path.realpath(lib._name)
        if loaded != os.path.realpath(path):
            pytest.fail(f"loaded {loaded} instead of the build output {path}")
        return lib
    finally:
        if saved is not None:
            os.environ["MOJO_PYPARDISO_LIB"] = saved


def _toolchain_present():
    """Whether a `mojo` compiler is reachable at all, ignoring its exit status.

    Decides whether a build failure means "no toolchain here" (skip) or "the
    source does not compile" (fail). Deliberately only looks for the binary:
    `mojo_command()` raising `BuildError` is the absence case, and a binary
    that exists but cannot be forked is a failure worth reporting.
    """
    from mojo_pypardiso import _lib

    if os.environ.get("MOJO_PYPARDISO_MOJO"):
        return True
    try:
        _lib.mojo_command()
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def upstream():
    """The real upstream `pypardiso` solver, or skip if MKL is unavailable."""
    pypardiso = pytest.importorskip("pypardiso")
    try:
        return pypardiso.PyPardisoSolver()
    except Exception as exc:
        pytest.skip(f"upstream pypardiso unavailable: {exc}")


@pytest.fixture(scope="session")
def ours(compiled_lib):
    """The port's solver class, bound to the freshly built shared library."""
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
