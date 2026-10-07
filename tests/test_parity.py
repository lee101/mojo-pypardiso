"""Numerical parity against the real upstream pypardiso (MKL PARDISO) and scipy.

Every assertion here is a numerical one: the port must produce the same solution
vectors as the MKL solver it replaces, and must reproduce `L @ U == P @ A` for the
factors it hands back.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import (banded, csr, dense_nonsym, dense_spd, laplacian_2d,
                      random_sparse)


def reconstruct(factors, n):
    """Rebuild dense L and U from the CSR factor arrays the kernel returns."""
    L = np.eye(n)
    U = np.zeros((n, n))
    for r in range(n):
        for t in range(factors["l_ptr"][r], factors["l_ptr"][r + 1]):
            L[r, factors["l_idx"][t]] = factors["l_val"][t]
        for t in range(factors["u_ptr"][r], factors["u_ptr"][r + 1]):
            U[r, factors["u_idx"][t]] = factors["u_val"][t]
        U[r, r] = factors["u_diag"][r]
    return L, U


MATRICES = {
    "1x1": csr(np.array([[2.0]])),
    "2x2": csr(np.array([[2.0, 1.0], [1.0, 3.0]])),
    "3x3_pivot": csr(np.array([[2.0, 1.0, 0.0],
                               [3.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0]])),
    "5x5": csr(dense_spd(5)),
    "20x20": csr(dense_spd(20, 1)),
    "40x20_nonsym": csr(dense_nonsym(20, 2)),
    "80x80": csr(dense_spd(80, 3)),
    "banded_200": banded(200, 4),
    "laplacian_20x20": laplacian_2d(20, 20),
    "random_sparse_300": random_sparse(300, 0.01, 5),
}


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_factorization_reproduces_permuted_matrix(ours, name):
    """`L @ U == A[perm]` to machine precision, which is the whole contract."""
    A = MATRICES[name]
    n = A.shape[0]
    factors = ours()._analyze(A)
    L, U = reconstruct(factors, n)

    assert sorted(factors["perm"].tolist()) == list(range(n))
    assert np.abs(L @ U - A.toarray()[factors["perm"]]).max() < 1e-9


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_solve_matches_upstream(ours, upstream, name):
    """Same solution vectors as the MKL PARDISO solver this port replaces."""
    A = MATRICES[name]
    n = A.shape[0]
    rng = np.random.default_rng(7)
    b = rng.standard_normal(n)

    mine = ours().solve(A, b)
    theirs = upstream.solve(A, b)

    assert mine.shape == theirs.shape
    scale = max(1.0, np.abs(theirs).max())
    assert np.abs(mine - theirs).max() / scale < 1e-9


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_residual_is_zero(ours, name):
    """`A @ x - b` small relative to `b`, independently of any other solver."""
    A = MATRICES[name]
    n = A.shape[0]
    b = np.random.default_rng(8).standard_normal(n)

    x = ours().solve(A, b)
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert residual < 1e-10


@pytest.mark.parametrize("n,nrhs", [(30, 1), (30, 4), (60, 2), (60, 9), (120, 7)])
def test_multiple_right_hand_sides_match_upstream(ours, upstream, n, nrhs):
    A = csr(dense_spd(n, 11))
    B = np.random.default_rng(12).standard_normal((n, nrhs))

    mine = ours().solve(A, B)
    theirs = upstream.solve(A, B)

    assert mine.shape == (n, nrhs)
    assert np.abs(mine - theirs).max() < 1e-9


def test_threaded_solve_matches_serial_solve(ours):
    """`mpp_lu_solve` splits right-hand sides over workers; check both agree."""
    import os
    A = csr(dense_spd(50, 13))
    B = np.random.default_rng(14).standard_normal((50, 16))
    assert os.cpu_count() > 1

    ours().solve(A, B)  # warm the module-level lib
    threaded = ours().solve(A, B)
    serial_solver = ours()
    from mojo_pypardiso import _lib
    real = _lib.nworkers
    _lib.nworkers = lambda: 1
    try:
        serial = serial_solver.solve(A, B)
    finally:
        _lib.nworkers = real

    assert np.abs(threaded - serial).max() == 0.0


def test_default_worker_count_is_capped_and_overridable(monkeypatch):
    """The default worker count must stay bounded, and the env override must win.

    `mpp_lu_solve` launches a MAX thread pool. The MAX runtime aborts the whole
    process when `pthread_create` fails, so a pool sized to every logical CPU
    on a large loaded host kills Python outright rather than returning an error
    the solver could fall back from. The cap is what prevents that, so it is
    pinned here rather than left as a tuning constant nobody re-checks.
    """
    import os

    from mojo_pypardiso import _lib

    monkeypatch.delenv("MOJO_PYPARDISO_THREADS", raising=False)
    default = _lib.nworkers()
    assert 1 <= default <= _lib._MAX_WORKERS

    monkeypatch.setenv("MOJO_PYPARDISO_THREADS", "1")
    assert _lib.nworkers() == 1

    monkeypatch.setenv("MOJO_PYPARDISO_THREADS", "3")
    assert _lib.nworkers() == 3

    # the override is validated, not passed straight through: a nonsense value
    # would otherwise reach the kernel as a zero worker count
    monkeypatch.setenv("MOJO_PYPARDISO_THREADS", "0")
    assert _lib.nworkers() == 1
    assert os.cpu_count() is None or os.cpu_count() >= 1


def test_csc_input_matches_csr_input(ours, upstream):
    """`_check_A`'s CSC branch: the explicit transposition is the same system."""
    A = csr(dense_spd(40, 15))
    b = np.random.default_rng(16).standard_normal(40)

    via_csr = ours().solve(A, b)
    via_csc = ours().solve(sp.csc_matrix(A), b)
    reference = upstream.solve(sp.csc_matrix(A), b)

    assert np.abs(via_csc - via_csr).max() == 0.0
    assert np.abs(via_csc - reference).max() < 1e-9


def widen_indices(M, dtype):
    """Re-cast a matrix's index arrays, as scipy does once a matrix gets big."""
    M = M.copy()
    M.indices = M.indices.astype(dtype)
    M.indptr = M.indptr.astype(dtype)
    return M


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_int64_index_arrays_are_narrowed_not_misread(ours, index_dtype, fmt):
    """The kernels read and write 32-bit indices, so wider ones must be narrowed.

    A CSC matrix whose `indices` is int64 used to be handed to
    `mpp_csr_transpose` with an int64 destination, and the kernel wrote Int32
    into the low half of each slot and left the high half at zero. That is
    silently wrong indices rather than an error, so it has to be pinned.
    """
    n = 40
    A = csr(dense_nonsym(n, 21))
    b = np.random.default_rng(22).standard_normal(n)
    M = widen_indices(A if fmt == "csr" else sp.csc_matrix(A), index_dtype)
    assert M.indices.dtype == index_dtype

    x = ours().solve(M, b)
    assert np.abs(A @ x - b).max() < 1e-9


def test_solve_handles_every_right_hand_side_stride(ours):
    """`_call_pardiso` flattens b column-major; C, F and strided views must agree."""
    n = 9
    A = csr(dense_spd(n, 23))  # well conditioned, so an absolute tolerance is meaningful

    for b in (np.ascontiguousarray(np.random.default_rng(24).standard_normal((n, 5))),
              np.asfortranarray(np.random.default_rng(24).standard_normal((n, 5))),
              np.random.default_rng(25).standard_normal((n, 10))[:, ::2]):
        reference = np.linalg.solve(A.toarray(), np.asarray(b, dtype=float))
        assert np.abs(ours().solve(A, b) - reference).max() < 1e-9


def test_repeated_solve_reuses_the_stored_factorisation(ours):
    """`solve(A, b)` twice on the same object must take the phase 33 path.

    The fast path in `_is_already_factorized` shortens the answer for the
    documented pattern -- one `factorize` followed by many `solve` calls with the
    same object -- but only for a matrix small enough that the content
    comparison behind it is guaranteed to have read every element. This pins
    both halves: that the fast path still reports the stored factorisation as
    reusable, and that it produces the same answer as a fresh solver.
    """
    A = csr(dense_spd(20, 31))
    assert A.nnz <= 4096                       # inside the fast path's bound
    rng = np.random.default_rng(32)
    b1, b2 = rng.standard_normal(20), rng.standard_normal(20)

    reused = ours()
    reused.solve(A, b1)                      # factorises on the first call
    assert reused._is_already_factorized(A) is True

    from_scratch = ours().solve(A, b2)
    reused.solve(A, b2)
    assert reused.phase == 33                # the second call must not refactorise
    assert np.abs(reused.solve(A, b2) - from_scratch).max() == 0.0


def test_the_same_object_above_the_fast_path_bound_still_reuses(ours):
    """The bound is a size limit on the fast path, not a correctness limit.

    A matrix too large for the fast path to trust takes the content comparison
    on every call, which is what upstream does unconditionally, and has to reach
    the same answer.
    """
    A = csr(dense_spd(80, 41))
    assert A.nnz > 4096
    b = np.random.default_rng(43).standard_normal(80)

    solver = ours()
    first = solver.solve(A, b)
    assert solver._is_already_factorized(A) is True
    solver.solve(A, b)
    assert solver.phase == 33
    assert np.abs(solver.solve(A, b) - first).max() == 0.0


def test_a_matrix_mutated_in_place_is_not_taken_as_factorized(ours):
    """The identity fast path must not serve a matrix whose contents changed.

    `solve` may be handed the same object after its contents changed -- an
    in-place `A[i, j] = v`, a rescale, an elimination sweep -- and scipy's
    sparse setters rewrite `data` in the same buffer without replacing it, so
    the object's identity survives a matrix that is no longer the one that was
    factorised. Serving it from the stored factors returns the solution of a
    different system with nothing to show that anything is wrong.
    """
    A = csr(dense_spd(20, 35))
    b = np.ones(20)

    solver = ours()
    stale = solver.solve(A, b)
    assert solver._is_already_factorized(A) is True

    A[0, 0] += 5.0                           # same object, new contents
    assert solver._is_already_factorized(A) is False

    x = solver.solve(A, b)
    assert solver.phase == 13                 # refactorised rather than reused
    assert np.abs(x - stale).max() > 1e-6
    assert np.abs(A @ x - b).max() < 1e-9


def test_identity_fast_path_does_not_survive_a_release(ours):
    """Dropping the factorisation must clear the object identity with it.

    The fast path in `_is_already_factorized` is only valid while the factors it
    refers to are still stored, so `remove_stored_factorization` has to forget
    the object too. Otherwise a released solver would report phase 33 for a
    matrix whose factors are gone.
    """
    A = csr(dense_spd(20, 33))
    b = np.random.default_rng(34).standard_normal(20)

    solver = ours()
    solver.solve(A, b)
    assert solver._is_already_factorized(A) is True

    solver.remove_stored_factorization()
    assert solver._factorized_obj is None
    assert solver._is_already_factorized(A) is False
    # and a released solver still solves, by factorising again
    assert np.abs(A @ solver.solve(A, b) - b).max() < 1e-9


def test_a_changed_matrix_is_not_taken_as_factorized(ours):
    """An equal-but-distinct matrix is the case the fast path does not cover.

    The fast path needs object identity, so a matrix that merely happens to be
    equal to the stored one still takes the content comparison and is reported
    as reusable -- and one that differs is not.
    """
    A = csr(dense_spd(20, 35))
    solver = ours()

    solver.factorize(A)
    assert solver._is_already_factorized(A.copy()) is True
    assert solver._is_already_factorized(csr(dense_spd(20, 37))) is False


def test_uninitialised_solve_buffers_are_fully_written(ours):
    """`solve` allocates `pb` and `x` with `np.empty`; both must be fully written.

    `mpp_permute_rhs` writes every element of `pb` and `mpp_lu_solve` writes
    every element of `x` before reading it, which is what makes skipping the
    zeroing safe. A right-hand side full of values that would survive as poison
    if the kernels left anything behind pins that.
    """
    n = 50
    A = csr(dense_spd(n, 38))
    rng = np.random.default_rng(39)

    for nrhs in (1, 2, 5):
        b = rng.standard_normal(n if nrhs == 1 else (n, nrhs))
        x = ours().solve(A, b)
        reference = np.linalg.solve(A.toarray(), np.asarray(b, dtype=float))
        assert x.shape == np.asarray(b).shape
        assert np.abs(x - reference).max() < 1e-9


def test_one_dimensional_right_hand_side_keeps_its_shape(ours):
    """A 1D `b` is a single column, so it is passed through without a copy.

    `_check_b` no longer forces a 1D `b` into Fortran order and `_call_pardiso`
    reshapes it directly, so a flat right-hand side and the same vector as a
    single column have to give the same answer.
    """
    n = 25
    A = csr(dense_spd(n, 40))
    v = np.random.default_rng(41).standard_normal(n)
    solver = ours()

    flat = solver.solve(A, v)
    column = solver.solve(A, v.reshape(-1, 1))

    assert flat.shape == (n,)
    assert column.shape == (n, 1)
    assert np.abs(flat - column.ravel()).max() == 0.0


def test_indptr_diff_scratch_is_reused_across_sizes(ours):
    """`_check_A` reuses one scratch buffer for the indptr-difference kernel.

    The buffer is grown on demand, so checking a small matrix and then a larger
    one must still see the larger matrix's differences rather than a truncated
    or stale scratch, and an empty row must still be found afterwards.
    """
    solver = ours()
    small = csr(dense_spd(20, 42))
    large = csr(dense_spd(200, 43))

    assert solver._empty_rows(small) is False
    assert solver._diff_scratch is not None
    assert solver._empty_rows(large) is False
    assert len(solver._diff_scratch) >= 200

    # a square matrix whose last row is empty, checked after the scratch grew
    data = np.ones(20)
    indices = np.arange(20)
    indptr = np.concatenate([np.arange(20), np.full(2, 20)])
    singular = sp.csr_matrix((data, indices, indptr), shape=(21, 21))
    assert solver._empty_rows(singular) is True


def random_family(rng, n, kind):
    """One matrix per structural family the solver has to handle."""
    if kind == 0:                                    # well conditioned
        return np.eye(n) + 0.3 * rng.standard_normal((n, n))
    if kind == 1:                                    # weak diagonal, forces pivoting
        return 0.01 * np.eye(n) + rng.standard_normal((n, n))
    if kind == 2:                                    # tridiagonal, no fill
        return (np.diag(rng.uniform(3, 5, n))
                + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1))
    M = np.diag(rng.uniform(3, 5, n))               # pentadiagonal
    for d in (1, 2, -1, -2):
        M += np.diag(rng.uniform(-1, 1, n - abs(d)), d)
    return M


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_randomised_family_sweep_matches_dense_solve(ours, kind, fmt):
    """Sweep the four structural families in both formats against ground truth.

    `numpy.linalg.solve` is the oracle rather than upstream MKL, because upstream
    is itself wrong on the CSC path for a pivoting-heavy matrix: given the CSC
    arrays with `iparm(12) = 1` it returned a solution with a residual of ~4 on
    this family while the CSR path on the same matrix was exact. The explicit
    `mpp_csr_transpose` in this port does not reproduce that, which is why the
    check here is against a dense solve.
    """
    rng = np.random.default_rng(2024 + kind)
    for _ in range(8):
        n = int(rng.integers(2, 40))
        M = random_family(rng, n, kind)
        if abs(np.linalg.det(M)) < 1e-10:           # keep it nonsingular
            M = M + 3 * np.eye(n)
        A = csr(M)
        nrhs = int(rng.integers(1, 6))
        b = rng.standard_normal(n if nrhs == 1 else (n, nrhs))
        M_in = sp.csc_matrix(A) if fmt == "csc" else A

        x = ours().solve(M_in, b)
        reference = np.linalg.solve(A.toarray(), np.asarray(b, dtype=float))
        assert np.abs(x - reference).max() < 1e-9


def test_csr_transpose_kernel_matches_scipy(ours):
    """`mpp_csr_transpose` is the same conversion as `csc_matrix.tocsr()`."""
    from mojo_pypardiso import _lib
    rng = np.random.default_rng(17)
    n = 60
    C = sp.random(n, n, density=0.05, format="csc", random_state=17,
                  data_rvs=lambda k: rng.standard_normal(k))

    indptr = np.ascontiguousarray(C.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(C.indices, dtype=np.int32)
    data = np.ascontiguousarray(C.data, dtype=np.float64)
    dst_ptr = np.zeros(n + 1, dtype=np.int32)
    dst_idx = np.zeros(C.nnz, dtype=np.int32)
    dst_val = np.zeros(C.nnz, dtype=np.float64)
    cursor = np.zeros(n, dtype=np.int32)

    err = _lib.lib().mpp_csr_transpose(
        _lib.addr(indptr), _lib.addr(indices), _lib.addr(data),
        _lib.addr(dst_ptr), _lib.addr(dst_idx), _lib.addr(dst_val),
        n, C.nnz, _lib.addr(cursor))

    assert err == 0
    got = sp.csr_matrix((dst_val, dst_idx, dst_ptr), shape=(n, n))
    assert np.array_equal(got.indptr, C.tocsr().indptr)
    assert np.array_equal(got.indices, C.tocsr().indices)
    assert np.allclose(got.data, C.tocsr().data)


def test_indptr_diff_kernel_matches_numpy(ours):
    """`_check_A`'s `np.diff(A.indptr)` is evaluated by the Mojo kernel."""
    from mojo_pypardiso import _lib
    A = csr(dense_spd(50, 18))
    n = A.shape[0]

    indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
    diff = np.zeros(n, dtype=np.int32)
    zeros = _lib.lib().mpp_indptr_diffs(_lib.addr(indptr), _lib.addr(diff), n)

    assert np.array_equal(diff, np.diff(A.indptr))
    assert (zeros == 0) == bool(np.diff(A.indptr).all())


def test_indptr_diff_kernel_finds_the_empty_row():
    from mojo_pypardiso import _lib
    A = sp.csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, 2))
    n = A.shape[0]
    indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
    diff = np.zeros(n, dtype=np.int32)
    assert _lib.lib().mpp_indptr_diffs(_lib.addr(indptr), _lib.addr(diff), n) == 0

    # a row with no entries makes the square matrix singular
    B = sp.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))
    indptr = np.ascontiguousarray(B.indptr, dtype=np.int32)
    assert _lib.lib().mpp_indptr_diffs(_lib.addr(indptr), _lib.addr(diff), n) == 1


def test_permute_rhs_kernel_matches_manual_gather(ours):
    """`mpp_permute_rhs` is exactly `Pb[i] = b[perm[i]]`."""
    from mojo_pypardiso import _lib
    n, nrhs = 37, 5
    perm = np.random.default_rng(19).permutation(n).astype(np.int32)
    b = np.asfortranarray(np.random.default_rng(20).standard_normal((n, nrhs)))

    bf = np.ascontiguousarray(b, dtype=np.float64).reshape(-1, order="F")
    pb = np.zeros_like(bf)
    _lib.lib().mpp_permute_rhs(n, nrhs, _lib.addr(perm), _lib.addr(bf), _lib.addr(pb))

    # the buffer is column-major, so column r starts at r * n
    expected = np.concatenate([bf[r * n:(r + 1) * n][perm] for r in range(nrhs)])
    assert np.array_equal(pb, expected)


def test_nnzf_counts_every_stored_factor_entry(ours):
    """`iparm(18)` is nnz(L) + nnz(U) with the split-out diagonal counted once."""
    for name in ("3x3_pivot", "20x20", "banded_200", "laplacian_20x20"):
        A = MATRICES[name]
        n = A.shape[0]
        f = ours()._analyze(A)
        stored = (f["l_ptr"][n] - f["l_ptr"][0]) + (f["u_ptr"][n] - f["u_ptr"][0]) + n
        assert f["nnzf"] == stored
        assert f["nnzf"] >= A.nnz


def test_symbolic_and_numeric_agree_on_a_matrix_that_needs_no_pivoting(ours):
    """The phase 22 walk builds the fill pattern with the arithmetic suppressed.

    Step 5 of `mpp_lu_factorize` is the only place the two walks share
    structure but not arithmetic, so this pins that the structural pass and the
    numeric pass create exactly the same entries on a matrix whose pivot rule
    they agree on. A row-permuted matrix keeps the diagonal dominant, so the
    numeric partial pivot also takes the diagonal, and the two walks must then
    produce identical index patterns.
    """
    rng = np.random.default_rng(23)
    n = 50
    A = csr(np.eye(n) + 0.1 * rng.standard_normal((n, n)))
    numeric = ours()._analyze(A, symbolic=False)
    symbolic = ours()._analyze(A, symbolic=True)

    for key in ("l_ptr", "u_ptr", "l_idx", "u_idx", "perm"):
        assert np.array_equal(numeric[key][:A.shape[0] + 1], symbolic[key][:A.shape[0] + 1])
    n_entries = int(numeric["nnzf"])
    # the index arrays beyond nnzf are scratch, so compare only what is filled
    assert np.array_equal(numeric["l_idx"][:n_entries], symbolic["l_idx"][:n_entries])
    assert np.array_equal(numeric["u_idx"][:n_entries], symbolic["u_idx"][:n_entries])


def test_update_survives_a_row_that_needs_fill(ours):
    """Step 5 streams the pivot row from the contiguous sorted U run.

    The fill path is only reached when the target row does not already hold a
    column of the pivot row, so a matrix that provably creates fill exercises
    the branch that appends to the target row's linked list from the streamed
    run rather than the entry workspace.
    """
    n = 40
    M = np.eye(n) * 0.01
    for i in range(n - 1):
        M[i, i + 1] = 1.0
        M[i + 1, i] = 1.0
    A = csr(M)
    factors = ours()._analyze(A)
    L, U = reconstruct(factors, n)
    # partial pivoting interleaves the weak diagonal and must create fill
    assert factors["nnzf"] > A.nnz
    assert np.abs(L @ U - A.toarray()[factors["perm"]]).max() < 1e-9


def test_phase22_reports_the_static_ordering_fill(ours):
    """Phase 22 is value independent, so it must not see the numeric fill."""
    n = 60
    M = np.eye(n) * 0.01
    for i in range(n - 1):
        M[i, i + 1] = 1.0
        M[i + 1, i] = 1.0
    A = csr(M)
    b = np.zeros(n)

    analysis = ours()
    analysis.set_phase(22)
    analysis._call_pardiso(A, b)
    symbolic_nnz = int(analysis.get_iparm(18))

    numeric = ours()
    numeric._call_pardiso(A, b)
    numeric_nnz = int(numeric.get_iparm(18))

    # the static ordering takes the diagonal every time, so it needs no fill
    assert symbolic_nnz == A.nnz
    # partial pivoting has to interleave the weak diagonal, which does fill
    assert numeric_nnz > symbolic_nnz
