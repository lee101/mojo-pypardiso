"""Direct calls into every `@export`ed kernel, through ctypes.

The rest of the suite reaches the kernels through the Python API, which is the
right level for a numerical assertion but hides a lot: a kernel that is wired
up wrong at its own boundary can still produce an answer the wrapper then fixes
up, or an answer that only happens to be right for the shapes the API happens to
produce. Each test here allocates the buffers itself, sizes them itself, calls
the exported symbol by name and asserts on what comes back.
"""

import ctypes
import re

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import csr
from mojo_pypardiso import _lib


def addr(a):
    return a.ctypes.data


class FactorWorkspace:
    """Buffers `mpp_lu_factorize` needs, sized and allocated by the caller.

    `mojo_pypardiso.pardiso_wrapper._analyze` allocates the same arrays with the
    same widths; this spells the layout out so a test can hand the kernel a
    deliberately too-small `cap` and see the `-3` it answers with.
    """

    def __init__(self, n, cap, nnzf_extra=0):
        self.n = n
        self.cap = cap
        self.w = {
            'val': np.zeros(cap, dtype=np.float64),
            'row': np.zeros(cap, dtype=np.int32),
            'col': np.zeros(cap, dtype=np.int32),
            'nextc': np.zeros(cap, dtype=np.int32),
            'nextr': np.zeros(cap, dtype=np.int32),
            'col_head': np.zeros(n + 1, dtype=np.int32),
            'row_head': np.zeros(n, dtype=np.int32),
            'row_label': np.zeros(n, dtype=np.int32),
            'pos': np.zeros(n, dtype=np.int32),
            'tag': np.zeros(n, dtype=np.int32),
            'l_lab': np.zeros(cap, dtype=np.int32),
            'l_entry': np.zeros(cap, dtype=np.int32),
            'scratch': np.zeros(n + 2 * cap, dtype=np.int32),
        }
        self.out = {
            'l_ptr': np.zeros(n + 1, dtype=np.int32),
            'l_idx': np.zeros(cap, dtype=np.int32),
            'l_val': np.zeros(cap, dtype=np.float64),
            'u_ptr': np.zeros(n + 1, dtype=np.int32),
            'u_idx': np.zeros(n + cap, dtype=np.int32),
            'u_val': np.zeros(n + cap, dtype=np.float64),
            'u_diag': np.zeros(n, dtype=np.float64),
            'perm': np.zeros(n, dtype=np.int32),
            'u_cur': np.zeros(n, dtype=np.int32),
            'nnzf': np.zeros(1, dtype=np.int32),
        }

    def call(self, lib, A, symbolic=0):
        """Run the kernel on `A` and return its return code."""
        n = A.shape[0]
        indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        data = np.ascontiguousarray(A.data, dtype=np.float64)
        w, o = self.w, self.out
        return lib.mpp_lu_factorize(
            n, self.cap, symbolic,
            addr(indptr), addr(indices), addr(data),
            addr(o['l_ptr']), addr(o['l_idx']), addr(o['l_val']),
            addr(o['u_ptr']), addr(o['u_idx']), addr(o['u_val']), addr(o['u_diag']),
            addr(o['perm']), addr(o['nnzf']),
            addr(w['val']), addr(w['row']), addr(w['col']),
            addr(w['nextc']), addr(w['nextr']),
            addr(w['col_head']), addr(w['row_head']),
            addr(w['row_label']), addr(w['pos']), addr(w['tag']),
            addr(w['l_lab']), addr(w['scratch']), addr(w['l_entry']),
            addr(o['u_cur']))


def dense_factors(ws, n):
    """Rebuild dense L and U from what the kernel wrote."""
    o = ws.out
    L = np.eye(n)
    U = np.zeros((n, n))
    for r in range(n):
        for t in range(o['l_ptr'][r], o['l_ptr'][r + 1]):
            L[r, o['l_idx'][t]] = o['l_val'][t]
        for t in range(o['u_ptr'][r], o['u_ptr'][r + 1]):
            U[r, o['u_idx'][t]] = o['u_val'][t]
        U[r, r] = o['u_diag'][r]
    return L, U


# --------------------------------------------------------------------------
# mpp_indptr_diffs
# --------------------------------------------------------------------------

def test_indptr_diffs_writes_every_difference(compiled_lib):
    """The kernel must produce exactly `np.diff(indptr)`, and count the zeros."""
    n = 64
    rng = np.random.default_rng(1)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.sort(rng.integers(0, 4 * n, size=n)).astype(np.int32)
    diff = np.full(n, -99, dtype=np.int32)

    zeros = compiled_lib.mpp_indptr_diffs(addr(indptr), addr(diff), n)

    assert np.array_equal(diff, np.diff(indptr).astype(np.int32))
    assert zeros == int((np.diff(indptr) == 0).sum())


def test_indptr_diffs_handles_n_equal_to_one(compiled_lib):
    """The loop is `for i in range(n)` and loads `i + 1`, so `n == 1` is the
    shortest legal call; `indptr` must have room for the two loads."""
    indptr = np.array([7, 9], dtype=np.int32)
    diff = np.zeros(1, dtype=np.int32)
    assert compiled_lib.mpp_indptr_diffs(addr(indptr), addr(diff), 1) == 0
    assert diff.tolist() == [2]


def test_indptr_diffs_handles_n_equal_to_zero(compiled_lib):
    """An empty matrix reaches the kernel with n == 0; it must not touch memory."""
    indptr = np.array([0], dtype=np.int32)
    diff = np.array([12345], dtype=np.int32)
    assert compiled_lib.mpp_indptr_diffs(addr(indptr), addr(diff), 0) == 0
    assert diff.tolist() == [12345]          # untouched, so no write past the end


# --------------------------------------------------------------------------
# mpp_csr_transpose
# --------------------------------------------------------------------------

def test_csr_transpose_round_trips_a_csc_matrix(compiled_lib):
    """`A.T` of a CSC matrix is CSR, and it has to come back exactly."""
    rng = np.random.default_rng(2)
    n = 40
    dense = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.1)
    dense = np.triu(dense, 1)                # strictly upper triangular, CSC friendly
    A = sp.csc_matrix(dense)

    src_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
    src_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
    src_val = np.ascontiguousarray(A.data, dtype=np.float64)
    dst_ptr = np.zeros(n + 1, dtype=np.int32)
    dst_idx = np.zeros(A.nnz, dtype=np.int32)
    dst_val = np.zeros(A.nnz, dtype=np.float64)
    cursor = np.zeros(n, dtype=np.int32)

    err = compiled_lib.mpp_csr_transpose(
        addr(src_ptr), addr(src_idx), addr(src_val),
        addr(dst_ptr), addr(dst_idx), addr(dst_val),
        n, A.nnz, addr(cursor))

    assert err == 0
    got = sp.csr_matrix((dst_val, dst_idx, dst_ptr), shape=(n, n))
    assert np.array_equal(got.indptr, A.tocsr().indptr)
    assert np.array_equal(got.indices, A.tocsr().indices)
    assert np.array_equal(got.data, A.tocsr().data)


def test_csr_transpose_rejects_a_mismatched_entry_count(compiled_lib):
    """The kernel's only defensive return is -1, for formats that disagree.

    `nnz` is passed by the caller, so a wrong one has to be caught rather than
    leaving `indptr[n]` short of `indices`. The matrix here really has 5 stored
    entries, so claiming 6 must be refused. It is built through the COO
    constructor with duplicates kept, because `csc_matrix` of a dense array
    would drop them and change the count.
    """
    n = 3
    A = sp.csc_matrix(([1.0, 2.0, 3.0, 4.0, 5.0],
                       ([0, 1, 1, 0, 2], [0, 1, 2, 1, 0])), shape=(n, n))
    assert A.nnz == 5

    src_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
    src_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
    src_val = np.ascontiguousarray(A.data, dtype=np.float64)
    dst_ptr = np.zeros(n + 1, dtype=np.int32)
    dst_idx = np.zeros(6, dtype=np.int32)
    dst_val = np.zeros(6, dtype=np.float64)
    cursor = np.zeros(n, dtype=np.int32)

    err = compiled_lib.mpp_csr_transpose(
        addr(src_ptr), addr(src_idx), addr(src_val),
        addr(dst_ptr), addr(dst_idx), addr(dst_val),
        n, 6, addr(cursor))                  # claimed nnz, actual nnz is 5

    assert err == -1
    # the kernel returns before writing `dst_ptr[n]`, so the guard really did
    # stop it rather than the buffers happening to agree
    assert dst_ptr[n] == 0


def test_csr_transpose_keeps_duplicates_and_sorts_each_row(compiled_lib):
    """The scatter walks columns in increasing order, so each row of the result
    comes out with ascending column indices, and stored duplicates are kept
    rather than summed -- the conversion is a reindex, not a coalesce.

    The arrays are assembled by hand because both scipy constructors coalesce
    duplicates on the way in, so there is no other way to get one into the
    kernel's input.
    """
    n = 4
    nnz = 3
    src_ptr = np.array([0, 1, 3, 3, 3], dtype=np.int32)      # row 1 holds 3
    src_idx = np.array([1, 1, 1], dtype=np.int32)           # all in column 1
    src_val = np.array([2.0, 3.0, 5.0], dtype=np.float64)

    dst_ptr = np.zeros(n + 1, dtype=np.int32)
    dst_idx = np.zeros(nnz, dtype=np.int32)
    dst_val = np.zeros(nnz, dtype=np.float64)
    cursor = np.zeros(n, dtype=np.int32)

    assert compiled_lib.mpp_csr_transpose(
        addr(src_ptr), addr(src_idx), addr(src_val),
        addr(dst_ptr), addr(dst_idx), addr(dst_val),
        n, nnz, addr(cursor)) == 0

    # the three stored entries all sit in column 1, so all three transpose into
    # row 1, and they come out reindexed rather than summed
    assert dst_ptr.tolist() == [0, 0, 3, 3, 3]
    assert dst_idx.tolist() == [0, 1, 1]
    assert dst_val.tolist() == [2.0, 3.0, 5.0]


# --------------------------------------------------------------------------
# mpp_permute_rhs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nrhs", [1, 2, 5])
def test_permute_rhs_gathers_each_column(compiled_lib, nrhs):
    """`pb[r*n + i] == b[r*n + perm[i]]`, column by column."""
    n = 23
    rng = np.random.default_rng(3)
    perm = rng.permutation(n).astype(np.int32)
    b = rng.standard_normal(n * nrhs)

    pb = np.full(n * nrhs, np.nan)
    compiled_lib.mpp_permute_rhs(n, nrhs, addr(perm), addr(b), addr(pb))

    for r in range(nrhs):
        np.testing.assert_array_equal(pb[r * n:(r + 1) * n], b[r * n:(r + 1) * n][perm])


def test_permute_rhs_with_the_identity_writes_the_input_unchanged(compiled_lib):
    """An identity permutation is the case that catches an uninitialised
    destination buffer, because every slot is overwritten with something a
    pre-zeroed buffer would have got wrong."""
    n = 8
    b = np.arange(1.0, n + 1.0)
    perm = np.arange(n, dtype=np.int32)
    pb = np.full(n, -1.0)
    compiled_lib.mpp_permute_rhs(n, 1, addr(perm), addr(b), addr(pb))
    np.testing.assert_array_equal(pb, b)


# --------------------------------------------------------------------------
# mpp_lu_factorize
# --------------------------------------------------------------------------

def test_lu_factorize_reproduces_the_permuted_matrix(compiled_lib):
    """`L @ U == A[perm]`, read straight out of the kernel's own output arrays."""
    rng = np.random.default_rng(4)
    n = 45
    dense = np.eye(n) * 0.05 + rng.standard_normal((n, n))   # forces pivoting
    A = csr(dense)

    ws = FactorWorkspace(n, cap=max(4 * A.nnz, n + 64))
    assert ws.call(compiled_lib, A) == 0

    L, U = dense_factors(ws, n)
    perm = ws.out['perm']

    assert sorted(perm.tolist()) == list(range(n))
    assert np.abs(L @ U - dense[perm]).max() < 1e-10
    # L is unit lower and U is upper with the diagonal split out, as the solve
    # in the next section relies on
    assert np.array_equal(np.diag(L), np.ones(n))
    assert np.abs(np.tril(U, -1)).max() == 0.0


def test_lu_factorize_pivots_to_the_largest_absolute_entry(compiled_lib):
    """Column 0 of the matrix below is (1, 9), so row 1 must be the pivot and
    `perm[0] == 1`. A kernel that ignored the magnitudes would emit `perm[0] == 0`.
    """
    A = csr(np.array([[1.0, 2.0],
                     [9.0, 3.0]]))
    ws = FactorWorkspace(2, cap=64)
    assert ws.call(compiled_lib, A) == 0
    assert ws.out['perm'][0] == 1
    assert ws.out['u_diag'][0] == 9.0
    # 1/9 is the multiplier for row 0, and it is the only entry of L
    assert ws.out['l_ptr'][0] == 0 and ws.out['l_ptr'][2] == 1
    assert ws.out['l_val'][0] == pytest.approx(1.0 / 9.0, rel=1e-12)


def test_lu_factorize_reports_minus_two_on_a_structural_zero_pivot(compiled_lib):
    """A column with no stored entry has no candidate at all.

    `_check_A` rejects a matrix with an empty *row*, so this reaches the kernel
    only for a matrix with an empty *column* -- square, every row populated.
    """
    A = csr(np.array([[1.0, 0.0, 1.0],
                     [0.0, 0.0, 2.0],
                     [3.0, 0.0, 3.0]]))
    ws = FactorWorkspace(3, cap=64)
    assert ws.call(compiled_lib, A) == -2


def test_lu_factorize_reports_minus_two_on_a_numerically_zero_pivot(compiled_lib):
    """A stored but all-zero column is the same failure by a different route:
    every row in it is populated, so `_check_A` lets it through."""
    A = csr(np.array([[1.0, 0.0, 1.0],
                     [1.0, 0.0, 2.0],
                     [2.0, 0.0, 3.0]]))
    ws = FactorWorkspace(3, cap=64)
    assert ws.call(compiled_lib, A) == -2


def test_lu_factorize_reports_minus_three_when_the_workspace_is_too_small(compiled_lib):
    """`cap` has to hold every input entry plus every fill entry, and the kernel
    cannot know that in advance, so it answers -3 and the caller retries.

    A matrix with fill is used deliberately: a workspace just big enough for the
    input entries alone must overflow, which is exactly the case the retry in
    `_analyze` exists for.
    """
    n = 30
    dense = np.eye(n) * 0.01
    for i in range(n - 1):
        dense[i, i + 1] = 1.0
        dense[i + 1, i] = 1.0
    A = csr(dense)

    tight = FactorWorkspace(n, cap=A.nnz + 2)
    assert tight.call(compiled_lib, A) == -3

    # the same kernel, same matrix, a workspace that fits: it has to agree with
    # the identity L @ U == A[perm]
    roomy = FactorWorkspace(n, cap=32 * A.nnz)
    assert roomy.call(compiled_lib, A) == 0
    L, U = dense_factors(roomy, n)
    assert np.abs(L @ U - dense[roomy.out['perm']]).max() < 1e-10
    assert roomy.out['nnzf'][0] > A.nnz       # which is why `cap = nnz` overflows


def test_lu_factorize_retry_recovers_and_the_python_side_drives_it(ours):
    """The -3 answer is only useful if `_analyze` acts on it: start from a
    `cap` that cannot fit and check the factors it eventually returns are the
    right ones. `_analyze`'s initial `cap` is `max(2 * nnz, n + 16, 1024)`, so a
    matrix whose fill is several times `nnz` forces a real retry.
    """
    n = 40
    dense = np.eye(n) * 0.01
    for i in range(n - 1):
        dense[i, i + 1] = 1.0
        dense[i + 1, i] = 1.0
    A = csr(dense)
    assert A.nnz < 1024                       # so `cap` starts at the 1024 floor

    f = ours()._analyze(A)
    L, U = reconstruct_dense(f, n)
    assert np.abs(L @ U - dense[f['perm']]).max() < 1e-10


def reconstruct_dense(factors, n):
    L = np.eye(n)
    U = np.zeros((n, n))
    for r in range(n):
        for t in range(factors["l_ptr"][r], factors["l_ptr"][r + 1]):
            L[r, factors["l_idx"][t]] = factors["l_val"][t]
        for t in range(factors["u_ptr"][r], factors["u_ptr"][r + 1]):
            U[r, factors["u_idx"][t]] = factors["u_val"][t]
        U[r, r] = factors["u_diag"][r]
    return L, U


def test_lu_factorize_symbolic_walk_builds_the_same_pattern_without_values(compiled_lib):
    """`symbolic != 0` is phase 22: same structure, no arithmetic.

    On a matrix the two pivot rules agree on (diagonally dominant, so the
    largest-magnitude entry in each column is the diagonal) the structural and
    the numeric walk must produce byte-identical index patterns and the same
    permutation.
    """
    rng = np.random.default_rng(5)
    n = 40
    dense = np.eye(n) * 5.0 + 0.5 * rng.standard_normal((n, n))
    A = csr(dense)

    numeric = FactorWorkspace(n, cap=64 * A.nnz)
    assert numeric.call(compiled_lib, A, symbolic=0) == 0
    structural = FactorWorkspace(n, cap=64 * A.nnz)
    assert structural.call(compiled_lib, A, symbolic=1) == 0

    for key in ('l_ptr', 'u_ptr', 'perm'):
        np.testing.assert_array_equal(numeric.out[key], structural.out[key])
    # `u_diag` is the one array the two walks legitimately disagree on: the
    # numeric one divides by the pivot it found, the structural one does no
    # arithmetic at all and stores 1.0 for every row, which is why the solve
    # never runs off a phase 22 factorisation

    stored = int(numeric.out['l_ptr'][n]) + int(numeric.out['u_ptr'][n]) + n
    assert stored == int(numeric.out['nnzf'][0])
    np.testing.assert_array_equal(numeric.out['l_idx'][:stored],
                                  structural.out['l_idx'][:stored])
    np.testing.assert_array_equal(numeric.out['u_idx'][:stored],
                                  structural.out['u_idx'][:stored])
    assert np.array_equal(structural.out['u_diag'], np.ones(n))


def test_lu_factorize_counts_nnzf_as_l_plus_u_plus_the_diagonal(compiled_lib):
    """`iparm(18)` is nnz(L) + nnz(U) with the split-out diagonal counted once."""
    rng = np.random.default_rng(6)
    for n in (12, 37):
        dense = np.eye(n) * 0.2 + rng.standard_normal((n, n))
        A = csr(dense)
        ws = FactorWorkspace(n, cap=64 * A.nnz)
        assert ws.call(compiled_lib, A) == 0
        o = ws.out
        stored = (int(o['l_ptr'][n]) - int(o['l_ptr'][0])) \
            + (int(o['u_ptr'][n]) - int(o['u_ptr'][0])) + n
        assert int(o['nnzf'][0]) == stored
        assert stored >= A.nnz


def test_lu_factorize_u_rows_come_out_sorted_by_column(compiled_lib):
    """Step 3 insertion-sorts each finished U row so the back substitution walks
    ascending columns. That invariant is load-bearing for `mpp_lu_solve`, and a
    sort that stopped working would still give a correct triangular solve on a
    row of length 0 or 1, so it is checked on rows that are long enough to have
    been reordered.
    """
    rng = np.random.default_rng(7)
    n = 50
    dense = np.eye(n) * 0.01 + rng.standard_normal((n, n))
    A = csr(dense)
    ws = FactorWorkspace(n, cap=64 * A.nnz)
    assert ws.call(compiled_lib, A) == 0

    lengths = []
    o = ws.out
    for r in range(n):
        cols = o['u_idx'][o['u_ptr'][r]:o['u_ptr'][r + 1]]
        assert list(cols) == sorted(cols)
        assert (cols > r).all()             # strictly right of the diagonal
        lengths.append(len(cols))
    assert max(lengths) > 1                 # the sort had something to do


def test_lu_factorize_handles_the_one_by_one_matrix(compiled_lib):
    """The smallest input, which is the tightest fit for the `n + 16` sizing."""
    ws = FactorWorkspace(1, cap=64)
    assert ws.call(compiled_lib, csr(np.array([[2.0]]))) == 0
    assert ws.out['perm'].tolist() == [0]
    assert ws.out['u_diag'][0] == 2.0
    assert int(ws.out['l_ptr'][1]) == 0 and int(ws.out['u_ptr'][1]) == 0


# --------------------------------------------------------------------------
# mpp_lu_solve
# --------------------------------------------------------------------------

def build_dense_factors(n, seed=8):
    """A dense L (unit lower) and U (upper), as `mpp_lu_factorize` would emit
    them: L by row with no diagonal, U by row with the diagonal split out.

    The off-diagonals are kept small relative to the diagonal. A unit-lower `L`
    and an upper `U` whose off-diagonals are full-strength random numbers
    multiply out to a condition number of order `1e5`, and then the test is
    measuring the backward error of a triangular solve rather than whether the
    kernel got the indices right. `mpp_lu_factorize`'s own factors are far
    better conditioned than that, so this is not a property of the kernel.
    """
    rng = np.random.default_rng(seed)
    L = np.eye(n) + 0.1 * np.tril(rng.standard_normal((n, n)), -1)
    U = np.eye(n) * 2.0 + 0.1 * np.triu(rng.standard_normal((n, n)), 1)
    return L, U


def pack_factors(L, U):
    """Pack dense L, U into the CSR arrays the solve kernel reads.

    Both are allocated at the full `n * n` an index could reach, not at the
    packed length: the solve kernel takes no lengths, so a short array would be
    read out of bounds rather than rejected.
    """
    n = L.shape[0]
    l_ptr = np.zeros(n + 1, dtype=np.int32)
    l_idx = np.zeros(n * n, dtype=np.int32)
    l_val = np.zeros(n * n, dtype=np.float64)
    u_ptr = np.zeros(n + 1, dtype=np.int32)
    u_idx = np.zeros(n * n, dtype=np.int32)
    u_val = np.zeros(n * n, dtype=np.float64)
    for r in range(n):
        # L is unit lower triangular with the diagonal implicit, so only the
        # strictly lower part is stored -- putting a 1.0 at (r, r) in `l_idx`
        # would make the forward substitution read `x[r]` before it is written.
        # `lower` is built in ascending column order because the forward
        # substitution reads a row in that order.
        lower = [c for c in range(r) if L[r, c] != 0.0]
        l_ptr[r + 1] = l_ptr[r] + len(lower)
        l_idx[l_ptr[r]:l_ptr[r + 1]] = lower
        l_val[l_ptr[r]:l_ptr[r + 1]] = [L[r, c] for c in lower]

        off_diagonal = [c for c in range(n) if c != r and U[r, c] != 0.0]
        u_ptr[r + 1] = u_ptr[r] + len(off_diagonal)
        u_idx[u_ptr[r]:u_ptr[r + 1]] = off_diagonal
        u_val[u_ptr[r]:u_ptr[r + 1]] = [U[r, c] for c in off_diagonal]

    return {
        'l_ptr': l_ptr,
        'l_idx': l_idx,
        'l_val': l_val,
        'u_ptr': u_ptr,
        'u_idx': u_idx,
        'u_val': u_val,
        'u_diag': np.diag(U).astype(np.float64).copy(),
    }


def test_lu_factorize_factors_solve_the_system_they_came_from(compiled_lib):
    """The contract between the two kernels, checked end to end.

    `mpp_lu_factorize` writes `P A = L U` in PIVOT order and `mpp_permute_rhs`
    applies `perm` to the right-hand side, so feeding its own factors straight
    into `mpp_lu_solve` has to answer `A x = b`.

    This is the one place the permutation is actually exercised. Everywhere else
    the port permutes with one kernel and solves with the other, so a mismatch
    in how the second one indexes `u_diag` or `l_ptr` cancels out and the port
    still gets the right answer. Here the reference is computed without either
    kernel: `L @ U @ x == A[perm] @ x == A @ b` must hold for the `x` the solve
    kernel produced, which cannot survive a permutation mismatch.
    """
    rng = np.random.default_rng(16)
    n = 25
    dense = np.eye(n) * 0.05 + rng.standard_normal((n, n))   # pivots hard
    A = csr(dense)
    b = rng.standard_normal(n)

    ws = FactorWorkspace(n, cap=64 * A.nnz)
    assert ws.call(compiled_lib, A) == 0
    o = ws.out

    perm = o['perm']
    pb = np.empty(n, dtype=np.float64)
    compiled_lib.mpp_permute_rhs(n, 1, addr(perm), addr(b), addr(pb))
    assert np.array_equal(pb, b[perm])          # the right-hand side is P b

    x = np.empty(n, dtype=np.float64)
    lib = ctypes.CDLL(compiled_lib._name)
    lib.mpp_lu_solve.argtypes = _lib._SIGNATURES["mpp_lu_solve"][0]
    lib.mpp_lu_solve(
        n, 1,
        addr(o['l_ptr']), addr(o['l_idx']), addr(o['l_val']),
        addr(o['u_ptr']), addr(o['u_idx']), addr(o['u_val']), addr(o['u_diag']),
        addr(pb), addr(x), 1)

    # `L @ U == P A` and `pb == P b`, so the solve answers `(P A) x == (P b)`,
    # which is the same system as `A x == b`. Both sides are checked against
    # identities that hold without any permutation bookkeeping, so a mismatch
    # cannot cancel.
    L, U = dense_factors(ws, n)
    assert np.abs(L @ U @ x - b[perm]).max() < 1e-9
    assert np.abs(dense @ x - b).max() < 1e-9
    assert np.abs(x - np.linalg.solve(dense, b)).max() < 1e-9


@pytest.mark.parametrize("nrhs", [1, 3, 6])
def test_lu_solve_on_hand_built_factors(compiled_lib, nrhs):
    """`L x = b`, `U x = y`, on factors this test packs itself.

    Nothing here comes from `mpp_lu_factorize`, so the solve kernel is checked
    against `np.linalg.solve(L @ U, b)` rather than against the thing that
    produced its input.

    Every factor array is allocated at full width, not packed tight: `mpp_lu_solve`
    indexes them through raw pointers and takes no lengths, so an array shorter
    than the last pointer would read out of bounds rather than fail, and one
    packed tight would only stay in bounds by accident.
    """
    n = 31
    L, U = build_dense_factors(n)
    f = pack_factors(L, U)
    rng = np.random.default_rng(9)
    b = rng.standard_normal(n * nrhs)

    # a private handle, so these factor pointers are typed from scratch rather
    # than through whatever `_lib` last left in the ctypes cache
    lib = ctypes.CDLL(compiled_lib._name)
    lib.mpp_lu_solve.argtypes = _lib._SIGNATURES["mpp_lu_solve"][0]

    x = np.empty(n * nrhs, dtype=np.float64)
    lib.mpp_lu_solve(
        n, nrhs,
        addr(f['l_ptr']), addr(f['l_idx']), addr(f['l_val']),
        addr(f['u_ptr']), addr(f['u_idx']), addr(f['u_val']), addr(f['u_diag']),
        addr(b), addr(x), 1)

    reference = np.linalg.solve(L @ U, b.reshape(nrhs, n).T).reshape(-1, order='F')
    assert np.abs(x - reference).max() < 1e-10


def test_lu_solve_leaves_the_right_hand_side_untouched(compiled_lib):
    """The forward substitution accumulates in `x` and reads `b` only, so a
    caller may keep `b`. `b` comes back with a copy of its original values."""
    n = 20
    L, U = build_dense_factors(n, seed=10)
    f = pack_factors(L, U)
    b = np.random.default_rng(11).standard_normal(n)
    original = b.copy()

    x = np.zeros(n, dtype=np.float64)
    compiled_lib.mpp_lu_solve(
        n, 1,
        addr(f['l_ptr']), addr(f['l_idx']), addr(f['l_val']),
        addr(f['u_ptr']), addr(f['u_idx']), addr(f['u_val']), addr(f['u_diag']),
        addr(b), addr(x), 1)

    np.testing.assert_array_equal(b, original)
    assert np.abs(L @ U @ x - original).max() < 1e-10


def test_lu_solve_reads_the_right_hand_side_column_major(compiled_lib):
    """Column `r` of an `(n, nrhs)` right-hand side starts at offset `r * n`.

    With three right-hand sides whose values differ, a kernel that assumed a
    row-major buffer would answer each column with another column's solution.
    """
    n = 12
    L, U = build_dense_factors(n, seed=12)
    f = pack_factors(L, U)
    B = np.arange(1.0, n * 3 + 1).reshape(n, 3)
    flat = np.asfortranarray(B).reshape(-1, order='F')

    x = np.zeros(n * 3, dtype=np.float64)
    compiled_lib.mpp_lu_solve(
        n, 3,
        addr(f['l_ptr']), addr(f['l_idx']), addr(f['l_val']),
        addr(f['u_ptr']), addr(f['u_idx']), addr(f['u_val']), addr(f['u_diag']),
        addr(flat), addr(x), 1)

    got = x.reshape(3, n).T
    assert np.abs(got - np.linalg.solve(L @ U, B)).max() < 1e-10


def test_lu_solve_overwrites_every_element_of_its_destination(compiled_lib):
    """`x` is allocated with `np.empty` on the Python side, so the kernel has to
    write all `n * nrhs` slots even when the right-hand side is all zeros.

    A solution of zeros is exactly the case a kernel that skipped writing would
    get wrong here, and it is the case a caller gets from a factorised solve
    with `b = 0`.
    """
    n = 15
    L, U = build_dense_factors(n, seed=13)
    f = pack_factors(L, U)
    b = np.zeros(n * 2)

    x = np.full(n * 2, -7.0, dtype=np.float64)   # poison, must be overwritten
    compiled_lib.mpp_lu_solve(
        n, 2,
        addr(f['l_ptr']), addr(f['l_idx']), addr(f['l_val']),
        addr(f['u_ptr']), addr(f['u_idx']), addr(f['u_val']), addr(f['u_diag']),
        addr(b), addr(x), 1)

    assert x.tolist() == [0.0] * (n * 2)


def test_lu_solve_threads_agree_with_the_serial_path(compiled_lib):
    """Above four right-hand sides `mpp_lu_solve` switches to
    `max.algorithm.parallelize`; the answer must not depend on the worker
    count. One worker forces the serial path, several force the threaded one.
    """
    n = 40
    L, U = build_dense_factors(n, seed=14)
    f = pack_factors(L, U)
    b = np.random.default_rng(15).standard_normal(n * 8)

    answers = []
    for workers in (1, 4, 8):
        x = np.zeros(n * 8, dtype=np.float64)
        compiled_lib.mpp_lu_solve(
            n, 8,
            addr(f['l_ptr']), addr(f['l_idx']), addr(f['l_val']),
            addr(f['u_ptr']), addr(f['u_idx']), addr(f['u_val']), addr(f['u_diag']),
            addr(b), addr(x), workers)
        answers.append(x)

    reference = np.linalg.solve(L @ U, b.reshape(8, n).T).reshape(-1, order='F')
    for x in answers:
        assert np.abs(x - reference).max() < 1e-10
    assert np.array_equal(answers[0], answers[1])
    assert np.array_equal(answers[1], answers[2])


def _declared_parameters():
    """`(kernel name, parameter list)` parsed out of `src/ported.mojo`.

    The ctypes `argtypes` table in `_lib.py` is hand-written, so nothing else
    checks it against the kernels. A table that is one entry short or has the
    wrong width does not fail loudly at import time -- ctypes fills a missing
    trailing argument and a wrong width truncates an address -- so it is
    compared against the source here.
    """
    import os

    import mojo_pypardiso._lib as lib_module

    src = os.path.join(os.path.dirname(lib_module.__file__),
                       "..", "..", "src", "ported.mojo")
    with open(src) as fh:
        text = fh.read()

    out = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r'\s*@export\("([A-Za-z_0-9]+)"\)', line)
        if not match:
            continue
        name = match.group(1)
        signature = []
        depth = 0
        started = False
        for body in lines[i + 1:]:
            signature.append(body)
            depth += body.count("(") - body.count(")")
            if "(" in body:
                started = True
            if started and depth <= 0:
                break
        text_of = " ".join(part.strip() for part in signature)
        inner = text_of[text_of.index("(") + 1:text_of.rindex(")")]
        params = [p.strip().split(":")[0].strip()
                  for p in _split_top_level(inner)]
        # `rindex(")")` lands on the parameter list's own close only because the
        # `abi("C")` effect is nested; drop it if it was read as a parameter
        out[name] = [p for p in params if p and not p.startswith(")")]
    return out


def _split_top_level(text):
    """Split a parameter list on commas that are not nested."""
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def test_ctypes_argtypes_match_the_declared_kernel_signatures():
    """Every parameter of every exported kernel is an `Int`, and `_lib`'s table
    has one `c_void_p` or `c_long` per parameter, in the same order.

    The widths are not all interchangeable. A buffer parameter is an address,
    so it has to reach the kernel as a `c_void_p`; ctypes would otherwise
    truncate a numpy address that does not fit in 32 bits, which it does not
    report. A count parameter is a number, so it must not be a `c_void_p`
    either -- `mpp_csr_transpose`'s `cursor` was declared that way and became
    the integer 1. Both directions are checked here.
    """
    from mojo_pypardiso import _lib

    declared = _declared_parameters()
    assert set(declared) == set(_lib._SIGNATURES)

    for name, params in declared.items():
        argtypes = _lib._SIGNATURES[name][0]
        assert len(argtypes) == len(params), (name, len(argtypes), len(params))
        for param, argtype in zip(params, argtypes):
            is_count = param in COUNTS[name]
            if is_count:
                assert argtype is _lib.I64, (name, param, argtype)
            else:
                assert argtype is _lib.A, (name, param, argtype)


# The `Int` parameters of each kernel that are counts rather than addresses.
COUNTS = {
    "mpp_indptr_diffs": {"n"},
    "mpp_csr_transpose": {"n", "nnz"},
    "mpp_permute_rhs": {"n", "nrhs"},
    "mpp_lu_factorize": {"n", "cap", "symbolic"},
    "mpp_lu_solve": {"n", "nrhs", "nworkers"},
}


def test_every_export_has_an_argtypes_entry(compiled_lib):
    """`@export`ing a kernel without adding it to `_SIGNATURES` would leave the
    call untyped, so ctypes would pass a Python int as a pointer and fail. This
    checks the table against the symbols the built library actually exports."""
    from mojo_pypardiso import _lib

    assert set(_lib._SIGNATURES) == set(_declared_parameters())
    for name in _lib._SIGNATURES:
        assert isinstance(getattr(compiled_lib, name), ctypes._CFuncPtr)
