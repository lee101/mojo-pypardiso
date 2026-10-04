"""Python interface to the Mojo sparse direct solver for Ax=b.

Ported from ``pypardiso/pardiso_wrapper.py`` in upstream pypardiso 0.4.7. The
public surface is unchanged: same class, same method names, same argument order
and defaults, same iparm/phase protocol, same exceptions. The single
substitution is the solver itself. Upstream forwards everything to
``mkl_pardiso``; this port hands the factorisation and the triangular solves to
``mpp_lu_factorize`` / ``mpp_lu_solve`` in the compiled Mojo library, which
produce the same PA=LU with partial pivoting.

Because MKL keeps the factorisation in its own address space, upstream can
answer phase 33 from state it holds privately. This port keeps the factors on
the Python side, so that state is visible as ``_factors`` and ``free_memory``
releases it. Every other method keeps its upstream behaviour.

--- Basic usage ---
matrix type: real (float64) and nonsymmetric
methods: solve, factorize

- use the "solve(A,b)" method to solve Ax=b for x, where A is a sparse CSR (or CSC) matrix and b is a numpy array
- use the "factorize(A)" method first, if you intend to solve the system more than once for different right-hand
  sides, the factorization will be reused automatically afterwards

--- Advanced usage ---
methods: get_iparm, get_iparms, set_iparm, set_matrix_type, set_phase

- additional options can be accessed by setting the iparms
- other matrix types can be chosen with the "set_matrix_type" method. complex matrix types are currently not
  supported; only mtype=11 (real and nonsymmetric) is implemented
- the solving phases can be set with the "set_phase" method

--- Statistical info ---
methods: set_statistical_info_on, set_statistical_info_off

--- Memory usage ---
methods: remove_stored_factorization, free_memory
"""

import hashlib
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning

from . import _lib

# A page-touched scratch buffer, used to keep a matrix's index and data arrays
# resident so the fast path in `_is_already_factorized` can trust that their
# contents have not changed. See that method for why it exists; `_lib.WARM`
# explains the mechanism.
_WARM_MIN_NNZ = 4096


def _touch(A):
    """Bring `A`'s index and data arrays into cache, to make a later content
    comparison trustworthy. See `_is_already_factorized`."""
    for arr in (A.data, A.indices, A.indptr):
        _lib.WARM[:max(1, len(arr))].sum()


class PyPardisoSolver:
    """
    Python interface to a sparse direct solver for large sparse linear systems of equations Ax=b.

    Pardiso documentation: https://software.intel.com/en-us/node/470282
    """

    def __init__(self, mtype=11, phase=13, size_limit_storage=5e7):

        self.libmkl = None
        self._lib = _lib.lib()

        # determine 32bit or 64bit architecture
        if np.dtype(np.intp).itemsize == 8:
            self._pt_type = (np.int64, np.int64)
        else:
            self._pt_type = (np.int32, np.int32)

        self.pt = np.zeros(64, dtype=self._pt_type[1])
        self.iparm = np.zeros(64, dtype=np.int32)
        self.perm = np.zeros(0, dtype=np.int32)

        self.mtype = mtype
        self.phase = phase
        self.msglvl = False

        self.factorized_A = sp.csr_matrix((0, 0))
        self.size_limit_storage = size_limit_storage
        self._solve_transposed = False

        # upstream holds the factorisation inside mkl; here the LU factors live
        # on the python side, so phase 33 can read them back
        self._factors = None
        # the exact matrix object `factorize` last took, so a repeated
        # `solve(A, b)` with the same object can skip the three array
        # comparisons `_is_already_factorized` would otherwise make
        self._factorized_obj = None
        # scratch for `_check_A`'s indptr-difference kernel, grown on demand
        self._diff_scratch = None
        # the arrays the stored factorisation was built from, so the identity
        # fast path in `_is_already_factorized` can tell an untouched matrix
        # from one whose contents were rewritten in place
        self._factorized_buffers = None

    def factorize(self, A):
        """
        Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
        solve method. This will drastically increase the speed of solve, if solve is called more than once for the
        same matrix A

        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
        """

        self._check_A(A)

        self._store_for_comparison(A)

        self.set_phase(12)
        b = np.zeros((A.shape[0], 1))
        self._call_pardiso(A, b)

    def _store_for_comparison(self, A):
        """Record the matrix these factors come from, and how to recognise it.

        Upstream keeps the matrix itself, a copy, or a hash of it depending on
        its size; this port does the same and additionally records the buffers
        the factorisation was built from, which is what
        `_is_already_factorized`'s fast path is guarded by.
        """
        if A.nnz > self.size_limit_storage:
            self.factorized_A = self._hash_csr_matrix(A)
        else:
            self.factorized_A = A.copy()
        self._factorized_buffers = self._buffer_ids(A)

    @staticmethod
    def _buffer_ids(A):
        """Addresses of the index and data buffers a matrix currently uses.

        Recorded with the factorisation so `_is_already_factorized` can tell a
        fresh factorisation from one it may reuse.
        """
        if getattr(A, "format", None) not in ("csr", "csc"):
            return None
        return (A.data.ctypes.data, A.indices.ctypes.data, A.indptr.ctypes.data)

    def solve(self, A, b):
        """
        solve Ax=b for x

        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]

        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as input b
        """

        self._check_A(A)
        b = self._check_b(A, b)

        if self._is_already_factorized(A):
            self.set_phase(33)
        else:
            self.set_phase(13)

        x = self._call_pardiso(A, b)

        return x

    def _is_already_factorized(self, A):
        # the documented usage is factorize(A) once and then solve(A, b) many
        # times with the same object. recognising that object answers the
        # question without comparing indptr, indices and data element by
        # element, which on a large matrix costs more than the solve itself.
        #
        # Identity alone is not enough to trust. A caller may mutate the matrix
        # in place between solves -- `A[0, 0] = 5.0`, a rescale, an in-place
        # elimination sweep -- and scipy's sparse setters rewrite `data` in the
        # same buffer without replacing it, so the object's identity survives a
        # matrix that is no longer the one that was factorised. Serving that
        # matrix from the stored factors returns the solution of a different
        # system with no indication that anything is wrong, which is strictly
        # worse than the comparison the fast path exists to avoid. So the guard
        # is:
        #
        #   same object AND the matrix is small enough for the content
        #   comparison to have read every element of it
        #
        # Above the threshold, where that guarantee stops holding, the identity
        # is not used at all and every call takes the full comparison, which is
        # what upstream does unconditionally. On a 500x500 dense matrix the
        # warmed comparison below costs about a quarter of a millisecond
        # against a 0.5 ms solve, so there is little left to win at that size
        # anyway; the fast path is for the many-small-solves case.
        if (self._factorized_buffers is not None
                and A is self._factorized_obj
                and A.nnz <= _WARM_MIN_NNZ
                and type(self.factorized_A) != str):
            _touch(A)
            if self._csr_matrix_equal(A, self.factorized_A):
                return True
        if type(self.factorized_A) == str:
            return self._hash_csr_matrix(A) == self.factorized_A
        else:
            return self._csr_matrix_equal(A, self.factorized_A)

    def _csr_matrix_equal(self, a1, a2):
        return all((np.array_equal(a1.indptr, a2.indptr),
                    np.array_equal(a1.indices, a2.indices),
                    np.array_equal(a1.data, a2.data)))

    def _hash_csr_matrix(self, matrix):
        return (hashlib.sha1(matrix.indices).hexdigest() +
                hashlib.sha1(matrix.indptr).hexdigest() +
                hashlib.sha1(matrix.data).hexdigest())

    def _check_A(self, A):
        if A.shape[0] != A.shape[1]:
            raise ValueError('Matrix A needs to be square, but has shape: {}'.format(A.shape))

        if sp.issparse(A) and A.format == "csr":
            self._solve_transposed = False
            self.set_iparm(12, 0)
        elif sp.issparse(A) and A.format == "csc":
            self._solve_transposed = True
            self.set_iparm(12, 1)
        else:
            msg = 'PyPardiso requires matrix A to be in CSR or CSC format, but matrix A is: {}'.format(type(A))
            raise TypeError(msg)

        # scipy allows unsorted csr-indices, which lead to completely wrong pardiso results
        if not A.has_sorted_indices:
            A.sort_indices()

        # scipy allows csr matrices with empty rows. a square matrix with an empty row is singular. calling
        # pardiso with a matrix A that contains empty rows leads to a segfault, same applies for csc with
        # empty columns
        if self._empty_rows(A):
            row_col = 'column' if self._solve_transposed else 'row'
            raise ValueError('Matrix A is singular, because it contains empty {}(s)'.format(row_col))

        if A.dtype != np.float64:
            raise TypeError('PyPardiso currently only supports float64, but matrix A has dtype: {}'.format(A.dtype))

    def _empty_rows(self, A):
        """`np.diff(A.indptr).all()`, evaluated by the Mojo kernel.

        The kernel only needs somewhere to write the differences, and only its
        count of zero rows is used, so the scratch is reused across calls rather
        than allocated per `_check_A`.
        """
        n = A.shape[0]
        indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        diff = self._diff_scratch
        if diff is None or len(diff) < max(n, 1):
            diff = np.empty(max(n, 1), dtype=np.int32)
            self._diff_scratch = diff
        zeros = self._lib.mpp_indptr_diffs(_lib.addr(indptr), _lib.addr(diff), n)
        return zeros != 0

    def _check_b(self, A, b):
        if sp.issparse(b):
            warnings.warn('PyPardiso requires the right-hand side b to be a dense array for maximum efficiency',
                          SparseEfficiencyWarning)
            # `todense` hands back a `np.matrix`, whose `flags.f_contiguous` is
            # False for an (n, 1) result, so the `np.asfortranarray` below would
            # silently turn an (n,) right-hand side into an (n, 1) one and the
            # answer would come back with the wrong shape. `toarray` gives a
            # plain ndarray of the same shape with the same values, which is
            # what the ctypes call downstream actually reads.
            b = b.toarray()

        # pardiso expects fortran (column-major) order for b. a 1D b is one column,
        # so both orders are the same buffer and there is nothing to convert.
        if b.ndim > 1:
            if not b.flags.f_contiguous:
                b = np.asfortranarray(b)
        elif not b.flags.c_contiguous:
            # A 1D b is not reshaped or copied again in `_call_pardiso`, so its
            # buffer is handed to `mpp_permute_rhs` as it stands. The kernel
            # indexes it as `b[i]` with no stride argument, so a strided 1D view
            # would be read with a stride of 1 against a buffer whose stride is
            # not -- upstream materialises such a view with `asfortranarray`,
            # which is a no-op on 1D, and reads it through `ctypes` the same way,
            # so it is materialised here too.
            b = np.ascontiguousarray(b)

        if b.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch: Matrix A {} and array b {}".format(A.shape, b.shape))

        if b.dtype != np.float64:
            if b.dtype in [np.float16, np.float32, np.int16, np.int32, np.int64]:
                warnings.warn("Array b's data type was converted from {} to float64".format(str(b.dtype)),
                              PyPardisoWarning)
                b = b.astype(np.float64)
            else:
                raise TypeError('Dtype {} for array b is not supported'.format(str(b.dtype)))

        return b

    def _call_pardiso(self, A, b):

        # phase 0 / -1: release the factorisation (see free_memory)
        if self.phase == 0 or self.phase == -1:
            self._factors = None
            return np.zeros_like(b)

        # the CSC path is PARDISO's iparm(12) = 1; mpp_csr_transpose applies the
        # same conversion csc_matrix.tocsr() performs, so the kernels only ever
        # see CSR
        Acsr = A
        if self._solve_transposed:
            n, nnz = A.shape[0], A.nnz
            # The kernels take 32-bit indices and 64-bit values, so the source is
            # narrowed on the way in and the destination is allocated at the
            # kernel's index width. scipy keeps int64 index arrays around once a
            # matrix gets large enough, and handing one of those to a kernel that
            # writes Int32 would leave the high half of each slot untouched --
            # i.e. silently wrong indices, not an error.
            src_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
            src_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
            src_val = np.ascontiguousarray(A.data, dtype=np.float64)
            cursor = np.zeros(n, dtype=np.int32)
            indptr = np.zeros(n + 1, dtype=np.int32)
            indices = np.zeros(nnz, dtype=np.int32)
            data = np.zeros(nnz, dtype=np.float64)
            err = self._lib.mpp_csr_transpose(
                _lib.addr(src_ptr), _lib.addr(src_idx), _lib.addr(src_val),
                _lib.addr(indptr), _lib.addr(indices), _lib.addr(data),
                n, nnz, _lib.addr(cursor))
            if err != 0:
                raise PyPardisoError(err)
            Acsr = sp.csr_matrix((data, indices, indptr), shape=(n, n))

        # phase 22: analysis only. The fill pattern does not depend on the
        # values, so this is the same structural walk with the arithmetic off.
        if self.phase == 22:
            self.iparm[17] = self._analyze(Acsr, symbolic=True)['nnzf']
            return np.zeros_like(b)

        # phases 12 and 13 factorise; phase 33 reuses what is already stored
        if self.phase == 33:
            if self._factors is None:
                raise PyPardisoError(-4)
        else:
            from_factorize = self.phase == 12
            self._factors = self._analyze(Acsr, symbolic=False)
            # remember the object whose factors these are, so a repeated solve
            # with it can skip the element-by-element comparison
            self._factorized_obj = A
            if not from_factorize:
                # a phase 13 call from `solve`: `factorized_A` is still empty, so
                # store the comparison copy here. `factorize` stores its own and
                # is the one that owns that record.
                self._store_for_comparison(A)

        n = Acsr.shape[0]
        nrhs = 1 if b.ndim == 1 else b.shape[1]
        # upstream hands PARDISO the Fortran-contiguous b, and PARDISO applies
        # `perm` to it internally; the kernels take that half-applied b.
        # `_check_b` has already made b float64 and Fortran-contiguous, so for
        # the 2D case its buffer is exactly the column-major layout the kernel
        # wants and no copy is needed; a 1D b is a single column already.
        if b.ndim == 1:
            bf = b
        elif b.flags.f_contiguous:
            bf = b.reshape(-1, order='F')
        else:
            bf = np.asfortranarray(b).reshape(-1, order='F')
        perm = self._factors['perm']
        # both kernels below write every element of their output buffer before
        # reading it, so these do not need to be zeroed first
        pb = np.empty_like(bf)
        self._lib.mpp_permute_rhs(n, nrhs, _lib.addr(perm), _lib.addr(bf), _lib.addr(pb))
        xf = np.empty_like(pb)
        self._lib.mpp_lu_solve(n, nrhs, *self._factor_args(), _lib.addr(pb), _lib.addr(xf),
                               _lib.nworkers())

        # the iparm statistics mkl fills in during phase 13/33
        self.iparm[17] = self._factors['nnzf']
        self.iparm[18] = 1

        # change memory-layout back from fortran to c order
        if b.ndim == 1:
            return xf.reshape(b.shape)
        return np.ascontiguousarray(xf.reshape(b.shape, order='F'))

    def _factor_args(self):
        f = self._factors
        return (_lib.addr(f['l_ptr']), _lib.addr(f['l_idx']), _lib.addr(f['l_val']),
                _lib.addr(f['u_ptr']), _lib.addr(f['u_idx']), _lib.addr(f['u_val']),
                _lib.addr(f['u_diag']))

    def _analyze(self, A, symbolic=False):
        """Run mpp_lu_factorize, growing the entry workspace until it fits.

        The workspace holds every input and fill entry the factorisation ever
        creates, and how many that is is not known until it is built, so a
        -3 from the kernel means "give me a bigger array and try again".
        """
        n = A.shape[0]
        nnz = A.nnz
        indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        data = np.ascontiguousarray(A.data, dtype=np.float64)

        cap = max(2 * nnz, n + 16, 1024)
        while True:
            w = {
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
                # one (label, entry-handle) pair per multiplier the factorisation
                # emits; the counting sort at the end reads them back
                'l_lab': np.zeros(cap, dtype=np.int32),
                'l_entry': np.zeros(cap, dtype=np.int32),
                # low half holds the live column, high half the U row being
                # sorted, and the tail holds the label -> position table
                'scratch': np.zeros(n + 2 * cap, dtype=np.int32),
            }
            out = {
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
            err = self._lib.mpp_lu_factorize(
                n, cap, 1 if symbolic else 0,
                _lib.addr(indptr), _lib.addr(indices), _lib.addr(data),
                _lib.addr(out['l_ptr']), _lib.addr(out['l_idx']), _lib.addr(out['l_val']),
                _lib.addr(out['u_ptr']), _lib.addr(out['u_idx']), _lib.addr(out['u_val']),
                _lib.addr(out['u_diag']), _lib.addr(out['perm']), _lib.addr(out['nnzf']),
                _lib.addr(w['val']), _lib.addr(w['row']), _lib.addr(w['col']),
                _lib.addr(w['nextc']), _lib.addr(w['nextr']),
                _lib.addr(w['col_head']), _lib.addr(w['row_head']),
                _lib.addr(w['row_label']), _lib.addr(w['pos']), _lib.addr(w['tag']),
                _lib.addr(w['l_lab']),
                _lib.addr(w['scratch']), _lib.addr(w['l_entry']),
                _lib.addr(out['u_cur']),
            )
            if err == -3:
                cap *= 4
                continue
            if err != 0:
                raise PyPardisoError(err)
            out['nnzf'] = int(out['nnzf'][0])
            return out

    def get_iparms(self):
        """Returns a dictionary of iparms"""
        return dict(enumerate(self.iparm, 1))

    def get_iparm(self, i):
        """Returns the i-th iparm (1-based indexing)"""
        return self.iparm[i-1]

    def set_iparm(self, i, value):
        """set the i-th iparm to 'value' (1-based indexing)"""
        if i not in {1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 18, 19, 21, 24, 25, 27, 28, 31, 34, 35, 36, 37, 56, 60}:
            warnings.warn('{} is no input iparm. See the Pardiso documentation.'.format(value), PyPardisoWarning)
        self.iparm[i-1] = value

    def set_matrix_type(self, mtype):
        """Set the matrix type (see Pardiso documentation)"""
        self.mtype = mtype

    def set_statistical_info_on(self):
        """Display statistical info (appears in notebook server console window if pypardiso is
        used from jupyter notebook, use wurlitzer to redirect info to the notebook)"""
        self.msglvl = 1

    def set_statistical_info_off(self):
        """Turns statistical info off"""
        self.msglvl = 0

    def set_phase(self, phase):
        """Set the phase(s) for the solver. See the Pardiso documentation for details."""
        self.phase = phase

    def remove_stored_factorization(self):
        """removes the stored factorization, this will free the memory in python, but the factorization in pardiso
        is still accessible with a direct call to self._call_pardiso(A,b) with phase=33"""
        self.factorized_A = sp.csr_matrix((0, 0))
        self._factors = None
        self._factorized_obj = None
        self._factorized_buffers = None

    def free_memory(self, everything=False):
        """release mkl's internal memory, either only for the factorization (ie the LU-decomposition) or all of
        mkl's internal memory if everything=True"""
        self.remove_stored_factorization()
        A = sp.csr_matrix((0, 0))
        b = np.zeros(0)
        self.set_phase(-1 if everything else 0)
        self._call_pardiso(A, b)
        self.set_phase(13)


class PyPardisoWarning(UserWarning):
    pass


class PyPardisoError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ('The Pardiso solver failed with error code {}. '
                'See Pardiso documentation for details.'.format(self.value))
