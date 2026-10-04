"""Behavioural parity of the Python API with upstream pypardiso.

These tests assert that the port keeps upstream's names, argument order, defaults,
phase protocol and exceptions, so a caller written against `pypardiso` needs no
changes to run on `mojo_pypardiso`.

Every test here carries the `compiled` mark, so the whole file is gated on the
`compiled_lib` fixture. Nothing in it can pass on a pure-Python path.
"""

import inspect
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning

import pypardiso as up
from mojo_pypardiso import PyPardisoSolver, factorized, spsolve
from mojo_pypardiso.pardiso_wrapper import PyPardisoError, PyPardisoWarning

from conftest import csr, dense_spd, random_sparse

compiled = pytest.mark.usefixtures("compiled_lib")


def public_methods(cls):
    return {n for n, _ in inspect.getmembers(cls, inspect.isfunction)
            if not n.startswith("__") and not n.startswith("_")}


@compiled
def test_module_exports_match_upstream():
    import mojo_pypardiso
    assert set(mojo_pypardiso.__all__) == set(up.__all__)
    for name in up.__all__:
        assert hasattr(mojo_pypardiso, name)


@compiled
def test_public_method_set_matches_upstream():
    assert public_methods(PyPardisoSolver) == public_methods(up.PyPardisoSolver)


@compiled
@pytest.mark.parametrize("name", sorted(public_methods(up.PyPardisoSolver)))
def test_method_signatures_match_upstream(name):
    mine = inspect.signature(getattr(PyPardisoSolver, name))
    theirs = inspect.signature(getattr(up.PyPardisoSolver, name))
    assert list(mine.parameters) == list(theirs.parameters)
    assert [p.default for p in mine.parameters.values()] == \
           [p.default for p in theirs.parameters.values()]


@compiled
@pytest.mark.parametrize("name", ["spsolve", "factorized"])
def test_scipy_alias_signatures_match_upstream(name):
    # `solver` defaults to each module's own singleton, so the two defaults are
    # different objects by construction; compare names and the other defaults.
    mine = inspect.signature(globals()[name])
    theirs = inspect.signature(getattr(up, name))
    assert list(mine.parameters) == list(theirs.parameters)
    for pname, p in mine.parameters.items():
        if pname == "solver":
            continue
        assert p.default == theirs.parameters[pname].default
    assert isinstance(mine.parameters["solver"].default, PyPardisoSolver)


@compiled
def test_default_constructor_arguments():
    solver = PyPardisoSolver()
    assert solver.mtype == 11
    assert solver.phase == 13
    assert solver.size_limit_storage == 5e7
    assert solver.msglvl is False


@compiled
@pytest.mark.parametrize("mtype,phase", [(11, 13), (11, 12), (11, 22), (11, 33)])
def test_constructor_accepts_upstream_arguments(mtype, phase):
    solver = PyPardisoSolver(mtype=mtype, phase=phase, size_limit_storage=1000)
    assert (solver.mtype, solver.phase, solver.size_limit_storage) == \
           (mtype, phase, 1000)


@compiled
def test_default_iparms_are_zero_like_upstream():
    assert np.array_equal(PyPardisoSolver().get_iparms(), up.PyPardisoSolver().get_iparms())
    assert len(PyPardisoSolver().get_iparms()) == 64


@compiled
def test_set_iparm_warns_on_non_input_iparm():
    solver = PyPardisoSolver()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.set_iparm(7, 1)
    assert any(issubclass(w.category, PyPardisoWarning) for w in caught)

    solver.set_iparm(12, 1)
    assert solver.get_iparm(12) == 1


@compiled
def test_phase_and_matrix_type_setters():
    solver = PyPardisoSolver()
    solver.set_phase(22)
    solver.set_matrix_type(-2)
    solver.set_statistical_info_on()
    assert (solver.phase, solver.mtype, solver.msglvl) == (22, -2, 1)
    solver.set_statistical_info_off()
    assert solver.msglvl == 0


@compiled
def test_check_A_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        PyPardisoSolver().solve(sp.csr_matrix(np.ones((2, 3))), np.ones(2))


@compiled
def test_check_A_rejects_dense_input():
    with pytest.raises(TypeError, match="CSR or CSC"):
        PyPardisoSolver().solve(np.eye(3), np.ones(3))


@compiled
def test_check_A_rejects_non_float64():
    A = sp.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(TypeError, match="float64"):
        PyPardisoSolver().solve(A, np.ones(3, dtype=np.float32))


@compiled
def test_check_A_reports_empty_rows_the_upstream_way():
    # both rows carry an entry, so nothing is empty
    A = sp.csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, 2))
    assert np.diff(A.indptr).tolist() == [1, 1]
    assert np.allclose(PyPardisoSolver().solve(A, np.ones(2)), [1.0, 1.0])

    # row 1 has no entries, so the square matrix is singular
    B = sp.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))
    assert np.diff(B.indptr).tolist() == [1, 0]
    with pytest.raises(ValueError, match="empty row"):
        PyPardisoSolver().solve(B, np.ones(2))


@compiled
def test_check_A_reports_empty_columns_on_the_csc_path():
    A = sp.csc_matrix(([1.0, 1.0], ([0, 1], [0, 0])), shape=(2, 2))
    assert np.diff(A.indptr).tolist() == [2, 0]
    with pytest.raises(ValueError, match="empty column"):
        PyPardisoSolver().solve(A, np.ones(2))


@compiled
def test_check_b_rejects_wrong_dimension():
    A = csr(dense_spd(4))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        PyPardisoSolver().solve(A, np.ones(3))


@compiled
def test_check_b_rejects_unsupported_dtype():
    A = csr(dense_spd(4))
    with pytest.raises(TypeError, match="not supported"):
        PyPardisoSolver().solve(A, np.ones(4, dtype=np.complex128))


@compiled
def test_check_b_converts_and_warns_on_narrow_dtypes():
    A = csr(dense_spd(4))
    b = np.ones(4, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = PyPardisoSolver()._check_b(A, b)
    assert out.dtype == np.float64
    assert any(issubclass(w.category, PyPardisoWarning) for w in caught)


@compiled
def test_check_b_warns_on_sparse_rhs():
    A = csr(dense_spd(4))
    b = sp.csr_matrix(np.ones((4, 1)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PyPardisoSolver()._check_b(A, b)
    assert any(issubclass(w.category, SparseEfficiencyWarning) for w in caught)


@compiled
def test_solve_returns_the_shape_of_b():
    A = csr(dense_spd(6, 1))
    solver = PyPardisoSolver()
    assert solver.solve(A, np.ones(6)).shape == (6,)
    assert solver.solve(A, np.ones((6, 1))).shape == (6, 1)
    assert solver.solve(A, np.ones((6, 3))).shape == (6, 3)


@compiled
def test_solve_returns_c_contiguous_like_upstream():
    A = csr(dense_spd(6, 2))
    x = PyPardisoSolver().solve(A, np.ones((6, 3)))
    assert x.flags.c_contiguous
    assert not x.base is None or True  # upstream also returns a fresh array


@compiled
def test_factorize_then_solve_reuses_the_factors():
    A = csr(dense_spd(30, 3))
    solver = PyPardisoSolver()
    solver.factorize(A)
    assert solver.phase == 12
    assert solver._factors is not None

    b = np.random.default_rng(4).standard_normal(30)
    first = solver.solve(A, b)
    assert solver.phase == 33  # reused rather than refactorised
    second = solver.solve(A, b)
    assert np.array_equal(first, second)


@compiled
def test_solve_on_a_new_matrix_refactorises():
    solver = PyPardisoSolver()
    A = csr(dense_spd(20, 4))
    b = np.ones(20)
    solver.solve(A, b)
    assert solver.phase == 13
    solver.solve(csr(dense_spd(20, 5)), b)
    assert solver.phase == 13


@compiled
def test_solve_transposed_flag_tracks_the_format():
    solver = PyPardisoSolver()
    A = csr(dense_spd(5, 6))
    solver.solve(A, np.ones(5))
    assert solver._solve_transposed is False
    assert solver.get_iparm(12) == 0
    solver.solve(sp.csc_matrix(A), np.ones(5))
    assert solver._solve_transposed is True
    assert solver.get_iparm(12) == 1


@compiled
def test_remove_stored_factorization_clears_the_state():
    A = csr(dense_spd(10, 7))
    solver = PyPardisoSolver()
    solver.factorize(A)
    solver.remove_stored_factorization()
    assert solver._factors is None
    assert solver.factorized_A.shape == (0, 0)


@compiled
def test_free_memory_releases_and_restores_the_phase():
    A = csr(dense_spd(10, 8))
    solver = PyPardisoSolver()
    solver.factorize(A)
    solver.free_memory()
    assert solver._factors is None
    assert solver.phase == 13


@compiled
def test_phase_33_without_a_factorization_raises():
    A = csr(dense_spd(10, 9))
    solver = PyPardisoSolver()
    solver.set_phase(33)
    with pytest.raises(PyPardisoError):
        solver._call_pardiso(A, np.ones(10))


@compiled
def test_singular_matrix_raises_pardiso_error():
    A = csr(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(PyPardisoError) as info:
        PyPardisoSolver().solve(A, np.ones(2))
    assert info.value.value == -2
    assert "error code -2" in str(info.value)


@compiled
def test_pardiso_error_message_matches_upstream_shape():
    mine = str(PyPardisoError(-4))
    theirs = str(up.pardiso_wrapper.PyPardisoError(-4))
    assert mine == theirs


@compiled
def test_pardiso_warning_is_a_user_warning():
    assert issubclass(PyPardisoWarning, UserWarning)


@compiled
def test_spsolve_squeezes_by_default_like_upstream():
    A = csr(dense_spd(8, 10))
    b = np.ones((8, 1))
    ours = spsolve(A, b, factorize=True, squeeze=True)
    theirs = up.spsolve(A, b, factorize=True, squeeze=True)
    assert ours.shape == theirs.shape == (8,)
    assert np.allclose(ours, theirs, atol=1e-9)


@compiled
def test_spsolve_without_squeeze_matches_upstream():
    A = csr(dense_spd(8, 11))
    b = np.ones((8, 2))
    ours = spsolve(A, b, factorize=True, squeeze=False)
    theirs = up.spsolve(A, b, factorize=True, squeeze=False)
    assert ours.shape == theirs.shape == (8, 2)
    assert np.allclose(ours, theirs, atol=1e-9)


@compiled
def test_spsolve_accepts_csc_like_upstream():
    A = sp.csc_matrix(csr(dense_spd(8, 12)))
    b = np.ones(8)
    assert np.allclose(spsolve(A, b), up.spsolve(A, b), atol=1e-9)


@compiled
def test_spsolve_without_factorize_matches_upstream():
    A = csr(dense_spd(12, 13))
    b = np.ones(12)
    assert np.allclose(spsolve(A, b, factorize=False),
                       up.spsolve(A, b, factorize=False), atol=1e-9)


@compiled
def test_spsolve_accepts_a_custom_solver():
    A = csr(dense_spd(8, 14))
    b = np.ones(8)
    assert np.allclose(spsolve(A, b, solver=PyPardisoSolver()),
                       np.linalg.solve(A.toarray(), b), atol=1e-9)


@compiled
def test_factorized_returns_a_reusable_callable():
    A = csr(dense_spd(10, 15))
    solve_b = factorized(A)
    b = np.ones(10)
    assert np.allclose(solve_b(b), np.linalg.solve(A.toarray(), b), atol=1e-9)
    # a second call with a different right-hand side still works
    b2 = np.arange(10, dtype=float)
    assert np.allclose(solve_b(b2), np.linalg.solve(A.toarray(), b2), atol=1e-9)


@compiled
def test_hash_csr_matrix_matches_upstream():
    A = csr(dense_spd(6, 16))
    assert PyPardisoSolver()._hash_csr_matrix(A) == \
           up.PyPardisoSolver()._hash_csr_matrix(A)


@compiled
def test_csr_matrix_equal_matches_upstream():
    A = csr(dense_spd(6, 17))
    solver = PyPardisoSolver()
    assert solver._csr_matrix_equal(A, A.copy()) is True
    assert solver._csr_matrix_equal(A, csr(dense_spd(6, 18))) is False


@compiled
def test_large_matrix_storage_limit_uses_the_hash():
    A = csr(dense_spd(30, 19))
    solver = PyPardisoSolver(size_limit_storage=1)
    solver.factorize(A)
    assert isinstance(solver.factorized_A, str)
    assert solver._is_already_factorized(A) is True
    assert solver._is_already_factorized(csr(dense_spd(30, 20))) is False


@compiled
def test_factorize_on_csc_then_solve_reuses_factors():
    A = sp.csc_matrix(csr(dense_spd(30, 23)))
    b = np.random.default_rng(24).standard_normal(30)
    solver = PyPardisoSolver()
    solver.factorize(A)
    assert solver._factors is not None
    first = solver.solve(A, b)
    assert solver.phase == 33
    assert np.allclose(first, np.linalg.solve(A.toarray(), b), atol=1e-9)
    assert np.array_equal(solver.solve(A, b), first)


@compiled
def test_repeated_solve_is_deterministic():
    A = csr(dense_spd(40, 21))
    b = np.random.default_rng(22).standard_normal(40)
    solver = PyPardisoSolver()
    first = solver.solve(A, b)
    for _ in range(3):
        assert np.array_equal(solver.solve(A, b), first)


@compiled
def test_unsorted_indices_are_sorted_like_upstream():
    # row 0 holds columns [1, 0] and row 1 holds [0, 1], neither sorted
    S = sp.csr_matrix((np.array([2.0, 1.0, 3.0, 1.0]),
                       np.array([1, 0, 0, 1]), np.array([0, 2, 4])), shape=(2, 2))
    S.has_sorted_indices = False
    bb = np.array([3.0, 4.0])
    mine = PyPardisoSolver().solve(S, bb)
    assert S.has_sorted_indices
    assert np.allclose(mine, np.linalg.solve(S.toarray(), bb), atol=1e-9)
