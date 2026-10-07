# mojo-pypardiso

A Mojo port of the [pypardiso](https://github.com/haasad/PyPardisoProject) sparse
direct solver, with the same Python API.

Upstream `pypardiso` is a thin ctypes wrapper: it loads Intel MKL's `mkl_pardiso`
and forwards every call to it. Almost nothing is computed in Python. This port
keeps the Python surface compatible and replaces the closed binary with a Mojo
implementation of the same computation — a sparse LU factorisation of `P A` with
partial pivoting, plus the triangular solves against the stored factors —
compiled to `dist/libmojo-pypardiso.so` and called through ctypes.

## Install

```bash
pixi install
pixi run build
```

`pixi install` fetches the pinned Mojo toolchain and upstream `pypardiso` (used
for parity testing and benchmarking). `pixi run build` compiles
`src/ported.mojo` into `dist/libmojo-pypardiso.so`.

## Usage

```python
import numpy as np
import scipy.sparse as sp
from mojo_pypardiso import spsolve

A = sp.csr_matrix(np.array([[4.0, 1.0, 0.0],
                            [1.0, 3.0, 1.0],
                            [0.0, 1.0, 2.0]]))
b = np.array([1.0, 2.0, 3.0])

x = spsolve(A, b)
print(x)                       # [0.22222222 0.11111111 1.44444444]
print(np.allclose(A @ x, b))   # True
```

The solver class works the same way as upstream:

```python
from mojo_pypardiso import PyPardisoSolver

solver = PyPardisoSolver()
solver.factorize(A)            # phase 12, reused by later solves
x = solver.solve(A, b)         # phase 33, reuses the factors

for rhs in many_rhs:
    x = solver.solve(A, rhs)   # only the triangular solves run
```

Everything upstream exposes is present with the same names, argument order and
defaults: `PyPardisoSolver(mtype=11, phase=13, size_limit_storage=5e7)`,
`factorize`, `solve`, `get_iparms`, `get_iparm`, `set_iparm`, `set_matrix_type`,
`set_phase`, `set_statistical_info_on/off`, `remove_stored_factorization`,
`free_memory`, `PyPardisoError`, `PyPardisoWarning`, `spsolve`, `factorized`,
and the `ps` singleton. The test suite asserts this equivalence directly, by
reflecting over both modules and comparing signatures.

## What is covered, and what is not

Everything in `pypardiso/pardiso_wrapper.py` and `pypardiso/scipy_aliases.py` is
ported. There are four places where the port is not identical to upstream: three
forced by MKL PARDISO being a proprietary binary, and one where upstream is
silently wrong and this port is not.

**1. `_call_pardiso`, phases 12/13 — the factorisation algorithm.**
MKL runs a multifrontal supernodal solver with a fill-reducing ordering and
multithreaded supernode updates. `mpp_lu_factorize` is a right-looking sparse LU
with partial pivoting over a doubly linked row/column structure. Both produce
`P A = L U` for the same partial pivoting, so the solutions agree to floating
point; the schedule, the fill ordering and the threading differ. This is the
reason the benchmark shows upstream winning by a wide margin on problems with
substantial fill-in — see the table below. The port deliberately makes no attempt
to reorder for fill.

**2. `_call_pardiso`, phase 22 — the analysis pass.**
PARDISO's phase 22 reuses the ordering phase 12 chose, so the reported `nnzf`
describes the fill of the *numeric* factorisation. Here `mpp_lu_factorize` is run
with `symbolic != 0`, which suppresses the arithmetic and switches the pivot rule
to one that does not read the values (the smallest active row index in the
column, i.e. the diagonal whenever the column has one). The phase 22 `nnzf` is
therefore exact for that static ordering and is usually *smaller* than the phase 13
`nnzf`. `test_phase22_reports_the_static_ordering_fill` pins this behaviour: on a
matrix that needs fill under numeric pivoting, phase 22 reports no fill and phase
13 reports some. The structural pass also stores a placeholder `u_diag` of 1.0
rather than a pivot, so a phase 22 factorisation must never be solved against;
nothing does that, and `test_lu_factorize_symbolic_walk_builds_the_same_pattern_without_values`
pins the difference.

**3. `_check_A`, the CSC branch.**
Upstream sets `iparm(12) = 1` and lets MKL apply the transposition internally.
`mpp_csr_transpose` applies it explicitly instead. It is the same conversion
`scipy.sparse.csc_matrix.tocsr()` performs — count, exclusive prefix sum,
scatter — so `solve(csc, b)` and `solve(csr, b)` return identical results.
`test_csc_input_matches_csr_input` pins that they are bit-for-bit equal.

**4. A structurally empty column raises here and is silently wrong upstream.**
`_check_A` only tests `np.diff(A.indptr).all()`, so a square CSR matrix with no
empty *row* but a column that holds no stored entries passes validation. This
port's step 1 then walks that column, finds no candidate, and returns -2, so
`solve()` raises `PyPardisoError(-2)`.
`test_lu_factorize_reports_minus_two_on_a_structural_zero_pivot` pins the kernel's
answer and `test_singular_matrix_raises_pardiso_error` pins what `solve()` does
with it. A caller that depended on upstream's behaviour on a singular matrix
would see a new exception here.

Two smaller divergences, both consequences of the same thing:

- **`ia = A.indptr + 1`, `ja = A.indices + 1` are dropped.** The `+ 1` exists
  because PARDISO's Fortran ABI is 1-based. These kernels are 0-based throughout
  and hand PARDISO-style 0-based CSR to the factorisation.
- **The factorisation lives on the Python side.** MKL keeps it in its own address
  space, so upstream can answer phase 33 from state it holds privately. Here it is
  visible as `PyPardisoSolver._factors`, and `free_memory()` releases it. Every
  other method keeps its upstream behaviour.

**Not covered:** complex matrix types. Upstream does not support them either
(`mtype` other than 11 is untested upstream and unsupported here); the kernels are
`Float64` throughout. `msglvl` is stored but the statistical info PARDISO prints on
stdout is not emitted, because there is no MKL runtime to print it — the methods
exist and set the flag exactly as upstream does.

**Index width.** The kernels read and write 32-bit indices and 64-bit values.
scipy promotes a sparse matrix's `indptr`/`indices` to `int64` once the matrix is
large enough, so both are narrowed on the way into the kernels and every kernel
destination array is allocated at `int32`. This is hardening rather than a
compatibility requirement: upstream hands the arrays straight to MKL, which reads
them as `int32` too. `test_int64_index_arrays_are_narrowed_not_misread` pins the
behaviour on both formats and both widths.

**Right-hand side layout.** `b` is handed to `mpp_permute_rhs` and
`mpp_lu_solve` as one column-major buffer, so a non-contiguous right-hand side is
materialised first. A 2D `b` that is not Fortran-contiguous becomes Fortran
order, and a 1D `b` that is not C-contiguous — a strided view such as
`v[::2]` — becomes contiguous. Upstream materialises 1D views too, via the
`asfortranarray` it applies unconditionally and which is a no-op on 1D.
`test_solve_handles_every_right_hand_side_stride` pins all three layouts.
A sparse `b` is densified with `toarray` rather than upstream's `todense`: both
hold the same values, but `todense` returns an `np.matrix` whose
`flags.f_contiguous` is false for an `(n, 1)` result, which would silently turn a
one-dimensional right-hand side into a column and return the wrong shape.

## How it works

**FFI strategy.** All kernels live in one compilation unit, `src/ported.mojo`,
compiled by `build/build.sh` with `mojo build --emit shared-lib` into
`dist/libmojo-pypardiso.so`. Each kernel is `@export("mpp_...")` with an explicit
`abi("C")` effect. Because `@export` rejects parametric functions, buffers cross
the ABI as `Int` addresses and are rebuilt inside the wrapper as
`Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=...)`. `_lib.py` holds
the `argtypes` table; the Python layer allocates the numpy buffers and passes
`.ctypes.data`. All allocation stays on the Python side — the kernels never
allocate.

The `argtypes` table is hand-written and nothing in the toolchain checks it, so
`tests/test_kernels.py` parses the parameter list out of `src/ported.mojo` and
compares it entry by entry against `_lib._SIGNATURES`. That is not ceremony: two
errors in it were live at the time of writing. `mpp_lu_solve` declared eleven
slots for a twelve-parameter kernel, so the extra `nworkers` argument was
silently dropped and the solve read uninitialised stack as its worker count; and
`mpp_csr_transpose`'s `cursor` was declared as a count rather than an address,
which made every CSC solve write its scatter cursors through a null pointer.

**Memory layout.** Everything is CSR and 0-based, matching `scipy.sparse`. Indices
are `Int32`, values `Float64`. The factorisation is right-looking: it walks the
columns in order and each step pushes the pivot row into the rows below it.

- `L` is unit lower triangular, stored **by row**, so the forward substitution
  reads each row as one contiguous dot product.
- `U` is upper triangular, stored by row with the diagonal split out into a
  separate `u_diag` array, so the back substitution is a contiguous dot product
  followed by one division.
- `perm[k]` is the original row label that ends up at position `k`, so `P b` is
  the gather `b[perm]`. `mpp_permute_rhs` is that gather.

Both `L` and `u_diag` are indexed by *pivot position*, which is what the
factorisation writes them in. `perm` is a label lookup, not an index into either
array; reading them through `perm` mixes two orderings up and produces a wrong
answer. The port happens to permute with one kernel and solve with the other, so
a mismatch between the two cancels and `solve` still returns the right vector.
`test_lu_factorize_factors_solve_the_system_they_came_from` is the test that
catches it: it feeds the factorisation's own factors into the solve kernel and
checks the result against `A x == b`, which no consistent amount of
mis-indexing survives.

The factorisation works on a flat entry workspace. Each entry is threaded into
both a row list and a column list, so a row and a column can be walked without
searching. Nothing is ever deleted: a row is eliminated by moving past its index,
and the entries left behind are its finished U row. A pivot swaps two rows by
exchanging their list heads and relabelling their entries — `O(1)` for the swap
plus `O(nnz)` for the relabel. The trailing-submatrix update is a rank-one update
per multiplier, using a scatter-gather through `wpos`/`wtag`: stamp the target
row's columns with a tag counter, then look each column up by tag. That finds or
creates the fill entry without an unmarking pass. The tag counter increments once
per multiplier, so the stamps need no clearing.

The workspace must hold every input and fill entry the factorisation ever
creates, and how many that is not known until it is built. The kernel returns
`-3` when the workspace overflows and the Python side retries with four times the
space; `-2` means no candidate pivot in the column, which surfaces as
`PyPardisoError(-2)` (see divergence 4 above for where that differs from
upstream). Both returns are asserted against the kernel directly, in
`test_lu_factorize_reports_minus_two_on_a_structural_zero_pivot`,
`test_lu_factorize_reports_minus_two_on_a_numerically_zero_pivot` and
`test_lu_factorize_reports_minus_three_when_the_workspace_is_too_small`, and the
retry that answers `-3` is pinned by
`test_lu_factorize_retry_recovers_and_the_python_side_drives_it`.

**Threading.** `mpp_lu_solve` spreads independent right-hand sides over worker
threads with `max.algorithm.parallelize`, which is where the capability lives in
this toolchain (`std.algorithm` no longer exports it). The worker count is
clamped to the number of right-hand sides and gated behind a threshold, because
`parallelize` with more workers than work items does not survive being called from
a C host, and because below four right-hand sides the thread launch costs more
than the solve. `MOJO_PYPARDISO_THREADS` overrides the count. The factorisation
is single threaded: its steps are genuinely dependent, so spreading them over
`parallelize` would need a different schedule (a left-looking Gilbert-Peierls
ordering), which is a rewrite rather than a flag.

**The cached-solve path.** `solve(A, b)` decides between phase 13 (factorise and
solve) and phase 33 (solve against what is stored) by asking whether `A` is the
matrix the stored factors came from. Upstream answers that with three
element-by-element `np.array_equal` comparisons over `indptr`, `indices` and
`data`, on every call.

Object identity is a cheap way to answer it, and this port uses it — but only as
a *guard in front of* the comparison, never instead of it. Identity alone is not
trustworthy: a caller may mutate the matrix in place between solves, and scipy's
sparse setters rewrite `data` in the same buffer without replacing it, so the
object survives a matrix that is no longer the one that was factorised. Serving
that matrix from the stored factors returns the solution of a different system
with nothing to show that anything is wrong. This was not hypothetical: before
the guard was added, `A[0, 0] = 30.0` between two `solve` calls on the same
object returned the old solution with a residual of 9.0 where upstream returns
the correct one.

The guard is therefore: same object, *and* small enough that the comparison which
follows is guaranteed to have read every element of the matrix. Above that
threshold the identity is not used at all and every call takes the full
comparison, which is what upstream does unconditionally. The threshold is
`nnz <= 4096` (`_WARM_MIN_NNZ`). The bound is not arbitrary in the other
direction either: on a cache-missed array the comparison can read from main
memory while the array is being rewritten by the caller's in-place edit, and so
see the old bytes, so `_lib.WARM` walks the array first to pin its pages and
load them. On a dense 500x500 matrix that walk costs about 424 us against about
274 us for the comparison it protects — which is why the fast path only pays
below the threshold, where the walk is small enough to be noise against a solve.

`tests/test_parity.py` pins this in both directions: that a repeated `solve` on
the same small object still takes phase 33 and gives the same answer as a fresh
solver, that the same object above the threshold still reaches phase 33 by the
other route, that an in-place mutation is *not* taken as factorised, that
releasing the factorisation clears the recorded object, and that an
equal-but-distinct matrix is handled by the comparison rather than the identity.

## Tests

```bash
pixi run test
```

151 tests, all asserting numerical or behavioural parity rather than "it ran":

- `L @ U == A[perm]` to machine precision for ten matrix shapes (1x1 through
  300x300, tridiagonal, 2D Laplacian, random sparse, weakly diagonal so pivoting
  is forced).
- Solution vectors compared against the real upstream `pypardiso` (MKL PARDISO)
  on the same inputs, to `1e-9`.
- Relative residual `||A x - b|| / ||b|| < 1e-10`, independent of any other solver.
- A randomised sweep of four structural families (well conditioned, weak diagonal
  with forced pivoting, tridiagonal, pentadiagonal) in both CSR and CSC, against
  `numpy.linalg.solve`. Upstream MKL PARDISO is deliberately not the oracle here,
  because on a CSC matrix from this family it returned a solution with a residual
  of order 4 while the CSR path on the same matrix was exact; the check is
  against a dense solve instead.
- Right-hand sides passed as C-contiguous, Fortran-contiguous and strided views,
  across 1, 2 and 5 right-hand sides, all agreeing with `numpy.linalg.solve`.
- `int32` and `int64` index arrays on both the CSR and the CSC path.
- **Every one of the five exported kernels called by name through ctypes**, with
  the test allocating and sizing its own buffers, in `tests/test_kernels.py`:
  - `mpp_indptr_diffs` against `np.diff`, at `n` of 0, 1 and 64.
  - `mpp_csr_transpose` against `csc_matrix.tocsr()`, against its own `-1`
    entry-count guard, and on an input with duplicate entries that it must
    reindex rather than coalesce.
  - `mpp_permute_rhs` against a manual gather, for 1, 2 and 5 right-hand sides,
    and with the identity permutation into a poisoned destination.
  - `mpp_lu_factorize` against `L @ U == A[perm]`, against its `-2` returns (a
    structurally empty column and a numerically zero pivot) and its `-3` return,
    against `perm[0]` naming the largest-magnitude row, against `nnzf` counting
    every stored entry once, against each U row coming out sorted by column, on a
    1x1 matrix, and on the structural walk that shares its pivot rule.
  - `mpp_lu_solve` against `numpy.linalg.solve(L @ U, b)` on factors the test
    packs itself, for 1, 3 and 6 right-hand sides; against `b` being left
    untouched; against the column-major layout of a 3-column right-hand side;
    against `x` being fully written when the right-hand side is zero; and serial
    against threaded.
  - The cross-kernel contract: `mpp_lu_factorize`'s factors fed through
    `mpp_permute_rhs` and `mpp_lu_solve`, checked against `A x == b`.
  - `_lib._SIGNATURES` compared against the parameter lists parsed out of
    `src/ported.mojo`, entry by entry, in both directions: a buffer parameter
    must be a `c_void_p` and a count parameter must not be.
- API equivalence with upstream by reflection: exported names, public method set,
  parameter names and defaults for every public method.
- Every documented error path: non-square, non-CSR/CSC, non-float64, empty rows
  and columns, dimension mismatch, unsupported dtype, singular matrix, phase 33
  with no stored factorisation.

**A build failure fails the run; only a missing toolchain skips it.** Every test
is gated on a `compiled_lib` fixture that force-rebuilds `src/ported.mojo` and
asserts the loaded `CDLL` really is `dist/libmojo-pypardiso.so`. Those two
conditions are what stop a pure-Python path from satisfying a kernel claim.

The fixture distinguishes two ways the library can be unavailable. If there is
no `mojo` compiler at all, there is nothing to assert against and the tests
`skip` with the reason printed. If a compiler *is* present and the build then
fails, the fixture `fail`s. That distinction matters: an earlier version
skipped on both, so a source that does not compile produced `exit 0` with every
kernel assertion unrun -- a green run that proved nothing.

```
$ MOJO_PYPARDISO_MOJO=/nonexistent/mojo pixi run test
2 passed, 149 skipped
```

The two tests that still run without the library are the two that never call a
kernel: `test_ctypes_argtypes_match_the_declared_kernel_signatures` compares the
`argtypes` table against the Mojo *source*, and
`test_indptr_diff_kernel_finds_the_empty_row` asks for neither the
`compiled_lib` nor the `ours` fixture, so it goes on to load whatever
`dist/libmojo-pypardiso.so` is already there and calls it. Neither can satisfy a
claim about kernel behaviour; they survive the missing toolchain by not being
gated on it, and neither replaces the gated tests that do.

## Benchmarks

Measured on a machine shared with other jobs. The table below is the real output
of `pixi run bench`, which takes a machine-wide `flock` so a concurrent factory
job cannot distort it. Best of 3 runs per cell, each measured in its own process.

The `flock` waits up to 900 s for the shared lock and then gives up rather than
hanging forever; if it times out, exit status 1 and no table is produced. The
lock is machine-wide and shared with other checkouts, so a long-running job
elsewhere can hold it; `pixi run bench-nolock` runs the identical benchmark
without it, for when that happens.

**This machine is shared and the noise is large.** Eight full runs of this
benchmark were taken on this machine while preparing this table, each with a
load average between 68 and 137 on 72 logical CPUs. Individual cells move a
lot — the upstream column alone ranged from 18.81 ms to 75.33 ms on
`dense 200x200` and from 36.04 ms to 155.19 ms on the Laplacian. Treat the ratio
column as indicative, not precise. The table below gives the ratio range
observed across those runs; `n` is how many of them measured that case, which is
lower for some only because three of the early runs were cut short by the 900 s
`flock` timeout before reaching them.

| case | n | ratio range | |
| --- | --- | --- | --- |
| solve: banded 20000 | 7 | 0.01x - 0.05x | a win on every run |
| spsolve alias: laplacian 60x60 | 4 | 0.07x - 0.18x | a win on every run |
| solve cached: dense 500x500, 8 rhs | 7 | 0.60x - 1.27x | around parity, on both sides |
| solve: dense 200x200 | 7 | 0.36x - 1.37x | on or near parity, on both sides |
| solve: dense 400x400 | 7 | 2.10x - 5.23x | a loss that grows with `n` |
| solve: dense 500x500, 32 rhs | 7 | 2.10x - 4.51x | a stable loss |
| factorize: dense 800x800 | 7 | 7.82x - 11.57x | a stable loss |
| solve: dense 800x800 | 7 | 5.47x - 10.06x | a loss that grows with `n` |
| solve: laplacian 80x80 | 7 | 9.73x - 33.52x | the worst row on every run |

```
machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs, Python 3.13.15, numpy 2.5.3
```

| case | mojo-pypardiso | upstream pypardiso (MKL) | ratio |
| --- | --- | --- | --- |
| solve: dense 200x200 (1 rhs) | 28.40 ms | 20.68 ms | 1.37x |
| solve: dense 400x400 (1 rhs) | 281.11 ms | 101.01 ms | 2.78x |
| solve: dense 800x800 (1 rhs) | 2430.36 ms | 291.71 ms | 8.33x |
| factorize: dense 800x800 | 2220.84 ms | 236.24 ms | 9.40x |
| solve: dense 500x500, 32 rhs | 575.05 ms | 127.53 ms | 4.51x |
| solve: banded 20000 (1 rhs) | 5.07 ms | 102.52 ms | 0.05x |
| solve: laplacian 80x80 grid (6400) | 1502.64 ms | 81.20 ms | 18.50x |
| spsolve alias: laplacian 60x60 (3600) | 1.17 ms | 6.56 ms | 0.18x |
| solve cached: dense 500x500, 8 rhs | 14.31 ms | 11.27 ms | 1.27x |
| phase 22 analysis: dense 800x800 | 1876.64 ms | crashed | - |

This table is the `bench` step of the final `pixi run build && pixi run test &&
pixi run bench` against this source, so it is the measurement that ships with
it. Two of the nine comparable cases are faster than upstream MKL PARDISO; the
median ratio is 2.78x, the best 0.05x, the worst 18.50x. Ratios above 1 are
losses, stated as they came out.

Reading the table:

- **Decisively faster on the banded case.** A tridiagonal factorisation stays
  banded, so there is no fill at all and the work reduces to a flat triangular
  solve. This is the case the port is genuinely good at: no fill, cache-resident
  rows, one pass. It is the one result that has held on every recorded run, and
  the one where upstream is paying for a general-purpose solver it does not need.
- **On or near parity on the two smallest cases** — `dense 200x200` (1.37x here,
  0.36x to 1.37x across runs) and the cached `dense 500x500, 8 rhs` solve (1.27x
  here, 0.60x to 1.27x across runs). Both have been recorded on both sides of
  parity. Neither should be read as a result; what is stable is that on these
  cells both solvers are in the same order of magnitude, unlike every other row.
  The `spsolve` alias on the 60x60 grid is a third, and a clearer, win: 0.18x
  here, 0.07x to 0.18x on every run that measured it.
- **Behind on the larger dense cases.** MKL's supernodal blocking and threaded
  BLAS inside a multifrontal solver are a different machine. The gap widens with
  `n` because this port's inner loop is a single-threaded scalar rank-one update.
- **Behind on the 2D Laplacian**, which is the fill-in story rather than a
  constant-factor story. Under the natural ordering a grid Laplacian produces
  quadratic fill, and MKL reorders (nested dissection / minimum degree) while this
  port does not. That is the worst row on every run.
- **Upstream crashed on phase 22** for a dense 800x800 matrix. That is an upstream
  bug, not a measurement failure: `mkl_pardiso` segfaults reproducibly on that
  path in this MKL build, which is why the bench measures each cell in its own
  subprocess and reports the crash in the table rather than taking the whole
  table down with it.

The honest summary: this is a correct, dependency-light reimplementation of the
algorithm, and it is fast exactly where sparse LU is easy — no fill,
cache-resident rows. It does not change the conclusion that this port loses to
MKL wherever fill-in matters. The two changes that would close most of that gap,
neither done here, are a fill-reducing ordering (minimum degree or nested
dissection) before the factorisation, and a left-looking schedule whose steps
are independent enough to thread.

## Layout

```
src/ported.mojo           the five exported kernels, in upstream source order
build/build.sh            mojo build --emit shared-lib -> dist/libmojo-pypardiso.so
python/mojo_pypardiso/    ctypes glue and the upstream-compatible Python API
    _lib.py               loads or rebuilds the .so, holds the argtypes table
    pardiso_wrapper.py    PyPardisoSolver, mirroring pypardiso/pardiso_wrapper.py
    scipy_aliases.py      spsolve / factorized, mirroring pypardiso/scipy_aliases.py
tests/                    pytest
    conftest.py           shared fixtures and matrix generators
    test_api.py           the Python API, compared against upstream by reflection
    test_parity.py        numerical parity with upstream, scipy, and ground truth
    test_kernels.py       every exported kernel called by name through ctypes
bench/bench.py            benchmarks against upstream, prints a markdown table
```

`src/ported.mojo` reads top-to-bottom in upstream's source order: each section is
named after the upstream method whose work it carries, and the sections appear in
the order upstream uses them.

## Licence

MIT, as shipped in `LICENSE`.
