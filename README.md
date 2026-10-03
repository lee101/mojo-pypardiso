# mojo-pypardiso

A Mojo port of the [pypardiso](https://github.com/haasad/PyPardisoProject) sparse
direct solver, with the same Python API.

Upstream `pypardiso` is a thin ctypes wrapper: it loads Intel MKL's `mkl_pardiso`
and forwards every call to it. Almost nothing is computed in Python. This port
keeps the Python surface byte-for-byte compatible and replaces the closed binary
with a Mojo implementation of the same computation — a sparse LU factorisation of
`P A` with partial pivoting, plus the triangular solves against the stored
factors — compiled to `dist/libmojo-pypardiso.so` and called through ctypes.

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
ported. There are three places where the port is not identical to upstream, all
of them forced by MKL PARDISO being a proprietary binary rather than by anything
Mojo cannot express.

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
13 reports some.

**3. `_check_A`, the CSC branch.**
Upstream sets `iparm(12) = 1` and lets MKL apply the transposition internally.
`mpp_csr_transpose` applies it explicitly instead. It is the same conversion
`scipy.sparse.csc_matrix.tocsr()` performs — count, exclusive prefix sum,
scatter — so `solve(csc, b)` and `solve(csr, b)` return identical results.

This is not only equivalent, it is more reliable than the upstream path. Feeding
upstream a CSC matrix with a weak diagonal, so that partial pivoting has to
interleave rows, it returns a wrong answer: on a deterministic 38x38 case its
relative error against `numpy.linalg.solve` was 1.5, repeated identically over
five runs, while the same matrix through its own CSR path was exact to 3e-15.
This port returns 6e-15 for both. That is an upstream defect, not a
compatibility break, but it means "identical results to upstream" is not
achievable on this path and should not be claimed.

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
them as `int32` too, so `int32` input already works there. But an `int64`-indexed
CSC matrix reaching `mpp_csr_transpose` used to leave the high half of each index
slot at whatever the destination buffer happened to hold, which is wrong indices
rather than an error. `test_int64_index_arrays_are_narrowed_not_misread` pins the
behaviour on both formats and both widths.

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
space; `-2` means a zero pivot, which upstream surfaces as `PyPardisoError(-2)`.

**Threading.** `mpp_lu_solve` spreads independent right-hand sides over worker
threads with `max.algorithm.parallelize`, which is where the capability lives in
this toolchain (`std.algorithm` no longer exports it). The worker count is
clamped to the number of right-hand sides and gated behind a threshold, because
`parallelize` with more workers than work items does not survive being called from
a C host, and because below four right-hand sides the thread launch costs more
than the solve. `MOJO_PYPARDISO_THREADS` overrides the count. The factorisation
is single threaded; see the benchmarking section for why it cannot simply be
made parallel without first changing its schedule.

## Tests

```bash
pixi run test
```

113 tests, all asserting numerical or behavioural parity rather than "it ran":

- `L @ U == A[perm]` to machine precision for ten matrix shapes (1x1 through
  300x300, tridiagonal, 2D Laplacian, random sparse, weakly diagonal so pivoting
  is forced).
- Solution vectors compared against the real upstream `pypardiso` (MKL PARDISO)
  on the same inputs, to `1e-9`.
- Relative residual `||A x - b|| / ||b|| < 1e-10`, independent of any other solver.
- A randomised sweep of four structural families (well conditioned, weak diagonal
  with forced pivoting, tridiagonal, pentadiagonal) in both CSR and CSC, against
  `numpy.linalg.solve`. Over 200 such cases the worst relative error observed
  was `1.5e-13`, with no exceptions. Upstream MKL PARDISO is deliberately not
  the oracle here, because it is wrong on the CSC path for a pivoting-heavy
  matrix; see the CSC branch note above.
- Right-hand sides passed as C-contiguous, Fortran-contiguous and strided views,
  across 1, 2 and 5 right-hand sides, all agreeing with `numpy.linalg.solve`.
- `int32` and `int64` index arrays on both the CSR and the CSC path.
- Each kernel called directly through ctypes and checked against its definition:
  `mpp_csr_transpose` against `csc_matrix.tocsr()`, `mpp_indptr_diffs` against
  `np.diff`, `mpp_permute_rhs` against the gather, threaded against serial solve.
- The phase 22 structural walk and the numeric walk are asserted to produce
  identical index patterns on a matrix whose two pivot rules agree, and a
  matrix that provably creates fill is checked for `L @ U == A[perm]` so the
  fill-appending branch of the rank-one update is covered.
- API equivalence with upstream by reflection: exported names, public method set,
  parameter names and defaults for every public method.
- Every documented error path: non-square, non-CSR/CSC, non-float64, empty rows
  and columns, dimension mismatch, unsupported dtype, singular matrix, phase 33
  with no stored factorisation.

## Benchmarks

Measured on a machine shared with other jobs. The table below is the real output
of `pixi run bench`, which takes a machine-wide `flock` so a concurrent factory
job cannot distort it. Best of 3 runs per cell, each measured in its own process.

The `flock` waits up to 900 s for the shared lock and then gives up rather than
hanging forever; if it times out, exit status 1 and no table is produced. The
lock is machine-wide and shared with other checkouts, so a long-running job
elsewhere can hold it; `pixi run bench-nolock` runs the identical benchmark
without it, for when that happens. The run below was taken on a free lock with a
load average near 83 on 72 logical CPUs from other jobs, so every number here is
a measurement under contention rather than an idle-machine figure.

Across three full runs of this benchmark, the `mojo-pypardiso` column varied by at
most 1.24x and the upstream column by at most 1.71x, so both are noisy to roughly
that degree and any single ratio within about 1.5x of parity should not be read as
a win in either direction. The banded case wins decisively on every run
(0.07x-0.14x). The `spsolve` alias case is the marginal one: it landed at 0.79x,
1.28x and 1.01x across the three runs, so it is a wash rather than a result. The
dense and Laplacian cases lose by margins far larger than the noise.

The table below is one complete run, reproduced verbatim.

```
machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs,
         Python 3.13.15, numpy 2.5.3
```

| case | mojo-pypardiso | upstream pypardiso (MKL) | ratio |
| --- | --- | --- | --- |
| solve: banded 20000 (1 rhs) | 3.69 ms | 33.64 ms | 0.11x |
| spsolve alias: laplacian 60x60 (3600) | 1.10 ms | 0.85 ms | 1.28x |
| solve: dense 200x200 (1 rhs) | 21.98 ms | 13.45 ms | 1.63x |
| solve: dense 400x400 (1 rhs) | 156.56 ms | 27.21 ms | 5.75x |
| solve: dense 500x500, 32 rhs | 302.29 ms | 57.67 ms | 5.24x |
| solve: dense 800x800 (1 rhs) | 1255.61 ms | 136.46 ms | 9.20x |
| factorize: dense 800x800 | 1244.73 ms | 99.94 ms | 12.45x |
| solve: laplacian 80x80 grid (6400) | 785.48 ms | 28.40 ms | 27.66x |
| phase 22 analysis: dense 800x800 | 1147.37 ms | crashed | - |

One of the eight comparable cases in this run is faster than upstream MKL
PARDISO, the banded case by a wide margin. The `spsolve` alias row landed on the
losing side here at 1.28x but won the next run at 0.79x, so treat it as a wash.
The ratios above 1 are losses, stated as they came out.

### Where the time goes

Profiling each stage separately (`bench/profile_kernels.py`) shows the
factorisation is essentially the whole runtime. A representative run:

| stage | dense 800 | laplacian 80x80 |
| --- | --- | --- |
| whole `solve()` | 1844.6 ms | 781.5 ms |
| `mpp_lu_factorize` (`_analyze`, numeric) | 2444.1 ms | 801.3 ms |
| of which `np.zeros` allocation | 7.0 ms | 0.2 ms |
| `mpp_permute_rhs` | 0.011 ms | 0.011 ms |
| `mpp_lu_solve` (triangular solves) | 1.24 ms | 1.87 ms |

(The `_analyze` column is timed on its own and so can exceed the whole `solve()`
column: it is the cold-cache single call, whereas `solve()` reuses the factorised
matrix across repeats. The point of the table is the ratio between the
factorisation and everything else, which is over 99% either way.)

So the factorisation's rank-one update is the only thing worth optimising, and
the two changes this pass made to `mpp_lu_factorize` both target it:

- **Step 5 streams the pivot row from the contiguous sorted run that step 3
  already wrote**, instead of walking the entry list again through `nextr`.
  Same entries, same order, but one sequential stream instead of a pointer
  chase, and the column and its value arrive together so the inner loop no
  longer touches the entry workspace.
- **The loop-invariant `symbolic` test is hoisted out of the inner loop** and
  the fill branch is split into its own path, so the numeric update is a
  straight line over the common case.

There is deliberately **no before/after speedup table** here. The comparison this
port could show would be against a revision of `mpp_lu_factorize` that is not in
this repository, so neither the "before" nor the "after" column could be
reproduced by anyone who clones it. The numbers in the table above are the ones
this repository actually produces today, and `bench/profile_kernels.py` and
`bench/profile_counts.py` regenerate the stage and iteration-count figures.

`bench/profile_counts.py` counts the step-5 inner loop, which runs once per
(multiplier, pivot-row-entry) pair:

| case | nnz(L+U) | step-5 inner iterations |
| --- | --- | --- |
| dense 200 | 39,800 | 1,313,400 |
| dense 400 | 159,600 | 10,586,800 |
| dense 800 | 639,200 | 85,013,600 |
| laplacian 80x80 | 1,011,358 | 40,192,117 |

One further idea was tried and **reverted**: packing the tag and the entry
position into a single 64-bit slot so the update probe is one load instead of a
dependent `wtag` then `wpos`. It was slower at every size, because the slot array
is the hottest randomly-indexed array in the loop and doubling it to 8 bytes per
column costs more than the saved load. `bench/probe_scatter.py` records why the
obvious "just SIMD it" answer does not apply here: the update is a dependent
scatter (one indirect address per entry), and on this machine's real access shape
a blocked-SIMD rewrite is not faster than the scalar scatter-gather. Measured on
the two shapes that matter, an ordered merge beats the stamped probe by only
1.22x (dense 800) and 1.08x (laplacian 80x80), which is not enough to pay for the
sorted invariant it would require.

### Why the factorisation is not threaded, and why there is no GPU path

- **Parallelism.** The factorisation is right-looking on the matrix columns, so
  step `k+1` reads the fill that step `k` produced. The steps are genuinely
  dependent, so they cannot be spread over `max.algorithm.parallelize` without
  first changing the schedule to one whose steps are independent (a left-looking
  Gilbert-Peierls ordering). That is a rewrite of the kernel, not a flag, and it
  was not attempted here. The triangular solves already use `parallelize`
  across right-hand sides where that decomposition is valid.
- **GPU.** The `max.gpu.host` path is available and works on this toolchain:
  `bench/probe_gpu.py` builds a `DeviceContext`, runs one kernel over 2^20
  elements with a correct tail, and synchronizes (the probe also confirms the
  API moved to `max.gpu`; `std.gpu` is gone). It is not shipped because the
  rank-one update is **0.06 flops per byte** — two flops against roughly 36
  bytes touched per entry — and, more fundamentally, it is a dependent scatter
  with a sequential dependency between steps. A GPU does not help a loop whose
  iterations must run in order and whose arithmetic intensity is an order of
  magnitude below the point where a GPU wins. That is an algorithmic property,
  not a missing API.

Reading the table:

- **About 9x faster on the banded case.** A tridiagonal factorisation stays banded,
  so there is no fill at all and the work reduces to a flat triangular solve. This
  is the case the port is genuinely good at: no fill, cache-resident rows, one pass.
- **Roughly level on the small `spsolve` alias case**, which measured 0.79x, 1.28x
  and 1.01x on three runs, i.e. inside the noise.
- **Behind on every dense case**, by 1.6x at n=200 rising to 12.5x on the
  factorisation alone at n=800. MKL's supernodal blocking and threaded BLAS
  inside a multifrontal solver are a different machine. The gap widens with n
  because this port's inner loop is a single-threaded scalar rank-one update.
- **Behind on the 2D Laplacian**, which is the fill-in story rather than a
  constant-factor story. Under the natural ordering a grid Laplacian produces
  quadratic fill — `bench/profile_counts.py` measures 1,011,358 factor entries
  for a matrix with 31,680 nonzeros — and MKL reorders (nested dissection /
  minimum degree) while this port does not. The same matrix at 60x60 is *level*
  because it is small enough that the constant wins over the asymptotic penalty.
- **Upstream crashed on phase 22** for a dense 800x800 matrix. That is an upstream
  bug, not a measurement failure: `mkl_pardiso` segfaults reproducibly on that
  path in this MKL build (confirmed twice in a row, dumping core), which is why
  the bench measures each cell in its own subprocess and reports the crash in the
  table.

The honest summary: this is a correct, complete, dependency-light
reimplementation of the algorithm, and it is fast exactly where sparse LU is easy
— no fill, cache-resident rows. It does not change the conclusion that this port
loses to MKL wherever fill-in matters. Two further changes would close most of
that gap and neither is done here: a fill-reducing ordering (minimum degree or
nested dissection) before the factorisation, and a left-looking schedule whose
steps are independent enough to thread.

## Layout

```
src/ported.mojo           the five exported kernels, in upstream source order
build/build.sh            mojo build --emit shared-lib -> dist/libmojo-pypardiso.so
python/mojo_pypardiso/    ctypes glue and the upstream-compatible Python API
    _lib.py               loads or rebuilds the .so, holds the argtypes table
    pardiso_wrapper.py    PyPardisoSolver, mirroring pypardiso/pardiso_wrapper.py
    scipy_aliases.py      spsolve / factorized, mirroring pypardiso/scipy_aliases.py
tests/                    pytest; numerical parity against upstream pypardiso
bench/bench.py            benchmarks against upstream, prints a markdown table
bench/profile_kernels.py  per-stage timing of one solve, the optimisation target
bench/profile_counts.py   how many inner iterations the factorisation performs
bench/probe_scatter.py    why the update loop is not vectorised
bench/probe_gpu.py        the max.gpu.host path, and this kernel's intensity
```

`src/ported.mojo` reads top-to-bottom in upstream's source order: each section is
named after the upstream method whose work it carries, and the sections appear in
the order upstream uses them.

## Licence

MIT, as shipped in `LICENSE`.