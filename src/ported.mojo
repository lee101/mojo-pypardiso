"""Mojo kernels for `mojo-pypardiso`, a port of the `pypardiso` sparse direct solver.

Upstream `pypardiso` computes nothing in Python: `pypardiso/pardiso_wrapper.py`
loads `mkl_rt` with ctypes and forwards every call to `mkl_pardiso`, a
proprietary binary. The two things that file actually computes in Python are

    _check_A   -> `np.diff(A.indptr).all()`        (kernel 1 below)
    _call_pardiso
        -> `ia = A.indptr + 1`, `ja = A.indices + 1`
        -> the ctypes invocation

Everything else is argument marshalling. The MKL PARDISO solver itself is the
compute-bound part of the library: the sparse LU factorisation of `PA` with
partial pivoting (phase 12/13), the value-independent analysis pass (phase 22),
and the triangular solves against the stored factors (phase 13/33).

This compilation unit reproduces those kernels, and it reads top-to-bottom in
upstream's source order: each section is named after the upstream method whose
work it carries, and within a section the branch structure follows the phase
protocol upstream drives through `self.phase`.

    _check_A      -> mpp_indptr_diffs, mpp_csr_transpose
    _call_pardiso -> mpp_permute_rhs, mpp_lu_factorize, mpp_lu_solve

Documented divergences from upstream, all of them forced by MKL PARDISO being
a closed binary rather than by anything Mojo cannot express:

* `_call_pardiso` (phases 12/13). MKL runs a multifrontal supernodal solver.
  `mpp_lu_factorize` runs a right-looking sparse LU with partial pivoting over
  a doubly linked row/column structure, which yields the same `PA = LU`. Same
  mathematics, different schedule.
* `_call_pardiso` (phase 22). MKL's analysis pass uses a static ordering that
  does not depend on the values. So does this one: `symbolic != 0` picks the
  pivot with the smallest row index in the column, which is the diagonal entry
  when the column has one. The `nnz` it reports therefore belongs to that
  static ordering, not to the numeric factorisation.
* `_check_A` (CSC branch). Upstream sets `iparm(12) = 1` and lets MKL apply the
  transposition internally. `mpp_csr_transpose` applies it explicitly instead,
  which is the same conversion `csc_matrix.tocsr()` performs.
* `_call_pardiso` (`ia`/`ja`). The `+ 1` exists because PARDISO's Fortran ABI
  is 1-based. These kernels are 0-based throughout, so the shift is dropped.

All indices are 0-based and all buffers are CSR, matching `scipy.sparse`.
"""

from max.algorithm import parallelize

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def fptr(a: Int) -> FPtr:
    return FPtr(unsafe_from_address=a)


def iptr(a: Int) -> IPtr:
    return IPtr(unsafe_from_address=a)


@export("mpp_indptr_diffs")
def mpp_indptr_diffs(indptr: Int, diff: Int, n: Int) abi("C") -> Int:
    """`_check_A`: `np.diff(A.indptr)`, counting how many differences are zero.

    `_check_A` raises when this is non-zero: a square matrix with an empty row
    is singular, and on the CSC path (which is PARDISO's `iparm(12) = 1`) the
    same test on the CSC arrays finds an empty column instead.
    """
    # the difference is kept in a local instead of being reloaded out of `d`
    # on the very next line, which is what this loop used to do
    var ip = iptr(indptr)
    var d = iptr(diff)
    var zero = 0
    for i in range(n):
        var diff = ip.unsafe_load(i + 1) - ip.unsafe_load(i)
        d.unsafe_store(i, diff)
        if diff == 0:
            zero += 1
    return zero


@export("mpp_csr_transpose")
def mpp_csr_transpose(
    src_ptr: Int,
    src_idx: Int,
    src_val: Int,
    dst_ptr: Int,
    dst_idx: Int,
    dst_val: Int,
    n: Int,
    nnz: Int,
    cursor: Int,
) abi("C") -> Int:
    """`_check_A` CSC branch: the transposition PARDISO does for `iparm(12) = 1`.

    `src` is the input in one sparse format and `dst` the same square matrix in
    the other, so this is exactly what `csc_matrix.tocsr()` computes: count the
    entries belonging to each row of the result, take the exclusive prefix sum
    for the destination `indptr`, then scatter. `cursor` is an `n`-element
    scratch holding the running fill position per row, because `indptr` is
    already written by the time the scatter starts. Returns 0, or -1 if the two
    formats disagree about the number of stored entries.
    """
    var sp = iptr(src_ptr)
    var si = iptr(src_idx)
    var sv = fptr(src_val)
    var dp = iptr(dst_ptr)
    var di = iptr(dst_idx)
    var dv = fptr(dst_val)
    var cur = iptr(cursor)

    for r in range(n):
        dp.unsafe_store(r, 0)
    for j in range(n):
        for k in range(sp.unsafe_load(j), sp.unsafe_load(j + 1)):
            var r = si.unsafe_load(k)
            dp.unsafe_store(r, dp.unsafe_load(r) + 1)

    var acc = 0
    for r in range(n):
        var c = dp.unsafe_load(r)
        dp.unsafe_store(r, Int32(acc))
        cur.unsafe_store(r, Int32(acc))
        acc += Int(c)
    if acc != nnz:
        return -1
    dp.unsafe_store(n, Int32(acc))

    for j in range(n):
        for k in range(sp.unsafe_load(j), sp.unsafe_load(j + 1)):
            var r = si.unsafe_load(k)
            var p = cur.unsafe_load(r)
            di.unsafe_store(p, Int32(j))
            dv.unsafe_store(p, sv.unsafe_load(k))
            cur.unsafe_store(r, p + 1)
    return 0


@export("mpp_permute_rhs")
def mpp_permute_rhs(n: Int, nrhs: Int, r_perm: Int, b: Int, pb: Int) abi("C"):
    """`_call_pardiso`: PARDISO's `(P A) x = (P b)` row permutation of `b`.

    `b` and `pb` are `(n, nrhs)` column-major, which is the Fortran-contiguous
    layout `_check_b` hands to PARDISO. MKL applies `perm` internally; here it
    is an explicit pass so the permuted right-hand side is visible.
    """
    var perm = iptr(r_perm)
    var bv = fptr(b)
    var pv = fptr(pb)
    for r in range(nrhs):
        var off = r * n
        for i in range(n):
            pv.unsafe_store(off + i, bv.unsafe_load(off + Int(perm.unsafe_load(i))))


@export("mpp_lu_factorize")
def mpp_lu_factorize(
    n: Int,
    cap: Int,
    symbolic: Int,
    a_ptr: Int,
    a_idx: Int,
    a_val: Int,
    l_ptr: Int,
    l_idx: Int,
    l_val: Int,
    u_ptr: Int,
    u_idx: Int,
    u_val: Int,
    u_diag: Int,
    r_perm: Int,
    o_nnzf: Int,
    w_val: Int,
    w_row: Int,
    w_col: Int,
    w_nextc: Int,
    w_nextr: Int,
    col_head: Int,
    row_head: Int,
    row_label: Int,
    w_pos: Int,
    w_tag: Int,
    l_lab: Int,
    scratch: Int,
    l_entry: Int,
    u_cur: Int,
) abi("C") -> Int:
    """`_call_pardiso` phases 12/13 (and 22): sparse LU with partial pivoting.

    Produces `P A = L U` for the CSR matrix `A`, with `L` unit lower
    triangular stored by row (row `j` holds `L[j, i]` for `i < j`) and `U`
    upper triangular stored by row (row `j` holds `U[j, c]` for `c > j`, with
    the diagonal split out into `u_diag`). Storing `L` by row is what lets the
    forward substitution of `mpp_lu_solve` read a row as one contiguous dot
    product.

    `r_perm[k]` is the original row label that ends up at position `k`, so
    `Pb[i] = b[r_perm[i]]`. `o_nnzf` receives `nnz(L) + nnz(U)`, the number of
    stored factor entries the diagonal counted once.

    `symbolic != 0` is phase 22: the identical structural walk with the
    arithmetic suppressed, so the fill pattern is filled in but no values are
    formed. That requires a pivot rule that does not look at the values, and it
    takes the smallest active row index in the column.

    Returns 0 on success, -2 on a structurally or numerically zero pivot, and
    -3 when the `cap`-sized entry workspace overflows; the caller then retries
    with a larger `cap`.
    """
    var ap = iptr(a_ptr)
    var ai = iptr(a_idx)
    var av = fptr(a_val)

    var val = fptr(w_val)
    var row = iptr(w_row)
    var col = iptr(w_col)
    var nextc = iptr(w_nextc)
    var nextr = iptr(w_nextr)
    var chead = iptr(col_head)
    var rhead = iptr(row_head)
    # `rlabel[r]` is the original matrix row that currently sits at position
    # `r`. The `row[]` field of an entry is that current position, so it cannot
    # answer "which original row was this"; only this table can, and `perm` is
    # read out of it.
    var rlabel = iptr(row_label)
    var wpos = iptr(w_pos)
    var wtag = iptr(w_tag)
    # Multipliers are appended column-major with the owning row's ORIGINAL
    # label alongside, because a later pivot step can still move a live row to
    # a different position and the entries already emitted move with it. The
    # row-major L is assembled from these in the closing pass.
    var llab = iptr(l_lab)
    var lcnt = 0
    var scr = iptr(scratch)
    var ccap = Int32(cap)

    var lp = iptr(l_ptr)
    var li = iptr(l_idx)
    var lv = fptr(l_val)
    var up = iptr(u_ptr)
    var ui = iptr(u_idx)
    var uv = fptr(u_val)
    var ud = fptr(u_diag)
    var perm = iptr(r_perm)
    # `l_entry` records the workspace entry each emitted multiplier came from,
    # so the closing pass can read its column and its value.
    var lentry = iptr(l_entry)
    var ucu = iptr(u_cur)
    var nnzf = iptr(o_nnzf)

    for r in range(n + 1):
        chead.unsafe_store(r, -1)
    for r in range(n):
        rhead.unsafe_store(r, -1)
        rlabel.unsafe_store(r, Int32(r))
    up.unsafe_store(0, 0)
    # `ucnt` is the running write position of U; every row's `u_ptr` entry is
    # written when that row is finalized, and `u_ptr[n]` at the end.
    var ucnt: Int32 = 0

    # Entries live in one flat workspace, threaded into both the row list and
    # the column list. Nothing is ever deleted: a row is eliminated by moving
    # past its index, and the entries left behind are its finished U row.
    var cnt: Int32 = 0
    for r in range(n):
        for p in range(ap.unsafe_load(r), ap.unsafe_load(r + 1)):
            if cnt >= ccap:
                return -3
            var e = cnt
            cnt += 1
            var c = ai.unsafe_load(p)
            val.unsafe_store(e, av.unsafe_load(p))
            row.unsafe_store(e, Int32(r))
            col.unsafe_store(e, c)
            nextr.unsafe_store(e, rhead.unsafe_load(r))
            rhead.unsafe_store(r, e)
            nextc.unsafe_store(e, chead.unsafe_load(c))
            chead.unsafe_store(c, e)

    # The tag counter stamps scatter-gather slots so a row can be marked
    # without an equally long unmarking pass at the end of the step.
    var tag: Int32 = 0
    for k in range(n):
        # 1. Walk column k. Entries in rows above k are the finished U rows
        #    and take no part; the rest is this column's live segment.
        var na: Int32 = 0
        var pv: Int32 = -1
        var bestabs = 0.0
        var bestrow: Int32 = -1
        var e = chead.unsafe_load(k)
        while e != -1:
            var rr = row.unsafe_load(e)
            if rr >= Int32(k):
                scr.unsafe_store(na, e)
                na += 1
                if symbolic != 0:
                    if bestrow == -1 or rr < bestrow:
                        bestrow = rr
                        pv = e
                else:
                    var a = val.unsafe_load(e)
                    if a < 0.0:
                        a = -a
                    if pv == -1 or a > bestabs:
                        bestabs = a
                        pv = e
            e = nextc.unsafe_load(e)
        if pv == -1:
            return -2
        if symbolic == 0 and bestabs == 0.0:
            return -2

        # 2. Partial pivot: the chosen entry's row becomes row k.
        var p = row.unsafe_load(pv)
        if p != Int32(k):
            var hk = rhead.unsafe_load(k)
            var hp = rhead.unsafe_load(p)
            rhead.unsafe_store(k, hp)
            rhead.unsafe_store(p, hk)
            var g = hp
            while g != -1:
                row.unsafe_store(g, Int32(k))
                g = nextr.unsafe_load(g)
            g = hk
            while g != -1:
                row.unsafe_store(g, p)
                g = nextr.unsafe_load(g)
            var hlabel = rlabel.unsafe_load(k)
            rlabel.unsafe_store(k, rlabel.unsafe_load(p))
            rlabel.unsafe_store(p, hlabel)
        perm.unsafe_store(k, rlabel.unsafe_load(Int32(k)))

        var akk: Float64 = 1.0
        if symbolic == 0:
            akk = val.unsafe_load(pv)

        # 3. Row k is final. Its entries right of the diagonal are U; sort them
        #    by column so the back substitution walks ascending columns. Rows
        #    are emitted in pivot order, so a single running cursor is the CSR
        #    row pointer and no per-row slot has to be reserved.
        up.unsafe_store(k, ucnt)
        var nu: Int32 = 0
        var g = rhead.unsafe_load(k)
        while g != -1:
            if col.unsafe_load(g) > Int32(k):
                scr.unsafe_store(ccap + nu, g)
                nu += 1
            g = nextr.unsafe_load(g)
        var t = Int32(1)
        while t < nu:
            var key = scr.unsafe_load(ccap + t)
            var q = t - 1
            while q >= 0 and col.unsafe_load(scr.unsafe_load(ccap + q)) > col.unsafe_load(key):
                scr.unsafe_store(ccap + q + 1, scr.unsafe_load(ccap + q))
                q -= 1
            scr.unsafe_store(ccap + q + 1, key)
            t += 1
        t = 0
        while t < nu:
            var ge = scr.unsafe_load(ccap + t)
            ui.unsafe_store(ucnt, col.unsafe_load(ge))
            uv.unsafe_store(ucnt, val.unsafe_load(ge))
            ucnt += 1
            t += 1
        # the contiguous sorted run step 3 just wrote, which step 5 streams
        var ustar = Int(up.unsafe_load(k))
        var uend = Int(ucnt)
        ud.unsafe_store(k, val.unsafe_load(pv) if symbolic == 0 else 1.0)

        # 4. Scale the column below the pivot. Those entries are the multipliers of
        #    column k, so they are appended with the row label they belong to;
        #    the row-major form of L is built from them at the end. The pivot
        #    itself is not a multiplier and is skipped.
        t = 0
        while t < na:
            var ge = scr.unsafe_load(t)
            if ge != pv:
                if symbolic == 0:
                    val.unsafe_store(ge, val.unsafe_load(ge) / akk)
                llab.unsafe_store(lcnt, rlabel.unsafe_load(row.unsafe_load(ge)))
                lentry.unsafe_store(lcnt, ge)
                lcnt += 1
            t += 1

        # 5. Rank-one update of the trailing submatrix: for every row below the
        #    pivot that holds L[rr, k], subtract that multiplier times row k of
        #    U. Scatter-gather through `wpos`/`wtag` finds the target entry or
        #    creates the fill. The pivot is skipped again, so the finalized U
        #    row is never updated into itself.
        #
        #    The pivot row is streamed from the contiguous sorted run step 3
        #    already wrote, instead of walking the entry list again. Same
        #    entries, same order, but one sequential stream instead of a
        #    `nextr` pointer chase, and the column and the value arrive
        #    together, so the inner loop never touches the entry workspace.
        t = 0
        while t < na:
            var ge = scr.unsafe_load(t)
            t += 1
            if ge == pv:
                continue
            var rr = row.unsafe_load(ge)
            var lv_ = val.unsafe_load(ge)
            tag += 1
            var m = rhead.unsafe_load(rr)
            while m != -1:
                var c = col.unsafe_load(m)
                wpos.unsafe_store(c, m)
                wtag.unsafe_store(c, tag)
                m = nextr.unsafe_load(m)
            if symbolic == 0:
                # numeric: the arithmetic is fused into the same pass that
                # finds the target, and `symbolic` is hoisted out of the
                # inner loop because it cannot change within a call
                var fu = ustar
                while fu < uend:
                    var c = ui.unsafe_load(fu)
                    var target: Int32
                    if wtag.unsafe_load(c) == tag:
                        target = wpos.unsafe_load(c)
                    else:
                        if cnt >= ccap:
                            return -3
                        target = cnt
                        cnt += 1
                        val.unsafe_store(target, -lv_ * uv.unsafe_load(fu))
                        row.unsafe_store(target, rr)
                        col.unsafe_store(target, c)
                        nextr.unsafe_store(target, rhead.unsafe_load(rr))
                        rhead.unsafe_store(rr, target)
                        nextc.unsafe_store(target, chead.unsafe_load(c))
                        chead.unsafe_store(c, target)
                        wpos.unsafe_store(c, target)
                        wtag.unsafe_store(c, tag)
                        fu += 1
                        continue
                    val.unsafe_store(target, val.unsafe_load(target) - lv_ * uv.unsafe_load(fu))
                    fu += 1
            else:
                # structural only: the fill pattern is built, no values formed
                var fu = ustar
                while fu < uend:
                    var c = ui.unsafe_load(fu)
                    if wtag.unsafe_load(c) != tag:
                        if cnt >= ccap:
                            return -3
                        var target = cnt
                        cnt += 1
                        val.unsafe_store(target, 0.0)
                        row.unsafe_store(target, rr)
                        col.unsafe_store(target, c)
                        nextr.unsafe_store(target, rhead.unsafe_load(rr))
                        rhead.unsafe_store(rr, target)
                        nextc.unsafe_store(target, chead.unsafe_load(c))
                        chead.unsafe_store(c, target)
                        wpos.unsafe_store(c, target)
                        wtag.unsafe_store(c, tag)
                    fu += 1

    # Close off U. `u_ptr` was written per row as each row was finalized; the
    # running cursor is the total number of stored off-diagonal U entries.
    up.unsafe_store(n, ucnt)
    nnzf.unsafe_store(0, Int32(Int(lcnt) + Int(ucnt) + n))

    # Assemble the row-major L. `perm` says which position each original label
    # ended up at, so L is built in pivot order. The multipliers were appended
    # column by column, which is already ascending in the column index, so the
    # counting pass only has to bucket them by row; each bucket therefore comes
    # out with ascending column indices, which the triangular solve needs.
    # `ucu` is finished with, so it doubles as the row counter here.
    for r in range(n):
        ucu.unsafe_store(r, 0)
    for t in range(lcnt):
        var q = llab.unsafe_load(t)
        ucu.unsafe_store(q, ucu.unsafe_load(q) + 1)
    # exclusive scan over the counts, in the destination (pivot) row order
    var acc = 0
    for k in range(n):
        var q = perm.unsafe_load(k)
        lp.unsafe_store(k, Int32(acc))
        acc += Int(ucu.unsafe_load(q))
    lp.unsafe_store(n, Int32(acc))
    # `wpos` holds the running fill position of each destination row; the label
    # -> destination table goes in the low end of `scr`, which the structural
    # walk is finished with.
    for k in range(n):
        wpos.unsafe_store(k, lp.unsafe_load(k))
        scr.unsafe_store(perm.unsafe_load(k), Int32(k))
    var t2 = 0
    while t2 < lcnt:
        var ge = lentry.unsafe_load(t2)
        var dest = scr.unsafe_load(llab.unsafe_load(t2))
        var pos = wpos.unsafe_load(dest)
        li.unsafe_store(pos, col.unsafe_load(ge))
        lv.unsafe_store(pos, val.unsafe_load(ge))
        wpos.unsafe_store(dest, pos + 1)
        t2 += 1
    return 0


@export("mpp_lu_solve")
def mpp_lu_solve(
    n: Int,
    nrhs: Int,
    l_ptr: Int,
    l_idx: Int,
    l_val: Int,
    u_ptr: Int,
    u_idx: Int,
    u_val: Int,
    u_diag: Int,
    b: Int,
    x: Int,
    nworkers: Int,
) abi("C"):
    """`_call_pardiso` phases 13/33: the triangular solves against the factors.

    `b` is the already-permuted right-hand side `(P b)` and `x` receives the
    solution; both are `(n, nrhs)` column-major, matching the Fortran-contiguous
    layout `_check_b` hands to PARDISO. The forward substitution reads `b` into
    `x` and never writes back to `b`, so `b` is only read; the caller must still
    keep the two buffers distinct, because `x` is the one that accumulates.

    Each right-hand side is independent, so a multi-column solve is spread over
    the worker threads the way PARDISO's own threaded solve is.
    """
    var lp = iptr(l_ptr)
    var li = iptr(l_idx)
    var lv = fptr(l_val)
    var up = iptr(u_ptr)
    var ui = iptr(u_idx)
    var uv = fptr(u_val)
    var ud = fptr(u_diag)
    var bv = fptr(b)
    var xv = fptr(x)

    def solve_rhs(r: Int) {imm}:
        var off = r * n
        # L is unit lower triangular: y[j] = Pb[j] - sum(L[j, i] * y[i], i < j).
        #
        # Both `L` and `u_diag` are indexed by PIVOT position, which is what
        # `mpp_lu_factorize` wrote them in: row `k` of each is the row that was
        # pivoted to position `k`, and `perm[k]` is the original label of that
        # row. `perm` is a label lookup and is not an index into either array.
        # `rhead`/`up`/`ud` were all written under that same index, so reading
        # them by `perm[j]` instead mixes two different orderings up.
        for j in range(n):
            var s = bv.unsafe_load(off + j)
            for t in range(lp.unsafe_load(j), lp.unsafe_load(j + 1)):
                s -= lv.unsafe_load(t) * xv.unsafe_load(off + Int(li.unsafe_load(t)))
            xv.unsafe_store(off + j, s)
        # U has its diagonal split out, so it is divided back in here.
        for j in range(n - 1, -1, -1):
            var s = xv.unsafe_load(off + j)
            for t in range(up.unsafe_load(j), up.unsafe_load(j + 1)):
                s -= uv.unsafe_load(t) * xv.unsafe_load(off + Int(ui.unsafe_load(t)))
            xv.unsafe_store(off + j, s / ud.unsafe_load(j))

    # Each right-hand side is one work item, so asking for more workers than
    # there are right-hand sides is a bad decomposition and `parallelize` does
    # not survive it from a C host. Below the threshold the thread launch costs
    # more than the solve, so it stays serial.
    if nrhs >= 4 and nworkers > 1:
        var workers = nworkers
        if workers > nrhs:
            workers = nrhs
        parallelize(solve_rhs, nrhs, workers)
    else:
        for r in range(nrhs):
            solve_rhs(r)
