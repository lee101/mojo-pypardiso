"""Micro-benchmark on the *real* access shape of the factorisation update.

The first version of this probe used 1M random indices over a 40 MB array,
which is the worst case for an indirect store and is not what
`mpp_lu_factorize` does. The shapes that matter here are the ones the dense
and Laplacian benchmark matrices produce, so they are measured directly:

    dense 800        ncols = 800,   row length ~ 400,   640000 entries
    laplacian 80x80  ncols = 6400,  row length ~ 80,   1011358 entries

and for each, the two things that differ between the current update loop and a
sorted-merge update:

    stamped probe    the shape in `mpp_lu_factorize` today: stamp the row into
                     `wpos`/`wtag`, then probe per (row, pivot column) pair
    ordered merge    walk both sides by column, no probe, no tag

Run with `pixi run python bench/probe_scatter.py`.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import tempfile
import time

import numpy as np

SRC = r'''
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()

# The shape in mpp_lu_factorize today: for one multiplier, stamp the target
# row into wpos/wtag, then walk the pivot row probing wtag per entry.
@export("stamped_probe")
def stamped_probe(
    wpos: Int, wtag: Int, rhead: Int, rnext: Int, rcol: Int, rval: Int,
    khead: Int, pos_of: Int, lv: Float64, tag: Int,
) abi("C"):
    var pp = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=wpos)
    var tp = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=wtag)
    var rh = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=rhead)
    var rn = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=rnext)
    var rc = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=rcol)
    var vv = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=rval)
    var kh = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=khead)
    var m = rh.unsafe_load(pos_of)
    while m != Int32(-1):
        var c = rc.unsafe_load(m)
        pp.unsafe_store(c, m)
        tp.unsafe_store(c, Int32(tag))
        m = rn.unsafe_load(m)
    var f = kh.unsafe_load(0)
    while f != Int32(-1):
        var c = rc.unsafe_load(f)
        if tp.unsafe_load(c) == Int32(tag):
            var target = pp.unsafe_load(c)
            vv.unsafe_store(target, vv.unsafe_load(target) - lv * vv.unsafe_load(f))
        f = rn.unsafe_load(f)

# The same arithmetic with both sides sorted and walked together: no tag, no
# probe, no branch on whether the column is present.
@export("ordered_merge")
def ordered_merge(
    wpos: Int, wtag: Int, rhead: Int, rnext: Int, rcol: Int, rval: Int,
    khead: Int, pos_of: Int, lv: Float64, tag: Int,
) abi("C"):
    var rn = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=rnext)
    var rc = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=rcol)
    var vv = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=rval)
    var pp = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=wpos)
    _ = wtag
    _ = rhead
    var kh = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=khead)
    var a = pp.unsafe_load(0)
    var b = kh.unsafe_load(0)
    var ca = rc.unsafe_load(a)
    var cb = rc.unsafe_load(b)
    while a != Int32(-1) and b != Int32(-1):
        if ca == cb:
            vv.unsafe_store(a, vv.unsafe_load(a) - lv * vv.unsafe_load(b))
            a = rn.unsafe_load(a)
            if a != Int32(-1):
                ca = rc.unsafe_load(a)
            b = rn.unsafe_load(b)
            if b != Int32(-1):
                cb = rc.unsafe_load(b)
        elif ca < cb:
            a = rn.unsafe_load(a)
            if a != Int32(-1):
                ca = rc.unsafe_load(a)
        else:
            b = rn.unsafe_load(b)
            if b != Int32(-1):
                cb = rc.unsafe_load(b)
    _ = pos_of
    _ = tag
'''

def build(tmpdir):
    path = os.path.join(tmpdir, "probe.mojo")
    with open(path, "w") as fh:
        fh.write(SRC)
    lib = os.path.join(tmpdir, "probe.so")
    found = subprocess.run(["which", "mojo"], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ)
    if found:
        env["MODULAR_HOME"] = os.path.join(
            os.path.dirname(os.path.dirname(found)), "share", "max")
    proc = subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", path, "-o", lib],
        capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print((proc.stderr or proc.stdout)[:8000])
        raise SystemExit("build failed")
    return ctypes.CDLL(lib)


def linked_rows(rng, nrows, ncols, rowlen, entries):
    """Build one linked-list row per entry slot, the shape step 5 walks."""
    col = np.empty(entries, dtype=np.int32)
    val = rng.standard_normal(entries)
    nxt = np.empty(entries, dtype=np.int32)
    nxt[:] = -1
    rhead = np.empty(nrows, dtype=np.int32)
    # each entry belongs to a row; entries of a row are chained
    row_of = np.empty(entries, dtype=np.int32)
    per = entries // nrows
    for r in range(nrows):
        lo = r * per
        hi = lo + per
        cols = rng.choice(ncols, size=hi - lo, replace=False)
        cols.sort()
        col[lo:hi] = cols
        row_of[lo:hi] = r
        for t in range(hi - lo - 1):
            nxt[lo + t] = lo + t + 1
        rhead[r] = lo
    return col, val, nxt, rhead, row_of


def main():
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(tmp)
        V, F, I, I32 = ctypes.c_void_p, ctypes.c_double, ctypes.c_int64, ctypes.c_int32
        for name in ("stamped_probe", "ordered_merge"):
            fn = getattr(lib, name)
            fn.argtypes = [V] * 7 + [I, F, I]
            fn.restype = None

        rng = np.random.default_rng(0)
        for label, ncols, nrows, entries in (
            ("dense 800", 800, 799, 799 * 400),
            ("laplacian 80x80", 6400, 6399, 6399 * 80),
        ):
            col, val, nxt, rhead, row_of = linked_rows(rng, nrows, ncols, entries // nrows, entries)
            wpos = np.zeros(ncols, dtype=np.int32)
            wtag = np.zeros(ncols, dtype=np.int32)
            khead = np.zeros(1, dtype=np.int32)
            lv = 0.75

            def call(fn, row, tag):
                fn(V(wpos.ctypes.data), V(wtag.ctypes.data), V(rhead.ctypes.data),
                   V(nxt.ctypes.data), V(col.ctypes.data), V(val.ctypes.data),
                   V(khead.ctypes.data), row, lv, tag)

            def best(fn, repeat=5):
                out = math.inf
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    for row in range(nrows):
                        call(fn, row, row + 1)
                    out = min(out, time.perf_counter() - t0)
                return out

            per = entries // nrows
            t_stamp = best(lib.stamped_probe)
            t_merge = best(lib.ordered_merge)
            print(f"{label:16s} ncols={ncols:5d} entries/row={per:5d}")
            print(f"  stamped probe  {t_stamp*1e3:8.2f} ms  {t_stamp/entries*1e9:6.2f} ns/entry")
            print(f"  ordered merge  {t_merge*1e3:8.2f} ms  {t_merge/entries*1e9:6.2f} ns/entry")
            print(f"  ratio          {t_stamp/t_merge:8.2f}x")
            print()


if __name__ == "__main__":
    main()