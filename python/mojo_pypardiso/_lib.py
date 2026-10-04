"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_PYPARDISO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pypardiso.so"
)

I = ctypes.c_int32
A = ctypes.c_void_p
I64 = ctypes.c_int64

# Every `@export`ed kernel, in the order `src/ported.mojo` declares them. The
# widths are checked against the kernel sources by
# `tests/test_kernels.py::test_ctypes_argtypes_match_the_declared_kernel_signatures`,
# so a kernel that gains a parameter without a matching table entry fails there
# rather than silently truncating the call.
_SIGNATURES = {
    "mpp_indptr_diffs": ([A, A, I64], I),
    # `n` and `nnz` are counts; `cursor` is a buffer address, and the kernel
    # declares it `Int` like every other address because `@export` has no
    # pointer type to hand it
    "mpp_csr_transpose": ([A] * 6 + [I64, I64, A], I),
    "mpp_permute_rhs": ([I64, I64] + [A] * 3, None),
    "mpp_lu_factorize": ([I64] * 3 + [A] * 26, I),
    "mpp_lu_solve": ([I64] * 2 + [A] * 9 + [I64], None),
}


class BuildError(RuntimeError):
    pass


# A page-touched scratch buffer. `_is_already_factorized` walks a matrix's index
# and data arrays before comparing them, so that the comparison which follows is
# reading the same bytes the kernel will; see that method for why the walk has to
# happen at all. Sized as a slice, so only the prefix covering the matrix is ever
# touched.
WARM = np.zeros(1 << 22, dtype=np.uint8)


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYPARDISO_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_PYPARDISO_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/ported.mojo` into `dist/libmojo-pypardiso.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", os.path.join(SRC, "ported.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(a) -> int:
    return a.ctypes.data


def nworkers() -> int:
    """Worker count handed to the kernels that spread work over threads."""
    env = os.environ.get("MOJO_PYPARDISO_THREADS")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def main() -> int:
    """`python -m mojo_pypardiso._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
