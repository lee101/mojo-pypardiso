"""mojo-pypardiso against upstream pypardiso (MKL PARDISO) and scipy.

    pixi run bench

Prints a markdown table. Upstream MKL PARDISO is a highly tuned multithreaded
commercial library, so it wins several rows outright; those rows are reported
as they came out.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_pypardiso as mp  # noqa: E402
import pypardiso  # noqa: E402


def timeit(fn, repeat: int = 3):
    """Best of `repeat` wall-clock seconds; returns None if the call fails."""
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        best = min(best, time.perf_counter() - t0)
    return best


def dense_spd(n, seed=0):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix(np.eye(n) + 0.2 * rng.standard_normal((n, n)))


def banded(n):
    main = 4.0 + np.random.default_rng(0).uniform(0, 1, n)
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def laplacian_2d(nx, ny):
    n = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            rows.append(k); cols.append(k); vals.append(4.0)
            if i > 0: rows.append(k); cols.append(k - ny); vals.append(-1.0)
            if i < nx - 1: rows.append(k); cols.append(k + ny); vals.append(-1.0)
            if j > 0: rows.append(k); cols.append(k - 1); vals.append(-1.0)
            if j < ny - 1: rows.append(k); cols.append(k + 1); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sort_indices()
    return A


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("solve: dense 200x200 (1 rhs)")
def _():
    A = dense_spd(200, 0)
    b = np.random.default_rng(1).standard_normal(200)
    return (lambda: mp.PyPardisoSolver().solve(A, b),
            lambda: pypardiso.PyPardisoSolver().solve(A, b))


@case("solve: dense 400x400 (1 rhs)")
def _():
    A = dense_spd(400, 1)
    b = np.random.default_rng(2).standard_normal(400)
    return (lambda: mp.PyPardisoSolver().solve(A, b),
            lambda: pypardiso.PyPardisoSolver().solve(A, b))


@case("solve: dense 800x800 (1 rhs)")
def _():
    A = dense_spd(800, 2)
    b = np.random.default_rng(3).standard_normal(800)
    return (lambda: mp.PyPardisoSolver().solve(A, b),
            lambda: pypardiso.PyPardisoSolver().solve(A, b))


@case("factorize: dense 800x800")
def _():
    A = dense_spd(800, 2)
    b = np.zeros((800, 1))
    def ours():
        s = mp.PyPardisoSolver()
        s.factorize(A)
    def theirs():
        s = pypardiso.PyPardisoSolver()
        s.factorize(A)
    return (ours, theirs)


@case("solve: dense 500x500, 32 rhs")
def _():
    A = dense_spd(500, 3)
    B = np.random.default_rng(4).standard_normal((500, 32))
    return (lambda: mp.PyPardisoSolver().solve(A, B),
            lambda: pypardiso.PyPardisoSolver().solve(A, B))


@case("solve: banded 20000 (1 rhs)")
def _():
    A = banded(20000)
    b = np.random.default_rng(5).standard_normal(20000)
    return (lambda: mp.PyPardisoSolver().solve(A, b),
            lambda: pypardiso.PyPardisoSolver().solve(A, b))


@case("solve: laplacian 80x80 grid (6400)")
def _():
    A = laplacian_2d(80, 80)
    b = np.random.default_rng(6).standard_normal(A.shape[0])
    return (lambda: mp.PyPardisoSolver().solve(A, b),
            lambda: pypardiso.PyPardisoSolver().solve(A, b))


@case("spsolve alias: laplacian 60x60 (3600)")
def _():
    A = laplacian_2d(60, 60)
    b = np.random.default_rng(9).standard_normal(A.shape[0])
    return (lambda: mp.spsolve(A, b), lambda: pypardiso.spsolve(A, b))


@case("solve cached: dense 500x500, 8 rhs")
def _():
    A = dense_spd(500, 3)
    B = np.random.default_rng(4).standard_normal((500, 8))
    solver = mp.PyPardisoSolver()
    solver.factorize(A)                 # the documented pattern: factorise once
    def ours():
        solver.solve(A, B)              # then solve repeatedly against it
    def theirs():
        solver.solve(A, B)
    return (ours, theirs)


@case("phase 22 analysis: dense 800x800")
def _():
    A = dense_spd(800, 2)
    b = np.zeros(800)
    def ours():
        s = mp.PyPardisoSolver()
        s.set_phase(22)
        s._call_pardiso(A, b)
    def theirs():
        s = pypardiso.PyPardisoSolver()
        s.set_phase(22)
        s._call_pardiso(A, b)
    return (ours, theirs)


def machine():
    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"{cpu or platform.processor()}, {os.cpu_count()} logical CPUs, "
            f"Python {platform.python_version()}, numpy {np.__version__}")


# Each measurement runs in its own process. MKL PARDISO segfaults on some of
# these paths (phase 22 on a dense matrix does, reproducibly, in this MKL
# build), and a crash in one cell must not take the whole table with it.
CHILD = """
import json, sys
sys.path.insert(0, "bench")
sys.path.insert(0, "python")
import numpy as np
import bench as B

index, which = int(sys.argv[1]), sys.argv[2]
name, build = B.CASES[index]
ours, theirs = build()
fn = ours if which == "ours" else theirs
seconds = B.timeit(fn)
print("@@RESULT@@" + json.dumps(seconds))
"""


def measure(index, which):
    """Best-of-N seconds for one cell, or None if it failed or crashed."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, str(index), which],
            cwd=root, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            value = json.loads(line[len("@@RESULT@@"):])
            return value
    return None


def main():
    print(f"machine: {machine()}")
    print()
    print("| case | mojo-pypardiso | upstream pypardiso (MKL) | ratio |")
    print("| --- | --- | --- | --- |")
    rows = []
    for index, (name, _) in enumerate(CASES):
        t_ours = measure(index, "ours")
        t_theirs = measure(index, "theirs")

        def fmt(t):
            return "-" if t is None else f"{t*1e3:.2f} ms"

        if t_ours is None or t_theirs is None:
            if t_ours is None:
                ours_cell = "failed"
            else:
                ours_cell = fmt(t_ours)
            theirs_cell = "crashed" if t_theirs is None else fmt(t_theirs)
            print(f"| {name} | {ours_cell} | {theirs_cell} | - |")
            rows.append((name, t_ours, t_theirs, None))
            continue
        ratio = t_ours / t_theirs
        rows.append((name, t_ours, t_theirs, ratio))
        print(f"| {name} | {fmt(t_ours)} | {fmt(t_theirs)} | {ratio:.2f}x |")

    done = [r for r in rows if r[3] is not None]
    if done:
        faster = [r for r in done if r[3] < 1.0]
        ratios = sorted(r[3] for r in done)
        print()
        print(f"{len(faster)} of {len(done)} comparable cases faster than "
              f"upstream MKL PARDISO; median ratio "
              f"{ratios[len(ratios)//2]:.2f}x, best {ratios[0]:.2f}x, "
              f"worst {ratios[-1]:.2f}x.")
        for name, _, _, ratio in sorted(done, key=lambda r: r[3]):
            print(f"  {ratio:6.2f}x  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
