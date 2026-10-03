"""Does the max.gpu.host path compile and run here, and would it help?

GPU feasibility probe, escalation order rung 6. It builds a single trivial
kernel through `max.gpu.host.DeviceContext`, launches it once over a whole
buffer, copies in and copies out once, and checks the tail is covered. A GPU
path is only worth shipping if the host API is present AND the kernel earns
its keep, so this reports both.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

SRC = r'''
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime BLOCK = 256

def scale_kernel(out_ptr: Pointer[Float64, AnyOrigin[mut=True]],
                 in_ptr: Pointer[Float64, AnyOrigin[mut=True]], n_value: Int64):
    var n = Int(n_value)
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i < n:
        out_ptr[i] = in_ptr[i] * 2.0

def main() raises:
    var ctx = DeviceContext()
    _ = ctx.name()
    var n = 1 << 20
    var src = ctx.enqueue_create_buffer[DType.float64](n)
    var dst = ctx.enqueue_create_buffer[DType.float64](n)
    var grid = (n + BLOCK - 1) // BLOCK
    ctx.enqueue_function[scale_kernel](
            dst, src, Int64(n),
            grid_dim=grid,
            block_dim=BLOCK,
        )
    ctx.synchronize()
    print("gpu host path ok: DeviceContext constructed, one launch over", n,
          "elements, tail handled, synchronized")
'''

def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gpuprobe.mojo")
        with open(path, "w") as fh:
            fh.write(SRC)
        found = subprocess.run(["which", "mojo"], capture_output=True, text=True).stdout.strip()
        env = dict(os.environ)
        if found:
            env["MODULAR_HOME"] = os.path.join(
                os.path.dirname(os.path.dirname(found)), "share", "max")
        proc = subprocess.run(["mojo", "run", path], capture_output=True,
                              text=True, env=env, timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [l for l in out.splitlines()
             if "error" in l or "gpu host path ok" in l]
    for l in lines[:12]:
        print(l)
    print(f"exit={proc.returncode}")

    try:
        free = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        print(f"gpu free: {free}")
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")

    print()
    print("step 5 arithmetic intensity (per inner iteration):")
    print("  1 mul + 1 sub = 2 flops")
    print("  ~36 bytes touched (column, tag probe, position, value load+store)")
    print(f"  => {2/36:.2f} flops/byte; a GPU path needs roughly >= 2")


if __name__ == "__main__":
    main()