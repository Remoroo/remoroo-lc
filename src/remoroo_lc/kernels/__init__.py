"""Batched kernel implementation of the same three layers.

The NumPy code in `remoroo_lc.reference` is the specification; this package is
the runtime.  It compiles one set of kernels to both CPU and CUDA, runs one
thread per environment, and is what both Isaac Lab (thousands of environments)
and the cell's edge box (one environment at command rate) actually execute.

One thread per environment is the load-bearing choice.  Layer 3 is a projected
Gauss-Seidel sweep in which each row update reads the q_dot the previous row just
wrote; expressed as batched tensor ops that is 32 * m tiny sequential kernel
launches, which is launch-bound at any batch size.  Inside a thread it is a plain
loop.  It also means there is no cross-thread reduction anywhere in the pipeline,
so there are no atomics, no run-to-run reordering, and bitwise determinism is
structural rather than something to be requested.

Use `available()` to check whether a backend is importable before constructing
anything.
"""

from __future__ import annotations


def available() -> bool:
    """True if a kernel backend can be imported."""
    try:
        import warp  # noqa: F401
    except ImportError:
        return False
    return True


def backend_name() -> str:
    return "warp" if available() else "none"


def __getattr__(name: str):
    if name in ("BatchedController", "cuda_available"):
        from remoroo_lc.kernels import warp_backend

        return getattr(warp_backend, name)
    if name == "CellStructure":
        from remoroo_lc.kernels.structure import CellStructure

        return CellStructure
    raise AttributeError(name)
