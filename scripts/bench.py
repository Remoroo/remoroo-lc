#!/usr/bin/env python3
"""Per-cell performance, single-instance and batched.

    python scripts/bench.py                 # every cell, CPU
    python scripts/bench.py --cuda          # add the CUDA sweep
    python scripts/bench.py --envs 4096

Budgets (soft, from the G0 brief):
  single instance, one CPU core : mean <= 1 ms, p99 <= 2 ms
  batched, 4096 environments    : full step <= 5 ms

Numbers that were not measured are printed as "not run".  There is no fallback
that estimates them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.kernels import available  # noqa: E402
from remoroo_lc.plant import Plant  # noqa: E402
from remoroo_lc.reference.controller import Controller  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402


def _time_reference(cell, ticks: int) -> dict:
    ctrl, plant = Controller(cell), Plant(cell)
    q = cell.rest_posture()
    plant.reset(q)
    ctrl.reset(q)
    ctrl.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
    samples = []
    for k in range(ticks):
        if k % 125 == 0:
            ctrl.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
        t0 = time.perf_counter()
        out = ctrl.step(q)
        samples.append((time.perf_counter() - t0) * 1e3)
        q, _ = plant.step(out.q_target)
    return _stats(samples)


def _time_kernels(cell, num_envs: int, device: str, ticks: int, stress: bool = False) -> dict:
    """Time the kernels, either from rest or over uniformly random postures.

    Driving a zero action chunk from `rest` is the BEST case and it flatters a
    cell with many collision pairs: cost scales with the ACTIVE set, and at rest
    almost nothing is within influence distance.  `stress=True` samples joint
    states uniformly inside the limits instead, which is the number to quote for
    a real cell -- it is what a teacher exploring against the filter produces.
    """
    from remoroo_lc.kernels.warp_backend import BatchedController
    import warp as wp

    ctrl = BatchedController(cell, num_envs=num_envs, device=device)
    q = np.tile(cell.rest_posture(), (num_envs, 1))
    ctrl.reset(q)
    ctrl.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
    # Warm up: the first launch compiles and allocates.
    for _ in range(20):
        ctrl.step(q)
    wp.synchronize_device(device)

    # Fixed seed: the stressed posture sequence is part of the measurement, so
    # two runs of this script compare like with like.
    rng = np.random.default_rng(20260730)
    lo, hi = (np.asarray(x, dtype=np.float32) for x in cell.joint_limits())
    zero_chunk = np.zeros((8, cell.action_dim), dtype=np.float32)

    samples = []
    for _ in range(ticks):
        if stress:
            q = (lo + (hi - lo) * rng.random((num_envs, cell.n_joints))).astype(np.float32)
            ctrl.reset(q)
            ctrl.set_chunk(zero_chunk, q)
            wp.synchronize_device(device)
        t0 = time.perf_counter()
        ctrl.step(q)
        wp.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    s = _stats(samples)
    s["per_env_us"] = s["mean_ms"] * 1e3 / num_envs
    s["stress"] = stress
    return s


def _stats(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": float(np.percentile(samples, 50)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": max(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", type=Path)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 64, 1024, 4096])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument(
        "--stress",
        action="store_true",
        help="drive uniformly random postures instead of rest; the honest number "
        "for a cell with many collision pairs, since cost scales with the active set",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "bench.json")
    a = ap.parse_args()

    cell_paths = a.cells or sorted((ROOT / "configs" / "cells").glob("*.yaml"))
    have_kernels = available()
    cuda_ok = False
    if have_kernels:
        from remoroo_lc.kernels.warp_backend import cuda_available

        cuda_ok = cuda_available()
    if a.cuda and not cuda_ok:
        print("CUDA requested but no device is available; the CUDA rows will say not run")

    out: dict = {"cells": {}, "cuda_available": cuda_ok}
    for path in cell_paths:
        cell = load_cell(path)
        ctrl = Controller(cell)
        row: dict = {
            "n_joints": cell.n_joints,
            "n_tcps": cell.n_tcps,
            "constraint_rows": ctrl.n_rows,
            "collision_pairs": len(ctrl.walls.pairs),
        }
        print(
            f"\n{cell.name}: n={cell.n_joints} T={cell.n_tcps} "
            f"rows={ctrl.n_rows} pairs={len(ctrl.walls.pairs)}"
        )

        row["reference_cpu"] = _time_reference(cell, a.ticks)
        r = row["reference_cpu"]
        print(
            f"  reference (NumPy, 1 env) mean {r['mean_ms']:.3f} ms  "
            f"p99 {r['p99_ms']:.3f} ms"
        )

        if not have_kernels:
            row["kernels"] = "not run: no backend installed"
            print("  kernels                  not run: no backend installed")
            out["cells"][cell.name] = row
            continue

        row["kernels_cpu"] = {}
        for n in a.envs:
            s = _time_kernels(
                cell, n, "cpu", a.ticks if n <= 64 else max(a.ticks // 8, 25), a.stress
            )
            row["kernels_cpu"][str(n)] = s
            print(
                f"  warp cpu  {n:>5} env  mean {s['mean_ms']:8.3f} ms  "
                f"p99 {s['p99_ms']:8.3f} ms  ({s['per_env_us']:7.2f} us/env)"
            )

        if cuda_ok:
            row["kernels_cuda"] = {}
            for n in a.envs:
                s = _time_kernels(cell, n, "cuda", a.ticks, a.stress)
                row["kernels_cuda"][str(n)] = s
                print(
                    f"  warp cuda {n:>5} env  mean {s['mean_ms']:8.3f} ms  "
                    f"p99 {s['p99_ms']:8.3f} ms  ({s['per_env_us']:7.2f} us/env)"
                )
        else:
            row["kernels_cuda"] = "not run: no CUDA device on this machine"
            print("  warp cuda                not run: no CUDA device on this machine")

        out["cells"][cell.name] = row

    dest = a.out if a.out.is_absolute() else (Path.cwd() / a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    shown = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
