#!/usr/bin/env python3
"""Calibrate the per-TCP manipulability threshold w_thresh for a cell.

w_thresh sets where the damped least-squares damping starts to rise.  It has to
be per-chain because the measure w = sqrt(det(J J^T)) carries units of
length^3 * (dimensionless)^3 and therefore scales with the chain's own link
lengths -- a number tuned for one robot is meaningless on the next one.  So it is
calibrated, not guessed: sample joint configurations uniformly inside the
configured limits and take a low percentile of the resulting w.

Writes the values back into the cell file's `tcps[].damping.w_thresh` when
--write is given; otherwise just prints them.

Usage:
    python scripts/calibrate_wthresh.py configs/cells/*.yaml --write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoroo_lc.reference.kinematics import KinematicTree  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402


def calibrate(cell_path: Path, samples: int, percentile: float, seed: int) -> dict[str, float]:
    cell = load_cell(cell_path)
    tree = KinematicTree(cell)
    lo, hi = cell.joint_limits()
    rng = np.random.default_rng(seed)
    w = np.zeros((samples, cell.n_tcps), dtype=np.float64)
    for s in range(samples):
        q = lo + (hi - lo) * rng.random(cell.n_joints).astype(np.float32)
        w[s] = tree.manipulability(tree.tcp_jacobians(tree.fk(q)))
    return {
        t.name: float(np.percentile(w[:, i], percentile)) for i, t in enumerate(cell.tcps)
    }


def write_back(cell_path: Path, values: dict[str, float]) -> None:
    """Rewrite w_thresh in place, preserving comments and layout."""
    text = cell_path.read_text()
    order = list(values)
    idx = {"i": 0}

    def sub(match: re.Match) -> str:
        name = order[idx["i"]] if idx["i"] < len(order) else None
        idx["i"] += 1
        return match.group(0) if name is None else f"w_thresh: {values[name]:.6g}"

    new = re.sub(r"w_thresh:\s*[-+0-9.eE]+", sub, text)
    cell_path.write_text(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--percentile", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    for cell_path in a.cells:
        vals = calibrate(cell_path, a.samples, a.percentile, a.seed)
        pretty = ", ".join(f"{k}={v:.6g}" for k, v in vals.items())
        print(f"{cell_path.name}: {pretty}")
        if a.write:
            write_back(cell_path, vals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
