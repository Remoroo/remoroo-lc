"""Shared fixtures.

Real customer cells live in `configs/rig/`, not here, and are deliberately NOT in
this matrix: they carry thousands of collision rows and belong to the rig scripts
and the benchmarks, not to a unit suite that has to stay fast.

The single most important thing in this file is that `cell` is parameterised over
the WHOLE cell matrix.  Any test that takes it runs five times: two 6-DOF chains,
two 7-DOF chains, a mixed 6+7 pair with different effector widths, a single
effectorless chain, and a branched tree whose two chains share a trunk joint.  A
test that only passes on one of them is a test that has caught an embodiment
assumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.plant import Plant  # noqa: E402
from remoroo_lc.reference.controller import Controller  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402

CELL_DIR = ROOT / "configs" / "cells"
# Skip macOS AppleDouble sidecars (`._name.yaml`), which survive a tar from a
# Dropbox-backed checkout and otherwise get parameterised as if they were cells.
CELL_FILES = sorted(p for p in CELL_DIR.glob("*.yaml") if not p.name.startswith("._"))
CELL_IDS = [p.stem for p in CELL_FILES]

assert CELL_FILES, f"no cell configs found under {CELL_DIR}"


@pytest.fixture(params=CELL_FILES, ids=CELL_IDS)
def cell_path(request) -> Path:
    return request.param


@pytest.fixture
def cell(cell_path):
    return load_cell(cell_path)


@pytest.fixture
def controller(cell):
    return Controller(cell)


@pytest.fixture
def plant(cell):
    return Plant(cell)


@pytest.fixture
def rng():
    return np.random.default_rng(20260729)


def random_states(cell, n: int, seed: int = 0, margin: float = 0.05) -> np.ndarray:
    """n joint configurations sampled uniformly inside the joint limits."""
    lo, hi = cell.joint_limits()
    g = np.random.default_rng(seed)
    span = hi - lo
    return (lo + margin * span + (1.0 - 2.0 * margin) * span * g.random((n, cell.n_joints))).astype(
        np.float32
    )


def zero_chunk(cell, k_steps: int = 8) -> np.ndarray:
    return np.zeros((k_steps, cell.action_dim), dtype=np.float32)


def run_closed_loop(cell, controller, plant, chunks, steps_per_chunk: int):
    """Drive controller + plant and collect per-tick records."""
    q = cell.rest_posture()
    plant.reset(q)
    controller.reset(q)
    records = []
    for chunk in chunks:
        controller.set_chunk(chunk, q)
        for _ in range(steps_per_chunk):
            out = controller.step(q)
            q, _ = plant.step(out.q_target)
            records.append(out)
    return records, q
