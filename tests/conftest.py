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
import yaml

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


def write_cell(tmp_dir, base_path: Path, mutate) -> Path:
    """Copy a shipped cell into `tmp_dir`, apply `mutate` to its raw dict, write it.

    Every relative path inside the cell (urdf, spheres, limits, gains) is
    absolutised against the ORIGINAL file's directory first, because the copy
    lands in a tmp directory where `../arm.urdf` means nothing.  This is the same
    trick `test_schema._write` plays; it lives here because the environment tests
    need it too and two copies of a path-rewriting rule is one copy too many.
    """
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    mutate(raw)
    for m in raw.get("models", []):
        for key in ("urdf", "spheres"):
            if key in m and not str(m[key]).startswith("/"):
                m[key] = str((base_path.parent / m[key]).resolve())
    for key in ("limits", "gains"):
        if isinstance(raw.get(key), str):
            raw[key] = str((base_path.parent / raw[key]).resolve())
    out = Path(tmp_dir) / "cell.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


#: One environment entry of EVERY primitive type the loader accepts.  No shipped
#: cell declares a sphere or a cylinder -- `configs/cells/*.yaml` between them use
#: only `plane` and `box` -- so without this literal two of the four encodings in
#: `structure._ENV_CODE` and two of the four branches in `walls.env_distance` are
#: never exercised by a cell-level test.  The numbers are arbitrary but FIXED:
#: tests/test_env_mesh.py pins the structure they flatten to, so changing one of
#: them here is a deliberate act that will fail that test.
PRIMITIVE_ENVIRONMENT = [
    {"name": "table", "type": "plane", "point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
    {
        "name": "back_wall",
        "type": "box",
        "xyz": [-0.50, 0.0, 0.75],
        "rpy": [0.0, 0.0, 0.0],
        "dims": [0.06, 2.00, 1.50],
    },
    {"name": "bulb", "type": "sphere", "xyz": [0.20, 0.10, 0.40], "radius": 0.05},
    {
        "name": "post",
        "type": "cylinder",
        "xyz": [0.0, 0.30, 0.50],
        "rpy": [0.1, -0.2, 0.3],
        "radius": 0.05,
        "height": 1.00,
    },
]
