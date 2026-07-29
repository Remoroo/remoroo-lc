"""The cell contract: every shipped config validates, and sizing is derived."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from remoroo_lc.constants import TASK_DIM
from remoroo_lc.schema import CELL_SPEC_VERSION, CellSpecError, load_cell, validate_cell


def test_every_shipped_cell_validates(cell):
    validate_cell(cell)  # must not raise
    assert cell.spec_version.split(".")[0] == CELL_SPEC_VERSION.split(".")[0]


def test_sizing_is_derived_not_declared(cell):
    n_from_models = sum(len(m.joint_names) for m in cell.models)
    assert cell.n_joints == n_from_models
    assert cell.action_dim == sum(TASK_DIM + g for g in cell.effector_widths)
    assert cell.task_dim == TASK_DIM * cell.n_tcps
    assert len(cell.joint_labels()) == cell.n_joints


def test_action_slices_tile_the_action_vector(cell):
    slices = cell.action_slices()
    assert len(slices) == cell.n_tcps
    covered = np.zeros(cell.action_dim, dtype=int)
    for pose, eff in slices:
        covered[pose] += 1
        covered[eff] += 1
    assert np.all(covered == 1), "action slices must tile the action vector exactly"


def test_joint_limits_and_rest_are_consistent(cell):
    lo, hi = cell.joint_limits()
    assert lo.shape == hi.shape == (cell.n_joints,)
    assert np.all(hi > lo)
    q_rest = cell.rest_posture()
    assert np.all(q_rest >= lo) and np.all(q_rest <= hi)
    assert np.all(cell.joint_velocity_limits() > 0.0)


def test_effectorless_tcp_is_supported():
    """g = 0 must not be a special case anywhere."""
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "single_6dof_leg.yaml")
    assert cell.effector_widths == (0,)
    assert cell.action_dim == TASK_DIM
    pose, eff = cell.action_slices()[0]
    assert pose.stop - pose.start == TASK_DIM
    assert eff.stop - eff.start == 0


def test_effector_width_is_config_not_code():
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "mixed.yaml")
    assert cell.effector_widths == (1, 2)
    assert cell.action_dim == (TASK_DIM + 1) + (TASK_DIM + 2)


def test_shared_joints_appear_once_in_the_global_vector():
    """The branched cell has one trunk joint, not one per chain."""
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "branched_trunk.yaml")
    labels = cell.joint_labels()
    assert len(labels) == len(set(labels))
    assert sum("trunk" in lb for lb in labels) == 1


# --------------------------------------------------------------------------- #
# validator rejects malformed contracts
# --------------------------------------------------------------------------- #


def _write(tmp_path, base_path, mutate):
    raw = yaml.safe_load(base_path.read_text())
    mutate(raw)
    out = tmp_path / "cell.yaml"
    # Resolve relative paths against the original file's directory.
    for m in raw.get("models", []):
        for key in ("urdf", "spheres"):
            if key in m and not str(m[key]).startswith("/"):
                m[key] = str((base_path.parent / m[key]).resolve())
    for key in ("limits", "gains"):
        if isinstance(raw.get(key), str):
            raw[key] = str((base_path.parent / raw[key]).resolve())
    out.write_text(yaml.safe_dump(raw))
    return out


def test_unsupported_version_is_rejected(tmp_path, cell_path):
    bad = _write(tmp_path, cell_path, lambda r: r.update(lc_spec_version="99.0"))
    with pytest.raises(CellSpecError, match="not supported"):
        load_cell(bad)


def test_unknown_tcp_frame_is_rejected(tmp_path, cell_path):
    def mutate(r):
        r["tcps"][0]["frame"] = "no_such_link"

    bad = _write(tmp_path, cell_path, mutate)
    with pytest.raises(CellSpecError, match="not a link"):
        load_cell(bad)


def test_unaccounted_joint_is_rejected(tmp_path, cell_path):
    def mutate(r):
        r["models"][0]["joints"] = []

    bad = _write(tmp_path, cell_path, mutate)
    with pytest.raises(CellSpecError):
        load_cell(bad)


def test_rest_outside_limits_is_rejected(tmp_path, cell_path):
    def mutate(r):
        n = len(r["models"][0].get("joints") or []) or None
        cur = r["models"][0].get("rest")
        if cur is None:
            pytest.skip("cell has no explicit rest posture")
        r["models"][0]["rest"] = [1000.0] * len(cur)
        assert n is None or n == len(cur)

    bad = _write(tmp_path, cell_path, mutate)
    with pytest.raises(CellSpecError, match="rest posture outside joint limits"):
        load_cell(bad)
