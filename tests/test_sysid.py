"""The sysid fit, validated against the plant it is meant to identify.

The gains that end up in a cell's gains.yaml come from this fitter, and every
number the scoreboard reports is downstream of them.  So the fitter is checked
the only way that means anything: excite a plant whose gains are known, and
confirm it recovers them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from remoroo_lc.adapters.mock import MockCellAdapter
from remoroo_lc.plant import Plant
from remoroo_lc.reference.kinematics import KIND_PRISMATIC, KIND_REVOLUTE, KinematicTree

ROOT = Path(__file__).resolve().parents[1]


def _sysid():
    spec = importlib.util.spec_from_file_location(
        "sysid_tapes", ROOT / "scripts" / "sysid_tapes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sysid_tapes"] = mod
    spec.loader.exec_module(mod)
    return mod


def _inertia(cell):
    tree = KinematicTree(cell)
    by_kind = cell.gains["by_kind"]
    key = {KIND_REVOLUTE: "revolute", KIND_PRISMATIC: "prismatic"}
    return np.asarray(
        [by_kind[key[int(k)]]["inertia"] for k in tree.joint_kind], dtype=float
    )


def test_step_and_chirp_tapes_stay_inside_the_limits(cell):
    m = _sysid()
    lo, hi = cell.joint_limits()
    for j in range(cell.n_joints):
        for tape in (m.step_tape(cell, j), m.chirp_tape(cell, j)):
            assert np.all(tape >= lo - 1e-6) and np.all(tape <= hi + 1e-6)
            # Only the joint under test may move.
            moving = np.where(np.ptp(tape, axis=0) > 1e-9)[0]
            assert moving.tolist() == [j]


def test_fit_recovers_the_plant_it_was_given(cell):
    """kp within 5%, delay exact -- the brief's bar, on every cell."""
    m = _sysid()
    adapter = MockCellAdapter(cell)
    adapter.connect()
    fits = m.identify(cell, adapter, _inertia(cell), verbose=False)
    adapter.disconnect()

    plant = Plant(cell)
    labels = cell.joint_labels()
    for j, name in enumerate(labels):
        got = fits[name]
        kp_true = float(plant.kp[j])
        kd_true = float(plant.kd[j])
        assert abs(got["kp"] - kp_true) <= 0.05 * kp_true, (
            f"{name}: kp {got['kp']:.1f} vs {kp_true:.1f}"
        )
        assert got["delay_ticks"] == plant.delay_ticks, (
            f"{name}: delay {got['delay_ticks']} vs {plant.delay_ticks}"
        )
        # kd is the harder one -- it only shows up in the shape of the approach --
        # so it gets a looser bar than kp, stated rather than hidden.
        assert abs(got["kd"] - kd_true) <= 0.15 * kd_true, (
            f"{name}: kd {got['kd']:.2f} vs {kd_true:.2f}"
        )


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_fit_recovers_a_delay_it_was_not_expecting(delay):
    """The delay is a vendor property; the fitter must find it, not assume it."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    m = _sysid()
    cell = load_cell(CELL_DIR / "single_6dof_leg.yaml")
    cell.gains["command_delay_ticks"] = delay
    adapter = MockCellAdapter(cell)
    adapter.connect()
    fits = m.identify(cell, adapter, _inertia(cell), verbose=False)
    adapter.disconnect()
    for name, got in fits.items():
        assert got["delay_ticks"] == delay, f"{name}: {got['delay_ticks']} vs {delay}"


def test_fit_recovers_modified_gains():
    """Change the plant, and the fit must follow it rather than the config file."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    m = _sysid()
    cell = load_cell(CELL_DIR / "single_6dof_leg.yaml")
    labels = cell.joint_labels()
    cell.gains["per_joint"] = {
        labels[0]: {"inertia": 0.5, "kp": 20000.0, "kd": 200.0},
        labels[2]: {"inertia": 0.5, "kp": 60000.0, "kd": 350.0},
    }
    adapter = MockCellAdapter(cell)
    adapter.connect()
    fits = m.identify(cell, adapter, _inertia(cell), verbose=False)
    adapter.disconnect()
    assert abs(fits[labels[0]]["kp"] - 20000.0) <= 0.05 * 20000.0
    assert abs(fits[labels[2]]["kp"] - 60000.0) <= 0.05 * 60000.0
    assert abs(fits[labels[1]]["kp"] - 45000.0) <= 0.05 * 45000.0
