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
import yaml

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


# --------------------------------------------------------------------------- #
# main() -- the session an engine phase runs unattended over ssh
#
# Everything above fits a plant in-process and never enters main(), which is
# where an unattended rig session actually breaks: the input-gains override, the
# evidence the writer keeps, the --out directory, and the cleanup after a failure
# in the window between connect() and identify().  None of the four was covered
# when all four were defects.
# --------------------------------------------------------------------------- #


def _gains_file(path: Path, **revolute) -> Path:
    """configs/gains.default.yaml with its revolute row amended, written to `path`.

    Derived from the shipped file rather than retyped so that a gains file that
    grows a key does not leave these tests fitting a shape nothing else uses.
    """
    doc = yaml.safe_load((ROOT / "configs" / "gains.default.yaml").read_text(encoding="utf-8"))
    doc["by_kind"]["revolute"].update(revolute)
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _rig_shaped_cell(tmp_path: Path, gains: str) -> Path:
    """A real cell copied outside configs/, with `gains:` pointed at `gains`.

    Point it at a path that does not exist and you have the shape EVERY rig cell
    has before its first sysid session: `gains:` must name the measured file the
    control stack reads afterwards (cells/<id>/lc/gains.yaml), and that file is
    the thing the session is about to create.

    The relative references are absolutised because load_cell resolves a cell
    file's paths against that file's own directory, and this copy lives in
    tmp_path -- writing the copy into configs/cells/ instead would be picked up
    by conftest's CELL_FILES glob and parameterise the whole suite over it.
    """
    from tests.conftest import CELL_DIR

    raw = yaml.safe_load((CELL_DIR / "single_6dof_leg.yaml").read_text(encoding="utf-8"))
    raw["limits"] = str((CELL_DIR / raw["limits"]).resolve())
    for model in raw["models"]:
        for key in ("urdf", "spheres"):
            if key in model:
                model[key] = str((CELL_DIR / model[key]).resolve())
    raw["gains"] = gains
    out = tmp_path / "cell.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def test_gains_override_runs_a_cell_whose_own_gains_file_does_not_exist_yet(
    tmp_path, monkeypatch
):
    """--gains breaks the chicken-and-egg that used to block the first session."""
    m = _sysid()
    missing = tmp_path / "lc" / "gains.yaml"
    cell_path = _rig_shaped_cell(tmp_path, str(missing))
    assert not missing.exists()

    # The premise, asserted instead of assumed: load_cell opens `gains:` eagerly,
    # so without the override the cell this session is FOR cannot even be loaded.
    monkeypatch.setattr(
        sys, "argv", ["sysid_tapes.py", str(cell_path), "--joints", "joint1"]
    )
    with pytest.raises(FileNotFoundError):
        m.main()

    out = tmp_path / "gains_raw.yaml"
    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(cell_path), "--joints", "joint1",
         "--gains", str(_gains_file(tmp_path / "prior.yaml")), "--out", str(out)],
    )
    assert m.main() == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert list(doc["per_joint"]) == ["limb/joint1"], doc["per_joint"]
    assert not missing.exists(), "the override still opened the cell's own target"


def test_gains_override_beats_the_gains_the_cell_declares(tmp_path, monkeypatch):
    """The override is an override: the cell's own file loses, and the inertia the
    fit was HANDED is the inertia that reaches the file (sysid identifies kp/M and
    kd/M, never M -- wrap_gains.py gauges that by comparing the two)."""
    m = _sysid()
    declared = _gains_file(tmp_path / "declared.yaml")
    override = _gains_file(tmp_path / "override.yaml", inertia=0.7)
    assert yaml.safe_load(declared.read_text(encoding="utf-8"))[
        "by_kind"]["revolute"]["inertia"] == 0.5
    cell_path = _rig_shaped_cell(tmp_path, str(declared))

    out = tmp_path / "gains_raw.yaml"
    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(cell_path), "--joints", "joint1",
         "--gains", str(override), "--out", str(out)],
    )
    assert m.main() == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["by_kind"]["revolute"]["inertia"] == 0.7, doc["by_kind"]
    assert doc["per_joint"]["limb/joint1"]["inertia"] == 0.7, doc["per_joint"]


def test_the_written_file_carries_the_fits_own_residual_and_delay(tmp_path, monkeypatch):
    """The writer must carry the measurement, not a stand-in for it."""
    from tests.conftest import CELL_DIR

    m = _sysid()
    out = tmp_path / "gains_raw.yaml"
    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(CELL_DIR / "single_6dof_leg.yaml"),
         "--joints", "joint1", "--out", str(out)],
    )
    assert m.main() == 0
    per = yaml.safe_load(out.read_text(encoding="utf-8"))["per_joint"]["limb/joint1"]

    from remoroo_lc.schema import load_cell

    cell = load_cell(CELL_DIR / "single_6dof_leg.yaml")
    adapter = MockCellAdapter(cell)
    adapter.connect()
    fits = m.identify(cell, adapter, _inertia(cell), verbose=False, joints=[0])
    adapter.disconnect()
    ref = fits["limb/joint1"]
    # the mock path is deterministic, so this is exact rather than approximate:
    # a rounded or re-derived residual would not survive it
    assert per["residual"] == ref["residual"], (per, ref)
    assert per["delay_ticks"] == ref["delay_ticks"], (per, ref)


def test_the_written_file_tells_a_garbage_joint_from_a_clean_one(tmp_path, monkeypatch):
    """Two joints, two wildly different fits, and the file has to say which.

    This is the defect residual closes.  Before it, a 4.99e-09 fit and a 3.7e+03
    one wrote per_joint entries of identical shape, and the only record of the
    difference was the scrollback of a 25-minute commanded-motion session.  The
    per-joint delay matters for the same reason: the top-level
    command_delay_ticks collapses the cell to one number (its maximum) and used
    to be the only delay in the file, so a joint that disagreed left no trace.

    fit_joint is replaced here on purpose: the fit itself is already proven above
    against a known plant, and what is under test is the plumbing from the fit to
    the file, which needs two DIFFERENT answers to have anything to prove.
    """
    from tests.conftest import CELL_DIR

    m = _sysid()
    planted = [(45000.0, 300.0, 2, 4.99e-09), (112.0, 8.0, 5, 3.7e03)]
    pending = list(planted)
    monkeypatch.setattr(m, "fit_joint", lambda *a, **k: pending.pop(0))

    out = tmp_path / "gains_raw.yaml"
    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(CELL_DIR / "single_6dof_leg.yaml"),
         "--joints", "joint1,joint3", "--out", str(out)],
    )
    assert m.main() == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    clean, garbage = doc["per_joint"]["limb/joint1"], doc["per_joint"]["limb/joint3"]
    assert (clean["residual"], clean["delay_ticks"]) == (4.99e-09, 2), clean
    assert (garbage["residual"], garbage["delay_ticks"]) == (3.7e03, 5), garbage
    # and the cell-wide number still reports the largest delay, unchanged
    assert doc["command_delay_ticks"] == 5, doc["command_delay_ticks"]


def test_out_creates_its_parent_directories(tmp_path, monkeypatch):
    """A 25-minute session must not lose every number it measured to a mkdir."""
    from tests.conftest import CELL_DIR

    m = _sysid()
    out = tmp_path / "run_007" / "wf_gap" / "rig" / "gains_raw.yaml"
    assert not out.parent.exists()
    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(CELL_DIR / "single_6dof_leg.yaml"),
         "--joints", "joint1", "--out", str(out)],
    )
    assert m.main() == 0
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["per_joint"], out


class _FakeHardwareCell:
    """HardwareCell's surface as far as main() reaches before identify().

    It records connect/disconnect so a test can ask the only question that
    matters after a failure in that window: did ANYTHING de-energise the arms?
    No hardware is involved, which is the point -- the window is a property of
    main()'s control flow, and a defect in it must be catchable on the mac.
    """

    substeps_per_tick = 1
    real_feedback = True

    def __init__(self, cell, hosts) -> None:
        self.cell, self.hosts, self.log = cell, hosts, []

    def connect(self) -> None:
        self.log.append("connect")

    def disconnect(self) -> None:
        self.log.append("disconnect")

    def estop(self) -> None:
        self.log.append("estop")

    def _read(self):
        q = np.asarray(self.cell.rest_posture(), dtype=float)
        return q, np.zeros_like(q)


@pytest.mark.parametrize("injection", ["kinematics", "no_by_kind", "velocity_limits"])
def test_a_failure_between_connect_and_identify_still_disconnects(monkeypatch, injection):
    """The energised-with-no-handler window, one injection per way it can fail.

    main() connected, then ran the amplitude clip, joint_velocity_limits(),
    KinematicTree(cell) and the by_kind/inertia lookup with no handler at all
    before re-establishing `finally: adapter.disconnect()`.  Each of the three
    injections here is one of the real exceptions that window can raise, and each
    of them used to leave both controllers in servo mode with motion enabled and
    nothing cleaning up -- the exact fault HardwareCell.connect() says "never
    again" about, reintroduced in its caller.
    """
    from tests.conftest import CELL_DIR

    from remoroo_lc.schema import CellSpec, CellSpecError, load_cell

    m = _sysid()
    built: list[_FakeHardwareCell] = []

    def build(cell, hosts):
        built.append(_FakeHardwareCell(cell, hosts))
        return built[-1]

    monkeypatch.setattr(m, "HardwareCell", build)

    if injection == "kinematics":
        def explode(cell):
            raise RuntimeError("kinematics failed to build")
        monkeypatch.setattr(m, "KinematicTree", explode)
        expected: type[BaseException] = RuntimeError
    elif injection == "no_by_kind":
        # a gains file with no by_kind table -- which is exactly what --gains
        # exists to supply -- made the inertia lookup raise KeyError: 'revolute'
        def loader(path, **kw):
            cell = load_cell(path, **kw)
            cell.gains.pop("by_kind")
            return cell
        monkeypatch.setattr(m, "load_cell", loader)
        expected = KeyError
    else:
        # ⚠ This one has to be installed AFTER the load: load_cell validates the
        # velocity limits itself (schema.py:700), so a cell that loaded cannot
        # then fail the window's `qd_max = cell.joint_velocity_limits()` for that
        # reason.  The leg is still worth having -- what it pins is where that
        # statement sits relative to the finally, not how likely it is to raise.
        def explode(self):
            raise CellSpecError("joint has no positive velocity limit")

        def loader(path, **kw):
            cell = load_cell(path, **kw)
            monkeypatch.setattr(CellSpec, "joint_velocity_limits", explode)
            return cell
        monkeypatch.setattr(m, "load_cell", loader)
        expected = CellSpecError

    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(CELL_DIR / "single_6dof_leg.yaml"), "--hardware",
         "--hosts", "127.0.0.1", "--go", "--joints", "joint1"],
    )
    with pytest.raises(expected):
        m.main()
    assert built, "main() never built the adapter; the injection fired too early"
    assert built[0].log == ["connect", "disconnect"], built[0].log


def test_gains_override_needs_only_inertia_on_the_hardware_path(tmp_path, monkeypatch):
    """The rig's prior file carries inertia and nothing else, and that is enough.

    The fit identifies kp/M and kd/M, never M, so inertia is the only thing it
    needs out of --gains.  kp and kd in a gains file are the REFERENCE PLANT's
    business -- the no---hardware self-check builds a Plant and refuses without
    them, which is exactly why the self-check cannot run on a cell whose gains
    are the thing being measured.  This leg pins that the hardware path reaches
    identify_hw on an inertia-only file, and that the inertia it hands the fit --
    and copies into the file -- is the override's, not the cell's.

    identify_hw is stubbed because everything past it is 250 Hz servo traffic;
    the fake adapter still has to survive connect and be disconnected.
    """
    m = _sysid()
    prior = tmp_path / "prior.yaml"
    prior.write_text(
        yaml.safe_dump({"by_kind": {"revolute": {"inertia": 0.73}}}), encoding="utf-8"
    )
    cell_path = _rig_shaped_cell(tmp_path, str(tmp_path / "lc" / "gains.yaml"))

    built: list[_FakeHardwareCell] = []

    def build(cell, hosts):
        built.append(_FakeHardwareCell(cell, hosts))
        return built[-1]

    monkeypatch.setattr(m, "HardwareCell", build)

    handed = {}

    def fake_identify_hw(cell, adapter, inertia, tapes_for, joints, verbose=True):
        handed["inertia"] = np.asarray(inertia, dtype=float).copy()
        labels = cell.joint_labels()
        return {
            labels[j]: {"inertia": float(inertia[j]), "kp": 1.0, "kd": 1.0,
                        "delay_ticks": 3, "residual": 0.5}
            for j in joints
        }

    monkeypatch.setattr(m, "identify_hw", fake_identify_hw)

    out = tmp_path / "gains_raw.yaml"
    monkeypatch.setattr(
        sys, "argv",
        ["sysid_tapes.py", str(cell_path), "--hardware", "--hosts", "127.0.0.1",
         "--go", "--joints", "joint1", "--gains", str(prior), "--out", str(out)],
    )
    assert m.main() == 0
    assert np.allclose(handed["inertia"], 0.73), handed
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["per_joint"]["limb/joint1"]["inertia"] == 0.73, doc["per_joint"]
    assert built[0].log == ["connect", "disconnect"], built[0].log
