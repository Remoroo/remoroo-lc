"""The 2026-10-07 arm sysid SESSION, dry-run end to end on the mock plant.

Before the robot moves (remoroo-world docs/2026-10-07_motor_sysid_vs_sota.md,
"Arm-1 session protocol"): every tape passes the guards, every tape writes its
raw record in contract (B), and the analytic fit run off those records -- the
same regrid the rig uses -- recovers a mock plant whose kp/kd/delay are known.
The mock is set to the SHAPE of the real arm (kp/M ~ 392, 7-tick delay, from
cells/corner_cell/lc/gains.yaml 2026-09-23), not to the 48 Hz placeholder, so
the recovery is tested where the rig actually lives.

    .venv/bin/python -m pytest -q tests/test_sysid_session.py -s     # prints the numbers
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import socket
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "configs" / "cells" / "dual_xarm6.yaml"
# The real two-arm cell (remoroo-world, beside this repo): the only cell with
# jaws, so the only one --gripper can name a drive joint on.
CORNER = ROOT.parent / "remoroo-world" / "cells" / "corner_cell" / "lc" / "cell.yaml"

# The mock plant's KNOWN answer: corner_cell joint1 as fitted on the rig
# 2026-09-23 (cells/corner_cell/lc/gains.yaml: kp 196.1796, kd 12.1869, M 0.5,
# command_delay_ticks 7).  The fit must find these, not read them.
KP, KD, M, DELAY = 196.1796, 12.1869, 0.5, 7
# The arm's rated joint acceleration as the SDK states it: XArmAPI.joint_acc_limit
# max = 20.0 rad/s^2 (xarm-python-sdk 1.18.4 x3/base.py:111).  On hardware the
# script reads it off the SDK; the mock has none, so the test hands it over.
QDD_MAX = 20.0
# Mock jaw: what the gripper fit must recover.  Invented for the mock -- no jaw
# has been measured, which is what the session is for.  1500 = the mock's
# CONFIGURED speed (what the rig reads off the jaw at connect, D4); 5000 = the
# SDK max (xarm.py _GRIPPER_SPEED_MAX).
GRIP_DELAY_S, GRIP_RATE_PER_SPEED, GRIP_CONFIGURED = 0.05, 0.2, 1500.0
GRIP_SPEEDS = (GRIP_CONFIGURED, 5000.0)
# P0 (training home) offset from the cell rest pose the mock parks at, so the
# session has to play the rest -> P0 transit tape first (D3)
P0_OFF = np.array([0.15, -0.1, 0.1, 0.2, -0.1, 0.3])
N_TAPES = 3 * 6 * 7 + 6 + 1      # 3 poses x 6 joints x 7, P0 repeats, the transit

CONTRACT_B = {
    "tape", "params_json", "joints", "excited", "command_hz", "start_pose",
    "send_t", "send_q", "frame_t_host", "frame_t_ctrl", "q", "qd", "tau",
    "target_q", "gripper_pos", "git_hash", "utc", "unit_host",
}


def _sysid():
    spec = importlib.util.spec_from_file_location("sysid_tapes", ROOT / "scripts" / "sysid_tapes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sysid_tapes"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Run the full per-joint session once at three poses; share the result."""
    m = _sysid()
    tmp = tmp_path_factory.mktemp("sysid")
    gains = tmp / "gains.mock.yaml"
    gains.write_text(yaml.safe_dump({
        "plant_rate_hz": 1000, "command_delay_ticks": DELAY,
        "by_kind": {"revolute": {"inertia": M, "kp": KP, "kd": KD},
                    "prismatic": {"inertia": 2.0, "kp": 4 * KP, "kd": 4 * KD}},
    }))
    from remoroo_lc.schema import load_cell

    cell = load_cell(CELL, gains_path=gains)
    rest = cell.rest_posture()[:6].astype(float)
    p0 = rest + P0_OFF
    tapes_dir = tmp / "sysid_tapes"
    tapes_dir.mkdir()
    poses = tapes_dir / "poses.yaml"
    poses.write_text(yaml.safe_dump({
        "joints": [f"joint{i}" for i in range(1, 7)],
        "poses": {
            "P0": {"q": p0.tolist(), "role": "fit"},
            "P1": {"q": (p0 + [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]).tolist(), "role": "fit"},
            "P2": {"q": (p0 + [-0.1, -0.1, 0.1, -0.1, -0.1, -0.1]).tolist(), "role": "heldout"},
        }}))
    # stand-in for track T2b's clearance-checked transit_rest_P0.npz (contract A)
    qt = np.vstack([rest[None], m.min_jerk(rest, p0, 250.0, 0.15, QDD_MAX)])
    np.savez(tapes_dir / "transit_rest_P0.npz", name="transit_rest_P0",
             source="test fixture: min_jerk rest -> P0",
             joints=np.array([f"joint{i}" for i in range(1, 7)]), hz=250.0,
             start_pose=rest, q_target=qt, pose_label="P0", cap_class="transit")
    out = tmp / "gains.fit.yaml"
    raw = tmp / "raw"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.main([str(CELL), "--dry-run", "--hosts", "mock-a,mock-b", "--gains", str(gains),
                     "--poses", str(poses), "--qdd-max", str(QDD_MAX), "--raw-dir", str(raw),
                     "--out", str(out)])
    assert rc == 0
    return {"m": m, "tmp": tmp, "gains": gains, "out": out, "raw": raw, "rest": rest,
            "p0": p0, "doc": yaml.safe_load(out.read_text()), "stdout": buf.getvalue()}


def test_fit_recovers_the_known_plant_from_raw_records(session):
    doc = session["doc"]
    errs = []
    for name, got in doc["per_joint"].items():
        e_kp = abs(got["kp"] - KP) / KP
        e_kd = abs(got["kd"] - KD) / KD
        errs.append((name, got["kp"], got["kd"], got["delay_ticks"], e_kp, e_kd))
        assert got["delay_ticks"] == DELAY, (name, got)
        assert e_kp <= 0.05, (name, got)
        assert e_kd <= 0.15, (name, got)
    for r in errs:
        print(f"  {r[0]:<14} kp {r[1]:8.3f} ({100 * r[4]:.3f}%)  kd {r[2]:7.3f} "
              f"({100 * r[5]:.3f}%)  delay {r[3]}")
    assert len(errs) == 6          # ONE arm: the other unit's joints never fitted


def test_every_tape_wrote_contract_b_and_stayed_inside_the_guards(session):
    files = sorted(session["raw"].glob("*.npz"))
    # 3 poses x 6 joints x 7 tapes, + the noise-floor repeat at P0 for 6 joints,
    # + the rest -> P0 transit
    assert len(files) == N_TAPES, len(files)
    # contract (B): the raw dir carries the poses its labels name
    assert yaml.safe_load((session["raw"] / "poses.yaml").read_text())["poses"]["P2"]["role"] == "heldout"
    lim = 1.5 * 3.14 / 250.0
    peak = {}
    n_moved_other = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert CONTRACT_B <= set(z.files), set(CONTRACT_B) - set(z.files)
        assert float(z["command_hz"]) == 250.0
        assert list(z["joints"]) == [f"left/joint{i}" for i in range(1, 7)]
        S, F = z["send_t"].size, z["frame_t_host"].size
        assert z["send_q"].shape == (S, 6) and z["q"].shape == (F, 6)
        assert z["target_q"].shape == (F, 6) and z["gripper_pos"].shape == (F,)
        assert np.max(np.abs(np.diff(z["send_q"], axis=0))) <= lim + 1e-9
        p = json.loads(str(z["params_json"]))
        assert p["pose_label"] in ("P0", "P1", "P2") and p["role"] in ("fit", "heldout")
        cls = p["peaks"]["cap_class"]
        assert cls == p["cap_class"], p
        c = peak.setdefault(cls, [0.0, 0.0])
        c[0], c[1] = max(c[0], p["peaks"]["jump"]), max(c[1], p["peaks"]["acc"])
        if cls in ("sweep", "transit"):
            # float32 tape quantisation: 4 ulp(1.7 rad) * 250^2 ~ 0.03 rad/s^2;
            # the peak INCLUDES the tape's boundaries (anchor before, hold after)
            assert p["peaks"]["acc"] <= QDD_MAX + 0.03, (f.name, p["peaks"])
        else:
            # D1 step: no tick above qd_max/hz, the 0.03 step included
            assert p["peaks"]["jump"] <= 3.14 / 250.0 + 1e-6, (f.name, p["peaks"])
        if p["kind"] == "replay":
            continue
        k = [f"left/joint{i}" for i in range(1, 7)].index(str(z["excited"][0]))
        others = np.delete(z["send_q"], k, axis=1)
        n_moved_other += int(np.ptp(others, axis=0).max() > 1e-9)
    assert n_moved_other == 0      # one joint at a time, as before
    for cls, (j, a) in sorted(peak.items()):
        print(f"  {cls:<8} peak jump {j:.4f} rad  peak accel incl. boundaries {a:.2f} rad/s^2")


def test_chirps_start_and_end_at_the_anchor_at_rest(session):
    """Round-2 item 1: the tapered chirps leave and rejoin the anchor with zero
    target velocity, so their boundary acceleration is inside ACCEL_CAP (the
    untapered ones stopped dead at 106-132 rad/s^2)."""
    m = session["m"]
    for f in sorted(session["raw"].glob("unit0_P1_*_chirp_*.npz")):
        z = np.load(f)
        p = json.loads(str(z["params_json"]))
        k = [f"left/joint{i}" for i in range(1, 7)].index(str(z["excited"][0]))
        sq, anchor = z["send_q"][:, k], z["start_pose"][k]
        assert abs(sq[0] - anchor) < 1e-6 and abs(sq[-1] - anchor) < 1e-6, f.name
        assert abs(sq[1] - sq[0]) * 250 < 1e-2 and abs(sq[-1] - sq[-2]) * 250 < 1e-2
        full = m.with_boundaries(z["send_q"], z["start_pose"])
        acc = float(np.max(m.tape_peaks(full, 250.0)["acc"]))
        print(f"  {f.name}: f1 {p['f1']:.2f} Hz, peak accel incl. boundaries {acc:.2f}")
        assert acc <= QDD_MAX + 0.03


def test_untapered_chirp_is_refused_at_its_boundary(session):
    """check_tape sees the stop the bare tape hides."""
    from remoroo_lc.schema import load_cell

    m = session["m"]
    cell = load_cell(CELL, gains_path=session["gains"])
    lo, hi = cell.joint_limits()
    qd = cell.joint_velocity_limits()
    f1 = m.chirp_f1(float(qd[0]), QDD_MAX, 0.08)
    bare = m.chirp_tape(cell, 0, amplitude=0.08, f1=f1, duration_s=6.0)
    interior = float(np.max(m.tape_peaks(bare.astype(float), 250.0)["acc"]))
    with pytest.raises(SystemExit, match="boundaries"):
        m.check_tape("bare", "sweep", bare, lo, hi, qd, QDD_MAX, 250.0,
                     cell.rest_posture())
    print(f"  untapered 0.08 chirp: interior {interior:.1f} rad/s^2, refused at the boundary")


def test_triangle_corners_sit_under_accel_cap_by_the_margin(session):
    """Round-3 item 2: the triangle's corners are rounded at ACCEL_CAP / 1.15
    (the margins rule, 10-15% headroom), not at the cap (20.0048 in float32)."""
    m = session["m"]
    corner = QDD_MAX / m.CORNER_HEADROOM
    assert m.CORNER_HEADROOM == 1.15
    accs = []
    for f in sorted(session["raw"].glob("*_triangle_0.05.npz")):
        p = json.loads(str(np.load(f)["params_json"]))
        assert p["corner_accel"] == corner, (f.name, p["corner_accel"])
        # float32 tape quantisation: 4 ulp(1.7 rad) * 250^2 ~ 0.03 rad/s^2
        assert p["peaks"]["acc"] <= corner + 0.03, (f.name, p["peaks"])
        accs.append(p["peaks"]["acc"])
    assert len(accs) == 3 * 6
    print(f"  triangle corners at {corner:.2f} rad/s^2 (ACCEL_CAP {QDD_MAX} / "
          f"{m.CORNER_HEADROOM}); peak incl. boundaries {min(accs):.4f}..{max(accs):.4f}")
    assert max(accs) < QDD_MAX / 1.10      # at least the margin rule's 10%


def test_new_chirp_top_frequency_is_set_by_the_acceleration_cap(session):
    z = np.load(next(session["raw"].glob("unit0_P0_joint1_chirp_0.02.npz")))
    p = json.loads(str(z["params_json"]))
    f_vel = 3.14 / (3 * 2 * np.pi * 0.02)
    f_acc = np.sqrt(QDD_MAX / 0.02) / (2 * np.pi)
    print(f"  0.02 rad chirp: f1 {p['f1']:.3f} Hz (velocity rule {f_vel:.2f}, "
          f"acceleration rule {f_acc:.2f}); peak target accel {p['peaks']['acc']:.2f}")
    assert abs(p["f1"] - min(f_vel, f_acc)) < 1e-9 and p["f1"] < f_vel


def test_provenance_adds_tape_pose_tapes_and_raw_paths(session):
    doc = session["doc"]
    assert set(doc["tape_pose"]) == {"P0", "P1", "P2"}
    assert doc["tape_pose"]["P2"]["role"] == "heldout"
    q1 = np.array(list(doc["tape_pose"]["P1"]["q"].values()))
    assert np.allclose(q1, session["p0"] + [0.1, 0.1, -0.1, 0.1, 0.1, 0.1], atol=2e-3)
    assert len(doc["sysid_raw_files"]) == N_TAPES
    assert "transit_rest_P0" in doc["sysid_raw_files"][0]       # first motion
    assert all(Path(p).exists() for p in doc["sysid_raw_files"])


def test_target_q_carries_the_delay_split(session):
    """The mock's TARGET_Q is the send delayed by the plant's own delay: the
    split the rig record will make between transport and servo lag."""
    z = np.load(next(session["raw"].glob("unit0_P0_joint2_step_up_0.15.npz")))
    sq, tq = z["send_q"][:, 1], z["target_q"][::4, 1]
    lag = int(np.argmax(np.abs(tq - tq[0]) > 1e-6) - np.argmax(np.abs(sq - sq[0]) > 1e-6))
    assert lag == DELAY, lag


def test_replay_mode_streams_a_contract_a_tape(session):
    m, tmp = session["m"], session["tmp"]
    rest = session["rest"]
    hz = 250.0
    t = np.arange(int(10 * hz)) / hz
    # a smooth all-joints tape inside every cap, boundaries included: a 1 - cos
    # bump of 0.03 rad at 1 Hz leaves and returns at rest (0.09 rad/s, 0.6
    # rad/s^2 peaks).  A plain sine here stops dead at 0.19 rad/s -- 47 rad/s^2
    # at the boundary, which check_tape now refuses
    start = rest + [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
    bump = lambda f: (0.015 * (1.0 - np.cos(2 * np.pi * f * t)))[:, None] * np.ones(6)  # noqa: E731
    qt = start + bump(1.0)
    tape = tmp / "multi.npz"
    np.savez(tape, name="multi_joint_fixture", source="test fixture",
             joints=np.array([f"joint{i}" for i in range(1, 7)]), hz=hz,
             start_pose=start, q_target=qt, pose_label="P1", cap_class="sweep")
    raw = tmp / "raw_replay"
    rc = m.main([str(CELL), "--dry-run", "--hosts", "mock-a,mock-b", "--gains",
                 str(session["gains"]), "--qdd-max", str(QDD_MAX), "--raw-dir", str(raw),
                 "--replay", str(tape)])
    assert rc == 0
    z = np.load(next(raw.glob("*.npz")))
    assert set(z["excited"]) == {f"left/joint{i}" for i in range(1, 7)}
    assert np.allclose(z["send_q"], qt)
    assert np.allclose(z["start_pose"], start)
    assert json.loads(str(z["params_json"]))["role"] == "heldout"

    # refusals: wrong rate, wrong joints -- before anything moves
    bad = tmp / "bad.npz"
    np.savez(bad, name="b", source="t", joints=np.array([f"joint{i}" for i in range(1, 7)]),
             hz=50.0, start_pose=start, q_target=qt, pose_label="P1", cap_class="sweep")
    with pytest.raises(SystemExit, match="command_hz"):
        m.main([str(CELL), "--dry-run", "--hosts", "a,b", "--gains", str(session["gains"]),
                "--qdd-max", str(QDD_MAX), "--raw-dir", str(raw), "--replay", str(bad)])
    np.savez(bad, name="b", source="t", joints=np.array([f"j{i}" for i in range(1, 7)]),
             hz=hz, start_pose=start, q_target=qt, pose_label="P1", cap_class="sweep")
    with pytest.raises(SystemExit, match="not this unit"):
        m.main([str(CELL), "--dry-run", "--hosts", "a,b", "--gains", str(session["gains"]),
                "--qdd-max", str(QDD_MAX), "--raw-dir", str(raw), "--replay", str(bad)])
    # a 'sweep' tape over ACCEL_CAP is refused, not clipped ...
    fast = start + bump(6.0)          # 21.3 rad/s^2, 0.0023 rad/tick
    args = [str(CELL), "--dry-run", "--hosts", "a,b", "--gains", str(session["gains"]),
            "--qdd-max", str(QDD_MAX), "--raw-dir", str(raw), "--replay", str(bad)]
    jn = np.array([f"joint{i}" for i in range(1, 7)])
    np.savez(bad, name="b", source="t", joints=jn, hz=hz, start_pose=start,
             q_target=fast, pose_label="P1", cap_class="sweep")
    with pytest.raises(SystemExit, match="acceleration"):
        m.main(args)
    # ... the SAME tape as 'edge' is what deployment may send (21.3 rad/s^2,
    # but 0.0023 rad/tick <= qd_max/hz): accepted, D1
    np.savez(bad, name="b", source="t", joints=jn, hz=hz, start_pose=start,
             q_target=fast, pose_label="P1", cap_class="edge")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert m.main(args) == 0
    # ... and the plan prints its uncapped peak beside D1 for the operator
    lines = buf.getvalue().splitlines()
    i_d1 = next(i for i, L in enumerate(lines) if L.startswith("  D1 caps"))
    edge_acc = float(np.max(m.tape_peaks(m.with_boundaries(fast, start), hz)["acc"]))
    print(lines[i_d1 + 1] + f"   (tape: {edge_acc:.2f})")
    assert lines[i_d1 + 1].startswith(
        f"     edge-class tapes: 1, peak target accel {edge_acc:.2f} rad/s^2")
    assert edge_acc > QDD_MAX
    # an 'edge' tape with a tick above qd_max/hz is refused
    jumpy = fast.copy()
    jumpy[100:, 0] += 0.013
    np.savez(bad, name="b", source="t", joints=jn, hz=hz, start_pose=start,
             q_target=jumpy, pose_label="P1", cap_class="edge")
    with pytest.raises(SystemExit, match="tick jump"):
        m.main(args)
    # no cap_class (contract A) -> refused
    np.savez(bad, name="b", source="t", joints=jn, hz=hz, start_pose=start,
             q_target=qt, pose_label="P1")
    with pytest.raises(SystemExit, match="cap_class"):
        m.main(args)


@pytest.mark.skipif(not CORNER.exists(), reason="remoroo-world corner_cell not beside this repo")
def test_gripper_mode_recovers_the_mock_jaw(session):
    m, tmp = session["m"], session["tmp"]
    raw = tmp / "raw_grip"
    base = [str(CORNER), "--dry-run", "--hosts", "mock-a,mock-b", "--gains",
            str(session["gains"]), "--qdd-max", str(QDD_MAX), "--raw-dir", str(raw),
            "--mock-gripper", f"{GRIP_DELAY_S},{GRIP_RATE_PER_SPEED},{GRIP_CONFIGURED}"]
    # the jaw is named per the cell: unit 0's drive joint, nothing else
    from remoroo_lc.schema import load_cell
    cell = load_cell(CORNER, gains_path=session["gains"])
    model, jaws = m.unit_jaw_joints(cell, slice(0, 6))
    assert jaws == ["drive_joint"] and model == cell.models[0].name
    assert m.unit_jaw_joints(cell, slice(6, 12)) == (model, ["drive_joint_1"])
    # bare, or qualified by THIS unit's model exactly -- not any '<x>/drive_joint'
    for wrong in ("drive_joint_1", "joint6", "left_finger_joint", "foo/drive_joint",
                  "right/drive_joint", f"{model}/drive_joint_1", f"x{model}/drive_joint"):
        with pytest.raises(SystemExit, match="jaw drive joint"):
            m.main(base + ["--gripper", wrong])
    qualified = [a if a != str(raw) else str(tmp / "raw_grip_qualified") for a in base]
    assert m.main(qualified + ["--gripper", f"{model}/drive_joint"]) == 0
    assert len(list((tmp / "raw_grip_qualified").glob("*.npz"))) == 2
    rc = m.main(base + ["--gripper", "drive_joint"])
    assert rc == 0
    files = sorted(raw.glob("*.npz"))
    assert len(files) == 2
    for f, sp in zip(sorted(files, key=lambda p: float(p.stem.split("speed")[1])), GRIP_SPEEDS):
        z = np.load(f)
        assert CONTRACT_B <= set(z.files)
        assert list(z["excited"]) == ["drive_joint"]
        assert z["send_t"].size == 2 * 5          # open->close->open, 5 repeats
        p = json.loads(str(z["params_json"]))
        assert p["speed_configured"] == GRIP_CONFIGURED and p["speed_max"] == 5000.0
        assert p["speed_source"] == ("configured" if sp == GRIP_CONFIGURED else "sdk_max")
        g = m.fit_gripper({k: z[k] for k in ("send_t", "send_q", "frame_t_host", "gripper_pos")})
        rate = GRIP_RATE_PER_SPEED * sp
        print(f"  gripper speed {sp:g} ({p['speed_source']}): delay {g['delay_s'] * 1e3:.1f} ms "
              f"(true {GRIP_DELAY_S * 1e3:.0f}), rate {g['rate_units_per_s']:.1f} "
              f"(true {rate:.1f})")
        assert abs(g["delay_s"] - GRIP_DELAY_S) <= 2.0 / 250.0
        assert abs(g["rate_units_per_s"] - rate) <= 0.03 * rate


def test_poses_session_starts_only_at_p0_or_rest(session, tmp_path):
    """D3: anywhere else is refused before anything moves; from rest without
    the transit tape is refused."""
    m = session["m"]
    p0, rest = session["p0"], session["rest"]
    tol = np.full(6, 3.14 / 250.0)
    assert m.session_start(p0 + 0.5 * tol, p0, rest, tol) == "P0"
    assert m.session_start(rest - 0.5 * tol, p0, rest, tol) == "rest"
    with pytest.raises(SystemExit, match="neither P0"):
        m.session_start(rest + 2 * tol, p0, rest, tol)
    poses = tmp_path / "poses.yaml"
    poses.write_text((session["tmp"] / "sysid_tapes" / "poses.yaml").read_text())
    with pytest.raises(SystemExit, match="transit_rest_P0.npz"):
        m.main([str(CELL), "--dry-run", "--hosts", "mock-a,mock-b", "--gains",
                str(session["gains"]), "--poses", str(poses), "--qdd-max", str(QDD_MAX),
                "--raw-dir", str(tmp_path / "raw")])


def test_session_ends_at_p0_and_the_plan_prices_the_motion(session):
    """The SESSION PLAN is printed before any motion; its total is the motion
    the mock then actually streamed; the session ends back at P0, not rest."""
    out = session["stdout"]
    lines = out.splitlines()
    i_plan = next(i for i, L in enumerate(lines) if L.startswith("SESSION PLAN"))
    i_first = next(i for i, L in enumerate(lines) if "transit_rest_P0 (transit):" in L)
    assert i_plan < i_first                       # printed before anything moved
    assert "D3 starts at rest; ends at P0" in out
    # round-3 item 3: the edge peak beside D1 (none here: per-joint tapes are
    # step/sweep) and the workspace check, arm 2 = the unit NOT excited
    assert "     edge-class tapes: none in this plan" in lines
    assert ("  CHECK: carton at nominal position or removed; arm 2 at rig home or "
            "lc rest; e-stop in hand") in lines
    for L in lines[i_plan:]:
        if L.strip().startswith(("class ", "TOTAL", "D1", "D2", "D3")) or "SESSION" in L:
            print(L)
    planned = float(next(L for L in lines if "TOTAL commanded motion" in L).split()[3])
    done = float(next(L for L in lines if "commanded motion:" in L).split()[2])
    print(f"  planned {planned:.2f} s, streamed {done:.2f} s")
    assert abs(planned - done) <= 0.01 * done
    back = next(L for L in lines if "back at P0" in L)
    q_end = np.array([float(x) for x in back.split("[")[1].split("]")[0].split()])
    assert np.allclose(q_end, session["p0"], atol=2e-3), (q_end, session["p0"])
    assert not np.allclose(q_end, session["rest"], atol=2e-2)


def test_d2_only_the_excited_unit_is_enabled(monkeypatch, session):
    """D2 on the REAL HardwareCell against a mocked SDK: unit 1 gets its
    port-30000 reader and nothing else -- no XArmAPI, no motion_enable, no
    set_mode, no servo_j."""
    from remoroo_lc.adapters import xarm_rt
    from remoroo_lc.schema import load_cell

    m = session["m"]
    cell = load_cell(CELL, gains_path=session["gains"])
    rest = cell.rest_posture().astype(float)
    calls: list[tuple[str, str]] = []

    class FakeArm:
        axis, joint_speed_limit, joint_acc_limit = 6, [0.0, 3.14], [0.0, QDD_MAX]

        def __init__(self, host, is_radian=True):
            self.host = host
            calls.append((host, "XArmAPI"))

        def __getattr__(self, name):
            def call(*_a, **_k):
                calls.append((self.host, name))
                return 0
            return call

    class FakeRT:
        def __init__(self, host, n_joints=6):
            self.host, self._buf = host, b""
            self._sock, self._peer = socket.socketpair()
            self.q = rest[:6] if host == "arm-a" else rest[6:]

        def connect(self):
            calls.append((self.host, "rt.connect"))

        def close(self):
            self._sock.close()
            self._peer.close()

        def read(self):
            return types.SimpleNamespace(q=self.q, qd=np.zeros(6))

    pkg, wrapper = types.ModuleType("xarm"), types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = FakeArm
    pkg.wrapper = wrapper
    monkeypatch.setitem(sys.modules, "xarm", pkg)
    monkeypatch.setitem(sys.modules, "xarm.wrapper", wrapper)
    monkeypatch.setattr(xarm_rt, "XArmRealTime", FakeRT)

    hc = m.HardwareCell(cell, ["arm-a", "arm-b"], 0)
    hc.connect()
    hc.reset(rest)             # a glide + 0.4 s hold, streamed at 250 Hz
    hc.disconnect()
    a = [c for h, c in calls if h == "arm-a"]
    b = [c for h, c in calls if h == "arm-b"]
    print(f"  arm-a: {sorted(set(a))}\n  arm-b: {b}")
    assert b == ["rt.connect"], b
    assert {"motion_enable", "set_mode", "set_state", "set_servo_angle_j"} <= set(a)
    assert a.count("set_servo_angle_j") >= int(m.HOLD_S * 250)


def test_the_old_self_check_path_is_unchanged():
    """No --dry-run/--hardware: the original lockstep self-check, no raw files."""
    m = _sysid()
    assert m.MAX_DELAY_TICKS == 12
    assert m.main([str(ROOT / "configs" / "cells" / "single_6dof_leg.yaml")]) == 0
