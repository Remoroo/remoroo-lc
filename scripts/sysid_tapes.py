#!/usr/bin/env python3
"""System identification: measure a cell's actuator gains and command delay.

Generates per-joint step and chirp excitations, streams them through a
CellAdapter, records the response, and fits the plant model

    M q_ddot = kp (q_target - q) - kd q_dot        with an N-tick command delay

writing a gains yaml of the same shape as configs/gains.default.yaml.

These are JOINT-space tapes, not action tapes: they drive the servos directly and
never go through the controller, because what is being measured is the thing the
controller sits on top of.

    python scripts/sysid_tapes.py configs/cells/dual_xarm6.yaml            # self-check
    python scripts/sysid_tapes.py configs/cells/dual_xarm6.yaml --out g.yaml

Self-check is the default and is what the test runs: excite the reference plant
with known gains and confirm the fit recovers them.  A fitter that has never been
shown to recover a known answer is not evidence about an unknown one.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.constants import DTYPE  # noqa: E402
from remoroo_lc.plant import Plant  # noqa: E402
from remoroo_lc.reference.kinematics import KIND_PRISMATIC, KIND_REVOLUTE, KinematicTree  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402

MAX_DELAY_TICKS = 8


def step_tape(cell, joint: int, amplitude: float = 0.15, hold_s: float = 1.2,
              ramp_ticks: int = 0) -> np.ndarray:
    """A single joint steps by `amplitude` and holds; everything else holds still.

    `ramp_ticks` spreads the rise over that many ticks.  The reference plant
    takes the pure step (which keeps the self-check exact); hardware gets a
    fast ramp, because an instant 8-degree target jump is a jerk command, and
    the fit regresses the APPLIED target sequence so a ramp costs it nothing.
    """
    hz = float(cell.limits["rates"]["command_hz"])
    n = int(round(hold_s * hz))
    q0 = cell.rest_posture()
    tape = np.repeat(q0[None], 2 * n, axis=0).astype(DTYPE)
    lo, hi = cell.joint_limits()
    amp = float(np.clip(amplitude, 0.0, 0.4 * (hi[joint] - lo[joint])))
    target = np.clip(q0[joint] + amp, lo[joint] + 1e-3, hi[joint] - 1e-3)
    tape[n:, joint] = target
    if ramp_ticks > 0:
        r = min(int(ramp_ticks), n)
        tape[n:n + r, joint] = np.linspace(q0[joint], target, r, endpoint=False)
    return tape


def chirp_tape(
    cell, joint: int, amplitude: float = 0.08, f0: float = 0.2, f1: float = 25.0,
    duration_s: float = 6.0,
) -> np.ndarray:
    """A linear-frequency chirp on one joint.

    The chirp is what pins kd down: a step mostly reports kp and the delay,
    because the damping only shows up in how the approach is shaped, whereas a
    sweep through the resonance reports both.
    """
    hz = float(cell.limits["rates"]["command_hz"])
    n = int(round(duration_s * hz))
    t = np.arange(n) / hz
    q0 = cell.rest_posture()
    lo, hi = cell.joint_limits()
    amp = float(np.clip(amplitude, 0.0, 0.2 * (hi[joint] - lo[joint])))
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    tape = np.repeat(q0[None], n, axis=0).astype(DTYPE)
    tape[:, joint] = np.clip(
        q0[joint] + amp * np.sin(phase), lo[joint] + 1e-3, hi[joint] - 1e-3
    )
    return tape


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def fit_joint(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dt: float,
    inertia: float,
    substeps: int = 1,
) -> tuple[float, float, int, float]:
    """Fit (kp, kd, delay) for one joint from a recorded response.

    Fitted against the EXACT discrete update rather than a continuous
    approximation of it:

        qd[k+1] = qd[k] + (dt/M) (kp (applied[k] - q[k]) - kd qd[k])

    A centred derivative of qd instead sits half a sample away from the
    acceleration that semi-implicit Euler actually applied, and that half sample
    biases kp low by a factor of four even when the feedback rate is ample -- the
    fit looks converged and is simply wrong.  Against real hardware the discrete
    model is an approximation either way; against this plant it is exact, which
    is what makes the self-check meaningful.

    Excitations are fitted as SEPARATE segments and their regression rows
    stacked.  Concatenating the recordings first would put one row across the
    join between two runs -- where the state was reset and nothing continuous
    happened -- and that single row is enough to move the answer by a factor of
    four, because it is a huge apparent acceleration with no cause.

    The model is linear in (kp, kd) once the delay is fixed, so the delay is
    searched over its small integer range and the gains come from a least-squares
    solve at each candidate.  Searching rather than optimising keeps it exact:
    the delay is an integer number of samples and there is nothing to descend.
    """
    best = None
    for d in range(MAX_DELAY_TICKS * substeps + 1):
        rows_A, rows_b = [], []
        for q_target, q, qd in segments:
            # Re-based by one sample so that d = 0 means "no delay".  The state
            # recorded at sample k is the state AFTER that sample's update, so the
            # update from k to k+1 used the target in force at k+1; without the
            # rebase a genuinely undelayed controller has no representable shift
            # and the fit runs away to a negative stiffness.
            base = np.concatenate([q_target[1:], q_target[-1:]])
            applied = np.concatenate([np.full(d, base[0]), base[:-d]]) if d else base
            rows_A.append(np.stack([applied[:-1] - q[:-1], -qd[:-1]], axis=1))
            rows_b.append(inertia * (qd[1:] - qd[:-1]) / dt)
        A = np.concatenate(rows_A)
        b = np.concatenate(rows_b)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = float(np.mean((A @ sol - b) ** 2))
        if best is None or resid < best[3]:
            best = (float(sol[0]), float(sol[1]), d, resid)
    kp, kd, d, resid = best
    # The delay is searched in FEEDBACK samples and reported in command ticks,
    # which is the unit the plant and the vendor both speak.
    return kp, kd, max(0, int(round(d / substeps))), resid


def identify(cell, adapter, inertia: np.ndarray, verbose: bool = True,
             tapes_for=None, joints=None) -> dict:
    """Run every excitation and fit every joint.

    State is read at the controller's feedback rate rather than the command rate
    where the adapter offers one.  A 48 Hz servo sampled at the 250 Hz command
    rate spans its own rise in about three samples, and a numerical derivative of
    that recovers kp an order of magnitude low -- which is the sort of quiet
    wrongness that would then propagate into every scoreboard number without
    ever announcing itself.
    """
    sub = int(getattr(adapter, "substeps_per_tick", 1))
    dt = 1.0 / (float(cell.limits["rates"]["command_hz"]) * sub)
    if verbose and sub == 1 and not getattr(adapter, "real_feedback", False):
        print(
            "  NOTE: no high-rate feedback available; fitting at the command rate, "
            "which is only trustworthy for servos well below command_hz / 10"
        )
    labels = cell.joint_labels()
    if tapes_for is None:
        tapes_for = lambda j: (step_tape(cell, j), chirp_tape(cell, j))  # noqa: E731
    out: dict[str, dict] = {}
    for j in (range(cell.n_joints) if joints is None else joints):
        segments = []
        for tape in tapes_for(j):
            rec_t, rec_q, rec_qd = [], [], []
            adapter.reset(cell.rest_posture())
            for row in tape:
                adapter.stream_targets(row)
                q, qd = adapter.tick(record=sub > 1) if sub > 1 else adapter.tick()
                if sub > 1:
                    sq, sqd = adapter.last_substeps
                    rec_t.extend([row[j]] * sub)
                    rec_q.extend(sq[:, j].tolist())
                    rec_qd.extend(sqd[:, j].tolist())
                else:
                    rec_t.append(row[j])
                    rec_q.append(q[j])
                    rec_qd.append(qd[j])
            segments.append(
                (
                    np.asarray(rec_t, float),
                    np.asarray(rec_q, float),
                    np.asarray(rec_qd, float),
                )
            )
        kp, kd, delay, resid = fit_joint(segments, dt, float(inertia[j]), substeps=sub)
        out[labels[j]] = {
            "inertia": float(inertia[j]),
            "kp": round(kp, 4),
            "kd": round(kd, 4),
            "delay_ticks": int(delay),
            "residual": resid,
        }
        if verbose:
            print(
                f"  {labels[j]:<24} kp={kp:10.2f}  kd={kd:8.2f}  "
                f"delay={delay} ticks  resid={resid:.3e}"
            )
    return out




# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #

class HardwareCell:
    """The real cell, shaped like the mock for identify(): reset / stream / tick.

    One lc model can span several controllers (this rig: one 12-joint tree on
    two xArm boxes), which the generic CellAdapter deliberately refuses to map;
    here the mapping is explicit and local: hosts are given IN MODEL JOINT
    ORDER and each takes an equal contiguous slice.

    Commands go through XArmUnit (the servo-mode handshake and its faults live
    there); feedback comes from the port-30000 real-time report at 250 Hz --
    the SDK's status API updates at ~100 Hz, and sampling that at 250 would
    hand the fit stale repeats.  Pacing is wall-clock: stream, sleep to the
    tick boundary, read the freshest frame.

    Safety, in order of appearance:
      * connect() cross-checks the cell's joint and velocity limits against
        what every controller reports about itself (a cell file wider than the
        machine is a fault mid-episode waiting to happen);
      * reset() REFUSES if any joint is more than 0.6 rad from the rest
        posture -- the operator parks the arm near home first; a blind
        joint-space lerp across the workspace is not this script's call;
      * every streamed target is clipped to limits and guarded against
        per-tick jumps beyond 1.5x the joint's own velocity limit -- tripping
        the guard estops;
      * falling behind the 250 Hz grid for 25 consecutive ticks estops: a
        starved servo_j faults the controller in a way we did not choose.
    """

    substeps_per_tick = 1
    real_feedback = True    # port-30000 IS the controller's own 250 Hz truth
    RAMP_QD = 0.15          # rad/s, the ramp-to-rest speed
    FAR_FROM_REST = 0.6     # rad, refuse-to-start threshold

    def __init__(self, cell, hosts: list[str]) -> None:
        from remoroo_lc.adapters.xarm import XArmUnit
        from remoroo_lc.adapters.xarm_rt import XArmRealTime

        self.cell = cell
        n = cell.n_joints
        if n % len(hosts):
            raise SystemExit(
                f"{n} joints across {len(hosts)} hosts does not divide evenly; "
                "this mapping wants explicit config, not a guess"
            )
        per = n // len(hosts)
        hz = float(cell.limits["rates"]["command_hz"])
        self.dt = 1.0 / hz
        self.lo, self.hi = cell.joint_limits()
        self.qd_max = cell.joint_velocity_limits()
        self.slices = [slice(i * per, (i + 1) * per) for i in range(len(hosts))]
        self.units = [
            XArmUnit(name=f"unit{i}@{h}", host=h, n_joints=per, command_hz=hz,
                     has_effector=False,
                     q_lo=self.lo[self.slices[i]], q_hi=self.hi[self.slices[i]],
                     qd_max=self.qd_max[self.slices[i]])
            for i, h in enumerate(hosts)
        ]
        self.readers = [XArmRealTime(h, n_joints=per) for h in hosts]
        self._last_target: np.ndarray | None = None
        self._late = 0

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        # Any failure past the first handshake DISCONNECTS everything: the
        # first hardware contact of this code left two arms energised in servo
        # mode behind a crashed limits() -- never again.
        try:
            for u in self.units:
                u.connect()
            for r in self.readers:
                r.connect()
            for u in self.units:
                u.limits()   # axis count + speed ceiling, checked by the unit
            q, _ = self._read()
            print(f"  [hw] connected {len(self.units)} unit(s); "
                  f"q = {np.array2string(q, precision=3)}")
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        for r in self.readers:
            try:
                r.close()
            except Exception:
                pass
        for u in self.units:
            try:
                u.disconnect()
            except Exception:
                pass

    def estop(self) -> None:
        for u in self.units:
            try:
                u.estop()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def _read(self) -> tuple[np.ndarray, np.ndarray]:
        """Freshest 250 Hz frame from every reader, stitched in model order."""
        from remoroo_lc.adapters.xarm_rt import FRAME_BYTES

        q = np.zeros(self.cell.n_joints, dtype=DTYPE)
        qd = np.zeros(self.cell.n_joints, dtype=DTYPE)
        for r, sl in zip(self.readers, self.slices):
            sample = None
            while True:                       # drain the backlog to the newest
                buffered = len(r._buf) >= FRAME_BYTES
                readable = bool(select.select([r._sock], [], [], 0)[0])
                if not buffered and not readable:
                    break
                sample = r.read()
            if sample is None:
                sample = r.read()             # block for the next frame
            q[sl] = sample.q
            qd[sl] = sample.qd
        return q, qd

    def _pace_soft(self, deadline: float) -> float:
        """Sleep to the next grid point.  NEVER aborts for ordinary lateness and
        NEVER bursts to catch up: first hardware contact measured the loop a
        mere 0.14 ms/tick over budget, and a hard guard turned that into an
        emergency stop.  Real-time is not required here -- the record is
        regridded afterwards -- so lateness just resyncs and is counted."""
        deadline += self.dt
        now = time.perf_counter()
        slack = deadline - now
        if slack > 0:
            time.sleep(slack)
        elif -slack > 0.2:
            self.estop()
            raise RuntimeError(
                f"loop wedged {-slack * 1e3:.0f} ms behind the grid -- stopped")
        else:
            self._late += 1
            deadline = now
        return deadline

    def _guard(self, row: np.ndarray) -> np.ndarray:
        row = np.clip(np.asarray(row, dtype=np.float64).reshape(-1),
                      self.lo + 1e-3, self.hi - 1e-3)
        if self._last_target is not None:
            step = np.abs(row - self._last_target)
            cap = self.qd_max * self.dt * 1.5 + 1e-6
            if np.any(step > cap):
                j = int(np.argmax(step - cap))
                self.estop()
                raise RuntimeError(
                    f"target jump {step[j]:.4f} rad on joint {j} exceeds "
                    f"{cap[j]:.4f} (1.5x its velocity limit per tick) -- stopped")
        return row

    def stream_targets(self, row: np.ndarray) -> None:
        row = self._guard(row)
        for u, sl in zip(self.units, self.slices):
            u.stream_targets(row[sl])
        self._last_target = row

    def run_tape(self, tape: np.ndarray, j: int):
        """Play one excitation and return (applied, q, qd) on the exact grid.

        Commands go ONLY to the unit that owns joint j (the other arm holds by
        itself in servo mode -- half the SDK round-trips).  A pump thread
        drains that unit's 250 Hz report the whole time.  Afterwards both
        event streams -- (wall time, target sent) and (wall time, frame) --
        are sample-and-held onto the exact command grid, so the fit sees a
        uniform dt regardless of send-loop jitter, and the constant transport
        offset lands where it belongs: in the fitted delay."""
        ui = next(i for i, sl in enumerate(self.slices)
                  if sl.start <= j < sl.stop)
        sl, unit, reader = self.slices[ui], self.units[ui], self.readers[ui]
        frames: list[tuple[float, float, float]] = []
        stop = threading.Event()

        def pump() -> None:
            while not stop.is_set():
                try:
                    smp = reader.read()
                except Exception:
                    return
                frames.append((time.perf_counter(),
                               float(smp.q[j - sl.start]),
                               float(smp.qd[j - sl.start])))

        # frames buffered during the preceding reset would enter the record
        # with arrival times far from their capture times (measured: 305-334
        # "frames/s" over a 250 Hz stream = backlog draining) -- flush first
        from remoroo_lc.adapters.xarm_rt import FRAME_BYTES
        while (len(reader._buf) >= FRAME_BYTES
               or select.select([reader._sock], [], [], 0)[0]):
            reader.read()
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        sends: list[tuple[float, float]] = []
        t_start = deadline = time.perf_counter()
        try:
            for i, row in enumerate(tape):
                row = self._guard(row)
                unit.stream_targets(row[sl])
                self._last_target = row
                sends.append((time.perf_counter(), float(row[j])))
                if i % 64 == 0 and i and (
                        not frames
                        or time.perf_counter() - frames[-1][0] > 0.5):
                    raise RuntimeError("feedback stream stalled >0.5 s -- stopped")
                deadline = self._pace_soft(deadline)
        except BaseException:
            self.estop()
            stop.set()
            raise
        stop.set()
        th.join(timeout=1.0)
        wall = time.perf_counter() - t_start
        rate = len(sends) / max(wall, 1e-9)
        if rate < 100.0:
            self.estop()
            raise RuntimeError(
                f"achieved only {rate:.0f} Hz of target sends; the excitation "
                "is too distorted to fit -- stopped")
        st = np.asarray([t for t, _ in sends])
        sv = np.asarray([v for _, v in sends])
        ft = np.asarray([f[0] for f in frames])
        if ft.size < 10:
            self.estop()
            raise RuntimeError("almost no feedback frames recorded -- stopped")
        fq = np.asarray([f[1] for f in frames])
        fqd = np.asarray([f[2] for f in frames])
        t0, t1 = max(st[0], ft[0]), min(st[-1], ft[-1])
        grid = np.arange(t0, t1, self.dt)
        gi = np.clip(np.searchsorted(st, grid, side="right") - 1, 0, sv.size - 1)
        fi = np.clip(np.searchsorted(ft, grid, side="right") - 1, 0, fq.size - 1)
        print(f"    sent {len(sends)} targets at {rate:.0f} Hz "
              f"({self._late} late resyncs), {len(frames)} frames at "
              f"{len(frames) / max(wall, 1e-9):.0f} Hz, fitting {grid.size} samples")
        self._late = 0
        return sv[gi], fq[fi], fqd[fi]

    def reset(self, q0: np.ndarray) -> None:
        """Settle at the anchor; ramp gently if the last tape left us offset."""
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
        q, _ = self._read()
        far = np.abs(q - q0)
        if np.any(far > self.FAR_FROM_REST):
            # a refusal to START is not an emergency: nothing is moving.
            j = int(np.argmax(far))
            raise SystemExit(
                f"joint {j} is {far[j]:.2f} rad from the anchor posture; "
                "re-run without --at-cell-rest to excite around the current "
                "pose, or park the arm near home first")
        self._last_target = q.copy()
        n = max(int(np.max(far) / (self.RAMP_QD * self.dt)), 1)
        deadline = time.perf_counter()
        for k in range(1, n + 1):
            self.stream_targets(q + (q0 - q) * (k / n))
            deadline = self._pace_soft(deadline)
        for _ in range(int(0.4 / self.dt)):
            self.stream_targets(q0)
            deadline = self._pace_soft(deadline)


def identify_hw(cell, adapter: "HardwareCell", inertia: np.ndarray,
                tapes_for, joints, verbose: bool = True) -> dict:
    """identify(), for the real cell: the SAME fit_joint math, fed from
    run_tape's regridded record instead of a mock stepped in lockstep."""
    labels = cell.joint_labels()
    out: dict[str, dict] = {}
    for j in (range(cell.n_joints) if joints is None else joints):
        segments = []
        for tape in tapes_for(j):
            adapter.reset(cell.rest_posture())
            segments.append(adapter.run_tape(np.asarray(tape, dtype=float), j))
        kp, kd, delay, resid = fit_joint(segments, adapter.dt,
                                         float(inertia[j]), substeps=1)
        out[labels[j]] = {
            "inertia": float(inertia[j]), "kp": round(kp, 4),
            "kd": round(kd, 4), "delay_ticks": int(delay), "residual": resid}
        if verbose:
            print(f"  {labels[j]:<24} kp={kp:10.2f}  kd={kd:8.2f}  "
                  f"delay={delay} ticks  resid={resid:.3e}")
    adapter.reset(cell.rest_posture())          # leave the cell at its anchor
    return out


def _use_current_pose_as_rest(cell, q_now: np.ndarray) -> None:
    """Anchor every excitation at where the robot actually is.

    rig_replay's rule, verbatim: a home pose baked into the cell file is the
    wrong reference for wherever the operator left the arm.  Sysid moves each
    joint a few degrees around its anchor; making the anchor the CURRENT pose
    deletes the largest motion of the whole session (the ramp to a file pose
    measured 1.94 rad away on first contact).
    """
    k = 0
    for m in cell.models:
        nm = len(m.joint_names)
        object.__setattr__(m, "rest", np.asarray(q_now[k:k + nm], dtype=np.float32))
        k += nm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--hardware", action="store_true",
        help="drive real hardware instead of the reference plant (needs adapters)",
    )
    ap.add_argument(
        "--hosts", default=os.environ.get("REMOROO_LC_HOSTS", ""),
        help="controller IPs for --hardware, in MODEL JOINT ORDER (arm1 first); "
             "defaults to $REMOROO_LC_HOSTS",
    )
    ap.add_argument("--go", action="store_true",
                    help="required to command motion (same contract as rig_replay)")
    ap.add_argument("--amplitude-scale", type=float, default=1.0,
                    help="scale every excitation amplitude; 0.5 for a first probe")
    ap.add_argument("--at-cell-rest", action="store_true",
                    help="ramp to the cell file's rest posture and excite there "
                         "(default: excite around the CURRENT pose)")
    ap.add_argument("--joints", default=None,
                    help="comma-separated joint labels to identify (default: all); "
                         "a first hardware session probes ONE joint")
    a = ap.parse_args()

    cell = load_cell(a.cell)
    joints = None
    if a.joints:
        labels = cell.joint_labels()
        want = [w.strip() for w in a.joints.split(",") if w.strip()]
        joints = []
        for w in want:
            match = [i for i, L in enumerate(labels) if L == w or L.endswith("/" + w)]
            if len(match) != 1:
                raise SystemExit(f"--joints {w!r}: matches {match} in {labels}")
            joints.append(match[0])

    if a.hardware:
        hosts = [h.strip() for h in a.hosts.split(",") if h.strip()]
        if not hosts:
            raise SystemExit("--hardware needs --hosts (or $REMOROO_LC_HOSTS)")
        if not a.go:
            raise SystemExit(
                "--hardware COMMANDS MOTION and needs --go: clear the workspace, "
                "stand at the E-stop, and pass --go deliberately"
            )
        adapter = HardwareCell(cell, hosts)
        adapter.connect()
        try:
            q_now, _ = adapter._read()
            if a.at_cell_rest:
                far = float(np.max(np.abs(q_now - cell.rest_posture())))
                print(f"  [hw] ramping to the cell rest posture "
                      f"(max delta {far:.2f} rad)")
            else:
                _use_current_pose_as_rest(cell, q_now)
                print("  [hw] excitations anchored at the CURRENT pose; "
                      f"q = {np.array2string(q_now, precision=3)}")
        except BaseException:
            adapter.disconnect()
            raise
        scale = float(np.clip(a.amplitude_scale, 0.05, 1.0))
        hz = float(cell.limits["rates"]["command_hz"])
        qd_max = cell.joint_velocity_limits()

        def hw_tapes(j: int):
            amp_c = 0.08 * scale
            # cap the chirp's top frequency so the commanded velocity never
            # exceeds a third of the joint's own limit: 2*pi*f1*A <= qd_max/3
            f1 = float(min(10.0, qd_max[j] / (3.0 * 2.0 * np.pi * amp_c)))
            print(f"  [hw] joint {j}: step 0.15*{scale:.2f} rad ramped 0.12 s, "
                  f"chirp {amp_c:.3f} rad 0.2->{f1:.1f} Hz")
            return (
                step_tape(cell, j, amplitude=0.15 * scale,
                          ramp_ticks=int(0.12 * hz)),
                chirp_tape(cell, j, amplitude=amp_c, f1=f1),
            )

        tapes_for = hw_tapes
    else:
        from remoroo_lc.adapters.mock import MockCellAdapter

        adapter = MockCellAdapter(cell)
        adapter.connect()
        tapes_for = None
    tree = KinematicTree(cell)
    by_kind = cell.gains.get("by_kind", {})
    inertia = np.asarray(
        [
            by_kind[{KIND_REVOLUTE: "revolute", KIND_PRISMATIC: "prismatic"}[int(k)]]["inertia"]
            for k in tree.joint_kind
        ],
        dtype=float,
    )

    n_id = cell.n_joints if joints is None else len(joints)
    print(f"{cell.name}: identifying {n_id} joints"
          + (" ON HARDWARE" if a.hardware else ""))
    try:
        if a.hardware:
            fits = identify_hw(cell, adapter, inertia, tapes_for, joints)
        else:
            fits = identify(cell, adapter, inertia, tapes_for=tapes_for,
                            joints=joints)
    finally:
        adapter.disconnect()

    delays = {v["delay_ticks"] for v in fits.values()}
    doc = {
        "plant_rate_hz": cell.gains.get("plant_rate_hz", 1000),
        "command_delay_ticks": int(max(delays)),
        "by_kind": by_kind,
        "per_joint": {
            k: {"inertia": v["inertia"], "kp": v["kp"], "kd": v["kd"]} for k, v in fits.items()
        },
    }
    if len(delays) > 1:
        print(
            f"  NOTE: joints disagree about the delay {sorted(delays)}; the plant "
            "model carries one delay for the whole cell, so the largest is used"
        )
    if a.out:
        a.out.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
