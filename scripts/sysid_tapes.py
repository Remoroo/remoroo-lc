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
    python scripts/sysid_tapes.py ~/cell/lc/cell.yaml --hardware --hosts <ips> --go \
        --gains configs/gains.default.yaml --out /tmp/gains_raw.yaml

The fit takes inertia as an INPUT (it identifies kp/M and kd/M, never M itself),
which is what --gains supplies; --out is where the measurement goes.  On a rig
cell the two cannot be the same file: the cell's `gains:` has to name the measured
file the control stack reads afterwards, and that file does not exist yet on the
session that is about to create it.

Self-check is the default and is what the test runs: excite the reference plant
with known gains and confirm the fit recovers them.  A fitter that has never been
shown to recover a known answer is not evidence about an unknown one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import select
import shutil
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

# Raised 8 -> 12 on 2026-10-07: the 2026-09-23 rig fit landed delays of 5-7 ticks
# (cells/corner_cell/lc/gains.yaml per_joint.delay_ticks), one under the old
# ceiling, so a slower day could have been clipped to 8 and called a fit.
# 12 = measured peak 7 + room for the TARGET_Q split to move it; the search cost
# is linear in it.  docs/2026-10-07_motor_sysid_vs_sota.md (remoroo-world) row 9.
MAX_DELAY_TICKS = 12

#: D1 cap classes (operator defaults 2026-10-07 round 2 -- OPERATOR TO CONFIRM
#: before any motion).  'sweep' (chirps, triangle, splines) and 'transit' (anchor
#: moves, reset glides) keep target acceleration under ACCEL_CAP, the SDK's
#: joint_acc_limit max (read at connect, 20 rad/s^2 on xarm-python-sdk 1.18.4);
#: 'step' and 'edge' are exempt from it and instead keep EVERY tick at or under
#: one rated-speed tick, qd_max/hz -- the bound commanderd's SlewLimiter puts on
#: what deployment sends (remoroo-world deploy/edge_student/motion.py
#: SlewLimiter.per_tick = rated speed x dt).  A contract-(A) tape names its class.
CAP_CLASSES = ("sweep", "step", "edge", "transit")
#: seconds held at the anchor after every reset glide (unchanged since first contact)
HOLD_S = 0.4
#: rad/s, the peak speed of every transit and reset glide (the ramp speed reset()
#: has used on the rig since first contact)
RAMP_QD = 0.15
#: the triangle's corners are rounded at ACCEL_CAP / CORNER_HEADROOM -- the
#: margins rule (a margin is the measured peak + 10-15%, never 2x; operator
#: 2026-10-06).  Rounded AT the cap they peaked at 20.0048 rad/s^2 in float32,
#: over it by quantisation alone; the tapered chirps already sit under it.
CORNER_HEADROOM = 1.15


def step_tape(cell, joint: int, amplitude: float = 0.15, hold_s: float = 1.2,
              ramp_ticks: int = 0) -> np.ndarray:
    """A single joint steps by `amplitude` and holds; everything else holds still.

    `ramp_ticks` spreads the rise over that many ticks.  The reference plant
    takes the pure step (which keeps the self-check exact); hardware gets a
    fast ramp, because an instant 8-degree target jump is a jerk command, and
    the fit regresses the APPLIED target sequence so a ramp costs it nothing.

    A NEGATIVE amplitude steps DOWN: a servo with friction or a gravity load
    need not answer the same way in both directions, and one direction alone
    cannot show it (2026-10-07 protocol, steps up and down).
    """
    hz = float(cell.limits["rates"]["command_hz"])
    n = int(round(hold_s * hz))
    q0 = cell.rest_posture()
    tape = np.repeat(q0[None], 2 * n, axis=0).astype(DTYPE)
    lo, hi = cell.joint_limits()
    amp = float(np.sign(amplitude)
                * np.clip(abs(amplitude), 0.0, 0.4 * (hi[joint] - lo[joint])))
    target = np.clip(q0[joint] + amp, lo[joint] + 1e-3, hi[joint] - 1e-3)
    tape[n:, joint] = target
    if ramp_ticks > 0:
        r = min(int(ramp_ticks), n)
        tape[n:n + r, joint] = np.linspace(q0[joint], target, r, endpoint=False)
    return tape


def chirp_tape(
    cell, joint: int, amplitude: float = 0.08, f0: float = 0.2, f1: float = 25.0,
    duration_s: float = 6.0, taper_s: float = 0.0,
) -> np.ndarray:
    """A linear-frequency chirp on one joint.

    The chirp is what pins kd down: a step mostly reports kp and the delay,
    because the damping only shows up in how the approach is shaped, whereas a
    sweep through the resonance reports both.

    `taper_s` raises the amplitude on a half-cosine over the first taper_s and
    lowers it over the last, so the chirp leaves and returns to the anchor with
    zero target velocity.  Untapered, both 2026-10-07 hardware chirps ended
    mid-swing at full speed and the hold after them stopped dead: 106-132
    rad/s^2 at the boundary, which a check of the bare tape never saw
    (check_tape now differences the tape WITH its boundaries).  The reference
    self-check keeps the untapered default.
    """
    hz = float(cell.limits["rates"]["command_hz"])
    n = int(round(duration_s * hz))
    t = np.arange(n) / hz
    q0 = cell.rest_posture()
    lo, hi = cell.joint_limits()
    amp = float(np.clip(amplitude, 0.0, 0.2 * (hi[joint] - lo[joint])))
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    env = np.ones(n)
    k = int(round(taper_s * hz))
    if k > 0:
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(k) / k))    # w[0] = 0
        env[:k] = w
        env[n - k:] = w[::-1]                                 # env[-1] = 0
    tape = np.repeat(q0[None], n, axis=0).astype(DTYPE)
    tape[:, joint] = np.clip(
        q0[joint] + amp * env * np.sin(phase), lo[joint] + 1e-3, hi[joint] - 1e-3
    )
    return tape


def triangle_tape(cell, joint: int, amplitude: float, speed: float,
                  reversals: int, qdd_max: float, hold_s: float = 0.5) -> np.ndarray:
    """A slow back-and-forth: 0 -> +A -> -A -> ... -> 0 at `speed`, `reversals`
    direction changes.

    This is where Coulomb friction and deadband show, and nothing else in the
    set has them: a step and a chirp are fast enough that a stiff servo hides
    a small constant force inside kd (2026-10-07 review, row 7).  The corners
    are rounded at `qdd_max` -- a raw triangle reverses its velocity inside one
    tick, which is a target acceleration of 2*speed/dt (50 rad/s^2 at 0.1 rad/s,
    250 Hz), past any rated joint acceleration.  The session builds them at
    ACCEL_CAP / CORNER_HEADROOM (1.15: the margins rule, 10-15% headroom under
    the cap), not at the cap itself (hw_tape_set).
    """
    hz = float(cell.limits["rates"]["command_hz"])
    dt = 1.0 / hz
    q0 = cell.rest_posture()
    lo, hi = cell.joint_limits()
    # the leg sequence: out to +A, then full swings, then back to the anchor
    legs = [+amplitude] + [-2.0 * amplitude * (-1) ** k for k in range(reversals - 1)]
    legs.append(-float(np.sum(legs)))
    # velocity switches at each leg boundary, each rounded by a linear ramp at
    # qdd_max CENTRED on the switch -- a centred ramp gives back exactly the
    # displacement it takes, so the legs keep their length (a causal limiter
    # overshoots every reversal by 2 v^2/a and the tape drifts off its anchor)
    t_sw = np.concatenate([[0.0], np.cumsum([abs(x) / speed for x in legs])])
    v_seg = [0.0] + [math.copysign(speed, x) for x in legs] + [0.0]
    dv = np.diff(v_seg)
    nh = int(round(hold_s * hz))
    t = np.arange(-nh, int(math.ceil(t_sw[-1] * hz)) + nh) / hz
    v = np.zeros_like(t)
    for ts, d in zip(t_sw, dv):
        tau = abs(d) / qdd_max
        v += d * np.clip((t - ts) / tau + 0.5, 0.0, 1.0)
    path = np.cumsum(v) * dt
    tape = np.repeat(q0[None], path.size, axis=0).astype(DTYPE)
    tape[:, joint] = np.clip(q0[joint] + path, lo[joint] + 1e-3, hi[joint] - 1e-3)
    return tape


def chirp_f1(qd_max: float, qdd_max: float, amplitude: float,
             ceiling: float = 10.0) -> float:
    """Top frequency of a chirp of `amplitude` that respects BOTH caps.

    Velocity: 2*pi*f*A <= qd_max/3 (the rule the 2026-09-23 session ran under,
    and why its 0.08 rad chirp stopped at 2.08 Hz).  Acceleration: (2*pi*f)^2*A
    <= qdd_max, the arm's rated joint acceleration -- the velocity rule alone
    lets a 0.02 rad chirp reach 8.3 Hz, where it would command 55 rad/s^2.
    `ceiling` is the 10 Hz cap the old hw chirp carried (sysid_tapes.py:631 at
    bc4f0c2), kept so a scaled-down probe does not run away in frequency.
    """
    f_vel = qd_max / (3.0 * 2.0 * np.pi * amplitude)
    f_acc = np.sqrt(qdd_max / amplitude) / (2.0 * np.pi)
    return float(min(ceiling, f_vel, f_acc))


def tape_peaks(tape: np.ndarray, hz: float) -> dict:
    """Per-joint peak |tick jump| (rad), |velocity| (rad/s) and |acceleration|
    (rad/s^2) of a target tape, from its first and second differences."""
    d1 = np.diff(tape, axis=0)
    d2 = np.diff(tape, n=2, axis=0)
    return {
        "jump": np.max(np.abs(d1), axis=0) if d1.size else np.zeros(tape.shape[1]),
        "vel": (np.max(np.abs(d1), axis=0) * hz) if d1.size else np.zeros(tape.shape[1]),
        "acc": (np.max(np.abs(d2), axis=0) * hz * hz) if d2.size else np.zeros(tape.shape[1]),
    }


def with_boundaries(tape: np.ndarray, anchor) -> np.ndarray:
    """The target sequence the servo actually receives around a tape: the
    anchor held before it (every tape follows a glide that ENDS in a hold
    there) and the tape's own last row held after it (the next glide starts
    from the last commanded target at zero velocity, HardwareCell._glide).  A
    tape that starts off its anchor or stops mid-motion shows the jump or the
    stop here and nowhere else."""
    a = np.asarray(anchor, dtype=np.float64).reshape(1, -1)
    t = np.asarray(tape, dtype=np.float64)
    return np.concatenate([a, a, t, t[-1:], t[-1:]])


def check_tape(name: str, cap_class: str, tape: np.ndarray, lo, hi, qd_max,
               accel_cap: float, hz: float, anchor) -> dict:
    """Refuse a tape BEFORE it moves anything, by its D1 cap class, on the tape
    WITH its boundaries (with_boundaries):

      every class      inside the joint limits with the guard's 1e-3 margin;
      sweep, transit   every tick under the e-stop guard (1.5 qd_max/hz) and
                       target acceleration under accel_cap (ACCEL_CAP);
      step, edge       every tick at or under qd_max/hz (rated speed, what
                       commanderd's SlewLimiter passes); NO acceleration cap --
                       a step is an impulse by construction and an edge tape is
                       what deployment sends.

    Returns the peaks (boundaries included), the interior acceleration and the
    caps they were held to: the session plan prints them, the raw record keeps
    them."""
    if cap_class not in CAP_CLASSES:
        raise SystemExit(f"{name}: cap_class {cap_class!r} is not one of "
                         f"{CAP_CLASSES} -- refused")
    tape = np.asarray(tape, dtype=np.float64)
    if np.any(tape < lo + 1e-3 - 1e-9) or np.any(tape > hi - 1e-3 + 1e-9):
        raise SystemExit(f"{name}: leaves the joint limits (1e-3 margin) -- refused")
    pk = tape_peaks(with_boundaries(tape, anchor), hz)
    # a float32 tape (DTYPE) cannot express a difference finer than a few ulps
    # of its values: 4 ulp(1.7 rad) = 4.8e-7 rad, * 250^2 = 0.03 rad/s^2
    ulp = 4.0 * np.spacing(np.abs(tape).max(axis=0).astype(np.float32)).astype(np.float64)
    swept = cap_class in ("sweep", "transit")
    jcap = (1.5 if swept else 1.0) * np.asarray(qd_max, dtype=np.float64) / hz
    if np.any(pk["jump"] > jcap + ulp):
        j = int(np.argmax(pk["jump"] - jcap))
        raise SystemExit(f"{name} ({cap_class}): tick jump {pk['jump'][j]:.4f} rad on "
                         f"joint {j} > cap {jcap[j]:.4f} -- refused")
    if swept and np.any(pk["acc"] > accel_cap + ulp * hz * hz):
        j = int(np.argmax(pk["acc"]))
        raise SystemExit(f"{name} ({cap_class}): peak target acceleration "
                         f"{pk['acc'][j]:.1f} rad/s^2 on joint {j} (boundaries "
                         f"included) > ACCEL_CAP {accel_cap:.1f} -- refused")
    return {"cap_class": cap_class,
            **{k: float(np.max(v)) for k, v in pk.items()},
            "acc_interior": float(np.max(tape_peaks(tape, hz)["acc"])),
            "jump_cap": float(np.min(jcap)),
            "acc_cap": float(accel_cap) if swept else None}


def min_jerk(q_from, q_to, hz: float, qd_peak: float, accel_cap: float) -> np.ndarray:
    """The glide every transit and reset uses: minimum jerk (s = 10u^3 - 15u^4
    + 6u^5) from q_from to q_to, rows k = 1..n, lasting the LONGER of what keeps
    peak speed (1.875 D/T) under qd_peak and peak acceleration (10/sqrt(3)
    D/T^2 = 5.77 D/T^2) under accel_cap.  Speed alone (the rule until
    2026-10-07) lets a short glide accelerate without bound: 0.15 rad/s over
    1 mrad is 37 rad/s^2."""
    q_from = np.asarray(q_from, dtype=np.float64).reshape(-1)
    q_to = np.asarray(q_to, dtype=np.float64).reshape(-1)
    D = float(np.max(np.abs(q_to - q_from)))
    T = max(1.875 * D / qd_peak, math.sqrt(10.0 / math.sqrt(3.0) * D / accel_cap))
    n = max(int(math.ceil(T * hz)), 1)
    u = np.arange(1, n + 1) / n
    sv = 10 * u**3 - 15 * u**4 + 6 * u**5
    return q_from + (q_to - q_from) * sv[:, None]


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
      * ONLY the excited unit is connected through the SDK (D2, operator
        default 2026-10-07): motion_enable, servo mode and servo_j all live in
        XArmUnit.connect, so every other unit is its port-30000 reader alone --
        never enabled, never in servo mode, never sent a target;
      * reset() REFUSES if any joint is more than 0.6 rad from the rest
        posture -- the operator parks the arm near home first; a blind
        joint-space lerp across the workspace is not this script's call --
        and glides there on min_jerk from the LAST COMMANDED target;
      * every streamed target is clipped to limits and guarded against
        per-tick jumps beyond 1.5x the joint's own velocity limit -- tripping
        the guard estops;
      * falling behind the 250 Hz grid for 25 consecutive ticks estops: a
        starved servo_j faults the controller in a way we did not choose.
    """

    substeps_per_tick = 1
    real_feedback = True    # port-30000 IS the controller's own 250 Hz truth
    RAMP_QD = RAMP_QD       # rad/s, peak speed of every transit / reset glide
    FAR_FROM_REST = 0.6     # rad, refuse-to-start threshold

    def __init__(self, cell, hosts: list[str], unit: int,
                 gripper: bool = False) -> None:
        from remoroo_lc.adapters.xarm import XArmUnit
        from remoroo_lc.adapters.xarm_rt import XArmRealTime

        per, hz = self._layout(cell, hosts, unit)
        sl = self.slices[unit]
        # D2: the excited unit is the ONLY XArmUnit; the others exist as readers
        self.units = {unit: XArmUnit(name=f"unit{unit}@{hosts[unit]}", host=hosts[unit],
                                     n_joints=per, command_hz=hz, has_effector=False,
                                     q_lo=self.lo[sl], q_hi=self.hi[sl],
                                     qd_max=self.qd_max[sl])}
        self.readers = [XArmRealTime(h, n_joints=per) for h in hosts]
        self.want_gripper = bool(gripper)

    def _layout(self, cell, hosts: list[str], unit: int):
        self.cell = cell
        self.hosts = list(hosts)
        #: the ONE unit this session connects, enables and streams (D2)
        self.only_unit = int(unit)
        #: the jaw's configured speed, read at connect in gripper mode (D4)
        self.gripper_speed_configured: float | None = None
        #: rated joint acceleration, rad/s^2; connect() reads it off the SDK
        self.qdd_max: float | None = None
        #: commanded-motion seconds streamed this session (ticks / command_hz)
        self.motion_s = 0.0
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
        self._last_target: np.ndarray | None = None
        self._late = 0
        return per, hz

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        # Any failure past the first handshake DISCONNECTS everything: the
        # first hardware contact of this code left two arms energised in servo
        # mode behind a crashed limits() -- never again.
        try:
            for u in self.units.values():
                u.connect()
            for r in self.readers:
                r.connect()
            for u in self.units.values():
                u.limits()   # axis count + speed ceiling, checked by the unit
            # The rated joint acceleration, from the SDK itself:
            # XArmAPI.joint_acc_limit = [min, max] rad/s^2 under is_radian
            # (xarm-python-sdk 1.18.4 x3/base.py:111 + :656, 20.0 rad/s^2).
            # The cell docs carry no joint acceleration, so this is the only
            # source; main() refuses an operator value above it.
            self.qdd_max = float(min(float(u._arm.joint_acc_limit[1])
                                     for u in self.units.values()))
            if self.want_gripper:
                # D4: deployment never passes a jaw speed (commanderd.py:397,
                # set_gripper_position(pos, wait=False)), so the jaw runs at
                # whatever its speed register holds -- read it, not assumed
                self.gripper_speed_configured = \
                    self.units[self.only_unit].gripper_speed()
            q, _ = self._read()
            print(f"  [hw] connected unit {self.only_unit} (enabled) + "
                  f"{len(self.readers)} passive 30000 reader(s); "
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
        for u in self.units.values():
            try:
                u.disconnect()
            except Exception:
                pass

    def estop(self) -> None:
        for u in self.units.values():
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
        self.units[self.only_unit].stream_targets(row[self.slices[self.only_unit]])
        self._last_target = row
        self.motion_s += self.dt

    def unit_of(self, j: int) -> int:
        return next(i for i, sl in enumerate(self.slices) if sl.start <= j < sl.stop)

    def _glide(self, q_to: np.ndarray, deadline: float) -> float:
        """min_jerk to q_to from the LAST COMMANDED target (the measured pose
        only before anything was commanded), under RAMP_QD and ACCEL_CAP.  From
        the last target, not the measured pose: the glide then starts with no
        jump and zero target velocity whatever the servo's tracking error is
        -- which is the hold check_tape assumes after every tape."""
        q_from = self._last_target
        if q_from is None:
            q_from, _ = self._read()
            self._last_target = np.asarray(q_from, dtype=np.float64).copy()
        for row in min_jerk(q_from, q_to, 1.0 / self.dt, self.RAMP_QD, self.qdd_max):
            self.stream_targets(row)
            deadline = self._pace_soft(deadline)
        return deadline

    def transit(self, q_to: np.ndarray) -> None:
        """Move to another anchor pose on the min_jerk glide (peak speed
        RAMP_QD, peak acceleration under ACCEL_CAP).  Same refuse-to-start rule
        as reset(): nothing over FAR_FROM_REST in one go, measured."""
        q_to = np.asarray(q_to, dtype=np.float64).reshape(-1)
        q, _ = self._read()
        far = np.abs(q_to - q)
        if np.any(far > self.FAR_FROM_REST):
            j = int(np.argmax(far))
            raise SystemExit(
                f"transit: joint {j} would travel {far[j]:.2f} rad "
                f"(> {self.FAR_FROM_REST}); park the arm nearer the pose first")
        self._glide(q_to, time.perf_counter())

    def play(self, tape: np.ndarray, ui: int) -> dict:
        """Play one tape on unit `ui` and return EVERYTHING it saw (contract B).

        Commands go ONLY to unit ui (the other arm holds by itself in servo
        mode -- half the SDK round-trips).  A pump thread drains that unit's
        250 Hz report the whole time and keeps every frame whole: host arrival
        time, the controller's own clock, q/qd/tau of ALL the unit's joints and
        the controller's TARGET_Q.  Until 2026-10-07 this kept one joint's q/qd
        and threw the rest away, so no recording of the real arm existed to
        check a sim against (review row 1).

        Each send is stamped BEFORE the SDK call (send_t: when the target left
        this host) and after it returns (send_t_ack: set_servo_angle_j is a
        blocking Modbus round-trip, xarm SDK 1.18.4 uxbus_cmd.py:139-151).  The
        fit regrids on send_t_ack, as it always has, so its delay stays
        comparable with the 2026-09-23 numbers."""
        sl, unit, reader = self.slices[ui], self.units[ui], self.readers[ui]
        frames: list[tuple] = []
        stop = threading.Event()

        def pump() -> None:
            while not stop.is_set():
                try:
                    smp = reader.read()
                except Exception:
                    return
                frames.append((time.perf_counter(), smp.timestamp_s, smp.q.copy(),
                               smp.qd.copy(), smp.tau.copy(), smp.q_target.copy()))

        # frames buffered during the preceding reset would enter the record
        # with arrival times far from their capture times (measured: 305-334
        # "frames/s" over a 250 Hz stream = backlog draining) -- flush first
        from remoroo_lc.adapters.xarm_rt import FRAME_BYTES
        while (len(reader._buf) >= FRAME_BYTES
               or select.select([reader._sock], [], [], 0)[0]):
            reader.read()
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        sends: list[tuple[float, float, np.ndarray]] = []
        t_start = deadline = time.perf_counter()
        try:
            for i, row in enumerate(tape):
                row = self._guard(row)
                t_send = time.perf_counter()
                unit.stream_targets(row[sl])
                sends.append((t_send, time.perf_counter(), row[sl].copy()))
                self._last_target = row
                self.motion_s += self.dt
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
        if len(frames) < 10:
            self.estop()
            raise RuntimeError("almost no feedback frames recorded -- stopped")
        print(f"    sent {len(sends)} targets at {rate:.0f} Hz "
              f"({self._late} late resyncs), {len(frames)} frames at "
              f"{len(frames) / max(wall, 1e-9):.0f} Hz")
        self._late = 0
        F = len(frames)
        return {
            "send_t": np.asarray([x[0] for x in sends]),
            "send_t_ack": np.asarray([x[1] for x in sends]),
            "send_q": np.stack([x[2] for x in sends]),
            "frame_t_host": np.asarray([f[0] for f in frames]),
            "frame_t_ctrl": np.asarray([f[1] for f in frames]),
            "q": np.stack([f[2] for f in frames]),
            "qd": np.stack([f[3] for f in frames]),
            "tau": np.stack([f[4] for f in frames]),
            "target_q": np.stack([f[5] for f in frames]),
            # offset 738 is all-zero on these controllers (xarm_rt.RtSample
            # gripper_pos_mm); NaN says "not recorded", 0.0 would say "closed"
            "gripper_pos": np.full(F, np.nan),
            "fit_exact": False,
        }

    def gripper_play(self, ui: int, open_pos: float, closed_pos: float,
                     speed: float, repeats: int) -> dict:
        """open -> close -> open, `repeats` times, nothing in the jaw; the arm
        is not streamed at all.  The jaw's position comes from the SDK's
        get_gripper_position on the tool bus (~107 Hz, the only live source:
        remoroo-world deploy/edge_student/transport/frame.py TRAP 1), each
        poll paired with the controller clock of the newest port-30000 frame.
        A move ends when the jaw is within 1% of its travel of the target and
        has stayed 0.3 s; one that has not arrived after 10 s e-stops."""
        unit, reader = self.units[ui], self.readers[ui]
        arm = unit._arm
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        latest = [float("nan")]
        stop = threading.Event()

        def pump() -> None:
            while not stop.is_set():
                try:
                    latest[0] = reader.read().timestamp_s
                except Exception:
                    return

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        sends, polls = [], []
        tol = 0.01 * abs(open_pos - closed_pos)
        try:
            for _ in range(repeats):
                for target in (closed_pos, open_pos):
                    t0 = time.perf_counter()
                    code = arm.set_gripper_position(target, wait=False, speed=speed)
                    sends.append((t0, time.perf_counter(), target))
                    if code != 0:
                        raise RuntimeError(f"set_gripper_position returned {code}")
                    arrived = None
                    while True:
                        code, pos = arm.get_gripper_position(check_baud=False)
                        th_ = time.perf_counter()
                        if code != 0:
                            raise RuntimeError(f"get_gripper_position returned {code}")
                        polls.append((th_, latest[0], float(pos)))
                        if abs(float(pos) - target) <= tol:
                            arrived = arrived or th_
                            if th_ - arrived >= 0.3:
                                break
                        elif th_ - t0 > 10.0:
                            raise RuntimeError("gripper did not arrive in 10 s -- stopped")
        except BaseException:
            self.estop()
            stop.set()
            raise
        finally:
            # leave the jaw as deployment finds it: set_gripper_position(speed=)
            # WROTE `speed` into the jaw's register (SDK 1.18.4 x3/gripper.py
            # :580-584) and commanderd, which sends none, would inherit it
            code = arm.set_gripper_speed(self.gripper_speed_configured)
            print(f"    jaw speed register restored to "
                  f"{self.gripper_speed_configured:g} r/min (code {code})")
        stop.set()
        th.join(timeout=1.0)
        return _gripper_record(sends, polls)

    def reset(self, q0: np.ndarray) -> None:
        """Settle at the anchor: the min_jerk glide from the last commanded
        target (2026-10-07: was a linear ramp from the MEASURED pose, whose
        first tick jumped by the tracking error and whose ends were velocity
        steps), then HOLD_S held there."""
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
        deadline = self._glide(q0, time.perf_counter())
        for _ in range(int(HOLD_S / self.dt)):
            self.stream_targets(q0)
            deadline = self._pace_soft(deadline)


class SimHardwareCell(HardwareCell):
    """HardwareCell's session -- same guard, reset, transit, tape set and raw
    files -- driven by the reference Plant instead of two xArm boxes: the dry
    run the 2026-10-07 protocol requires before the robot moves.

    Time is the plant's own clock, counted in integer plant substeps, so a
    frame and the next send stamped at the same instant compare EQUAL and the
    record regrids onto exactly what identify() feeds fit_joint -- the self-
    check stays exact.  TARGET_Q is the plant's delayed target (Plant.
    last_applied), the mock's stand-in for what the controller reports.
    """

    def __init__(self, cell, hosts: list[str], qdd_max: float, unit: int,
                 gripper=None, gripper_speed: float | None = None) -> None:
        self._layout(cell, hosts, unit)
        self.gripper_speed_configured = gripper_speed
        self.plant = Plant(cell)
        self.substeps_per_tick = self.plant.substeps
        self.qdd_max = float(qdd_max)
        self.gripper = gripper
        self._k = 0
        self._pending: np.ndarray | None = None
        self.estopped = False

    def _now(self, extra: int = 0) -> float:
        return (self._k + extra) / self.plant.plant_hz

    def connect(self) -> None:
        self.plant.reset(self.cell.rest_posture())
        print(f"  [dry] mock plant, {len(self.slices)} unit(s), "
              f"delay {self.plant.delay_ticks} ticks")

    def disconnect(self) -> None:
        pass

    def estop(self) -> None:
        self.estopped = True

    def _read(self):
        q, qd = self.plant.state
        return q.astype(np.float64), qd.astype(np.float64)

    def stream_targets(self, row: np.ndarray) -> None:
        row = self._guard(row)
        self._pending = row
        self._last_target = row
        self.motion_s += self.dt

    def _pace_soft(self, deadline: float) -> float:
        q, _ = self.plant.state
        self.plant.step(q if self._pending is None else self._pending)
        self._k += self.plant.substeps
        return deadline

    def play(self, tape: np.ndarray, ui: int) -> dict:
        sl, sub = self.slices[ui], self.plant.substeps
        sends, frames = [], []
        for row in tape:
            row = self._guard(row)
            t = self._now()
            sends.append((t, t, row[sl].copy()))
            self._last_target = row
            self.motion_s += self.dt
            self.plant.step(row, record=True)
            sq, sqd = self.plant.last_substeps
            for k in range(sub):
                tf = self._now(k + 1)
                frames.append((tf, tf, sq[k, sl].astype(np.float64),
                               sqd[k, sl].astype(np.float64),
                               self.plant.last_applied[sl].astype(np.float64)))
            self._k += sub
        F, n = len(frames), sl.stop - sl.start
        return {
            "send_t": np.asarray([x[0] for x in sends]),
            "send_t_ack": np.asarray([x[1] for x in sends]),
            "send_q": np.stack([x[2] for x in sends]),
            "frame_t_host": np.asarray([f[0] for f in frames]),
            "frame_t_ctrl": np.asarray([f[1] for f in frames]),
            "q": np.stack([f[2] for f in frames]),
            "qd": np.stack([f[3] for f in frames]),
            "tau": np.full((F, n), np.nan),      # the plant has no torque sensor
            "target_q": np.stack([f[4] for f in frames]),
            "gripper_pos": np.full(F, np.nan),
            "fit_exact": True,
        }

    def gripper_play(self, ui: int, open_pos: float, closed_pos: float,
                     speed: float, repeats: int) -> dict:
        g, dt = self.gripper, self.dt
        tol = 0.01 * abs(open_pos - closed_pos)
        sends, polls, t = [], [], self._now()
        for _ in range(repeats):
            for target in (closed_pos, open_pos):
                sends.append((t, t, target))
                g.command(t, target, speed)
                arrived = None
                t0 = t
                while True:
                    t += dt
                    pos = g.advance(t, dt)
                    polls.append((t, t, pos))
                    if abs(pos - target) <= tol:
                        arrived = arrived if arrived is not None else t
                        if t - arrived >= 0.3:
                            break
                    elif t - t0 > 10.0:
                        raise RuntimeError("mock gripper did not arrive in 10 s")
        self.motion_s += t - self._now()
        self._k = int(round(t * self.plant.plant_hz))
        return _gripper_record(sends, polls)


def _gripper_record(sends, polls) -> dict:
    """Contract-B arrays for a gripper session: one 'joint' (the drive), send_q
    = commanded jaw position, q = gripper_pos = the jaw's reported position,
    both in the SDK's own units (0..850 on the xArm gripper)."""
    F = len(polls)
    pos = np.asarray([x[2] for x in polls], dtype=float)
    return {
        "send_t": np.asarray([x[0] for x in sends]),
        "send_t_ack": np.asarray([x[1] for x in sends]),
        "send_q": np.asarray([[x[2]] for x in sends], dtype=float),
        "frame_t_host": np.asarray([x[0] for x in polls]),
        "frame_t_ctrl": np.asarray([x[1] for x in polls]),
        "q": pos[:, None].copy(),
        "qd": np.full((F, 1), np.nan),
        "tau": np.full((F, 1), np.nan),
        "target_q": np.full((F, 1), np.nan),
        "gripper_pos": pos,
        "fit_exact": False,
    }


def fit_gripper(rec: dict) -> dict:
    """Delay + rate of a rate-limited jaw, per commanded move, from a raw record.

    rate = least-squares slope of position against time between 10% and 90%
    of the travel (units/s); delay = where that line leaves the start position,
    minus the send.  A first-crossing threshold instead reads the time the jaw
    needs to cover the threshold as delay (2% of 850 at 300 units/s = 57 ms,
    measured on the mock).  Medians over the moves."""
    st, sv = rec["send_t"], rec["send_q"][:, 0]
    t, pos = rec["frame_t_host"], rec["gripper_pos"]
    delays, rates = [], []
    for i in range(st.size):
        w = (t > st[i]) & (t < (st[i + 1] if i + 1 < st.size else np.inf))
        tw, pw = t[w], pos[w]
        if tw.size < 3:
            continue
        p0 = float(pos[np.searchsorted(t, st[i]) - 1]) if np.searchsorted(t, st[i]) else float(pw[0])
        travel = float(sv[i]) - p0
        if abs(travel) < 1e-9:
            continue
        frac = (pw - p0) / travel
        mid = (frac >= 0.1) & (frac <= 0.9)
        if mid.sum() >= 2:
            slope, icpt = np.polyfit(tw[mid] - st[i], pw[mid], 1)
            rates.append(abs(float(slope)))
            delays.append(float((p0 - icpt) / slope))
    return {"delay_s": float(np.median(delays)) if delays else float("nan"),
            "rate_units_per_s": float(np.median(rates)) if rates else float("nan"),
            "moves": len(delays)}


def regrid(rec: dict, k: int, dt: float):
    """Contract-B record -> (applied, q, qd) for joint k of the unit, the shape
    fit_joint takes.  Both event streams are sample-and-held onto the exact
    command grid, so the fit sees a uniform dt regardless of send-loop jitter,
    and the constant transport offset lands where it belongs: in the fitted
    delay.  The mock's frames already ARE the grid (one per plant substep); a
    frame stamped at the same instant as a send belongs to the PREVIOUS target,
    which side='left' gives exactly."""
    st, sv = rec["send_t_ack"], rec["send_q"][:, k]
    ft, fq, fqd = rec["frame_t_host"], rec["q"][:, k], rec["qd"][:, k]
    if rec["fit_exact"]:
        gi = np.searchsorted(st, ft, side="left") - 1
        keep = gi >= 0
        return sv[gi[keep]], fq[keep], fqd[keep]
    t0, t1 = max(st[0], ft[0]), min(st[-1], ft[-1])
    grid = np.arange(t0, t1, dt)
    gi = np.clip(np.searchsorted(st, grid, side="right") - 1, 0, sv.size - 1)
    fi = np.clip(np.searchsorted(ft, grid, side="right") - 1, 0, fq.size - 1)
    return sv[gi], fq[fi], fqd[fi]


def _git_hash() -> str:
    try:
        h = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception as exc:  # noqa: BLE001 -- provenance, not control
        return f"unknown ({exc})"


def write_raw(path: Path, rec: dict, *, tape: str, params: dict, joints, excited,
              command_hz: float, start_pose, git_hash: str, utc: str,
              unit_host: str) -> Path:
    """One tape's raw record, contract (B), written BEFORE any regrid or fit
    touches it.  Extra key: send_t_ack (see HardwareCell.play)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, tape=np.array(tape), params_json=np.array(json.dumps(params)),
        joints=np.array([str(x) for x in joints]),
        excited=np.array([str(x) for x in excited]),
        command_hz=np.float64(command_hz),
        start_pose=np.asarray(start_pose, dtype=np.float64),
        send_t=rec["send_t"], send_q=rec["send_q"],
        frame_t_host=rec["frame_t_host"], frame_t_ctrl=rec["frame_t_ctrl"],
        q=rec["q"], qd=rec["qd"], tau=rec["tau"], target_q=rec["target_q"],
        gripper_pos=rec["gripper_pos"], git_hash=np.array(git_hash),
        utc=np.array(utc), unit_host=np.array(unit_host),
        send_t_ack=rec["send_t_ack"])
    return path


def _match_unit_joints(cell, sl: slice, names, what: str) -> None:
    labels = cell.joint_labels()[sl]
    names = [str(x) for x in names]
    ok = len(names) == len(labels) and all(
        L == n or L.endswith("/" + n) for L, n in zip(labels, names))
    if not ok:
        raise SystemExit(f"{what}: joints {names} are not this unit's {labels} "
                         "in cell order -- refused")


def load_poses(path: Path, cell, sl: slice) -> list[dict]:
    """poses.yaml -> [{label, role, q}] for the unit's joints, in file order.

    Accepted shape (written by remoroo-world's tape builder,
    cells/<cell>/lc/sysid_tapes/poses.yaml): an optional top-level `joints:`
    list and a `poses:` mapping (or the mapping at top level) of
    label -> {q|joints|values: [...], role: fit|heldout}.  Joint names are
    checked against the unit, never assumed."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    names = doc.get("joints")
    body = doc.get("poses", {k: v for k, v in doc.items() if isinstance(v, dict)})
    lo, hi = cell.joint_limits()
    out = []
    for label, ent in body.items():
        q = None
        for key in ("q", "values", "joints"):
            v = ent.get(key)
            if isinstance(v, dict):
                names_here, q = list(v.keys()), list(v.values())
                _match_unit_joints(cell, sl, names_here, f"poses {label}")
                break
            if isinstance(v, list) and v and not isinstance(v[0], str):
                q = v
                break
        if q is None:
            raise SystemExit(f"poses {label}: no joint values (q/values/joints)")
        if names is not None:
            _match_unit_joints(cell, sl, names, f"poses {label}")
        q = np.asarray(q, dtype=np.float64)
        role = str(ent.get("role", ""))
        if role not in ("fit", "heldout"):
            raise SystemExit(f"poses {label}: role {role!r} is not fit/heldout")
        if q.size != sl.stop - sl.start or np.any(q < lo[sl] + 1e-3) or np.any(q > hi[sl] - 1e-3):
            raise SystemExit(f"poses {label}: {q.size} values or outside the limits")
        out.append({"label": str(label), "role": role, "q": q})
    return out


def load_replay(path: Path, cell, sl: slice, hz: float) -> dict:
    """A contract-(A) replay tape, refused unless it is this unit's joints in
    cell order at this cell's command rate, starts at its own start_pose and
    names its D1 cap class (cap_class, one of CAP_CLASSES)."""
    z = np.load(path, allow_pickle=False)
    _match_unit_joints(cell, sl, z["joints"], f"replay {path.name}")
    if abs(float(z["hz"]) - hz) > 1e-9:
        raise SystemExit(f"replay {path.name}: hz {float(z['hz'])} != command_hz {hz}")
    qt = np.asarray(z["q_target"], dtype=np.float64)
    sp = np.asarray(z["start_pose"], dtype=np.float64)
    if qt.ndim != 2 or qt.shape[1] != sl.stop - sl.start:
        raise SystemExit(f"replay {path.name}: q_target shape {qt.shape}")
    if np.max(np.abs(qt[0] - sp)) > 1e-6:
        raise SystemExit(f"replay {path.name}: q_target[0] is not start_pose")
    if "cap_class" not in z.files or str(z["cap_class"]) not in CAP_CLASSES:
        raise SystemExit(f"replay {path.name}: cap_class "
                         f"{str(z['cap_class']) if 'cap_class' in z.files else None!r}"
                         f" is not one of {CAP_CLASSES} (contract A) -- refused")
    return {"name": str(z["name"]), "source": str(z["source"]),
            "pose_label": str(z["pose_label"]), "start_pose": sp, "q_target": qt,
            "cap_class": str(z["cap_class"]), "path": str(path)}


def hw_tape_set(cell, j: int, scale: float, qd_max, accel_cap: float,
                repeat: bool) -> list[tuple[str, str, np.ndarray, dict]]:
    """The 2026-10-07 per-joint set (review: Arm-1 session protocol), as
    (name, cap_class, tape, params).  Every number that is a limit comes from
    the cell or the SDK; the amplitudes, speeds and durations are the
    protocol's; the classes are D1's; the triangle's corners sit CORNER_HEADROOM
    under ACCEL_CAP."""
    hz = float(cell.limits["rates"]["command_hz"])
    rated = float(qd_max[j]) / hz         # one tick at rated speed
    out = []
    for amp in (0.03, 0.15):
        a = amp * scale
        # D1 'step': big = the 2026-09-23 proven shape, a 0.12 s linear ramp
        # (0.15 rad -> 0.005 rad/tick); small = a ramp of no tick above
        # qd_max/hz (0.03 rad at 3.14 rad/s, 250 Hz -> 3 ticks of 0.010 rad)
        r = int(0.12 * hz) if amp >= 0.15 else int(math.ceil(a / rated))
        for sign, tag in ((+1, "up"), (-1, "down")):
            out.append((f"step_{tag}_{amp:g}", "step",
                        step_tape(cell, j, amplitude=sign * a, ramp_ticks=r),
                        {"kind": "step", "cap_class": "step", "amplitude": sign * a,
                         "ramp_ticks": r, "hold_s": 1.2}))
    corner = accel_cap / CORNER_HEADROOM          # the margins rule, not the cap
    out.append(("triangle_0.05", "sweep",
                triangle_tape(cell, j, 0.05 * scale, 0.1, 3, corner),
                {"kind": "triangle", "cap_class": "sweep", "amplitude": 0.05 * scale,
                 "speed": 0.1, "reversals": 3, "corner_accel": corner}))
    for amp, dur in ((0.08, 6.0), (0.02, 12.0)):
        a = amp * scale
        f1 = chirp_f1(float(qd_max[j]), accel_cap, a)
        pr = {"kind": "chirp", "cap_class": "sweep", "amplitude": a, "f0": 0.2,
              "f1": f1, "duration_s": dur, "taper_s": 0.5}
        tape = chirp_tape(cell, j, amplitude=a, f1=f1, duration_s=dur, taper_s=0.5)
        out.append((f"chirp_{amp:g}", "sweep", tape, pr))
        if amp == 0.02 and repeat:
            out.append((f"chirp_{amp:g}_repeat", "sweep", tape, dict(pr, repeat=True)))
    return out


def build_pose_plan(cell, q_now: np.ndarray, sl: slice, anchors: list[dict], joints,
                    scale: float, qd_max, accel_cap: float) -> list[dict]:
    """Every per-joint tape of the session, per anchor, built and checked by
    its cap class (boundaries included) BEFORE anything moves.  identify_hw
    plays exactly these arrays; plan_rows prices them."""
    hz = float(cell.limits["rates"]["command_hz"])
    lo, hi = cell.joint_limits()
    plan = []
    for pi, anc in enumerate(anchors):
        full = np.asarray(q_now, dtype=np.float64).copy()
        if anc["q"] is not None:
            full[sl] = anc["q"]
        _use_current_pose_as_rest(cell, full)
        full = cell.rest_posture().astype(np.float64)     # the tapes' own anchor
        tapes = []
        for j in joints:
            for name, cls, tape, params in hw_tape_set(cell, j, scale, qd_max,
                                                       accel_cap, repeat=pi == 0):
                tape = np.asarray(tape, dtype=np.float64)
                pk = check_tape(name, cls, tape, lo, hi, qd_max, accel_cap, hz, full)
                tapes.append({"joint": j, "name": name, "cap_class": cls,
                              "tape": tape, "params": params, "peaks": pk})
        plan.append(dict(anc, full=full, tapes=tapes))
    return plan


def plan_rows(cell, plan: list[dict], q_start: np.ndarray, accel_cap: float, *,
              lead: dict | None = None, home: np.ndarray | None = None) -> list[dict]:
    """The SESSION PLAN: every motion in the order it will run -- the lead
    transit tape, each anchor glide, each tape with the reset glide + hold in
    front of it, the final settle and the return home -- with its duration,
    class and peaks against caps.  Glides are priced with the SAME min_jerk
    the adapter streams, so the total is the motion the session will command."""
    hz = float(cell.limits["rates"]["command_hz"])
    lo, hi = cell.joint_limits()
    qd_max = cell.joint_velocity_limits()
    hold = int(HOLD_S * hz)
    rows = []

    def glide(label, pose, last, to):
        g = min_jerk(last, to, hz, RAMP_QD, accel_cap)
        rows.append({"pose": pose, "tape": label, "cap_class": "transit",
                     "ticks": len(g), "glide_ticks": 0,
                     "peaks": check_tape(label, "transit", g, lo, hi, qd_max,
                                         accel_cap, hz, last)})
        return g[-1]

    last = np.asarray(q_start, dtype=np.float64)
    if lead is not None:
        last = glide(f"glide -> {lead['name']} start", lead["pose_label"], last,
                     lead["full"][0])
        rows.append({"pose": lead["pose_label"], "tape": lead["name"],
                     "cap_class": lead["cap_class"], "ticks": len(lead["full"]),
                     "glide_ticks": 0, "peaks": lead["peaks"]})
        last = lead["full"][-1]
    for anc in plan:
        full = anc["full"]
        if anc["q"] is not None:
            last = glide(f"transit -> {anc['label']}", anc["label"], last, full)
        for t in anc["tapes"]:
            gl = len(min_jerk(last, full, hz, RAMP_QD, accel_cap)) + hold
            rows.append({"pose": anc["label"], "joint": cell.joint_labels()[t["joint"]],
                         "tape": t["name"], "cap_class": t["cap_class"],
                         "ticks": len(t["tape"]), "glide_ticks": gl, "peaks": t["peaks"]})
            last = t["tape"][-1]
        rows.append({"pose": anc["label"], "tape": "settle", "cap_class": "transit",
                     "ticks": 0,
                     "glide_ticks": len(min_jerk(last, full, hz, RAMP_QD, accel_cap)) + hold,
                     "peaks": None})
        last = full
    if home is not None:
        glide("return -> P0", "P0", last, home)
    return rows


def print_plan(rows: list[dict], hz: float, *, ui: int, n_units: int,
               accel_cap: float, start: str, extra: str = "") -> float:
    """Print the plan; returns the total commanded motion in seconds."""
    total = sum(r["ticks"] + r["glide_ticks"] for r in rows) / hz
    print("SESSION PLAN (operator defaults D1-D4 of 2026-10-07 -- OPERATOR TO "
          "CONFIRM before --go)")
    print(f"  D1 caps: sweep/transit accel <= ACCEL_CAP {accel_cap:.2f} rad/s^2 "
          "(SDK joint_acc_limit max); step/edge every tick <= qd_max/hz, no accel cap")
    # edge has no accel cap, so its peak is what the operator confirms instead
    edge = [r["peaks"]["acc"] for r in rows if r["cap_class"] == "edge"]
    print("     edge-class tapes: " + (
        f"{len(edge)}, peak target accel {max(edge):.2f} rad/s^2 (boundaries "
        "included, uncapped) -- OPERATOR TO CONFIRM" if edge else "none in this plan"))
    print(f"  D2 only unit {ui} is enabled + streamed; every other unit is a "
          "passive port-30000 reader")
    print(f"  D3 starts at {start}" + (extra and f"; {extra}"))
    others = ", ".join(f"arm {k + 1}" for k in range(n_units) if k != ui)
    print("  CHECK: " + "; ".join(["carton at nominal position or removed"]
                                  + ([f"{others} at rig home or lc rest"] if others else [])
                                  + ["e-stop in hand"]))
    print(f"  {'#':>3} {'pose':<6} {'joint':<14} {'tape':<26} {'class':<8} "
          f"{'tape_s':>7} {'+glide_s':>8}  {'jump/cap rad':>15}  {'acc/cap rad/s^2':>17}")
    for i, r in enumerate(rows):
        p = r["peaks"]
        jc = (f"{p['jump']:.4f}/{p['jump_cap']:.4f}" if p else "")
        ac = (f"{p['acc']:.2f}/" + (f"{p['acc_cap']:.1f}" if p["acc_cap"] is not None
                                    else "none") if p else "")
        print(f"  {i:>3} {r['pose']:<6} {r.get('joint', ''):<14} {r['tape']:<26} "
              f"{r['cap_class']:<8} {r['ticks'] / hz:>7.2f} {r['glide_ticks'] / hz:>8.2f}"
              f"  {jc:>15}  {ac:>17}")
    by = {}
    for r in rows:
        if r["peaks"]:
            c = by.setdefault(r["cap_class"], {"jump": 0.0, "acc": 0.0, "n": 0,
                                               "jump_cap": r["peaks"]["jump_cap"],
                                               "acc_cap": r["peaks"]["acc_cap"]})
            c["jump"] = max(c["jump"], r["peaks"]["jump"])
            c["acc"] = max(c["acc"], r["peaks"]["acc"])
            c["n"] += 1
    for k, c in by.items():
        print(f"  class {k:<8} x{c['n']:<4} peak jump {c['jump']:.4f} (cap "
              f"{c['jump_cap']:.4f})  peak accel incl. boundaries {c['acc']:.2f} "
              f"(cap {'none' if c['acc_cap'] is None else format(c['acc_cap'], '.1f')})")
    print(f"  TOTAL commanded motion {total:.1f} s = {total / 60.0:.2f} min")
    return total


def session_start(q_unit: np.ndarray, p0: np.ndarray, rest: np.ndarray,
                  tol: np.ndarray) -> str:
    """D3: a --poses session starts ONLY at the training home P0 or at the
    cell's rest pose (where the arm parks), every joint within `tol` -- one
    rated-speed tick, qd_max/hz, the most commanderd's own slew lets a single
    command move a joint.  'P0' or 'rest'; anything else is refused."""
    if np.all(np.abs(q_unit - p0) <= tol):
        return "P0"
    if np.all(np.abs(q_unit - rest) <= tol):
        return "rest"
    raise SystemExit(
        f"--poses: the arm is at neither P0 {np.round(p0, 3).tolist()} nor the cell "
        f"rest pose {np.round(rest, 3).tolist()} (tolerance {float(np.min(tol)):.4f} "
        f"rad/joint); measured {np.round(q_unit, 3).tolist()} -- park it at one "
        "of them first")


def unit_jaw_joints(cell, sl: slice) -> tuple[str, list[str]]:
    """(model name, jaw drive joint(s)) of the unit that owns joints `sl`, per
    the cell: locked joints of its model that hang below the unit's LAST arm
    joint in the URDF and are not mimics (a parallel jaw is one drive plus
    mimics).  A jaw is named by its bare joint name or by its model-qualified
    label, '<model>/<joint>' (the form cell.joint_labels() uses)."""
    from remoroo_lc.urdf import load_urdf

    k = 0
    for m in cell.models:
        nm = len(m.joint_names)
        if k <= sl.start < k + nm:
            break
        k += nm
    urdf = load_urdf(m.urdf_path, name=m.name)
    links, out = [urdf.by_name[m.joint_names[sl.stop - 1 - k]].child], []
    while links:
        ln = links.pop()
        for c in urdf.children.get(ln, []):
            jn = urdf.parent_joint[c]
            if jn in m.locked and urdf.by_name[jn].mimic is None:
                out.append(jn)
            links.append(c)
    return m.name, out


def identify_hw(cell, adapter: "HardwareCell", inertia: np.ndarray, *, ui: int,
                joints, plan: list[dict], raw_dir: Path, utc: str,
                home: np.ndarray | None = None, verbose: bool = True) -> tuple[dict, dict]:
    """The per-joint session: for each anchor of the plan (build_pose_plan),
    each joint of unit ui, the whole tape set -- raw record written per tape
    BEFORE the fit -- then the SAME fit_joint math over every tape recorded at
    a 'fit' pose.  Tapes at a 'heldout' pose are recorded and never fitted.
    With `home`, ends with a transit there (D3: back to P0, not rest).
    Returns (fits, provenance)."""
    labels = cell.joint_labels()
    sl = adapter.slices[ui]
    hz = float(cell.limits["rates"]["command_hz"])
    sub = int(adapter.substeps_per_tick)
    dt_fit = adapter.dt / sub
    gh = _git_hash()
    segments: dict[int, list] = {j: [] for j in joints}
    prov = {"tape_pose": {}, "tapes": [], "raw_files": []}
    for anc in plan:
        full = anc["full"]
        if anc["q"] is not None:
            adapter.transit(full)
        q_at, _ = adapter._read()
        prov["tape_pose"][anc["label"]] = {
            "role": anc["role"],
            "q": {labels[k]: round(float(q_at[k]), 6) for k in range(sl.start, sl.stop)}}
        if verbose:
            print(f"  [{anc['label']}] ({anc['role']}) q = "
                  f"{np.array2string(q_at[sl], precision=3)}")
        for t in anc["tapes"]:
            j, name = t["joint"], t["name"]
            adapter.reset(full)
            rec = adapter.play(t["tape"], ui)
            path = raw_dir / f"unit{ui}_{anc['label']}_{labels[j].split('/')[-1]}_{name}.npz"
            write_raw(path, rec, tape=name,
                      params=dict(t["params"], pose_label=anc["label"], role=anc["role"],
                                  peaks=t["peaks"]),
                      joints=labels[sl], excited=[labels[j]], command_hz=hz,
                      start_pose=full[sl], git_hash=gh, utc=utc,
                      unit_host=adapter.hosts[ui])
            prov["tapes"].append({"pose": anc["label"], "joint": labels[j],
                                  "tape": name, "params": t["params"]})
            prov["raw_files"].append(str(path))
            if anc["role"] == "fit":
                segments[j].append(regrid(rec, j - sl.start, dt_fit))
        adapter.reset(full)                      # settle at the anchor
    if home is not None:
        adapter.transit(home)
        print(f"  back at P0: q = {np.array2string(adapter._read()[0][sl], precision=3)}")
    out: dict[str, dict] = {}
    for j in joints:
        kp, kd, delay, resid = fit_joint(segments[j], dt_fit, float(inertia[j]),
                                         substeps=sub)
        out[labels[j]] = {
            "inertia": float(inertia[j]), "kp": round(kp, 4),
            "kd": round(kd, 4), "delay_ticks": int(delay), "residual": resid}
        if verbose:
            print(f"  {labels[j]:<24} kp={kp:10.2f}  kd={kd:8.2f}  "
                  f"delay={delay} ticks  resid={resid:.3e}")
    return out, prov


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


def main(argv=None) -> int:
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
    ap.add_argument(
        "--gains", type=Path, default=None,
        help="gains yaml supplying the fit's INPUT -- the by_kind inertia prior -- "
             "overriding whatever the cell file declares.  NOT the output: the "
             "measurement goes to --out.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="run the WHOLE hardware session (tapes, poses, replay, "
                         "gripper, raw files) against the mock plant; --hosts "
                         "names the mock units, nothing connects")
    ap.add_argument("--unit", type=int, default=0,
                    help="index into --hosts of the ONE arm this session moves "
                         "(default 0 = the first host = arm 1); the others are "
                         "never streamed")
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="where every tape's raw .npz goes (default: "
                         "<--out's dir>/sysid_raw/<utc>)")
    ap.add_argument("--poses", type=Path, default=None,
                    help="poses.yaml: run the per-joint set at each pose "
                         "(role fit|heldout).  Starts only at its P0 or at the "
                         "cell rest pose (from rest it first plays "
                         "transit_rest_P0.npz beside it); ends back at P0")
    ap.add_argument("--replay", type=Path, nargs="+", default=None,
                    help="contract-(A) .npz tapes to stream on --unit, all "
                         "joints at once; recorded, not fitted")
    ap.add_argument("--gripper", default=None, metavar="DRIVE_JOINT",
                    help="gripper mode: open->close->open on --unit's jaw, at the "
                         "jaw's configured speed (read at connect) and at the "
                         "SDK maximum; must name --unit's jaw drive joint")
    ap.add_argument("--gripper-repeats", type=int, default=5)
    ap.add_argument("--qdd-max", type=float, default=None,
                    help="rated joint acceleration rad/s^2. --hardware reads the "
                         "SDK's joint_acc_limit and refuses a value above it; "
                         "--dry-run has no SDK to ask and requires this")
    ap.add_argument("--mock-gripper", default=None,
                    help="--dry-run only: 'delay_s,rate_per_speed,configured_speed' "
                         "of the mock jaw")
    a = ap.parse_args(argv)

    # ⚠ --gains is the fit's INPUT and it exists to break a chicken-and-egg.  A
    # rig cell's `gains:` must name the MEASURED file (cells/<id>/lc/gains.yaml --
    # that is what the control stack reads after this session), but load_cell
    # opens whatever `gains:` names eagerly, so before the first session the cell
    # this session is FOR cannot be loaded at all: FileNotFoundError, zero joints
    # measured, and the operator standing at the E-stop for nothing.  Passing
    # gains_path short-circuits that branch in load_cell, which is why the
    # override works when the cell's own target is ABSENT -- the path is never
    # opened.  The fit needs only inertia out of this file (see the by_kind lookup
    # below); kp and kd in it are the reference plant's business, not the rig's.
    cell = load_cell(a.cell, gains_path=a.gains)
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

    if a.hardware or a.dry_run:
        return _session(a, cell, joints)
    from remoroo_lc.adapters.mock import MockCellAdapter

    adapter = MockCellAdapter(cell)
    adapter.connect()
    try:
        tree = KinematicTree(cell)
        by_kind = cell.gains.get("by_kind", {})
        inertia = _inertia_of(cell, tree, by_kind)
        n_id = cell.n_joints if joints is None else len(joints)
        print(f"{cell.name}: identifying {n_id} joints")
        fits = identify(cell, adapter, inertia, joints=joints)
    finally:
        adapter.disconnect()
    _write_gains(a, cell, by_kind, fits, extra={})
    return 0


def _inertia_of(cell, tree, by_kind) -> np.ndarray:
    return np.asarray(
        [
            by_kind[{KIND_REVOLUTE: "revolute", KIND_PRISMATIC: "prismatic"}[int(k)]]["inertia"]
            for k in tree.joint_kind
        ],
        dtype=float,
    )


def _session(a, cell, joints) -> int:
    """--hardware / --dry-run: one arm, one mode per invocation (per-joint
    tapes, --replay, or --gripper), every tape's raw stream saved."""
    from datetime import datetime, timezone

    hosts = [h.strip() for h in a.hosts.split(",") if h.strip()]
    if not hosts:
        raise SystemExit("--hardware/--dry-run need --hosts (or $REMOROO_LC_HOSTS)")
    if a.hardware and not a.go:
        raise SystemExit(
            "--hardware COMMANDS MOTION and needs --go: clear the workspace, "
            "stand at the E-stop, and pass --go deliberately"
        )
    if not 0 <= a.unit < len(hosts):
        raise SystemExit(f"--unit {a.unit}: only {len(hosts)} host(s)")
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # the raw streams ARE the measurement: with neither flag they still land
    # somewhere (the working directory), never nowhere
    raw_dir = Path(a.raw_dir or ((a.out.parent if a.out else Path.cwd())
                                 / "sysid_raw" / utc))
    if a.gripper:
        n_per = cell.n_joints // len(hosts)
        model, jaws = unit_jaw_joints(cell, slice(a.unit * n_per, (a.unit + 1) * n_per))
        named = jaws + [f"{model}/{jn}" for jn in jaws]
        if a.gripper not in named:
            raise SystemExit(f"--gripper {a.gripper!r} is not unit {a.unit}'s jaw "
                             f"drive joint per the cell ({named}) -- refused")
    if a.hardware:
        adapter = HardwareCell(cell, hosts, a.unit, gripper=bool(a.gripper))
    else:
        if a.qdd_max is None:
            raise SystemExit("--dry-run needs --qdd-max: the mock has no SDK "
                             "joint_acc_limit to read")
        grip, sp0 = None, None
        if a.gripper:
            from remoroo_lc.plant import GripperPlant
            from remoroo_lc.adapters.xarm import _GRIPPER_OPEN

            d, r, sp0 = (float(x) for x in (a.mock_gripper or "").split(","))
            grip = GripperPlant(delay_s=d, rate_per_speed=r, pos0=float(_GRIPPER_OPEN))
        adapter = SimHardwareCell(cell, hosts, a.qdd_max, a.unit, gripper=grip,
                                  gripper_speed=sp0)

    ui = a.unit
    hz = float(cell.limits["rates"]["command_hz"])
    labels = cell.joint_labels()
    fits, extra = None, {}

    adapter.connect()
    # ⚠ EVERYTHING between connect() and this finally has to stay inside it.
    # This try used to begin below, and the setup in between -- the amplitude
    # clip, joint_velocity_limits(), KinematicTree(cell), the by_kind/inertia
    # lookup and the header print -- ran with NO handler at all.  A CellSpecError
    # out of joint_velocity_limits(), a kinematics failure, or a
    # `KeyError: 'revolute'` from a gains file with no by_kind table therefore
    # exited main() with both controllers still energised in servo mode with
    # motion enabled and nothing cleaning up after them: exactly the fault
    # connect() (~line 287) says "never again" about, reintroduced 300 lines
    # later in the caller.  What was missing is the cleanup, not the guards, so
    # not one guard changed -- they raise inside the protection now.
    try:
        scale = float(np.clip(a.amplitude_scale, 0.05, 1.0))
        qd_max = cell.joint_velocity_limits()
        tree = KinematicTree(cell)
        by_kind = cell.gains.get("by_kind", {})
        inertia = _inertia_of(cell, tree, by_kind)
        sl = adapter.slices[ui]
        if joints is None:
            joints = list(range(sl.start, sl.stop))
        if any(not sl.start <= j < sl.stop for j in joints):
            raise SystemExit(f"--joints must all belong to --unit {ui} "
                             f"({labels[sl]}): one arm per session")
        if a.hardware and a.qdd_max is not None:
            if a.qdd_max > adapter.qdd_max:
                raise SystemExit(f"--qdd-max {a.qdd_max} exceeds the SDK's "
                                 f"joint_acc_limit {adapter.qdd_max}")
            adapter.qdd_max = float(a.qdd_max)
        print(f"  ACCEL_CAP (rated joint acceleration) {adapter.qdd_max:.2f} rad/s^2 "
              "-- OPERATOR TO CONFIRM")
        q_now, _ = adapter._read()
        rest_file = cell.rest_posture().astype(np.float64).copy()
        # excitations anchor at the CURRENT pose unless a pose is named
        _use_current_pose_as_rest(cell, q_now)
        gh = _git_hash()

        if a.gripper:
            from remoroo_lc.adapters.xarm import (_GRIPPER_CLOSED, _GRIPPER_OPEN,
                                                  _GRIPPER_SPEED_MAX)

            # D4: what deployment runs the jaw at, then the SDK's top speed
            sp_cfg = float(adapter.gripper_speed_configured)
            speeds = list(dict.fromkeys([sp_cfg, float(_GRIPPER_SPEED_MAX)]))
            print(f"{cell.name}: gripper {a.gripper} on unit {ui}"
                  + (" ON HARDWARE" if a.hardware else " (dry run)")
                  + f"; speeds {speeds} r/min (configured {sp_cfg:g}, SDK max "
                  f"{_GRIPPER_SPEED_MAX} -- OPERATOR TO CONFIRM)")
            extra["gripper"] = {}
            for sp in speeds:
                rec = adapter.gripper_play(ui, float(_GRIPPER_OPEN),
                                           float(_GRIPPER_CLOSED), sp,
                                           a.gripper_repeats)
                path = write_raw(
                    raw_dir / f"unit{ui}_gripper_speed{sp:g}.npz", rec,
                    tape=f"gripper_speed{sp:g}",
                    params={"kind": "gripper", "speed": sp,
                            "speed_source": ("configured" if sp == sp_cfg
                                             else "sdk_max"),
                            "speed_configured": sp_cfg,
                            "speed_max": float(_GRIPPER_SPEED_MAX),
                            "repeats": a.gripper_repeats,
                            "open": _GRIPPER_OPEN, "closed": _GRIPPER_CLOSED,
                            "units": "xArm SDK gripper units",
                            "pose_label": "current", "role": "fit"},
                    joints=[a.gripper], excited=[a.gripper], command_hz=hz,
                    start_pose=[_GRIPPER_OPEN], git_hash=gh, utc=utc,
                    unit_host=hosts[ui])
                g = fit_gripper(rec)
                extra["gripper"][str(sp)] = dict(g, raw=str(path))
                print(f"  speed {sp:g}: delay {g['delay_s'] * 1e3:.1f} ms, "
                      f"rate {g['rate_units_per_s']:.1f} units/s over "
                      f"{g['moves']} moves -> {path}")
        elif a.replay:
            print(f"{cell.name}: {len(a.replay)} replay tape(s) on unit {ui}"
                  + (" ON HARDWARE" if a.hardware else " (dry run)"))
            tapes = [load_replay(pth, cell, sl, hz) for pth in a.replay]
            for t in tapes:          # check ALL of them before ANYTHING moves
                _prep_replay(t, cell, q_now, sl, adapter.qdd_max)
            rows, last = [], q_now
            for t in tapes:
                rows += plan_rows(cell, [], last, adapter.qdd_max, lead=t)
                last = t["full"][-1]
            print_plan(rows, hz, ui=ui, n_units=len(hosts), accel_cap=adapter.qdd_max,
                       start="current pose")
            extra["raw_files"] = [
                str(_play_replay(adapter, t, ui, raw_dir, cell, hz, gh, utc))
                for t in tapes]
        else:
            if a.poses:
                anchors = load_poses(a.poses, cell, sl)
                p0 = next((x for x in anchors if x["label"] == "P0"), None)
                if p0 is None:
                    raise SystemExit(f"--poses {a.poses}: no P0 (the training home "
                                     "the session starts and ends at) -- refused")
                tol = qd_max[sl] / hz
                start = session_start(q_now[sl], p0["q"], rest_file[sl], tol)
                lead = None
                if start == "rest":
                    # D3: rest -> P0 is a precomputed, clearance-checked tape
                    # (remoroo-world engine/sysid writes it beside poses.yaml)
                    tp = a.poses.parent / "transit_rest_P0.npz"
                    if not tp.exists():
                        raise SystemExit(f"--poses: the arm is at rest and {tp} "
                                         "does not exist -- refused")
                    lead = load_replay(tp, cell, sl, hz)
                    if lead["cap_class"] != "transit":
                        raise SystemExit(f"{tp.name}: cap_class {lead['cap_class']!r}"
                                         ", not 'transit' -- refused")
                    if (np.any(np.abs(lead["start_pose"] - rest_file[sl]) > tol)
                            or np.any(np.abs(lead["q_target"][-1] - p0["q"]) > tol)):
                        raise SystemExit(f"{tp.name}: does not run cell rest -> P0 "
                                         "-- refused")
                    _prep_replay(lead, cell, q_now, sl, adapter.qdd_max)
                q_plan = q_now.copy()
                if lead is not None:
                    q_plan[sl] = lead["q_target"][-1]
                home = q_plan.copy()
                home[sl] = p0["q"]
            elif a.at_cell_rest:
                anchors = [{"label": "cell_rest", "role": "fit", "q": rest_file[sl]}]
                start, lead, q_plan, home = "current pose", None, q_now, None
            else:
                anchors = [{"label": "current", "role": "fit", "q": None}]
                start, lead, q_plan, home = "current pose", None, q_now, None
            plan = build_pose_plan(cell, q_plan, sl, anchors, joints, scale, qd_max,
                                   adapter.qdd_max)
            if home is not None:
                home = next(x["full"] for x in plan if x["label"] == "P0")
            rows = plan_rows(cell, plan, q_now, adapter.qdd_max, lead=lead, home=home)
            print_plan(rows, hz, ui=ui, n_units=len(hosts), accel_cap=adapter.qdd_max,
                       start=start, extra=("ends at P0" if home is not None else ""))
            n_id = len(joints)
            print(f"{cell.name}: identifying {n_id} joints at "
                  f"{[x['label'] for x in anchors]}"
                  + (" ON HARDWARE" if a.hardware else " (dry run)"))
            raw_dir.mkdir(parents=True, exist_ok=True)
            if a.poses:
                # contract (B): the raw dir carries the poses its labels name
                shutil.copy2(a.poses, raw_dir / "poses.yaml")
            lead_raw = ([str(_play_replay(adapter, lead, ui, raw_dir, cell, hz, gh, utc))]
                        if lead is not None else [])
            fits, extra = identify_hw(cell, adapter, inertia, ui=ui, joints=joints,
                                      plan=plan, raw_dir=raw_dir, utc=utc, home=home)
            extra["raw_files"] = lead_raw + extra["raw_files"]
        print(f"  commanded motion: {adapter.motion_s:.1f} s "
              f"({adapter.motion_s / 60.0:.2f} min); raw files in {raw_dir}")
        extra["motion_s"] = round(adapter.motion_s, 2)
    finally:
        adapter.disconnect()
    if fits is not None:
        _write_gains(a, cell, by_kind, fits, extra)
    return 0


def _prep_replay(t: dict, cell, q_now: np.ndarray, sl: slice, accel_cap: float) -> None:
    """A loaded contract-(A) tape -> full-width rows (the other unit's joints
    at their measured, untouched values) checked by ITS cap class."""
    hz = float(cell.limits["rates"]["command_hz"])
    lo, hi = cell.joint_limits()
    full = np.repeat(np.asarray(q_now, dtype=np.float64)[None],
                     t["q_target"].shape[0], axis=0)
    full[:, sl] = t["q_target"]
    t["full"] = full
    t["peaks"] = check_tape(t["name"], t["cap_class"], full, lo, hi,
                            cell.joint_velocity_limits(), accel_cap, hz, full[0])


def _play_replay(adapter, t: dict, ui: int, raw_dir: Path, cell, hz: float,
                 gh: str, utc: str) -> Path:
    """Glide to the tape's start, play it, write its raw record.  Replays are
    recorded, never fitted: role 'heldout'."""
    sl = adapter.slices[ui]
    adapter.transit(t["full"][0])
    rec = adapter.play(t["full"], ui)
    path = write_raw(
        raw_dir / f"unit{ui}_{t['pose_label']}_replay_{t['name']}.npz", rec,
        tape=t["name"], params={"kind": "replay", "cap_class": t["cap_class"],
                                "source": t["source"], "pose_label": t["pose_label"],
                                "role": "heldout", "peaks": t["peaks"]},
        joints=cell.joint_labels()[sl], excited=cell.joint_labels()[sl],
        command_hz=hz, start_pose=t["start_pose"], git_hash=gh, utc=utc,
        unit_host=adapter.hosts[ui])
    err = rec["q"][-1] - rec["send_q"][-1]
    print(f"  {t['name']} ({t['cap_class']}): {rec['send_q'].shape[0]} ticks, end "
          f"error max {np.max(np.abs(err)):.4f} rad -> {path}")
    return path


def _write_gains(a, cell, by_kind, fits, extra) -> None:
    delays = {v["delay_ticks"] for v in fits.values()}
    doc = {
        "plant_rate_hz": cell.gains.get("plant_rate_hz", 1000),
        "command_delay_ticks": int(max(delays)),
        "by_kind": by_kind,
        # `residual` and `delay_ticks` are per-joint EVIDENCE, and they used to be
        # printed to the operator's terminal and then dropped on the floor: a
        # garbage fit and a clean one wrote indistinguishable files, and the
        # 2026-08-30 rig file (cells/robot_setup_3/lc/gains.yaml) carries twelve
        # kp/kd pairs with no way left to ask how well any of them fitted, or
        # which joints disagreed about the delay that command_delay_ticks above
        # collapses to its maximum.  residual is the mean squared error of
        # fit_joint's least-squares solve at the winning delay (units: N^2, so it
        # is comparable across joints only through this file's own inertias);
        # delay_ticks is THIS joint's fitted delay in command ticks.
        # ⚠ Added keys only, never renamed or reshaped: engine/rigmeas/wrap_gains.py
        # and engine/convert/actuators_from_sysid.py read inertia/kp/kd out of
        # these same per_joint entries, and wrap_gains decides inertia's gauge by
        # comparing it against by_kind above.
        "per_joint": {
            k: {"inertia": v["inertia"], "kp": v["kp"], "kd": v["kd"],
                "delay_ticks": int(v["delay_ticks"]),
                "residual": float(v["residual"])}
            for k, v in fits.items()
        },
    }
    # 2026-10-07, ADDED keys only: where each tape ran (tape_pose, read by
    # remoroo-world engine/convert/actuators_from_sysid.py:323, "unrecorded"
    # until now), what ran (sysid_tapes) and the raw records (sysid_raw_files)
    # the sim-vs-real check replays.
    if "tape_pose" in extra:
        doc["tape_pose"] = extra["tape_pose"]
        doc["sysid_tapes"] = extra["tapes"]
        doc["sysid_raw_files"] = extra["raw_files"]
    if len(delays) > 1:
        print(
            f"  NOTE: joints disagree about the delay {sorted(delays)}; the plant "
            "model carries one delay for the whole cell, so the largest is used"
        )
    if a.out:
        # ⚠ mkdir before write: write_text does NOT create parents, and the only
        # thing standing between a 25-minute commanded-motion session and a
        # FileNotFoundError that throws away every number it measured was the
        # operator having happened to name a directory that already existed.  An
        # engine phase writing into a per-run artifact directory does not.
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
