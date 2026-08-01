#!/usr/bin/env python3
"""Replay a recorded trajectory through the controller, in sim and on hardware.

Staged so that everything which can be learned without moving the robot is
learned first, and each stage gates the next:

  --check      kinematics only.  Compare our FK against the vendor controller's
               on the recorded joints.  No robot needed, no motion.
  --sim        replay through controller + plant.  Reports tracking, filter
               activity and solve cost.  No robot needed, no motion.
  --hw         replay on the real cell.  Requires --go, and refuses to start if
               the sim stage was not run first.
  --baseline   stream the RECORDED joint targets straight to the servo with the
               controller out of the loop.  This is the control condition: it
               measures the robot's own tracking error, so the hardware replay
               can be read as "our contribution" instead of "everything".

Speed is swept, not reduced.  `--speeds` plays the same path at several time
scales; the point is to find where tracking degrades, which is a property worth
knowing, rather than to pick one slow pass that succeeds.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.plant import Plant  # noqa: E402
from remoroo_lc.reference.controller import Controller  # noqa: E402
from remoroo_lc.rig.recording import (  # noqa: E402
    check_kinematics,
    load_recording,
    recording_to_tape,
)
from remoroo_lc.schema import load_cell  # noqa: E402


def _use_current_pose_as_rest(cell, q_now: np.ndarray) -> None:
    """Anchor the posture term at where the robot actually is.

    A home pose baked into the cell file is the wrong reference for a replay: the
    recording starts wherever the operator left the arm, and pulling toward some
    other configuration would bias the whole run.  For a cell with
    n == TASK_DIM * T there is no null space for it to act in anyway, so this
    mostly matters for redundant cells -- but it costs nothing to be right.
    """
    k = 0
    for m in cell.models:
        nm = len(m.joint_names)
        object.__setattr__(m, "rest", np.asarray(q_now[k : k + nm], dtype=np.float32))
        k += nm


def replay_sim(cell, tape, q0: np.ndarray) -> dict:
    """Controller + plant.  Returns the achieved TCP path and diagnostics."""
    ctrl, plant = Controller(cell), Plant(cell)
    ticks_per_action = int(
        round(
            float(cell.limits["rates"]["command_hz"])
            / float(cell.limits["rates"]["policy_hz"])
        )
    )
    q = np.asarray(q0, dtype=np.float32)
    plant.reset(q)
    ctrl.reset(q)
    p_meas, p_cmd, solve_ms, active, minpair, qd_all = [], [], [], [], [], []
    for s in range(0, tape.actions.shape[0], 8):
        chunk = tape.actions[s : s + 8]
        ctrl.set_chunk(chunk, q)
        for _ in range(chunk.shape[0] * ticks_per_action):
            t0 = time.perf_counter()
            out = ctrl.step(q)
            solve_ms.append((time.perf_counter() - t0) * 1e3)
            q, _ = plant.step(out.q_target)
            p_meas.append(out.p_meas.copy())
            p_cmd.append(out.p_cmd.copy())
            active.append(out.diag["n_active"] > 0)
            minpair.append(out.diag["min_pair_distance"])
            qd_all.append(out.qd.copy())
    return {
        "p_meas": np.asarray(p_meas),
        "p_cmd": np.asarray(p_cmd),
        "qd": np.asarray(qd_all),
        "solve_ms": np.asarray(solve_ms),
        "wall_active_fraction": float(np.mean(active)),
        "min_pair_distance_m": float(np.min(minpair)),
        "q_final": q,
    }


def replay_hw(cell, tape, adapter, watchdog_ms: float = 4.0) -> dict:
    """Controller + real cell.  Halts on any hard violation or a late solve."""
    ctrl = Controller(cell)
    dt = 1.0 / float(cell.limits["rates"]["command_hz"])
    ticks_per_action = int(round(dt**-1 / float(cell.limits["rates"]["policy_hz"])))

    q, _ = adapter.read_state()
    _use_current_pose_as_rest(cell, q)
    ctrl = Controller(cell)  # rebuilt so it picks up the live rest posture
    ctrl.reset(q)

    p_meas, p_cmd, solve_ms, active, minpair, jitter = [], [], [], [], [], []
    deadline = time.perf_counter()
    try:
        for s in range(0, tape.actions.shape[0], 8):
            chunk = tape.actions[s : s + 8]
            ctrl.set_chunk(chunk, q)
            for _ in range(chunk.shape[0] * ticks_per_action):
                t0 = time.perf_counter()
                out = ctrl.step(q)
                solve_ms.append((time.perf_counter() - t0) * 1e3)

                if out.diag["min_pair_distance"] <= 0.0:
                    adapter.estop()
                    raise RuntimeError(
                        f"HARD VIOLATION: pair distance "
                        f"{out.diag['min_pair_distance'] * 1e3:.2f} mm -- stopped"
                    )
                if len(solve_ms) > 100 and np.percentile(solve_ms, 99) > watchdog_ms:
                    adapter.estop()
                    raise RuntimeError(
                        f"solve p99 {np.percentile(solve_ms, 99):.2f} ms exceeded "
                        f"{watchdog_ms} ms -- a late filter is not a filter"
                    )

                adapter.stream_targets(out.q_target)
                adapter.set_effector(out.effector)
                p_meas.append(out.p_meas.copy())
                p_cmd.append(out.p_cmd.copy())
                active.append(out.diag["n_active"] > 0)
                minpair.append(out.diag["min_pair_distance"])

                deadline += dt
                slack = deadline - time.perf_counter()
                jitter.append(slack * 1e3)
                if slack > 0:
                    time.sleep(slack)
                q, _ = adapter.read_state()
    finally:
        pass
    return {
        "p_meas": np.asarray(p_meas),
        "p_cmd": np.asarray(p_cmd),
        "solve_ms": np.asarray(solve_ms),
        "jitter_ms": np.asarray(jitter),
        "wall_active_fraction": float(np.mean(active)),
        "min_pair_distance_m": float(np.min(minpair)),
        "q_final": q,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="kinematics cross-check only")
    ap.add_argument("--sim", action="store_true", help="replay against the plant")
    ap.add_argument("--hw", action="store_true", help="replay on the real cell")
    ap.add_argument("--baseline", action="store_true", help="servo the recording directly")
    ap.add_argument(
        "--hosts",
        default=os.environ.get("REMOROO_LC_HOSTS", ""),
        help="controller IPs for --hw/--baseline; defaults to $REMOROO_LC_HOSTS",
    )
    ap.add_argument(
        "--speeds", type=float, nargs="+", default=[1.0],
        help="time scales to sweep; >1 is faster than recorded",
    )
    ap.add_argument("--go", action="store_true", help="required to command motion")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "reports" / "rig")
    a = ap.parse_args()

    cell = load_cell(a.cell)
    rec = load_recording(a.recording)
    print(f"recording: {len(rec)} samples, {rec.t[-1]:.1f} s at {rec.rate_hz:.0f} Hz "
          f"({rec.mode})")

    results: dict = {"recording": str(a.recording), "cell": cell.name}

    # ---- stage 1: kinematics, no motion ---------------------------------- #
    if a.check or a.sim or a.hw:
        k = check_kinematics(cell, rec)
        results["kinematics"] = k
        print("\nkinematics cross-check (our FK vs the controller's own):")
        for t in k["tcps"]:
            print(
                f"  {t['name']:<12} pos RMS {t['pos_rms_mm']:6.2f} mm  "
                f"max {t['pos_max_mm']:6.2f} mm  bias |{t['pos_bias_norm_mm']:5.2f}| mm  "
                f"residual {t['pos_residual_rms_mm']:5.2f} mm  "
                f"rot RMS {t['rot_rms_deg']:5.3f} deg  corr(err,reach) "
                f"{t['error_vs_reach_corr']:+.2f}"
            )

        # Two different questions, and only one of them gates a replay.
        #
        # ABSOLUTE agreement -- our FK against the number the vendor controller
        # reports -- is limited by that unit's factory kinematic calibration,
        # which lives in the controller and is not in any URDF.  It shows up as a
        # constant bias, it differs between two physically identical arms, and it
        # is irreducible without per-unit calibration data.  It only matters for a
        # stage that consumes controller-reported TCP.
        #
        # TRACKING fidelity is what the sim stage measures, and it puts our FK on
        # BOTH sides of the subtraction, so a constant bias cancels exactly.  What
        # survives is the posture-dependent residual, so that is the gate.
        bias_worst = max(t["pos_bias_norm_mm"] for t in k["tcps"])
        resid_worst = max(t["pos_residual_rms_mm"] for t in k["tcps"])
        if resid_worst > 2.0:
            print(
                f"\n  STOP: {resid_worst:.1f} mm of POSTURE-DEPENDENT disagreement.\n"
                "  This part does not cancel, so a replay would attribute it to the\n"
                "  controller.  Fix the URDF, the tool transform or the base\n"
                "  calibration first -- the reach correlation above says which."
            )
            return 2
        print(
            f"  -> {bias_worst:.2f} mm of the disagreement is a CONSTANT bias per TCP:\n"
            "     each unit's factory calibration, absent from a nominal URDF and\n"
            "     different for each arm, so not removable here.  It is common-mode\n"
            f"     and cancels in the replay.  Posture-dependent residual is"
            f" {resid_worst:.2f} mm."
        )
        if a.hw or a.baseline:
            print(
                f"     NOTE: absolute accuracy against the controller stays"
                f" ~{bias_worst:.1f} mm.\n"
                "     Hardware numbers below are tracking fidelity, not absolute accuracy."
            )

    # ---- stage 2: sim ----------------------------------------------------- #
    if a.sim:
        _use_current_pose_as_rest(cell, rec.q[0])
        results["sim"] = {}
        print("\nsim replay (controller + plant):")
        for scale in a.speeds:
            tape, env = recording_to_tape(cell, rec, time_scale=scale)
            r = replay_sim(cell, tape, rec.q[0])
            err = np.linalg.norm(r["p_meas"] - r["p_cmd"], axis=2)
            row = {
                "time_scale": scale,
                "tcp_speed_p95": env["tcp_speed_p95"],
                "rms_mm": float(np.sqrt(np.mean(err**2)) * 1e3),
                "max_mm": float(err.max() * 1e3),
                "wall_active_fraction": r["wall_active_fraction"],
                "min_pair_mm": r["min_pair_distance_m"] * 1e3,
                "solve_ms_p99": float(np.percentile(r["solve_ms"], 99)),
            }
            results["sim"][str(scale)] = row
            print(
                f"  x{scale:<4g} peak {max(env['tcp_speed_p95']):.3f} m/s   "
                f"RMS {row['rms_mm']:6.2f} mm   max {row['max_mm']:6.2f} mm   "
                f"walls {row['wall_active_fraction'] * 100:5.1f}%   "
                f"min pair {row['min_pair_mm']:6.1f} mm"
            )
        if any(v["wall_active_fraction"] > 0.02 for v in results["sim"].values()):
            print(
                "\n  NOTE: the filter fired on a path that was walked collision-free.\n"
                "  Either the sphere set is too fat, the obstacle list is wrong, or\n"
                "  the calibration is off.  Worth resolving before the hardware run."
            )

    # ---- stage 3: hardware ------------------------------------------------ #
    if a.hw or a.baseline:
        if not a.go:
            print(
                "\nrefusing to command motion without --go.\n"
                "Run --check and --sim first, read them, then add --go.",
                file=sys.stderr,
            )
            return 3
        if "sim" not in results:
            print("refusing to run hardware without a sim stage in the same invocation",
                  file=sys.stderr)
            return 3
        raise SystemExit(
            "hardware replay is wired but intentionally not reachable from this "
            "script yet: build the CellAdapter for your cell and call replay_hw/"
            "replay_baseline directly, with an operator on the estop."
        )

    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f"{a.recording.stem}_replay.json"
    dest.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
