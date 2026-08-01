#!/usr/bin/env python3
"""Record a trajectory from a real xArm cell.  Reads only; never commands motion.

Two ways to get a trajectory onto tape:

  --teach     put the arm in manual mode and hand-guide it.  The arm is
              back-drivable and the controller commands nothing, so this is the
              safest possible way to obtain a real trajectory, and the path is
              whatever a human actually wanted rather than whatever we thought to
              synthesise.

  --observe   record while something else drives the arm (the vendor's own
              planner, an existing script, a teleop session).  Also read-only.

What is recorded, per sample: the joint positions and velocities the controller
reports, AND the controller's own forward-kinematic TCP pose.  That last one is
the point: it lets the replay stage check our URDF and our FK against the
vendor's before anything moves, which is the cheapest possible way to catch a
calibration error.

    python scripts/rig_record.py --cell configs/cells/rig_bimanual_xarm6.yaml \
        --hosts $REMOROO_LC_HOSTS --teach --seconds 40 \
        -o recordings/reach_and_place.jsonl
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

from remoroo_lc.adapters.xarm_rt import XArmRealTime  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402

MODE_POSITION = 0
MODE_TEACH = 2  # manual / back-drivable


def _connect(host: str, teach: bool, report_type: str = "real"):
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:  # pragma: no cover - hardware only
        raise SystemExit(
            "the UFACTORY SDK is not installed: pip install xarm-python-sdk"
        ) from exc
    # report_type="real" is the 250 Hz-class real-time stream on port 30003.
    # The default ("rich", port 30002) and "normal" (30001) both push at about
    # 5 Hz on this firmware -- measured 5.3 vs 99.7 packets/s -- and a recording
    # taken from those arrives in 0.09 rad steps every 172 ms, which is not a
    # trajectory, it is a slideshow of one.
    arm = XArmAPI(host, is_radian=True, report_type=report_type)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    if teach:
        # Manual mode. The servos hold no target; the arm is free to be pushed.
        arm.set_mode(MODE_TEACH)
    else:
        arm.set_mode(MODE_POSITION)
    arm.set_state(0)
    return arm


def record(
    hosts: list[str], seconds: float, rate_hz: float, teach_hosts: set[str],
    report_type: str = "real",
) -> list[dict]:
    """Read every host; put only `teach_hosts` into manual mode.

    The split matters: an arm nobody is holding should not be made compliant just
    because its neighbour is being guided.  State is read from every host either
    way, because the recording has to cover the whole cell's joint vector.
    """
    # The SDK connection is used only to set the mode; every sample comes from
    # the 250 Hz port-30000 stream, which the SDK does not expose.
    arms = [_connect(h, h in teach_hosts, report_type) for h in hosts]
    # Capture each controller's configured tool offset.  It is a per-arm SETTING,
    # not a property of the robot, and without recording it the replay cannot tell
    # a kinematic error from a bookkeeping one -- on this rig the two arms have
    # 200 mm and 0 mm configured.
    tcp_offsets = []
    time.sleep(1.5)  # tcp_offset arrives on the report stream, not at connect time
    for arm in arms:
        off = list(arm.tcp_offset or [0.0] * 6)
        tcp_offsets.append([off[0] / 1000.0, off[1] / 1000.0, off[2] / 1000.0,
                            off[3], off[4], off[5]])
    streams = [XArmRealTime(h) for h in hosts]
    for st in streams:
        st.connect()
    if teach_hosts:
        print(
            "\n  MANUAL MODE ENGAGED on " + ", ".join(sorted(teach_hosts))
            + "\n  Those arms are back-drivable -- support them before letting go."
            + ("\n  Holding position: " + ", ".join(h for h in hosts if h not in teach_hosts)
               if len(teach_hosts) < len(hosts) else "")
            + "\n  Guide the path you want recorded; recording starts now.\n"
        )
    # The report stream IS the clock.  It delivers at 250 Hz, so reading one
    # frame per arm per iteration paces the loop at exactly the controller's rate
    # with no sleep and no drift.  Imposing our own clock on top of it -- which is
    # what the earlier version did -- only produces a loop that is sometimes early
    # (and re-reads a stale frame) and sometimes late (and falls behind).
    rows: list[dict] = []
    t0 = time.perf_counter()
    lag_s: list[float] = []
    try:
        while time.perf_counter() - t0 < seconds:
            frames = [st.read() for st in streams]
            now = time.perf_counter()
            sample = {"t": round(now - t0, 6), "units": []}
            for r in frames:
                sample["units"].append(
                    {
                        "q": [float(v) for v in r.q],
                        "qd": [float(v) for v in r.qd],
                        "tcp_xyz": [float(v) for v in r.tcp_xyz],
                        "tcp_rotvec": [float(v) for v in r.tcp_rotvec],
                        "controller_t": r.timestamp_s,
                        "gripper_mm": r.gripper_pos_mm,
                    }
                )
            rows.append(sample)
            # Falling behind shows up as our wall clock advancing faster than the
            # controller's timestamps: the socket buffers and every frame we hand
            # back gets older.  Recorded so it is visible instead of silent.
            if len(rows) == 1:
                ctrl_t0 = frames[0].timestamp_s
            lag_s.append((now - t0) - (frames[0].timestamp_s - ctrl_t0))
    finally:
        for st in streams:
            st.close()
        for arm, host in zip(arms, hosts):
            try:
                if host in teach_hosts:
                    # Leave the arm in position mode holding where it stands, so
                    # it does not stay back-drivable after the operator walks away.
                    arm.set_mode(MODE_POSITION)
                    arm.set_state(0)
                arm.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: clean-up on {host}: {exc}", file=sys.stderr)
    return rows, tcp_offsets


def report(rows: list[dict], rate_hz: float) -> dict:
    """What was actually captured, including how well the clock held."""
    t = np.asarray([r["t"] for r in rows])
    gaps = np.diff(t)
    n_units = len(rows[0]["units"])
    # How many samples are actually NEW.  The loop can spin faster than the
    # controller's report socket updates, and a recording padded with repeats
    # would read as a robot that kept stopping.
    ctrl_t = np.asarray([r["units"][0].get("controller_t", np.nan) for r in rows])
    ctrl_gaps = np.diff(ctrl_t)
    q_all = np.asarray([[v for u in r["units"] for v in u["q"]] for r in rows])
    distinct = int(np.count_nonzero(np.any(np.diff(q_all, axis=0) != 0.0, axis=1))) + 1
    stats = {
        "samples": len(rows),
        "distinct_samples": distinct,
        "effective_rate_hz": float((distinct - 1) / t[-1]),
        "duration_s": float(t[-1]),
        "rate_hz_nominal": rate_hz,
        "rate_hz_actual": float((len(rows) - 1) / t[-1]),
        "gap_ms_p50": float(np.percentile(gaps, 50) * 1e3),
        "gap_ms_p99": float(np.percentile(gaps, 99) * 1e3),
        "gap_ms_max": float(gaps.max() * 1e3),
        "controller_gap_ms_p50": float(np.nanpercentile(ctrl_gaps, 50) * 1e3),
        "controller_gap_ms_p99": float(np.nanpercentile(ctrl_gaps, 99) * 1e3),
        "controller_frames_skipped": int(np.nansum(ctrl_gaps > 0.006)),
        "units": [],
    }
    for u in range(n_units):
        p = np.asarray([r["units"][u]["tcp_xyz"] for r in rows])
        q = np.asarray([r["units"][u]["q"] for r in rows])
        # Speed against the CONTROLLER's clock, and only between samples that
        # actually changed.  Dividing by our wall clock instead turns a
        # sub-millisecond loop hiccup into a reported 172 m/s, because the
        # numerator is a real 4 ms of motion and the denominator is scheduling
        # noise.
        ct = np.asarray([r["units"][u].get("controller_t", np.nan) for r in rows])
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dts = np.diff(ct)
        ok = np.isfinite(dts) & (dts > 1e-4)
        speed = step[ok] / dts[ok] if ok.any() else np.zeros(1)
        stats["units"].append(
            {
                "path_length_m": float(np.sum(step)),
                "moving_fraction": float(np.mean(speed > 0.01)) if ok.any() else 0.0,
                "tcp_speed_p50": float(np.percentile(speed, 50)),
                "tcp_speed_p95": float(np.percentile(speed, 95)),
                "tcp_speed_max": float(speed.max()),
                "joint_range_rad": [float(v) for v in np.ptp(q, axis=0)],
            }
        )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument(
        "--hosts",
        default=os.environ.get("REMOROO_LC_HOSTS", ""),
        help="comma-separated controller IPs; defaults to $REMOROO_LC_HOSTS (see .env.example)",
    )
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--teach", action="store_true", help="hand-guide (manual mode)")
    ap.add_argument(
        "--teach-hosts", default="",
        help="comma-separated subset to make compliant; default is all of --hosts",
    )
    ap.add_argument(
        "--report-type", default="real", choices=["real", "rich", "normal"],
        help="xArm report stream; 'real' is port 30003, the only fast one",
    )
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    cell = load_cell(a.cell)
    hosts = [h.strip() for h in a.hosts.split(",") if h.strip()]
    if len(hosts) * 6 != cell.n_joints:
        print(
            f"  NOTE: {len(hosts)} hosts x 6 joints != the cell's {cell.n_joints}; "
            "the recording will be checked against the cell at replay time",
            file=sys.stderr,
        )

    teach_hosts: set[str] = set()
    if a.teach:
        teach_hosts = (
            {h.strip() for h in a.teach_hosts.split(",") if h.strip()}
            if a.teach_hosts
            else set(hosts)
        )
        unknown = teach_hosts - set(hosts)
        if unknown:
            raise SystemExit(f"--teach-hosts names hosts not in --hosts: {sorted(unknown)}")
    rows, tcp_offsets = record(hosts, a.seconds, a.rate, teach_hosts, a.report_type)
    stats = report(rows, a.rate)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "meta",
                    "cell": cell.name,
                    "hosts": hosts,
                    "mode": "teach" if a.teach else "observe",
                    "teach_hosts": sorted(teach_hosts),
                    "tcp_offset": tcp_offsets,
                    "stats": stats,
                }
            )
            + "\n"
        )
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\nwrote {a.out}  ({stats['samples']} samples, {stats['duration_s']:.1f} s)")
    moved = max((u["path_length_m"] for u in stats["units"]), default=0.0) > 0.05
    if moved and stats["effective_rate_hz"] < 0.3 * stats["rate_hz_actual"]:
        print(
            f"  WARNING: only {stats['effective_rate_hz']:.1f} Hz of the samples are "
            f"new.  The controller's report stream, not this loop, is the limit; "
            f"the recorded path is undersampled and should not be replayed."
        )
    print(
        f"  clock: {stats['rate_hz_actual']:.1f} Hz sampled, "
        f"{stats['effective_rate_hz']:.1f} Hz distinct "
        f"({stats['distinct_samples']}/{stats['samples']} new), "
        f"gap p99 {stats['gap_ms_p99']:.2f} ms, max {stats['gap_ms_max']:.2f} ms\n"
        f"  controller clock: dt p50 {stats['controller_gap_ms_p50']:.3f} ms, "
        f"p99 {stats['controller_gap_ms_p99']:.3f} ms, "
        f"{stats['controller_frames_skipped']} frames skipped"
    )
    for i, u in enumerate(stats["units"]):
        print(
            f"  unit {i}: path {u['path_length_m']:.3f} m, TCP speed "
            f"p50 {u['tcp_speed_p50']:.3f} / p95 {u['tcp_speed_p95']:.3f} / "
            f"max {u['tcp_speed_max']:.3f} m/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
