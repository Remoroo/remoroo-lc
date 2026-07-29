"""Load a rig recording, check it against our kinematics, and turn it into a tape.

Three things happen here, in this order on purpose:

1. **Load** what the robot reported: joints, joint velocities, and the vendor
   controller's own TCP pose.
2. **Cross-check our forward kinematics against the vendor's**, using the TCP pose
   the controller reported for the very same joint angles.  This costs nothing,
   requires no motion, and is the only cheap way to find out that the URDF or the
   calibration is wrong before a replay blames the controller for it.
3. **Convert** to an action tape in the frozen schema, which is what a policy
   would have emitted to produce this motion.

The conversion is where a replay can quietly stop being a fair test, so the
choices are explicit:

* Actions are sampled at policy rate from the recording; the recording is at
  command rate or faster.  Downsampling is by nearest sample, not by averaging,
  because averaging invents poses the robot never held.
* Deltas are relative to the PREVIOUS WAYPOINT, matching `delta_mode="cumulative"`.
  A replay therefore reproduces the recorded *shape*; it does not re-anchor to
  the recorded absolute poses, so any tracking error accumulates exactly the way
  it would for a real policy.  That is the honest behaviour to measure.
* The recording's own speed is reported against the controller's achievable
  envelope, so a replay that cannot keep up is identified as such rather than
  being read as a tracking failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from remoroo_lc.constants import DTYPE, POINT_DIM, TASK_DIM
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import exp_so3, log_so3, make_transform, transform_inv
from remoroo_lc.tapes import Tape


@dataclass
class Recording:
    """What the robot reported, as arrays."""

    cell: str
    hosts: list[str]
    mode: str
    t: np.ndarray  # (N,)
    q: np.ndarray  # (N, n_joints) in the cell's joint order
    qd: np.ndarray  # (N, n_joints)
    tcp_xyz: np.ndarray  # (N, n_units, 3) as the CONTROLLER reported it
    tcp_rotvec: np.ndarray  # (N, n_units, 3) axis-angle, see xarm_rt
    #: Per-arm tool offset the CONTROLLER had configured, if the recording says.
    tcp_offset: np.ndarray | None
    meta: dict

    @property
    def rate_hz(self) -> float:
        return float((len(self.t) - 1) / (self.t[-1] - self.t[0]))

    def __len__(self) -> int:
        return int(self.t.shape[0])


def load_recording(path: str | Path) -> Recording:
    meta, rows = None, []
    with open(path) as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("kind") == "meta":
                meta = obj
            else:
                rows.append(obj)
    if meta is None:
        raise ValueError(f"{path}: no metadata line")
    if not rows:
        raise ValueError(f"{path}: no samples")
    return Recording(
        cell=meta["cell"],
        hosts=meta.get("hosts", []),
        mode=meta.get("mode", "unknown"),
        t=np.asarray([r["t"] for r in rows], dtype=np.float64),
        q=np.asarray(
            [[v for u in r["units"] for v in u["q"]] for r in rows], dtype=DTYPE
        ),
        qd=np.asarray(
            [[v for u in r["units"] for v in u["qd"]] for r in rows], dtype=DTYPE
        ),
        tcp_xyz=np.asarray(
            [[u["tcp_xyz"] for u in r["units"]] for r in rows], dtype=DTYPE
        ),
        tcp_rotvec=np.asarray(
            [
                [u.get("tcp_rotvec", u.get("tcp_rpy")) for u in r["units"]]
                for r in rows
            ],
            dtype=DTYPE,
        ),
        tcp_offset=(
            np.asarray(meta["tcp_offset"], dtype=DTYPE)
            if meta.get("tcp_offset") is not None
            else None
        ),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# 2. kinematics cross-check
# --------------------------------------------------------------------------- #


def chain_frames(cell: CellSpec, tcp_index: int) -> tuple[str, str]:
    """(base link, tip link) for one TCP, derived from the joint graph.

    A vendor controller reports its TCP in ITS OWN chain's base frame, which for a
    cell whose URDF carries several chains is not the model's root.  And it reports
    it at whatever tool offset happens to be configured on that controller, which
    is a setting, not a fact about the robot.

    So the comparison is anchored at two places that no setting can move: the
    first actuated joint's parent (that chain's base) and the last actuated joint's
    child (its tip).  Both fall out of the chain; neither is written down
    anywhere per robot.
    """
    tcp = cell.tcps[tcp_index]
    urdf = cell.urdfs[tcp.model]
    model = next(m for m in cell.models if m.name == tcp.model)
    chain = [j for j in urdf.chain_to(tcp.frame) if j in model.joint_names]
    if not chain:
        raise ValueError(f"{tcp.name}: no actuated joint on its chain")
    return urdf.by_name[chain[0]].parent, urdf.by_name[chain[-1]].child


def check_kinematics(cell: CellSpec, rec: Recording, stride: int = 10) -> dict:
    """Compare our FK against the vendor controller's, on the recorded joints.

    Both sides are computing the same thing from the same joint angles, so any
    disagreement is a disagreement about the ROBOT -- the URDF's link lengths, the
    tool transform, or the base calibration -- and not about control.  Finding it
    here costs nothing.  Finding it after a replay means re-running the replay.

    A constant offset points at the tool transform; an offset that grows with
    distance from the base points at link lengths; an offset that swings with
    configuration points at the base calibration.  The returned per-axis bias and
    the correlation with reach are there to tell those apart.
    """
    if rec.q.shape[1] != cell.n_joints:
        raise ValueError(
            f"recording has {rec.q.shape[1]} joints, cell {cell.name!r} has "
            f"{cell.n_joints}; the recording was made on a different cell"
        )
    tree = KinematicTree(cell)
    n_units = rec.tcp_xyz.shape[1]
    if n_units != cell.n_tcps:
        raise ValueError(f"recording has {n_units} units, cell has {cell.n_tcps} TCPs")

    idx = np.arange(0, len(rec), stride)
    frames = [chain_frames(cell, u) for u in range(cell.n_tcps)]
    err = np.zeros((idx.size, cell.n_tcps, POINT_DIM))
    rot_err = np.zeros((idx.size, cell.n_tcps))
    reach = np.zeros((idx.size, cell.n_tcps))
    tool = np.zeros(cell.n_tcps)
    for k, i in enumerate(idx):
        fk = tree.fk(rec.q[i])
        for u in range(cell.n_tcps):
            base_link, tip = frames[u]
            T_base = fk.link_T[tree.link_id(cell.tcps[u].model, base_link)]
            T_tip = fk.link_T[tree.link_id(cell.tcps[u].model, tip)]
            T_ours = transform_inv(T_base) @ T_tip

            # Theirs, with the controller's own tool offset divided out so the
            # comparison lands on the tip too.  `tcp_offset` is a per-arm
            # setting; on the reference rig the two arms have 200 mm and 0 mm
            # configured while the URDF models 171.5 mm, and comparing without
            # removing it reports a 173 mm "kinematic error" that is nothing of
            # the sort.
            off = rec.tcp_offset[u] if rec.tcp_offset is not None else np.zeros(6)
            T_off = make_transform(exp_so3(off[3:]), off[:3])
            T_rep = make_transform(exp_so3(rec.tcp_rotvec[i, u]), rec.tcp_xyz[i, u])
            T_theirs = T_rep @ transform_inv(T_off)

            err[k, u] = T_ours[:POINT_DIM, POINT_DIM] - T_theirs[:POINT_DIM, POINT_DIM]
            rot_err[k, u] = float(
                np.linalg.norm(
                    log_so3(T_theirs[:POINT_DIM, :POINT_DIM] @ T_ours[:POINT_DIM, :POINT_DIM].T)
                )
            )
            reach[k, u] = float(np.linalg.norm(T_theirs[:POINT_DIM, POINT_DIM]))
            if k == 0:
                T_tool = transform_inv(T_tip) @ fk.link_T[
                    tree.link_id(cell.tcps[u].model, cell.tcps[u].frame)
                ]
                tool[u] = float(np.linalg.norm(T_tool[:POINT_DIM, POINT_DIM]))

    out = {"samples": int(idx.size), "tcps": []}
    for u in range(cell.n_tcps):
        d = np.linalg.norm(err[:, u], axis=1)
        bias = err[:, u].mean(axis=0)
        centred = d - d.mean()
        r = reach[:, u] - reach[:, u].mean()
        corr = (
            float(np.dot(centred, r) / (np.linalg.norm(centred) * np.linalg.norm(r)))
            if np.linalg.norm(centred) > 1e-12 and np.linalg.norm(r) > 1e-12
            else 0.0
        )
        out["tcps"].append(
            {
                "name": cell.tcps[u].name,
                "pos_rms_mm": float(np.sqrt(np.mean(d**2)) * 1e3),
                "pos_max_mm": float(d.max() * 1e3),
                "pos_bias_mm": [float(v * 1e3) for v in bias],
                "pos_bias_norm_mm": float(np.linalg.norm(bias) * 1e3),
                "rot_rms_deg": float(np.degrees(np.sqrt(np.mean(rot_err[:, u] ** 2)))),
                "rot_max_deg": float(np.degrees(rot_err[:, u].max())),
                "error_vs_reach_corr": corr,
                "our_tip_to_tcp_mm": float(tool[u] * 1e3),
                "their_tcp_offset_mm": (
                    float(np.linalg.norm(rec.tcp_offset[u][:POINT_DIM]) * 1e3)
                    if rec.tcp_offset is not None
                    else None
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 3. conversion to an action tape
# --------------------------------------------------------------------------- #


def recording_to_tape(
    cell: CellSpec,
    rec: Recording,
    name: str = "replay",
    time_scale: float = 1.0,
    category: str = "clean",
) -> tuple[Tape, dict]:
    """Convert a recording into the action tape a policy would have emitted.

    `time_scale` > 1 plays the path back FASTER than it was recorded, < 1 slower.
    It is here so the same path can be swept across a speed ladder, which is how
    a controller's usable envelope gets measured rather than assumed -- not so a
    failing test can be slowed down until it passes.
    """
    tree = KinematicTree(cell)
    policy_hz = float(cell.limits["rates"]["policy_hz"])
    duration = float(rec.t[-1] - rec.t[0]) / max(time_scale, 1e-9)
    n_steps = max(int(round(duration * policy_hz)), 2)

    # Nearest recorded sample at each policy instant.  Not interpolated: an
    # interpolated pose is a pose the robot never actually held.
    want = rec.t[0] + np.linspace(0.0, float(rec.t[-1] - rec.t[0]), n_steps + 1)
    pick = np.searchsorted(rec.t, want).clip(0, len(rec) - 1)

    # Every waypoint's pose in its own model base frame, resolved first; the
    # deltas are differenced out of them afterwards.
    inv_base = [
        transform_inv(next(m for m in cell.models if m.name == t.model).base)
        for t in cell.tcps
    ]
    p_base = np.zeros((n_steps + 1, cell.n_tcps, POINT_DIM), dtype=DTYPE)
    R_base = np.zeros((n_steps + 1, cell.n_tcps, POINT_DIM, POINT_DIM), dtype=DTYPE)
    for k, i in enumerate(pick):
        fk = tree.fk(rec.q[i])
        p_w, R_w = tree.tcp_poses(fk)
        for u in range(cell.n_tcps):
            Ri = inv_base[u][:POINT_DIM, :POINT_DIM]
            p_base[k, u] = Ri @ p_w[u] + inv_base[u][:POINT_DIM, POINT_DIM]
            R_base[k, u] = Ri @ R_w[u]

    actions = np.zeros((n_steps, cell.action_dim), dtype=DTYPE)
    for u, (pose_sl, eff_sl) in enumerate(cell.action_slices()):
        actions[:, pose_sl.start : pose_sl.start + POINT_DIM] = np.diff(p_base[:, u], axis=0)
        for k in range(n_steps):
            # Increment from waypoint k to k+1, left-composed in the base frame,
            # which is what the frozen action schema says a rotation delta is.
            actions[k, pose_sl.start + POINT_DIM : pose_sl.stop] = log_so3(
                R_base[k + 1, u] @ R_base[k, u].T
            )
        if cell.tcps[u].effector.width:
            actions[:, eff_sl] = 0.0  # recordings carry no effector channel yet

    step_dist = np.linalg.norm(np.diff(p_base, axis=0), axis=2)
    speed = step_dist * policy_hz
    envelope = {
        "policy_steps": n_steps,
        "duration_s": duration,
        "time_scale": time_scale,
        "path_length_m": [float(step_dist[:, u].sum()) for u in range(cell.n_tcps)],
        "tcp_speed_p50": [float(np.percentile(speed[:, u], 50)) for u in range(cell.n_tcps)],
        "tcp_speed_p95": [float(np.percentile(speed[:, u], 95)) for u in range(cell.n_tcps)],
        "tcp_speed_max": [float(speed[:, u].max()) for u in range(cell.n_tcps)],
        "task_v_max": float(cell.limits["task"]["linear"]["v_max"]),
    }
    tape = Tape(
        name=name,
        category=category,
        cell=cell.name,
        actions=actions,
        policy_hz=policy_hz,
        notes=(
            f"replay of a {rec.mode} recording, time_scale {time_scale:g}, "
            f"{n_steps} policy steps"
        ),
    )
    return tape, envelope
