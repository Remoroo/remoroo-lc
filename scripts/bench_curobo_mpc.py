#!/usr/bin/env python3
"""Benchmark cuRobo V2's MPC as a candidate replacement for layers 2+3.

Two questions, answered on the same 60 s teach recording our own stack is
measured against, so the numbers are comparable rather than merely impressive:

  ACCURACY  can it follow a path -- not reach a pose -- while avoiding
            collisions?  Its goal tensor is [batch, horizon, links, goalset, 3]
            with `non_terminal_tool_pose_weight_factor` explicitly there to
            weight the interior knots, so a VLA chunk maps onto its horizon with
            no adaptation.  This feeds it exactly the chunk our follower gets and
            measures the achieved TCP path against our FK on the recorded joints.

  SPEED     `optimization_dt` is 0.02 -- the VLA action period -- and
            `interpolation_steps` is fixed at 4, so its command rate is 200 Hz,
            not our 250.  Reported as-is; the shim is a separate question from
            whether the solve fits the budget at all.

CUDA only: there is no CPU path in this solver, which is itself a finding for the
edge box, where our own kernels currently run on the CPU at 0.80 ms p99.

Run on the Orin  : ~/dev/venvs/curobo-v2/bin/python scripts/bench_curobo_mpc.py ...
Run on the A10G  : same, in the sibling's curobo env.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), the convention cuRobo uses.

    Branch on the largest diagonal term rather than assuming w is big: the
    w-first formula loses precision when the trace is near -1, which is exactly
    where a downward-pointing tool sits.
    """
    m = np.asarray(R, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array(
        [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=1, help="goal knots; see note, only 1 works")
    ap.add_argument("--replan", type=int, default=8, help="knots executed per re-anchor")
    ap.add_argument("--seconds", type=float, default=60.0, help="how much of the recording")
    ap.add_argument("--collision", action="store_true", help="load spheres + self-collision")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import torch

    from remoroo_lc.reference.kinematics import KinematicTree
    from remoroo_lc.rig.recording import load_recording
    from remoroo_lc.schema import load_cell

    # Public API only -- curobo.types / curobo.model_predictive_control -- so
    # this keeps working across their internal _src reshuffles.
    from curobo.model_predictive_control import (  # noqa: PLC0415
        ModelPredictiveControl,
        ModelPredictiveControlCfg,
    )
    from curobo._src.types.robot import RobotCfg  # noqa: PLC0415  (no public alias)
    from curobo.types import GoalToolPose, JointState  # noqa: PLC0415

    dev = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    # --- our ground truth, from our own FK on the recorded joints ----------- #
    cell = load_cell(a.cell)
    tree = KinematicTree(cell)
    rec = load_recording(a.recording)
    t_rec = np.asarray(rec.t, dtype=float)
    t_rec = t_rec - t_rec[0]
    keep = t_rec <= a.seconds
    q_rec = np.asarray(rec.q)[keep]
    t_rec = t_rec[keep]

    n_tcps = cell.n_tcps
    gt_p = np.zeros((len(q_rec), n_tcps, 3))
    gt_R = np.zeros((len(q_rec), n_tcps, 3, 3))
    for i in range(len(q_rec)):
        p_w, R_w = tree.tcp_poses(tree.fk(q_rec[i]))
        gt_p[i] = np.asarray(p_w)
        gt_R[i] = np.asarray(R_w)

    # Resample onto the 20 ms action grid: this is the chunk a VLA would emit.
    dt_a = 0.02
    n_knots = int(t_rec[-1] / dt_a)
    grid = np.arange(n_knots + 1) * dt_a
    pick = np.searchsorted(t_rec, grid).clip(0, len(q_rec) - 1)
    knot_p = gt_p[pick]  # (n_knots+1, T, 3)
    knot_R = gt_R[pick]
    knot_q = q_rec[pick]

    tool_frames = [t.frame for t in cell.tcps]
    print(f"cell: {cell.name}  joints {cell.n_joints}  tools {tool_frames}")
    print(f"recording: {len(q_rec)} samples, {t_rec[-1]:.1f} s -> {n_knots} knots at 20 ms")

    # --- build the solver --------------------------------------------------- #
    # Root of the model: the one link that is no joint's child.  Derived, so
    # this works on a cell nobody here has seen.
    m0 = cell.models[0].name
    urdf0 = cell.urdfs[m0]
    children = {j.child for j in urdf0.joints}
    base_link = next(ln for ln in urdf0.links if ln not in children)

    t0 = time.perf_counter()
    robot = RobotCfg.from_basic(
        urdf_path=str(a.urdf), base_link=base_link, tool_frames=tool_frames
    )
    cfg = ModelPredictiveControlCfg.create(
        robot=robot,
        self_collision_check=bool(a.collision),
        load_collision_spheres=bool(a.collision),
        optimization_dt=dt_a,
        # CUDA graphs must be off to reshape the goal buffer to a multi-knot
        # horizon (reset_shape -> reset_cuda_graph raises otherwise).  This
        # costs launch overhead, so the solve times below are an UPPER bound on
        # what a graph-captured fixed-horizon setup would achieve.
        use_cuda_graph=False,
    )
    mpc = ModelPredictiveControl(cfg)
    build_s = time.perf_counter() - t0
    print(f"build: {build_s:.1f} s   horizon knots {mpc.action_horizon}  action_dim {mpc.action_dim}")
    print(f"collision: spheres+self = {bool(a.collision)}")

    q0 = torch.as_tensor(knot_q[0], dtype=torch.float32, device=dev).view(1, -1)
    state = JointState.from_position(q0, joint_names=mpc.joint_names)
    # tool_frames=None means "track all tool links", which is both TCPs here.
    # Passing the explicit list cannot work: their validator tests
    # `tool_frames not in self.tool_frames` -- a membership test where a subset
    # test was intended -- so any correct list is rejected (cuRobo V2
    # solver_mpc.py:349).
    mpc.setup(current_state=state, tool_frames=None)
    # setup() sizes the goal buffer from FK of the current state, i.e. horizon 1
    # (a static goal), and update_goal_tool_poses then hard-rejects any other
    # horizon.  reset_shape() drops the buffer so the first per-knot goal
    # allocates at the horizon we actually want.  There is no public way to
    # declare "I will send an H-knot path" at setup time, which is the one real
    # integration friction for the VLA use case.
    # NOTE: horizon>1 is unreachable through the public API.  setup() sizes the
    # goal buffer from FK of the current state -- ToolPose.as_goal gives
    # [B, H=1, L, 1, 3] -- update_goal_tool_poses hard-rejects any other shape,
    # setup() takes no goal argument, current_state must be 2D so H cannot be
    # raised that way, and reset_shape() does not clear link_goal_poses.  So
    # despite the documented [batch, horizon, ...] goal tensor and the
    # non_terminal_tool_pose_weight_factor knob, a multi-knot PATH cannot be
    # handed to this solver as shipped.  We therefore measure it the only way it
    # allows: horizon-1 goal re-anchored every action, i.e. MPC chasing a moving
    # target rather than tracking a known path.

    # --- run ---------------------------------------------------------------- #
    achieved_p, solve_ms = [], []
    q_cur = knot_q[0].astype(np.float32)
    H = min(a.horizon, mpc.action_horizon)

    for s in range(0, n_knots - 1, a.replan):
        k = min(H, n_knots - s)
        pos = np.zeros((1, k, len(tool_frames), 1, 3), dtype=np.float32)
        quat = np.zeros((1, k, len(tool_frames), 1, 4), dtype=np.float32)
        for j in range(k):
            for u in range(len(tool_frames)):
                pos[0, j, u, 0] = knot_p[s + j + 1, u]
                quat[0, j, u, 0] = _quat_wxyz(knot_R[s + j + 1, u])
        goal = GoalToolPose(
            tool_frames=tool_frames,
            position=torch.as_tensor(pos, device=dev),
            quaternion=torch.as_tensor(quat, device=dev),
        )
        mpc.update_goal_tool_poses(goal, run_ik=False)

        for _ in range(min(a.replan, k)):
            st = JointState.from_position(
                torch.as_tensor(q_cur, dtype=torch.float32, device=dev).view(1, -1),
                joint_names=mpc.joint_names,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            res = mpc.optimize_next_action(st)
            torch.cuda.synchronize()
            solve_ms.append((time.perf_counter() - t1) * 1e3)
            q_cur = res.next_action.position.view(-1).detach().cpu().numpy().astype(np.float32)
            p_w, _ = tree.tcp_poses(tree.fk(q_cur))
            achieved_p.append(np.asarray(p_w).copy())

    achieved = np.asarray(achieved_p)  # (N, T, 3), one per 20 ms action
    # Compare at the action grid the solver actually stepped on.
    n = achieved.shape[0]
    ref = knot_p[1 : n + 1]
    err = np.linalg.norm(achieved - ref, axis=2).reshape(-1) * 1e3

    def st(x):
        return dict(
            mean=float(statistics.fmean(x)),
            p50=float(np.percentile(x, 50)),
            p99=float(np.percentile(x, 99)),
            max=float(np.max(x)),
        )

    acc, spd = st(err), st(solve_ms)
    print()
    print(f"ACCURACY vs recorded path   RMS {np.sqrt((err**2).mean()):7.1f} mm   "
          f"p99 {acc['p99']:7.1f}   max {acc['max']:7.1f}")
    print(f"SOLVE TIME per action       mean {spd['mean']:7.2f} ms   "
          f"p99 {spd['p99']:7.2f}   max {spd['max']:7.2f}")
    print(f"  (its own command period is optimization_dt/interpolation_steps = "
          f"{dt_a / 4 * 1e3:.1f} ms = {4 / dt_a:.0f} Hz)")

    report = {
        "device": torch.cuda.get_device_name(0),
        "curobo_build_s": build_s,
        "horizon": H,
        "replan": a.replan,
        "collision": bool(a.collision),
        "n_actions": n,
        "accuracy_mm": acc,
        "accuracy_rms_mm": float(np.sqrt((err**2).mean())),
        "solve_ms": spd,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
