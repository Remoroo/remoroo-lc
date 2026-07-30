#!/usr/bin/env python3
"""Draw what the controller actually did against the recorded ground truth.

Summary statistics hid two different failures behind one number, so this plots
the paths instead.  Everything is in the WORLD frame and the ground truth is our
own FK on the recorded joints -- not the vendor-reported TCP, which lives in each
arm's own base frame and carries that unit's factory calibration.  Mixing those
two frames is what made an earlier version of this comparison read 300 mm.

The recording stands in for the policy: VLA-style chunks -- K actions at the
50 Hz action rate, a new chunk anchored at the MEASURED pose after every
`replan` executed actions -- are cut from the recorded path, with the first
delta of each chunk aiming back onto it (a policy that observes state corrects
toward its intended path; one that couldn't would not be worth testing under).

Both layer-1 tracking laws run on identical chunks:

  follower        the preview law: one C1 Hermite through the chunk, velocity
                  carried through each waypoint  (task.mode: follower)
  point_to_point  the no-preview law: brake toward every waypoint as if it
                  were terminal  (the mode the plots convicted)

Usage:
    python scripts/plot_replay.py recordings/teach_rt.jsonl \
        --cell configs/rig/rig_bimanual_xarm6.yaml [--ideal] [--chunk 16 --replan 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from remoroo_lc.constants import DTYPE, POINT_DIM  # noqa: E402
from remoroo_lc.plant import Plant  # noqa: E402
from remoroo_lc.reference.controller import Controller  # noqa: E402
from remoroo_lc.reference.kinematics import KinematicTree  # noqa: E402
from remoroo_lc.rig.recording import load_recording, recording_to_tape  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402
from remoroo_lc.spatial import log_so3, transform_inv  # noqa: E402


def ground_truth(cell, tree, rec) -> np.ndarray:
    """World-frame TCP positions from OUR FK on the recorded joints."""
    out = np.zeros((len(rec.q), cell.n_tcps, POINT_DIM))
    for i in range(len(rec.q)):
        p_w, _ = tree.tcp_poses(tree.fk(rec.q[i]))
        out[i] = np.asarray(p_w)
    return out


def _base_frames(cell):
    return [
        transform_inv(next(m for m in cell.models if m.name == t.model).base)
        for t in cell.tcps
    ]


def _reference_track(cell, tree, rec, policy_hz: float):
    """The recorded path resampled at the action rate, in each model base frame."""
    t = np.asarray(rec.t, dtype=float)
    t = t - t[0]
    n_steps = int(t[-1] * policy_hz)
    inv_base = _base_frames(cell)
    want = np.linspace(0.0, t[-1], n_steps + 1)
    pick = np.searchsorted(t, want).clip(0, len(rec.q) - 1)
    p_t = np.zeros((n_steps + 1, cell.n_tcps, POINT_DIM), dtype=DTYPE)
    R_t = np.zeros((n_steps + 1, cell.n_tcps, POINT_DIM, POINT_DIM), dtype=DTYPE)
    for k, i in enumerate(pick):
        p_w, R_w = tree.tcp_poses(tree.fk(rec.q[i]))
        for u in range(cell.n_tcps):
            Ri = inv_base[u][:POINT_DIM, :POINT_DIM]
            p_t[k, u] = Ri @ p_w[u] + inv_base[u][:POINT_DIM, POINT_DIM]
            R_t[k, u] = Ri @ R_w[u]
    return n_steps, inv_base, p_t, R_t


def run(cell, tree, rec, mode: str, ideal: bool, chunk: int, replan: int):
    """Replay VLA-style chunks under the given tracking law.

    Returns (achieved world TCP positions, achieved joints)."""
    cell.limits["task"]["mode"] = mode
    hz = float(cell.limits["rates"]["command_hz"])
    policy_hz = float(cell.limits["rates"]["policy_hz"])
    ticks = int(round(hz / policy_hz))
    n_steps, inv_base, p_t, R_t = _reference_track(cell, tree, rec, policy_hz)

    ctrl, plant = Controller(cell), Plant(cell)
    q = np.asarray(rec.q[0], dtype=np.float32)
    ctrl.reset(q)
    plant.reset(q)
    achieved, joints = [], []

    for s in range(0, n_steps - 1, replan):
        k_steps = min(chunk, n_steps - s)
        # The oracle policy: from the MEASURED pose, deltas that walk back onto
        # the recorded path and then along it.
        p_w, R_w = tree.tcp_poses(tree.fk(q))
        action = np.zeros((k_steps, cell.action_dim), dtype=DTYPE)
        for u, (pose_sl, _eff) in enumerate(cell.action_slices()):
            Ri = inv_base[u][:POINT_DIM, :POINT_DIM]
            p_c = Ri @ p_w[u] + inv_base[u][:POINT_DIM, POINT_DIM]
            R_c = Ri @ R_w[u]
            b = pose_sl.start
            for j in range(k_steps):
                tgt = s + j + 1
                prev_p = p_c if j == 0 else p_t[tgt - 1, u]
                prev_R = R_c if j == 0 else R_t[tgt - 1, u]
                action[j, b : b + POINT_DIM] = p_t[tgt, u] - prev_p
                action[j, b + POINT_DIM : b + 2 * POINT_DIM] = log_so3(
                    R_t[tgt, u] @ prev_R.T
                )
        ctrl.set_chunk(action, q)
        for _ in range(min(replan, k_steps) * ticks):
            out = ctrl.step(q)
            q = out.q_target.astype(np.float32) if ideal else plant.step(out.q_target)[0]
            achieved.append(out.p_meas.copy())
            joints.append(np.asarray(q).copy())

    return np.asarray(achieved), np.asarray(joints)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--ideal", action="store_true", help="perfect servo; removes the plant")
    ap.add_argument("--chunk", type=int, default=16, help="actions per chunk (VLA horizon)")
    ap.add_argument("--replan", type=int, default=8, help="actions executed before re-anchor")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "rig")
    a = ap.parse_args()

    cell = load_cell(a.cell)
    tree = KinematicTree(cell)
    rec = load_recording(a.recording)
    t_rec = np.asarray(rec.t, dtype=float)
    t_rec = t_rec - t_rec[0]
    gt = ground_truth(cell, tree, rec)
    hz = float(cell.limits["rates"]["command_hz"])

    runs = {}
    for mode in ("follower", "point_to_point"):
        p, qj = run(cell, tree, rec, mode, a.ideal, a.chunk, a.replan)
        runs[mode] = {"p": p, "q": qj, "t": np.arange(p.shape[0]) / hz}
        print(f"{mode:14s} {p.shape[0]} ticks")

    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.recording.stem
    colour = {"point_to_point": "tab:red", "follower": "tab:blue"}
    n_t = cell.n_tcps

    # ---- 3D paths --------------------------------------------------------- #
    fig = plt.figure(figsize=(7 * n_t, 6.5))
    for u in range(n_t):
        ax = fig.add_subplot(1, n_t, u + 1, projection="3d")
        ax.plot(*gt[:, u].T, color="k", lw=2.0, label="recorded (ground truth)")
        for mode, r in runs.items():
            ax.plot(*r["p"][:, u].T, color=colour[mode], lw=0.9, alpha=0.85, label=mode)
        ax.scatter(*gt[0, u], color="k", s=45, marker="o", label="start")
        ax.set_title(f"{cell.tcps[u].name}: TCP path in world frame")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        if u == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"{stem}: achieved vs recorded  "
        f"({'ideal servo' if a.ideal else 'simulated plant'})"
    )
    fig.tight_layout()
    fig.savefig(a.out / f"{stem}_paths3d.png", dpi=130)
    plt.close(fig)

    # ---- per-axis vs time, and error -------------------------------------- #
    fig, axes = plt.subplots(POINT_DIM + 1, n_t, figsize=(8 * n_t, 12), sharex=True)
    axes = np.atleast_2d(axes.reshape(POINT_DIM + 1, n_t))
    for u in range(n_t):
        for k, name in enumerate("xyz"):
            ax = axes[k, u]
            ax.plot(t_rec, gt[:, u, k], color="k", lw=1.6, label="recorded")
            for mode, r in runs.items():
                ax.plot(r["t"], r["p"][:, u, k], color=colour[mode], lw=0.8, alpha=0.85,
                        label=mode)
            ax.set_ylabel(f"{name} [m]")
            if k == 0:
                ax.set_title(f"{cell.tcps[u].name}")
                ax.legend(fontsize=8, ncol=3)
        ax = axes[POINT_DIM, u]
        for mode, r in runs.items():
            ref = np.stack(
                [np.interp(r["t"], t_rec, gt[:, u, k]) for k in range(POINT_DIM)], axis=1
            )
            err = np.linalg.norm(r["p"][:, u] - ref, axis=1) * 1e3
            ax.plot(r["t"], err, color=colour[mode], lw=0.8,
                    label=f"{mode}: RMS {np.sqrt((err**2).mean()):.0f} mm")
        ax.set_ylabel("error [mm]")
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{stem}: per-axis position and tracking error")
    fig.tight_layout()
    fig.savefig(a.out / f"{stem}_axes.png", dpi=130)
    plt.close(fig)

    # ---- joint space ------------------------------------------------------ #
    n_j = cell.n_joints
    rows = int(np.ceil(n_j / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 2.6 * rows), sharex=True)
    axes = axes.ravel()
    labels = cell.joint_labels()
    for j in range(n_j):
        ax = axes[j]
        ax.plot(t_rec, np.asarray(rec.q)[:, j], color="k", lw=1.4, label="recorded")
        for mode, r in runs.items():
            ax.plot(r["t"], r["q"][:, j], color=colour[mode], lw=0.8, alpha=0.85, label=mode)
        ax.set_title(labels[j], fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(n_j, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{stem}: joint trajectories [rad]")
    fig.tight_layout()
    fig.savefig(a.out / f"{stem}_joints.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote {a.out / (stem + '_paths3d.png')}")
    print(f"wrote {a.out / (stem + '_axes.png')}")
    print(f"wrote {a.out / (stem + '_joints.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
