"""Composition of the three layers into one command-rate step.

    action chunk --[L1 interpolator]--> task pose targets
                 --[L2 stacked DLS]---> q_dot_des
                 --[L3 walls + PGS]---> q_dot
                 --[output stage]-----> q_target = q_meas + q_dot dt_c

The output stage integrates from the MEASURED joint position, not from the
previous target.  Integrating from the previous target would let the controller's
internal model drift away from the machine silently and, worse, drift
*differently* in simulation than on hardware -- which is precisely the gap this
component exists to close.  Integrating from measurement costs a little tracking
lag and buys sim/real parity, and that trade is the whole point of G0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from remoroo_lc.constants import BIG, DTYPE, POINT_DIM, TASK_DIM
from remoroo_lc.reference.diffik import DiffIk, posture_velocity
from remoroo_lc.reference.interpolator import ChunkInterpolator
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.solver import solve_qp
from remoroo_lc.reference.walls import WallBuilder
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import transform_inv


@dataclass
class StepOutput:
    """One command tick's worth of output."""

    q_target: np.ndarray  # (n,) joint position target
    effector: np.ndarray  # (sum g_i,) normalised effector command
    qd: np.ndarray  # (n,) filtered joint velocity actually commanded
    p_cmd: np.ndarray  # (T, 3) commanded TCP position, world
    R_cmd: np.ndarray  # (T, 3, 3) commanded TCP orientation, world
    p_meas: np.ndarray  # (T, 3) measured TCP position, world
    R_meas: np.ndarray  # (T, 3, 3) measured TCP orientation, world
    diag: dict = field(default_factory=dict)


class Controller:
    """The G0 controller for one cell instance."""

    def __init__(self, cell: CellSpec, delta_mode: str = "cumulative") -> None:
        self.cell = cell
        self.tree = KinematicTree(cell)
        self.interp = ChunkInterpolator(cell, delta_mode=delta_mode)
        self.diffik = DiffIk(cell)
        self.walls = WallBuilder(cell, self.tree)

        sol = cell.limits["solver"]
        self.iterations = int(sol["iterations"])
        self.box_mode = str(sol.get("box_mode", "scale"))
        self.rho = float(sol["rho"])
        self.w_joint = np.full(cell.n_joints, DTYPE(sol["joint_weight"]), dtype=DTYPE)
        self.w_post = float(cell.limits["posture"]["weight"])
        self.k_post = float(cell.limits["posture"]["k_post"])
        self.q_rest = cell.rest_posture()
        self.qd_max = cell.joint_velocity_limits()
        self.dt_c = DTYPE(1.0 / float(cell.limits["rates"]["command_hz"]))

        # Per-TCP base transforms, so pose deltas can live in the model base
        # frame while the Jacobians live in the world frame.
        self.T_base = np.stack(
            [next(m for m in cell.models if m.name == t.model).base for t in cell.tcps]
        ).astype(DTYPE)
        self.T_base_inv = np.stack([transform_inv(T) for T in self.T_base]).astype(DTYPE)

        self._lam = np.zeros(self.walls.n_rows, dtype=DTYPE)
        self.tick = 0

    # ------------------------------------------------------------------ #
    @property
    def n_rows(self) -> int:
        return self.walls.n_rows

    def tcp_pose_base(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Measured TCP poses expressed in each model's own base frame."""
        fk = self.tree.fk(q)
        p_w, R_w = self.tree.tcp_poses(fk)
        return self._to_base(p_w, R_w)

    def _to_base(self, p_w: np.ndarray, R_w: np.ndarray):
        n_t = self.cell.n_tcps
        p_b = np.zeros_like(p_w)
        R_b = np.zeros_like(R_w)
        for i in range(n_t):
            Ti = self.T_base_inv[i]
            p_b[i] = Ti[:POINT_DIM, :POINT_DIM] @ p_w[i] + Ti[:POINT_DIM, POINT_DIM]
            R_b[i] = Ti[:POINT_DIM, :POINT_DIM] @ R_w[i]
        return p_b, R_b

    def _to_world(self, p_b: np.ndarray, R_b: np.ndarray):
        n_t = self.cell.n_tcps
        p_w = np.zeros_like(p_b)
        R_w = np.zeros_like(R_b)
        for i in range(n_t):
            T = self.T_base[i]
            p_w[i] = T[:POINT_DIM, :POINT_DIM] @ p_b[i] + T[:POINT_DIM, POINT_DIM]
            R_w[i] = T[:POINT_DIM, :POINT_DIM] @ R_b[i]
        return p_w, R_w

    # ------------------------------------------------------------------ #
    def reset(self, q: np.ndarray) -> None:
        """Zero all carried state and anchor the command to the current pose."""
        p_b, R_b = self.tcp_pose_base(np.asarray(q, dtype=DTYPE))
        self.interp.reset(p_b, R_b)
        self._lam = np.zeros(self.walls.n_rows, dtype=DTYPE)
        self.tick = 0

    def set_chunk(self, actions: np.ndarray, q: np.ndarray) -> None:
        """Hand the controller a new chunk of policy actions."""
        p_b, R_b = self.tcp_pose_base(np.asarray(q, dtype=DTYPE))
        self.interp.set_chunk(actions, p_b, R_b)

    # ------------------------------------------------------------------ #
    def step(self, q_meas: np.ndarray, qd_meas: np.ndarray | None = None) -> StepOutput:
        q = np.asarray(q_meas, dtype=DTYPE).reshape(-1)

        fk = self.tree.fk(q)
        p_w, R_w = self.tree.tcp_poses(fk)
        J = self.tree.tcp_jacobians(fk)
        w = self.tree.manipulability(J)

        p_cb, R_cb, eff = self.interp.step()
        p_cw, R_cw = self._to_world(p_cb, R_cb)

        v_task = self.diffik.task_velocity(p_w, R_w, p_cw, R_cw)
        ik = self.diffik.solve(J, w, v_task)
        qd_post = posture_velocity(q, self.q_rest, self.k_post)

        walls = self.walls.assemble(q, fk)
        sol = solve_qp(
            ik.qd_des,
            qd_post,
            walls.G,
            walls.h,
            self.qd_max,
            self.w_joint,
            self.w_post,
            self.rho,
            self.iterations,
            lam_init=self._lam,
            box_mode=self.box_mode,
        )
        self._lam = sol.lam
        q_target = (q + sol.qd * self.dt_c).astype(DTYPE)
        self.tick += 1

        pair_d = walls.distance[walls.n_joint_rows :]
        joint_d = walls.distance[: walls.n_joint_rows]
        diag = {
            "tick": self.tick,
            "q_at_step": q.copy(),
            "n_active": sol.n_active,
            "n_active_rows": int(np.count_nonzero(walls.active)),
            "wall_active": bool(sol.n_active > 0),
            "max_violation": sol.max_violation,
            "slack_norm": sol.slack_norm,
            "lam": sol.lam.copy(),
            "residual": sol.residual,
            "box_clamped": sol.box_clamped,
            "min_pair_distance": float(np.min(pair_d)) if pair_d.size else float(BIG),
            "min_joint_margin": float(np.min(joint_d)) if joint_d.size else float(BIG),
            "manipulability": w.copy(),
            "damping": ik.lam.copy(),
            "task_velocity": v_task.copy(),
            "qd_des": ik.qd_des.copy(),
        }
        return StepOutput(
            q_target=q_target,
            effector=eff,
            qd=sol.qd,
            p_cmd=p_cw,
            R_cmd=R_cw,
            p_meas=p_w,
            R_meas=R_w,
            diag=diag,
        )

    # ------------------------------------------------------------------ #
    def task_error(self, out: StepOutput) -> tuple[np.ndarray, np.ndarray]:
        """Per-TCP (position error in metres, orientation error in radians)."""
        from remoroo_lc.spatial import log_so3  # noqa: PLC0415

        n_t = self.cell.n_tcps
        ep = np.zeros(n_t, dtype=DTYPE)
        er = np.zeros(n_t, dtype=DTYPE)
        for i in range(n_t):
            ep[i] = np.linalg.norm(out.p_cmd[i] - out.p_meas[i])
            er[i] = np.linalg.norm(log_so3(out.R_cmd[i] @ out.R_meas[i].T))
        return ep, er

    def stacked_task_dim(self) -> int:
        return TASK_DIM * self.cell.n_tcps
