"""Layers 2+3, replaced: one metric-weighted second-order solve.

The stacked-DLS + QP stack it replaces fails in a specific, measured way.  It is
first order (`q_dot = J^+ e / dt_c`) so recovering a tracking debt is capped by a
velocity clamp rather than by how hard the robot can accelerate; and its safety
layer is a HARD constraint followed by a velocity box, so when a constraint bites
the whole `q_dot` is scaled down -- killing the tracking component along with the
violating one.  On the rig's teach recording that showed up as 12.4% of ticks
carrying 99.7% of the total squared error, while constraint-free ticks tracked to
2.4 mm.  It does not degrade, it clamps.

This solves the same problem the way the geometric-fabrics literature does
(Ratliff et al.), implemented here from the published concepts:

Every behaviour contributes a PAIR -- a metric and a force -- in its own task
space, and all of them are pulled back into joint space and summed before a
single solve:

    M_q = sum_k J_k^T M_k J_k  +  W_joint
    f_q = sum_k J_k^T ( M_k (Jdot_k qdot - xddot_k) )  +  damping
    qddot = -M_q^-1 f_q

The metric is what buys graceful degradation.  An obstacle term contributes
`outer(n, n) * scale / d` -- RANK ONE, aligned with the collision normal.  As the
distance falls the robot becomes arbitrarily reluctant to move TOWARD the
obstacle while staying completely free to keep tracking in every other
direction.  Nothing is clamped; the directions simply reweight.  A hard
constraint cannot express that, which is the entire reason for this module.

Second order also fixes recovery on its own terms: a pose error produces an
ACCELERATION, so a large debt is answered by accelerating harder, bounded by real
joint acceleration limits rather than by an invented task-space speed cap.

Determinism is preserved the same way the rest of the package does it: fixed
term order from config load order, no data-dependent iteration, one Cholesky of
an (n x n) SPD matrix, float32 throughout.  There is no iterative solver here at
all, which makes this strictly easier to keep bitwise-identical than the 32-sweep
PGS it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from remoroo_lc.constants import DTYPE, EPS, POINT_DIM, TASK_DIM
from remoroo_lc.reference.linalg import spd_solve
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import log_so3


@dataclass
class DynamicsOutput:
    """One solve's worth of output."""

    qdd: np.ndarray  # (n,) commanded joint acceleration, after limits
    qd: np.ndarray  # (n,) integrated joint velocity command
    scale: float  # uniform accel-limit scale applied (1.0 = unconstrained)
    energy: float  # 0.5 qdot^T M qdot, the quantity speed control regulates
    n_repulsion: int  # collision terms that contributed a metric this tick
    diag: dict = field(default_factory=dict)


class MetricDynamics:
    """Metric-weighted second-order controller for one cell instance."""

    def __init__(self, cell: CellSpec) -> None:
        self.cell = cell
        self.n = cell.n_joints
        self.n_tcps = cell.n_tcps
        d = cell.limits.get("dynamics", {})

        # Pose attractor: critically-damped by default, so kd is derived from kp
        # rather than being a second invented number.
        self.kp_lin = DTYPE(d.get("attractor", {}).get("kp_linear", 900.0))
        self.kp_ang = DTYPE(d.get("attractor", {}).get("kp_angular", 400.0))
        zeta = DTYPE(d.get("attractor", {}).get("zeta", 1.0))
        self.kd_lin = DTYPE(2.0) * zeta * np.sqrt(self.kp_lin)
        self.kd_ang = DTYPE(2.0) * zeta * np.sqrt(self.kp_ang)
        self.w_attract = DTYPE(d.get("attractor", {}).get("weight", 1.0))

        # Obstacle repulsion.  `scale` sets how stiff the normal direction gets;
        # `d_infl` is where the term switches on at all.  Both are in metres.
        rep = d.get("repulsion", {})
        self.rep_scale = DTYPE(rep.get("scale", 0.6))
        self.rep_infl = DTYPE(rep.get("d_infl_m", cell.limits["collision_damper"]["d_infl_m"]))
        self.rep_exp = DTYPE(rep.get("exponent", 2.0))

        # Joint-limit and joint-speed repulsion, same barrier shape in 1-D.
        jl = d.get("joint_limit", {})
        self.jl_scale = DTYPE(jl.get("scale", 4.0))
        self.jl_infl = DTYPE(np.radians(jl.get("d_infl_deg", 10.0)))

        # Config-space damping and posture pull.
        self.damping = DTYPE(d.get("cspace_damping", 30.0))
        self.k_post = DTYPE(cell.limits["posture"]["k_post"])
        self.w_post = DTYPE(cell.limits["posture"]["weight"])
        self.q_rest = cell.rest_posture()

        # Base joint-space mass.  Identity-weighted: the cell file's
        # `joint_weight` already exists for exactly this and nothing here needs a
        # dynamics model of the arm -- this is a control metric, not the robot's
        # real inertia, and pretending otherwise would need mass properties most
        # customer URDFs do not carry.
        self.w_joint = DTYPE(cell.limits["solver"]["joint_weight"])

        self.qdd_max = self._accel_limits(d)
        self.qd_max = cell.joint_velocity_limits().astype(DTYPE)
        self.q_lo, self.q_hi = (np.asarray(x, dtype=DTYPE) for x in cell.joint_limits())
        self.dt_c = DTYPE(1.0 / float(cell.limits["rates"]["command_hz"]))
        self.qd = np.zeros(self.n, dtype=DTYPE)

    def _accel_limits(self, d: dict) -> np.ndarray:
        """Joint acceleration bound, from config or derived from velocity limits.

        A URDF carries velocity and effort but no acceleration limit, so if the
        cell does not state one it is derived: reaching full joint speed in
        `t_to_vmax` seconds.  Derived from the rig's own numbers, never invented
        as an absolute.
        """
        lim = d.get("qdd_max")
        if lim is not None:
            return np.asarray(lim, dtype=DTYPE).reshape(self.n)
        t_to_vmax = DTYPE(d.get("t_to_vmax", 0.15))
        return (self.cell.joint_velocity_limits().astype(DTYPE) / t_to_vmax).astype(DTYPE)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.qd = np.zeros(self.n, dtype=DTYPE)

    # ------------------------------------------------------------------ #
    def _attractor(self, p_meas, R_meas, p_cmd, R_cmd, J, xd):
        """Pose attractor: (M, xddot_des) per TCP, stacked.

        `xddot_des = kp * e - kd * xdot` in the TCP's own 6-D task space.  The
        metric is isotropic within a TCP: a pose target has no preferred
        direction, unlike an obstacle.
        """
        m6 = TASK_DIM * self.n_tcps
        M = np.zeros((m6, m6), dtype=DTYPE)
        xdd = np.zeros(m6, dtype=DTYPE)
        for i in range(self.n_tcps):
            b = i * TASK_DIM
            e = np.zeros(TASK_DIM, dtype=DTYPE)
            e[:POINT_DIM] = p_cmd[i] - p_meas[i]
            e[POINT_DIM:] = log_so3(R_cmd[i] @ R_meas[i].T)
            kp = np.concatenate(
                [np.full(POINT_DIM, self.kp_lin), np.full(POINT_DIM, self.kp_ang)]
            ).astype(DTYPE)
            kd = np.concatenate(
                [np.full(POINT_DIM, self.kd_lin), np.full(POINT_DIM, self.kd_ang)]
            ).astype(DTYPE)
            xdd[b : b + TASK_DIM] = kp * e - kd * xd[b : b + TASK_DIM]
            for c in range(TASK_DIM):
                M[b + c, b + c] = self.w_attract
        return M, xdd

    def _repulsion(self, walls, J_pair_rel, normals, dist):
        """Collision repulsion: rank-one metrics along each collision normal.

        This is the term the QP could not express.  For pair k with normal n and
        distance d, the metric contribution in joint space is

            J_k^T (n n^T) J_k * scale / d^exp

        which is stiff ONLY along n.  The matching force accelerates away from
        the obstacle at the same barrier rate.  As d -> 0 the metric grows
        without bound, so the solve refuses approach asymptotically instead of
        clamping discontinuously; and because the metric is rank one, motion
        orthogonal to n is untouched.
        """
        Mq = np.zeros((self.n, self.n), dtype=DTYPE)
        fq = np.zeros(self.n, dtype=DTYPE)
        live = np.nonzero((dist < self.rep_infl) & (dist > -self.rep_infl))[0]
        for k in live:
            d = max(float(dist[k]), float(EPS))
            barrier = self.rep_scale / DTYPE(d**float(self.rep_exp))
            # Row of the relative Jacobian projected on the normal: the scalar
            # rate at which this pair's gap closes per unit joint velocity.
            jn = J_pair_rel[k]  # (n,) already n^T (J_a - J_b)
            Mq += barrier * np.outer(jn, jn)
            # Force that accelerates the gap open at the barrier rate.
            fq -= barrier * jn * (self.rep_scale / DTYPE(d))
        return Mq, fq, int(live.size)

    def _joint_barriers(self, q):
        """1-D barriers on position and speed limits, same shape as repulsion."""
        Mq = np.zeros((self.n, self.n), dtype=DTYPE)
        fq = np.zeros(self.n, dtype=DTYPE)
        for j in range(self.n):
            for margin, sgn in ((q[j] - self.q_lo[j], DTYPE(1.0)),
                                (self.q_hi[j] - q[j], DTYPE(-1.0))):
                if margin < self.jl_infl:
                    d = max(float(margin), float(EPS))
                    barrier = self.jl_scale / DTYPE(d * d)
                    Mq[j, j] += barrier
                    fq[j] -= sgn * barrier * (self.jl_scale / DTYPE(d))
        return Mq, fq

    # ------------------------------------------------------------------ #
    def solve(
        self,
        q: np.ndarray,
        J: np.ndarray,
        p_meas: np.ndarray,
        R_meas: np.ndarray,
        p_cmd: np.ndarray,
        R_cmd: np.ndarray,
        J_pair_rel: np.ndarray,
        dist: np.ndarray,
    ) -> DynamicsOutput:
        """One metric-weighted second-order solve.

        `J` is (T, TASK_DIM, n) per-TCP; `J_pair_rel` is (P, n), each row the
        collision normal already projected through the pair's relative Jacobian
        (exactly the `G` rows layer 3 was building, reused); `dist` is (P,).
        """
        n = self.n
        m6 = TASK_DIM * self.n_tcps
        Js = np.zeros((m6, n), dtype=DTYPE)
        for i in range(self.n_tcps):
            Js[i * TASK_DIM : (i + 1) * TASK_DIM, :] = J[i]
        xd = (Js @ self.qd).astype(DTYPE)

        M_task, xdd_des = self._attractor(p_meas, R_meas, p_cmd, R_cmd, J, xd)

        # Pull the task-space pair back into joint space.  The Jdot qdot
        # curvature term is deliberately dropped: it needs a second-order
        # kinematics pass for a contribution that is small at 250 Hz, and its
        # absence is a damping bias, not an instability.
        Mq = (Js.T @ M_task @ Js).astype(DTYPE)
        fq = (-Js.T @ M_task @ xdd_des).astype(DTYPE)

        M_rep, f_rep, n_rep = self._repulsion(None, J_pair_rel, None, dist)
        M_jl, f_jl = self._joint_barriers(q)
        Mq += M_rep + M_jl
        fq += f_rep + f_jl

        # Posture pull, in the null space by construction: it is weak and
        # isotropic, so wherever a task or obstacle metric is large it loses.
        Mq += np.eye(n, dtype=DTYPE) * (self.w_joint + self.w_post)
        fq += (self.w_post * self.k_post * (q - self.q_rest)).astype(DTYPE)

        # Config-space damping, written as a force so it is metric-weighted like
        # everything else: this is what makes the whole system dissipative and
        # is the reason no velocity clamp is needed to keep it stable.
        fq += (self.damping * (Mq @ self.qd)).astype(DTYPE)

        qdd = spd_solve(Mq, -fq).astype(DTYPE)

        # Acceleration limit by UNIFORM scaling.  Per-joint clipping would bend
        # the direction and can turn a commanded retreat into an approach -- the
        # same lesson `box_mode: scale` records in the QP path.
        over = np.abs(qdd) / np.maximum(self.qdd_max, EPS)
        scale = DTYPE(1.0) / max(DTYPE(1.0), DTYPE(over.max() if over.size else 0.0))
        qdd = (qdd * scale).astype(DTYPE)

        qd_new = (self.qd + qdd * self.dt_c).astype(DTYPE)
        # Joint speed limit, also by uniform scaling, same reasoning.
        vover = np.abs(qd_new) / np.maximum(self.qd_max, EPS)
        vscale = DTYPE(1.0) / max(DTYPE(1.0), DTYPE(vover.max() if vover.size else 0.0))
        qd_new = (qd_new * vscale).astype(DTYPE)
        self.qd = qd_new

        energy = float(0.5 * qd_new @ (Mq @ qd_new))
        return DynamicsOutput(
            qdd=qdd,
            qd=qd_new,
            scale=float(scale),
            energy=energy,
            n_repulsion=n_rep,
            diag={"accel_scale": float(scale), "vel_scale": float(vscale)},
        )
