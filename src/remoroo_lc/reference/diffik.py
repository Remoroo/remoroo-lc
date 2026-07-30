"""Layer 2: stacked damped differential inverse kinematics.

Every TCP contributes TASK_DIM rows to one stacked system over the whole cell:

    q_dot_des = J^T (J J^T + Lambda)^-1 v,     Lambda = blockdiag(lambda_i^2 I)

with J in R^(TASK_DIM*T x n) built from the full-cell Jacobians, zero in the
columns of joints that are off a given TCP's path.

There is exactly one solve, never one per chain.  When the chains are disjoint
the stacked system is block diagonal and this reduces, algebraically, to
independent per-chain damped least squares; when they share joints -- a trunk
feeding two limbs -- the shared columns appear in both row blocks and the single
solve arbitrates them.  Solving per chain and summing would double-count the
shared joint and is wrong; that is the whole reason for the stacked form, and the
branched cell in the test matrix is what keeps it honest.

Damping is per TCP and rises as that TCP's own manipulability falls, so a chain
approaching a singularity gets damped without dragging down a healthy chain
sharing the same solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from remoroo_lc.constants import DTYPE, EPS, POINT_DIM, TASK_DIM
from remoroo_lc.reference.linalg import spd_solve
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import log_so3


@dataclass
class DiffIkResult:
    qd_des: np.ndarray  # (n,)
    v_task: np.ndarray  # (TASK_DIM*T,) stacked, world frame
    lam: np.ndarray  # (T,) damping actually applied
    w: np.ndarray  # (T,) manipulability measure
    #: Uniform scale the feasibility governor applied (1.0 = request was feasible).
    governor: float = 1.0


class DiffIk:
    """Stacked damped least squares over every TCP in the cell."""

    def __init__(self, cell: CellSpec) -> None:
        self.cell = cell
        d = cell.limits["diffik"]
        self.lambda_min = DTYPE(d["lambda_min"])
        self.lambda_max = DTYPE(d["lambda_max"])
        self.w_thresh = np.asarray([t.w_thresh for t in cell.tcps], dtype=DTYPE)
        task = cell.limits["task"]
        # See limits.yaml: this bounds the SERVO CORRECTION, not the trajectory.
        scale = DTYPE(d.get("v_clamp_scale", 4.0))
        self.v_max_lin = DTYPE(task["linear"]["v_max"]) * scale
        self.v_max_ang = DTYPE(task["angular"]["v_max"]) * scale
        self.dt_c = DTYPE(1.0 / float(cell.limits["rates"]["command_hz"]))
        # Feedback gain, 1/s.  Defaults to 1/dt_c, which is what this layer used
        # when it had no feedforward term and had to generate the whole motion
        # from position error.  With feedforward it should be a servo bandwidth.
        inv_dt = DTYPE(1.0) / self.dt_c
        self.kp_lin = DTYPE(d.get("kp_linear", float(inv_dt)))
        self.kp_ang = DTYPE(d.get("kp_angular", float(inv_dt)))

    # ------------------------------------------------------------------ #
    def damping(self, w: np.ndarray) -> np.ndarray:
        """lambda_i = lambda_min + lambda_max * max(0, 1 - w_i/w_thresh_i)^2."""
        frac = np.maximum(DTYPE(0.0), DTYPE(1.0) - w / self.w_thresh).astype(DTYPE)
        return (self.lambda_min + self.lambda_max * frac * frac).astype(DTYPE)

    def task_velocity(
        self,
        p: np.ndarray,
        R: np.ndarray,
        p_cmd: np.ndarray,
        R_cmd: np.ndarray,
        v_ff: np.ndarray | None = None,
        windup: float = 1.0,
    ) -> np.ndarray:
        """Stacked task velocity, world frame, clamped in norm per TCP.

        FEEDFORWARD PLUS FEEDBACK.  `v_ff` is the command trajectory's own task
        velocity, which Layer 1 already knows -- with a policy that hands over
        future poses, the velocity along the path is not something this layer
        should have to infer.  Without it this reduces to `e / dt_c`, a pure
        proportional servo that must BUILD UP position error before it produces
        any speed, and at path speed that error is what then demands an
        impossible recovery velocity and saturates the joint box.  Measured on
        the rig recording, that feedback loop -- error, clamp, more error -- is
        where essentially all of the tracking error came from.  With
        feedforward, zero error still commands the right velocity and the
        feedback term only has to correct the residual.

        Clamping by norm rather than per axis keeps the commanded direction
        intact under saturation; a per-axis clamp here would bend the path.  The
        per-axis clamps live in Layer 1, where they shape the trajectory on
        purpose.  The bound here is the trajectory limit times
        diffik.v_clamp_scale, because this quantity is a correction and not a
        speed -- see limits.yaml.
        """
        n_t = p.shape[0]
        v = np.zeros(n_t * TASK_DIM, dtype=DTYPE)
        # Feedback gain, 1/s.  The historical value is 1/dt_c = 250 -- correct
        # when feedback was the ONLY source of motion, and far too stiff once
        # feedforward carries the path: at 250/s a 10 mm residual asks for
        # 2.5 m/s of correction, which saturates the joint box and manufactures
        # the very error it is reacting to.  With feedforward the feedback only
        # has to close a residual, so it wants a servo bandwidth, not 1/dt.
        # ANTI-WINDUP.  `windup` is the previous tick's feasibility-governor
        # scale.  When the governor is cutting the request, the residual is not
        # evidence that we should push harder -- it is evidence the kinematics
        # cannot deliver that direction right now.  Left ungoverned the feedback
        # term amplifies that residual into a larger request, which the governor
        # cuts again: the classic integrator-windup shape, and measured on the
        # rig recording the governed 9.4% of ticks carried 82% of all squared
        # error.  Scaling the FEEDBACK by the same factor keeps the feedforward
        # (which is feasible by construction) at full authority and stops the
        # loop pushing into a wall it cannot move.
        kp_lin = self.kp_lin * DTYPE(windup)
        kp_ang = self.kp_ang * DTYPE(windup)
        for i in range(n_t):
            lin = (p_cmd[i] - p[i]) * kp_lin
            ang = log_so3(R_cmd[i] @ R[i].T) * kp_ang
            if v_ff is not None:
                lin = lin + v_ff[i, :POINT_DIM]
                ang = ang + v_ff[i, POINT_DIM:]
            nl = DTYPE(np.linalg.norm(lin))
            na = DTYPE(np.linalg.norm(ang))
            if nl > self.v_max_lin:
                lin = lin * (self.v_max_lin / (nl + EPS))
            if na > self.v_max_ang:
                ang = ang * (self.v_max_ang / (na + EPS))
            v[i * TASK_DIM : i * TASK_DIM + POINT_DIM] = lin
            v[i * TASK_DIM + POINT_DIM : (i + 1) * TASK_DIM] = ang
        return v

    def solve(
        self,
        J_tcp: np.ndarray,
        w: np.ndarray,
        v_task: np.ndarray,
        qd_max: np.ndarray | None = None,
    ) -> DiffIkResult:
        """One damped least-squares solve over the whole cell.

        With `qd_max`, the result is put through a FEASIBILITY GOVERNOR: if the
        solution asks a joint to exceed its own URDF velocity limit, the whole
        vector is scaled down uniformly until it fits.

        This matters more than it looks.  Near a singular direction the damped
        inverse still amplifies by up to 1/lambda^2, and on the rig recording that
        produced requests of 202 rad/s against a 3.14 rad/s limit -- 130x what
        the arm actually needed for the same motion.  Handing that to Layer 3
        makes its velocity box scale the whole solution down by the same 130x,
        which annihilates the tracking component along with the excess and is the
        "clamps rather than degrades" failure.  Governing here instead means the
        command that reaches Layer 3 is one the robot can execute, so the box is
        left to do its real job and the dampers arbitrate against a sane request.
        Because the map from v_task to qd_des is linear, scaling qd_des uniformly
        IS slowing down along the commanded path with its direction preserved --
        the honest response to a command the kinematics cannot deliver, and the
        bound comes from the URDF rather than from a number anybody chose.
        """
        n_t, _, n = J_tcp.shape
        m = n_t * TASK_DIM
        J = J_tcp.reshape(m, n)
        lam = self.damping(w)
        A = (J @ J.T).astype(DTYPE)
        for i in range(n_t):
            d = lam[i] * lam[i]
            for r in range(TASK_DIM):
                k = i * TASK_DIM + r
                A[k, k] = A[k, k] + d
        y = spd_solve(A, v_task)
        qd_des = (J.T @ y).astype(DTYPE)
        gov = DTYPE(1.0)
        if qd_max is not None:
            over = np.abs(qd_des) / np.maximum(np.asarray(qd_max, dtype=DTYPE), EPS)
            worst = DTYPE(over.max()) if over.size else DTYPE(0.0)
            if worst > DTYPE(1.0):
                gov = DTYPE(1.0) / worst
                qd_des = (qd_des * gov).astype(DTYPE)
        return DiffIkResult(qd_des=qd_des, v_task=v_task, lam=lam, w=w, governor=float(gov))


def posture_velocity(q: np.ndarray, q_rest: np.ndarray, k_post: float) -> np.ndarray:
    """q_dot_post = -k_post (q - q_rest).

    Present for every cell.  Harmless when n == TASK_DIM*T (it is competing with
    a task term that already determines q_dot uniquely), and load-bearing when
    n > TASK_DIM*T or when joints are shared, where it is what makes the null
    space resolve to the same place every time instead of wherever the solver
    happened to start.
    """
    return (DTYPE(-k_post) * (q - q_rest)).astype(DTYPE)
