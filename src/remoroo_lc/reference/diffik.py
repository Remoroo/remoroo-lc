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
    ) -> np.ndarray:
        """Stacked task velocity, world frame, clamped in norm per TCP.

        Clamping by norm rather than per axis keeps the commanded direction
        intact under saturation; a per-axis clamp here would bend the path.  The
        per-axis clamps live in Layer 1, where they shape the trajectory on
        purpose.  The bound here is the trajectory limit times
        diffik.v_clamp_scale, because this quantity is a correction and not a
        speed -- see limits.yaml.
        """
        n_t = p.shape[0]
        v = np.zeros(n_t * TASK_DIM, dtype=DTYPE)
        inv_dt = DTYPE(1.0) / self.dt_c
        for i in range(n_t):
            lin = (p_cmd[i] - p[i]) * inv_dt
            ang = log_so3(R_cmd[i] @ R[i].T) * inv_dt
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
    ) -> DiffIkResult:
        """One damped least-squares solve over the whole cell."""
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
        return DiffIkResult(qd_des=qd_des, v_task=v_task, lam=lam, w=w)


def posture_velocity(q: np.ndarray, q_rest: np.ndarray, k_post: float) -> np.ndarray:
    """q_dot_post = -k_post (q - q_rest).

    Present for every cell.  Harmless when n == TASK_DIM*T (it is competing with
    a task term that already determines q_dot uniquely), and load-bearing when
    n > TASK_DIM*T or when joints are shared, where it is what makes the null
    space resolve to the same place every time instead of wherever the solver
    happened to start.
    """
    return (DTYPE(-k_post) * (q - q_rest)).astype(DTYPE)
