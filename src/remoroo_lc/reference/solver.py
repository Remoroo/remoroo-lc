"""Layer 3b: the safety-filter QP, solved by fixed-iteration dual PGS.

    minimise  0.5 ||qd - qd_des||_W^2 + 0.5 w_post ||qd - qd_post||^2 + 0.5 rho ||s||^2
    s.t.      G qd <= h + s,   s >= 0,   |qd_j| <= qd_max_j

W is diagonal, so H = W + w_post I is diagonal and H^-1 is free.  The two
quadratic tracking terms collapse into one:

    H = W + w_post I,   f = -(W qd_des + w_post qd_post)

Slack elimination.  For row k the slack optimum is s_k = lambda_k / rho, so the
soft constraint is exactly the hard one with rho^-1 added to the diagonal of the
dual system.  No extra variables, and rho -> infinity recovers the hard QP.

Dual coordinate descent.  With phi_k(lambda) = G_k qd - h_k - lambda_k / rho and
D_k = G_k H^-1 G_k^T + 1/rho, the update

    lambda_k <- max(0, lambda_k + phi_k / D_k)

is the exact minimiser of the dual objective in coordinate k, so the dual
objective is non-increasing every iteration -- which is the monotonicity the unit
test checks.  Exactly 32 sweeps, always, in the row order the cell config fixed.
No convergence test: an early exit would make the arithmetic depend on the data,
and two devices that disagree about when to stop do not produce the same robot.

The box clamp is applied once, after the sweeps, per the layer's contract.  A
saturated box therefore does not corrupt the dual iteration, and the returned
q_dot is bounded no matter how badly the rows conflict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from remoroo_lc.constants import BIG, DTYPE


@dataclass
class SolveResult:
    qd: np.ndarray  # (n,) after the box clamp
    lam: np.ndarray  # (m,) dual variables, for warm starting the next tick
    n_active: int
    max_violation: float
    slack_norm: float
    residual: float
    box_clamped: int


def _dual_objective(
    lam: np.ndarray, qd_free: np.ndarray, Hinv: np.ndarray, G: np.ndarray,
    h: np.ndarray, rho: float
) -> float:
    """The dual of the slack-eliminated QP, up to a constant.

    d(lam) = 0.5 lam^T (G H^-1 G^T + I/rho) lam + lam^T (h - G qd_free)
    Minimised over lam >= 0; used only for the monotonicity test and diagnostics.
    """
    Gq = G @ qd_free
    M = (G * Hinv) @ G.T + np.eye(G.shape[0], dtype=DTYPE) / DTYPE(rho)
    return float(0.5 * lam @ (M @ lam) + lam @ (h - Gq))


def solve_qp(
    qd_des: np.ndarray,
    qd_post: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    qd_max: np.ndarray,
    w_joint: np.ndarray,
    w_post: float,
    rho: float,
    iterations: int,
    lam_init: np.ndarray | None = None,
    trace: bool = False,
    box_mode: str = "scale",
) -> SolveResult:
    """Fixed-iteration dual projected Gauss-Seidel.

    `lam_init` warm-starts from the previous tick; because the row set is fixed
    per cell, row k means the same thing on every tick and the warm start is
    meaningful rather than accidental.
    """
    n = qd_des.shape[0]
    m = G.shape[0]
    H = (w_joint + DTYPE(w_post)).astype(DTYPE)
    Hinv = (DTYPE(1.0) / H).astype(DTYPE)
    qd_free = (Hinv * (w_joint * qd_des + DTYPE(w_post) * qd_post)).astype(DTYPE)

    lam = (
        np.zeros(m, dtype=DTYPE)
        if lam_init is None or lam_init.shape[0] != m
        else lam_init.astype(DTYPE).copy()
    )
    inv_rho = DTYPE(1.0 / rho)

    # Effective diagonal of the dual system, recomputed each tick because G is.
    D = np.maximum((np.einsum("kj,j,kj->k", G, Hinv, G) + inv_rho).astype(DTYPE), DTYPE(1.0e-12))

    qd = (qd_free - Hinv * (G.T @ lam)).astype(DTYPE)

    # A row parked at h = BIG with lambda_k = 0 is provably a no-op: phi_k is
    # about -BIG, so the projected update returns 0, dl is 0, and neither lambda
    # nor q_dot changes.  Skipping those rows removes no arithmetic that had any
    # effect, so the result is identical to sweeping all m rows -- it just does
    # not spend a sweep on the hundreds of pairs that are nowhere near anything.
    # The mask depends only on h and the incoming lambda, both of which are
    # themselves deterministic, so the row set is the same on every device.
    live = np.where((h < BIG) | (lam > DTYPE(0.0)))[0]

    history: list[float] = []
    if trace:
        history.append(_dual_objective(lam, qd_free, Hinv, G, h, rho))
    for _ in range(int(iterations)):
        for k in live:
            phi = DTYPE(np.dot(G[k], qd) - h[k] - lam[k] * inv_rho)
            new = lam[k] + phi / D[k]
            new = new if new > DTYPE(0.0) else DTYPE(0.0)
            dl = DTYPE(new - lam[k])
            if dl != DTYPE(0.0):
                lam[k] = new
                qd = (qd - Hinv * (G[k] * dl)).astype(DTYPE)
        if trace:
            history.append(_dual_objective(lam, qd_free, Hinv, G, h, rho))

    # The velocity box, applied after the sweeps.  How it is applied matters more
    # than it looks.
    #
    # Clipping per joint changes the DIRECTION of q_dot, not just its magnitude,
    # and the direction is what the collision rows constrain: -n^T(J_a - J_b)q_dot
    # is a projection, so bending q_dot can turn a commanded retreat into an
    # approach on a pair the solver had already decided to back away from.
    # Measured on the 7-DOF cell driving one chain into the other: three joints
    # pinned at the box, lambda climbing to 4.0 as the damper pushed harder, and
    # the two tool spheres interpenetrating by 29 mm anyway -- with the row
    # active and the QP solved to convergence the whole time.
    #
    # Scaling the whole vector to fit instead keeps the direction exactly, so a
    # retreat stays a retreat and only gets slower.  It is a strictly weaker
    # command than the QP asked for either way, so nothing else is put at risk.
    if box_mode == "scale":
        over = float(np.max(np.abs(qd) / qd_max)) if qd.size else 0.0
        qd_boxed = (qd / DTYPE(over)).astype(DTYPE) if over > 1.0 else qd.copy()
        n_clamped = int(np.count_nonzero(np.abs(qd) > qd_max)) if over > 1.0 else 0
    elif box_mode == "clip":
        qd_boxed = np.clip(qd, -qd_max, qd_max).astype(DTYPE)
        n_clamped = int(np.count_nonzero(qd_boxed != qd))
    else:
        raise ValueError(f"box_mode must be 'scale' or 'clip', got {box_mode!r}")

    viol = (G @ qd_boxed - h).astype(DTYPE)
    result = SolveResult(
        qd=qd_boxed,
        lam=lam,
        n_active=int(np.count_nonzero(lam > DTYPE(0.0))),
        max_violation=float(np.max(viol)) if m else 0.0,
        slack_norm=float(np.linalg.norm(lam * inv_rho)) if m else 0.0,
        residual=float(np.max(np.maximum(viol, DTYPE(0.0)))) if m else 0.0,
        box_clamped=n_clamped,
    )
    if trace:
        result.dual_history = history  # type: ignore[attr-defined]
    return result


def unconstrained(
    qd_des: np.ndarray, qd_post: np.ndarray, w_joint: np.ndarray, w_post: float
) -> np.ndarray:
    """The QP's optimum when no row is active: the weighted blend of the two
    tracking terms.  Used by the property test that removing the walls reproduces
    Layer 2 exactly.

    Written as a multiply by the reciprocal, matching `solve_qp` exactly rather
    than merely mathematically: `x / H` and `x * (1/H)` differ in the last bit in
    float32, and a property test that says "exactly" has to mean it.
    """
    Hinv = (DTYPE(1.0) / (w_joint + DTYPE(w_post))).astype(DTYPE)
    return (Hinv * (w_joint * qd_des + DTYPE(w_post) * qd_post)).astype(DTYPE)
