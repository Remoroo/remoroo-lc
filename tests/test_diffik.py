"""Layer 2: the stacked solve, damping ramp, and the shared-joint claim."""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import TASK_DIM
from remoroo_lc.reference.diffik import DiffIk, posture_velocity
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.linalg import spd_solve
from tests.conftest import random_states


def _models_are_disjoint(cell) -> bool:
    """True when no two TCPs share a model (and therefore share no joints)."""
    return len({t.model for t in cell.tcps}) == cell.n_tcps


def test_damping_ramp_endpoints(cell):
    ik = DiffIk(cell)
    thresh = np.asarray([t.w_thresh for t in cell.tcps], dtype=np.float32)
    assert np.allclose(ik.damping(thresh), ik.lambda_min, atol=1e-6)
    assert np.allclose(ik.damping(thresh * 2.0), ik.lambda_min, atol=1e-6)
    at_zero = ik.damping(np.zeros_like(thresh))
    assert np.allclose(at_zero, ik.lambda_min + ik.lambda_max, atol=1e-6)
    # Monotone non-increasing in w.
    ws = np.linspace(0.0, float(thresh[0]) * 1.5, 25, dtype=np.float32)
    lams = [float(ik.damping(np.full(cell.n_tcps, w, dtype=np.float32))[0]) for w in ws]
    assert all(b <= a + 1e-7 for a, b in zip(lams, lams[1:]))


def test_stacked_solve_matches_per_chain_when_disjoint(cell):
    """Disjoint chains: the stacked system is block diagonal and must reduce."""
    if not _models_are_disjoint(cell):
        return
    tree = KinematicTree(cell)
    ik = DiffIk(cell)
    for q in random_states(cell, 6, seed=11):
        fk = tree.fk(q)
        J = tree.tcp_jacobians(fk)
        w = tree.manipulability(J)
        v = np.arange(cell.task_dim, dtype=np.float32) * 0.01 - 0.05
        stacked = ik.solve(J, w, v).qd_des
        lam = ik.damping(w)
        per_chain = np.zeros(cell.n_joints, dtype=np.float32)
        for i in range(cell.n_tcps):
            Ji = J[i]
            A = (Ji @ Ji.T).astype(np.float32) + (lam[i] ** 2) * np.eye(TASK_DIM, dtype=np.float32)
            y = spd_solve(A, v[i * TASK_DIM : (i + 1) * TASK_DIM])
            per_chain += Ji.T @ y
        assert np.max(np.abs(stacked - per_chain)) < 2e-4, (
            "disjoint chains must reduce exactly to independent per-chain DLS"
        )


def test_shared_joints_are_not_double_counted():
    """The branched cell: naive per-chain solving would over-drive the trunk."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "branched_trunk.yaml")
    tree = KinematicTree(cell)
    ik = DiffIk(cell)
    q = cell.rest_posture()
    fk = tree.fk(q)
    J = tree.tcp_jacobians(fk)
    w = tree.manipulability(J)
    trunk = cell.joint_labels().index("torso/trunk")

    # Both TCPs asked to move the same way about the trunk axis.
    v = np.zeros(cell.task_dim, dtype=np.float32)
    v[0] = 0.10
    v[TASK_DIM] = 0.10
    stacked = ik.solve(J, w, v).qd_des

    lam = ik.damping(w)
    summed = np.zeros(cell.n_joints, dtype=np.float32)
    for i in range(cell.n_tcps):
        Ji = J[i]
        A = (Ji @ Ji.T).astype(np.float32) + (lam[i] ** 2) * np.eye(TASK_DIM, dtype=np.float32)
        summed += Ji.T @ spd_solve(A, v[i * TASK_DIM : (i + 1) * TASK_DIM])

    assert abs(stacked[trunk]) < abs(summed[trunk]) - 1e-6, (
        "the stacked solve must arbitrate the shared trunk, not add both demands"
    )


def test_stacked_solve_reproduces_the_task_velocity_when_well_conditioned(cell):
    """Away from singularities, J q_dot should recover v to within the damping."""
    tree = KinematicTree(cell)
    ik = DiffIk(cell)
    q = cell.rest_posture()
    fk = tree.fk(q)
    J = tree.tcp_jacobians(fk)
    w = tree.manipulability(J)
    v = np.zeros(cell.task_dim, dtype=np.float32)
    for i in range(cell.n_tcps):
        v[i * TASK_DIM] = 0.05  # 5 cm/s along world x on every TCP
    qd = ik.solve(J, w, v).qd_des
    achieved = J.reshape(cell.task_dim, cell.n_joints) @ qd
    if cell.n_joints >= cell.task_dim:
        assert np.max(np.abs(achieved - v)) < 5e-3
    else:
        # Over-constrained (the branched cell): a least-squares compromise, but a
        # bounded and correlated one.
        assert np.dot(achieved, v) > 0.0
        assert np.max(np.abs(achieved)) < 2.0 * np.max(np.abs(v)) + 1e-3


def test_task_velocity_is_norm_clamped(cell):
    tree = KinematicTree(cell)
    ik = DiffIk(cell)
    fk = tree.fk(cell.rest_posture())
    p, R = tree.tcp_poses(fk)
    p_cmd = p + np.float32(10.0)  # absurdly far
    v = ik.task_velocity(p, R, p_cmd, R)
    for i in range(cell.n_tcps):
        lin = v[i * TASK_DIM : i * TASK_DIM + 3]
        assert np.linalg.norm(lin) <= float(ik.v_max_lin) + 1e-5
        # Direction preserved: a norm clamp must not bend the commanded path.
        want = np.ones(3, dtype=np.float32) / np.sqrt(3.0)
        assert np.allclose(lin / np.linalg.norm(lin), want, atol=1e-4)


def test_posture_velocity_points_home(cell):
    q_rest = cell.rest_posture()
    q = q_rest + np.float32(0.3)
    qd = posture_velocity(q, q_rest, 1.0)
    assert np.all(qd < 0.0)
    assert np.allclose(posture_velocity(q_rest, q_rest, 1.0), 0.0)
