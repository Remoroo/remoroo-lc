"""Tree FK and Jacobians, checked against an independent float64 implementation.

The reference implementation runs in float32 by design.  A finite-difference
check at float32 precision would be dominated by rounding, so this file carries a
second, deliberately naive FK written in float64 straight from the URDF -- no
shared code with `reference.kinematics` beyond the parser -- and differentiates
that.  Two independent implementations agreeing is a stronger statement than one
implementation agreeing with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.constants import POINT_DIM, TASK_DIM
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.spatial import exp_so3, log_so3
from tests.conftest import random_states


# --------------------------------------------------------------------------- #
# independent float64 forward kinematics
# --------------------------------------------------------------------------- #


def _fk64(cell, q):
    """Link world transforms for the whole cell, in float64, from the URDF."""
    out = {}
    k = 0
    for m in cell.models:
        urdf = cell.urdfs[m.name]
        vals = {}
        for jn in m.joint_names:
            vals[jn] = float(q[k])
            k += 1
        vals.update({jn: v for jn, v in m.locked.items()})
        T = {urdf.root: np.asarray(m.base, dtype=np.float64)}
        for link in urdf.ordered_links():
            if link == urdf.root:
                continue
            j = urdf.by_name[urdf.parent_joint[link]]
            Tj = T[j.parent] @ np.asarray(j.origin, dtype=np.float64)
            v = vals.get(j.name, 0.0 if j.mimic is None else float(j.mimic[2]))
            M = np.eye(4)
            axis = np.asarray(j.axis, dtype=np.float64)
            if j.jtype in ("revolute", "continuous"):
                th = v
                K = np.array(
                    [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
                )
                M[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            elif j.jtype == "prismatic":
                M[:3, 3] = axis * v
            Tj = Tj @ M
            T[link] = Tj
        for link, Tl in T.items():
            out[f"{m.name}/{link}"] = Tl
    return out


def _tcp_pose64(cell, q, i):
    T = _fk64(cell, q)[f"{cell.tcps[i].model}/{cell.tcps[i].frame}"]
    T = T @ np.asarray(cell.tcps[i].offset, dtype=np.float64)
    return T[:3, 3].copy(), T[:3, :3].copy()


# --------------------------------------------------------------------------- #


def test_fk_matches_independent_implementation(cell):
    tree = KinematicTree(cell)
    for q in random_states(cell, 12, seed=1):
        fk = tree.fk(q)
        ref = _fk64(cell, q.astype(np.float64))
        for lid, name in enumerate(tree.link_names):
            assert np.allclose(fk.link_T[lid], ref[name], atol=1e-5), name


def test_task_jacobian_matches_finite_differences(cell):
    """<= 1e-5 against central differences of the independent float64 FK."""
    tree = KinematicTree(cell)
    h = 1e-6
    for q in random_states(cell, 6, seed=2):
        q64 = q.astype(np.float64)
        J = tree.tcp_jacobians(tree.fk(q))
        for i in range(cell.n_tcps):
            J_fd = np.zeros((TASK_DIM, cell.n_joints))
            for j in range(cell.n_joints):
                qp, qm = q64.copy(), q64.copy()
                qp[j] += h
                qm[j] -= h
                pp, Rp = _tcp_pose64(cell, qp, i)
                pm, Rm = _tcp_pose64(cell, qm, i)
                J_fd[:POINT_DIM, j] = (pp - pm) / (2 * h)
                dR = Rp @ Rm.T
                J_fd[POINT_DIM:, j] = log_so3(dR.astype(np.float32)).astype(np.float64) / (2 * h)
            assert np.max(np.abs(J[i] - J_fd)) <= 1e-5, (
                f"tcp {cell.tcps[i].name}: max err {np.max(np.abs(J[i] - J_fd)):.2e}"
            )


def test_point_jacobian_matches_finite_differences(cell):
    tree = KinematicTree(cell)
    h = 1e-6
    q = random_states(cell, 1, seed=3)[0]
    fk = tree.fk(q)
    # A few links spread across the tree, including leaves.
    probe = list(range(0, tree.n_links, max(1, tree.n_links // 5)))
    for lid in probe:
        p = fk.link_T[lid][:POINT_DIM, POINT_DIM]
        J = tree.point_jacobian(fk, lid, p)
        name = tree.link_names[lid]
        J_fd = np.zeros((POINT_DIM, cell.n_joints))
        for j in range(cell.n_joints):
            qp, qm = q.astype(np.float64), q.astype(np.float64)
            qp[j] += h
            qm[j] -= h
            J_fd[:, j] = (_fk64(cell, qp)[name][:3, 3] - _fk64(cell, qm)[name][:3, 3]) / (2 * h)
        assert np.max(np.abs(J - J_fd)) <= 1e-5, name


def test_batched_point_jacobians_match_scalar(cell):
    tree = KinematicTree(cell)
    q = random_states(cell, 1, seed=4)[0]
    fk = tree.fk(q)
    links = np.arange(tree.n_links, dtype=np.int32)
    pts = fk.link_T[:, :POINT_DIM, POINT_DIM].copy()
    batched = tree.point_jacobians(fk, links, pts)
    for lid in range(tree.n_links):
        assert np.allclose(batched[lid], tree.point_jacobian(fk, lid, pts[lid]), atol=0.0)


def test_columns_off_the_path_are_exactly_zero(cell):
    """Non-zero columns for a TCP are precisely the joints on its own path."""
    tree = KinematicTree(cell)
    for q in random_states(cell, 4, seed=5):
        J = tree.tcp_jacobians(tree.fk(q))
        for i, tcp in enumerate(cell.tcps):
            urdf = cell.urdfs[tcp.model]
            model = next(m for m in cell.models if m.name == tcp.model)
            on_path = {
                jn for jn in urdf.chain_to(tcp.frame) if jn in model.joint_names
            }
            jidx = cell.joint_index()
            expect = np.zeros(cell.n_joints, dtype=bool)
            for jn in on_path:
                expect[jidx[(tcp.model, jn)]] = True
            nonzero = np.any(J[i] != 0.0, axis=0)
            assert np.array_equal(nonzero[~expect], np.zeros(int((~expect).sum()), dtype=bool)), (
                "a joint off the TCP's path has a non-zero Jacobian column"
            )


def test_shared_trunk_column_is_nonzero_for_both_tcps():
    """The point of the branched cell: one column serves two TCPs."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "branched_trunk.yaml")
    tree = KinematicTree(cell)
    trunk = cell.joint_labels().index("torso/trunk")
    J = tree.tcp_jacobians(tree.fk(cell.rest_posture()))
    assert cell.n_tcps == 2
    for i in range(2):
        assert np.any(J[i][:, trunk] != 0.0), "trunk must move both TCPs"


def test_manipulability_is_positive_away_from_singularities(cell):
    tree = KinematicTree(cell)
    w = tree.manipulability(tree.tcp_jacobians(tree.fk(cell.rest_posture())))
    assert np.all(w > 0.0)
    assert np.all(w > np.asarray([t.w_thresh for t in cell.tcps])), (
        "the shipped rest posture should not be inside the damping ramp"
    )


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 0.5, 3.0, np.pi - 1e-4])
def test_rotation_vector_round_trip(scale, rng):
    for _ in range(50):
        axis = rng.normal(size=POINT_DIM)
        axis = axis / np.linalg.norm(axis)
        w = (axis * scale).astype(np.float32)
        R = exp_so3(w)
        back = log_so3(R)
        assert np.linalg.norm(back - w) <= 1e-4 * max(1.0, scale), (w, back)
        assert np.allclose(R @ R.T, np.eye(POINT_DIM), atol=1e-5)
        assert abs(np.linalg.det(R.astype(np.float64)) - 1.0) < 1e-5


def test_rotation_log_near_pi():
    for axis in np.eye(POINT_DIM, dtype=np.float32):
        w = (axis * np.float32(np.pi - 1e-5)).astype(np.float32)
        R = exp_so3(w)
        back = log_so3(R)
        assert np.linalg.norm(np.abs(back) - np.abs(w)) < 1e-3
