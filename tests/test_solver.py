"""Layer 3b: dual PGS monotonicity, feasibility, and graceful degradation."""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import BIG
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.solver import solve_qp, unconstrained
from remoroo_lc.reference.walls import WallBuilder
from tests.conftest import random_states


def _problem(cell, seed, tighten=0.0):
    """A constraint set taken from a real state, optionally tightened."""
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    q = random_states(cell, 1, seed=seed)[0]
    walls = wb.assemble(q, tree.fk(q))
    g = np.random.default_rng(seed)
    qd_des = g.normal(0.0, 0.4, cell.n_joints).astype(np.float32)
    qd_post = g.normal(0.0, 0.1, cell.n_joints).astype(np.float32)
    h = walls.h.copy()
    if tighten:
        h[h < BIG] = np.float32(-tighten)
    return cell, walls.G, h, qd_des, qd_post, cell.joint_velocity_limits()


def _defaults(cell):
    sol = cell.limits["solver"]
    return (
        np.full(cell.n_joints, np.float32(sol["joint_weight"]), dtype=np.float32),
        float(cell.limits["posture"]["weight"]),
        float(sol["rho"]),
        int(sol["iterations"]),
    )


def test_dual_objective_is_monotone(cell):
    """Each coordinate step is an exact dual minimisation, so it cannot increase."""
    for seed in (31, 32, 33):
        c, G, h, qd_des, qd_post, qd_max = _problem(cell, seed, tighten=0.2)
        w_j, w_p, rho, iters = _defaults(c)
        res = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, iters, trace=True)
        hist = res.dual_history
        assert len(hist) == iters + 1
        for a, b in zip(hist, hist[1:]):
            assert b <= a + 1e-4, f"dual objective rose: {a} -> {b}"
        assert hist[-1] <= hist[0] + 1e-6


def test_feasible_problem_satisfies_its_constraints(cell):
    """On a satisfiable instance the only residual is the intended soft slack.

    The instance is built from joint-rate rows rather than from a random state:
    a uniformly random joint configuration frequently has spheres already
    interpenetrating, and those rows legitimately demand separation velocities
    large enough to hit the box clamp -- which is the graceful-degradation
    behaviour tested separately below, not a feasible problem.
    """
    n = cell.n_joints
    w_j, w_p, rho, iters = _defaults(cell)
    qd_max = cell.joint_velocity_limits()
    G = np.concatenate([np.eye(n, dtype=np.float32), -np.eye(n, dtype=np.float32)])
    for seed in (41, 42, 43):
        g = np.random.default_rng(seed)
        qd_des = g.normal(0.0, 0.4, n).astype(np.float32)
        qd_post = g.normal(0.0, 0.1, n).astype(np.float32)
        h = np.full(2 * n, np.float32(0.2))  # |qd_j| <= 0.2, always satisfiable
        res = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, iters)
        viol = G @ res.qd - h
        assert np.max(viol) <= max(1e-3, 10.0 / rho), np.max(viol)
        assert np.all(np.abs(res.qd) <= qd_max + 1e-6)
        assert res.box_clamped == 0


def test_inactive_problem_reproduces_the_unconstrained_optimum(cell):
    """No active rows must give exactly the closed-form blend of the two terms."""
    c, G, h, qd_des, qd_post, qd_max = _problem(cell, 51)
    h = np.full_like(h, BIG)
    w_j, w_p, rho, iters = _defaults(c)
    res = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, iters)
    want = np.clip(unconstrained(qd_des, qd_post, w_j, w_p), -qd_max, qd_max)
    assert np.max(np.abs(res.qd - want)) == 0.0
    assert res.n_active == 0


def test_conflicting_constraints_degrade_gracefully(cell):
    """Mutually impossible rows must still yield a bounded, finite answer."""
    c, G, _h, qd_des, qd_post, qd_max = _problem(cell, 61)
    n = c.n_joints
    # Demand every joint simultaneously move up by >= 1 and down by >= 1.
    G_bad = np.concatenate([np.eye(n, dtype=np.float32), -np.eye(n, dtype=np.float32)])
    h_bad = np.full(2 * n, np.float32(-1.0))
    w_j, w_p, rho, iters = _defaults(c)
    res = solve_qp(qd_des, qd_post, G_bad, h_bad, qd_max, w_j, w_p, rho, iters)
    assert np.all(np.isfinite(res.qd))
    assert np.all(np.abs(res.qd) <= qd_max + 1e-6)
    assert res.slack_norm > 0.0, "an infeasible problem must show up in the slack"
    # The compromise should be near the midpoint, not blown out to one side.
    assert np.max(np.abs(res.qd)) < 1.0 + 1e-3


def test_box_clamp_is_respected(cell):
    c, G, h, qd_des, qd_post, _ = _problem(cell, 71)
    tiny = np.full(c.n_joints, np.float32(0.05))
    w_j, w_p, rho, iters = _defaults(c)
    res = solve_qp(qd_des * 20.0, qd_post, G, h, tiny, w_j, w_p, rho, iters)
    assert np.all(np.abs(res.qd) <= tiny + 1e-7)
    assert res.box_clamped > 0


def test_repeatability_is_bitwise(cell):
    c, G, h, qd_des, qd_post, qd_max = _problem(cell, 81, tighten=0.1)
    w_j, w_p, rho, iters = _defaults(c)
    a = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, iters)
    b = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, iters)
    assert np.array_equal(a.qd, b.qd)
    assert np.array_equal(a.lam, b.lam)


def test_skipping_parked_rows_changes_nothing(cell):
    """The BIG-row skip must be an optimisation, not an approximation."""
    c, G, h, qd_des, qd_post, qd_max = _problem(cell, 91, tighten=0.05)
    w_j, w_p, rho, iters = _defaults(c)
    fast = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, iters)
    # Force every row to be swept by nudging BIG down to a still-inactive value.
    h_dense = h.copy()
    h_dense[h_dense >= BIG] = np.float32(1.0e5)
    dense = solve_qp(qd_des, qd_post, G, h_dense, qd_max, w_j, w_p, rho, iters)
    assert np.max(np.abs(fast.qd - dense.qd)) == 0.0


def test_row_order_matters_and_ours_is_fixed(cell):
    """Gauss-Seidel is order dependent; that is why the order is a cell property.

    Shuffling the rows changes the answer, which is exactly why the shipped order
    is derived from config load order and never from state.
    """
    n = cell.n_joints
    w_j, w_p, rho, _ = _defaults(cell)
    qd_max = cell.joint_velocity_limits()
    g = np.random.default_rng(101)
    # Dense, coupled, mutually tight rows: rows on disjoint joints would decouple
    # through the diagonal H and order genuinely would not matter for them.
    m = max(8, n)
    G = g.normal(size=(m, n)).astype(np.float32)
    h = np.full(m, np.float32(-0.5))
    qd_des = g.normal(0.0, 0.4, n).astype(np.float32)
    qd_post = g.normal(0.0, 0.1, n).astype(np.float32)
    # One sweep: the sequential dependency between rows is at its most visible.
    base = solve_qp(qd_des, qd_post, G, h, qd_max, w_j, w_p, rho, 1)
    perm = np.random.default_rng(5).permutation(G.shape[0])
    shuffled = solve_qp(qd_des, qd_post, G[perm], h[perm], qd_max, w_j, w_p, rho, 1)
    assert not np.array_equal(base.qd, shuffled.qd), (
        "if row order did not matter, warm starting and determinism would be moot"
    )
    # And the shipped order is stable across repeated builds of the same cell.
    labels_a = WallBuilder(cell, KinematicTree(cell)).row_labels()
    labels_b = WallBuilder(cell, KinematicTree(cell)).row_labels()
    assert labels_a == labels_b
