"""Closed-loop properties of the composed controller.

These are the claims the safety argument rests on, checked against the plant
rather than against the controller's own idea of what happened.
"""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import POINT_DIM
from remoroo_lc.reference.solver import unconstrained
from remoroo_lc.schema import load_cell
from tests.conftest import CELL_DIR


def _drive(cell, controller, plant, action_fn, ticks, chunk_len=8):
    """Run a closed loop, refreshing the chunk every chunk_len policy steps."""
    q = cell.rest_posture()
    plant.reset(q)
    controller.reset(q)
    outs = []
    ticks_per_chunk = int(
        chunk_len * float(cell.limits["rates"]["command_hz"])
        / float(cell.limits["rates"]["policy_hz"])
    )
    t = 0
    while t < ticks:
        controller.set_chunk(action_fn(t), q)
        for _ in range(min(ticks_per_chunk, ticks - t)):
            out = controller.step(q)
            q, _ = plant.step(out.q_target)
            outs.append(out)
            t += 1
    return outs, q


def test_target_never_leaves_joint_limits(cell, controller, plant, rng):
    """q_target must stay inside limits, and so must the plant that follows it."""
    lo, hi = cell.joint_limits()

    def actions(_t):
        # Deliberately aggressive: large deltas, random directions.
        a = rng.normal(0.0, 0.12, size=(8, cell.action_dim)).astype(np.float32)
        return a

    outs, _ = _drive(cell, controller, plant, actions, ticks=900)
    for out in outs:
        assert np.all(out.q_target >= lo - 1e-4), "q_target below a joint limit"
        assert np.all(out.q_target <= hi + 1e-4), "q_target above a joint limit"
    q_final, _ = plant.state
    assert np.all(q_final >= lo - 1e-3) and np.all(q_final <= hi + 1e-3)


def test_no_hard_collision_when_driven_into_a_wall(cell, controller, plant):
    """Push every TCP straight down through the table; nothing may interpenetrate."""

    def actions(_t):
        a = np.zeros((8, cell.action_dim), dtype=np.float32)
        for pose, _eff in cell.action_slices():
            a[:, pose.start + 2] = -0.05  # 5 cm down per policy step, forever
        return a

    outs, _ = _drive(cell, controller, plant, actions, ticks=1500)
    worst = min(o.diag["min_pair_distance"] for o in outs)
    assert worst > 0.0, f"hard collision: min pair distance {worst * 1000:.2f} mm"


def test_approach_speed_vanishes_at_the_safety_distance(cell, controller, plant):
    """The damper's contract, stated at the two places it actually holds.

    COMMANDED rate: for every damper row, -n^T (J_a - J_b) q_dot <= h.  This is
    exact, and it is what Layer 3 is responsible for.

    ACHIEVED rate: bounded only up to actuation lag.  The plant is still
    executing a command issued two ticks ago while the damper is already slowing
    the next one, so the achieved closing rate transiently exceeds the
    instantaneous bound by tens of mm/s.  That gap is not a defect to be tuned
    away -- it is why d_safe is a margin rather than a target, and the property
    that survives it is non-penetration.

    Distances are read per pair; differencing `min_pair_distance` would measure
    the gap between two different pairs whenever the closest pair changes.
    """
    d_safe = float(controller.walls.cd_safe)
    n_j = controller.walls.n_joint_rows

    def actions(_t):
        a = np.zeros((8, cell.action_dim), dtype=np.float32)
        for pose, _eff in cell.action_slices():
            a[:, pose.start + 2] = -0.05
        return a

    outs, _ = _drive(cell, controller, plant, actions, ticks=1500)
    if len(controller.walls.pairs) == 0:
        return

    worst_cmd, checked, dmin = 0.0, 0, np.inf
    for out in outs:
        q = out.diag["q_at_step"]
        walls = controller.walls.assemble(q, controller.tree.fk(q))
        dmin = min(dmin, float(walls.distance[n_j:].min()))
        if out.diag["box_clamped"]:
            # The box clamp is applied AFTER the QP by design, so on these ticks
            # the delivered q_dot is deliberately not the constrained one.
            continue
        # The rows are soft: the contract is G q_dot <= h + s with s = lambda/rho,
        # not G q_dot <= h.  Asserting the hard form would fail on any row the
        # solver has to lean on hard -- the leg driven into the floor loads one to
        # lambda ~ 60, i.e. 6 mm/s of intended give.
        slack = out.diag["lam"] / np.float32(controller.rho)
        worst_cmd = max(worst_cmd, float(np.max(walls.G @ out.qd - walls.h - slack)))
        checked += 1

    assert checked > 0
    # The residual left by the FIXED 32 sweeps, quantified against an exact QP
    # solver in test_oracles.py.  Cells whose rows conflict -- the branched one,
    # where a single trunk joint serves two chains being driven into the same
    # floor -- leave the most of it.
    assert worst_cmd <= 1e-2, (
        f"commanded approach exceeded row + its own slack by {worst_cmd:.5f}, "
        "which is beyond the 32-sweep residual"
    )
    assert dmin > 0.0, "hard collision"
    assert dmin >= d_safe - 2e-3, f"pushed inside d_safe: {dmin:.4f} m"


def test_no_active_wall_means_the_unconstrained_optimum(cell, controller, plant):
    """When nothing binds, Layer 3 must be a no-op beyond the posture blend."""

    def actions(_t):
        return np.zeros((8, cell.action_dim), dtype=np.float32)

    outs, _ = _drive(cell, controller, plant, actions, ticks=400)
    checked = 0
    for out in outs:
        if out.diag["n_active"] != 0:
            continue
        want = unconstrained(
            out.diag["qd_des"],
            -np.float32(controller.k_post) * (out.diag["q_at_step"] - controller.q_rest),
            controller.w_joint,
            controller.w_post,
        )
        want = np.clip(want, -controller.qd_max, controller.qd_max)
        assert np.max(np.abs(out.qd - want)) == 0.0
        checked += 1
    assert checked > 100, "expected most ticks of a hold to have no active row"


def test_rest_posture_is_a_fixed_point(cell, controller, plant):
    """Zero task command at q_rest must stay at q_rest, on every cell."""
    q = cell.rest_posture()
    plant.reset(q)
    controller.reset(q)
    for k in range(600):
        if k % 125 == 0:
            controller.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
        out = controller.step(q)
        q, _ = plant.step(out.q_target)
    assert np.max(np.abs(q - cell.rest_posture())) < 1e-3, (
        "q_rest must be a fixed point of the zero command"
    )


def test_posture_term_alone_determines_the_null_space(cell, controller, plant):
    """The exact claim: q_dot's null-space component comes only from the posture.

    q_dot_des lies in range(J^T) by construction, so it contributes nothing to the
    null space; with no wall active the QP optimum is the fixed blend, and the
    null-space component of q_dot must therefore be exactly
    w_post/(w_j + w_post) times that of q_dot_post.  This is what "the null space
    resolves deterministically" means -- it is an algebraic identity, not a
    tendency, and it holds on redundant, exactly determined and over-constrained
    cells alike.
    """
    q = cell.rest_posture().copy()
    q[0] += np.float32(0.25)  # somewhere off the rest posture
    plant.reset(q)
    controller.reset(q)
    controller.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
    checked = 0
    for _ in range(200):
        out = controller.step(q)
        q, _ = plant.step(out.q_target)
        if out.diag["n_active"] or out.diag["box_clamped"]:
            continue
        J = controller.tree.tcp_jacobians(controller.tree.fk(out.diag["q_at_step"]))
        J = J.reshape(cell.task_dim, cell.n_joints).astype(np.float64)
        N = np.eye(cell.n_joints) - np.linalg.pinv(J) @ J
        qd_post = -np.float64(controller.k_post) * (
            out.diag["q_at_step"].astype(np.float64) - controller.q_rest.astype(np.float64)
        )
        scale = controller.w_post / (float(controller.w_joint[0]) + controller.w_post)
        assert np.max(np.abs(N @ out.qd.astype(np.float64) - scale * (N @ qd_post))) < 1e-5
        checked += 1
    assert checked > 100


def test_redundant_cell_returns_to_rest_within_its_null_space():
    """Perturbed along a self-motion direction, a redundant cell must come back."""
    from remoroo_lc.plant import Plant
    from remoroo_lc.reference.controller import Controller

    cell = load_cell(CELL_DIR / "dual_7dof.yaml")
    assert cell.n_joints > cell.task_dim, "this cell must actually be redundant"
    controller, plant = Controller(cell), Plant(cell)
    q_rest = cell.rest_posture()

    # A direction that moves the joints without moving either TCP, to first order.
    J = controller.tree.tcp_jacobians(controller.tree.fk(q_rest))
    J = J.reshape(cell.task_dim, cell.n_joints).astype(np.float64)
    null = np.linalg.svd(J)[2][cell.task_dim :]
    assert null.shape[0] == cell.n_joints - cell.task_dim
    q0 = (q_rest + np.float32(0.30) * null[0].astype(np.float32)).astype(np.float32)

    plant.reset(q0)
    controller.reset(q0)
    q = q0
    dev = [float(np.linalg.norm(q - q_rest))]
    tcp0 = None
    # The posture time constant with the shipped defaults (w_post = 1e-2,
    # k_post = 1) is about 100 s: the term exists to make the instantaneous
    # solution unique, not to regulate posture quickly.  Raise k_post if active
    # regulation is ever wanted; the shape of the claim does not change.
    for k in range(4000):
        if k % 250 == 0:
            controller.set_chunk(np.zeros((16, cell.action_dim), dtype=np.float32), q)
        out = controller.step(q)
        q, _ = plant.step(out.q_target)
        if tcp0 is None:
            tcp0 = out.p_meas.copy()
        if k % 500 == 499:
            dev.append(float(np.linalg.norm(q - q_rest)))
    assert all(b < a for a, b in zip(dev, dev[1:])), f"deviation not monotone: {dev}"
    # The measured decay over this 16 s window is ~4%, which is what the
    # ~100 s posture time constant and the ~0.21 plant velocity gain predict
    # together.  The claim under test is the direction and monotonicity of the
    # return, not its speed.
    assert dev[-1] < dev[0] * 0.985
    assert np.max(np.abs(out.p_meas - tcp0)) < 0.01, "the return disturbed the TCP"


def test_branched_conflict_is_bounded_and_deterministic():
    """Two TCPs fighting over a shared trunk: bounded, repeatable, regression-locked."""
    from remoroo_lc.plant import Plant
    from remoroo_lc.reference.controller import Controller

    cell = load_cell(CELL_DIR / "branched_trunk.yaml")
    trunk = cell.joint_labels().index("torso/trunk")

    def run():
        controller, plant = Controller(cell), Plant(cell)
        q = cell.rest_posture()
        plant.reset(q)
        controller.reset(q)
        # Opposite tangential demands: only the shared trunk can serve either.
        a = np.zeros((8, cell.action_dim), dtype=np.float32)
        slices = cell.action_slices()
        a[:, slices[0][0].start + 1] = 0.04
        a[:, slices[1][0].start + 1] = 0.04
        outs = []
        for k in range(400):
            if k % 125 == 0:
                controller.set_chunk(a, q)
            out = controller.step(q)
            q, _ = plant.step(out.q_target)
            outs.append(out)
        return outs, q

    outs_a, q_a = run()
    outs_b, q_b = run()
    assert np.array_equal(q_a, q_b), "the compromise must be deterministic"
    for x, y in zip(outs_a, outs_b):
        assert np.array_equal(x.qd, y.qd)

    trunk_rate = np.array([float(o.qd[trunk]) for o in outs_a])
    assert np.all(np.isfinite(trunk_rate))
    assert np.max(np.abs(trunk_rate)) <= float(cell.joint_velocity_limits()[trunk]) + 1e-6

    # Regression lock: this exact compromise is the behaviour under test.  If it
    # changes, either the arbitration changed or something upstream of it did,
    # and either way that is a decision, not a detail.
    np.testing.assert_allclose(
        q_a,
        np.array(
            [
                0.25909373, 1.4282929, 0.26799116, 1.1618012, -0.27512884,
                0.10483745, 0.9346017, 1.2062315, 0.45409423, -0.6052772,
                0.8705812,
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=2e-4,
    )


def test_effectorless_cell_runs_every_layer():
    """g = 0 must exercise the full stack, not a reduced path."""
    from remoroo_lc.plant import Plant
    from remoroo_lc.reference.controller import Controller

    cell = load_cell(CELL_DIR / "single_6dof_leg.yaml")
    assert cell.effector_widths == (0,)
    controller, plant = Controller(cell), Plant(cell)
    q = cell.rest_posture()
    plant.reset(q)
    controller.reset(q)
    a = np.zeros((8, cell.action_dim), dtype=np.float32)
    # +z in the MODEL BASE frame.  This limb is mounted inverted, so its base z
    # points at the floor; the action schema is base-frame relative, not world.
    a[:, 2] = 0.04
    controller.set_chunk(a, q)
    saw_wall = False
    for k in range(1200):
        if k % 125 == 0:
            controller.set_chunk(a, q)
        out = controller.step(q)
        q, _ = plant.step(out.q_target)
        assert out.effector.shape == (0,)
        assert out.diag["min_pair_distance"] > 0.0
        saw_wall |= out.diag["n_active"] > 0
    assert saw_wall, "the leg should have been stopped by the floor damper"


def test_output_integrates_from_measurement_not_from_target(cell, controller, plant):
    """q_target - q_meas must equal exactly q_dot * dt_c."""
    q = cell.rest_posture()
    plant.reset(q)
    controller.reset(q)
    controller.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
    for _ in range(50):
        out = controller.step(q)
        expect = (q + out.qd * controller.dt_c).astype(np.float32)
        assert np.array_equal(out.q_target, expect)
        q, _ = plant.step(out.q_target)
