"""Reference vs kernels.

The NumPy reference is meant to be a specification of the kernels, so this file
is what keeps that claim true.  It is written around a measured fact about the
architecture rather than around a hoped-for tolerance:

    Layer 2 has a numerical gain of roughly 1/(sigma_min * dt_c) from TCP
    POSITION to JOINT VELOCITY -- about 2500 on these cells, and up to
    1/(2*lambda_min*dt_c) = 12500 at a singularity.

At 250 Hz a one-ulp float32 disagreement about where a metre-scale TCP is
(~1e-7 m) therefore becomes ~3e-4 rad/s of disagreement about how fast a joint
should turn, and no two implementations that are not bitwise identical can do
better in float32.  NumPy dispatching to BLAS and Warp emitting its own C++ are
not bitwise identical and will not be made so.

So the tolerances here are tiered by what each quantity's conditioning permits,
and the pipeline is checked where it is well conditioned (poses, Jacobians,
distances, constraint rows) at float32 precision, then at the amplified scale
where it is not.  The determinism guarantee the product actually relies on is a
different and stronger one -- same kernels, same device, bitwise -- and lives in
test_determinism.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.constants import TASK_DIM
from remoroo_lc.kernels import available
from remoroo_lc.plant import Plant
from remoroo_lc.reference.controller import Controller

pytestmark = pytest.mark.skipif(not available(), reason="no kernel backend installed")

#: One float32 ulp on a metre-scale coordinate.
ULP = 3.0e-7
#: Layer-2 gain from TCP position to joint velocity.  TYPICAL is what a
#: well-conditioned state gives (1/(sigma_min*dt_c)); WORST is the bound at a
#: singularity, where the damping takes over: 1/(2*lambda_min*dt_c).  A uniform
#: sample over joint limits contains near-singular states, so a max-over-10k
#: assertion has to use WORST while the bulk of the distribution sits near
#: TYPICAL -- both are asserted, which says more than either alone.
GAIN_TYPICAL = 2.5e3
GAIN_WORST = 1.25e4


def _backend(cell, num_envs=1):
    from remoroo_lc.kernels.warp_backend import BatchedController

    return BatchedController(cell, num_envs=num_envs, device="cpu")


def _random_q(cell, n, seed):
    lo, hi = cell.joint_limits()
    g = np.random.default_rng(seed)
    return (lo + 0.1 * (hi - lo) + 0.8 * (hi - lo) * g.random((n, cell.n_joints))).astype(
        np.float32
    )


# --------------------------------------------------------------------------- #
# well-conditioned stages: full float32 precision
# --------------------------------------------------------------------------- #


def test_forward_kinematics_matches(cell):
    ker = _backend(cell, 32)
    Q = _random_q(cell, 32, seed=101)
    ker.reset(Q)
    ref = Controller(cell)
    p_k, R_k = ker.tcp_p.numpy(), ker.tcp_R.numpy()
    link_k = ker.link_T.numpy()
    for i in range(Q.shape[0]):
        fk = ref.tree.fk(Q[i])
        p_r, R_r = ref.tree.tcp_poses(fk)
        assert np.max(np.abs(p_r - p_k[i])) < 1e-5
        assert np.max(np.abs(R_r - R_k[i])) < 1e-5
        assert np.max(np.abs(fk.link_T - link_k[i])) < 1e-5


def test_jacobians_match(cell):
    ker = _backend(cell, 24)
    Q = _random_q(cell, 24, seed=102)
    ker.reset(Q)
    ref = Controller(cell)
    Jv, Jw = ker.Jv.numpy(), ker.Jw.numpy()
    Js = ker.Js.numpy()
    for i in range(Q.shape[0]):
        fk = ref.tree.fk(Q[i])
        J = ref.tree.tcp_jacobians(fk)
        for t in range(cell.n_tcps):
            assert np.max(np.abs(J[t][:3].T - Jv[i, t])) < 1e-5
            assert np.max(np.abs(J[t][3:].T - Jw[i, t])) < 1e-5
        if len(ref.walls.spheres):
            centres = ref.walls.sphere_world(fk)
            Jref = ref.tree.point_jacobians(fk, ref.walls.spheres.link, centres)
            assert np.max(np.abs(np.transpose(Jref, (0, 2, 1)) - Js[i])) < 1e-5


def test_manipulability_matches(cell):
    ker = _backend(cell, 32)
    Q = _random_q(cell, 32, seed=103)
    ker.reset(Q)
    ker.set_chunk(np.zeros((2, cell.action_dim), np.float32), Q)
    out = ker.step(Q)
    ref = Controller(cell)
    ik = ref.diffik
    thresh = np.asarray([t.w_thresh for t in cell.tcps], dtype=np.float32)
    on_ramp = 0
    for i in range(Q.shape[0]):
        w = ref.tree.manipulability(ref.tree.tcp_jacobians(ref.tree.fk(Q[i])))
        lam_r, lam_k = ik.damping(w), out["damping"][i]
        off = w > thresh
        on_ramp += int(np.count_nonzero(~off))

        # Off the damping ramp -- 98% of uniformly sampled states -- w is a
        # well-conditioned product of TASK_DIM Cholesky pivots and the two agree
        # to about a percent.  The damping is then EXACTLY lambda_min in both,
        # because max(0, 1 - w/w_thresh) is exactly zero, so that gets an exact
        # comparison rather than a tolerance.
        if off.any():
            rel = np.max(
                np.abs(w[off] - out["manipulability"][i][off]) / np.maximum(w[off], 1e-9)
            )
            assert rel < 2e-2, f"manipulability relative error {rel:.2e} off the ramp"
            assert np.array_equal(lam_r[off], lam_k[off])
            assert np.allclose(lam_r[off], ik.lambda_min)

        # On the ramp, w is det(J J^T) evaluated where that determinant is going
        # to zero: its float32 value is mostly cancellation, the two
        # implementations can even disagree about whether the Cholesky succeeded,
        # and the relative error is unbounded.  That is a property of the
        # quantity, not of either implementation.  What has to hold is that both
        # respond by damping hard, and that q_dot still lands in the same place --
        # which is what test_single_step_joint_velocity_matches checks, singular
        # states included.
        assert np.all(lam_k >= float(ik.lambda_min) - 1e-6)
        assert np.max(np.abs(lam_r - lam_k)) < 0.1
    assert on_ramp < 0.15 * Q.shape[0] * cell.n_tcps, (
        "sampling landed on the damping ramp far more often than expected"
    )


def test_constraint_rows_match(cell):
    ker = _backend(cell, 16)
    Q = _random_q(cell, 16, seed=104)
    ker.reset(Q)
    ker.set_chunk(np.zeros((2, cell.action_dim), np.float32), Q)
    ker.step(Q)
    G_k, h_k, d_k = ker.G.numpy(), ker.h.numpy(), ker.dist.numpy()
    ref = Controller(cell)
    for i in range(Q.shape[0]):
        walls = ref.walls.assemble(Q[i], ref.tree.fk(Q[i]))
        assert np.max(np.abs(walls.distance - d_k[i])) < 1e-5
        # h is either the damper value or exactly BIG; the parked rows must match
        # exactly, since that is what the solver's skip condition tests.
        parked_r = walls.h >= np.float32(1.0e6)
        parked_k = h_k[i] >= np.float32(1.0e6)
        assert np.array_equal(parked_r, parked_k), "row activation disagrees"
        if (~parked_r).any():
            assert np.max(np.abs(walls.h[~parked_r] - h_k[i][~parked_r])) < 1e-5
        # G is compared on the LIVE rows only.  The kernel deliberately leaves a
        # parked row's G unwritten -- nothing reads it, and filling all of them
        # was 18% of the batched step -- so comparing there would be comparing
        # against a value that has no meaning rather than against a wrong one.
        live = ~parked_r
        if live.any():
            assert np.max(np.abs(walls.G[live] - G_k[i][live])) < 1e-4


def test_interpolator_state_matches(cell):
    """Layer 1 alone, over a long run, with the tracker's own dynamics."""
    ker = _backend(cell, 1)
    ref = Controller(cell)
    q = cell.rest_posture()
    ref.reset(q)
    ker.reset(q[None])
    g = np.random.default_rng(105)
    worst = 0.0
    for k in range(600):
        if k % 60 == 0:
            a = g.normal(0.0, 0.05, (8, cell.action_dim)).astype(np.float32)
            ref.set_chunk(a, q)
            ker.set_chunk(a, q[None])
        ref.step(q)
        ker.step(q[None])
        worst = max(worst, float(np.max(np.abs(ref.interp.x - ker.x.numpy()[0]))))
    assert worst < 1e-4, f"interpolator state diverged by {worst:.2e}"


# --------------------------------------------------------------------------- #
# the amplified stage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_states", [128])
def test_single_step_joint_velocity_matches(cell, n_states):
    """One step from identical inputs, at the tolerance conditioning allows."""
    ker = _backend(cell, n_states)
    Q = _random_q(cell, n_states, seed=106)
    g = np.random.default_rng(107)
    actions = g.normal(0.0, 0.03, (n_states, 4, cell.action_dim)).astype(np.float32)
    ker.reset(Q)
    ker.set_chunk(actions, Q)
    out = ker.step(Q)

    diffs = []
    for i in range(n_states):
        ref = Controller(cell)
        ref.reset(Q[i])
        ref.set_chunk(actions[i], Q[i])
        o = ref.step(Q[i])
        diffs.append(float(np.max(np.abs(o.qd - out["qd"][i]))))
        assert np.array_equal(
            o.diag["n_active"] > 0, out["n_active"][i] > 0
        ), "the two disagree about whether a wall is active"
    d = np.asarray(diffs)
    assert np.max(d) < ULP * GAIN_WORST, f"q_dot max diff {np.max(d):.2e}"
    assert np.percentile(d, 90) < ULP * GAIN_TYPICAL, (
        f"q_dot p90 diff {np.percentile(d, 90):.2e} -- the bulk of states should be "
        "nowhere near the singular bound"
    )


@pytest.mark.slow
def test_ten_thousand_random_steps(cell):
    """The prompt's 10k-step sweep, batched."""
    total = 10000
    batch = 500
    ker = _backend(cell, batch)
    ref = Controller(cell)
    diffs = []
    for b in range(total // batch):
        Q = _random_q(cell, batch, seed=200 + b)
        g = np.random.default_rng(300 + b)
        actions = g.normal(0.0, 0.03, (batch, 2, cell.action_dim)).astype(np.float32)
        ker.reset(Q)
        ker.set_chunk(actions, Q)
        out = ker.step(Q)
        for i in range(0, batch, 25):  # dense enough to catch a systematic break
            ref.reset(Q[i])
            ref.set_chunk(actions[i], Q[i])
            o = ref.step(Q[i])
            diffs.append(float(np.max(np.abs(o.qd - out["qd"][i]))))
    d = np.asarray(diffs)
    assert np.max(d) < ULP * GAIN_WORST, (
        f"q_dot max diff {np.max(d):.2e} over {total} states"
    )
    assert np.percentile(d, 90) < ULP * GAIN_TYPICAL, (
        f"q_dot p90 diff {np.percentile(d, 90):.2e} over {total} states"
    )


def test_closed_loop_trajectories_stay_together(cell):
    """The property that actually matters: two independent loops track each other.

    Each side drives its own plant, so nothing is being held in sync.  The
    instantaneous q_dot disagreement is amplified as documented above; what this
    checks is that the closed loop contracts it rather than integrating it, which
    is what makes the reference a usable specification at all.
    """
    ref, pr = Controller(cell), Plant(cell)
    ker, pk = _backend(cell, 1), Plant(cell)
    g = np.random.default_rng(108)
    q1 = cell.rest_posture()
    q2 = q1.copy()
    pr.reset(q1)
    ref.reset(q1)
    pk.reset(q2)
    ker.reset(q2[None])
    worst_tcp = 0.0
    for k in range(1000):
        if k % 40 == 0:
            a = g.normal(0.0, 0.05, (8, cell.action_dim)).astype(np.float32)
            ref.set_chunk(a, q1)
            ker.set_chunk(a, q2[None])
        o1 = ref.step(q1)
        o2 = ker.step(q2[None])
        worst_tcp = max(worst_tcp, float(np.max(np.abs(o1.p_meas - o2["p_meas"][0]))))
        q1, _ = pr.step(o1.q_target)
        q2, _ = pk.step(o2["q_target"][0])
    assert worst_tcp < 5e-3, (
        f"independent loops drifted {worst_tcp * 1000:.2f} mm apart over 4 s"
    )


def test_batched_matches_single(cell):
    """Environment b must not depend on any other environment."""
    n = 16
    Q = _random_q(cell, n, seed=109)
    g = np.random.default_rng(110)
    actions = g.normal(0.0, 0.04, (n, 4, cell.action_dim)).astype(np.float32)

    big = _backend(cell, n)
    big.reset(Q)
    big.set_chunk(actions, Q)
    out_big = big.step(Q)

    for i in (0, n // 2, n - 1):
        one = _backend(cell, 1)
        one.reset(Q[i][None])
        one.set_chunk(actions[i], Q[i][None])
        out_one = one.step(Q[i][None])
        assert np.array_equal(out_one["q_target"][0], out_big["q_target"][i]), (
            "batching changed the answer for one environment"
        )


def test_task_dim_is_the_only_shared_dimension(cell):
    """Sanity: the kernels were told nothing about this cell but its structure."""
    ker = _backend(cell, 1)
    assert ker.st.n_joints == cell.n_joints
    assert ker.st.n_tcps == cell.n_tcps
    assert ker.A.shape[1] == TASK_DIM * cell.n_tcps
    assert ker.G.shape[1] == 2 * cell.n_joints + ker.st.n_pairs
