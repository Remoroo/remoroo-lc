"""The safety layers must say how much motion they removed.

remoroo-lc slows a command down in two places, and both do it by scaling the WHOLE
joint-velocity vector uniformly:

  * layer 2, `k_diffik`'s feasibility governor -- the damped inverse can ask for far
    more joint speed than the URDF allows near a singular direction, so the request is
    scaled to fit before it reaches the constraint solver;
  * layer 3, `k_solve`'s output velocity box -- the same idea one layer later, after
    the collision and joint-limit rows have had their say.

Because both are UNIFORM, neither shows up in any signal that existed before:
`box_clamped` counts joints over their individual limit and reads 0 while the entire
command is being halved, and `n_active` counts constraint rows, which the governor
does not touch at all.

Measured on this campaign: the layer-2 governor was active on 97-99% of full-scale
ticks, uniformly cutting motion to about 0.45, and nothing anywhere recorded it.  The
controller simply looked like it was faithfully obeying a slow command.  A governor
that is on almost always is not a safety net, it is a secret speed limit, and the only
thing that distinguishes those two is a number in a log.

These tests assert the signal exists, is per-environment, is bounded, and -- the part
that matters -- actually MOVES when the governor engages.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.kernels import available

pytestmark = pytest.mark.skipif(not available(), reason="no kernel backend installed")


def _backend(cell, num_envs=1):
    from remoroo_lc.kernels.warp_backend import BatchedController

    return BatchedController(cell, num_envs=num_envs, device="cpu")


def _random_q(cell, n, seed):
    lo, hi = cell.joint_limits()
    g = np.random.default_rng(seed)
    return (lo + 0.1 * (hi - lo) + 0.8 * (hi - lo) * g.random((n, cell.n_joints))).astype(
        np.float32
    )


def _far_chunk(cell, n_envs, k=2, reach_m=5.0):
    """A chunk demanding metres of translation -- far past any cell in the matrix."""
    act = np.zeros((n_envs, k, cell.action_dim), np.float32)
    for sl_pose, _ in cell.action_slices():
        act[:, :, sl_pose.start : sl_pose.start + 3] = reach_m
    return act


def _drive(cell, n_envs, seed, ticks, reach_m=5.0):
    """Run a far-away command for `ticks` control ticks, collecting diagnostics.

    Multiple ticks, not one, because layer 1 shapes the trajectory under jerk and
    acceleration limits: on tick one the reference has barely left rest and asks for
    nothing the arm cannot do, so a single-step test would report the governor idle no
    matter how broken it was.  The governor engages once the trajectory reaches speed --
    which is precisely the regime the 97-99% measurement came from.
    """
    ker = _backend(cell, n_envs)
    Q = _random_q(cell, n_envs, seed=seed)
    ker.reset(Q)
    ker.set_chunk(_far_chunk(cell, n_envs, reach_m=reach_m), Q)

    q, outs = Q.copy(), []
    for _ in range(ticks):
        out = ker.step(q)
        outs.append({k: np.asarray(v).copy() for k, v in out.items()})
        q = out["q_target"].astype(np.float32)
    return outs


def test_diagnostics_expose_both_scales(cell):
    ker = _backend(cell, 8)
    Q = _random_q(cell, 8, seed=201)
    ker.reset(Q)
    ker.set_chunk(np.zeros((8, 2, cell.action_dim), np.float32), Q)
    out = ker.step(Q)

    for key in ("governor", "box_scale"):
        assert key in out, f"{key} is not exported; the slowdown stays invisible"
        arr = np.asarray(out[key])
        assert arr.shape == (8,), f"{key} must be per environment, got {arr.shape}"
        assert np.all(np.isfinite(arr))
        # A scale, not a flag: (0, 1].  1.0 means the layer passed the request through.
        assert np.all(arr > 0.0), f"{key} must be positive"
        assert np.all(arr <= 1.0 + 1e-6), f"{key} is a scale and cannot exceed 1"


def test_scales_read_unimpeded_before_the_first_step(cell):
    """A read before any step must say "removed nothing", not "fully governed".

    Seeded at 1.0 rather than the zeros every other buffer gets, because a zeroed
    scale reads as a total stop and would make the very first logged tick a lie.
    """
    ker = _backend(cell, 4)
    ker.reset(_random_q(cell, 4, seed=202))
    out = ker.read()
    assert np.allclose(out["governor"], 1.0)
    assert np.allclose(out["box_scale"], 1.0)


def test_a_still_command_does_not_engage_the_governor(cell):
    """Holding position asks for no joint speed, so nothing should be cut.

    This is the control half of the experiment: without it, a governor stuck at some
    constant below 1 would pass the "it moves" test below by accident.
    """
    ker = _backend(cell, 8)
    Q = _random_q(cell, 8, seed=203)
    ker.reset(Q)
    ker.set_chunk(np.zeros((8, 2, cell.action_dim), np.float32), Q)
    out = ker.step(Q)
    assert np.allclose(out["governor"], 1.0, atol=1e-5), (
        "the governor engaged on a zero command -- it is not measuring feasibility"
    )


def test_an_infeasible_command_engages_the_governor(cell):
    """THE POINT.  Drive a command the arm cannot deliver and watch the scale drop.

    This reproduces the audited condition on every cell in the matrix: with a far-away
    target the governor engages on the large majority of ticks and cuts motion to well
    under half.  Before this export existed, all of that was invisible.
    """
    outs = _drive(cell, n_envs=4, seed=204, ticks=60)
    gov = np.stack([o["governor"] for o in outs])          # (ticks, envs)

    assert np.all(gov > 0.0) and np.all(gov <= 1.0 + 1e-6)
    assert np.min(gov) < 1.0, (
        "a command far beyond the arm's joint velocity limits never engaged the "
        "governor -- either the export is dead or the governor is"
    )

    # Not "it fired once": the measured pathology is that it fires almost ALWAYS, which
    # makes it a speed limit rather than a safety net.  Asserting the shape of the
    # pathology is what stops a future change from quietly restoring it.
    engaged = float(np.mean(gov < 1.0 - 1e-6))
    assert engaged > 0.5, f"governor engaged on only {engaged:.0%} of ticks"
    assert np.min(gov) < 0.9, (
        f"governor never cut motion by more than 10% (min {np.min(gov):.3g}); the "
        f"exported number is not tracking the scale that was applied"
    )


def test_the_governor_scale_is_consistent_with_the_request(cell):
    """The exported number must BE the scale, not merely correlate with it.

    After the governor runs, `qd_des` is the scaled request, so no joint may exceed its
    own velocity limit; and whenever the scale is strictly below 1 the request must be
    riding that limit exactly -- that is what "scaled until it just fits" means.
    """
    outs = _drive(cell, n_envs=4, seed=205, ticks=40)
    qd_max = np.asarray(cell.joint_velocity_limits())

    seen_engaged = False
    for out in outs:
        gov = out["governor"]
        worst = np.max(np.abs(out["qd_des"]) / np.maximum(qd_max, 1e-9), axis=1)
        assert np.all(worst <= 1.0 + 1e-3), "governed request still exceeds a joint limit"

        engaged = gov < 1.0 - 1e-6
        if np.any(engaged):
            seen_engaged = True
            assert np.allclose(worst[engaged], 1.0, atol=1e-3), (
                "the governor engaged but the request is not riding the limit, so the "
                "exported scale is not the scale that was applied"
            )
    assert seen_engaged, "the governor never engaged; this test proved nothing"


def test_reference_and_kernel_agree_that_the_governor_engaged(cell):
    """The NumPy reference already returned `governor`; the kernel now does too.

    They are separate implementations, so this checks the two agree about WHETHER the
    governor fired, which is the property anyone reading a log actually depends on.
    """
    from remoroo_lc.reference.controller import Controller

    n_envs, ticks = 2, 30
    ker = _backend(cell, n_envs)
    Q = _random_q(cell, n_envs, seed=206)
    ker.reset(Q)
    act = _far_chunk(cell, n_envs)
    ker.set_chunk(act, Q)

    refs = []
    for i in range(n_envs):
        ref = Controller(cell)
        ref.reset(Q[i])
        ref.set_chunk(act[i], Q[i])
        refs.append(ref)

    q_k = Q.copy()
    q_r = [Q[i].copy() for i in range(n_envs)]
    agreed = 0
    for _ in range(ticks):
        out = ker.step(q_k)
        q_k = out["q_target"].astype(np.float32)
        for i in range(n_envs):
            ref_out = refs[i].step(q_r[i])
            ref_gov = float(ref_out.diag["governor"])
            q_r[i] = ref_out.q_target.astype(np.float32)

            assert 0.0 < ref_gov <= 1.0 + 1e-6
            # Agreement on the BOOLEAN -- did it fire -- which is the property a log
            # reader depends on.  The scales themselves sit downstream of a damped
            # inverse whose conditioning makes bitwise agreement impossible in float32
            # (see test_kernel_equivalence for that argument in full), and the two
            # paths' joint states drift apart over a rollout for the same reason.
            agreed += int((ref_gov < 1.0 - 1e-6) == (out["governor"][i] < 1.0 - 1e-6))

    total = ticks * n_envs
    assert agreed > 0.9 * total, (
        f"reference and kernel disagreed about whether the governor engaged on "
        f"{total - agreed}/{total} ticks"
    )
