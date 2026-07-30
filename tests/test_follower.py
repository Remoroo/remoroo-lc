"""The preview follower: layer 1's VLA mode.

The point-to-point law treats every waypoint as a stopping target and is the
right law for sparse terminal goals; on a densely sampled path it structurally
under-travels (measured ~2x amplitude loss on the rig's teach recording).  The
follower exists because a VLA hands over K future poses at once, so interior
waypoints are via points with KNOWN future.  These tests pin down the properties
that make that mode trustworthy: it passes through what it was given, it is C1
across chunk replacement, it stops on its own past the horizon, and it is the
same trajectory in NumPy and in the kernels, twice over.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.constants import DTYPE, POINT_DIM
from remoroo_lc.reference.interpolator import ChunkInterpolator
from remoroo_lc.schema import load_cell

TPA = 5  # command ticks per action at 250 Hz / 50 Hz


@pytest.fixture
def fcell(cell_path):
    """The cell with layer 1 switched to follower mode at the VLA action rate."""
    cell = load_cell(cell_path)
    cell.limits["task"]["mode"] = "follower"
    cell.limits["rates"]["policy_hz"] = 50
    return cell


def _interp(cell) -> tuple[ChunkInterpolator, np.ndarray, np.ndarray]:
    it = ChunkInterpolator(cell)
    p0 = np.zeros((cell.n_tcps, POINT_DIM), dtype=DTYPE)
    p0[:, 0] = 0.4
    R0 = np.stack([np.eye(POINT_DIM, dtype=DTYPE)] * max(cell.n_tcps, 1))[: cell.n_tcps]
    it.reset(p0, R0)
    return it, p0, R0


def _pose_chunk(cell, k_steps: int, dp) -> np.ndarray:
    """A chunk whose position deltas are dp[k] on every TCP's x axis."""
    acts = np.zeros((k_steps, cell.action_dim), dtype=DTYPE)
    for pose_sl, _eff in cell.action_slices():
        acts[:, pose_sl.start] = dp
    return acts


def _current_pose(it):
    p = it.x[:, :POINT_DIM].copy()
    R = np.stack([it.R_ref[i] @ _exp(it.x[i, POINT_DIM:]) for i in range(it.n_tcps)])
    return p, R


def _exp(r):
    from remoroo_lc.spatial import exp_so3

    return exp_so3(np.asarray(r, dtype=DTYPE))


def test_mode_comes_from_config_and_default_is_point_to_point(cell):
    assert ChunkInterpolator(cell).mode == "point_to_point"


def test_follower_passes_through_every_waypoint(fcell):
    """At each knot time the command IS the waypoint -- no chase, no lag."""
    it, p0, R0 = _interp(fcell)
    rng = np.random.default_rng(11)
    acts = (rng.standard_normal((8, fcell.action_dim)) * 0.01).astype(DTYPE)
    it.set_chunk(acts, p0, R0)
    hits = []
    for n in range(1, 8 * TPA + 1):
        it.step()
        if n % TPA == 0:
            k = n // TPA - 1
            hits.append(float(np.abs(it.x - it.waypoints[k]).max()))
    assert max(hits) < 1e-5, hits


def test_follower_reproduces_a_constant_velocity_line_exactly(fcell):
    """Central differences make the Hermite exact on a line.  The only error
    allowed is the blend-in from rest over segment 0, which is real physics."""
    it, p0, R0 = _interp(fcell)
    dp = 0.02  # 1.0 m/s at 50 Hz
    it.set_chunk(_pose_chunk(fcell, 8, dp), p0, R0)
    for _ in range(8 * TPA):
        it.step()
    # Second chunk: now moving at exactly the line's velocity.
    p1, R1 = _current_pose(it)
    it.set_chunk(_pose_chunk(fcell, 8, dp), p1, R1)
    x_start = float(it.seam_x[0, 0])
    errs = []
    for n in range(1, 8 * TPA + 1):
        it.step()
        want = x_start + dp * n / TPA
        errs.append(abs(float(it.x[0, 0]) - want))
    assert max(errs) < 1e-5, max(errs)


def test_follower_is_c1_across_chunk_replacement(fcell):
    """Replacing the chunk mid-flight must not step the velocity."""
    it, p0, R0 = _interp(fcell)
    rng = np.random.default_rng(7)
    acts = (rng.standard_normal((16, fcell.action_dim)) * 0.008).astype(DTYPE)
    it.set_chunk(acts, p0, R0)
    for _ in range(8 * TPA):  # execute half the horizon
        it.step()
    v_before = it.v.copy()
    a_scale = float(np.abs(it.a).max())
    p1, R1 = _current_pose(it)
    acts2 = (rng.standard_normal((16, fcell.action_dim)) * 0.008).astype(DTYPE)
    it.set_chunk(acts2, p1, R1)
    it.step()
    jump = float(np.abs(it.v - v_before).max())
    # One tick of the chunk's own acceleration is the honest bound on how much
    # the velocity may change across a C1 seam.
    dt_c = 1.0 / float(fcell.limits["rates"]["command_hz"])
    assert jump < 10.0 * max(a_scale, 1.0) * dt_c + 1e-4, jump


def test_follower_runway_is_a_constant_deceleration_stop(fcell):
    """Past the horizon: rest, at distance v_K * T / 2, decelerating linearly."""
    it, p0, R0 = _interp(fcell)
    k_steps = 8
    dp = 0.02
    it.set_chunk(_pose_chunk(fcell, k_steps, dp), p0, R0)
    for _ in range(k_steps * TPA):
        it.step()
    v_k = float(it.v[0, 0])
    x_k = float(it.x[0, 0])
    horizon_s = k_steps / 50.0
    vs = []
    for _ in range(2 * k_steps * TPA):
        it.step()
        vs.append(float(it.v[0, 0]))
    assert abs(float(it.v[0, 0])) < 1e-6
    assert abs(float(it.a[0, 0])) < 1e-6
    travel = float(it.x[0, 0]) - x_k
    assert travel == pytest.approx(v_k * horizon_s / 2.0, rel=1e-3)
    dec = np.diff(np.asarray(vs[: k_steps * TPA]))
    assert dec.max() < 1e-6  # never speeds back up
    assert dec.std() < 1e-4  # constant deceleration, not a shaped one


def test_follower_beats_point_to_point_on_a_sampled_sine(fcell):
    """The regression that motivated the mode: a 0.4 Hz path at 50 Hz sampling.
    Point-to-point structurally under-travels; the follower must not."""
    results = {}
    for mode in ("follower", "point_to_point"):
        fcell.limits["task"]["mode"] = mode
        it, p0, R0 = _interp(fcell)
        t = np.arange(0, 41) / 50.0
        path = 0.15 * np.sin(2 * np.pi * 0.4 * t)
        deltas = np.diff(path)
        errs = []
        for s in range(0, 40, 8):
            p_now, R_now = _current_pose(it)
            it.set_chunk(_pose_chunk(fcell, 8, deltas[s : s + 8]), p_now, R_now)
            for n in range(8 * TPA):
                it.step()
                want = 0.4 + np.interp((s * TPA + n + 1) / 250.0, t, path)
                errs.append(abs(float(it.x[0, 0]) - want))
        results[mode] = float(np.sqrt(np.mean(np.square(errs))))
    assert results["follower"] < 2e-3, results
    assert results["follower"] < 0.2 * results["point_to_point"], results


def test_follower_reference_matches_kernel_and_is_bitwise_repeatable(fcell):
    """Same trajectory in NumPy and Warp CPU; and the kernel agrees with itself
    to the bit across two runs."""
    kernels = pytest.importorskip("remoroo_lc.kernels.warp_backend")
    from remoroo_lc.reference.controller import Controller

    ref = Controller(fcell)
    q0 = fcell.rest_posture()

    def kernel_run():
        ker = kernels.BatchedController(fcell, num_envs=1, device="cpu")
        ker.reset(q0[None])
        q = q0.copy()
        rng = np.random.default_rng(5)
        out_p = []
        for _ in range(2):
            acts = (rng.standard_normal((8, fcell.action_dim)) * 0.006).astype(DTYPE)
            ker.set_chunk(acts, q[None])
            for _ in range(8 * TPA):
                o = ker.step(q[None])
                q = o["q_target"][0].astype(np.float32)
                out_p.append(o["p_cmd"][0].copy())
        return np.asarray(out_p)

    k1 = kernel_run()
    k2 = kernel_run()
    assert np.array_equal(k1, k2), "kernel follower is not repeatable"

    ref.reset(q0)
    q = q0.copy()
    rng = np.random.default_rng(5)
    ref_p = []
    for _ in range(2):
        acts = (rng.standard_normal((8, fcell.action_dim)) * 0.006).astype(DTYPE)
        ref.set_chunk(acts, q)
        for _ in range(8 * TPA):
            o = ref.step(q)
            q = o.q_target.astype(np.float32)
            ref_p.append(o.p_cmd.copy())
    diff = float(np.abs(np.asarray(ref_p) - k1).max())
    assert diff < 5e-4, diff


def test_follower_effector_behaviour_is_unchanged(fcell):
    """Effector channels keep the rate-limited law regardless of mode."""
    widths = [t.effector.width for t in fcell.tcps]
    if not any(widths):
        pytest.skip("effectorless cell")
    it, p0, R0 = _interp(fcell)
    acts = np.zeros((4, fcell.action_dim), dtype=DTYPE)
    for i, (_pose, eff_sl) in enumerate(fcell.action_slices()):
        if widths[i]:
            acts[:, eff_sl] = 1.0
    it.set_chunk(acts, p0, R0)
    prev = it.eff.copy()
    dt_c = 1.0 / float(fcell.limits["rates"]["command_hz"])
    for _ in range(4 * TPA):
        it.step()
        step = np.abs(it.eff - prev)
        assert float(step.max()) <= float(it.eff_rate.max()) * dt_c + 1e-6
        prev = it.eff.copy()
    assert it.eff.max() <= 1.0 + 1e-6
