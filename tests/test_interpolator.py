"""Layer 1: limits are respected structurally, and seams carry state."""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import POINT_DIM, TASK_DIM
from remoroo_lc.reference.interpolator import AxisLimits, ChunkInterpolator, jerk_limited_step
from remoroo_lc.spatial import exp_so3


def _fresh(cell):
    interp = ChunkInterpolator(cell)
    p = np.zeros((cell.n_tcps, POINT_DIM), dtype=np.float32)
    R = np.stack([np.eye(POINT_DIM, dtype=np.float32)] * cell.n_tcps)
    interp.reset(p, R)
    return interp


def test_limits_are_respected_on_random_chunks(cell, rng):
    interp = _fresh(cell)
    lim = interp.lim
    dt = float(interp.dt_c)
    prev_a = interp.a.copy()
    worst = {"v": 0.0, "a": 0.0, "j": 0.0}
    for _ in range(12):
        actions = rng.normal(0.0, 0.08, size=(6, cell.action_dim)).astype(np.float32)
        p = interp.x[:, :POINT_DIM].copy()
        R = np.stack([exp_so3(interp.x[i, POINT_DIM:]) for i in range(cell.n_tcps)])
        interp.set_chunk(actions, p, R)
        for _ in range(int(round(6 / float(interp.dt_p) * float(interp.dt_c) / float(interp.dt_c)))
                       * 0 + 120):
            interp.step()
            j = (interp.a - prev_a) / dt
            worst["v"] = max(worst["v"], float(np.max(np.abs(interp.v) - lim.v_max)))
            worst["a"] = max(worst["a"], float(np.max(np.abs(interp.a) - lim.a_max)))
            worst["j"] = max(worst["j"], float(np.max(np.abs(j) - lim.j_max)))
            prev_a = interp.a.copy()
    tol = 1e-3
    assert worst["v"] <= tol, worst
    assert worst["a"] <= tol, worst
    assert worst["j"] <= 1e-2, worst  # float32 division by dt amplifies rounding


def test_converges_to_a_held_target(cell):
    """A chunk that stops moving must be reached and held, not orbited."""
    interp = _fresh(cell)
    actions = np.zeros((4, cell.action_dim), dtype=np.float32)
    for pose, _eff in cell.action_slices():
        actions[:, pose.start] = 0.05  # 5 cm step on x, held for the chunk
    p = interp.x[:, :POINT_DIM].copy()
    R = np.stack([np.eye(POINT_DIM, dtype=np.float32)] * cell.n_tcps)
    interp.set_chunk(actions, p, R)
    for _ in range(1500):
        interp.step()
    target = 4 * 0.05  # cumulative mode: four 5 cm deltas
    assert np.allclose(interp.x[:, 0], target, atol=1e-4), interp.x[:, 0]
    assert np.max(np.abs(interp.v)) < 1e-3
    assert np.max(np.abs(interp.a)) < 1e-2


def test_state_is_carried_across_a_seam(cell):
    """(x, v, a) must not be reset at a chunk boundary."""
    interp = _fresh(cell)
    actions = np.zeros((4, cell.action_dim), dtype=np.float32)
    for pose, _eff in cell.action_slices():
        actions[:, pose.start] = 0.10
    p = interp.x[:, :POINT_DIM].copy()
    R = np.stack([np.eye(POINT_DIM, dtype=np.float32)] * cell.n_tcps)
    interp.set_chunk(actions, p, R)
    for _ in range(40):
        interp.step()
    v_before, a_before, x_before = interp.v.copy(), interp.a.copy(), interp.x.copy()
    assert np.max(np.abs(v_before)) > 1e-3, "test needs the axis to be moving"
    interp.set_chunk(actions, interp.x[:, :POINT_DIM].copy(), R)
    assert np.array_equal(interp.v, v_before)
    assert np.array_equal(interp.a, a_before)
    assert np.array_equal(interp.x, x_before)


def test_effector_channels_are_rate_limited(cell):
    if sum(cell.effector_widths) == 0:
        assert _fresh(cell).eff.shape == (0,)
        return
    interp = _fresh(cell)
    actions = np.zeros((2, cell.action_dim), dtype=np.float32)
    for _pose, eff in cell.action_slices():
        actions[:, eff] = 1.0
    p = interp.x[:, :POINT_DIM].copy()
    R = np.stack([np.eye(POINT_DIM, dtype=np.float32)] * cell.n_tcps)
    interp.set_chunk(actions, p, R)
    prev = interp.eff.copy()
    max_rate = float(np.max(interp.eff_rate))
    for _ in range(400):
        interp.step()
        step = np.max(np.abs(interp.eff - prev)) / float(interp.dt_c)
        assert step <= max_rate + 1e-3
        prev = interp.eff.copy()
    assert np.allclose(interp.eff, 1.0, atol=1e-4)
    assert np.all(interp.eff <= 1.0) and np.all(interp.eff >= 0.0)


def test_effector_stays_in_unit_range(cell, rng):
    if sum(cell.effector_widths) == 0:
        return
    interp = _fresh(cell)
    for _ in range(6):
        actions = rng.uniform(-3.0, 3.0, size=(4, cell.action_dim)).astype(np.float32)
        p = interp.x[:, :POINT_DIM].copy()
        R = np.stack([exp_so3(interp.x[i, POINT_DIM:]) for i in range(cell.n_tcps)])
        interp.set_chunk(actions, p, R)
        for _ in range(80):
            interp.step()
            assert np.all(interp.eff >= 0.0) and np.all(interp.eff <= 1.0)


def test_rotation_state_is_continuous_through_half_turns(cell):
    """The rotation-vector state must unwrap, not jump by 2 pi."""
    interp = _fresh(cell)
    R = np.stack([np.eye(POINT_DIM, dtype=np.float32)] * cell.n_tcps)
    actions = np.zeros((8, cell.action_dim), dtype=np.float32)
    for pose, _eff in cell.action_slices():
        actions[:, pose.start + POINT_DIM + 2] = 0.7  # ~40 deg per action about z
    prev = interp.x[:, POINT_DIM:].copy()
    for _ in range(4):
        p = interp.x[:, :POINT_DIM].copy()
        Rc = np.stack([exp_so3(interp.x[i, POINT_DIM:]) for i in range(cell.n_tcps)])
        interp.set_chunk(actions, p, Rc)
        for _ in range(200):
            interp.step()
            jump = np.max(np.abs(interp.x[:, POINT_DIM:] - prev))
            assert jump < 0.05, "rotation state jumped; the unwrap is broken"
            prev = interp.x[:, POINT_DIM:].copy()
    total = np.abs(interp.x[:, POINT_DIM + 2])
    assert np.all(total > 3.5), "should have accumulated well past pi without wrapping"
    _ = R


def test_jerk_limited_step_is_a_pure_function(rng):
    lim = AxisLimits(
        np.full(TASK_DIM, 0.5, dtype=np.float32),
        np.full(TASK_DIM, 2.5, dtype=np.float32),
        np.full(TASK_DIM, 50.0, dtype=np.float32),
    )
    x = rng.normal(size=TASK_DIM).astype(np.float32)
    v = rng.normal(size=TASK_DIM).astype(np.float32) * 0.1
    a = rng.normal(size=TASK_DIM).astype(np.float32) * 0.1
    tgt = rng.normal(size=TASK_DIM).astype(np.float32)
    r1 = jerk_limited_step(x, v, a, tgt, lim, 0.004)
    r2 = jerk_limited_step(x, v, a, tgt, lim, 0.004)
    for a1, a2 in zip(r1, r2):
        assert np.array_equal(a1, a2)
