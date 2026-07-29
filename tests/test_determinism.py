"""Determinism.

This is the guarantee the product actually rests on, and it is stronger than the
reference-vs-kernel agreement in test_kernel_equivalence.py: the SAME kernels on
the SAME device must produce identical bits, run after run, so that a policy
trained in simulation and deployed on hardware is acting through arithmetic that
behaves the same way in both places.

That is why it can be asserted bitwise while the cross-implementation comparison
cannot: one thread per environment means no cross-thread reduction, no atomics,
and no scheduling dependence, and the fixed iteration counts and fixed constraint
ordering mean no data-dependent control flow that could reorder arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.kernels import available
from remoroo_lc.plant import Plant
from remoroo_lc.reference.controller import Controller

pytestmark = pytest.mark.skipif(not available(), reason="no kernel backend installed")


def _backend(cell, num_envs=1, device="cpu"):
    from remoroo_lc.kernels.warp_backend import BatchedController

    return BatchedController(cell, num_envs=num_envs, device=device)


def _cuda() -> bool:
    from remoroo_lc.kernels.warp_backend import cuda_available

    return cuda_available()


def _run(ctrl, cell, ticks=400, seed=7, device_batch=1):
    g = np.random.default_rng(seed)
    q = np.tile(cell.rest_posture(), (device_batch, 1))
    ctrl.reset(q)
    out = None
    trace = []
    for k in range(ticks):
        if k % 40 == 0:
            a = g.normal(0.0, 0.05, (8, cell.action_dim)).astype(np.float32)
            ctrl.set_chunk(a, q)
        out = ctrl.step(q)
        q = out["q_target"]
        trace.append(q.copy())
    return np.stack(trace), out


def test_same_device_is_bitwise_repeatable(cell):
    a_trace, a_out = _run(_backend(cell), cell)
    b_trace, b_out = _run(_backend(cell), cell)
    assert np.array_equal(a_trace, b_trace), "two identical runs differed"
    for key in ("q_target", "qd", "effector", "n_active", "max_violation"):
        assert np.array_equal(a_out[key], b_out[key]), f"{key} differed"


def test_reference_is_bitwise_repeatable(cell):
    """The NumPy path too -- it is what the tests and the scoreboard run on."""

    def run():
        ctrl, plant = Controller(cell), Plant(cell)
        g = np.random.default_rng(11)
        q = cell.rest_posture()
        plant.reset(q)
        ctrl.reset(q)
        trace = []
        for k in range(300):
            if k % 40 == 0:
                ctrl.set_chunk(g.normal(0.0, 0.05, (8, cell.action_dim)).astype(np.float32), q)
            o = ctrl.step(q)
            q, _ = plant.step(o.q_target)
            trace.append(q.copy())
        return np.stack(trace)

    assert np.array_equal(run(), run())


def test_environments_are_independent(cell):
    """Environment b's answer must not depend on what environment c is doing."""
    n = 8
    g = np.random.default_rng(13)
    Q = np.tile(cell.rest_posture(), (n, 1))
    Q += g.normal(0.0, 0.05, Q.shape).astype(np.float32)
    actions = g.normal(0.0, 0.04, (n, 4, cell.action_dim)).astype(np.float32)

    together = _backend(cell, n)
    together.reset(Q)
    together.set_chunk(actions, Q)
    out_all = together.step(Q)

    # Same environments, different neighbours: reverse the batch order.
    order = np.arange(n)[::-1]
    shuffled = _backend(cell, n)
    shuffled.reset(Q[order])
    shuffled.set_chunk(actions[order], Q[order])
    out_shuf = shuffled.step(Q[order])

    assert np.array_equal(out_all["q_target"], out_shuf["q_target"][order]), (
        "an environment's result changed when its batch neighbours did"
    )


def test_batch_size_does_not_change_the_answer(cell):
    q = cell.rest_posture()
    a = np.random.default_rng(17).normal(0.0, 0.04, (4, cell.action_dim)).astype(np.float32)
    results = []
    for n in (1, 3, 16):
        ctrl = _backend(cell, n)
        Q = np.tile(q, (n, 1))
        ctrl.reset(Q)
        ctrl.set_chunk(np.broadcast_to(a, (n,) + a.shape), Q)
        results.append(ctrl.step(Q)["q_target"][0])
    for r in results[1:]:
        assert np.array_equal(results[0], r), "batch size changed the answer"


def test_warm_start_does_not_break_repeatability(cell):
    """Carried dual state must not make a run depend on how it was reached."""
    ctrl = _backend(cell)
    trace_a, _ = _run(ctrl, cell, ticks=200)
    # Same controller object, reset and run again: the reset must clear lambda.
    trace_b, _ = _run(ctrl, cell, ticks=200)
    assert np.array_equal(trace_a, trace_b), (
        "a second run on the same controller differed; reset is not clearing state"
    )


def test_constraint_order_is_a_property_of_the_cell(cell):
    """The shipped row order must be reproducible from the config alone."""
    from remoroo_lc.kernels.structure import build_structure

    a = build_structure(cell)
    b = build_structure(cell)
    assert np.array_equal(a.pair_kind, b.pair_kind)
    assert np.array_equal(a.pair_a, b.pair_a)
    assert np.array_equal(a.pair_b, b.pair_b)
    assert a.walls.row_labels() == b.walls.row_labels()
    assert a.n_rows == b.n_rows


@pytest.mark.cuda
@pytest.mark.skipif(not available() or not _cuda(), reason="no CUDA device")
def test_cpu_and_cuda_agree(cell):
    """CPU vs CUDA, on the quantities whose conditioning permits a tight bound.

    Not run here -- this machine has no CUDA device -- but written, and it is the
    check that matters most for sim/real parity, because training runs on GPU and
    the cell's edge box does not.  The tolerances mirror
    test_kernel_equivalence.py for the same reason: the well-conditioned stages
    get float32 precision, and q_dot gets the amplified bound.
    """
    q = cell.rest_posture()
    a = np.random.default_rng(19).normal(0.0, 0.04, (4, cell.action_dim)).astype(np.float32)
    outs = {}
    for device in ("cpu", "cuda"):
        ctrl = _backend(cell, 1, device=device)
        ctrl.reset(q[None])
        ctrl.set_chunk(a, q[None])
        outs[device] = ctrl.step(q[None])
    c, g = outs["cpu"], outs["cuda"]
    assert np.max(np.abs(c["p_meas"] - g["p_meas"])) <= 1e-5
    assert np.max(np.abs(c["p_cmd"] - g["p_cmd"])) <= 1e-5
    assert np.array_equal(c["n_active"], g["n_active"])
    rel = np.max(np.abs(c["qd"] - g["qd"])) / max(float(np.max(np.abs(c["qd"]))), 1e-3)
    assert rel <= 1e-5 * 1.25e4, "q_dot beyond the documented diff-IK amplification"
