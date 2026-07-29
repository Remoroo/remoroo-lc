"""The adapter layer.

The integration tests here close the loop through `CellAdapter` rather than
around it, so the composition logic -- unit splitting, joint ordering, effector
routing, limit cross-checks, estop fan-out -- is exercised by the same code path
hardware will use.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.adapters.base import CellAdapter, RobotAdapter, UnitLimits
from remoroo_lc.adapters.mock import MockCellAdapter, MockUnit
from remoroo_lc.reference.controller import Controller


def test_mock_units_implement_the_protocol(cell):
    ad = MockCellAdapter(cell)
    assert len(ad.adapters) == len(cell.models)
    for unit in ad.adapters.values():
        assert isinstance(unit, RobotAdapter)
        assert isinstance(unit.limits(), UnitLimits)


def test_joint_slices_tile_the_global_vector(cell):
    ad = MockCellAdapter(cell)
    covered = np.zeros(cell.n_joints, dtype=int)
    for sl in ad.slices.values():
        covered[sl] += 1
    assert np.all(covered == 1)
    assert ad.n_joints == cell.n_joints
    assert ad.eff_dim == sum(cell.effector_widths)


def test_connect_checks_the_cell_against_the_hardware(cell):
    ad = MockCellAdapter(cell)
    ad.connect()  # the mock reports slightly wider limits, so this passes
    ad.disconnect()

    # Now make the hardware narrower than the cell claims.
    ad = MockCellAdapter(cell)
    first = next(iter(ad.adapters.values()))
    first._q_hi = first._q_hi - np.float32(1.0)
    with pytest.raises(ValueError, match="wider than the hardware"):
        ad.connect()


def test_wrong_adapter_set_is_rejected(cell):
    from remoroo_lc.plant import Plant

    plant = Plant(cell)
    good = MockCellAdapter(cell, plant)
    with pytest.raises(ValueError, match="does not match the cell"):
        CellAdapter(cell, {"not_a_model": next(iter(good.adapters.values()))})


def test_closed_loop_through_the_adapter(cell):
    """Controller + CellAdapter + plant, on the standard hold command."""
    ad = MockCellAdapter(cell)
    ctrl = Controller(cell)
    q = cell.rest_posture()
    ad.reset(q)
    ad.connect()
    ctrl.reset(q)

    lo, hi = cell.joint_limits()
    for k in range(400):
        if k % 125 == 0:
            ctrl.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32), q)
        out = ctrl.step(q)
        ad.stream_targets(out.q_target)
        ad.set_effector(out.effector)
        q, _ = ad.tick()
        assert np.all(q >= lo - 1e-3) and np.all(q <= hi + 1e-3)

    assert np.max(np.abs(q - cell.rest_posture())) < 1e-3
    q_read, qd_read = ad.read_state()
    assert np.allclose(q_read, q)
    assert ad.effector_state.shape == (sum(cell.effector_widths),)
    ad.disconnect()


def test_effector_routes_to_the_right_unit():
    """A cell whose TCPs have different effector widths must not cross them."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "mixed.yaml")
    assert cell.effector_widths == (1, 2)
    ad = MockCellAdapter(cell)
    ad.connect()
    ad.set_effector(np.asarray([0.25, 0.5, 0.75], dtype=np.float32))
    left = ad.adapters[cell.tcps[0].model].effector
    right = ad.adapters[cell.tcps[1].model].effector
    assert np.allclose(left, [0.25])
    assert np.allclose(right, [0.5, 0.75])
    ad.disconnect()


def test_one_model_with_several_tcps_gets_all_its_channels():
    """A torso with two limbs is ONE unit with TWO effectors."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "branched_trunk.yaml")
    assert len({t.model for t in cell.tcps}) == 1 and cell.n_tcps == 2
    ad = MockCellAdapter(cell)
    assert ad.eff_index[cell.tcps[0].model].tolist() == [0, 1]
    ad.connect()
    ad.set_effector(np.asarray([0.3, 0.8], dtype=np.float32))
    assert np.allclose(ad.effector_state, [0.3, 0.8])
    ad.disconnect()


def test_effectorless_cell_has_no_effector_channels():
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_DIR

    cell = load_cell(CELL_DIR / "single_6dof_leg.yaml")
    ad = MockCellAdapter(cell)
    assert ad.eff_dim == 0
    assert ad.eff_index == {}
    ad.connect()
    ad.set_effector(np.zeros(0, dtype=np.float32))  # must be a no-op, not an error
    ad.disconnect()


def test_estop_reaches_every_unit_and_holds_position(cell):
    ad = MockCellAdapter(cell)
    q = cell.rest_posture()
    ad.reset(q)
    ad.connect()
    ad.stream_targets(q + np.float32(0.2))
    ad.estop()
    for unit in ad.adapters.values():
        assert unit.estopped
    # After an estop, streamed targets are ignored and the plant holds.
    ad.stream_targets(q + np.float32(0.5))
    for _ in range(50):
        q_now, _ = ad.tick()
    assert np.max(np.abs(q_now - q)) < 1e-2, "estopped units should hold position"


def test_estop_tries_every_unit_even_if_one_throws(cell):
    if len(cell.models) < 2:
        pytest.skip("needs a multi-unit cell")
    ad = MockCellAdapter(cell)
    ad.connect()
    names = list(ad.adapters)

    class Exploding(MockUnit):
        def estop(self):
            raise RuntimeError("boom")

    bad = ad.adapters[names[0]]
    ad.adapters[names[0]] = Exploding(
        bad.name, bad._plant, bad._slice, bad._effector_width,
        bad._command_hz, bad._q_lo, bad._q_hi, bad._qd_max,
    )
    with pytest.raises(RuntimeError, match="estop failed"):
        ad.estop()
    assert ad.adapters[names[1]].estopped, "the other unit must still have been stopped"


def test_context_manager_estops_on_exception(cell):
    ad = MockCellAdapter(cell)
    with pytest.raises(ValueError):
        with ad:
            raise ValueError("something went wrong mid-episode")
    for unit in ad.adapters.values():
        assert unit.estopped


# --------------------------------------------------------------------------- #
# Isaac Lab seam, against a fake environment
# --------------------------------------------------------------------------- #


class FakeEnv:
    """Implements exactly the two methods the adapter needs."""

    def __init__(self, cell, num_envs, order=None):
        from remoroo_lc.plant import Plant

        self.num_envs = num_envs
        self.cell = cell
        self.order = np.arange(cell.n_joints) if order is None else np.asarray(order)
        self.plants = [Plant(cell) for _ in range(num_envs)]
        for p in self.plants:
            p.reset(cell.rest_posture())
        self.writes = 0

    def read_joint_state(self):
        q = np.stack([p.state[0] for p in self.plants])
        qd = np.stack([p.state[1] for p in self.plants])
        inv = np.argsort(self.order)
        return q[:, inv], qd[:, inv]

    def write_joint_targets(self, q_target):
        self.writes += 1
        t = np.asarray(q_target)[:, self.order]
        for i, p in enumerate(self.plants):
            p.step(t[i])


@pytest.mark.skipif(
    not __import__("remoroo_lc.kernels", fromlist=["available"]).available(),
    reason="no kernel backend",
)
def test_isaaclab_adapter_drives_a_fake_env(cell):
    from remoroo_lc.adapters.isaaclab import IsaacLabAdapter

    env = FakeEnv(cell, num_envs=4)
    ad = IsaacLabAdapter(cell, env, device="cpu")
    ad.reset()
    ad.set_chunk(np.zeros((8, cell.action_dim), dtype=np.float32))
    for _ in range(60):
        out = ad.step()
    assert env.writes == 60
    assert out["q_target"].shape == (4, cell.n_joints)
    q, _ = env.read_joint_state()
    assert np.max(np.abs(q - cell.rest_posture())) < 1e-2


@pytest.mark.skipif(
    not __import__("remoroo_lc.kernels", fromlist=["available"]).available(),
    reason="no kernel backend",
)
def test_isaaclab_adapter_respects_joint_order(cell):
    """A permuted environment must produce the same physical result."""
    from remoroo_lc.adapters.isaaclab import IsaacLabAdapter

    order = np.arange(cell.n_joints)[::-1].copy()
    plain = FakeEnv(cell, num_envs=2)
    permuted = FakeEnv(cell, num_envs=2, order=order)

    a = IsaacLabAdapter(cell, plain, device="cpu")
    b = IsaacLabAdapter(cell, permuted, device="cpu", joint_order=order.tolist())
    for ad in (a, b):
        ad.reset()
        ad.set_chunk(np.zeros((4, cell.action_dim), dtype=np.float32))
    for _ in range(40):
        a.step()
        b.step()
    # Both FakeEnvs hold their plants in CELL order and permute only at the
    # boundary, so if the adapter's permutation is right the two are identical.
    qa, _ = plain.read_joint_state()
    qb, _ = permuted.read_joint_state()
    assert np.allclose(qa, qb[:, np.argsort(order)], atol=1e-5), (
        "the permuted environment ended up somewhere else, so joint_order is wrong"
    )


@pytest.mark.skipif(
    not __import__("remoroo_lc.kernels", fromlist=["available"]).available(),
    reason="no kernel backend",
)
def test_isaaclab_adapter_rejects_a_bad_permutation(cell):
    from remoroo_lc.adapters.isaaclab import IsaacLabAdapter

    env = FakeEnv(cell, num_envs=1)
    with pytest.raises(ValueError, match="permutation"):
        IsaacLabAdapter(cell, env, device="cpu", joint_order=[0] * cell.n_joints)


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


@pytest.mark.hw
def test_xarm_servo_smoke():
    """Stream a 5 s clean tape in servo mode, with a guard on the diagnostics.

    NEVER run in CI.  Requires a real cell, a real estop within reach, and a
    person watching.  Halts on any hard violation or on a solve p99 above 4 ms,
    because a controller that is late is a controller that is not filtering.
    """
    import os
    import time

    from remoroo_lc.adapters.base import CellAdapter
    from remoroo_lc.adapters.xarm import XArmUnit
    from remoroo_lc.schema import load_cell
    from remoroo_lc.tapes import figure_eight

    cell_path = os.environ.get("REMOROO_LC_HW_CELL")
    if not cell_path:
        pytest.skip("set REMOROO_LC_HW_CELL to the real cell file")
    cell = load_cell(cell_path)
    hosts = os.environ["REMOROO_LC_HW_HOSTS"].split(",")
    assert len(hosts) == len(cell.models)

    lo, hi = cell.joint_limits()
    qd_max = cell.joint_velocity_limits()
    units, k = {}, 0
    for m, host in zip(cell.models, hosts):
        n = len(m.joint_names)
        sl = slice(k, k + n)
        k += n
        units[m.name] = XArmUnit(
            name=m.name,
            host=host,
            n_joints=n,
            command_hz=float(cell.limits["rates"]["command_hz"]),
            q_lo=lo[sl],
            q_hi=hi[sl],
            qd_max=qd_max[sl],
            has_effector=any(t.model == m.name and t.effector.width for t in cell.tcps),
        )
    adapter = CellAdapter(cell, units)

    # Conservative: a quarter of the shipped task limits for a first contact.
    for axis in ("linear", "angular"):
        for key in ("v_max", "a_max", "j_max"):
            cell.limits["task"][axis][key] *= 0.25

    ctrl = Controller(cell)
    tape = figure_eight(cell, size=0.03, speed=0.02)
    dt = 1.0 / float(cell.limits["rates"]["command_hz"])

    with adapter:
        q, _ = adapter.read_state()
        ctrl.reset(q)
        solve_ms = []
        deadline = time.perf_counter()
        for s in range(0, tape.actions.shape[0], 8):
            ctrl.set_chunk(tape.actions[s : s + 8], q)
            for _ in range(8 * int(round(1.0 / dt / float(cell.limits["rates"]["policy_hz"])))):
                t0 = time.perf_counter()
                out = ctrl.step(q)
                solve_ms.append((time.perf_counter() - t0) * 1e3)
                if out.diag["min_pair_distance"] <= 0.0:
                    adapter.estop()
                    pytest.fail(f"hard violation: {out.diag['min_pair_distance']:.4f} m")
                if len(solve_ms) > 100 and np.percentile(solve_ms, 99) > 4.0:
                    adapter.estop()
                    pytest.fail(f"solve p99 {np.percentile(solve_ms, 99):.2f} ms > 4 ms")
                adapter.stream_targets(out.q_target)
                adapter.set_effector(out.effector)
                deadline += dt
                time.sleep(max(0.0, deadline - time.perf_counter()))
                q, _ = adapter.read_state()
