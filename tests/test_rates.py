"""One rate, one number.

These tests exist because of a measured failure, not a hypothetical one.  The cell
declared `policy_hz: 50` and the training worldspec declared `25`; remoroo-lc sized
its action chunk from the cell (20 ms) while the trainer fed it ticks from the spec
(40 ms).  Every chunk was therefore executed as a 2x-speed sprint followed by the
interpolator's deceleration runway -- a 0.92 -> 1.53 sawtooth in the per-tick
profile -- on every action of every episode of roughly twenty training runs.

Nothing raised.  Nothing logged.  The curve merely looked bad.

The contract now has two layers, and both are tested here:

  1. `resolve_rates` -- a rate may be DECLARED by a caller, and the cell's value wins;
     a disagreement raises rather than being silently resolved in either direction.
  2. `assert_drive_rate` -- the invariant on the built object: one action lasts exactly
     as long as the chunk the controller sized for it.  This one catches a divergence
     arriving from anywhere at all, including places nobody has thought of yet.
"""

from __future__ import annotations

import copy

import pytest

from remoroo_lc.kernels.structure import assert_drive_rate, build_structure
from remoroo_lc.schema import RateMismatch, resolve_rates

#: Ticks per action used by the invariant tests below.
_TICKS = 10


@pytest.fixture
def chunked_cell(cell):
    """`cell`, re-rated so a whole number of control ticks spans one action.

    The shipped test cells declare 16 Hz against 250 Hz -- 15.625 ticks per action --
    which is legal for a tick-by-tick driver but cannot express "half the tick budget",
    the exact shape of the bug these tests exist to catch.  Skipping on those cells
    would leave the regression untested on every cell in the matrix, so the rate is
    adjusted here instead: the point of the test is the invariant, not the fixture.
    """
    out = copy.deepcopy(cell)
    out.limits = copy.deepcopy(cell.limits)
    out.limits["rates"]["policy_hz"] = float(out.limits["rates"]["command_hz"]) / _TICKS
    return out


# --------------------------------------------------------------------------- #
# layer 1: two declarations of the same rate must agree
# --------------------------------------------------------------------------- #


def test_agreeing_declaration_is_accepted(cell):
    """Declaring exactly what the cell says is how a caller proves it read it."""
    own = cell.limits["rates"]
    out = resolve_rates(
        cell, policy_hz=float(own["policy_hz"]), command_hz=float(own["command_hz"])
    )
    assert out["policy_hz"] == float(own["policy_hz"])
    assert out["command_hz"] == float(own["command_hz"])
    assert out["dt_p"] == pytest.approx(1.0 / float(own["policy_hz"]))
    assert out["dt_c"] == pytest.approx(1.0 / float(own["command_hz"]))


def test_no_declaration_is_accepted(cell):
    """A caller that does not claim to know the rate is not forced to guess one."""
    out = resolve_rates(cell)
    assert out["policy_hz"] == float(cell.limits["rates"]["policy_hz"])


def test_disagreeing_policy_hz_raises(cell):
    """THE REGRESSION.  The exact shape of the bug: cell 50, caller 25."""
    own = float(cell.limits["rates"]["policy_hz"])
    with pytest.raises(RateMismatch) as exc:
        resolve_rates(cell, policy_hz=own * 2.0, source="the training worldspec")
    msg = str(exc.value)
    # The message must carry BOTH numbers and the factor -- an operator holding two
    # config files needs to know which one to change and by how much.
    assert "policy_hz" in msg
    assert "the training worldspec" in msg
    assert f"{own:g}" in msg
    assert "2" in msg


def test_disagreeing_command_hz_raises(cell):
    own = float(cell.limits["rates"]["command_hz"])
    with pytest.raises(RateMismatch):
        resolve_rates(cell, command_hz=own + 10.0)


def test_a_tiny_disagreement_still_raises(cell):
    """No tolerance band: 25.0001 Hz against 25 Hz is a typo, and typos scale."""
    own = float(cell.limits["rates"]["policy_hz"])
    with pytest.raises(RateMismatch):
        resolve_rates(cell, policy_hz=own + 1e-4)


def test_missing_rate_raises(cell):
    bad = copy.deepcopy(cell)
    bad.limits = copy.deepcopy(cell.limits)
    del bad.limits["rates"]["policy_hz"]
    with pytest.raises(RateMismatch, match="not declared"):
        resolve_rates(bad)


def test_nonpositive_rate_raises(cell):
    bad = copy.deepcopy(cell)
    bad.limits = copy.deepcopy(cell.limits)
    bad.limits["rates"]["policy_hz"] = 0.0
    with pytest.raises(RateMismatch):
        resolve_rates(bad)


def test_exact_ticks_only_enforced_when_asked(cell):
    """Exactness binds on whoever CHUNKS, not on merely holding two periods.

    A controller stepped tick-by-tick against a dt_p-long chunk is fine at a
    fractional ratio; a driver running a fixed tick count per action is not.
    """
    bad = copy.deepcopy(cell)
    bad.limits = copy.deepcopy(cell.limits)
    bad.limits["rates"]["policy_hz"] = 16.0
    bad.limits["rates"]["command_hz"] = 250.0        # 15.625 ticks per action

    loose = resolve_rates(bad)                       # tolerated
    assert loose["ticks_exact"] == 0.0
    assert loose["ticks_per_action"] == 0.0

    with pytest.raises(RateMismatch, match="exact integer multiple"):
        resolve_rates(bad, require_exact_ticks=True)


# --------------------------------------------------------------------------- #
# layer 2: the invariant on the built object
# --------------------------------------------------------------------------- #


def test_structure_carries_the_resolved_rates(cell):
    st = build_structure(cell)
    assert st.policy_hz == float(cell.limits["rates"]["policy_hz"])
    assert st.command_hz == float(cell.limits["rates"]["command_hz"])
    assert st.dt_p == pytest.approx(1.0 / st.policy_hz)
    assert st.dt_c == pytest.approx(1.0 / st.command_hz)


def test_build_structure_rejects_a_disagreeing_declaration(cell):
    own = float(cell.limits["rates"]["policy_hz"])
    with pytest.raises(RateMismatch):
        build_structure(cell, policy_hz=own / 2.0)


def test_drive_rate_invariant_accepts_the_matching_tick_count(chunked_cell):
    """ticks * dt_c == dt_p is the whole contract."""
    st = build_structure(chunked_cell)
    assert st.ticks_per_action == _TICKS
    assert_drive_rate(st, _TICKS, "a test driver")


def test_drive_rate_invariant_catches_the_measured_2x(chunked_cell):
    """THE REGRESSION, caught on the live object rather than on yaml text.

    A driver that feeds HALF the ticks the chunk was sized for runs it at 2x speed.
    This is the check that would have fired on step one of run one.
    """
    st = build_structure(chunked_cell)
    with pytest.raises(RateMismatch) as exc:
        assert_drive_rate(st, _TICKS // 2, "the training seam")
    msg = str(exc.value)
    assert "the training seam" in msg
    # The message must name the multiplier, because that number IS the bug.
    assert "x the commanded speed" in msg
    assert "2x" in msg


def test_drive_rate_invariant_catches_an_overlong_tick_budget(chunked_cell):
    """The mirror image: too many ticks runs the chunk in slow motion."""
    st = build_structure(chunked_cell)
    with pytest.raises(RateMismatch, match="0.5x the commanded speed"):
        assert_drive_rate(st, _TICKS * 2, "the training seam")


def test_drive_rate_invariant_catches_an_off_by_one(chunked_cell):
    """Not just gross factors: 9 ticks where 10 were planned is a 1.11x sprint."""
    st = build_structure(chunked_cell)
    with pytest.raises(RateMismatch):
        assert_drive_rate(st, _TICKS - 1, "the training seam")


def test_drive_rate_invariant_rejects_a_nonsense_tick_count(cell):
    st = build_structure(cell)
    with pytest.raises(RateMismatch):
        assert_drive_rate(st, 0, "a broken driver")


def test_drive_rate_is_reachable_from_the_controller(chunked_cell):
    """The seam calls this through the controller, so the controller must expose it."""
    pytest.importorskip("warp")
    from remoroo_lc.kernels.warp_backend import BatchedController

    ctrl = BatchedController(chunked_cell, num_envs=2, device="cpu")
    assert ctrl.policy_hz == float(chunked_cell.limits["rates"]["policy_hz"])
    assert ctrl.command_hz == float(chunked_cell.limits["rates"]["command_hz"])
    assert ctrl.ticks_per_action == _TICKS

    ctrl.assert_drive_rate(_TICKS, "a test driver")
    with pytest.raises(RateMismatch):
        ctrl.assert_drive_rate(_TICKS // 2, "the training seam")


def test_controller_rejects_a_disagreeing_policy_hz(cell):
    pytest.importorskip("warp")
    from remoroo_lc.kernels.warp_backend import BatchedController

    own = float(cell.limits["rates"]["policy_hz"])
    with pytest.raises(RateMismatch):
        BatchedController(cell, num_envs=2, device="cpu", policy_hz=own * 2.0)
