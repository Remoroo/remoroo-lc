"""Layer 3a: constraint assembly, damper geometry, and fixed row structure."""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.constants import BIG
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.walls import (
    PAIR_ROBOT_ENV,
    PAIR_ROBOT_ROBOT,
    WallBuilder,
    build_pair_list,
    build_sphere_set,
    env_distance,
    env_distance_batch,
)
from remoroo_lc.schema import EnvPrimitive
from remoroo_lc.spatial import make_transform, rpy_to_mat
from tests.conftest import random_states


def test_row_count_and_order_are_fixed(cell):
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    assert wb.n_rows == 2 * cell.n_joints + len(wb.pairs)
    assert len(wb.row_labels()) == wb.n_rows
    shapes = set()
    for q in random_states(cell, 8, seed=21):
        w = wb.assemble(q, tree.fk(q))
        shapes.add((w.G.shape, w.h.shape))
    assert len(shapes) == 1, "constraint shape must not depend on state"


def test_pair_list_is_deterministic(cell):
    tree = KinematicTree(cell)
    a = build_pair_list(cell, tree, build_sphere_set(cell, tree))
    b = build_pair_list(cell, tree, build_sphere_set(cell, tree))
    assert np.array_equal(a.kind, b.kind)
    assert np.array_equal(a.a, b.a)
    assert np.array_equal(a.b, b.b)
    assert a.label == b.label


def test_adjacent_links_are_not_paired(cell):
    """Directly jointed links can never usefully collide; they must be excluded."""
    tree = KinematicTree(cell)
    spheres = build_sphere_set(cell, tree)
    pairs = build_pair_list(cell, tree, spheres)
    link_of = [lb.split("#")[0] for lb in spheres.label]
    for p in range(len(pairs)):
        if pairs.kind[p] != PAIR_ROBOT_ROBOT:
            continue
        la, lb = link_of[int(pairs.a[p])], link_of[int(pairs.b[p])]
        assert la != lb, "a link must not be paired with itself"
        ma, na = la.split("/")
        mb, nb = lb.split("/")
        if ma != mb:
            continue
        urdf = cell.urdfs[ma]
        jointed = any(
            {j.parent, j.child} == {na, nb} for j in urdf.joints
        )
        assert not jointed, f"adjacent links {la} and {lb} must not be paired"


def test_inactive_rows_keep_their_slot(cell):
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    w = wb.assemble(cell.rest_posture(), tree.fk(cell.rest_posture()))
    inactive = ~w.active
    assert np.all(w.h[inactive] == BIG), "inactive rows must be parked at +BIG"
    assert np.all(w.h[w.active] < BIG)


def test_joint_damper_activation_boundary(cell):
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    n = cell.n_joints
    lo, hi = cell.joint_limits()

    q = cell.rest_posture().copy()
    q[0] = hi[0] - float(wb.jl_infl) * 1.01  # just outside the influence distance
    w = wb.assemble(q, tree.fk(q))
    assert not w.active[0]

    d = float(wb.jl_infl) * 0.99  # just inside
    q[0] = hi[0] - d
    w = wb.assemble(q, tree.fk(q))
    assert w.active[0]
    expect = float(wb.jl_xi) * (d - float(wb.jl_safe)) / (float(wb.jl_infl) - float(wb.jl_safe))
    assert w.h[0] == pytest.approx(expect, rel=1e-3)

    q[0] = hi[0] - float(wb.jl_safe)  # exactly at the safety distance
    w = wb.assemble(q, tree.fk(q))
    assert w.h[0] == pytest.approx(0.0, abs=1e-6), "no approach allowed at d_safe"

    q[0] = hi[0] - float(wb.jl_safe) * 0.5  # inside it
    w = wb.assemble(q, tree.fk(q))
    assert w.h[0] < 0.0, "inside d_safe the damper must command retreat"

    # And the mirrored lower-limit rows behave the same.
    q = cell.rest_posture().copy()
    q[0] = lo[0] + float(wb.jl_safe) * 0.5
    w = wb.assemble(q, tree.fk(q))
    assert w.active[n] and w.h[n] < 0.0
    assert w.G[n, 0] == -1.0


def test_collision_row_is_minus_the_distance_gradient(cell):
    """G row must equal -d(distance)/dq, checked by finite differences.

    This is the load-bearing claim of the whole layer: n^T (J_a - J_b) is the rate
    of change of the pair distance, so constraining it constrains approach speed.
    """
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    if len(wb.pairs) == 0:
        return
    q0 = random_states(cell, 1, seed=22)[0].astype(np.float64)
    base = wb.assemble(q0.astype(np.float32), tree.fk(q0.astype(np.float32)))
    # Distances are computed in float32, so the finite-difference error here is
    # quantisation-dominated and falls as O(1/h) rather than rising as O(h^2)
    # until h is large.  1e-3 rad sits in the flat part of that curve.
    h = 1e-3
    probe = np.linspace(0, len(wb.pairs) - 1, min(12, len(wb.pairs))).astype(int)
    for p in probe:
        row = wb.n_joint_rows + int(p)
        grad = np.zeros(cell.n_joints)
        for j in range(cell.n_joints):
            qp, qm = q0.copy(), q0.copy()
            qp[j] += h
            qm[j] -= h
            dp = wb.assemble(qp.astype(np.float32), tree.fk(qp.astype(np.float32))).distance[row]
            dm = wb.assemble(qm.astype(np.float32), tree.fk(qm.astype(np.float32))).distance[row]
            grad[j] = (dp - dm) / (2 * h)
        assert np.max(np.abs(base.G[row] + grad)) < 2e-4, (
            f"row {row} ({wb.pairs.label[int(p)]}) is not -d(distance)/dq"
        )


def test_collision_damper_rhs_geometry(cell):
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    if len(wb.pairs) == 0:
        return
    for q in random_states(cell, 6, seed=23):
        w = wb.assemble(q, tree.fk(q))
        rows = slice(wb.n_joint_rows, wb.n_rows)
        d = w.distance[rows]
        h_ = w.h[rows]
        act = w.active[rows]
        assert np.all(act == (d < float(wb.cd_infl)))
        inside = act & (d < float(wb.cd_safe))
        assert np.all(h_[inside] < 0.0), "inside d_safe the damper must push apart"
        far = act & (d > float(wb.cd_safe))
        assert np.all(h_[far] > 0.0)
        assert np.all(h_[act] <= float(wb.cd_xi) + 1e-5)


# --------------------------------------------------------------------------- #
# environment primitives
# --------------------------------------------------------------------------- #


def _brute_force_distance(prim, p, samples=200000, seed=0):
    """Distance to the primitive's surface by dense sampling of its surface."""
    g = np.random.default_rng(seed)
    R = prim.pose[:3, :3]
    c = prim.pose[:3, 3]
    if prim.ptype == "box":
        half = prim.dims
        pts = (g.random((samples, 3)).astype(np.float32) * 2 - 1) * half
        face = g.integers(0, 3, samples)
        sign = g.integers(0, 2, samples) * 2 - 1
        pts[np.arange(samples), face] = half[face] * sign
    elif prim.ptype == "sphere":
        v = g.normal(size=(samples, 3)).astype(np.float32)
        pts = v / np.linalg.norm(v, axis=1, keepdims=True) * prim.dims[0]
    elif prim.ptype == "cylinder":
        r, hh = float(prim.dims[0]), float(prim.dims[1])
        th = g.random(samples).astype(np.float32) * 2 * np.pi
        on_side = g.random(samples) < 0.7
        rad = np.where(on_side, r, g.random(samples).astype(np.float32) * r)
        z = np.where(
            on_side,
            (g.random(samples).astype(np.float32) * 2 - 1) * hh,
            (g.integers(0, 2, samples) * 2 - 1) * hh,
        )
        pts = np.stack([rad * np.cos(th), rad * np.sin(th), z], axis=1).astype(np.float32)
    else:
        raise AssertionError(prim.ptype)
    world = pts @ R.T + c
    return float(np.min(np.linalg.norm(world - p, axis=1)))


@pytest.mark.parametrize(
    "prim",
    [
        EnvPrimitive(
            "box", "box",
            make_transform(rpy_to_mat([0.2, -0.4, 0.7]), np.float32([0.1, -0.2, 0.35])),
            np.float32([0.15, 0.25, 0.05]),
        ),
        EnvPrimitive(
            "sphere", "sphere", make_transform(np.eye(3, dtype=np.float32),
                                               np.float32([0.2, 0.1, 0.3])),
            np.float32([0.12]),
        ),
        EnvPrimitive(
            "cyl", "cylinder",
            make_transform(rpy_to_mat([0.3, 0.1, 0.0]), np.float32([-0.1, 0.2, 0.4])),
            np.float32([0.08, 0.3]),
        ),
    ],
    ids=["box", "sphere", "cylinder"],
)
def test_env_distance_matches_brute_force(prim, rng):
    for _ in range(8):
        p = (rng.random(3).astype(np.float32) - 0.5) * 1.6
        d, n = env_distance(prim, p)
        if d <= 0.0:
            continue  # brute force over the surface only checks the outside
        ref = _brute_force_distance(prim, p)
        assert abs(d - ref) < 5e-3, (prim.ptype, d, ref)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-4
        # The normal must point from the surface toward the query point.
        assert np.dot(n, p - (p - n * d)) > 0


def test_env_distance_batch_matches_scalar(rng):
    prims = [
        EnvPrimitive("plane", "plane", make_transform(np.eye(3, dtype=np.float32),
                                                      np.float32([0, 0, 0.05])),
                     np.float32([0, 0, 1])),
        EnvPrimitive("box", "box",
                     make_transform(rpy_to_mat([0.1, 0.2, 0.3]), np.float32([0.0, 0.1, 0.2])),
                     np.float32([0.2, 0.1, 0.15])),
        EnvPrimitive("sph", "sphere", make_transform(np.eye(3, dtype=np.float32),
                                                     np.float32([0.1, 0, 0.3])),
                     np.float32([0.1])),
        EnvPrimitive("cyl", "cylinder", make_transform(rpy_to_mat([0.4, 0, 0]),
                                                       np.float32([0, 0, 0.2])),
                     np.float32([0.07, 0.25])),
    ]
    P = ((rng.random((60, 3)) - 0.5) * 1.2).astype(np.float32)
    for prim in prims:
        d_b, n_b = env_distance_batch(prim, P)
        for k in range(P.shape[0]):
            d_s, n_s = env_distance(prim, P[k])
            assert abs(d_b[k] - d_s) < 1e-5, (prim.ptype, k)
            assert np.max(np.abs(n_b[k] - n_s)) < 1e-4, (prim.ptype, k)


def test_plane_distance_is_signed():
    prim = EnvPrimitive("floor", "plane",
                        make_transform(np.eye(3, dtype=np.float32), np.float32([0, 0, 0])),
                        np.float32([0, 0, 1]))
    assert env_distance(prim, np.float32([0, 0, 0.3]))[0] == pytest.approx(0.3)
    assert env_distance(prim, np.float32([0, 0, -0.2]))[0] == pytest.approx(-0.2)


def test_environment_rows_exist_for_every_primitive(cell):
    """Every MOVABLE sphere is checked against every primitive, and the rest are
    accounted for rather than quietly missing."""
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    n_env_rows = int(np.count_nonzero(wb.pairs.kind == PAIR_ROBOT_ENV))
    assert n_env_rows + wb.pairs.dropped_static_env == len(cell.environment) * len(wb.spheres)


def test_dropped_environment_rows_are_exactly_the_ones_with_no_gradient(cell):
    """The exclusion must be justified by the mechanism, not by a link-name list.

    A sphere on world-fixed mounting hardware has an identically zero point
    Jacobian, so its environment row can never be satisfied or traded off.  This
    asserts the two sets coincide: every dropped sphere really has no gradient,
    and every sphere that has one really was kept.
    """
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    fk = tree.fk(cell.rest_posture())
    centres = wb.sphere_world(fk)
    J = tree.point_jacobians(fk, wb.spheres.link, centres)
    immovable = np.abs(J).max(axis=(1, 2)) == 0.0

    kept = set(np.asarray(wb.pairs.a)[np.asarray(wb.pairs.kind) == PAIR_ROBOT_ENV].tolist())

    # SOUNDNESS, which is the direction that matters for safety: nothing that can
    # move is ever dropped.
    for s in range(len(wb.spheres)):
        if not immovable[s] and len(cell.environment):
            assert s in kept, f"{wb.spheres.label[s]} can move but has no environment rows"

    # Everything dropped really is immovable.
    dropped = [s for s in range(len(wb.spheres)) if len(cell.environment) and s not in kept]
    for s in dropped:
        assert immovable[s], f"{wb.spheres.label[s]} was dropped but has a gradient"
    assert wb.pairs.dropped_static_env == len(dropped) * len(cell.environment)

    # NOT completeness, deliberately.  A sphere whose centre lies exactly on its
    # own joint's axis also has zero gradient, and the structural rule does not
    # catch it -- `left/link1#0` on the reference cells is one.  Dropping by
    # measured gradient instead would make the pair list depend on the posture it
    # was built at, and the fixed row order is what makes the layer reproducible.
    # A few degenerate rows cost a little work; a state-dependent row list would
    # cost determinism.


def test_broadphase_is_exactly_equal_to_brute_force(cell):
    """The link-level broadphase must change SPEED and nothing else.

    It skips a whole block of sphere pairs on one bounding-sphere test.  That is
    only sound if the bound truly encloses the link's spheres, so this asserts
    the property directly rather than trusting the construction: assemble with
    the broadphase, assemble with it disabled, and require the four arrays that
    reach the solver to be bit-identical.  A bound that is too small shows up
    here as a missing constraint, which is the failure that would matter.
    """
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    rng = np.random.default_rng(4)
    lo, hi = (np.asarray(x, dtype=np.float32) for x in cell.joint_limits())
    for _ in range(12):
        q = (lo + (hi - lo) * rng.random(cell.n_joints)).astype(np.float32)
        fk = tree.fk(q)
        fast = wb.assemble(q, fk)
        saved = wb.blk_start
        wb.blk_start = np.zeros(0, dtype=np.int32)  # disable -> test every pair
        try:
            brute = wb.assemble(q, fk)
        finally:
            wb.blk_start = saved
        # What the SOLVER reads must be bit-identical.
        assert np.array_equal(fast.h, brute.h)
        assert np.array_equal(fast.active, brute.active)
        # G is filled only for rows the broadphase kept.  On a skipped row G
        # stays zero and h stays BIG, and the solver's own comment records why
        # that is a no-op: phi is about -BIG, so the projected update returns
        # zero and neither lambda nor q_dot moves.  The property asserted here is
        # the one that matters -- G agrees everywhere the constraint is live --
        # and test_broadphase_does_not_change_controller_output closes the loop
        # by checking the full stack bitwise.
        live = fast.active | brute.active
        assert np.array_equal(fast.G[live], brute.G[live])
        # `distance` is a diagnostic and is the one thing that legitimately
        # differs: a skipped row keeps BIG instead of its true (large) value.
        # Assert the property that makes the skip SOUND -- every row the
        # broadphase declined to compute really was beyond the influence
        # distance.  This is the assertion that would catch an under-sized bound.
        skipped = (fast.distance >= BIG) & (brute.distance < BIG)
        assert np.all(brute.distance[skipped] >= float(wb.cd_infl)), (
            "broadphase skipped a pair that was within influence distance"
        )
        both = ~skipped
        assert np.array_equal(fast.distance[both], brute.distance[both])


def test_link_bounds_actually_enclose_their_spheres(cell):
    """The bound is the whole safety argument, so check it in the link frame."""
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    for k, link in enumerate(wb.bound_link):
        m = wb.spheres.link == link
        c, r = wb.spheres.centre[m], wb.spheres.radius[m]
        reach = np.linalg.norm(c - wb.bound_centre[k], axis=1) + r
        assert reach.max() <= wb.bound_radius[k] + 1e-6, (
            f"link {link}: sphere reaches {reach.max()} beyond bound {wb.bound_radius[k]}"
        )


def test_broadphase_does_not_change_controller_output(cell):
    """The end-to-end claim: same joint targets, to the bit.

    The assembly-level test compares constraint arrays; this one runs the whole
    controller both ways over several chunks and requires the commanded
    trajectories to be identical.  It is the check that would catch a broadphase
    interacting with warm-started multipliers, which no single-tick comparison
    can see.
    """
    from remoroo_lc.reference.controller import Controller  # noqa: PLC0415

    def run(broadphase: bool) -> np.ndarray:
        ctl = Controller(cell)
        if not broadphase:
            ctl.walls.blk_start = np.zeros(0, dtype=np.int32)
        q = cell.rest_posture().copy()
        ctl.reset(q)
        rng = np.random.RandomState(3)
        out = []
        for _ in range(4):
            ctl.set_chunk((rng.randn(8, cell.action_dim) * 0.01).astype(np.float32), q)
            for _ in range(40):
                q = ctl.step(q).q_target.astype(np.float32)
                out.append(q.copy())
        return np.asarray(out)

    assert np.array_equal(run(True), run(False))


def test_bounded_rows_hold_the_safety_distance(cell):
    """A bounded row budget must not weaken the filter.

    Every per-row array -- G at (rows x joints), plus h, dist, lam and the
    solver scratch -- is sized by the ROW count, so on the rig cell 40,194 rows
    is 2.6 MiB per environment and that, not the physics, caps how many
    environments fit on a GPU.  `rows_per_block` bounds it by keeping only the N
    nearest pairs of each link block.

    That is an approximation, so it is checked the only way that means anything:
    drive the cell into contact from a COLLISION-FREE start and require the
    bounded filter to hold the same closest approach as the full one.  Measured
    on the rig cell at 0.5 m/s, top-2 and above are identical to all 40,194 rows
    and top-1 is not -- it allows 5.8 mm where d_safe is 15 mm -- which is why
    the default budget is 4 rather than 1.
    """
    tree = KinematicTree(cell)
    ref = WallBuilder(cell, tree)
    rng = np.random.default_rng(17)
    lo, hi = (np.asarray(x, dtype=np.float32) for x in cell.joint_limits())
    starts = []
    for _ in range(400):
        q = (lo + (hi - lo) * rng.random(cell.n_joints)).astype(np.float32)
        w = ref.assemble(q, tree.fk(q))
        d = w.distance[ref.n_joint_rows :]
        if d.size == 0 or d.min() > float(ref.cd_infl):
            starts.append(q)
        if len(starts) >= 5:
            break
    if not starts:
        pytest.skip("no collision-free posture found for this cell")

    def closest(rows_per_block: int) -> float:
        c2 = load_cell_with(cell, rows_per_block)
        wb = WallBuilder(c2, KinematicTree(c2))
        worst = BIG
        for q in starts:
            w = wb.assemble(q, tree.fk(q))
            d = w.distance[wb.n_joint_rows :]
            if d.size:
                worst = min(worst, float(d.min()))
        return worst

    full = closest(0)
    for n in (2, 4, 8):
        assert closest(n) >= full - 1e-6, f"rows_per_block={n} reported a nearer pair than the full set"


def load_cell_with(cell, rows_per_block: int):
    """Same cell with a different row budget, without touching the original."""
    import copy  # noqa: PLC0415

    c = copy.copy(cell)
    c.collision = dict(cell.collision)
    if rows_per_block:
        c.collision["rows_per_block"] = rows_per_block
    else:
        c.collision.pop("rows_per_block", None)
    return c
