"""Layer 3a: generic constraint assembly.

Two families of rows, both of them velocity dampers of the same shape: the
allowed approach speed toward a wall falls linearly to zero as the distance to it
falls from d_infl to d_safe, and goes negative (i.e. commands retreat) inside
d_safe.

    joint limit:  q_dot_j            <=  xi (q_max_j - q_j - d_safe) / (d_infl - d_safe)
    collision:   -n^T (J_a - J_b) q_dot <= xi (d       - d_safe) / (d_infl - d_safe)

The pair list is built once at load and never changes: which sphere pairs are
checked, and in what order, is a property of the cell, not of the state.  Rows
whose distance is beyond d_infl keep their slot with h = +BIG rather than being
dropped, so the constraint matrix has the same shape and the same row meaning on
every tick, on every device, in every environment of a batch.  That is what makes
a warm-started Gauss-Seidel meaningful and the whole layer bitwise reproducible.

Nothing here knows what the spheres are attached to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from remoroo_lc.constants import BIG, DTYPE, EPS, POINT_DIM
from remoroo_lc.reference.kinematics import FkResult, KinematicTree
from remoroo_lc.schema import CellSpec, EnvObstacle, refuse_mesh_obstacles

PAIR_ROBOT_ROBOT = 0
PAIR_ROBOT_ENV = 1


@dataclass
class SphereSet:
    """Flattened collision spheres across every model in the cell."""

    link: np.ndarray  # (S,) global link id
    centre: np.ndarray  # (S, 3) in the link frame
    radius: np.ndarray  # (S,)
    label: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.link.shape[0])


@dataclass
class PairList:
    """Fixed collision-pair list, resolved at load."""

    kind: np.ndarray  # (P,) PAIR_*
    a: np.ndarray  # (P,) sphere index
    b: np.ndarray  # (P,) sphere index, or environment index for PAIR_ROBOT_ENV
    label: list[str] = field(default_factory=list)
    #: Environment rows dropped because the sphere sits on world-fixed mounting
    #: hardware.  Reported rather than silently swallowed: it is the difference
    #: between "nothing to check" and "we forgot to check".
    dropped_static_env: int = 0

    def __len__(self) -> int:
        return int(self.kind.shape[0])


def build_sphere_set(cell: CellSpec, tree: KinematicTree) -> SphereSet:
    """Flatten every model's sphere file into one indexed set, in config order."""
    link, centre, radius, label = [], [], [], []
    for m in cell.models:
        per_link = cell.spheres.get(m.name, {})
        for link_name in sorted(per_link):
            for k, (c, r) in enumerate(per_link[link_name]):
                link.append(tree.link_id(m.name, link_name))
                centre.append(c)
                radius.append(r)
                label.append(f"{m.name}/{link_name}#{k}")
    if not link:
        return SphereSet(
            np.zeros(0, dtype=np.int32),
            np.zeros((0, POINT_DIM), dtype=DTYPE),
            np.zeros(0, dtype=DTYPE),
            [],
        )
    return SphereSet(
        np.asarray(link, dtype=np.int32),
        np.stack(centre).astype(DTYPE),
        np.asarray(radius, dtype=DTYPE),
        label,
    )


def _rigid_groups(cell: CellSpec, model_name: str) -> dict[str, int]:
    """Group links that cannot move relative to each other into one rigid body.

    Links in the same rigid body can never collide with each other, and neither
    can two rigid bodies joined by a single actuated joint -- the URDF adjacency
    the brief refers to.  Deriving this from the joint graph rather than a
    hand-written ignore list is what keeps it correct on a robot nobody here has
    seen.

    A LOCKED joint is rigid too.  It is declared in the cell file as held at a
    fixed value and is never actuated, so the links it joins are as welded
    together as a fixed joint's are.  Treating it as articulated instead leaves
    an effector's own internal DOF -- six of them on a parallel jaw, each joining
    two links that are bolted to the same casting -- generating collision pairs
    against each other and against the wrist they are mounted on, none of which
    can ever happen.  On the reference bimanual cell that alone is the difference
    between 25 rigid bodies and 13.
    """
    urdf = cell.urdfs[model_name]
    model = next(m for m in cell.models if m.name == model_name)
    parent = {ln: ln for ln in urdf.links}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j in urdf.joints:
        if not j.moves or j.name in model.locked:
            ra, rb = find(j.parent), find(j.child)
            if ra != rb:
                parent[rb] = ra
    roots = sorted({find(ln) for ln in urdf.links})
    rid = {r: i for i, r in enumerate(roots)}
    return {ln: rid[find(ln)] for ln in urdf.links}


def _world_rigid_links(cell: CellSpec, model_name: str, groups: dict[str, int]) -> set[str]:
    """Links no actuated joint can move: the root's own rigid group.

    A model's root is placed by `models[].base` and never moves, so every link
    welded to it through fixed and locked joints is world-fixed too.  The point
    Jacobian of a sphere on such a link is identically zero, which makes its
    distance to an environment primitive a CONSTANT.

    That matters because it is exactly the mounting hardware: the plinth the arm
    is bolted to, a camera on the overhead beam, a rail the base is clamped to.
    Those overlap the obstacle they are mounted on BY CONSTRUCTION, and a
    constraint row for one is degenerate -- `G` is all zeros, so no joint motion
    changes it, and it can be neither satisfied nor traded off.  It cannot warn
    anybody either: the geometry is static, so it says the same thing forever.
    Left in, it reports a permanent collision and every diagnostic that counts
    active rows reads 100%, hiding the ones that are real.
    """
    urdf = cell.urdfs[model_name]
    children = {j.child for j in urdf.joints}
    roots = [ln for ln in urdf.links if ln not in children]
    if not roots:  # a cycle; not a tree, and not this function's problem
        return set()
    rg = groups[roots[0]]
    return {ln for ln in urdf.links if groups[ln] == rg}


def _adjacent_groups(cell: CellSpec, model_name: str, groups: dict[str, int]) -> set:
    urdf = cell.urdfs[model_name]
    adj = set()
    for j in urdf.joints:
        ga, gb = groups[j.parent], groups[j.child]
        if ga != gb:
            adj.add((min(ga, gb), max(ga, gb)))
    return adj


def link_bounds(spheres: SphereSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One bounding sphere per link, in that LINK's own frame.

    Computed once at load and never again: a link's spheres are rigid in its
    frame, so the bound's radius is invariant and only its centre needs the
    link transform at runtime.  Returns (link_ids, centres, radii).
    """
    ids = np.unique(spheres.link) if len(spheres) else np.zeros(0, dtype=np.int32)
    cen = np.zeros((ids.size, POINT_DIM), dtype=DTYPE)
    rad = np.zeros(ids.size, dtype=DTYPE)
    for k, link in enumerate(ids):
        m = spheres.link == link
        c, r = spheres.centre[m], spheres.radius[m]
        lo = (c - r[:, None]).min(axis=0)
        hi = (c + r[:, None]).max(axis=0)
        mid = ((lo + hi) * DTYPE(0.5)).astype(DTYPE)
        cen[k] = mid
        rad[k] = DTYPE(np.max(np.linalg.norm(c - mid, axis=1) + r))
    return ids.astype(np.int32), cen, rad


def group_pairs_by_link(
    pairs: PairList, spheres: SphereSet
) -> tuple[PairList, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort robot-robot pairs into contiguous per-link-pair BLOCKS.

    This is the data structure a broadphase needs: if two links' bounding
    spheres are further apart than the influence distance, then every sphere
    pair between them is too, and the whole block can be skipped with ONE test
    instead of hundreds.  Measured on the rig cell: 38,532 sphere pairs fall
    into 401 link pairs (96 per block on average), and along the teach recording
    only ~15 of those 401 blocks are ever close.

    The sort is a permutation of the pair list applied once at load, so the row
    order stays fixed for the lifetime of the cell -- which is what the
    determinism argument needs.  It is NOT the same order as before, and that
    matters: the Gauss-Seidel sweep in Layer 3 reads rows in order, so a
    different fixed order is a different (equally valid) fixed point.

    Returns the reordered PairList plus (block_link_a, block_link_b,
    block_start, block_count) covering only the robot-robot rows, which are
    placed first.
    """
    kind = pairs.kind
    rr = np.where(kind == PAIR_ROBOT_ROBOT)[0]
    env = np.where(kind != PAIR_ROBOT_ROBOT)[0]
    if rr.size == 0:
        return pairs, *(np.zeros(0, dtype=np.int32) for _ in range(4))

    la = spheres.link[pairs.a[rr]].astype(np.int64)
    lb = spheres.link[pairs.b[rr]].astype(np.int64)
    lo_, hi_ = np.minimum(la, lb), np.maximum(la, lb)
    key = lo_ * (int(spheres.link.max()) + 1) + hi_
    # Stable sort so that, within a block, the original relative order survives.
    order = rr[np.argsort(key, kind="stable")]
    skey = key[np.argsort(key, kind="stable")]
    uniq, start, count = np.unique(skey, return_index=True, return_counts=True)
    perm = np.concatenate([order, env])

    reordered = PairList(
        pairs.kind[perm],
        pairs.a[perm],
        pairs.b[perm],
        [pairs.label[i] for i in perm],
        pairs.dropped_static_env,
    )
    blk_a = (uniq // (int(spheres.link.max()) + 1)).astype(np.int32)
    blk_b = (uniq % (int(spheres.link.max()) + 1)).astype(np.int32)
    return reordered, blk_a, blk_b, start.astype(np.int32), count.astype(np.int32)


def build_pair_list(cell: CellSpec, tree: KinematicTree, spheres: SphereSet) -> PairList:
    """Resolve the fixed collision-pair list for this cell."""
    # ⚠ Before a single pair is decided, because the failure this prevents is
    # SILENT rather than loud.  corner_cell (Siemens rig, measured 2026-09-23)
    # declares one mesh obstacle and no `spheres:` file at all, so the
    # environment loop below emits zero rows whether or not the obstacle is
    # something this package can collide with -- the caller gets a pair list that
    # looks complete and an environment that was never checked.  Refusing here
    # also covers the two loops downstream that consume this list and cannot be
    # reached without it: `WallBuilder.__init__`'s `_env_rows` partition and
    # `WallBuilder.assemble`'s per-primitive row block (which additionally hits
    # `env_distance_batch`'s own refusal).  No gate is placed in those two: a gate
    # that cannot fire is not a gate.
    refuse_mesh_obstacles(cell.environment, consumer="reference.walls.build_pair_list")
    n_s = len(spheres)
    model_of: list[str] = []
    link_of: list[str] = []
    for m in cell.models:
        per_link = cell.spheres.get(m.name, {})
        for link_name in sorted(per_link):
            for _ in per_link[link_name]:
                model_of.append(m.name)
                link_of.append(link_name)

    groups = {m.name: _rigid_groups(cell, m.name) for m in cell.models}
    adjacent = {m.name: _adjacent_groups(cell, m.name, groups[m.name]) for m in cell.models}
    ignore = {tuple(sorted(p)) for p in cell.collision.get("ignore_pairs", [])}

    kind, ia, ib, label = [], [], [], []
    if cell.collision.get("self_pairs", True) or cell.collision.get("cross_model", True):
        for i in range(n_s):
            for j in range(i + 1, n_s):
                same_model = model_of[i] == model_of[j]
                if same_model:
                    if not cell.collision.get("self_pairs", True):
                        continue
                    gi = groups[model_of[i]][link_of[i]]
                    gj = groups[model_of[j]][link_of[j]]
                    if gi == gj:
                        continue
                    if (min(gi, gj), max(gi, gj)) in adjacent[model_of[i]]:
                        continue
                else:
                    if not cell.collision.get("cross_model", True):
                        continue
                if tuple(sorted((link_of[i], link_of[j]))) in ignore:
                    continue
                kind.append(PAIR_ROBOT_ROBOT)
                ia.append(i)
                ib.append(j)
                label.append(f"{spheres.label[i]} | {spheres.label[j]}")
    dropped_static = 0
    if cell.collision.get("environment", True):
        # Mounting hardware gets no environment rows.  See _world_rigid_links:
        # the sphere cannot move, so the distance is a constant and `G` is zero.
        static = {m.name: _world_rigid_links(cell, m.name, groups[m.name]) for m in cell.models}
        for e, prim in enumerate(cell.environment):
            for i in range(n_s):
                if link_of[i] in static[model_of[i]]:
                    dropped_static += 1
                    continue
                kind.append(PAIR_ROBOT_ENV)
                ia.append(i)
                ib.append(e)
                label.append(f"{spheres.label[i]} | env:{prim.name}")

    max_pairs = cell.collision.get("max_pairs")
    if max_pairs is not None and len(kind) > int(max_pairs):
        raise ValueError(
            f"{cell.name}: {len(kind)} collision pairs exceeds max_pairs={max_pairs}; "
            "prune the sphere set or widen the cap -- silently truncating would "
            "make the safety layer lie about what it is checking"
        )
    if not kind:
        return PairList(
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            [],
            dropped_static,
        )
    return PairList(
        np.asarray(kind, dtype=np.int32),
        np.asarray(ia, dtype=np.int32),
        np.asarray(ib, dtype=np.int32),
        label,
        dropped_static,
    )


# --------------------------------------------------------------------------- #
# environment distance functions
# --------------------------------------------------------------------------- #


def env_distance_batch(prim: EnvObstacle, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `env_distance` over (S, 3) points.  Returns (d (S,), n (S, 3))."""
    refuse_mesh_obstacles((prim,), consumer="reference.walls.env_distance_batch")
    n_pts = P.shape[0]
    if n_pts == 0:
        return np.zeros(0, dtype=DTYPE), np.zeros((0, POINT_DIM), dtype=DTYPE)
    if prim.ptype == "plane":
        nrm = prim.dims
        d = (P - prim.pose[:POINT_DIM, POINT_DIM]) @ nrm
        return d.astype(DTYPE), np.broadcast_to(nrm, (n_pts, POINT_DIM)).astype(DTYPE)
    R = prim.pose[:POINT_DIM, :POINT_DIM]
    c = prim.pose[:POINT_DIM, POINT_DIM]
    local = (P - c) @ R  # == (R^T (p - c))^T row-wise
    if prim.ptype == "sphere":
        r = DTYPE(prim.dims[0])
        dist = np.linalg.norm(local, axis=1).astype(DTYPE)
        nl = local / np.maximum(dist, EPS)[:, None]
        return (dist - r).astype(DTYPE), (nl @ R.T).astype(DTYPE)
    if prim.ptype == "box":
        half = prim.dims
        closest = np.clip(local, -half, half).astype(DTYPE)
        diff = (local - closest).astype(DTYPE)
        dist = np.linalg.norm(diff, axis=1).astype(DTYPE)
        outside = dist > EPS
        nl = np.zeros_like(local)
        nl[outside] = diff[outside] / dist[outside, None]
        if (~outside).any():
            gap = (half - np.abs(local[~outside])).astype(DTYPE)
            k = np.argmin(gap, axis=1)
            sub = np.zeros((int(np.count_nonzero(~outside)), POINT_DIM), dtype=DTYPE)
            rows = np.arange(sub.shape[0])
            sgn = np.sign(local[~outside][rows, k]).astype(DTYPE)
            sgn[sgn == 0] = DTYPE(1.0)
            sub[rows, k] = sgn
            nl[~outside] = sub
            dist = dist.copy()
            dist[~outside] = -gap[rows, k]
        return dist.astype(DTYPE), (nl @ R.T).astype(DTYPE)
    if prim.ptype == "cylinder":
        r, half_h = DTYPE(prim.dims[0]), DTYPE(prim.dims[1])
        radial = local[:, :2]
        rho = np.linalg.norm(radial, axis=1).astype(DTYPE)
        u = radial / np.maximum(rho, EPS)[:, None]
        d_rad = (rho - r).astype(DTYPE)
        sgn_z = np.sign(local[:, 2]).astype(DTYPE)
        sgn_z[sgn_z == 0] = DTYPE(1.0)
        d_ax = (np.abs(local[:, 2]) - half_h).astype(DTYPE)
        rim = (d_rad > 0.0) & (d_ax > 0.0)
        side = (~rim) & (d_rad > d_ax)
        cap = (~rim) & (~side)
        d = np.zeros(n_pts, dtype=DTYPE)
        nl = np.zeros((n_pts, POINT_DIM), dtype=DTYPE)
        if rim.any():
            dd = np.sqrt(d_rad[rim] ** 2 + d_ax[rim] ** 2).astype(DTYPE)
            d[rim] = dd
            nl[rim] = (
                np.stack(
                    [u[rim, 0] * d_rad[rim], u[rim, 1] * d_rad[rim], sgn_z[rim] * d_ax[rim]],
                    axis=1,
                )
                / np.maximum(dd, EPS)[:, None]
            )
        if side.any():
            d[side] = d_rad[side]
            nl[side] = np.stack(
                [u[side, 0], u[side, 1], np.zeros(int(side.sum()), dtype=DTYPE)], axis=1
            )
        if cap.any():
            d[cap] = d_ax[cap]
            z = np.zeros((int(cap.sum()), POINT_DIM), dtype=DTYPE)
            z[:, 2] = sgn_z[cap]
            nl[cap] = z
        return d, (nl @ R.T).astype(DTYPE)
    raise ValueError(f"unknown primitive type {prim.ptype!r}")


def env_distance(prim: EnvObstacle, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed distance from a point to a primitive's surface, and the outward
    unit normal at the closest point (pointing from the primitive to the point).

    A mesh obstacle is refused here by name rather than falling through to the
    `unknown primitive type` raise at the bottom: "unknown type 'mesh'" reads as a
    typo in a cell file, when what actually happened is that a caller asked this
    layer for a capability it does not have.
    """
    refuse_mesh_obstacles((prim,), consumer="reference.walls.env_distance")
    if prim.ptype == "plane":
        n = prim.dims  # normal was stashed in dims by the loader
        d = float(np.dot(n, p - prim.pose[:POINT_DIM, POINT_DIM]))
        return d, n.astype(DTYPE)
    R = prim.pose[:POINT_DIM, :POINT_DIM]
    c = prim.pose[:POINT_DIM, POINT_DIM]
    local = R.T @ (p - c)
    if prim.ptype == "sphere":
        r = float(prim.dims[0])
        dist = float(np.linalg.norm(local))
        n_local = local / DTYPE(max(dist, float(EPS)))
        return dist - r, (R @ n_local).astype(DTYPE)
    if prim.ptype == "box":
        half = prim.dims
        closest = np.clip(local, -half, half).astype(DTYPE)
        diff = local - closest
        dist = float(np.linalg.norm(diff))
        if dist > float(EPS):
            n_local = diff / DTYPE(dist)
            return dist, (R @ n_local).astype(DTYPE)
        # Inside: nearest face determines both sign and direction.
        gap = half - np.abs(local)
        k = int(np.argmin(gap))
        n_local = np.zeros(POINT_DIM, dtype=DTYPE)
        n_local[k] = DTYPE(np.sign(local[k]) if local[k] != 0 else 1.0)
        return -float(gap[k]), (R @ n_local).astype(DTYPE)
    if prim.ptype == "cylinder":
        r, half_h = float(prim.dims[0]), float(prim.dims[1])
        radial = np.asarray([local[0], local[1]], dtype=DTYPE)
        rho = float(np.linalg.norm(radial))
        d_rad = rho - r
        d_ax = abs(float(local[2])) - half_h
        if d_rad > 0.0 and d_ax > 0.0:  # outside the rim
            dist = float(np.sqrt(d_rad * d_rad + d_ax * d_ax))
            u = radial / DTYPE(max(rho, float(EPS)))
            n_local = np.asarray(
                [u[0] * d_rad, u[1] * d_rad, np.sign(local[2]) * d_ax], dtype=DTYPE
            ) / DTYPE(max(dist, float(EPS)))
            return dist, (R @ n_local).astype(DTYPE)
        if d_rad > d_ax:  # nearest the curved side
            u = radial / DTYPE(max(rho, float(EPS)))
            n_local = np.asarray([u[0], u[1], 0.0], dtype=DTYPE)
            return d_rad, (R @ n_local).astype(DTYPE)
        n_local = np.asarray([0.0, 0.0, np.sign(local[2]) or 1.0], dtype=DTYPE)
        return d_ax, (R @ n_local).astype(DTYPE)
    raise ValueError(f"unknown primitive type {prim.ptype!r}")


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


@dataclass
class Walls:
    """Assembled constraint rows plus the distances that produced them."""

    G: np.ndarray  # (m, n)
    h: np.ndarray  # (m,)
    active: np.ndarray  # (m,) bool
    distance: np.ndarray  # (m,) metres for collision rows, radians for joint rows
    n_joint_rows: int


class WallBuilder:
    """Builds the fixed-shape constraint set for a cell, tick by tick."""

    def __init__(self, cell: CellSpec, tree: KinematicTree) -> None:
        self.cell = cell
        self.tree = tree
        self.spheres = build_sphere_set(cell, tree)
        self.pairs = build_pair_list(cell, tree, self.spheres)
        # Link-level broadphase.  See group_pairs_by_link: pairs are reordered
        # into contiguous per-link-pair blocks so one bounding-sphere test can
        # retire a whole block.  Zero fidelity loss -- a skipped block is one
        # whose every sphere pair is provably beyond the influence distance --
        # and it is what keeps the wall check from being O(all pairs) forever.
        self.broadphase = bool(cell.collision.get("broadphase", True))
        (
            self.pairs,
            self.blk_link_a,
            self.blk_link_b,
            self.blk_start,
            self.blk_count,
        ) = group_pairs_by_link(self.pairs, self.spheres)
        self.bound_link, self.bound_centre, self.bound_radius = link_bounds(self.spheres)
        #: Rows emitted per block.  0 keeps one row per sphere pair (the original
        #: form).  N > 0 emits the N NEAREST pairs of each block instead, which
        #: bounds the row count at n_blocks * N regardless of how crowded the
        #: cell gets.  Rows are what every per-row array is sized by -- G, h,
        #: dist, lam, the solver scratch -- so the row count, not the pair count,
        #: is what decides how many environments fit on a GPU.
        self.rows_per_block = int(cell.collision.get("rows_per_block", 0))
        if not self.broadphase:
            # Keep the reordering (row order is a cell property either way) but
            # drop the block table so every pair is tested.  Measured: the
            # broadphase is a clear win on CPU and a LOSS on CUDA, where one
            # thread per environment means neighbouring threads keep different
            # blocks live and the warp executes the union of their branches.
            self.blk_start = np.zeros(0, dtype=np.int32)
            self.blk_count = np.zeros(0, dtype=np.int32)
            self.blk_link_a = np.zeros(0, dtype=np.int32)
            self.blk_link_b = np.zeros(0, dtype=np.int32)
        slot = {int(L): k for k, L in enumerate(self.bound_link)}
        self._blk_ia = np.asarray([slot[int(x)] for x in self.blk_link_a], dtype=np.int32)
        self._blk_ib = np.asarray([slot[int(x)] for x in self.blk_link_b], dtype=np.int32)
        # Row indices per block, precomputed: the broadphase gathers these, so the
        # per-tick cost is a gather rather than an arange-and-concatenate.
        self._pair_block_np = np.full(len(self.pairs), -1, dtype=np.int64)
        for _k, (_a, _c) in enumerate(zip(self.blk_start, self.blk_count)):
            self._pair_block_np[int(_a) : int(_a) + int(_c)] = _k
        #: Blocks that had more pairs within influence than `rows_per_block`.
        #: Non-zero means the bound is biting and should be raised.
        self.saturated_blocks = 0
        self._blk_rows_all = [
            np.arange(int(a), int(a) + int(c), dtype=np.int64)
            for a, c in zip(self.blk_start, self.blk_count)
        ]
        jl = cell.limits["joint_limit_damper"]
        cd = cell.limits["collision_damper"]
        self.jl_xi = DTYPE(jl["xi"])
        self.jl_safe = DTYPE(np.radians(jl["d_safe_deg"]))
        self.jl_infl = DTYPE(np.radians(jl["d_infl_deg"]))
        self.cd_xi = DTYPE(cd["xi"])
        self.cd_safe = DTYPE(cd["d_safe_m"])
        self.cd_infl = DTYPE(cd["d_infl_m"])
        self.q_lo, self.q_hi = cell.joint_limits()
        self.n = cell.n_joints
        self.n_joint_rows = 2 * self.n
        if self.rows_per_block > 0 and self.blk_start.size:
            self._blk_row0 = (
                self.n_joint_rows
                + np.arange(self.blk_start.size, dtype=np.int64) * self.rows_per_block
            )
            self.n_rows = self.n_joint_rows + int(self.blk_start.size) * self.rows_per_block
            # Env pairs keep one row each: there are few of them and they are
            # already grouped per primitive.
            self._env_row0 = self.n_rows
            self.n_rows += int(np.count_nonzero(self.pairs.kind != PAIR_ROBOT_ROBOT))
        else:
            self.rows_per_block = 0
            self.n_rows = self.n_joint_rows + len(self.pairs)

        # Row-index partition, fixed at load: which rows are robot-robot pairs and
        # which belong to each environment primitive.  Order within each group is
        # the pair list's order, so the overall row order is still exactly the
        # order build_pair_list produced.
        kind = self.pairs.kind
        self._rr = np.where(kind == PAIR_ROBOT_ROBOT)[0]
        self._rr_a = self.pairs.a[self._rr]
        self._rr_b = self.pairs.b[self._rr]
        self._env_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        for e in range(len(cell.environment)):
            sel = np.where((kind == PAIR_ROBOT_ENV) & (self.pairs.b == e))[0]
            if sel.size:
                self._env_rows.append((e, sel, self.pairs.a[sel]))

    def _assemble_topn(self, keep, centres, J_s, G, h, active, distance, base, live_blk):
        """Emit the N NEAREST pairs of each live block instead of all of them.

        Why a bound is needed at all: every per-row array -- G at
        (rows x joints), plus h, dist, lam and the solver scratch -- is sized by
        the ROW count, so 40,194 rows is 2.6 MiB per environment and that, not
        the physics, is what caps how many environments fit on a GPU.  Pruning
        which rows are *computed* does not help; the storage is allocated either
        way.

        Why the N nearest: within one link pair the binding constraint is the
        closest approach, and the next nearest are the ones that could become
        binding within a tick.  Keeping several rather than one matters because
        two links can touch at more than one place -- measured up to 324
        simultaneously active pairs inside a single block -- and constraining
        only the closest leaves the pair free to rotate about it.

        Selection is a stable partition by distance with ties broken by pair
        index, so it is a deterministic function of the state like everything
        else here.  A block that has more than N pairs within influence is
        SATURATED; the count is reported rather than swallowed, because that is
        the condition under which this is an approximation rather than an
        identity.
        """
        n = self.rows_per_block
        ka, kb = self.pairs.a[keep], self.pairs.b[keep]
        diff = (centres[ka] - centres[kb]).astype(DTYPE)
        dist = np.linalg.norm(diff, axis=1).astype(DTYPE)
        d = (dist - self.spheres.radius[ka] - self.spheres.radius[kb]).astype(DTYPE)
        blk_of = self._pair_block_np[keep]
        cspan = self.cd_infl - self.cd_safe
        for b_i in live_blk:
            m = np.where(blk_of == b_i)[0]
            if m.size == 0:
                continue
            dm = d[m]
            take = m[np.argsort(dm, kind="stable")[:n]]
            if int(np.count_nonzero(d[m] < self.cd_infl)) > n:
                self.saturated_blocks += 1
            row0 = self.n_joint_rows + int(b_i) * n
            for slot, idx in enumerate(take):
                if d[idx] >= self.cd_infl:
                    break
                row = row0 + slot
                a_i, b_s = int(ka[idx]), int(kb[idx])
                nrm = diff[idx] / (dist[idx] + EPS)
                G[row] = -(nrm @ (J_s[a_i] - J_s[b_s]))
                distance[row] = d[idx]
                h[row] = self.cd_xi * (d[idx] - self.cd_safe) / cspan
                active[row] = True

    def _blk_rows(self, live_blk: np.ndarray) -> np.ndarray:
        if live_blk.size == self.blk_start.size:
            return np.arange(self._rr.size, dtype=np.int64)
        return np.concatenate([self._blk_rows_all[int(k)] for k in live_blk])

    # ------------------------------------------------------------------ #
    def row_labels(self) -> list[str]:
        labels = [f"jmax:{lb}" for lb in self.cell.joint_labels()]
        labels += [f"jmin:{lb}" for lb in self.cell.joint_labels()]
        labels += [f"pair:{lb}" for lb in self.pairs.label]
        return labels

    def sphere_world(self, fk: FkResult) -> np.ndarray:
        n_s = len(self.spheres)
        if n_s == 0:
            return np.zeros((0, POINT_DIM), dtype=DTYPE)
        T = fk.link_T[self.spheres.link]
        return (
            np.einsum("sij,sj->si", T[:, :POINT_DIM, :POINT_DIM], self.spheres.centre)
            + T[:, :POINT_DIM, POINT_DIM]
        ).astype(DTYPE)

    def assemble(self, q: np.ndarray, fk: FkResult) -> Walls:
        n = self.n
        m = self.n_rows
        G = np.zeros((m, n), dtype=DTYPE)
        h = np.full(m, BIG, dtype=DTYPE)
        active = np.zeros(m, dtype=bool)
        distance = np.full(m, BIG, dtype=DTYPE)

        # --- joint position limit dampers (rows 0 .. 2n) ------------------- #
        span = self.jl_infl - self.jl_safe
        idx = np.arange(n)
        G[idx, idx] = DTYPE(1.0)
        G[n + idx, idx] = DTYPE(-1.0)
        d_up = (self.q_hi - q).astype(DTYPE)
        d_lo = (q - self.q_lo).astype(DTYPE)
        distance[:n] = d_up
        distance[n : 2 * n] = d_lo
        up_on = d_up < self.jl_infl
        lo_on = d_lo < self.jl_infl
        h[:n][up_on] = self.jl_xi * (d_up[up_on] - self.jl_safe) / span
        h[n : 2 * n][lo_on] = self.jl_xi * (d_lo[lo_on] - self.jl_safe) / span
        active[:n] = up_on
        active[n : 2 * n] = lo_on

        # --- collision dampers -------------------------------------------- #
        if len(self.pairs):
            centres = self.sphere_world(fk)
            J_s = self.tree.point_jacobians(fk, self.spheres.link, centres)
            cspan = self.cd_infl - self.cd_safe
            base = self.n_joint_rows

            if self._rr.size:
                # BROADPHASE.  One bounding-sphere test retires a whole link
                # block.  A block is skipped only when its two link bounds are
                # further apart than cd_infl, and every sphere inside a bound is
                # by construction within it -- so a skipped pair is provably
                # beyond influence and its row keeps the h = BIG it was
                # initialised with.  Speed only; tests/test_walls.py asserts the
                # assembled (G, h, active, distance) are IDENTICAL either way.
                keep = self._rr
                if self.blk_start.size:
                    Tb = fk.link_T[self.bound_link]
                    bc = (
                        np.einsum("kij,kj->ki", Tb[:, :POINT_DIM, :POINT_DIM], self.bound_centre)
                        + Tb[:, :POINT_DIM, POINT_DIM]
                    )
                    gap = (
                        np.linalg.norm(bc[self._blk_ia] - bc[self._blk_ib], axis=1)
                        - self.bound_radius[self._blk_ia]
                        - self.bound_radius[self._blk_ib]
                    )
                    live_blk = np.where(gap < self.cd_infl)[0]
                    keep = (
                        self._rr[self._blk_rows(live_blk)]
                        if live_blk.size
                        else self._rr[:0]
                    )
                if keep.size and self.rows_per_block > 0:
                    self._assemble_topn(
                        keep, centres, J_s, G, h, active, distance, base, live_blk
                    )
                elif keep.size:
                    ka = self.pairs.a[keep]
                    kb = self.pairs.b[keep]
                    diff = (centres[ka] - centres[kb]).astype(DTYPE)
                    dist = np.linalg.norm(diff, axis=1).astype(DTYPE)
                    normal = diff / (dist + EPS)[:, None]
                    d = (
                        dist - self.spheres.radius[ka] - self.spheres.radius[kb]
                    ).astype(DTYPE)
                    Jrel = J_s[ka] - J_s[kb]
                    rows = base + keep
                    G[rows] = -np.einsum("pi,pij->pj", normal, Jrel)
                    distance[rows] = d
                    on = d < self.cd_infl
                    sel = rows[on]
                    h[sel] = self.cd_xi * (d[on] - self.cd_safe) / cspan
                    active[sel] = True

            env_slot = 0
            for e, sel, sph in self._env_rows:
                prim = self.cell.environment[e]
                sd, normal = env_distance_batch(prim, centres[sph])
                d = (sd - self.spheres.radius[sph]).astype(DTYPE)
                if self.rows_per_block > 0:
                    rows = self._env_row0 + env_slot + np.arange(sel.size)
                    env_slot += sel.size
                else:
                    rows = base + sel
                G[rows] = -np.einsum("pi,pij->pj", normal, J_s[sph])
                distance[rows] = d
                on = d < self.cd_infl
                act = rows[on]
                h[act] = self.cd_xi * (d[on] - self.cd_safe) / cspan
                active[act] = True

        return Walls(G, h, active, distance, self.n_joint_rows)
