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
from remoroo_lc.schema import CellSpec, EnvPrimitive

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


def _adjacent_groups(cell: CellSpec, model_name: str, groups: dict[str, int]) -> set:
    urdf = cell.urdfs[model_name]
    adj = set()
    for j in urdf.joints:
        ga, gb = groups[j.parent], groups[j.child]
        if ga != gb:
            adj.add((min(ga, gb), max(ga, gb)))
    return adj


def build_pair_list(cell: CellSpec, tree: KinematicTree, spheres: SphereSet) -> PairList:
    """Resolve the fixed collision-pair list for this cell."""
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
    if cell.collision.get("environment", True):
        for e, prim in enumerate(cell.environment):
            for i in range(n_s):
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
        )
    return PairList(
        np.asarray(kind, dtype=np.int32),
        np.asarray(ia, dtype=np.int32),
        np.asarray(ib, dtype=np.int32),
        label,
    )


# --------------------------------------------------------------------------- #
# environment distance functions
# --------------------------------------------------------------------------- #


def env_distance_batch(prim: EnvPrimitive, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `env_distance` over (S, 3) points.  Returns (d (S,), n (S, 3))."""
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


def env_distance(prim: EnvPrimitive, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed distance from a point to a primitive's surface, and the outward
    unit normal at the closest point (pointing from the primitive to the point)."""
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
                diff = (centres[self._rr_a] - centres[self._rr_b]).astype(DTYPE)
                dist = np.linalg.norm(diff, axis=1).astype(DTYPE)
                normal = diff / (dist + EPS)[:, None]
                d = (
                    dist - self.spheres.radius[self._rr_a] - self.spheres.radius[self._rr_b]
                ).astype(DTYPE)
                Jrel = J_s[self._rr_a] - J_s[self._rr_b]
                rows = base + self._rr
                G[rows] = -np.einsum("pi,pij->pj", normal, Jrel)
                distance[rows] = d
                on = d < self.cd_infl
                sel = rows[on]
                h[sel] = self.cd_xi * (d[on] - self.cd_safe) / cspan
                active[sel] = True

            for e, sel, sph in self._env_rows:
                prim = self.cell.environment[e]
                sd, normal = env_distance_batch(prim, centres[sph])
                d = (sd - self.spheres.radius[sph]).astype(DTYPE)
                rows = base + sel
                G[rows] = -np.einsum("pi,pij->pj", normal, J_s[sph])
                distance[rows] = d
                on = d < self.cd_infl
                act = rows[on]
                h[act] = self.cd_xi * (d[on] - self.cd_safe) / cspan
                active[act] = True

        return Walls(G, h, active, distance, self.n_joint_rows)
