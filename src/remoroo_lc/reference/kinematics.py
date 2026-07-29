"""Generic fixed-base tree kinematics for a whole cell.

One KinematicTree covers every model in the cell at once, with a single global
joint vector q in R^n formed by concatenating each model's actuated joints in
config order.  Task Jacobians are always n columns wide; columns belonging to
joints that are not on a frame's path are exactly zero.  That is what lets Layer 2
stack every TCP into one damped least-squares solve and get correct arbitration
of shared joints for free, with no special case for the branched topology.

There is no notion here of a chain "belonging" to a TCP beyond which joints
happen to lie on its path, which is derived from the URDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from remoroo_lc.constants import DTYPE, EPS, POINT_DIM, TASK_DIM
from remoroo_lc.reference.linalg import chol_det, cholesky_spd
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import exp_so3, make_transform

# Joint kind codes, shared with the kernels.
KIND_FIXED = 0
KIND_REVOLUTE = 1
KIND_PRISMATIC = 2

_KIND_OF = {
    "fixed": KIND_FIXED,
    "revolute": KIND_REVOLUTE,
    "continuous": KIND_REVOLUTE,
    "prismatic": KIND_PRISMATIC,
}


@dataclass
class FkResult:
    """Everything a downstream layer needs from one forward-kinematics pass."""

    link_T: np.ndarray  # (n_links, 4, 4) world poses of every link
    joint_p: np.ndarray  # (n, 3) world position of each actuated joint's axis
    joint_z: np.ndarray  # (n, 3) world direction of each actuated joint's axis
    joint_kind: np.ndarray  # (n,) KIND_*


class KinematicTree:
    """Forward kinematics and Jacobians for every model in a cell.

    Structure is resolved once at construction: link ordering, parent links,
    which global joint (if any) drives each link, and the ordered list of global
    joint indices on the path to each link.  Nothing about it depends on how many
    joints, chains, or models there are.
    """

    def __init__(self, cell: CellSpec) -> None:
        self.cell = cell
        self.n = cell.n_joints
        jidx = cell.joint_index()

        self.link_names: list[str] = []  # "model/link"
        self._link_id: dict[tuple[str, str], int] = {}
        self._parent: list[int] = []  # parent link id, -1 for a model root
        self._origin: list[np.ndarray] = []  # 4x4 parent-link -> joint frame
        self._axis: list[np.ndarray] = []
        self._kind: list[int] = []  # joint kind driving this link
        self._qidx: list[int] = []  # global joint index, -1 if not actuated
        self._locked_value: list[float] = []
        self._base: list[np.ndarray] = []  # 4x4 for roots, identity otherwise
        self._path: list[list[int]] = []  # global joint indices, root-first

        for m in cell.models:
            urdf = cell.urdfs[m.name]
            for link in urdf.ordered_links():
                lid = len(self.link_names)
                self._link_id[(m.name, link)] = lid
                self.link_names.append(f"{m.name}/{link}")
                if link == urdf.root:
                    self._parent.append(-1)
                    self._origin.append(np.eye(4, dtype=DTYPE))
                    self._axis.append(np.zeros(POINT_DIM, dtype=DTYPE))
                    self._kind.append(KIND_FIXED)
                    self._qidx.append(-1)
                    self._locked_value.append(0.0)
                    self._base.append(m.base.astype(DTYPE))
                    self._path.append([])
                    continue
                j = urdf.by_name[urdf.parent_joint[link]]
                pid = self._link_id[(m.name, j.parent)]
                kind = _KIND_OF.get(j.jtype)
                if kind is None:
                    raise ValueError(f"{m.name}: unsupported joint type {j.jtype!r}")
                value = 0.0
                gi = -1
                if j.moves:
                    if j.name in m.joint_names:
                        gi = jidx[(m.name, j.name)]
                    elif j.name in m.locked:
                        value = m.locked[j.name]
                    elif j.mimic is not None:
                        # A mimic joint that is not itself actuated contributes a
                        # fixed offset here; its source joint carries the motion.
                        value = float(j.mimic[2])
                    else:
                        raise ValueError(f"{m.name}: joint {j.name!r} is unaccounted for")
                self._parent.append(pid)
                self._origin.append(j.origin.astype(DTYPE))
                self._axis.append(j.axis.astype(DTYPE))
                self._kind.append(kind if gi >= 0 or value != 0.0 else kind)
                self._qidx.append(gi)
                self._locked_value.append(value)
                self._base.append(np.eye(4, dtype=DTYPE))
                path = list(self._path[pid])
                if gi >= 0:
                    path.append(gi)
                self._path.append(path)

        self.n_links = len(self.link_names)
        self._parent_arr = np.asarray(self._parent, dtype=np.int32)
        self._qidx_arr = np.asarray(self._qidx, dtype=np.int32)
        self._kind_arr = np.asarray(self._kind, dtype=np.int32)
        self._locked_arr = np.asarray(self._locked_value, dtype=DTYPE)
        self._origin_arr = np.stack(self._origin).astype(DTYPE)
        self._axis_arr = np.stack(self._axis).astype(DTYPE)
        self._base_arr = np.stack(self._base).astype(DTYPE)

        # Global joint kind vector, indexed by global joint id.
        self.joint_kind = np.zeros(self.n, dtype=np.int32)
        for lid in range(self.n_links):
            gi = self._qidx[lid]
            if gi >= 0:
                self.joint_kind[gi] = self._kind[lid]

        # Frames: one per TCP, plus one per collision sphere.
        self.tcp_link = np.asarray(
            [self._link_id[(t.model, t.frame)] for t in cell.tcps], dtype=np.int32
        )
        self.tcp_offset = (
            np.stack([t.offset for t in cell.tcps]).astype(DTYPE)
            if cell.n_tcps
            else np.zeros((0, 4, 4), dtype=DTYPE)
        )
        self.tcp_base_rot = np.stack(
            [
                next(m for m in cell.models if m.name == t.model).base[
                    :POINT_DIM, :POINT_DIM
                ]
                for t in cell.tcps
            ]
        ).astype(DTYPE)

        # Support masks: which global joints can move each link.
        self.link_support = np.zeros((self.n_links, self.n), dtype=bool)
        for lid, path in enumerate(self._path):
            for gi in path:
                self.link_support[lid, gi] = True

    # ------------------------------------------------------------------ #
    def link_id(self, model: str, link: str) -> int:
        return self._link_id[(model, link)]

    def fk(self, q: np.ndarray) -> FkResult:
        """Forward kinematics for the whole cell."""
        q = np.asarray(q, dtype=DTYPE).reshape(-1)
        if q.shape[0] != self.n:
            raise ValueError(f"q has {q.shape[0]} entries, expected {self.n}")
        link_T = np.zeros((self.n_links, 4, 4), dtype=DTYPE)
        joint_p = np.zeros((self.n, POINT_DIM), dtype=DTYPE)
        joint_z = np.zeros((self.n, POINT_DIM), dtype=DTYPE)
        for lid in range(self.n_links):
            pid = self._parent[lid]
            if pid < 0:
                link_T[lid] = self._base_arr[lid]
                continue
            T_joint = link_T[pid] @ self._origin_arr[lid]
            gi = self._qidx[lid]
            kind = self._kind[lid]
            value = float(q[gi]) if gi >= 0 else float(self._locked_arr[lid])
            if kind == KIND_REVOLUTE:
                M = np.eye(4, dtype=DTYPE)
                M[:POINT_DIM, :POINT_DIM] = exp_so3(self._axis_arr[lid] * DTYPE(value))
            elif kind == KIND_PRISMATIC:
                M = np.eye(4, dtype=DTYPE)
                M[:POINT_DIM, POINT_DIM] = self._axis_arr[lid] * DTYPE(value)
            else:
                M = np.eye(4, dtype=DTYPE)
            link_T[lid] = T_joint @ M
            if gi >= 0:
                joint_p[gi] = T_joint[:POINT_DIM, POINT_DIM]
                joint_z[gi] = T_joint[:POINT_DIM, :POINT_DIM] @ self._axis_arr[lid]
        return FkResult(link_T, joint_p, joint_z, self.joint_kind)

    # ------------------------------------------------------------------ #
    def frame_pose(self, fk: FkResult, link: int, offset: np.ndarray | None = None):
        """World (position, rotation) of a frame attached to `link`."""
        T = fk.link_T[link]
        if offset is not None:
            T = T @ offset
        return T[:POINT_DIM, POINT_DIM].copy(), T[:POINT_DIM, :POINT_DIM].copy()

    def point_jacobian(self, fk: FkResult, link: int, p_world: np.ndarray) -> np.ndarray:
        """(3, n) positional Jacobian of a point rigidly attached to `link`.

        Columns for joints off the path are exactly zero, which is what makes the
        stacked formulation in Layer 2 and the per-pair rows in Layer 3 correct
        without any per-chain bookkeeping.
        """
        J = np.zeros((POINT_DIM, self.n), dtype=DTYPE)
        for gi in self._path[link]:
            if fk.joint_kind[gi] == KIND_REVOLUTE:
                J[:, gi] = np.cross(fk.joint_z[gi], p_world - fk.joint_p[gi])
            else:
                J[:, gi] = fk.joint_z[gi]
        return J

    def point_jacobians(
        self, fk: FkResult, links: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """(S, 3, n) positional Jacobians for many attached points at once.

        Same arithmetic as `point_jacobian`, evaluated joint-major so the Python
        loop is over joints (a handful) instead of over points (hundreds).  The
        support mask decides which points a joint moves; everything else stays
        exactly zero.
        """
        n_pts = links.shape[0]
        J = np.zeros((n_pts, POINT_DIM, self.n), dtype=DTYPE)
        if n_pts == 0:
            return J
        support = self.link_support[links]
        for gi in range(self.n):
            sel = support[:, gi]
            if not sel.any():
                continue
            if fk.joint_kind[gi] == KIND_REVOLUTE:
                J[sel, :, gi] = np.cross(fk.joint_z[gi], points[sel] - fk.joint_p[gi])
            else:
                J[sel, :, gi] = fk.joint_z[gi]
        return J

    def frame_jacobian(
        self, fk: FkResult, link: int, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """(TASK_DIM, n) world-frame geometric Jacobian of a frame on `link`."""
        p, _ = self.frame_pose(fk, link, offset)
        J = np.zeros((TASK_DIM, self.n), dtype=DTYPE)
        for gi in self._path[link]:
            if fk.joint_kind[gi] == KIND_REVOLUTE:
                J[:POINT_DIM, gi] = np.cross(fk.joint_z[gi], p - fk.joint_p[gi])
                J[POINT_DIM:, gi] = fk.joint_z[gi]
            else:
                J[:POINT_DIM, gi] = fk.joint_z[gi]
        return J

    # ------------------------------------------------------------------ #
    def tcp_poses(self, fk: FkResult) -> tuple[np.ndarray, np.ndarray]:
        """(T, 3) positions and (T, 3, 3) rotations of every TCP, world frame."""
        n_t = self.cell.n_tcps
        P = np.zeros((n_t, POINT_DIM), dtype=DTYPE)
        R = np.zeros((n_t, POINT_DIM, POINT_DIM), dtype=DTYPE)
        for i in range(n_t):
            P[i], R[i] = self.frame_pose(fk, int(self.tcp_link[i]), self.tcp_offset[i])
        return P, R

    def tcp_jacobians(self, fk: FkResult) -> np.ndarray:
        """(T, TASK_DIM, n) stacked per-TCP Jacobians in the world frame."""
        n_t = self.cell.n_tcps
        J = np.zeros((n_t, TASK_DIM, self.n), dtype=DTYPE)
        for i in range(n_t):
            J[i] = self.frame_jacobian(fk, int(self.tcp_link[i]), self.tcp_offset[i])
        return J

    def manipulability(self, J_tcp: np.ndarray) -> np.ndarray:
        """w_i = sqrt(det(J_i J_i^T)) per TCP, over that TCP's TASK_DIM rows.

        Computed through the same Cholesky the kernels use so the two agree bit
        for bit; a non-positive-definite Gram matrix (an exactly singular
        configuration) yields w = 0.
        """
        n_t = J_tcp.shape[0]
        w = np.zeros(n_t, dtype=DTYPE)
        for i in range(n_t):
            A = (J_tcp[i] @ J_tcp[i].T).astype(DTYPE)
            L, ok = cholesky_spd(A, floor=float(EPS))
            w[i] = chol_det(L) if ok else DTYPE(0.0)
        return w


def sphere_world_centres(
    tree: KinematicTree, fk: FkResult, link_ids: np.ndarray, centres_local: np.ndarray
) -> np.ndarray:
    """Transform link-frame sphere centres into the world frame."""
    out = np.zeros((link_ids.shape[0], POINT_DIM), dtype=DTYPE)
    for k in range(link_ids.shape[0]):
        T = fk.link_T[int(link_ids[k])]
        out[k] = T[:POINT_DIM, :POINT_DIM] @ centres_local[k] + T[:POINT_DIM, POINT_DIM]
    return out


def frame_from_offset(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Convenience: build a 4x4 from a position and rotation."""
    return make_transform(R.astype(DTYPE), p.astype(DTYPE))
