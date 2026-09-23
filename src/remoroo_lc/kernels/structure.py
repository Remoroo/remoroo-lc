"""Flatten a CellSpec into plain arrays the kernels can consume.

Everything that is a property of the cell rather than of the state is resolved
here, once, at construction: tree topology, which joints move which links, the
collision-pair list and its order, environment primitives, limits and gains.  The
kernels then contain no branching on cell shape at all -- they loop over counts
that arrive as arguments.

This is also the single place where the reference and kernel paths are made to
agree on structure.  Both build their pair list through
`reference.walls.build_pair_list`, so there is no second implementation of the
ordering rule that could drift from the first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from remoroo_lc.constants import DTYPE, POINT_DIM, TASK_DIM
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.walls import WallBuilder
from remoroo_lc.schema import (
    CellSpec,
    RateMismatch,
    refuse_mesh_obstacles,
    resolve_rates,
)

# Environment primitive type codes, shared with the kernels.
ENV_PLANE = 0
ENV_BOX = 1
ENV_SPHERE = 2
ENV_CYLINDER = 3

_ENV_CODE = {"plane": ENV_PLANE, "box": ENV_BOX, "sphere": ENV_SPHERE, "cylinder": ENV_CYLINDER}
# ⚠ There is no code in here for a mesh, and none can be added without mesh
# collision in the kernels.  `warp_backend.env_distance` dispatches on this code
# and its LAST branch is the fall-through, so an unrecognised code is computed as
# a cylinder: a placeholder for a mesh would upload a degenerate cylinder at the
# mesh's pose and the QP would steer around an obstacle that does not exist.
# `build_structure` therefore refuses a mesh on the host, before any of this is
# built -- a device kernel has no way to raise.


@dataclass
class CellStructure:
    """Cell-constant data, as flat NumPy arrays ready to upload."""

    cell: CellSpec
    tree: KinematicTree
    walls: WallBuilder

    # counts
    n_joints: int
    n_links: int
    n_tcps: int
    n_spheres: int
    n_pairs: int
    n_rows: int
    n_env: int
    eff_dim: int

    # topology, per link
    parent: np.ndarray  # (L,) int32, -1 for a model root
    qindex: np.ndarray  # (L,) int32, global joint index or -1
    kind: np.ndarray  # (L,) int32, KIND_*
    locked: np.ndarray  # (L,) float32, value of a non-actuated joint
    origin: np.ndarray  # (L, 4, 4) float32, parent link -> joint frame
    axis: np.ndarray  # (L, 3) float32
    base: np.ndarray  # (L, 4, 4) float32, world pose of a model root
    support: np.ndarray  # (L, n) int32, 1 if joint j moves link l

    # frames
    tcp_link: np.ndarray  # (T,) int32
    tcp_offset: np.ndarray  # (T, 4, 4) float32
    tcp_base: np.ndarray  # (T, 4, 4) float32, that TCP's model base transform
    tcp_base_inv: np.ndarray  # (T, 4, 4) float32
    w_thresh: np.ndarray  # (T,) float32
    eff_offset: np.ndarray  # (T,) int32, start of this TCP's effector channels
    eff_width: np.ndarray  # (T,) int32

    # collision
    sphere_link: np.ndarray  # (S,) int32
    sphere_centre: np.ndarray  # (S, 3) float32, link frame
    sphere_radius: np.ndarray  # (S,) float32
    # Link-level broadphase tables (see reference.walls.group_pairs_by_link).
    # Robot-robot rows come FIRST and are grouped into contiguous blocks, one per
    # link pair, so one bounding-sphere test can retire a whole block.
    blk_ia: np.ndarray  # (B,) index into bound_* for the block's first link
    blk_ib: np.ndarray  # (B,)
    blk_start: np.ndarray  # (B,) first pair index of the block
    blk_count: np.ndarray  # (B,)
    bound_link: np.ndarray  # (L,) global link id
    bound_centre: np.ndarray  # (L, 3) in the link frame
    bound_radius: np.ndarray  # (L,)
    n_blocks: int
    n_rr: int  # robot-robot pairs; env pairs occupy [n_rr, n_pairs)
    #: Rows emitted per link block.  0 keeps one row per sphere pair; N > 0
    #: emits the N nearest of each block, which bounds n_rows independently of
    #: how many pairs the cell has.  See reference.walls.
    rows_per_block: int
    env_row0: int  # first row of the environment-pair block
    pair_block: np.ndarray  # (P,) block index per pair, -1 if not broadphased
    pair_kind: np.ndarray  # (P,) int32
    pair_a: np.ndarray  # (P,) int32
    pair_b: np.ndarray  # (P,) int32
    env_type: np.ndarray  # (E,) int32
    env_pose: np.ndarray  # (E, 4, 4) float32
    env_dims: np.ndarray  # (E, 3) float32

    # limits
    q_lo: np.ndarray
    q_hi: np.ndarray
    qd_max: np.ndarray
    q_rest: np.ndarray
    w_joint: np.ndarray
    task_v: np.ndarray  # (TASK_DIM,)
    task_a: np.ndarray
    task_j: np.ndarray
    eff_rate: np.ndarray  # (eff_dim,)
    eff_default: np.ndarray  # (eff_dim,)

    # scalars
    dt_c: float
    dt_p: float
    #: The rates `dt_c`/`dt_p` came from, and the exact number of control ticks
    #: in one action.  Carried on the structure so a caller that DRIVES this
    #: controller can check the invariant on the built object rather than on the
    #: yaml text: `ticks_per_action * dt_c` must equal `dt_p`, or the chunk is
    #: being consumed at a different rate than it is being fed.  See
    #: `schema.resolve_rates` and `assert_drive_rate`.
    policy_hz: float
    command_hz: float
    #: 0 when command_hz is not an exact multiple of policy_hz -- i.e. when no
    #: whole number of control ticks spans one action.  Such a cell can still be
    #: driven tick-by-tick; it cannot be driven in fixed-size chunks.
    ticks_per_action: int
    brake_margin: float
    accel_lag_ticks: float
    pos_lag_ticks: float
    v_clamp_scale: float
    kp_lin: float
    kp_ang: float
    lambda_min: float
    lambda_max: float
    k_post: float
    w_post: float
    rho: float
    iterations: int
    jl_xi: float
    jl_safe: float
    jl_infl: float
    cd_xi: float
    cd_safe: float
    cd_infl: float
    delta_mode: int  # 0 = cumulative, 1 = per_observation
    track_mode: int  # 0 = point_to_point, 1 = follower (see ChunkInterpolator)


def assert_drive_rate(structure: CellStructure, ticks_per_action: int, source: str) -> None:
    """Check the invariant on the LIVE object: one action == one chunk duration.

    `structure.dt_p` is how long the interpolator plans for a chunk to last.
    `ticks_per_action * structure.dt_c` is how long the caller will actually
    spend executing it.  These are the same physical quantity arrived at down
    two different paths, so comparing them catches a divergence introduced
    ANYWHERE -- a second yaml, a hand-set constructor argument, a stale cache --
    and not merely two config keys that happen to be spelled differently.

    This is the check that would have caught the 2x sprint on the first step of
    the first run instead of after twenty of them.
    """
    ticks = int(ticks_per_action)
    if ticks < 1:
        raise RateMismatch(f"{source}: ticks_per_action must be >= 1, got {ticks}")
    planned = float(structure.dt_p)
    executed = ticks * float(structure.dt_c)
    if abs(planned - executed) > 1e-9 * max(planned, executed):
        raise RateMismatch(
            f"chunk duration disagrees with the tick budget: remoroo-lc sizes a chunk "
            f"to {planned * 1e3:.4g} ms (policy_hz {structure.policy_hz:g}), but {source} "
            f"will execute it over {ticks} ticks = {executed * 1e3:.4g} ms "
            # planned / executed: squeezing a 40 ms chunk into 20 ms of ticks runs it
            # at 2x speed, not 0.5x.  The inverse reads plausible and is backwards.
            f"(command_hz {structure.command_hz:g}). Every chunk would run at "
            f"{planned / executed:.4g}x the commanded speed."
        )


def _pair_block(walls) -> np.ndarray:
    """Block index per pair, -1 where the broadphase does not apply.

    Lets the per-(env, pair) kernel consult block liveness with one lookup
    instead of searching, which is what makes the parallel form as cheap as the
    serial one per row.
    """
    pb = np.full(len(walls.pairs), -1, dtype=np.int32)
    for k, (a, c) in enumerate(zip(walls.blk_start, walls.blk_count)):
        pb[int(a) : int(a) + int(c)] = k
    return pb


def build_structure(
    cell: CellSpec,
    delta_mode: str = "cumulative",
    *,
    policy_hz: float | None = None,
    command_hz: float | None = None,
    rate_source: str = "build_structure(policy_hz=...)",
) -> CellStructure:
    """Flatten `cell` into kernel-ready arrays.

    `policy_hz` / `command_hz` are a DECLARATION, not an override: pass what the
    caller believes it will drive this controller at and `resolve_rates` raises
    if the cell disagrees.  See `schema.resolve_rates` for why the cell wins.
    """
    # FIRST, before a single array exists.  This is the door of the layer that
    # flattens the obstacle world for the kernels, and it is one of the four
    # places a mesh obstacle can enter the collision path; the others are
    # `walls.build_pair_list` (reached from here and from `WallBuilder`, which
    # `Controller` builds without ever calling this function) and
    # `walls.env_distance` / `env_distance_batch`.  See `_ENV_CODE` above for why
    # there is nothing this function could put in `env_type` instead.
    refuse_mesh_obstacles(cell.environment, consumer="kernels.structure.build_structure")
    tree = KinematicTree(cell)
    walls = WallBuilder(cell, tree)
    lim = cell.limits
    rates = resolve_rates(
        cell, policy_hz=policy_hz, command_hz=command_hz, source=rate_source
    )

    n = cell.n_joints
    n_links = tree.n_links
    n_tcps = cell.n_tcps
    n_spheres = len(walls.spheres)
    n_pairs = len(walls.pairs)

    support = tree.link_support.astype(np.int32)

    eff_offset, eff_width, off = [], [], 0
    eff_rate, eff_default = [], []
    for t in cell.tcps:
        eff_offset.append(off)
        eff_width.append(t.effector.width)
        eff_rate.extend([t.effector.rate_limit] * t.effector.width)
        eff_default.extend(list(t.effector.default))
        off += t.effector.width

    env_type = np.asarray([_ENV_CODE[p.ptype] for p in cell.environment], dtype=np.int32)
    env_pose = (
        np.stack([p.pose for p in cell.environment]).astype(DTYPE)
        if cell.environment
        else np.zeros((0, 4, 4), dtype=DTYPE)
    )
    env_dims = np.zeros((len(cell.environment), POINT_DIM), dtype=DTYPE)
    for e, p in enumerate(cell.environment):
        env_dims[e, : p.dims.shape[0]] = p.dims

    task = lim["task"]
    def axis_vec(key: str) -> np.ndarray:
        return np.asarray(
            [task["linear"][key]] * POINT_DIM + [task["angular"][key]] * POINT_DIM, dtype=DTYPE
        )

    from remoroo_lc.reference.interpolator import ChunkInterpolator

    if delta_mode not in ChunkInterpolator.DELTA_MODES:
        raise ValueError(f"unknown delta_mode {delta_mode!r}")

    return CellStructure(
        cell=cell,
        tree=tree,
        walls=walls,
        n_joints=n,
        n_links=n_links,
        n_tcps=n_tcps,
        n_spheres=n_spheres,
        n_pairs=n_pairs,
        n_rows=walls.n_rows,
        n_env=len(cell.environment),
        eff_dim=off,
        parent=np.asarray(tree._parent, dtype=np.int32),
        qindex=np.asarray(tree._qidx, dtype=np.int32),
        kind=np.asarray(tree._kind, dtype=np.int32),
        locked=np.asarray(tree._locked_value, dtype=DTYPE),
        origin=tree._origin_arr.astype(DTYPE),
        axis=tree._axis_arr.astype(DTYPE),
        base=tree._base_arr.astype(DTYPE),
        support=support,
        tcp_link=tree.tcp_link.astype(np.int32),
        tcp_offset=tree.tcp_offset.astype(DTYPE),
        tcp_base=np.stack(
            [next(m for m in cell.models if m.name == t.model).base for t in cell.tcps]
        ).astype(DTYPE),
        tcp_base_inv=np.stack(
            [
                _inv(next(m for m in cell.models if m.name == t.model).base)
                for t in cell.tcps
            ]
        ).astype(DTYPE),
        w_thresh=np.asarray([t.w_thresh for t in cell.tcps], dtype=DTYPE),
        eff_offset=np.asarray(eff_offset, dtype=np.int32),
        eff_width=np.asarray(eff_width, dtype=np.int32),
        sphere_link=walls.spheres.link.astype(np.int32),
        sphere_centre=walls.spheres.centre.astype(DTYPE),
        sphere_radius=walls.spheres.radius.astype(DTYPE),
        blk_ia=walls._blk_ia.astype(np.int32),
        blk_ib=walls._blk_ib.astype(np.int32),
        blk_start=walls.blk_start.astype(np.int32),
        blk_count=walls.blk_count.astype(np.int32),
        bound_link=walls.bound_link.astype(np.int32),
        bound_centre=walls.bound_centre.astype(DTYPE),
        bound_radius=walls.bound_radius.astype(DTYPE),
        n_blocks=int(walls.blk_start.size),
        n_rr=int(np.count_nonzero(walls.pairs.kind == 0)),
        rows_per_block=int(walls.rows_per_block),
        env_row0=int(getattr(walls, "_env_row0", walls.n_joint_rows + len(walls.pairs))),
        pair_block=_pair_block(walls),
        pair_kind=walls.pairs.kind.astype(np.int32),
        pair_a=walls.pairs.a.astype(np.int32),
        pair_b=walls.pairs.b.astype(np.int32),
        env_type=env_type,
        env_pose=env_pose,
        env_dims=env_dims,
        q_lo=walls.q_lo.astype(DTYPE),
        q_hi=walls.q_hi.astype(DTYPE),
        qd_max=cell.joint_velocity_limits().astype(DTYPE),
        q_rest=cell.rest_posture().astype(DTYPE),
        w_joint=np.full(n, DTYPE(lim["solver"]["joint_weight"]), dtype=DTYPE),
        task_v=axis_vec("v_max"),
        task_a=axis_vec("a_max"),
        task_j=axis_vec("j_max"),
        eff_rate=np.asarray(eff_rate, dtype=DTYPE),
        eff_default=np.asarray(eff_default, dtype=DTYPE),
        dt_c=rates["dt_c"],
        dt_p=rates["dt_p"],
        policy_hz=rates["policy_hz"],
        command_hz=rates["command_hz"],
        ticks_per_action=int(rates["ticks_per_action"]),
        brake_margin=float(task.get("brake_margin", 0.6)),
        accel_lag_ticks=float(task.get("accel_lag_ticks", 4.0)),
        pos_lag_ticks=float(task.get("pos_lag_ticks", 12.0)),
        v_clamp_scale=float(lim["diffik"].get("v_clamp_scale", 4.0)),
        kp_lin=float(lim["diffik"].get("kp_linear", rates["command_hz"])),
        kp_ang=float(lim["diffik"].get("kp_angular", rates["command_hz"])),
        lambda_min=float(lim["diffik"]["lambda_min"]),
        lambda_max=float(lim["diffik"]["lambda_max"]),
        k_post=float(lim["posture"]["k_post"]),
        w_post=float(lim["posture"]["weight"]),
        rho=float(lim["solver"]["rho"]),
        iterations=int(lim["solver"]["iterations"]),
        jl_xi=float(lim["joint_limit_damper"]["xi"]),
        jl_safe=float(np.radians(lim["joint_limit_damper"]["d_safe_deg"])),
        jl_infl=float(np.radians(lim["joint_limit_damper"]["d_infl_deg"])),
        cd_xi=float(lim["collision_damper"]["xi"]),
        cd_safe=float(lim["collision_damper"]["d_safe_m"]),
        cd_infl=float(lim["collision_damper"]["d_infl_m"]),
        delta_mode=ChunkInterpolator.DELTA_MODES.index(delta_mode),
        track_mode=ChunkInterpolator.TRACK_MODES.index(
            str(task.get("mode", "point_to_point"))
        ),
    )


def _inv(T: np.ndarray) -> np.ndarray:
    R = T[:POINT_DIM, :POINT_DIM]
    p = T[:POINT_DIM, POINT_DIM]
    out = np.eye(4, dtype=DTYPE)
    out[:POINT_DIM, :POINT_DIM] = R.T
    out[:POINT_DIM, POINT_DIM] = -(R.T @ p)
    return out


def stacked_task_dim(structure: CellStructure) -> int:
    return TASK_DIM * structure.n_tcps
