"""Action tapes: recorded policy output, generated from a cell.

A tape is JSON lines.  The first line is metadata; every line after it is one
policy step:

    {"kind": "meta", "name": ..., "category": ..., "cell": ..., "action_dim": ...}
    {"t": 0.0,    "action": [...]}
    {"t": 0.0625, "action": [...]}

The action vector is the frozen schema: per TCP, in config order, a position
delta and a rotation-vector delta in that model's base frame, then g_i normalised
effector floats.  A tape for one cell is not a tape for another, and the metadata
says which, so a mismatch is caught at load rather than producing plausible
nonsense.

Generators are parameterised by the cell and by workspace bounds -- never by a
joint count, a chain count, or a link length.  A cell with one effectorless chain
and a cell with a shared trunk both get tapes from the same functions; what
changes is what the functions read out of the config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from remoroo_lc.constants import DTYPE, POINT_DIM, TASK_DIM
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import transform_inv

CATEGORIES = ("clean", "adversarial", "singularity")

#: Decimal places for the timestamp column.  Named rather than inline because the
#: agnosticism gate forbids bare 6 and 7 in core code -- see test_agnosticism.py --
#: and a rounding precision that happens to collide with a DOF count is exactly
#: the kind of thing that gate cannot tell apart, which is the price of it being
#: blunt enough to work.
_TIME_DECIMALS = 9


@dataclass
class Tape:
    """A sequence of policy actions plus what produced it."""

    name: str
    category: str
    cell: str
    actions: np.ndarray  # (N, action_dim)
    policy_hz: float
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {self.category!r}")
        self.actions = np.asarray(self.actions, dtype=DTYPE)

    @property
    def duration(self) -> float:
        return len(self.actions) / self.policy_hz

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "name": self.name,
                        "category": self.category,
                        "cell": self.cell,
                        "action_dim": int(self.actions.shape[1]),
                        "policy_hz": self.policy_hz,
                        "notes": self.notes,
                        **self.meta,
                    }
                )
                + "\n"
            )
            for i, a in enumerate(self.actions):
                fh.write(
                    json.dumps({"t": round(i / self.policy_hz, _TIME_DECIMALS), "action": [float(v) for v in a]})
                    + "\n"
                )
        return path

    @staticmethod
    def read(path: str | Path) -> Tape:
        rows, meta = [], None
        with open(path) as fh:
            for line in fh:
                obj = json.loads(line)
                if obj.get("kind") == "meta":
                    meta = obj
                else:
                    rows.append(obj["action"])
        if meta is None:
            raise ValueError(f"{path}: no metadata line")
        actions = np.asarray(rows, dtype=DTYPE)
        if actions.shape[1] != meta["action_dim"]:
            raise ValueError(f"{path}: action width does not match its own metadata")
        return Tape(
            name=meta["name"],
            category=meta["category"],
            cell=meta["cell"],
            actions=actions,
            policy_hz=float(meta["policy_hz"]),
            notes=meta.get("notes", ""),
        )

    def check_against(self, cell: CellSpec) -> None:
        if self.actions.shape[1] != cell.action_dim:
            raise ValueError(
                f"tape {self.name!r} has action width {self.actions.shape[1]}, "
                f"cell {cell.name!r} expects {cell.action_dim}"
            )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _rest_pose_base(cell: CellSpec) -> tuple[np.ndarray, list[np.ndarray]]:
    """Each TCP's rest position and orientation, in its own model base frame."""
    tree = KinematicTree(cell)
    fk = tree.fk(cell.rest_posture())
    p_w, R_w = tree.tcp_poses(fk)
    P, R = [], []
    for i, t in enumerate(cell.tcps):
        base = next(m for m in cell.models if m.name == t.model).base
        inv = transform_inv(base)
        P.append(inv[:POINT_DIM, :POINT_DIM] @ p_w[i] + inv[:POINT_DIM, POINT_DIM])
        R.append(inv[:POINT_DIM, :POINT_DIM] @ R_w[i])
    return np.stack(P).astype(DTYPE), R


def _from_paths(
    cell: CellSpec,
    paths: list[np.ndarray],
    name: str,
    category: str,
    rot_paths: list[np.ndarray] | None = None,
    eff: np.ndarray | None = None,
    notes: str = "",
) -> Tape:
    """Turn per-TCP base-frame position paths into a cumulative-delta tape.

    `paths[i]` is (N+1, 3), absolute positions starting AT the TCP's current
    pose; the tape carries the differences, which is what the frozen action
    schema is.  `rot_paths[i]` is (N+1, 3) of rotation vectors, differenced the
    same way (small increments, so composing them on SO(3) and differencing the
    vectors agree to the order this is used at).
    """
    n_steps = paths[0].shape[0] - 1
    actions = np.zeros((n_steps, cell.action_dim), dtype=DTYPE)
    slices = cell.action_slices()
    for i in range(cell.n_tcps):
        pose, eff_sl = slices[i]
        d = np.diff(paths[i], axis=0)
        actions[:, pose.start : pose.start + POINT_DIM] = d
        if rot_paths is not None:
            actions[:, pose.start + POINT_DIM : pose.stop] = np.diff(rot_paths[i], axis=0)
        g = cell.tcps[i].effector.width
        if g and eff is not None:
            actions[:, eff_sl] = eff[:n_steps, None]
    return Tape(
        name=name,
        category=category,
        cell=cell.name,
        actions=actions,
        policy_hz=float(cell.limits["rates"]["policy_hz"]),
        notes=notes,
    )


def _radial_and_up(cell: CellSpec, i: int, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit "radially outward" and "world up" at TCP i, in its model base frame.

    Both are derived from the cell -- the mount transform and gravity -- so a
    limb bolted to a bench and one hanging upside down both get a sensible plane
    without anyone writing down which is which.
    """
    base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
    up = (base[:POINT_DIM, :POINT_DIM].T @ np.asarray([0.0, 0.0, 1.0], dtype=DTYPE)).astype(DTYPE)
    radial = p0 - np.dot(p0, up) * up  # the part of the reach perpendicular to up
    n = float(np.linalg.norm(radial))
    if n < 1e-6:
        # Directly under (or over) its own mount: any perpendicular will do, and
        # which one is chosen has to be deterministic.
        alt = np.asarray([1.0, 0.0, 0.0], dtype=DTYPE)
        radial = alt - float(np.dot(alt, up)) * up
        n = float(np.linalg.norm(radial))
    return (radial / DTYPE(n)).astype(DTYPE), up


def clearance_of(cell: CellSpec, paths: list[np.ndarray], samples: int = 40) -> float:
    """Smallest pair distance the cell would see if it tracked `paths` exactly.

    Kinematic only -- FK plus the wall builder, no dynamics -- because this is
    used to SIZE a tape, not to score one.  It needs the joint configuration that
    reaches each waypoint, which comes from a short damped-least-squares descent
    seeded at the rest posture: the same Layer-2 mathematics, run to convergence
    instead of one step per tick.
    """
    from remoroo_lc.reference.diffik import DiffIk
    from remoroo_lc.reference.walls import WallBuilder

    tree = KinematicTree(cell)
    walls = WallBuilder(cell, tree)
    ik = DiffIk(cell)
    lo, hi = cell.joint_limits()
    q = cell.rest_posture().copy()
    n_way = paths[0].shape[0]
    idx = np.unique(np.linspace(0, n_way - 1, min(samples, n_way)).astype(int))

    worst = np.inf
    for k in idx:
        for _ in range(60):
            fk = tree.fk(q)
            p_now, R_now = tree.tcp_poses(fk)
            J = tree.tcp_jacobians(fk)
            v = np.zeros(cell.task_dim, dtype=DTYPE)
            for i in range(cell.n_tcps):
                base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
                tgt = base[:POINT_DIM, :POINT_DIM] @ paths[i][k] + base[:POINT_DIM, POINT_DIM]
                v[i * TASK_DIM : i * TASK_DIM + POINT_DIM] = tgt - p_now[i]
            w = tree.manipulability(J)
            q = np.clip(q + DTYPE(0.5) * ik.solve(J, w, v).qd_des, lo, hi).astype(DTYPE)
        fk = tree.fk(q)
        d = walls.assemble(q, fk).distance[walls.n_joint_rows :]
        if d.size:
            worst = min(worst, float(np.min(d)))
    return worst


def fit_clean_amplitude(
    cell: CellSpec, make_paths, margin: float | None = None, floor: float = 0.1
) -> float:
    """Largest scale in (0, 1] whose path keeps every pair clear of the dampers.

    A tape called clean has to BE clean, on a cell nobody here has seen.  Rather
    than guessing amplitudes per robot -- which is the hardcoding this package
    exists to avoid -- each clean tape is sized by bisection against the cell's
    own collision geometry, and the amplitude it settles on is recorded in the
    tape's notes so the report says how much workspace the cell actually had.
    """
    if margin is None:
        d_infl = float(cell.limits["collision_damper"]["d_infl_m"])
        d_safe = float(cell.limits["collision_damper"]["d_safe_m"])
        # Never demand more clearance than the cell has standing still.  A
        # compact torso whose own forearm sits 61 mm from its upper arm at rest
        # cannot keep 60 mm through any motion at all, and insisting would shrink
        # every clean tape to nothing rather than sizing it.
        at_rest = clearance_of(cell, [p[:1] for p in make_paths(0.0)])
        margin = max(1.5 * d_safe, min(d_infl, 0.9 * at_rest))
    if clearance_of(cell, make_paths(1.0)) >= margin:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if clearance_of(cell, make_paths(mid)) >= margin:
            lo = mid
        else:
            hi = mid
    return max(lo, floor)


def _hold(paths: list[np.ndarray], n: int) -> list[np.ndarray]:
    return [np.concatenate([p, np.repeat(p[-1:], n, axis=0)]) for p in paths]


# --------------------------------------------------------------------------- #
# clean tapes
# --------------------------------------------------------------------------- #


def figure_eight(cell: CellSpec, size: float, speed: float, cycles: float = 2.0) -> Tape:
    """A lemniscate per TCP in its own base frame, at a chosen size and speed.

    `speed` is the peak Cartesian speed in m/s, which is what sets how hard this
    leans on the interpolator and how much tracking lag the actuation delay
    produces.  Both are reported by the scoreboard, so the tape names its speed.
    """
    hz = float(cell.limits["rates"]["policy_hz"])
    # Peak speed of a lemniscate of half-width `size` traversed in period T is
    # 2*pi*size/T; solve for the number of policy steps per cycle.
    period = 2.0 * np.pi * size / max(speed, 1e-6)
    n = max(int(round(period * hz * cycles)), 8)
    s = np.linspace(0.0, cycles * 2.0 * np.pi, n + 1)
    P0, _ = _rest_pose_base(cell)
    # An over-constrained cell (n < TASK_DIM * T, e.g. two chains sharing a
    # trunk) cannot serve independent commands to every TCP at once -- the
    # stacked solve returns a least-squares compromise and the "tracking error"
    # would be measuring that impossibility rather than the controller.  On those
    # cells the clean tapes move one TCP at a time and hold the rest, which IS
    # achievable.  The condition is read off the cell, not written per cell.
    solo = cell.n_joints < cell.task_dim

    def make_paths(scale: float) -> list[np.ndarray]:
        amp = size * scale
        out = []
        for i in range(cell.n_tcps):
            # The figure lies in the plane spanned by "radially outward from this
            # model's base" and "up in the world", both expressed in the model
            # base frame.  A figure drawn in the base frame's own x-y plane
            # instead sweeps horizontally, which on a two-chain cell walks the
            # TCPs straight into each other and makes a tape called clean
            # anything but.  Deriving the plane from the mount and from gravity
            # keeps it right for a chain mounted upside down as readily as for
            # one bolted to a bench.
            u, w = _radial_and_up(cell, i, P0[i])
            p = np.repeat(P0[i][None], n + 1, axis=0).astype(DTYPE)
            if not solo or i == 0:
                p = p + (amp * np.sin(s))[:, None] * u[None, :]
                p = p + (amp * np.sin(s) * np.cos(s))[:, None] * w[None, :]
            out.append(p.astype(DTYPE))
        return out

    scale = fit_clean_amplitude(cell, make_paths)
    paths = make_paths(scale)
    eff = 0.5 * (1.0 - np.cos(s[:-1])) if sum(cell.effector_widths) else None
    return _from_paths(
        cell,
        paths,
        name=f"figure_eight_{int(size * 1000)}mm_{int(speed * 100)}cms",
        category="clean",
        eff=None if eff is None else eff.astype(DTYPE),
        notes=(
            f"lemniscate, requested half-width {size} m, fitted to "
            f"{size * scale * 1000:.0f} mm by the cell's own clearance, peak speed "
            f"{speed} m/s" + (", one TCP at a time (over-constrained cell)" if solo else "")
        ),
    )


def approach_retreat(cell: CellSpec, depth: float, speed: float, cycles: int = 3) -> Tape:
    """Move each TCP toward the workspace floor and back, repeatedly."""
    hz = float(cell.limits["rates"]["policy_hz"])
    n_seg = max(int(round(depth / max(speed, 1e-6) * hz)), 4)
    P0, _ = _rest_pose_base(cell)
    ramp = np.linspace(0.0, 1.0, n_seg + 1)
    cycle = np.concatenate([ramp, ramp[::-1][1:]])
    prof = np.concatenate([cycle] + [cycle[1:]] * (cycles - 1))

    def make_paths(scale: float) -> list[np.ndarray]:
        out = []
        for i in range(cell.n_tcps):
            p = np.repeat(P0[i][None], prof.shape[0], axis=0).astype(DTYPE)
            # "Toward the floor" is -z in the MODEL BASE frame only if the model
            # is mounted upright; the sign comes from where the base points.
            base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
            down_world = np.asarray([0.0, 0.0, -1.0], dtype=DTYPE)
            down_base = base[:POINT_DIM, :POINT_DIM].T @ down_world
            out.append((p + (prof[:, None] * depth * scale) * down_base[None, :]).astype(DTYPE))
        return out

    scale = fit_clean_amplitude(cell, make_paths)
    paths = make_paths(scale)
    eff = np.clip(prof[:-1] * 1.5, 0.0, 1.0) if sum(cell.effector_widths) else None
    return _from_paths(
        cell,
        paths,
        name=f"approach_retreat_{int(depth * 1000)}mm",
        category="clean",
        eff=None if eff is None else eff.astype(DTYPE),
        notes=(
            f"{cycles} approach/retreat cycles, requested {depth} m, fitted to "
            f"{depth * scale * 1000:.0f} mm by the cell's own clearance, at {speed} m/s"
        ),
    )


def handover(cell: CellSpec, speed: float = 0.12, closeness: float = 0.65) -> Tape | None:  # noqa: C901
    """Bring two effector-bearing TCPs toward their shared midpoint and back.

    `closeness` is how far along the way each TCP travels.  It is not 1.0 on
    purpose: sending both effectors to the SAME point is asking them to occupy
    the same space, the collision damper correctly refuses, and the tape then
    measures the filter rather than tracking -- which is a fine thing to measure
    but not a clean tape.  0.65 brings them to a plausible transfer separation
    with the walls still quiet.

    Returns None for cells with fewer than two effector-bearing TCPs, which is a
    property of the cell and not a failure.
    """
    bearing = [i for i, t in enumerate(cell.tcps) if t.effector.width > 0]
    if len(bearing) < 2:
        return None
    hz = float(cell.limits["rates"]["policy_hz"])
    P0, _ = _rest_pose_base(cell)
    tree = KinematicTree(cell)
    fk = tree.fk(cell.rest_posture())
    p_w, _ = tree.tcp_poses(fk)
    mid_world = 0.5 * (p_w[bearing[0]] + p_w[bearing[1]])

    targets = []
    for i in range(cell.n_tcps):
        base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
        inv = transform_inv(base)
        mid = inv[:POINT_DIM, :POINT_DIM] @ mid_world + inv[:POINT_DIM, POINT_DIM]
        tgt = P0[i] + DTYPE(closeness) * (mid - P0[i])
        targets.append(tgt if i in bearing else P0[i])
    travel = max(float(np.max([np.linalg.norm(targets[i] - P0[i]) for i in bearing])), 1e-3)
    n_seg = max(int(round(travel / max(speed, 1e-6) * hz)), 8)
    ramp = np.linspace(0.0, 1.0, n_seg + 1)
    prof = np.concatenate([ramp, np.ones(n_seg // 2), ramp[::-1][1:]])
    def make_paths(scale: float) -> list[np.ndarray]:
        return [
            (P0[i][None] + (prof[:, None] * scale) * (targets[i] - P0[i])[None, :]).astype(DTYPE)
            for i in range(cell.n_tcps)
        ]

    scale = fit_clean_amplitude(cell, make_paths)
    paths = make_paths(scale)
    eff = np.concatenate(
        [np.zeros(n_seg), np.ones(n_seg // 2), np.zeros(prof.shape[0] - 1 - n_seg - n_seg // 2)]
    )
    # A cell with fewer joints than TASK_DIM * T cannot serve two independent
    # TCP commands at once, so on those cells this tape measures the DOF deficit
    # rather than the controller and belongs in the adversarial set -- where the
    # bar it has to meet, zero hard violations, is still exactly right.
    over_constrained = cell.n_joints < cell.task_dim
    return _from_paths(
        cell,
        paths,
        name="handover",
        category="adversarial" if over_constrained else "clean",
        eff=eff.astype(DTYPE),
        notes=(
            "both effector-bearing TCPs converge toward their shared midpoint, "
            f"fitted to {closeness * scale * 100:.0f}% of the way by clearance"
            + (
                "; classed adversarial because this cell has fewer joints than "
                "TASK_DIM * T and cannot satisfy both commands"
                if over_constrained
                else ""
            )
        ),
    )


# --------------------------------------------------------------------------- #
# adversarial tapes
# --------------------------------------------------------------------------- #


def through_the_floor(cell: CellSpec, overshoot: float = 0.8) -> Tape:
    """Command every TCP straight through the lowest environment plane."""
    hz = float(cell.limits["rates"]["policy_hz"])
    n = int(round(4.0 * hz))
    P0, _ = _rest_pose_base(cell)
    paths = []
    for i in range(cell.n_tcps):
        base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
        down_base = base[:POINT_DIM, :POINT_DIM].T @ np.asarray([0, 0, -1], dtype=DTYPE)
        ramp = np.linspace(0.0, overshoot, n + 1)[:, None]
        paths.append((P0[i][None] + ramp * down_base[None, :]).astype(DTYPE))
    return _from_paths(
        cell, paths, "through_the_floor", "adversarial",
        notes=f"{overshoot} m of commanded descent, straight through the table",
    )


def into_another_chain(cell: CellSpec) -> Tape | None:
    """Command each TCP to where another TCP currently is."""
    if cell.n_tcps < 2:
        return None
    hz = float(cell.limits["rates"]["policy_hz"])
    n = int(round(4.0 * hz))
    tree = KinematicTree(cell)
    fk = tree.fk(cell.rest_posture())
    p_w, _ = tree.tcp_poses(fk)
    P0, _ = _rest_pose_base(cell)
    paths = []
    for i in range(cell.n_tcps):
        other = p_w[(i + 1) % cell.n_tcps]
        base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
        inv = transform_inv(base)
        tgt = inv[:POINT_DIM, :POINT_DIM] @ other + inv[:POINT_DIM, POINT_DIM]
        ramp = np.linspace(0.0, 1.0, n + 1)[:, None]
        paths.append((P0[i][None] + ramp * (tgt - P0[i])[None, :]).astype(DTYPE))
    return _from_paths(
        cell, paths, "into_another_chain", "adversarial",
        notes="each TCP commanded to another TCP's current position",
    )


def into_joint_limits(cell: CellSpec) -> Tape:
    """Rotate hard about the base axis until joints run out of travel."""
    hz = float(cell.limits["rates"]["policy_hz"])
    n = int(round(5.0 * hz))
    P0, _ = _rest_pose_base(cell)
    paths, rots = [], []
    for i in range(cell.n_tcps):
        ang = np.linspace(0.0, 3.0, n + 1)
        r = np.linalg.norm(P0[i][:2]) or 0.3
        phi0 = np.arctan2(P0[i][1], P0[i][0])
        p = np.stack(
            [r * np.cos(phi0 + ang), r * np.sin(phi0 + ang), np.full(n + 1, P0[i][2])], axis=1
        ).astype(DTYPE)
        paths.append(p)
        rots.append(np.stack([np.zeros(n + 1), np.zeros(n + 1), ang], axis=1).astype(DTYPE))
    return _from_paths(
        cell, paths, "into_joint_limits", "adversarial", rot_paths=rots,
        notes="3 rad of commanded yaw sweep, well past what the joints allow",
    )


def beyond_reach(cell: CellSpec, distance: float = 1.5) -> Tape:
    """Command every TCP radially outward, far past any plausible envelope."""
    hz = float(cell.limits["rates"]["policy_hz"])
    n = int(round(4.0 * hz))
    P0, _ = _rest_pose_base(cell)
    paths = []
    for i in range(cell.n_tcps):
        d = P0[i].copy()
        d[2] = 0.0
        norm = float(np.linalg.norm(d))
        out = d / norm if norm > 1e-6 else np.asarray([1.0, 0.0, 0.0], dtype=DTYPE)
        ramp = np.linspace(0.0, distance, n + 1)[:, None]
        paths.append((P0[i][None] + ramp * out[None, :]).astype(DTYPE))
    return _from_paths(
        cell, paths, "beyond_reach", "adversarial",
        notes=f"{distance} m of commanded radial extension",
    )


# --------------------------------------------------------------------------- #
# singularity sweeps
# --------------------------------------------------------------------------- #


def singularity_sweep(cell: CellSpec, samples: int = 4000, seed: int = 0) -> Tape:
    """Drive each TCP through the low-manipulability pose found by sampling.

    The pose is found, not assumed: sample joint configurations, take the one
    with the lowest w for each TCP, and command a straight line from the rest
    pose through it and out the other side.  Nothing here knows what makes a
    given chain singular.
    """
    tree = KinematicTree(cell)
    lo, hi = cell.joint_limits()
    g = np.random.default_rng(seed)
    best_w = np.full(cell.n_tcps, np.inf)
    best_p = [None] * cell.n_tcps
    for _ in range(samples):
        q = (lo + (hi - lo) * g.random(cell.n_joints)).astype(DTYPE)
        fk = tree.fk(q)
        w = tree.manipulability(tree.tcp_jacobians(fk))
        p_w, _ = tree.tcp_poses(fk)
        for i in range(cell.n_tcps):
            if w[i] < best_w[i]:
                best_w[i] = float(w[i])
                best_p[i] = p_w[i].copy()

    hz = float(cell.limits["rates"]["policy_hz"])
    n = int(round(6.0 * hz))
    P0, _ = _rest_pose_base(cell)
    ramp = np.concatenate(
        [np.linspace(0.0, 1.3, n // 2 + 1), np.linspace(1.3, 0.0, n - n // 2 + 1)[1:]]
    )
    paths = []
    for i in range(cell.n_tcps):
        base = next(m for m in cell.models if m.name == cell.tcps[i].model).base
        inv = transform_inv(base)
        tgt = inv[:POINT_DIM, :POINT_DIM] @ best_p[i] + inv[:POINT_DIM, POINT_DIM]
        paths.append((P0[i][None] + ramp[:, None] * (tgt - P0[i])[None, :]).astype(DTYPE))
    return _from_paths(
        cell, paths, "singularity_sweep", "singularity",
        notes=(
            "straight line through the lowest-w pose found in "
            f"{samples} samples (w = {np.min(best_w):.2e}), and 30% past it"
        ),
    )


# --------------------------------------------------------------------------- #
# the standard set
# --------------------------------------------------------------------------- #


def standard_tapes(cell: CellSpec) -> list[Tape]:
    """Every tape the G0 scoreboard runs, for any cell."""
    out: list[Tape] = []
    for size in (0.04, 0.08, 0.15):
        for speed in (0.05, 0.12):
            out.append(figure_eight(cell, size=size, speed=speed))
    out.append(approach_retreat(cell, depth=0.12, speed=0.10))
    transfer = handover(cell)
    if transfer is not None:
        out.append(transfer)

    out.append(through_the_floor(cell))
    other = into_another_chain(cell)
    if other is not None:
        out.append(other)
    out.append(into_joint_limits(cell))
    out.append(beyond_reach(cell))

    out.append(singularity_sweep(cell))
    for tape in out:
        tape.check_against(cell)
    return out
