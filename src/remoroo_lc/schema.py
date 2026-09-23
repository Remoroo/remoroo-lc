"""The cell contract: dataclasses, loader, validator, and derived sizing.

This module defines the product contract that `remoroo setup` emits upstream and
`remoroo world` consumes downstream.  See CELL_SPEC.md for the normative
description of the file format; this file is its executable definition.

A cell is:
  * one or more URDF models, each mounted at a configured base transform,
  * a list of TCPs (model + frame), each optionally carrying an effector of
    configured command width g_i,
  * environment primitives and collision bookkeeping,
  * limits / gains, either inline or by reference to a sibling yaml.

Nothing here knows what kind of machine it is describing.  Chains may share
joints; a TCP may have g_i = 0; a model may have any number of joints of any
type.  Every size in the pipeline is derived from this file at load time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml

from remoroo_lc.constants import DTYPE, POINT_DIM, QUAT_DIM, TASK_DIM
from remoroo_lc.spatial import make_transform, quat_to_mat, rpy_to_mat
from remoroo_lc.urdf import UrdfModel, load_urdf

#: Version of the cell-file contract this loader implements.  Bump the major on
#: any change that would make a previously valid file invalid or change its
#: meaning; bump the minor on backward-compatible additions.
CELL_SPEC_VERSION = "1.0"

_SUPPORTED_MAJORS = ("1",)

_PRIMITIVE_TYPES = ("plane", "box", "sphere", "cylinder")

#: A mesh obstacle is PARSED AND CARRIED here, and collided against nowhere.
#:
#: corner_cell -- the real Siemens rig cell, measured 2026-09-23 -- describes its
#: surroundings as one STL (`meshes/cell_obstacles.stl`, scale 0.001) because a
#: corner of a room is not a handful of boxes.  This loader used to raise on
#: `type: mesh`, and the consequence was not "no mesh collision": it was that the
#: whole measured cell could not be loaded by ANY part of this package, including
#: scripts/sysid_tapes.py, which reads joints and rates and never looks at the
#: environment at all.  That is refusing in the wrong place.  The layers that lack
#: the capability are the collision builders, and they now refuse by name -- see
#: `refuse_mesh_obstacles`.
_MESH_TYPE = "mesh"

#: Every `environment[].type` this loader accepts.  A mesh is accepted by the
#: SCHEMA and rejected by the COLLISION layer; those are different doors.
_ENV_TYPES = (*_PRIMITIVE_TYPES, _MESH_TYPE)


class CellSpecError(ValueError):
    """Raised when a cell file does not satisfy the contract."""


class MeshCollisionUnsupported(CellSpecError):
    """A collision consumer was handed a mesh obstacle.  lc cannot collide with one.

    Its own exception type, for the same reason `RateMismatch` is: the failure it
    reports is NOT a malformed file.  The cell is right -- the mesh is what the
    installer measured -- and it is this package that is missing a capability.  A
    caller that can proceed without an environment (sysid, kinematics, limits,
    tape replay) should never see this; a caller that is building a collision
    world must see it rather than a collision world with a hole in it.

    It subclasses CellSpecError so existing `except CellSpecError` handlers keep
    catching it, and CellSpecError is a ValueError, so the older
    `raise ValueError("unknown primitive type ...")` contract in
    reference/walls.py is preserved for callers that catch that.
    """


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EffectorSpec:
    """A TCP's effector.

    `width` is the number of normalised command floats this effector consumes in
    the action vector.  Zero is a first-class value: a bare frame or a foot has
    no command channels and must flow through every layer unchanged.
    """

    kind: str = "none"
    width: int = 0
    rate_limit: float = 4.0  # units/s on the normalised [0, 1] command
    default: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width < 0:
            raise CellSpecError(f"effector width must be >= 0, got {self.width}")
        if len(self.default) != self.width:
            object.__setattr__(self, "default", tuple([0.0] * self.width))


@dataclass(frozen=True)
class TcpSpec:
    """A task frame at the end of a chain."""

    name: str
    model: str
    frame: str
    offset: np.ndarray  # 4x4, frame -> TCP
    effector: EffectorSpec
    w_thresh: float  # manipulability threshold for damping, calibrated per chain


@dataclass(frozen=True)
class ModelSpec:
    """One URDF mounted into the cell at a fixed base transform."""

    name: str
    urdf_path: Path
    base: np.ndarray  # 4x4, world -> model root link
    spheres_path: Path | None
    joint_names: tuple[str, ...]  # actuated joints, in load order
    locked: dict[str, float]  # joints held at a fixed value (never actuated)
    rest: np.ndarray | None  # optional posture target for this model's joints


@dataclass(frozen=True)
class EnvPrimitive:
    """A static environment obstacle expressed in world coordinates."""

    name: str
    ptype: str
    pose: np.ndarray  # 4x4
    dims: np.ndarray  # box: half-extents; sphere: [r]; cylinder: [r, half-height]


@dataclass(frozen=True)
class EnvMesh:
    """A static environment obstacle given as a mesh file, in world coordinates.

    Deliberately NOT an `EnvPrimitive` and deliberately WITHOUT a `dims` field.
    A mesh that carried `dims` could be read by primitive code by accident, and
    the accident is not a crash -- it is a box of half-extents [0, 0, 0] quietly
    joining the planner's obstacle set.  With no `dims` at all, primitive code
    either dispatches on `ptype` (and is made to refuse: see
    `refuse_mesh_obstacles`) or raises AttributeError.  Neither is silent.

    `mesh_path` is resolved against the cell file's directory at load, like every
    other path in the contract, and its existence is checked there: an obstacle
    carried as a path that points at nothing is not a carried measurement.
    """

    name: str
    pose: np.ndarray  # 4x4, world -> mesh origin
    mesh_path: Path  # absolute, resolved against the cell file's directory
    scale: np.ndarray  # (3,) per-axis multiplier on the file's own units
    #: A ClassVar, not a field, so `EnvMesh(..., ptype="box")` is impossible while
    #: `obstacle.ptype` still answers for both kinds (config_stamp reads it).
    ptype: ClassVar[str] = _MESH_TYPE


#: Anything that can appear in `CellSpec.environment`.
EnvObstacle = EnvPrimitive | EnvMesh


def refuse_mesh_obstacles(environment: tuple[EnvObstacle, ...], *, consumer: str) -> None:
    """Refuse, by name, if `environment` contains a mesh.  Call this at the door.

    Every layer that BUILDS OR CONSULTS collision geometry calls this before it
    builds anything, because the alternative is silent in exactly the case that
    matters.  corner_cell declares one mesh obstacle and no `spheres:` file, so
    the environment row loops produce zero rows with or without a mesh: the
    caller would be handed a pair list that looks complete while nothing about
    the environment had been checked at all.

    ⚠ This is unconditional -- `collision: {environment: false}` does not make a
    mesh acceptable here.  The reason is `kernels/structure.py`: it flattens
    every obstacle into an `env_type` code and uploads the array whole, and
    `warp_backend.env_distance` dispatches on that code with the CYLINDER branch
    as its fall-through, so there is no value on earth that can stand for a mesh
    in that array -- a placeholder becomes a phantom degenerate cylinder at the
    mesh's pose, inside the QP, on the device, where nothing can raise.  One
    unconditional rule at four doors beats a flag that means something different
    on the reference path than on the kernel path.
    """
    for obstacle in environment:
        if isinstance(obstacle, EnvMesh):
            raise MeshCollisionUnsupported(
                f"{consumer}: cannot build collision geometry for environment obstacle "
                f"{obstacle.name!r} -- it is a MESH ({obstacle.mesh_path}), and "
                "remoroo-lc has no mesh collision support.  Its collision world is "
                f"spheres against primitives only ({', '.join(_PRIMITIVE_TYPES)}), on "
                "the reference path (remoroo_lc/reference/walls.py) and in the Warp "
                "kernels (remoroo_lc/kernels/warp_backend.py) alike.  Either declare "
                "this obstacle as those primitives in the cell's `environment:` list, "
                "or use a path that does not consult the environment -- load_cell, "
                "validate_cell, KinematicTree and scripts/sysid_tapes.py all carry a "
                "mesh obstacle untouched.  lc will not substitute a box: an invented "
                "box is a phantom obstacle in the planner, which is worse than this "
                "refusal."
            )


@dataclass
class CellSpec:
    """A fully resolved, validated cell."""

    name: str
    spec_version: str
    source: Path
    models: tuple[ModelSpec, ...]
    tcps: tuple[TcpSpec, ...]
    environment: tuple[EnvObstacle, ...]
    collision: dict[str, Any]
    limits: dict[str, Any]
    gains: dict[str, Any]
    urdfs: dict[str, UrdfModel] = field(default_factory=dict)
    spheres: dict[str, dict[str, list[tuple[np.ndarray, float]]]] = field(default_factory=dict)

    # ---------------- derived sizing (never hardcoded anywhere else) -------- #

    @property
    def n_joints(self) -> int:
        """Total actuated joints in the cell."""
        return sum(len(m.joint_names) for m in self.models)

    @property
    def n_tcps(self) -> int:
        return len(self.tcps)

    @property
    def effector_widths(self) -> tuple[int, ...]:
        return tuple(t.effector.width for t in self.tcps)

    @property
    def action_dim(self) -> int:
        """sum over TCPs of (TASK_DIM + g_i), in config order."""
        return sum(TASK_DIM + g for g in self.effector_widths)

    @property
    def task_dim(self) -> int:
        """Stacked task-space dimension, TASK_DIM per TCP."""
        return TASK_DIM * self.n_tcps

    def action_slices(self) -> list[tuple[slice, slice]]:
        """Per TCP: (pose-delta slice, effector slice) into the action vector."""
        out, off = [], 0
        for t in self.tcps:
            pose = slice(off, off + TASK_DIM)
            eff = slice(off + TASK_DIM, off + TASK_DIM + t.effector.width)
            out.append((pose, eff))
            off += TASK_DIM + t.effector.width
        return out

    def joint_index(self) -> dict[tuple[str, str], int]:
        """(model name, joint name) -> global joint index."""
        idx, k = {}, 0
        for m in self.models:
            for jn in m.joint_names:
                idx[(m.name, jn)] = k
                k += 1
        return idx

    def joint_labels(self) -> list[str]:
        return [f"{m.name}/{jn}" for m in self.models for jn in m.joint_names]

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Global (lower, upper) position limit vectors, length n_joints."""
        lo, hi = [], []
        for m in self.models:
            urdf = self.urdfs[m.name]
            for jn in m.joint_names:
                j = urdf.by_name[jn]
                lo.append(j.lower)
                hi.append(j.upper)
        return np.asarray(lo, dtype=DTYPE), np.asarray(hi, dtype=DTYPE)

    def joint_velocity_limits(self) -> np.ndarray:
        """Global per-joint velocity limits, length n_joints."""
        override = self.limits.get("joint_velocity_override") or {}
        out = []
        for m in self.models:
            urdf = self.urdfs[m.name]
            for jn in m.joint_names:
                v = override.get(f"{m.name}/{jn}", override.get(jn, urdf.by_name[jn].velocity))
                if v is None or float(v) <= 0.0:
                    raise CellSpecError(
                        f"{self.name}: joint {m.name}/{jn} has no positive velocity limit; "
                        "set it in the URDF <limit velocity=...> or in "
                        "limits.joint_velocity_override"
                    )
                out.append(float(v))
        return np.asarray(out, dtype=DTYPE)

    def rest_posture(self) -> np.ndarray:
        """Global posture target, defaulting to each joint's mid-range."""
        lo, hi = self.joint_limits()
        q_rest = ((lo + hi) * DTYPE(0.5)).astype(DTYPE)
        k = 0
        for m in self.models:
            nm = len(m.joint_names)
            if m.rest is not None:
                q_rest[k : k + nm] = m.rest
            k += nm
        return q_rest


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _resolve(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _pose_from(block: dict[str, Any] | None, where: str) -> np.ndarray:
    if not block:
        return np.eye(4, dtype=DTYPE)
    xyz = np.asarray(block.get("xyz", [0.0, 0.0, 0.0]), dtype=DTYPE)
    if xyz.shape != (POINT_DIM,):
        raise CellSpecError(f"{where}: xyz must have {POINT_DIM} entries")
    if "quat" in block and "rpy" in block:
        raise CellSpecError(f"{where}: give either quat or rpy, not both")
    if "quat" in block:
        q = np.asarray(block["quat"], dtype=DTYPE)
        if q.shape != (QUAT_DIM,):
            raise CellSpecError(f"{where}: quat must have {QUAT_DIM} entries (w, x, y, z)")
        R = quat_to_mat(q)
    else:
        rpy = np.asarray(block.get("rpy", [0.0, 0.0, 0.0]), dtype=DTYPE)
        if rpy.shape != (POINT_DIM,):
            raise CellSpecError(f"{where}: rpy must have {POINT_DIM} entries")
        R = rpy_to_mat(rpy)
    return make_transform(R, xyz)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_limits(path: str | Path) -> dict[str, Any]:
    """Load a limits yaml."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# control rates -- the single source of truth
# --------------------------------------------------------------------------- #


class RateMismatch(CellSpecError):
    """Two declarations of the same control rate disagree.

    This is its own exception type because the failure it prevents is not a
    typo, it is a SILENT factor.  Measured on this campaign: the cell declared
    ``policy_hz: 50`` while the training worldspec declared ``25``, and nothing
    compared them.  remoroo-lc sized its chunk to 20 ms; the trainer fed it
    40 ms worth of control ticks.  Every chunk was therefore executed as a
    2x-speed sprint followed by the interpolator's deceleration runway -- a
    0.92 -> 1.53 sawtooth in the per-tick profile -- and the policy was learning
    against a plant that moved at twice the speed it had asked for.  Nothing
    raised, nothing logged, and the training curve merely looked bad.

    A rate that is declared twice is a rate that can disagree with itself.
    """


def resolve_rates(
    cell: "CellSpec",
    *,
    policy_hz: float | None = None,
    command_hz: float | None = None,
    source: str = "caller",
    require_exact_ticks: bool = False,
) -> dict[str, float]:
    """The one place a control rate is decided.  Everything else reads this.

    THE CELL WINS.  ``cell.limits["rates"]`` is the rig deployment contract --
    the rate the arm will actually be driven at when this controller ships --
    and a training run that quietly uses a different one is training against a
    plant it will never meet.  So a caller may DECLARE the rates it believes it
    is using, and this function checks that belief; it may not override it.

    To run the same rig at a second rate, point the cell at a second limits file
    (``load_cell(..., limits_path=...)``) or use its ``limits_override:`` block.
    Both are explicit, both are recorded in ``config_stamp``, and both leave
    exactly one number in play for any given controller.

    Returns policy_hz, command_hz, the exact number of control ticks per action,
    and the two periods the kernels consume (``dt_p``, ``dt_c``).
    """
    rates = (cell.limits or {}).get("rates") or {}
    out: dict[str, float] = {}
    for key, declared in (("policy_hz", policy_hz), ("command_hz", command_hz)):
        if key not in rates:
            raise RateMismatch(
                f"{cell.name}: limits.rates.{key} is not declared; "
                f"remoroo-lc cannot size a chunk without it"
            )
        own = float(rates[key])
        if own <= 0.0:
            raise RateMismatch(f"{cell.name}: limits.rates.{key} must be positive, got {own:g}")
        if declared is not None and abs(float(declared) - own) > 1e-9:
            raise RateMismatch(
                f"{cell.name}: {key} is declared twice and the two disagree -- "
                f"the cell's limits say {own:g} Hz, {source} says {float(declared):g} Hz "
                f"(a factor of {float(declared) / own:.4g}). "
                f"Whichever is wrong, one of them is silently rescaling every command: "
                f"reconcile them, or give this run its own limits file via "
                f"load_cell(limits_path=...)."
            )
        out[key] = own

    # Exactness binds on whoever CHUNKS -- a driver that runs a fixed number of
    # control ticks per action.  The controller itself only ever needs the two
    # periods, and the reference path advances by dt_c against a dt_p-long
    # chunk quite happily at a fractional ratio, so requiring it here would
    # reject cells that are fine.  `assert_drive_rate` is where it is enforced,
    # against the tick count the driver will actually use.
    ratio = out["command_hz"] / out["policy_hz"]
    ticks = int(round(ratio))
    exact = ticks >= 1 and abs(ratio - ticks) <= 1e-9
    if require_exact_ticks and not exact:
        raise RateMismatch(
            f"{cell.name}: command_hz ({out['command_hz']:g}) must be an exact integer "
            f"multiple of policy_hz ({out['policy_hz']:g}); the ratio is {ratio:g}. "
            f"A fractional ratio makes the number of control ticks behind one action "
            f"depend on accumulated float error rather than on anything declared."
        )
    out["ticks_per_action"] = float(ticks) if exact else 0.0
    out["ticks_exact"] = 1.0 if exact else 0.0
    out["dt_p"] = 1.0 / out["policy_hz"]
    out["dt_c"] = 1.0 / out["command_hz"]
    return out


def load_spheres(path: str | Path) -> dict[str, list[tuple[np.ndarray, float]]]:
    """Load a collision-sphere file in cuRobo layout.

    Accepts either the nested cuRobo form
    ``{robot_cfg: {kinematics: {collision_spheres: {link: [...]}}}}`` or the flat
    form ``{link: [{center: [x, y, z], radius: r}, ...]}``.  JSON and YAML are
    both accepted; centres are in the link frame, metres.
    """
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)  # a superset of JSON
    if not isinstance(raw, dict):
        raise CellSpecError(f"{path}: sphere file must be a mapping")
    node = raw
    for key in ("robot_cfg", "kinematics", "collision_spheres"):
        if isinstance(node, dict) and key in node:
            node = node[key]
    if not isinstance(node, dict):
        raise CellSpecError(f"{path}: could not locate a link -> spheres mapping")
    out: dict[str, list[tuple[np.ndarray, float]]] = {}
    for link, entries in sorted(node.items()):
        if entries is None:
            continue
        spheres = []
        for i, e in enumerate(entries):
            try:
                c = np.asarray(e["center"], dtype=DTYPE)
                r = float(e["radius"])
            except (KeyError, TypeError) as exc:
                raise CellSpecError(f"{path}: {link}[{i}] needs center and radius") from exc
            if c.shape != (POINT_DIM,):
                raise CellSpecError(f"{path}: {link}[{i}] center must have {POINT_DIM} entries")
            if r <= 0.0:
                raise CellSpecError(f"{path}: {link}[{i}] radius must be positive")
            spheres.append((c, r))
        if spheres:
            out[link] = spheres
    return out


def _env_obstacle(entry: dict[str, Any], i: int, base_dir: Path) -> EnvObstacle:
    """Parse one `environment[]` entry.  The only place the accepted types are decided."""
    where = f"environment[{i}]"
    ptype = entry.get("type")
    if ptype not in _ENV_TYPES:
        raise CellSpecError(f"{where}: type must be one of {_ENV_TYPES}, got {ptype!r}")
    if ptype == _MESH_TYPE:
        return _env_mesh(entry, i, base_dir)
    return _env_primitive(entry, i)


def _env_mesh(entry: dict[str, Any], i: int, base_dir: Path) -> EnvMesh:
    """Parse a `type: mesh` obstacle: file, scale and 4x4 pose, carried verbatim.

    Both `file` and `scale` are REQUIRED, and `scale` is required for a reason
    that cost real corner geometry: an STL carries no units.  corner_cell's
    obstacle mesh is authored in millimetres and declared `scale: [0.001, 0.001,
    0.001]`; defaulting a missing scale to 1.0 would be inventing a number, and
    the number it invents is wrong by 1000x -- an obstacle the size of a building
    around a 0.8 m cell.  An absent measurement must fail, not be guessed.
    """
    where = f"environment[{i}]"
    name = entry.get("name", f"env{i}")
    if not entry.get("file"):
        raise CellSpecError(
            f"{where}: a mesh obstacle needs `file:`, the path to the mesh, resolved "
            "against the cell file's directory"
        )
    mesh_path = _resolve(base_dir, str(entry["file"]))
    if not mesh_path.is_file():
        raise CellSpecError(
            f"{where}: mesh file {mesh_path} does not exist.  Nothing in lc opens this "
            "file -- it is carried for the consumers that do -- which is exactly why it "
            "is checked here: a broken path in a carried measurement would otherwise "
            "surface only downstream, in whatever tool finally tried to load the cell's "
            "obstacle world"
        )
    if "scale" not in entry:
        raise CellSpecError(
            f"{where}: a mesh obstacle needs an explicit `scale:` (3 entries).  A mesh "
            "file has no units, so there is no safe default: millimetres vs metres is a "
            "1000x error in the position AND the size of the obstacle.  Write "
            "[1.0, 1.0, 1.0] if the file is already in metres"
        )
    scale = np.asarray(entry["scale"], dtype=DTYPE)
    if scale.shape != (POINT_DIM,):
        raise CellSpecError(f"{where}: mesh scale must have {POINT_DIM} entries")
    if not bool(np.all(scale > 0.0)):
        raise CellSpecError(
            f"{where}: mesh scale must be positive on every axis, got {scale.tolist()}"
        )
    return EnvMesh(
        name=name, pose=_pose_from(entry, where), mesh_path=mesh_path, scale=scale
    )


def _env_primitive(entry: dict[str, Any], i: int) -> EnvPrimitive:
    # `_env_obstacle` has already checked that `type` is one of _PRIMITIVE_TYPES;
    # this function is not a second door and must not grow one.
    where = f"environment[{i}]"
    ptype = entry["type"]
    name = entry.get("name", f"env{i}")
    if ptype == "plane":
        normal = np.asarray(entry.get("normal", [0.0, 0.0, 1.0]), dtype=DTYPE)
        nn = float(np.linalg.norm(normal))
        if nn < 1.0e-9:
            raise CellSpecError(f"{where}: plane normal is degenerate")
        normal = normal / DTYPE(nn)
        point = np.asarray(entry.get("point", [0.0, 0.0, 0.0]), dtype=DTYPE)
        pose = make_transform(np.eye(POINT_DIM, dtype=DTYPE), point)
        return EnvPrimitive(name, ptype, pose, normal.astype(DTYPE))
    pose = _pose_from(entry, where)
    if ptype == "box":
        dims = np.asarray(entry["dims"], dtype=DTYPE) * DTYPE(0.5)
        if dims.shape != (POINT_DIM,):
            raise CellSpecError(f"{where}: box dims must have {POINT_DIM} entries")
    elif ptype == "sphere":
        dims = np.asarray([float(entry["radius"])], dtype=DTYPE)
    else:  # cylinder, axis along local z
        dims = np.asarray(
            [float(entry["radius"]), float(entry["height"]) * 0.5], dtype=DTYPE
        )
    return EnvPrimitive(name, ptype, pose, dims)


def config_stamp(cell: "CellSpec") -> dict:
    """Identity of the resolved configuration, for stamping artifacts.

    A dataset or a trained policy is only valid against the controller that
    produced it, and this package is still moving: the tracking law, the feedback
    gain and the joint-velocity governor all changed materially in one day.  A
    stamp makes that visible instead of silent -- an artifact whose stamp does not
    match the current build is suspect by construction, which is the only way to
    catch a controller change that alters behaviour without erroring.

    The hash covers everything that changes what the controller DOES: the
    resolved limits and gains dicts, and the cell's own structure (joint order,
    tool frames, effector widths, rest posture, sphere and pair counts).  It
    deliberately does NOT cover file paths or comments, so moving a config
    without changing it keeps the stamp -- and it deliberately DOES cover
    `remoroo_lc.__version__`, because a code change with identical config is
    exactly the case that would otherwise slip through.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    from remoroo_lc import __version__  # noqa: PLC0415

    payload = {
        "version": __version__,
        "spec": cell.spec_version,
        "limits": cell.limits,
        "gains": cell.gains,
        "joints": list(cell.joint_labels()),
        "tcps": [(t.name, t.model, t.frame, t.effector.width) for t in cell.tcps],
        "rest": [float(v) for v in cell.rest_posture()],
        "n_spheres": sum(len(v) for m in cell.spheres.values() for v in m.values()),
        "environment": [(e.name, e.ptype) for e in cell.environment],
        "collision": cell.collision,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return {
        "version": __version__,
        "cell": cell.name,
        "stamp": hashlib.sha256(blob).hexdigest()[:16],
    }


def load_cell(
    path: str | Path,
    limits_path: str | Path | None = None,
    gains_path: str | Path | None = None,
) -> CellSpec:
    """Load, resolve and validate a cell file.

    `limits_path` / `gains_path` override whatever the cell file references.
    Relative paths inside the cell file resolve against the cell file's directory.
    """
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base_dir = path.parent

    version = str(raw.get("lc_spec_version", raw.get("schema_version", "")))
    if not version:
        raise CellSpecError(f"{path}: missing lc_spec_version")
    if version.split(".")[0] not in _SUPPORTED_MAJORS:
        raise CellSpecError(
            f"{path}: lc_spec_version {version} is not supported by this loader "
            f"(supports major {_SUPPORTED_MAJORS})"
        )

    # ---- limits / gains ---------------------------------------------------- #
    limits: dict[str, Any] = {}
    if limits_path is not None:
        limits = load_limits(limits_path)
    elif "limits" in raw:
        limits = (
            load_limits(_resolve(base_dir, raw["limits"]))
            if isinstance(raw["limits"], str)
            else dict(raw["limits"])
        )
    if isinstance(raw.get("limits_override"), dict):
        limits = _deep_merge(limits, raw["limits_override"])

    gains: dict[str, Any] = {}
    if gains_path is not None:
        gains = load_limits(gains_path)
    elif "gains" in raw:
        gains = (
            load_limits(_resolve(base_dir, raw["gains"]))
            if isinstance(raw["gains"], str)
            else dict(raw["gains"])
        )

    # ---- models ------------------------------------------------------------ #
    if not raw.get("models"):
        raise CellSpecError(f"{path}: a cell needs at least one model")
    models: list[ModelSpec] = []
    urdfs: dict[str, UrdfModel] = {}
    spheres: dict[str, dict[str, list[tuple[np.ndarray, float]]]] = {}
    for i, m in enumerate(raw["models"]):
        where = f"models[{i}]"
        name = m.get("name")
        if not name:
            raise CellSpecError(f"{where}: missing name")
        if any(x.name == name for x in models):
            raise CellSpecError(f"{where}: duplicate model name {name!r}")
        urdf_path = _resolve(base_dir, m["urdf"])
        urdf = load_urdf(urdf_path, name=name)
        urdfs[name] = urdf

        locked = {str(k): float(v) for k, v in (m.get("locked_joints") or {}).items()}
        for jn in locked:
            if jn not in urdf.by_name:
                raise CellSpecError(f"{where}: locked joint {jn!r} not in {urdf_path.name}")

        if "joints" in m:
            # Present but empty is an error, not "please guess": a model with no
            # actuated joints is almost always a typo, and silently falling back
            # to the URDF order would hide it.
            joint_names = tuple(str(j) for j in (m["joints"] or ()))
            for jn in joint_names:
                if jn not in urdf.by_name:
                    raise CellSpecError(f"{where}: joint {jn!r} not in {urdf_path.name}")
                if not urdf.by_name[jn].moves:
                    raise CellSpecError(f"{where}: joint {jn!r} is fixed and cannot be actuated")
                if jn in locked:
                    raise CellSpecError(f"{where}: joint {jn!r} is both actuated and locked")
        else:
            joint_names = tuple(
                jn
                for jn in urdf.ordered_moving_joints()
                if jn not in locked and urdf.by_name[jn].mimic is None
            )
        if not joint_names:
            raise CellSpecError(f"{where}: model {name!r} has no actuated joints")

        rest = None
        if m.get("rest") is not None:
            rest = np.asarray(m["rest"], dtype=DTYPE)
            if rest.shape != (len(joint_names),):
                raise CellSpecError(
                    f"{where}: rest has {rest.shape[0]} entries, expected {len(joint_names)}"
                )

        sph_path = _resolve(base_dir, m["spheres"]) if m.get("spheres") else None
        if sph_path is not None:
            loaded = load_spheres(sph_path)
            unknown = sorted(set(loaded) - set(urdf.links))
            if unknown:
                raise CellSpecError(
                    f"{where}: sphere file names links absent from the URDF: {unknown}"
                )
            spheres[name] = loaded

        models.append(
            ModelSpec(
                name=name,
                urdf_path=urdf_path,
                base=_pose_from(m.get("base"), where),
                spheres_path=sph_path,
                joint_names=joint_names,
                locked=locked,
                rest=rest,
            )
        )

    # ---- TCPs -------------------------------------------------------------- #
    if not raw.get("tcps"):
        raise CellSpecError(f"{path}: a cell needs at least one tcp")
    default_w = float((limits.get("diffik") or {}).get("w_thresh_default", 0.01))
    tcps: list[TcpSpec] = []
    for i, t in enumerate(raw["tcps"]):
        where = f"tcps[{i}]"
        name = t.get("name")
        if not name:
            raise CellSpecError(f"{where}: missing name")
        if any(x.name == name for x in tcps):
            raise CellSpecError(f"{where}: duplicate tcp name {name!r}")
        model_name = t.get("model")
        if model_name not in urdfs:
            raise CellSpecError(f"{where}: unknown model {model_name!r}")
        frame = t.get("frame")
        if frame not in urdfs[model_name].links:
            raise CellSpecError(f"{where}: frame {frame!r} is not a link of model {model_name!r}")
        eff_raw = t.get("effector") or {}
        effector = EffectorSpec(
            kind=str(eff_raw.get("kind", "none")),
            width=int(eff_raw.get("width", 0)),
            rate_limit=float(
                eff_raw.get("rate_limit", (limits.get("effector") or {}).get("rate_limit", 4.0))
            ),
            default=tuple(float(v) for v in (eff_raw.get("default") or ())),
        )
        damping = t.get("damping") or {}
        tcps.append(
            TcpSpec(
                name=name,
                model=model_name,
                frame=frame,
                offset=_pose_from(t.get("offset"), where),
                effector=effector,
                w_thresh=float(damping.get("w_thresh", default_w)),
            )
        )

    environment = tuple(
        _env_obstacle(e, i, base_dir) for i, e in enumerate(raw.get("environment") or [])
    )

    collision = {
        "self_pairs": True,
        "cross_model": True,
        "environment": True,
        "ignore_pairs": [],
        "max_pairs": None,
    }
    collision.update(raw.get("collision") or {})
    collision["ignore_pairs"] = [
        tuple(sorted(str(x) for x in pair)) for pair in collision["ignore_pairs"]
    ]

    cell = CellSpec(
        name=str(raw.get("name", path.stem)),
        spec_version=version,
        source=path,
        models=tuple(models),
        tcps=tuple(tcps),
        environment=environment,
        collision=collision,
        limits=limits,
        gains=gains,
        urdfs=urdfs,
        spheres=spheres,
    )
    validate_cell(cell)
    return cell


def validate_cell(cell: CellSpec) -> None:
    """Assert the invariants the rest of the pipeline relies on.

    Raises CellSpecError with an actionable message.  Called by load_cell; also
    exported so that `remoroo setup` can validate what it emits before shipping.
    """
    if cell.spec_version.split(".")[0] not in _SUPPORTED_MAJORS:
        raise CellSpecError(f"{cell.name}: unsupported lc_spec_version {cell.spec_version}")
    if cell.n_joints == 0:
        raise CellSpecError(f"{cell.name}: cell has zero actuated joints")
    if cell.n_tcps == 0:
        raise CellSpecError(f"{cell.name}: cell has zero TCPs")

    for m in cell.models:
        urdf = cell.urdfs[m.name]
        for jn in m.joint_names:
            j = urdf.by_name[jn]
            if j.jtype != "continuous" and j.upper <= j.lower:
                raise CellSpecError(
                    f"{cell.name}: joint {m.name}/{jn} has empty range "
                    f"[{j.lower}, {j.upper}]"
                )
        # Every moving joint must be either actuated, locked, or a mimic.
        for jn in urdf.ordered_moving_joints():
            known = jn in m.joint_names or jn in m.locked or urdf.by_name[jn].mimic is not None
            if not known:
                raise CellSpecError(
                    f"{cell.name}: model {m.name} joint {jn!r} is neither actuated nor locked; "
                    "add it to models[].joints or models[].locked_joints"
                )

    # Every TCP's chain must consist of joints this cell knows about.
    for t in cell.tcps:
        urdf = cell.urdfs[t.model]
        model = next(m for m in cell.models if m.name == t.model)
        for jn in urdf.chain_to(t.frame):
            j = urdf.by_name[jn]
            if j.moves and jn not in model.joint_names and jn not in model.locked:
                raise CellSpecError(
                    f"{cell.name}: tcp {t.name} depends on joint {jn!r} which is not actuated "
                    "and not locked"
                )
        if t.w_thresh <= 0.0:
            raise CellSpecError(f"{cell.name}: tcp {t.name} has non-positive w_thresh")

    try:
        cell.joint_velocity_limits()
    except CellSpecError:
        raise
    _ = cell.rest_posture()

    lo, hi = cell.joint_limits()
    q_rest = cell.rest_posture()
    below = np.where(q_rest < lo)[0]
    above = np.where(q_rest > hi)[0]
    if below.size or above.size:
        labels = cell.joint_labels()
        bad = [labels[i] for i in np.concatenate([below, above])]
        raise CellSpecError(f"{cell.name}: rest posture outside joint limits for {bad}")

    # A mesh obstacle is VALID cell content and is not refused here: a validator
    # that rejected it would put this package's missing capability in front of the
    # installer's measurement, and would keep the whole cell -- joints, TCPs,
    # rates, limits -- out of reach of every consumer that never touches the
    # environment.  The collision builders refuse it by name instead; see
    # `refuse_mesh_obstacles`.  This loop is still the door for a CellSpec that
    # was hand-built rather than loaded (validate_cell is exported for exactly
    # that, so `remoroo setup` can check what it emits before shipping it).
    for prim in cell.environment:
        if prim.ptype not in _ENV_TYPES:
            raise CellSpecError(f"{cell.name}: unknown environment type {prim.ptype!r}")
