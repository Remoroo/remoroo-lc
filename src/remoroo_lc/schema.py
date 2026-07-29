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
from typing import Any

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


class CellSpecError(ValueError):
    """Raised when a cell file does not satisfy the contract."""


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


@dataclass
class CellSpec:
    """A fully resolved, validated cell."""

    name: str
    spec_version: str
    source: Path
    models: tuple[ModelSpec, ...]
    tcps: tuple[TcpSpec, ...]
    environment: tuple[EnvPrimitive, ...]
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


def _env_primitive(entry: dict[str, Any], i: int) -> EnvPrimitive:
    where = f"environment[{i}]"
    ptype = entry.get("type")
    if ptype not in _PRIMITIVE_TYPES:
        raise CellSpecError(f"{where}: type must be one of {_PRIMITIVE_TYPES}, got {ptype!r}")
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
        _env_primitive(e, i) for i, e in enumerate(raw.get("environment") or [])
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

    for prim in cell.environment:
        if prim.ptype not in _PRIMITIVE_TYPES:
            raise CellSpecError(f"{cell.name}: unknown environment type {prim.ptype!r}")
