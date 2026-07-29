"""Minimal URDF reader.

Deliberately internal and stdlib-only (xml.etree + numpy) so that the core has no
parser dependency to license-audit or version-pin.  It reads exactly what a
controller needs -- the kinematic tree -- and ignores everything it does not:
inertial, visual and collision geometry, materials, transmissions, gazebo tags.

Handles arbitrary trees: any branching factor, fixed joints anywhere, chains that
share a common trunk, and any joint ordering in the file.  Joint ordering in the
parsed model is a deterministic depth-first walk with siblings sorted by joint
name, so two readers of the same file always agree regardless of file order.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from remoroo_lc.constants import DTYPE, POINT_DIM
from remoroo_lc.spatial import make_transform, rpy_to_mat

MOVING_TYPES = ("revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    jtype: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4, parent link frame -> joint frame
    axis: np.ndarray  # 3, unit, in the joint frame
    lower: float
    upper: float
    velocity: float
    effort: float
    mimic: tuple | None = None  # (source_joint, multiplier, offset)

    @property
    def moves(self) -> bool:
        return self.jtype in MOVING_TYPES


@dataclass
class UrdfModel:
    """A parsed URDF: links, joints, and the derived tree topology."""

    name: str
    path: Path
    links: list[str]
    joints: list[UrdfJoint]
    root: str
    parent_joint: dict[str, str] = field(default_factory=dict)  # link -> joint name
    children: dict[str, list[str]] = field(default_factory=dict)  # link -> child links
    by_name: dict[str, UrdfJoint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_name = {j.name: j for j in self.joints}
        self.parent_joint = {j.child: j.name for j in self.joints}
        self.children = {ln: [] for ln in self.links}
        for j in self.joints:
            self.children[j.parent].append(j.child)
        for ln in self.links:
            self.children[ln].sort(key=lambda c: self.parent_joint[c])

    def ordered_links(self) -> list[str]:
        """Links in a deterministic parent-before-child order."""
        out: list[str] = []
        stack = [self.root]
        while stack:
            ln = stack.pop()
            out.append(ln)
            stack.extend(reversed(self.children[ln]))
        return out

    def ordered_moving_joints(self) -> list[str]:
        """Moving joint names in deterministic depth-first order."""
        return [
            self.parent_joint[ln]
            for ln in self.ordered_links()
            if ln in self.parent_joint and self.by_name[self.parent_joint[ln]].moves
        ]

    def chain_to(self, link: str) -> list[str]:
        """Joint names from the root down to `link`, root-first."""
        if link not in self.links:
            raise KeyError(f"{self.name}: no link named {link!r}")
        out: list[str] = []
        cur = link
        while cur in self.parent_joint:
            jn = self.parent_joint[cur]
            out.append(jn)
            cur = self.by_name[jn].parent
        return list(reversed(out))


def _parse_origin(el: ET.Element | None) -> np.ndarray:
    if el is None:
        return np.eye(4, dtype=DTYPE)
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
    return make_transform(rpy_to_mat(rpy), np.asarray(xyz, dtype=DTYPE))


def _parse_axis(el: ET.Element | None) -> np.ndarray:
    raw = [1.0, 0.0, 0.0] if el is None else [float(v) for v in (el.get("xyz") or "1 0 0").split()]
    a = np.asarray(raw, dtype=DTYPE)
    n = float(np.linalg.norm(a))
    if n < 1.0e-9:
        raise ValueError("degenerate joint axis")
    return (a / DTYPE(n)).astype(DTYPE)


def load_urdf(path: str | Path, name: str | None = None) -> UrdfModel:
    """Parse a URDF file into a UrdfModel."""
    path = Path(path)
    root_el = ET.parse(path).getroot()
    if root_el.tag != "robot":
        raise ValueError(f"{path}: root element is <{root_el.tag}>, expected <robot>")

    links = [el.get("name") for el in root_el.findall("link")]
    if any(ln is None for ln in links):
        raise ValueError(f"{path}: a <link> is missing its name attribute")

    joints: list[UrdfJoint] = []
    for el in root_el.findall("joint"):
        jname = el.get("name")
        jtype = el.get("type")
        if jname is None or jtype is None:
            raise ValueError(f"{path}: a <joint> is missing name or type")
        parent_el, child_el = el.find("parent"), el.find("child")
        if parent_el is None or child_el is None:
            raise ValueError(f"{path}: joint {jname!r} is missing parent or child")
        parent, child = parent_el.get("link"), child_el.get("link")
        lim = el.find("limit")
        if jtype == "continuous":
            lower, upper = -np.pi, np.pi
        elif lim is not None:
            lower = float(lim.get("lower", "0"))
            upper = float(lim.get("upper", "0"))
        elif jtype in MOVING_TYPES:
            raise ValueError(f"{path}: joint {jname!r} of type {jtype} has no <limit>")
        else:
            lower = upper = 0.0
        vel = float(lim.get("velocity", "0")) if lim is not None else 0.0
        eff = float(lim.get("effort", "0")) if lim is not None else 0.0
        mim_el = el.find("mimic")
        mimic = None
        if mim_el is not None:
            mimic = (
                mim_el.get("joint"),
                float(mim_el.get("multiplier", "1")),
                float(mim_el.get("offset", "0")),
            )
        joints.append(
            UrdfJoint(
                name=jname,
                jtype=jtype,
                parent=parent,
                child=child,
                origin=_parse_origin(el.find("origin")),
                axis=_parse_axis(el.find("axis")) if jtype in MOVING_TYPES else _parse_axis(None),
                lower=lower,
                upper=upper,
                velocity=vel,
                effort=eff,
                mimic=mimic,
            )
        )

    link_set = set(links)
    for j in joints:
        for ln in (j.parent, j.child):
            if ln not in link_set:
                raise ValueError(f"{path}: joint {j.name!r} references unknown link {ln!r}")
    child_links = {j.child for j in joints}
    roots = sorted(link_set - child_links)
    if len(roots) != 1:
        raise ValueError(f"{path}: expected exactly one root link, found {roots}")
    if len({j.child for j in joints}) != len(joints):
        raise ValueError(f"{path}: a link has more than one parent joint (not a tree)")

    model = UrdfModel(
        name=name or root_el.get("name") or path.stem,
        path=path,
        links=sorted(link_set),
        joints=joints,
        root=roots[0],
    )
    # Reachability check catches cycles disconnected from the root.
    if len(model.ordered_links()) != len(links):
        raise ValueError(f"{path}: kinematic graph is not a single connected tree")
    return model


def link_axis_origin(
    T_parent_link: np.ndarray, joint: UrdfJoint
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame joint origin and axis given the parent link's world transform."""
    T_joint = T_parent_link @ joint.origin
    p = T_joint[:POINT_DIM, POINT_DIM]
    z = T_joint[:POINT_DIM, :POINT_DIM] @ joint.axis
    return p.astype(DTYPE), z.astype(DTYPE)
