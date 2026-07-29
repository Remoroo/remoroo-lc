#!/usr/bin/env python3
"""Generate the placeholder URDFs and collision-sphere files under assets/.

These are placeholders in the sense that their link lengths are invented, not in
the sense that they are toys: every file this writes is an exact, validating
instance of the contract in CELL_SPEC.md, carries inertial and collision geometry
so third-party tools (MuJoCo, and therefore the mink oracle) can load it
unchanged, and is replaced one-for-one by what `remoroo setup` emits for a real
cell.  Nothing downstream of here knows the difference.

Sphere sets are fitted mechanically: spheres are laid along each link's segment
from its own origin to each child joint origin.  A production sphere set from
`remoroo setup` drops into the same file, same format.

Run:  python scripts/gen_placeholder_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = ROOT / "assets" / "urdf"
SPHERE_DIR = ROOT / "assets" / "spheres"

# (joint name, axis, origin xyz, child link, lower, upper)
Rev = tuple[str, str, tuple[float, float, float], str, float, float]

CHAIN_6DOF: list[Rev] = [
    ("joint1", "0 0 1", (0.0, 0.0, 0.150), "link1", -3.05, 3.05),
    ("joint2", "0 1 0", (0.0, 0.0, 0.100), "link2", -2.09, 2.09),
    ("joint3", "0 1 0", (0.0, 0.0, 0.300), "link3", -0.19, 3.10),
    ("joint4", "0 0 1", (0.0, 0.0, 0.100), "link4", -3.05, 3.05),
    ("joint5", "0 1 0", (0.0, 0.0, 0.250), "link5", -3.10, 1.69),
    ("joint6", "0 0 1", (0.0, 0.0, 0.080), "link6", -3.05, 3.05),
]

CHAIN_7DOF: list[Rev] = [
    ("joint1", "0 0 1", (0.0, 0.0, 0.150), "link1", -2.90, 2.90),
    ("joint2", "0 1 0", (0.0, 0.0, 0.050), "link2", -2.09, 2.09),
    ("joint3", "0 0 1", (0.0, 0.0, 0.250), "link3", -2.90, 2.90),
    ("joint4", "0 1 0", (0.0, 0.0, 0.050), "link4", -0.19, 2.90),
    ("joint5", "0 0 1", (0.0, 0.0, 0.250), "link5", -2.90, 2.90),
    ("joint6", "0 1 0", (0.0, 0.0, 0.050), "link6", -3.10, 1.69),
    ("joint7", "0 0 1", (0.0, 0.0, 0.100), "link7", -3.05, 3.05),
]

BRANCH_5DOF: list[Rev] = [
    ("j1", "0 1 0", (0.0, 0.0, 0.240), "link1", -2.40, 2.40),
    ("j2", "0 1 0", (0.0, 0.0, 0.240), "link2", -0.19, 2.90),
    ("j3", "0 0 1", (0.0, 0.0, 0.060), "link3", -3.05, 3.05),
    ("j4", "0 1 0", (0.0, 0.0, 0.200), "link4", -3.10, 1.69),
    ("j5", "0 0 1", (0.0, 0.0, 0.060), "link5", -3.05, 3.05),
]

VELOCITY = 3.14
EFFORT = 50.0
LINK_RADIUS = 0.045
SPHERE_RADIUS = 0.045


def _inertial(mass: float = 1.0, com_z: float = 0.05) -> str:
    return (
        f'    <inertial>\n'
        f'      <origin xyz="0 0 {com_z:.4f}" rpy="0 0 0"/>\n'
        f'      <mass value="{mass:.4f}"/>\n'
        f'      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>\n'
        f'    </inertial>\n'
    )


def _geom(length: float, radius: float = LINK_RADIUS) -> str:
    if length <= 1e-6:
        return ""
    half = length * 0.5
    body = (
        f'      <origin xyz="0 0 {half:.4f}" rpy="0 0 0"/>\n'
        f"      <geometry><cylinder radius=\"{radius:.4f}\" length=\"{length:.4f}\"/></geometry>\n"
    )
    return f"    <collision>\n{body}    </collision>\n    <visual>\n{body}    </visual>\n"


def _link(name: str, length: float, mass: float = 1.0) -> str:
    return (
        f'  <link name="{name}">\n'
        + _inertial(mass, com_z=max(length * 0.5, 0.01))
        + _geom(length)
        + "  </link>\n"
    )


def _joint(
    name: str, jtype: str, parent: str, child: str, xyz, axis: str | None, lo=0.0, hi=0.0
) -> str:
    out = f'  <joint name="{name}" type="{jtype}">\n'
    out += f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
    out += f'    <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" rpy="0 0 0"/>\n'
    if axis is not None:
        out += f'    <axis xyz="{axis}"/>\n'
        out += (
            f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" '
            f'velocity="{VELOCITY}" effort="{EFFORT}"/>\n'
        )
    out += "  </joint>\n"
    return out


def _serial_urdf(robot_name: str, chain: list[Rev], tool_offset: float = 0.090) -> str:
    """A single serial chain rooted at base_link, ending in a fixed tool0."""
    parts = [f'<?xml version="1.0"?>\n<robot name="{robot_name}">\n']
    # Segment length of a link = distance to its own child's joint origin.
    lengths = [chain[0][2][2]] + [chain[i + 1][2][2] for i in range(len(chain) - 1)]
    lengths.append(tool_offset)
    parts.append(_link("base_link", lengths[0], mass=2.0))
    prev = "base_link"
    for i, (jn, axis, xyz, child, lo, hi) in enumerate(chain):
        parts.append(_joint(jn, "revolute", prev, child, xyz, axis, lo, hi))
        parts.append(_link(child, lengths[i + 1]))
        prev = child
    parts.append(_joint("tool_joint", "fixed", prev, "tool0", (0.0, 0.0, tool_offset), None))
    parts.append(_link("tool0", 0.0, mass=0.2))
    parts.append("</robot>\n")
    return "".join(parts)


def _branched_urdf() -> str:
    """One 1-DOF trunk joint feeding two 5-DOF branches.

    Each branch's TCP therefore depends on 6 joints, one of which -- the trunk --
    is shared with the other branch.  n = 11 while the stacked task space is
    2 * 6 = 12, so the two TCPs cannot both be satisfied exactly and the solver
    has to arbitrate.  That is the point of this asset.
    """
    parts = ['<?xml version="1.0"?>\n<robot name="branched_trunk">\n']
    parts.append(_link("base_link", 0.200, mass=3.0))
    parts.append(_joint("trunk", "revolute", "base_link", "trunk_link", (0, 0, 0.200), "0 0 1",
                        -2.60, 2.60))
    parts.append(_link("trunk_link", 0.120, mass=2.0))
    for side, sign in (("l", 1.0), ("r", -1.0)):
        prev = "trunk_link"
        lengths = [BRANCH_5DOF[i + 1][2][2] for i in range(len(BRANCH_5DOF) - 1)] + [0.090]
        for i, (jn, axis, xyz, child, lo, hi) in enumerate(BRANCH_5DOF):
            origin = (xyz[0], sign * 0.170, xyz[2] * 0.5) if i == 0 else xyz
            parts.append(
                _joint(f"{side}_{jn}", "revolute", prev, f"{side}_{child}", origin, axis, lo, hi)
            )
            parts.append(_link(f"{side}_{child}", lengths[i]))
            prev = f"{side}_{child}"
        parts.append(
            _joint(f"{side}_tool_joint", "fixed", prev, f"{side}_tool0", (0, 0, 0.090), None)
        )
        parts.append(_link(f"{side}_tool0", 0.0, mass=0.2))
    parts.append("</robot>\n")
    return "".join(parts)


def _spheres_for(urdf_path: Path) -> dict:
    """Lay spheres along each link's segments to its children."""
    sys.path.insert(0, str(ROOT / "src"))
    from remoroo_lc.urdf import load_urdf  # noqa: PLC0415

    model = load_urdf(urdf_path)
    out: dict[str, list[dict]] = {}
    for link in model.ordered_links():
        seg_ends = []
        for child in model.children[link]:
            j = model.by_name[model.parent_joint[child]]
            seg_ends.append(j.origin[:3, 3])
        if not seg_ends:
            seg_ends = [[0.0, 0.0, 0.0]]
        spheres = []
        for end in seg_ends:
            end = [float(v) for v in end]
            n_s = 2 if max(abs(v) for v in end) > 0.12 else 1
            for k in range(n_s):
                t = (k + 0.5) / n_s
                spheres.append(
                    {
                        "center": [round(v * t, 6) for v in end],
                        "radius": SPHERE_RADIUS,
                    }
                )
        # Deduplicate identical centres (a link with several identical children).
        seen, uniq = set(), []
        for s in spheres:
            key = tuple(s["center"])
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        out[link] = uniq
    return {"robot_cfg": {"kinematics": {"urdf_path": urdf_path.name,
                                         "collision_spheres": out}}}


def main() -> int:
    URDF_DIR.mkdir(parents=True, exist_ok=True)
    SPHERE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, text in (
        ("chain6.urdf", _serial_urdf("chain6", CHAIN_6DOF)),
        ("chain7.urdf", _serial_urdf("chain7", CHAIN_7DOF)),
        ("branched_trunk.urdf", _branched_urdf()),
    ):
        p = URDF_DIR / fname
        p.write_text(text)
        written.append(p)
    for p in list(written):
        sp = SPHERE_DIR / (p.stem + "_spheres.json")
        sp.write_text(json.dumps(_spheres_for(p), indent=2) + "\n")
        written.append(sp)
    for p in written:
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
