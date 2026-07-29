#!/usr/bin/env python3
"""Conservatively reduce a collision-sphere set to a size the filter can afford.

`remoroo setup` fits spheres to approximate the robot's SHAPE, and does it well:
the reference bimanual cell gets 1480 of them across 32 links.  The collision
filter does not need that resolution, and cannot pay for it -- pairs go as the
product of the two sides, so 1480 spheres is 865,000 pairs, and the whole
constraint budget at 250 Hz is a few hundred.

This merges spheres until the count fits, and the merge is CONSERVATIVE: the
enclosing sphere of two spheres contains both, so the reduced set's union
contains the original's.  Anything the full set would have called a collision,
the reduced set still calls a collision.  The cost is false positives -- the
robot thinks it is fatter than it is -- and the report prints how much fatter so
that cost is visible rather than assumed.

Never merges across links: a sphere belongs to the link it is rigidly attached
to, and merging across links would be merging things that move apart.

    python scripts/decimate_spheres.py in.yml --target 120 -o out.json
    python scripts/decimate_spheres.py in.yml --target 120 --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.schema import load_spheres  # noqa: E402


def enclosing(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float):
    """The smallest sphere containing both inputs.  Exact, not a bound."""
    d = float(np.linalg.norm(c2 - c1))
    if d + r2 <= r1:
        return c1.copy(), r1
    if d + r1 <= r2:
        return c2.copy(), r2
    R = 0.5 * (d + r1 + r2)
    c = c1 + (c2 - c1) * ((R - r1) / d) if d > 1e-12 else c1.copy()
    return c, R


def decimate_link(
    spheres: list[tuple[np.ndarray, float]], max_radius: float, floor: int = 1
):
    """Greedily merge while the result stays under `max_radius`.

    The control is the RADIUS, not the count.  Budgeting by count instead asks a
    370 mm upper arm and an 80 mm camera to give up the same fraction, and the
    upper arm answers by inflating its 24 mm spheres to 170 mm -- a robot that
    believes it is the size of a beach ball and whose filter therefore never
    stops firing.  Capping the radius bounds how much fatter the robot gets and
    lets the count fall out of the geometry, which is where it belongs.

    At each step it merges the pair whose enclosing sphere is smallest, which is
    the pair that costs the least accuracy.  O(n^2) per step and n is per-link
    (tens), so this is fast enough to run from a Makefile and never in a control
    loop.
    """
    items = [(np.asarray(c, dtype=np.float64), float(r)) for c, r in spheres]
    while len(items) > floor:
        best = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c, r = enclosing(items[i][0], items[i][1], items[j][0], items[j][1])
                if best is None or r < best[0]:
                    best = (r, c, i, j)
        if best is None or best[0] > max_radius:
            break
        r, c, i, j = best
        items = [it for k, it in enumerate(items) if k not in (i, j)] + [(c, r)]
    return items


def verify_containment(original, reduced) -> float:
    """Every original sphere must lie inside some reduced sphere.

    Returns the worst protrusion in metres; anything above zero means the
    reduction is NOT conservative and the safety argument does not hold.
    """
    worst = -np.inf
    for c0, r0 in original:
        best = np.inf
        for c1, r1 in reduced:
            # How far the original pokes out of this reduced sphere.
            best = min(best, float(np.linalg.norm(np.asarray(c0) - c1)) + r0 - r1)
        worst = max(worst, best)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument(
        "--max-growth", type=float, default=0.02,
        help="metres a link's largest sphere may grow by (the conservatism cost)",
    )
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--verify", action="store_true", help="prove containment (slow)")
    ap.add_argument("--min-per-link", type=int, default=1)
    a = ap.parse_args()

    per_link = load_spheres(a.source)
    total = sum(len(v) for v in per_link.values())
    print(
        f"{a.source.name}: {total} spheres across {len(per_link)} links; "
        f"max radius growth {a.max_growth * 1e3:.0f} mm"
    )

    names = sorted(per_link)
    out: dict[str, list[dict]] = {}
    worst_growth = 0.0
    worst_link = ""
    for name in names:
        original = per_link[name]
        cap = max(r for _, r in original) + a.max_growth
        reduced = decimate_link(original, cap, floor=a.min_per_link)
        r_before = max(r for _, r in original)
        r_after = max(r for _, r in reduced)
        growth = r_after - r_before
        if growth > worst_growth:
            worst_growth, worst_link = growth, name
        if a.verify:
            protrusion = verify_containment(original, reduced)
            if protrusion > 1e-9:
                print(
                    f"  FAIL {name}: an original sphere protrudes {protrusion * 1e3:.3f} mm",
                    file=sys.stderr,
                )
                return 1
        out[name] = [
            {"center": [round(float(x), 6) for x in c], "radius": round(float(r), 6)}
            for c, r in reduced
        ]
        print(f"  {name:<28} {len(original):4d} -> {len(reduced):3d}   "
              f"max radius {r_before * 1e3:5.1f} -> {r_after * 1e3:5.1f} mm")

    kept = sum(len(v) for v in out.values())
    print(f"\nkept {kept} spheres; largest radius growth {worst_growth * 1e3:.1f} mm "
          f"on {worst_link}")
    if a.verify:
        print("containment verified: every original sphere lies inside a reduced one")
    else:
        print("run with --verify to prove containment before trusting this on hardware")

    if a.out:
        doc = {"robot_cfg": {"kinematics": {"collision_spheres": out}}}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
