#!/usr/bin/env python3
"""Fail if any installed dependency is outside the permissive allowlist.

This is a hard gate, not a report.  remoroo-lc ships to customer sites inside a
commercial product, so a copyleft or non-commercial dependency anywhere in the
runtime or dev tree is a blocker, and "we'll swap it later" is not a plan once a
kernel has been written against it.

The check reads installed distribution metadata rather than parsing pyproject,
because what matters is what is actually importable in the environment the tests
and the kernels run in -- including transitive dependencies nobody chose.

Usage:
    python scripts/license_check.py                 # check everything installed
    python scripts/license_check.py --runtime-only  # just remoroo-lc's own deps
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys

#: SPDX-ish identifiers we accept.  Permissive only.
ALLOWED = {
    "MIT",
    "MIT LICENSE",
    "MIT-0",
    "BSD",
    "BSD LICENSE",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "APACHE-2.0",
    "APACHE 2.0",
    "APACHE SOFTWARE LICENSE",
    "APACHE LICENSE 2.0",
    "ISC",
    "ISC LICENSE",
    "ZLIB",
    "PYTHON SOFTWARE FOUNDATION LICENSE",
    "PSF-2.0",
    "UNLICENSE",
    "0BSD",
}

#: Substrings that are always a failure, whatever else the metadata says.
DENY_SUBSTRINGS = (
    "GPL",  # also catches LGPL / AGPL; "LGPL" is intentionally not carved out
    "NVIDIA",
    "NON-COMMERCIAL",
    "NONCOMMERCIAL",
    "CC BY-NC",
    "PROPRIETARY",
    "SSPL",
    "BUSL",
    "COMMONS CLAUSE",
    "ELASTIC LICENSE",
)

#: Distributions that ship no usable license metadata but whose license is known.
#: Every entry needs a source, and the list should stay short.
KNOWN: dict[str, str] = {
    # https://github.com/pypa/setuptools/blob/main/LICENSE
    "setuptools": "MIT",
    # https://github.com/pypa/wheel/blob/main/LICENSE.txt
    "wheel": "MIT",
    # https://github.com/pypa/pip/blob/main/LICENSE.txt
    "pip": "MIT",
}

_CLASSIFIER = re.compile(r"License :: (?:OSI Approved :: )?(.+)")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().upper().rstrip(".")


def license_of(dist: md.Distribution) -> tuple[str, str]:
    """Best available license string for a distribution, and where it came from."""
    meta = dist.metadata
    name = (meta.get("Name") or "").lower()
    if name in KNOWN:
        return KNOWN[name], "known-list"
    expr = meta.get("License-Expression")
    if expr:
        return expr, "License-Expression"
    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    for c in classifiers:
        m = _CLASSIFIER.match(c)
        if m and m.group(1) != "OSI Approved":
            return m.group(1), "classifier"
    lic = meta.get("License")
    if lic:
        # Some projects dump the whole license text into this field.
        return lic.splitlines()[0] if len(lic) > 200 else lic, "License"
    return "", "missing"


def verdict(license_text: str) -> str:
    norm = _normalise(license_text)
    if not norm:
        return "unknown"
    for bad in DENY_SUBSTRINGS:
        if bad in norm:
            return "denied"
    # Split on common separators for dual-licensed distributions; allowed if any
    # single alternative is allowed.
    parts = [p.strip() for p in re.split(r"\bOR\b|/|;|,", norm) if p.strip()]
    for p in parts + [norm]:
        if p in ALLOWED:
            return "allowed"
    for p in parts + [norm]:
        for ok in ALLOWED:
            if p.startswith(ok):
                return "allowed"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runtime-only",
        action="store_true",
        help="check only what remoroo-lc declares, not the whole environment",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    wanted = None
    if args.runtime_only:
        try:
            reqs = md.requires("remoroo-lc") or []
        except md.PackageNotFoundError:
            print("remoroo-lc is not installed; run `pip install -e .` first", file=sys.stderr)
            return 2
        wanted = {
            re.split(r"[<>=!\[; ]", r, 1)[0].strip().lower()
            for r in reqs
            if "extra ==" not in r
        }

    rows, bad = [], []
    for dist in sorted(md.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
        name = dist.metadata.get("Name")
        if not name:
            continue
        if wanted is not None and name.lower() not in wanted:
            continue
        lic, source = license_of(dist)
        v = verdict(lic)
        rows.append((name, dist.version, lic or "-", source, v))
        if v != "allowed":
            bad.append((name, lic, v))

    if not args.quiet:
        width = max((len(r[0]) for r in rows), default=4)
        for name, version, lic, source, v in rows:
            mark = "ok " if v == "allowed" else "FAIL"
            print(f"{mark} {name:<{width}} {version:<12} {lic[:44]:<44} ({source})")

    if bad:
        print(f"\n{len(bad)} package(s) outside the allowlist:", file=sys.stderr)
        for name, lic, v in bad:
            print(f"  {name}: {lic or '<no license metadata>'} [{v}]", file=sys.stderr)
        print(
            "\nAllowed: MIT, BSD-2/3, Apache-2.0, ISC, Zlib.  Add a sourced entry to "
            "KNOWN only if the metadata is missing but the licence is verifiably "
            "permissive.",
            file=sys.stderr,
        )
        return 1

    print(f"\nlicense check passed: {len(rows)} package(s), all permissive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
