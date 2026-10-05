#!/usr/bin/env python3
"""Licence and notice guard.

Checks, from the repository root:
- LICENSE is the Apache License, Version 2.0;
- every directory under vendor/ has a LICENSE* file and is named in NOTICE;
- every bundled font under app/ (*.ttf, *.otf) has an OFL*.txt beside it;
- every dependency in build.zig.zon is named in NOTICE;
- NOTICE points at COMMERCIAL.md, and COMMERCIAL.md, SUPPORT.md, CONTRIBUTING.md exist;
- README's "## Licence" section names Apache-2.0 and COMMERCIAL.md.
Exit 0 when everything holds; prints each failure otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check(root: Path) -> list[str]:
    fails: list[str] = []
    licence = root / "LICENSE"
    if not licence.is_file():
        fails.append("LICENSE missing")
    else:
        lines = licence.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2 or [line.strip() for line in lines[:2]] != [
            "Apache License", "Version 2.0, January 2004"
        ]:
            fails.append("LICENSE does not start with the Apache 2.0 title and version")
    notice_path = root / "NOTICE"
    notice = notice_path.read_text(encoding="utf-8") if notice_path.is_file() else ""
    if not notice:
        fails.append("NOTICE missing")
    if "COMMERCIAL.md" not in notice:
        fails.append("NOTICE does not point at COMMERCIAL.md")
    for name in ("COMMERCIAL.md", "SUPPORT.md", "CONTRIBUTING.md"):
        if not (root / name).is_file():
            fails.append(f"{name} missing")
    vendor = root / "vendor"
    if vendor.is_dir():
        for entry in sorted(vendor.iterdir()):
            if not entry.is_dir():
                continue
            if not list(entry.glob("LICENSE*")):
                fails.append(f"vendor/{entry.name} has no LICENSE file")
            if f"vendor/{entry.name}" not in notice:
                fails.append(f"vendor/{entry.name} is not named in NOTICE")
    fonts_root = root / "app"
    if fonts_root.is_dir():
        for font in sorted(list(fonts_root.rglob("*.ttf")) + list(fonts_root.rglob("*.otf"))):
            if not list(font.parent.glob("OFL*.txt")):
                rel = font.relative_to(root)
                fails.append(f"{rel} ships without an OFL licence text beside it")
    zon = root / "build.zig.zon"
    if zon.is_file():
        text = zon.read_text(encoding="utf-8")
        block = re.search(r"\.dependencies\s*=\s*\.\{(.*?)\n    \}", text, re.S)
        deps = re.findall(r"^\s{8}\.(\w+)\s*=\s*\.\{", block.group(1), re.M) if block else []
        for dep in deps:
            if dep not in notice:
                fails.append(f"build.zig.zon dependency {dep!r} is not named in NOTICE")
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        section = text.split("## Licence", 1)
        body = section[1] if len(section) == 2 else ""
        if "Apache-2.0" not in body:
            fails.append("README '## Licence' section does not name Apache-2.0")
        if "COMMERCIAL.md" not in body:
            fails.append("README '## Licence' section does not link COMMERCIAL.md")
    return fails


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "LICENSE").write_text("                    Apache License\nVersion 2.0, January 2004\n")
        (root / "NOTICE").write_text("see COMMERCIAL.md\nvendor/x\nstanza\n")
        for name in ("COMMERCIAL.md", "SUPPORT.md", "CONTRIBUTING.md"):
            (root / name).write_text("x")
        (root / "vendor" / "x").mkdir(parents=True)
        (root / "vendor" / "x" / "LICENSE").write_text("MIT")
        (root / "build.zig.zon").write_text(
            ".{\n    .dependencies = .{\n        .stanza = .{\n            .url = \"u\",\n        },\n    },\n}\n"
        )
        (root / "README.md").write_text("# r\n## Licence\nApache-2.0 ... COMMERCIAL.md\n")
        assert check(root) == [], check(root)
        (root / "LICENSE").write_text("GNU AFFERO GENERAL PUBLIC LICENSE\n")
        assert any("Apache 2.0" in f for f in check(root)), check(root)
        (root / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n")
        (root / "app" / "f").mkdir(parents=True)
        (root / "app" / "f" / "Face.ttf").write_bytes(b"\0")
        (root / "vendor" / "y").mkdir()
        fails = check(root)
        assert any("vendor/y" in f for f in fails), fails
        assert any("Face.ttf" in f for f in fails), fails
        (root / "app" / "f" / "OFL.txt").write_text("OFL")
        assert not any("Face.ttf" in f for f in check(root)), check(root)
    print("licence_guard self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    fails = check(ROOT)
    if fails:
        print("licence_guard: FAIL")
        for f in fails:
            print(f"  {f}")
        return 1
    print("licence_guard: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
