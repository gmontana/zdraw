#!/usr/bin/env python3
"""Keep ZDraw's environment override surface explicit and owned."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
INVENTORY = Path("docs/env-flags.md")
FLAG = re.compile(r"\bZDRAW_[A-Z0-9_]+\b")
TABLE_ROW = re.compile(
    r"^\|\s*(ZDRAW_[A-Z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*"
    r"([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$"
)
CLASSIFICATIONS = frozenset(
    {
        "default-config",
        "deployment",
        "instrument",
        "bench-selector",
        "quarantine",
        "candidate",
        "killed",
        "retired",
    }
)
# This is a C compile-time constant, not an environment variable.
EXCLUDED_NAMES = frozenset({"ZDRAW_BATCH_MAX_SPLITS"})
# Generated C ABI macros (src/abi_gen.zig -> zdraw_abi.h), not env flags.
EXCLUDED_PREFIXES = ("ZDRAW_ABI_",)


@dataclass(frozen=True)
class InventoryEntry:
    owner: str
    classification: str
    default_source: str
    disposition: str


def tracked_override_paths(repo: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode())
        if (
            str(relative).startswith(("src/", "cmd/", "tools/", "examples/"))
            or len(relative.parts) == 1
            and relative.suffix == ".zig"
        ):
            paths.append(repo / relative)
    return tuple(paths)


def flags_in_paths(paths: tuple[Path, ...]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found.update(FLAG.findall(text))
    found -= EXCLUDED_NAMES
    return {n for n in found if not n.startswith(EXCLUDED_PREFIXES)}


def parse_inventory(text: str) -> tuple[dict[str, InventoryEntry], list[str]]:
    entries: dict[str, InventoryEntry] = {}
    failures: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("| ZDRAW_"):
            continue
        match = TABLE_ROW.fullmatch(line)
        if match is None:
            failures.append(
                f"docs/env-flags.md:{line_number} is not one canonical flag row"
            )
            continue
        name, owner, classification, default_source, disposition = match.groups()
        if name in entries:
            failures.append(f"docs/env-flags.md duplicates {name}")
            continue
        if classification not in CLASSIFICATIONS:
            failures.append(
                f"docs/env-flags.md gives {name} invalid classification "
                f"{classification!r}"
            )
        if (
            classification == "killed"
            and "remove after " not in disposition.lower()
        ):
            failures.append(
                f"docs/env-flags.md gives killed flag {name} no concrete "
                "removal dependency"
            )
        entries[name] = InventoryEntry(
            owner,
            classification,
            default_source,
            disposition,
        )
    if "probe-unruled" in text:
        failures.append("docs/env-flags.md retains an unruled probe")
    return entries, failures


def compare_inventory(
    used: set[str],
    entries: dict[str, InventoryEntry],
) -> list[str]:
    failures: list[str] = []
    documented = set(entries)
    for name in sorted(used - documented):
        failures.append(f"{name} is used but absent from {INVENTORY}")
    for name in sorted(used & documented):
        if entries[name].classification == "retired":
            failures.append(f"{name} is used but classified as retired")
    for name in sorted(documented - used):
        if entries[name].classification != "retired":
            failures.append(f"{name} is documented but unused")
    return failures


def check_owners(entries: dict[str, InventoryEntry], repo: Path) -> list[str]:
    """The Canonical-owner column is a claim, not decoration: when it names a
    source file, that file must actually reference the flag (drift here sent
    a reader to metal_api.m for a flag whose only reader was gemm_mode.zig).
    Non-file owners (descriptive phrases) are not checked."""
    failures: list[str] = []
    for name in sorted(entries):
        entry = entries[name]
        if entry.classification == "retired":
            continue
        owners = [
            o.strip().strip("`")
            for o in entry.owner.split(",")
            if o.strip().strip("`")
        ]
        for owner in owners:
            if not owner.endswith((".zig", ".m", ".py", ".sh")):
                continue
            candidates = [
                repo / "src" / owner,
                repo / "cmd" / owner,
                repo / "tools" / owner,
                repo / owner,
            ]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                nested = sorted(repo.glob(f"tools/**/{owner}"))
                path = nested[0] if nested else None
            if path is None:
                failures.append(f"{name}: canonical owner {owner} not found")
                continue
            if name not in path.read_text(encoding="utf-8", errors="replace"):
                failures.append(
                    f"{name}: canonical owner {owner} does not reference it"
                )
    return failures


def check(repo: Path = REPO) -> list[str]:
    inventory_path = repo / INVENTORY
    if not inventory_path.is_file():
        return [f"{INVENTORY} is missing"]
    entries, failures = parse_inventory(
        inventory_path.read_text(encoding="utf-8")
    )
    used = flags_in_paths(tracked_override_paths(repo))
    failures.extend(compare_inventory(used, entries))
    failures.extend(check_owners(entries, repo))
    return failures


def self_test() -> None:
    sample = "Z" + "DRAW_ONE"
    valid = (
        "| Flag | Canonical owner | Classification | Default source | "
        "Disposition |\n"
        "|---|---|---|---|---|\n"
        f"| {sample} | owner | instrument | unset | keep |\n"
    )
    entries, failures = parse_inventory(valid)
    assert not failures and set(entries) == {sample}
    assert not compare_inventory({sample}, entries)
    second = "Z" + "DRAW_TWO"
    assert compare_inventory({sample, second}, entries) == [
        f"{second} is used but absent from docs/env-flags.md"
    ]
    assert compare_inventory(set(), entries) == [
        f"{sample} is documented but unused"
    ]
    retired, failures = parse_inventory(valid.replace("instrument", "retired"))
    assert not failures and not compare_inventory(set(), retired)
    assert compare_inventory({sample}, retired) == [
        f"{sample} is used but classified as retired"
    ]

    _, failures = parse_inventory(valid + valid.splitlines()[-1] + "\n")
    assert failures == [f"docs/env-flags.md duplicates {sample}"]

    invalid = valid.replace("instrument", "mystery")
    _, failures = parse_inventory(invalid)
    assert "invalid classification" in failures[0]

    grouped = valid.replace(sample, f"{sample} / _TWO")
    _, failures = parse_inventory(grouped)
    assert failures == [
        "docs/env-flags.md:3 is not one canonical flag row"
    ]

    _, failures = parse_inventory(valid + "probe-unruled\n")
    assert failures == ["docs/env-flags.md retains an unruled probe"]

    killed = valid.replace("instrument", "killed")
    _, failures = parse_inventory(killed)
    assert failures == [
        f"docs/env-flags.md gives killed flag {sample} no concrete "
        "removal dependency"
    ]
    _, failures = parse_inventory(killed.replace("keep", "remove after issue 1"))
    assert not failures

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.zig"
        path.write_text(
            f'const flag = "{sample}"; const max = ZDRAW_BATCH_MAX_SPLITS;\n',
            encoding="utf-8",
        )
        assert flags_in_paths((path,)) == {sample}

    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        (repo / "src").mkdir()
        (repo / "src" / "good.zig").write_text(
            f'const v = "{sample}";\n', encoding="utf-8"
        )
        (repo / "src" / "empty.zig").write_text("// nothing\n", encoding="utf-8")
        owned, _ = parse_inventory(valid.replace("| owner |", "| `good.zig` |"))
        assert not check_owners(owned, repo)
        drifted, _ = parse_inventory(valid.replace("| owner |", "| `empty.zig` |"))
        assert check_owners(drifted, repo) == [
            f"{sample}: canonical owner empty.zig does not reference it"
        ]
        missing, _ = parse_inventory(valid.replace("| owner |", "| `gone.zig` |"))
        assert check_owners(missing, repo) == [
            f"{sample}: canonical owner gone.zig not found"
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("env_guard self-test: ok")
        return 0
    failures = check()
    if failures:
        print("env_guard: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("env_guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
