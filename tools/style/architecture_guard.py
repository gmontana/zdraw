#!/usr/bin/env python3
"""Enforce ZDraw's dependency and memory-ownership boundaries."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
IMPORT = re.compile(r'@import\("([^"]+)"\)')

# These modules are policy/value boundaries. Expanding an allow-list is an
# architecture decision because it changes which layer may influence them.
ALLOWED_IMPORTS = {
    "model_kind.zig": {"std"},
    "progress_sink.zig": {"std"},
    "progress_bar.zig": {"std", "progress_sink.zig"},
    "progress.zig": {"std", "progress_sink.zig", "metrics.zig", "progress_bar.zig", "progress_load.zig"},
    "preview.zig": {"std", "image.zig", "progress_sink.zig", "preview_map.zig", "progress.zig"},
    "session_files.zig": {"std"},
    "model_paths.zig": {"std", "model_kind.zig"},
    "execution_plan.zig": {"std", "model_kind.zig"},
    "execution_receipt.zig": {
        "std",
        "execution_plan.zig",
        "model_kind.zig",
    },
    "hook_bypass_control.zig": {"std"},
    "token_selection_control.zig": {"std"},
    "toma_control.zig": {"std", "toma_config.zig"},
    "token_selection_runtime.zig": {
        "std",
        "metal_c.zig",
        "mpipe.zig",
        "token_selection_control.zig",
    },
    "mbuffer.zig": {
        "std",
        "metal_c.zig",
        "mpacked.zig",
        "tensor.zig",
        "zpack_file.zig",
    },
    "mchain_pool.zig": {"std", "metal_c.zig", "mbuffer.zig"},
    "mlend.zig": {"std", "mchain_pool.zig"},
}

# Large CPU allocations in these two VAE modules deliberately bypass an arena
# so their storage can be returned at a phase boundary. No other product module
# may introduce an implicit process allocator.
PAGE_ALLOCATOR_OWNERS = {
    "mvres_stream_chain.zig",
    "vdecode.zig",
}
SMP_ALLOCATOR_OWNERS = set()
RAW_BUFFER_RELEASE_OWNERS = {
    "mbuffer.zig",
    "metal_c.zig",
    "mpacked.zig",
}

def imports(path: Path) -> set[str]:
    """Import targets by file name: modules live in src/ subfolders and reach
    each other with relative paths, and the boundary table names files."""
    return {Path(t).name for t in IMPORT.findall(path.read_text(encoding="utf-8"))}


def locate(src: Path, name: str) -> Path:
    """The one file called `name` under src/ (flat or in a subfolder)."""
    direct = src / name
    if direct.is_file():
        return direct
    matches = sorted(src.rglob(name))
    if len(matches) != 1:
        raise SystemExit(f"architecture_guard: expected one {name} under {src}, found {len(matches)}")
    return matches[0]


def check_imports(
    src: Path = SRC,
    allowed_imports: dict = None,
) -> list[str]:
    if allowed_imports is None:
        allowed_imports = ALLOWED_IMPORTS
    failures: list[str] = []
    for name, allowed in allowed_imports.items():
        actual = imports(locate(src, name))
        unexpected = sorted(actual - allowed)
        if unexpected:
            failures.append(
                f"src/{name} crosses its dependency boundary: {unexpected}"
            )
    return failures


def check_safety_exit(src: Path = SRC) -> list[str]:
    text = locate(src, "safety.zig").read_text(encoding="utf-8")
    if "std.process.exit(" in text or "std.c.exit(" in text:
        return ["safety.zig must return errors so interactive callers can unwind"]
    return []


def check_allocators(src: Path = SRC) -> list[str]:
    failures: list[str] = []
    for path in sorted(src.rglob("*.zig")):
        text = path.read_text(encoding="utf-8")
        if (
            "std.heap.page_allocator" in text
            and path.name not in PAGE_ALLOCATOR_OWNERS
        ):
            failures.append(
                f"{path.relative_to(REPO)} uses page_allocator without owning that policy"
            )
        if (
            "std.heap.smp_allocator" in text
            and path.name not in SMP_ALLOCATOR_OWNERS
        ):
            failures.append(
                f"{path.relative_to(REPO)} uses smp_allocator outside an application boundary"
            )
    return failures


def check_metal_buffers(src: Path = SRC) -> list[str]:
    failures: list[str] = []
    for path in sorted(src.rglob("*.zig")):
        text = path.read_text(encoding="utf-8")
        if path.name != "mbuffer.zig" and re.search(
            r"\bmbuffer\.Buffer\s*\{\s*\.handle",
            text,
        ):
            failures.append(
                f"{path.relative_to(REPO)} constructs Buffer directly; use Buffer.borrow "
                "or an owning constructor"
            )
        if (
            "zdraw_metal_release_buffer(" in text
            and path.name not in RAW_BUFFER_RELEASE_OWNERS
        ):
            failures.append(
                f"{path.relative_to(REPO)} releases a raw Metal buffer outside its owner"
            )
    buffer_source = locate(src, "mbuffer.zig").read_text(encoding="utf-8")
    required = (
        "pub const Ownership = enum",
        "pub fn borrow(handle: *anyopaque) Buffer",
        "std.debug.assert(self.ownership == .owned)",
    )
    for contract in required:
        if contract not in buffer_source:
            failures.append(f"src/mbuffer.zig lacks ownership contract: {contract}")
    return failures


def check_metal_pipelines() -> list[str]:
    failures: list[str] = []
    for path in sorted(SRC.rglob("*.zig")):
        text = path.read_text(encoding="utf-8")
        if "mpipe.required(" in text and "zdraw_metal_release_pipeline(" not in text:
            failures.append(
                f"{path.relative_to(REPO)} compiles Metal pipelines without an explicit "
                "release owner"
            )
        lines = text.splitlines()
        # \s* spans newlines so a binding wrapped by zig fmt still matches.
        for match in re.finditer(r"\bconst (\w+) =\s*try mpipe\.required\(", text):
            index = text.count("\n", 0, match.start())
            name = match.group(1)
            cleanup = "\n".join(lines[index + 1 : index + 5])
            if f"zdraw_metal_release_pipeline({name})" not in cleanup:
                failures.append(
                    f"{path.relative_to(REPO)}:{index + 1} does not immediately protect "
                    f"pipeline {name} with errdefer"
                )
    return failures


def check_release_boundary(repo: Path = REPO) -> list[str]:
    failures = []
    for name in ("app", "swift", "include", "src/engine_c.zig", "src/workbench.zig"):
        if (repo / name).exists():
            failures.append(f"{name} is outside the engine CLI release boundary")
    for name in ("build.zig", "build.zig.zon", "docs/architecture.md"):
        if not (repo / name).is_file():
            failures.append(f"{name} is missing")
    build = (repo / "build.zig").read_text(encoding="utf-8")
    paths = re.findall(r'b.path\("([^"]+)"\)', build)
    paths += re.findall(r'\.root = "([^"]+)"', build)
    for path in sorted(set(paths)):
        if not (repo / path).exists():
            failures.append(f"build.zig references missing input {path}")
    return failures


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        src = Path(directory)
        path = src / "safety.zig"
        path.write_text("return error.SafetyBlocked;", encoding="utf-8")
        assert not check_safety_exit(src)
        path.write_text("std.process.exit(3);", encoding="utf-8")
        assert check_safety_exit(src)
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        (repo / "docs").mkdir()
        (repo / "docs/architecture.md").write_text("architecture", encoding="utf-8")
        (repo / "build.zig.zon").write_text(".{}", encoding="utf-8")
        (repo / "build.zig").write_text('b.path("absent.zig")', encoding="utf-8")
        assert any("missing input" in v for v in check_release_boundary(repo))
        (repo / "absent.zig").write_text("", encoding="utf-8")
        assert not check_release_boundary(repo)
        (repo / "app").mkdir()
        assert any("release boundary" in v for v in check_release_boundary(repo))

    with tempfile.TemporaryDirectory() as directory:
        src = Path(directory)
        (src / "a.zig").write_text(
            'const b = @import("b.zig");\n', encoding="utf-8"
        )
        assert not check_imports(src, {"a.zig": {"b.zig"}})
        failures = check_imports(src, {"a.zig": {"std"}})
        assert any("crosses its dependency boundary" in v for v in failures)

        (src / "rogue_alloc.zig").write_text(
            "const a = std.heap.page_allocator;\n", encoding="utf-8"
        )
        failures = check_allocators(src)
        assert any(
            "page_allocator without owning that policy" in v for v in failures
        )
        (src / "rogue_alloc.zig").unlink()
        (src / "vdecode.zig").write_text(
            "const a = std.heap.page_allocator;\n", encoding="utf-8"
        )
        assert not check_allocators(src)

        (src / "mbuffer.zig").write_text(
            "pub const Ownership = enum { owned, borrowed };\n"
            "pub fn borrow(handle: *anyopaque) Buffer {}\n"
            "std.debug.assert(self.ownership == .owned)\n",
            encoding="utf-8",
        )
        assert not check_metal_buffers(src)
        (src / "rogue_buf.zig").write_text(
            "const b = mbuffer.Buffer{ .handle = h };\n", encoding="utf-8"
        )
        failures = check_metal_buffers(src)
        assert any("constructs Buffer directly" in v for v in failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("architecture_guard self-test: ok")
        return 0
    failures = [
        *check_imports(),
        *check_allocators(),
        *check_safety_exit(),
        *check_metal_buffers(),
        *check_metal_pipelines(),
        *check_release_boundary(),
    ]
    if failures:
        print("architecture_guard: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("architecture_guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
