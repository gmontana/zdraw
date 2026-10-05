#!/usr/bin/env python3
"""Project-specific Zig quality linter with ratchet budgets.

Counts patterns we don't want growing (unsafe builtins, catch unreachable,
anytype, page_allocator in src/, TODOs, oversized files) and fails when any
budget is exceeded. The budget JSON is committed; bumps are explicit decisions.

Modes:
    --check     compare current counts to budget; exit 1 on regression  (default)
    --snapshot  rewrite the budget file from current counts             (deliberate)
    --report    print current counts; never fails
    --staged    restrict to git-staged Zig files (for pre-commit; reports only)

Ratchet philosophy: existing debt is frozen as a budget; only NEW debt
fails. Enforces the repository style limits.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / "tools" / "style" / "quality_budget.json"
SRC = REPO / "src"

# Patterns counted across src/. SAFETY-comment is a separate rule (below).
COUNT_PATTERNS: dict[str, re.Pattern] = {
    "ptr_cast":         re.compile(r"@ptrCast\b"),
    "align_cast":       re.compile(r"@alignCast\b"),
    "const_cast":       re.compile(r"@constCast\b"),
    "bit_cast":         re.compile(r"@bitCast\b"),
    "int_from_ptr":     re.compile(r"@intFromPtr\b"),
    "ptr_from_int":     re.compile(r"@ptrFromInt\b"),
    "catch_unreachable":re.compile(r"\bcatch\s+unreachable\b"),
    "anytype":          re.compile(r"\banytype\b"),
    "page_allocator":   re.compile(r"std\.heap\.page_allocator\b"),
    "c_allocator":      re.compile(r"std\.heap\.c_allocator\b"),
    "todo":             re.compile(r"//[^\n]*\b(TODO|FIXME|HACK)\b"),
}

# Size facts are tracked as ratchet counts (files_over_1500_loc, largest_file_loc)
# in collect_size_facts; there are no separate hard-limit constants. Function
# length is measured by the brace-aware scanner below.

# Directory policy: page_allocator/c_allocator only allowed in these prefixes.
# The architecture guard enforces these exact VAE owners independently.
ALLOC_ALLOWLIST = ("src/mvres_stream_chain.zig", "src/vdecode.zig")


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def diff_added_lines(staged: bool) -> dict[str, set[int]]:
    """Parse `git diff -U0` and return {file: {added line numbers}} (post-image)."""
    cmd = ["git", "diff", "-U0", "--no-color"]
    if staged:
        cmd.append("--cached")
    cmd += ["--", "*.zig"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False).stdout
    added: dict[str, set[int]] = {}
    cur_file: str | None = None
    cur_line = 0
    for raw in out.splitlines():
        if raw.startswith("+++ b/"):
            cur_file = raw[6:]
            added.setdefault(cur_file, set())
        elif (m := HUNK_RE.match(raw)):
            cur_line = int(m.group(1))
        elif raw.startswith("+") and not raw.startswith("+++") and cur_file is not None:
            added[cur_file].add(cur_line)
            cur_line += 1
        elif raw.startswith(" "):
            cur_line += 1
    return added


UNSAFE_BUILTIN_RE = re.compile(r"@(ptrCast|alignCast|constCast|bitCast|ptrFromInt|intFromPtr)\b")
SAFETY_RE = re.compile(r"//\s*SAFETY\s*:", re.IGNORECASE)


def unsafe_violations_on_lines(rel: str, line_set: set[int]) -> list[tuple[str, int, str]]:
    """Return (rel, line, snippet) for unsafe builtins on the given (post-image)
    lines that lack a SAFETY: comment in the 3 lines above."""
    if rel.startswith("tools/") or rel.startswith("tests/"):
        return []
    path = REPO / rel
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[tuple[str, int, str]] = []
    for ln in sorted(line_set):
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        line = lines[idx]
        if not UNSAFE_BUILTIN_RE.search(line):
            continue
        window = lines[max(0, idx - 3): idx]
        if any(SAFETY_RE.search(w) for w in window):
            continue
        out.append((rel, ln, line.strip()[:120]))
    return out


def zig_files(staged_only: bool) -> list[Path]:
    if staged_only:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", "*.zig"],
            cwd=REPO, capture_output=True, text=True, check=False,
        ).stdout.split()
        return [REPO / p for p in out if p.endswith(".zig") and (REPO / p).is_file()]
    return [p for p in SRC.rglob("*.zig") if p.is_file()]


FN_KW_RE = re.compile(r"\bfn\b")


def strip_zig(text: str) -> str:
    """Replace Zig comments, string literals, and char literals with spaces
    (newlines preserved) so pattern counts and brace-matching see only code.
    Zig has no `/*...*/` block comments; a line starting `\\\\` is a multiline
    string. This removes the regex-over-raw-text defect: a `@ptrCast` named in a
    `//!` design block or a `'}'` char literal no longer affects counting."""
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]
        if state == "code":
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                state, i = "line", i + 2
                out.append("  ")
            elif ch == "\\" and i + 1 < n and text[i + 1] == "\\":
                state, i = "line", i + 2  # multiline-string line: skip to EOL
                out.append("  ")
            elif ch == '"':
                state, i = "str", i + 1
                out.append(" ")
            elif ch == "'":
                state, i = "chr", i + 1
                out.append(" ")
            else:
                out.append(ch)
                i += 1
        elif state == "line":
            if ch == "\n":
                state = "code"
                out.append("\n")
            else:
                out.append(" ")
            i += 1
        else:  # "str" or "chr": consume until the matching quote, honoring \ escapes
            quote = '"' if state == "str" else "'"
            if ch == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
            elif ch == quote or ch == "\n":
                state = "code"
                out.append("\n" if ch == "\n" else " ")
                i += 1
            else:
                out.append(" ")
                i += 1
    return "".join(out)


def fn_span_lines(code: str, fn_pos: int) -> int | None:
    """Line span of the function whose `fn` keyword is at code[fn_pos], or None for
    a bodyless prototype / fn-type. Brace-matches the body block, so methods nested
    in a struct are measured (the depth-0 regex blind spot)."""
    n = len(code)
    i = fn_pos
    while i < n and code[i] != "(":  # to the parameter list
        if code[i] in ";{":
            return None
        i += 1
    depth = 0
    while i < n:  # match the parameter parens
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    i += 1
    kw = ("error", "struct", "union", "enum")
    while i < n:  # to the body open brace; `;` first = no body
        if code[i] == ";":
            return None
        if code[i] == "{":
            # A brace-opening RETURN TYPE (error{...}!T, struct { ... }) is
            # not the body: brace-match the group and keep scanning.
            j = i - 1
            while j >= 0 and code[j] in " \t\r\n":
                j -= 1
            k = j
            while k >= 0 and (code[k].isalnum() or code[k] == "_"):
                k -= 1
            if code[k + 1 : j + 1] in kw:
                gdepth = 0
                while i < n:
                    if code[i] == "{":
                        gdepth += 1
                    elif code[i] == "}":
                        gdepth -= 1
                        if gdepth == 0:
                            break
                    i += 1
                i += 1
                continue
            break
        i += 1
    if i >= n or code[i] != "{":
        return None
    bdepth = 0
    while i < n:  # match the body braces
        if code[i] == "{":
            bdepth += 1
        elif code[i] == "}":
            bdepth -= 1
            if bdepth == 0:
                break
        i += 1
    if i >= n:
        return None
    return code.count("\n", fn_pos, i) + 1


def fn_length_counts(files: list[Path]) -> dict[str, int]:
    # 60 = "fits on one screen" soft cap (sweet spot; 40 forced artificial
    # splits). 120 = hard cap. Both ratcheted via the budget for justified cases.
    over_60 = 0
    over_120 = 0
    for f in files:
        code = strip_zig(f.read_text(encoding="utf-8", errors="replace"))
        for m in FN_KW_RE.finditer(code):
            span = fn_span_lines(code, m.start())
            if span is None:
                continue
            if span > 60:
                over_60 += 1
            if span > 120:
                over_120 += 1
    return {"functions_over_60": over_60, "functions_over_120": over_120}


def count_in_file(path: Path) -> dict[str, int]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    code = strip_zig(raw)
    # `todo` targets comments, so it scans raw text; every other pattern counts
    # CODE only, so a mention in a comment/string can neither inflate nor mask it.
    return {k: len(p.findall(raw if k == "todo" else code)) for k, p in COUNT_PATTERNS.items()}


def collect_counts(files: list[Path]) -> dict[str, int]:
    totals = {k: 0 for k in COUNT_PATTERNS}
    for f in files:
        for k, v in count_in_file(f).items():
            totals[k] += v
    totals.update(count_arch_imports(files))
    totals.update(fn_length_counts(files))
    totals["lines_over_100"] = count_long_lines(files)
    return totals


def count_long_lines(files: list[Path]) -> int:
    """Total number of lines in src/ longer than 100 columns. TIGER_STYLE
    §"Style by numbers" hard limit. Ratcheted, not retroactively enforced —
    the count cannot grow, but the existing baseline is grandfathered until
    the cleanup sprint reduces it module by module."""
    total = 0
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if len(line) > 100:
                total += 1
    return total


# Directional layer-import counters, keyed (path-prefix, regex). Empty in
# zdraw: the compiler-to-backends rule left with the source project; add a
# row here plus a budget key when a ratcheted import direction exists.
ARCH_IMPORTS: dict = {}


def count_arch_imports(files: list[Path]) -> dict[str, int]:
    out = {k: 0 for k in ARCH_IMPORTS}
    for f in files:
        rel = str(f.relative_to(REPO))
        for key, (prefix, pat) in ARCH_IMPORTS.items():
            if not rel.startswith(prefix):
                continue
            out[key] += len(pat.findall(f.read_text(encoding="utf-8", errors="replace")))
    return out


def file_sizes(files: list[Path]) -> dict[str, int]:
    return {str(f.relative_to(REPO)): sum(1 for _ in f.open("r", encoding="utf-8", errors="replace"))
            for f in files}


def find_unsafe_without_safety(files: list[Path]) -> list[tuple[str, int, str]]:
    """Return (file, line, snippet) for unsafe builtins lacking a SAFETY comment
    within 3 lines above. Skips test files and tools/.
    """
    out = []
    for f in files:
        rel = str(f.relative_to(REPO))
        if rel.startswith("tools/") or rel.startswith("tests/"):
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if not UNSAFE_BUILTIN_RE.search(line):
                continue
            window = lines[max(0, i - 3): i]
            if any(SAFETY_RE.search(w) for w in window):
                continue
            out.append((rel, i + 1, line.strip()[:120]))
    return out


def collect_size_facts(files: list[Path]) -> dict:
    sizes = file_sizes(files)
    over_1500 = sum(1 for v in sizes.values() if v > 1500)
    largest = max(sizes.values()) if sizes else 0
    return {"files_over_1500_loc": over_1500, "largest_file_loc": largest}


def allocator_violations(files: list[Path]) -> list[tuple[str, int, str]]:
    out = []
    pat = re.compile(r"std\.heap\.(?:page_allocator|c_allocator)\b")
    for f in files:
        rel = str(f.relative_to(REPO))
        if not rel.startswith("src/"):
            continue
        if rel in ALLOC_ALLOWLIST:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines()):
            if pat.search(line):
                out.append((rel, i + 1, line.strip()[:120]))
    return out


def load_budget() -> dict:
    if not BUDGET.exists():
        return {}
    return json.loads(BUDGET.read_text())


def write_budget(data: dict) -> None:
    BUDGET.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def cmd_snapshot() -> int:
    files = zig_files(staged_only=False)
    counts = collect_counts(files)
    sizes = collect_size_facts(files)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=False).stdout.strip()
    # Preserve sibling sections written by other tools (e.g. the "zlint" budget
    # owned by zlint_check.py) — only this tool's keys are rewritten here.
    data = load_budget()
    data.update({
        "version": 1,
        "as_of_commit": git_head,
        "counts": counts,
        "sizes": sizes,
        "_notes": "Run `tools/style/zig_guard.py --snapshot` to re-baseline. "
                  "Bumps must accompany a deliberate decision; cleanups should lower numbers.",
    })
    write_budget(data)
    print(f"snapshot written -> {BUDGET.relative_to(REPO)}")
    for k, v in counts.items():
        print(f"  {k:20s} {v}")
    for k, v in sizes.items():
        print(f"  {k:20s} {v}")
    return 0


def cmd_check() -> int:
    budget = load_budget()
    if not budget:
        print("no budget yet; run with --snapshot to baseline", file=sys.stderr)
        return 2
    files = zig_files(staged_only=False)
    counts = collect_counts(files)
    sizes = collect_size_facts(files)
    fails: list[str] = []
    warns: list[str] = []
    print("pattern                          current  budget  delta")
    for k in sorted(set(counts) | set(budget.get("counts", {}))):
        cur, bud = counts.get(k, 0), budget["counts"].get(k, 0)
        d = cur - bud
        flag = ""
        if d > 0:
            flag = "  REGRESSION"
            fails.append(f"{k}: {cur} > budget {bud} (+{d})")
        elif d < 0:
            flag = "  (improvement — consider --snapshot to lower budget)"
        print(f"  {k:30s} {cur:6d}  {bud:6d}  {d:+d}{flag}")
    for k, v in sizes.items():
        bv = budget["sizes"].get(k, 0)
        d = v - bv
        flag = "  REGRESSION" if d > 0 else ""
        print(f"  {k:20s} {v:6d}  {bv:6d}  {d:+d}{flag}")
        if d > 0:
            fails.append(f"{k}: {v} > budget {bv} (+{d})")
    # SAFETY-comment & allocator rules — WARN for v0 (would false-positive on
    # existing 1000+ sites). Promote to fails once the SAFETY convention (CONTRIBUTING.md) is seeded and
    # diff-based checking lands. Only inspect staged files to stay cheap.
    staged = zig_files(staged_only=True)
    if staged:
        for (rel, ln, snip) in find_unsafe_without_safety(staged)[:10]:
            warns.append(f"unsafe without SAFETY (warn): {rel}:{ln}  {snip}")
        for (rel, ln, snip) in allocator_violations(staged)[:10]:
            warns.append(f"raw allocator in src/ outside allowlist (warn): {rel}:{ln}  {snip}")
    if fails:
        print("\nFAIL:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        print("\nIf intentional: re-snapshot with `tools/style/zig_guard.py --snapshot` "
              "in a separate commit and justify.", file=sys.stderr)
        return 1
    for w in warns:
        print(f"WARN: {w}", file=sys.stderr)
    print("\nzig_guard: ok")
    return 0


def cmd_check_diff(staged: bool) -> int:
    """Fail on NEWLY-ADDED unsafe builtins without a SAFETY comment.
    Operates on `git diff` (staged if --staged); existing sites are not retrofit."""
    added = diff_added_lines(staged)
    fails: list[tuple[str, int, str]] = []
    for rel, lines in added.items():
        fails.extend(unsafe_violations_on_lines(rel, lines))
    if not fails:
        print("zig_guard --check-diff: ok (no new unsafe sites without SAFETY)")
        return 0
    print("FAIL: new unsafe builtin(s) without `// SAFETY:` within 3 lines above",
          file=sys.stderr)
    print("See CONTRIBUTING.md (Unsafe operations) for the SAFETY-comment convention.", file=sys.stderr)
    for (rel, ln, snip) in fails[:20]:
        print(f"  {rel}:{ln}  {snip}", file=sys.stderr)
    return 1


def cmd_report(staged: bool) -> int:
    files = zig_files(staged_only=staged)
    if not files:
        print("no files to scan")
        return 0
    counts = collect_counts(files)
    sizes = collect_size_facts(files)
    print(f"scanned {len(files)} files{' (staged)' if staged else ''}")
    for k, v in counts.items():
        print(f"  {k:20s} {v}")
    for k, v in sizes.items():
        print(f"  {k:20s} {v}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check",      action="store_true", help="default: fail on regression vs budget")
    g.add_argument("--check-diff", action="store_true", help="fail on NEW unsafe builtins without SAFETY (diff-based)")
    g.add_argument("--snapshot",   action="store_true", help="rewrite budget from current state")
    g.add_argument("--report",     action="store_true", help="print counts without failing")
    ap.add_argument("--staged", action="store_true", help="restrict to staged diff/files")
    args = ap.parse_args()
    if args.snapshot:   return cmd_snapshot()
    if args.check_diff: return cmd_check_diff(args.staged)
    if args.report:     return cmd_report(args.staged)
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
