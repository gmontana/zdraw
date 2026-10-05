#!/usr/bin/env python3
"""Check new Zig function names against the repository naming contract.

The repository still has historical long names. This tool blocks new drift by
checking added function declarations in a git diff. Use --all when planning a
cleanup pass over the existing backlog.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


max_name_len = 15
fn_re = re.compile(r"^\s*(?:(?:pub|inline|export)\s+)*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class Finding:
    def __init__(self, path: str, line: int, name: str, reason: str) -> None:
        self.path = path
        self.line = line
        self.name = name
        self.reason = reason

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.name}: {self.reason}"


def run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def is_source(path: str) -> bool:
    return path.startswith("src/") and path.endswith(".zig")


def check_line(path: str, line_no: int, text: str) -> list[Finding]:
    match = fn_re.match(text)
    if match is None:
        return []

    name = match.group(1)
    findings: list[Finding] = []
    if name.startswith("_"):
        findings.append(Finding(path, line_no, name, "leading underscores are forbidden"))
    if "_" in name:
        findings.append(Finding(path, line_no, name, "functions use camelCase inside their namespace"))
    if len(name) > max_name_len:
        findings.append(Finding(path, line_no, name, f"name is {len(name)} chars; limit is {max_name_len}"))
    return findings


def diff_findings(staged: bool) -> list[Finding]:
    args = ["diff"]
    if staged:
        args.append("--cached")
    args.extend(["--unified=0", "--", "*.zig"])

    current_path: str | None = None
    new_line = 0
    findings: list[Finding] = []

    for raw in run_git(args).splitlines():
        if raw.startswith("+++ b/"):
            path = raw[len("+++ b/") :]
            current_path = path if is_source(path) else None
            continue
        if raw.startswith("+++ /dev/null"):
            current_path = None
            continue

        hunk = hunk_re.match(raw)
        if hunk is not None:
            new_line = int(hunk.group(1))
            continue

        if current_path is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            findings.extend(check_line(current_path, new_line, raw[1:]))
            new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw:
            new_line += 1

    return findings


def all_findings() -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(Path("src").rglob("*.zig")):
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), start=1):
            findings.extend(check_line(path.as_posix(), idx, line))
    return findings


def print_findings(findings: list[Finding]) -> None:
    print("Function naming violations:")
    for finding in findings[:80]:
        print(f"  {finding.format()}")
    if len(findings) > 80:
        print(f"  ... {len(findings) - 80} more")
    print("")
    print("Use camelCase names of 15 chars or fewer inside the local namespace.")
    print("If a name needs every dispatch axis spelled out, extract a namespace first.")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="check added lines in the staged diff")
    mode.add_argument("--worktree", action="store_true", help="check added lines in the unstaged diff")
    mode.add_argument("--all", action="store_true", help="check all current src/*.zig declarations")
    args = parser.parse_args()

    if args.staged:
        findings = diff_findings(staged=True)
    elif args.worktree:
        findings = diff_findings(staged=False)
    else:
        findings = all_findings()

    if findings:
        print_findings(findings)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
