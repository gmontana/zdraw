#!/usr/bin/env python3
"""zlint ratchet: freeze zlint findings per rule so new ones fail the gate.

    --check     compare current zlint counts to budget; exit 1 on regression (default)
    --snapshot  rewrite the zlint budget from current counts (deliberate)
    --report    print current counts without failing

Reads/writes the "zlint" section of tools/style/quality_budget.json. zlint is the
DonIsaac Zig linter (v0.8.1); its directory traversal finds 0 files here, so the
file list is fed via stdin from ripgrep. If zlint is not installed the check warns
and passes — CI hosts must install zlint v0.8.1 to enforce the gate.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / "tools" / "style" / "quality_budget.json"


def current_counts():
    """Run zlint over src/ and return a Counter of findings by rule code, or None
    if zlint is not installed."""
    if shutil.which("zlint") is None:
        return None, False
    # ripgrep is not guaranteed on every host (the macOS CI runner lacks it);
    # the file list only needs the tracked src/*.zig paths.
    files = "\n".join(
        str(p.relative_to(REPO)) for p in sorted((REPO / "src").rglob("*.zig")) if p.is_file()
    ) + "\n"
    file_count = sum(1 for line in files.splitlines() if line.strip())
    res = subprocess.run(
        ["zlint", "--stdin", "-f", "json"],
        cwd=REPO, input=files, capture_output=True, text=True, check=False,
    )
    counts = Counter()
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            counts[json.loads(line)["code"]] += 1
        except (json.JSONDecodeError, KeyError):
            continue
    # This tree has known findings. Empty output may mean zlint did not scan it
    # or changed its JSON format; refuse to treat that as a clean result.
    broke = file_count > 0 and sum(counts.values()) == 0
    return counts, broke


def load_budget():
    return json.loads(BUDGET.read_text()) if BUDGET.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="default: fail on regression")
    g.add_argument("--snapshot", action="store_true", help="rewrite zlint budget")
    g.add_argument("--report", action="store_true", help="print counts, never fail")
    args = ap.parse_args()

    counts, broke = current_counts()
    if counts is None:
        print("zlint not installed — skipping zlint gate (CI must install zlint v0.8.1)")
        return 0
    if broke:
        print("zlint_check: ERROR — fed src/*.zig to `zlint --stdin -f json` but parsed 0 findings.")
        print("The invocation or output format likely changed; refusing to pass silently.")
        return 1

    data = load_budget()
    if args.snapshot:
        data["zlint"] = dict(sorted(counts.items()))
        BUDGET.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"zlint snapshot written -> {BUDGET.relative_to(REPO)}")
        return 0

    budget = data.get("zlint", {})
    regressed = False
    print(f"{'rule':<20}{'current':>9}{'budget':>9}  delta")
    for r in sorted(set(counts) | set(budget)):
        cur, bud = counts.get(r, 0), budget.get(r, 0)
        mark = "  <-- REGRESSION" if cur > bud else ("  (improvement)" if cur < bud else "")
        if cur > bud:
            regressed = True
        print(f"{r:<20}{cur:>9}{bud:>9}  {cur - bud:+d}{mark}")

    if args.report:
        return 0
    if regressed:
        print("\nzlint_check: REGRESSION — new findings above budget.")
        print("Remove them, or (if deliberate) re-snapshot: tools/style/zlint_check.py --snapshot")
        return 1
    print("zlint_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
