#!/usr/bin/env python3
"""Write a durable, hash-bound receipt for the standing Z-Image gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def tree_identity(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(value for value in resolved.rglob("*") if value.is_file()):
        relative = path.relative_to(resolved).as_posix()
        identity = file_identity(path)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(identity["bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(identity["sha256"]).encode())
        digest.update(b"\n")
        count += 1
        total += int(identity["bytes"])
    return {
        "path": str(resolved),
        "files": count,
        "bytes": total,
        "manifest_sha256": digest.hexdigest(),
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def hardware() -> dict[str, str]:
    result = {
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    if sys.platform == "darwin":
        model = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if model:
            result["processor"] = model
    return result


def read_cases(path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            image = Path(row["image"])
            log = Path(row["log"])
            generate_rc = int(row["generate_rc"])
            content_rc = int(row["content_rc"])
            passed = generate_rc == 0 and content_rc == 0 and image.is_file()
            results.append(
                {
                    "case": row["case"],
                    "profile": row["profile"],
                    "size": int(row["size"]),
                    "prompt": row["prompt"],
                    "seed": int(row["seed"]),
                    "generate_rc": generate_rc,
                    "content_rc": content_rc,
                    "passed": passed,
                    "image": file_identity(image) if image.is_file() else None,
                    "log": file_identity(log) if log.is_file() else None,
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--visual-review",
        choices=("pass", "fail", "not-reviewed"),
        default="not-reviewed",
    )
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args()
    if args.visual_review == "pass" and not args.reviewer.strip():
        parser.error("--reviewer is required when visual review passes")

    results = read_cases(args.cases)
    automated_pass = len(results) == 12 and all(
        bool(result["passed"]) for result in results
    )
    passed = automated_pass and args.visual_review == "pass"
    receipt = {
        "receipt_schema": "zdraw.zimage-gate.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "tree_clean": not bool(git("status", "--porcelain")),
        "hardware": hardware(),
        "engine": file_identity(args.binary),
        "weights": tree_identity(args.weights),
        "pack": file_identity(args.pack),
        "automated_pass": automated_pass,
        "visual_review": {
            "status": args.visual_review,
            "reviewer": args.reviewer,
        },
        "passed": passed,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
