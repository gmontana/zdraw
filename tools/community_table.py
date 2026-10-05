#!/usr/bin/env python3
"""Community benchmark cards: validate `community/results.jsonl` and render
`community/RESULTS.md`.

A card is the JSON line `zdraw bench --card` prints (schema_version 1), plus
two optional submitter keys: `handle` (GitHub login) and `date` (YYYY-MM-DD).
Only the census case enters the table (1024x1024, 4 steps, seed 46, the
certified prompt), so every row is comparable.

  python3 tools/community_table.py --check community/results.jsonl
  python3 tools/community_table.py --render community/results.jsonl --out community/RESULTS.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROMPT = "a red fox sitting in deep snow, golden hour light"
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
CARD_KEYS = {
    "schema_version", "chip", "ram_gb", "macos", "zdraw_version", "commit", "model", "profile",
    "size", "steps", "seed", "prompt_sha256", "wall_s", "runs_s", "phases", "gpu_active_s",
    "rss_gb", "footprint_gb", "output_sha256", "expected_sha256", "hash_match", "routes",
    "fallbacks", "thermal_before", "thermal_after", "env_overrides", "safety",
}
OPTIONAL_KEYS = {"handle", "date", "png_sha256"}
MODELS = {"z-image-turbo", "flux2-klein-4b", "flux2-klein-base-4b"}
PROFILES = {"product", "strict", "-"}
ATTENTION = {"steel", "mfa"}
GEMM = {"metal4-matmul2d", "simdgroup-direct", "ours-f16", "exact"}
DECODER = {"winograd", "direct", "reference-f32"}
THERMAL = {"unknown", "nominal", "fair", "serious", "critical"}
TEXT = re.compile(r"^[A-Za-z0-9 ._+/-]{0,64}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
OVERRIDE = re.compile(r"^ZDRAW_[A-Z0-9_]+=[A-Za-z0-9._/-]*$")
HANDLE = re.compile(r"^[A-Za-z0-9-]{1,39}$")


def positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def check_card(card: dict, index: int) -> list[str]:
    e: list[str] = []
    keys = set(card)
    missing = CARD_KEYS - keys
    unknown = keys - CARD_KEYS - OPTIONAL_KEYS
    if missing:
        e.append(f"missing keys {sorted(missing)}")
    if unknown:
        e.append(f"unknown keys {sorted(unknown)}")
    if e:
        return e
    if card["schema_version"] != 1:
        e.append("schema_version must be 1")
    if card["size"] != 1024 or card["steps"] != 4 or card["seed"] != 46:
        e.append("only the census case (1024, 4 steps, seed 46) enters the table")
    if card["prompt_sha256"] != PROMPT_SHA:
        e.append("prompt_sha256 is not the census prompt")
    if not isinstance(card["hash_match"], bool):
        e.append("hash_match must be true or false (a card without a certified entry cannot enter)")
    if card["model"] not in MODELS:
        e.append(f"model {card['model']!r} not accepted")
    if card["profile"] not in PROFILES:
        e.append(f"profile {card['profile']!r} not accepted")
    for key in ("wall_s", "gpu_active_s", "rss_gb", "footprint_gb", "ram_gb"):
        if not positive(card[key]):
            e.append(f"{key} must be a positive number")
    if not isinstance(card["macos"], int) or isinstance(card["macos"], bool) or card["macos"] < 12:
        e.append("macos must be the major version as an integer")
    if not isinstance(card["runs_s"], list) or not card["runs_s"] or not all(positive(r) for r in card["runs_s"]):
        e.append("runs_s must be a non-empty list of positive numbers")
    phases = card["phases"]
    if not isinstance(phases, dict) or set(phases) != {"encode_s", "denoise_s", "decode_s"}:
        e.append("phases must have encode_s, denoise_s, decode_s")
    elif not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in phases.values()):
        e.append("phase seconds must be finite and >= 0")
    if not HEX64.match(str(card["output_sha256"])):
        e.append("output_sha256 must be 64 hex characters")
    if card["expected_sha256"] is not None and not HEX64.match(str(card["expected_sha256"])):
        e.append("expected_sha256 must be 64 hex characters or null")
    routes = card["routes"]
    if not isinstance(routes, dict) or set(routes) != {"attention", "gemm", "decoder"}:
        e.append("routes must have attention, gemm, decoder")
    else:
        if routes["attention"] not in ATTENTION:
            e.append(f"routes.attention {routes['attention']!r} not accepted")
        if routes["gemm"] not in GEMM:
            e.append(f"routes.gemm {routes['gemm']!r} not accepted")
        if routes["decoder"] not in DECODER:
            e.append(f"routes.decoder {routes['decoder']!r} not accepted")
    fb = card["fallbacks"]
    if not isinstance(fb, dict) or set(fb) != {"steel", "mpp", "mps"} or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in fb.values()
    ):
        e.append("fallbacks must have non-negative integers steel, mpp, mps")
    for key in ("thermal_before", "thermal_after"):
        if card[key] not in THERMAL:
            e.append(f"{key} {card[key]!r} not accepted")
    for key in ("chip", "zdraw_version", "commit", "model", "profile"):
        if not isinstance(card[key], str) or not TEXT.match(card[key]):
            e.append(f"{key} must be short plain text")
    if not isinstance(card["env_overrides"], list) or not all(
        isinstance(o, str) and OVERRIDE.match(o) for o in card["env_overrides"]
    ):
        e.append("env_overrides must be NAME=value strings for ZDRAW_ flags")
    if card["safety"] not in ("on", "off"):
        e.append("safety must be on or off")
    if "handle" in card and not (isinstance(card["handle"], str) and HANDLE.match(card["handle"])):
        e.append("handle must be a GitHub login")
    if "date" in card:
        try:
            dt.date.fromisoformat(str(card["date"]))
        except ValueError:
            e.append("date must be YYYY-MM-DD")
    return e


def load(path: Path) -> tuple[list[dict], list[str]]:
    cards: list[dict] = []
    errors: list[str] = []
    seen: set[tuple] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            card = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: not JSON ({exc})")
            continue
        if not isinstance(card, dict):
            errors.append(f"line {number}: not a JSON object")
            continue
        for problem in check_card(card, number):
            errors.append(f"line {number}: {problem}")
        key = (card.get("chip"), card.get("commit"), card.get("model"), card.get("profile"),
               card.get("output_sha256"), card.get("wall_s"))
        if key in seen:
            errors.append(f"line {number}: duplicate card")
        seen.add(key)
        cards.append(card)
    return cards, errors


def render(cards: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cards:
        groups[(c["chip"], int(round(c["ram_gb"])), c["model"], c["profile"])].append(c)
    lines = [
        "# Community benchmark cards",
        "",
        "Every row is `zdraw bench --card` on someone's Mac: the certified prompt at",
        "1024x1024, 4 steps, seed 46, one cold process. `hash OK` counts cards whose",
        "output matched the certified hash for that model and profile (a mismatch",
        "usually means a fallback route or an env override - the card says which).",
        "Walls are medians of cold one-shots; warm runs are in the raw cards.",
        f"Rendered from `results.jsonl` ({len(cards)} cards).",
        "",
        "| chip | RAM | model | profile | n | wall s (median) | RSS GB (median) | hash OK | routes | zdraw | contributors |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(groups):
        chip, ram, model, profile = key
        rows = groups[key]
        walls = statistics.median(r["wall_s"] for r in rows)
        rss = statistics.median(r["rss_gb"] for r in rows)
        ok = sum(1 for r in rows if r["hash_match"])
        routes = sorted({f"{r['routes']['attention']}/{r['routes']['gemm']}/{r['routes']['decoder']}" for r in rows})
        versions = sorted({r["zdraw_version"] for r in rows})
        handles = sorted({"@" + r["handle"] for r in rows if r.get("handle")})
        lines.append(
            f"| {chip} | {ram} GB | {model} | {profile} | {len(rows)} | {walls:.1f} | {rss:.2f} | "
            f"{ok}/{len(rows)} | {', '.join(routes)} | {', '.join(versions)} | {', '.join(handles) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="JSONL")
    parser.add_argument("--render", metavar="JSONL")
    parser.add_argument("--out", metavar="MD")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    path = Path(args.check or args.render or "")
    if not path.is_file():
        parser.error("give --check JSONL or --render JSONL")
    cards, errors = load(path)
    if errors:
        print("community_table: FAIL")
        for err in errors:
            print(f"  {err}")
        return 1
    if args.check:
        print(f"community_table: ok ({len(cards)} cards)")
    if args.render:
        text = render(cards)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"community_table: wrote {args.out}")
        else:
            sys.stdout.write(text)
    return 0


def sample_card() -> dict:
    return {
        "schema_version": 1, "chip": "Apple M4 Max", "ram_gb": 128.0, "macos": 26,
        "zdraw_version": "0.1.0", "commit": "5d4ea75", "model": "flux2-klein-4b", "profile": "-",
        "size": 1024, "steps": 4, "seed": 46, "prompt_sha256": PROMPT_SHA, "wall_s": 12.2,
        "runs_s": [12.2, 11.9], "phases": {"encode_s": 0.38, "denoise_s": 10.7, "decode_s": 0.77},
        "gpu_active_s": 11.4, "rss_gb": 4.2, "footprint_gb": 4.9, "output_sha256": "f" * 64,
        "expected_sha256": "f" * 64, "hash_match": True,
        "routes": {"attention": "steel", "gemm": "metal4-matmul2d", "decoder": "winograd"},
        "fallbacks": {"steel": 0, "mpp": 0, "mps": 0}, "thermal_before": "nominal",
        "thermal_after": "nominal", "env_overrides": [], "safety": "on", "handle": "gmontana",
        "date": "2026-08-28",
    }


def self_test() -> int:
    good = sample_card()
    assert check_card(good, 1) == [], check_card(good, 1)
    bad = dict(good, hash_match=None)
    assert any("hash_match" in m for m in check_card(bad, 1))
    bad = dict(good, size=512)
    assert any("census" in m for m in check_card(bad, 1))
    bad = dict(good, chip="<script>")
    assert any("chip" in m for m in check_card(bad, 1))
    bad = dict(good, notes="free text")
    assert any("unknown keys" in m for m in check_card(bad, 1))
    assert "| Apple M4 Max | 128 GB | flux2-klein-4b | - | 1 | 12.2 |" in render([good])
    print("community_table self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
