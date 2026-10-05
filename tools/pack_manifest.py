#!/usr/bin/env python3
"""The manifest beside a .zpack: what the file holds (every weight class, its
code width, parameters and bytes) and what it measured (the ledger rows the
pack's tier is certified and evaluated by, quoted by id). The pack states its
own cost; an application can choose a size knowing what it gives up.

usage: pack_manifest.py --pack FILE --tier NAME [--rows id,id,...] [--ledger PATH] [--out FILE]
       pack_manifest.py --self-test
Reads only the entry headers (a 2.7 GB pack takes well under a second)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import subprocess
import sys
from datetime import date
from pathlib import Path

MAGIC = b"ZDWPACK1"
ALIGN = 16
FAMILY = {1: "main", 2: "noise", 3: "context", 4: "flux2", 5: "text"}
KIND = {1: "ffn_down", 2: "ffn_gate", 3: "ffn_up", 4: "q", 5: "k", 6: "v", 7: "proj",
        8: "ffn_gateup", 9: "weight", 10: "raw", 11: "norm"}
DOUBLE = ["to_q", "to_k", "to_v", "add_q", "add_k", "add_v", "to_out", "to_add_out",
          "ff_in", "ff_out", "ffc_in", "ffc_out"]
SINGLE = ["qkv_mlp", "out"]
GLOBAL = ["x_embed", "context_embed", "mod_img", "mod_txt", "mod_single", "time_in_1",
          "time_in_2", "proj_out", "norm_out"]
QWEN = ["input_norm", "post_norm", "q", "k", "v", "o", "q_norm", "k_norm", "gate", "up", "down"]
RAW_DTYPE = {1: "f32", 2: "f16", 3: "bf16", 4: "u8"}


def entry_bits(group: int) -> int:
    hi = group >> 16
    if hi:
        return hi
    return 16 if group == 0 else 8


def class_of(family: str, layer: int, kind: str) -> str:
    """The weight class an entry belongs to, from its family and slot."""
    if family == "flux2":
        if layer >= 1000:
            return "single.norm" if kind == "norm" else "single." + SINGLE[(layer - 1000) % 2]
        if layer >= 100:
            return "double.norm" if kind == "norm" else "double." + DOUBLE[(layer - 100) % 12]
        return "global." + (GLOBAL[layer] if layer < len(GLOBAL) else str(layer))
    if family == "text":
        if layer == 0:
            return "text.embed"
        if layer == 1:
            return "text.final_norm"
        return "text." + QWEN[(layer - 100) % 16]
    return f"{family}.{kind}"


def read_entries(f) -> tuple[int, list[dict]]:
    head = f.read(16)
    if head[:8] != MAGIC:
        raise SystemExit("not a zpack: bad magic")
    fmt, count = struct.unpack("<II", head[8:16])
    pos = 16
    entries = []
    for _ in range(count):
        if fmt >= 3:
            raw = f.read(4 + 1 + 1 + 4 + 4 + 4 + 8)
            layer, family, kind, rows, cols, group, length = struct.unpack("<IBBIIIQ", raw)
        else:
            raw = f.read(4 + 1 + 4 + 4 + 4 + 8)
            layer, kind, rows, cols, group, length = struct.unpack("<IBIIIQ", raw)
            family = 1
        pos += len(raw)
        pad = (-pos) % ALIGN
        f.seek(pad + length, io.SEEK_CUR)
        pos += pad + length
        entries.append({"family": FAMILY.get(family, str(family)), "layer": layer,
                        "kind": KIND.get(kind, str(kind)), "rows": rows, "cols": cols,
                        "group": group, "bytes": length})
    return fmt, entries


def summarise(entries: list[dict]) -> list[dict]:
    classes: dict[tuple, dict] = {}
    for e in entries:
        packed = e["kind"] in ("weight", "q", "k", "v", "proj", "ffn_down", "ffn_gate", "ffn_up", "ffn_gateup")
        bits = entry_bits(e["group"]) if packed else None
        dtype = None if packed else RAW_DTYPE.get(e["group"] >> 16, "?")
        name = class_of(e["family"], e["layer"], e["kind"])
        key = (name, bits, dtype)
        c = classes.setdefault(key, {"class": name, "bits": bits, "dtype": dtype, "count": 0,
                                     "params": 0, "bytes": 0})
        c["count"] += 1
        c["params"] += e["rows"] * e["cols"]
        c["bytes"] += e["bytes"]
    return sorted(classes.values(), key=lambda c: c["class"])


def ledger_rows(path: Path, ids: list[str]) -> dict:
    want = set(ids)
    found = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id") in want:
                found[row["id"]] = {k: row[k] for k in ("date", "what", "measured", "verdict") if k in row}
    missing = want - set(found)
    if missing:
        raise SystemExit(f"ledger rows not found: {sorted(missing)}")
    return found


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def manifest(pack: Path, tier: str, rows: dict, digest: str) -> dict:
    with open(pack, "rb") as fh:
        fmt, entries = read_entries(fh)
    classes = summarise(entries)
    packed = [c for c in classes if c["bits"] is not None]
    params = sum(c["params"] for c in packed)
    bits = sum(c["params"] * c["bits"] for c in packed) / params if params else None
    return {
        "schema_version": 1,
        "file": pack.name,
        "bytes": pack.stat().st_size,
        "sha256": digest,
        "format_version": fmt,
        "tier": tier,
        "entries": len(entries),
        "packed_params": params,
        "nominal_bits_per_packed_param": None if bits is None else round(bits, 3),
        "stored_bits_per_packed_param": None if not params else round(8 * sum(c["bytes"] for c in packed) / params, 3),
        "classes": classes,
        "measured": rows,
        "engine_commit": commit(),
        "generated": date.today().isoformat(),
    }


def synthetic_pack() -> bytes:
    out = bytearray(MAGIC + struct.pack("<II", 4, 4))
    def put(layer, family, kind, rows, cols, group, payload):
        out.extend(struct.pack("<IBBIIIQ", layer, family, kind, rows, cols, group, len(payload)))
        while len(out) % ALIGN:
            out.append(0)
        out.extend(payload)
    put(100 + 2 * 12 + 2, 4, 9, 2, 64, 64 | (6 << 16), bytes(2 * 48 + 2 * 2))   # double 2 to_v at 6 bits
    put(1000 + 3 * 2 + 1, 4, 9, 2, 64, 64 | (2 << 16), bytes(2 * 16 + 2 * 2))   # single 3 out at 2 bits
    put(0, 5, 10, 4, 64, 3 << 16, bytes(4 * 64 * 2))                              # text embed bf16 verbatim
    put(1000 + 3 * 2 + 0, 4, 11, 1, 64, 3 << 16, bytes(64 * 2))                    # single 3 norm_q verbatim
    return bytes(out)


def self_test() -> int:
    fmt, entries = read_entries(io.BytesIO(synthetic_pack()))
    assert fmt == 4 and len(entries) == 4, entries
    classes = {c["class"]: c for c in summarise(entries)}
    assert classes["double.to_v"]["bits"] == 6 and classes["double.to_v"]["params"] == 128
    assert classes["single.out"]["bits"] == 2
    assert classes["text.embed"]["dtype"] == "bf16" and classes["text.embed"]["bits"] is None
    assert classes["single.norm"]["dtype"] == "bf16" and classes["single.norm"]["params"] == 64
    assert entry_bits(0) == 16 and entry_bits(64) == 8 and entry_bits(64 | (4 << 16)) == 4
    print("pack_manifest self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack")
    parser.add_argument("--tier")
    parser.add_argument("--rows", default="")
    parser.add_argument("--ledger", default="docs/experiments.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.pack or not args.tier:
        parser.error("--pack and --tier are required")
    pack = Path(args.pack)
    ids = [r for r in args.rows.split(",") if r]
    rows = ledger_rows(Path(args.ledger), ids) if ids else {}
    m = manifest(pack, args.tier, rows, sha256_of(pack))
    text = json.dumps(m, indent=2) + "\n"
    out = Path(args.out) if args.out else pack.with_suffix(pack.suffix + ".manifest.json")
    out.write_text(text, encoding="utf-8")
    print(f"pack_manifest: wrote {out} ({m['bytes']} bytes, {m['entries']} entries, "
          f"{m['nominal_bits_per_packed_param']} nominal bits/param)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
