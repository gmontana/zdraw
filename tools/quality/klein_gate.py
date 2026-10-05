#!/usr/bin/env python3
"""Run a fixed FLUX.2 Klein smoke-quality gate.

The gate is deliberately conservative and dependency-free. It runs zdraw on a
small prompt set, parses timing/memory logs, decodes the output PNGs directly,
rejects blank/corrupt files, and writes a durable manifest plus a Markdown
review sheet. It is a smoke gate, not a perceptual substitute for human review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
import zlib
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = Path("runs/quality/klein")
DEFAULT_MODEL = "flux2-klein-4b"
RESULT_RE = re.compile(
    r"result\s+(?P<width>\d+)x(?P<height>\d+)\s+\|\s+"
    r"(?P<steps>\d+)\s+steps\s+\|\s+seed\s+(?P<seed>\d+)\s+\|\s+"
    r"(?P<seconds>[0-9.]+)s\s+\|\s+mem\s+(?P<mem_gb>[0-9.]+)\s+GB"
)
STAGE_RE = re.compile(r"zdraw: (?P<stage>.+?) done \((?P<ms>\d+) ms\)")


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    tags: tuple[str, ...]


DEFAULT_CASES: tuple[Case, ...] = (
    Case(
        "fox_snow_512",
        "a red fox in deep snow, golden hour light, detailed fur, natural photograph",
        512,
        512,
        4,
        7,
        ("animal", "texture", "outdoor"),
    ),
    Case(
        "perfume_marble_512",
        "studio product photo of a clear glass perfume bottle on white marble, softbox reflections",
        512,
        512,
        4,
        19,
        ("product", "glass", "studio"),
    ),
    Case(
        "camera_desk_512",
        "a vintage black camera on a walnut desk, shallow depth of field, warm window light",
        512,
        512,
        4,
        31,
        ("object", "material", "depth"),
    ),
    Case(
        "ceramic_breakfast_512",
        "a white ceramic coffee mug beside a croissant on a linen tablecloth, morning light",
        512,
        512,
        4,
        43,
        ("food", "ceramic", "soft-light"),
    ),
    Case(
        "shoe_catalog_512",
        "catalog photograph of a black running shoe floating on a neutral gray background",
        512,
        512,
        4,
        59,
        ("product", "silhouette", "catalog"),
    ),
    Case(
        "city_rain_512",
        "cinematic street photograph of a rainy city at night, neon reflections on wet pavement",
        512,
        512,
        4,
        71,
        ("scene", "night", "reflection"),
    ),
)

LARGE_CASES: tuple[Case, ...] = (
    Case(
        "fox_snow_1024",
        "a red fox in deep snow, golden hour light, detailed fur, natural photograph",
        1024,
        1024,
        4,
        7,
        ("animal", "texture", "outdoor", "large"),
    ),
    Case(
        "perfume_marble_1024",
        "studio product photo of a clear glass perfume bottle on white marble, softbox reflections",
        1024,
        1024,
        4,
        19,
        ("product", "glass", "studio", "large"),
    ),
)


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None, timeout: int | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def shell(cmd: str, cwd: Path, timeout: int | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        shell=True,
        check=False,
    )


def remote_shell(host: str, command: str, timeout: int | None) -> subprocess.CompletedProcess[str]:
    # OpenSSH concatenates argv after the host into one remote shell command.
    # Quote the bash -lc payload explicitly so `cd repo && ...` stays a single
    # argument to bash instead of becoming `bash -lc cd ...`.
    return run(["ssh", host, "bash", "-lc", shlex.quote(command)], REPO, None, timeout)


def quote_path(path: str | Path) -> str:
    return shlex.quote(str(path))


def quote_remote_path(path: str | Path) -> str:
    text = str(path)
    if text.startswith("~/"):
        return "~/" + shlex.quote(text[2:])
    if text.startswith("$HOME/"):
        return "$HOME/" + shlex.quote(text[6:])
    if text.startswith("${HOME}/"):
        return "${HOME}/" + shlex.quote(text[8:])
    return shlex.quote(text)


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_metrics(text: str) -> dict[str, object]:
    metrics: dict[str, object] = {"stages_ms": {}}
    for match in STAGE_RE.finditer(text):
        stage = match.group("stage").strip().replace(" ", "_")
        metrics["stages_ms"][stage] = int(match.group("ms"))
    if match := RESULT_RE.search(text):
        metrics.update(
            {
                "reported_width": int(match.group("width")),
                "reported_height": int(match.group("height")),
                "reported_steps": int(match.group("steps")),
                "reported_seed": int(match.group("seed")),
                "reported_seconds": float(match.group("seconds")),
                "reported_mem_gb": float(match.group("mem_gb")),
            }
        )
    return metrics


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG")
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("unsupported PNG encoding")
            if bit_depth != 8:
                raise ValueError("unsupported PNG bit depth")
        elif kind == b"IDAT":
            idat.extend(chunk)
        elif kind == b"IEND":
            break
    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("missing IHDR")
    channels_by_type = {0: 1, 2: 3, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"unsupported PNG color type {color_type}")
    channels = channels_by_type[color_type]
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    rows: list[bytes] = []
    offset = 0
    prev = bytearray(stride)
    for _ in range(height):
        if offset >= len(raw):
            raise ValueError("truncated PNG data")
        filt = raw[offset]
        offset += 1
        cur = bytearray(raw[offset : offset + stride])
        offset += stride
        if len(cur) != stride:
            raise ValueError("truncated PNG scanline")
        for i in range(stride):
            left = cur[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filt == 0:
                value = cur[i]
            elif filt == 1:
                value = cur[i] + left
            elif filt == 2:
                value = cur[i] + up
            elif filt == 3:
                value = cur[i] + ((left + up) >> 1)
            elif filt == 4:
                value = cur[i] + paeth(left, up, up_left)
            else:
                raise ValueError(f"unsupported PNG filter {filt}")
            cur[i] = value & 0xFF
        rows.append(bytes(cur))
        prev = cur
    pixels: list[tuple[int, int, int]] = []
    for row in rows:
        for x in range(width):
            base = x * channels
            if color_type == 0:
                v = row[base]
                pixels.append((v, v, v))
            else:
                pixels.append((row[base], row[base + 1], row[base + 2]))
    return width, height, pixels


def image_stats(path: Path, expected_width: int, expected_height: int) -> dict[str, object]:
    width, height, pixels = read_png_rgb(path)
    if not pixels:
        raise ValueError("empty image")
    values = [v for px in pixels for v in px]
    luma = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels]
    sample_step = max(1, len(pixels) // 4096)
    unique_sample = len(set(pixels[::sample_step]))
    band_stats = []
    bands = 4
    for band in range(bands):
        y0 = height * band // bands
        y1 = height * (band + 1) // bands
        band_luma: list[float] = []
        for y in range(y0, y1):
            band_luma.extend(luma[y * width : (y + 1) * width])
        band_stats.append(
            {
                "mean": statistics.fmean(band_luma),
                "stddev": statistics.pstdev(band_luma) if len(band_luma) > 1 else 0.0,
            }
        )
    span = max(values) - min(values)
    rgb_std = statistics.pstdev(values) if len(values) > 1 else 0.0
    luma_std = statistics.pstdev(luma) if len(luma) > 1 else 0.0
    warnings: list[str] = []
    failures: list[str] = []
    if width != expected_width or height != expected_height:
        failures.append(f"size {width}x{height} != expected {expected_width}x{expected_height}")
    if span < 16 or rgb_std < 4.0 or luma_std < 4.0:
        failures.append("blank-like image")
    if unique_sample < 32:
        failures.append("too few sampled colors")
    for idx, stats in enumerate(band_stats):
        if stats["stddev"] < 2.5 and luma_std > 12.0:
            warnings.append(f"flat horizontal band {idx}")
    return {
        "width": width,
        "height": height,
        "mean_rgb": statistics.fmean(values),
        "stddev_rgb": rgb_std,
        "luma_stddev": luma_std,
        "min": min(values),
        "max": max(values),
        "span": span,
        "unique_sample": unique_sample,
        "bands": band_stats,
        "warnings": warnings,
        "failures": failures,
    }


def case_command(
    binary: str, weights: str, zpack: str, case: Case, out_path: str,
    model: str = DEFAULT_MODEL,
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["ZDRAW_KLEIN_ZPACK"] = zpack
    return (
        [
            binary,
            "generate",
            "--model",
            model,
            "--weights",
            weights,
            "--prompt",
            case.prompt,
            "--width",
            str(case.width),
            "--height",
            str(case.height),
            "--steps",
            str(case.steps),
            "--seed",
            str(case.seed),
            "--out",
            out_path,
        ],
        env,
    )


def remote_case_command(
    remote_repo: str, weights: str, zpack: str, case: Case, out_path: str,
    model: str = DEFAULT_MODEL,
) -> str:
    parts = [
        f"cd {quote_remote_path(remote_repo)}",
        f"ZDRAW_KLEIN_ZPACK={quote_remote_path(zpack)}",
        "./zig-out/bin/zdraw",
        "generate",
        "--model",
        quote_path(model),
        "--weights",
        quote_remote_path(weights),
        "--prompt",
        quote_path(case.prompt),
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(case.steps),
        "--seed",
        str(case.seed),
        "--out",
        quote_remote_path(out_path),
    ]
    return parts[0] + " && " + " ".join(parts[1:])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_local_case(args: argparse.Namespace, case: Case, out_dir: Path) -> dict[str, object]:
    image_path = out_dir / f"{case.id}.png"
    cmd, env = case_command(
        "./zig-out/bin/zdraw", args.weights, args.zpack, case, str(image_path), args.model
    )
    if args.dry_run:
        return {
            "case": asdict(case),
            "command": cmd,
            "dry_run": True,
            "passed": True,
            "wall_seconds": 0.0,
            "image": str(image_path),
            "metrics": {},
            "image_stats": None,
            "failures": [],
        }
    start = time.monotonic()
    proc = run(cmd, args.repo, env, args.timeout)
    elapsed = time.monotonic() - start
    log = proc.stdout + proc.stderr
    write_text(out_dir / f"{case.id}.log", log)
    return finalize_case(case, image_path, cmd, proc.returncode, elapsed, log)


def run_remote_case(args: argparse.Namespace, case: Case, local_out: Path, remote_out: str) -> dict[str, object]:
    remote_png = f"{remote_out.rstrip('/')}/{case.id}.png"
    cmd = remote_case_command(
        args.remote_repo, args.weights, args.zpack, case, remote_png, args.model
    )
    if args.dry_run:
        return {
            "case": asdict(case),
            "command": cmd,
            "dry_run": True,
            "passed": True,
            "wall_seconds": 0.0,
            "image": str(local_out / f"{case.id}.png"),
            "metrics": {},
            "image_stats": None,
            "failures": [],
        }
    start = time.monotonic()
    proc = remote_shell(args.remote, cmd, args.timeout)
    elapsed = time.monotonic() - start
    log = proc.stdout + proc.stderr
    write_text(local_out / f"{case.id}.log", log)
    local_png = local_out / f"{case.id}.png"
    if proc.returncode == 0:
        scp = run(["scp", f"{args.remote}:{remote_png}", str(local_png)], REPO, None, args.timeout)
        if scp.returncode != 0:
            log += "\n--- scp ---\n" + scp.stdout + scp.stderr
            write_text(local_out / f"{case.id}.log", log)
            return finalize_case(case, local_png, cmd, scp.returncode, elapsed, log)
    return finalize_case(case, local_png, cmd, proc.returncode, elapsed, log)


def finalize_case(
    case: Case,
    image_path: Path,
    command: list[str] | str,
    exit_code: int,
    elapsed: float,
    log: str,
) -> dict[str, object]:
    failures: list[str] = []
    stats: dict[str, object] | None = None
    if exit_code != 0:
        failures.append(f"command exited {exit_code}")
    if not image_path.exists():
        failures.append("output image missing")
    else:
        try:
            stats = image_stats(image_path, case.width, case.height)
            failures.extend(str(v) for v in stats.get("failures", []))
        except Exception as exc:  # noqa: BLE001 - report image parser failures in manifest.
            failures.append(f"image decode failed: {exc}")
    return {
        "case": asdict(case),
        "command": command,
        "exit_code": exit_code,
        "wall_seconds": elapsed,
        "image": str(image_path),
        "metrics": parse_metrics(log),
        "image_stats": stats,
        "failures": failures,
        "passed": not failures,
    }


def write_summary(out_dir: Path, manifest: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append("# Klein Quality Gate")
    lines.append("")
    lines.append(f"- commit: `{manifest['commit']}`")
    lines.append(f"- model: `{manifest.get('model', DEFAULT_MODEL)}`")
    lines.append(f"- host: `{manifest['host']}`")
    lines.append(f"- passed: `{manifest['passed']}`")
    lines.append(f"- cases: `{manifest['passed_cases']}/{manifest['case_count']}`")
    lines.append("")
    for result in manifest["results"]:
        case = result["case"]
        status = "PASS" if result["passed"] else "FAIL"
        stats = result.get("image_stats") or {}
        metrics = result.get("metrics") or {}
        lines.append(f"## {case['id']} - {status}")
        lines.append("")
        lines.append(f"Prompt: {case['prompt']}")
        lines.append("")
        if "reported_seconds" in metrics:
            lines.append(
                f"- zdraw: {metrics['reported_seconds']:.2f}s, mem {metrics.get('reported_mem_gb', 0.0):.2f} GB"
            )
        lines.append(f"- wall: {result['wall_seconds']:.2f}s")
        stages = metrics.get("stages_ms") or {}
        if stages:
            stage_text = ", ".join(
                f"{name.replace('_', ' ')} {ms / 1000.0:.2f}s" for name, ms in stages.items()
            )
            lines.append(f"- stages: {stage_text}")
        if stats:
            lines.append(
                "- image: {width}x{height}, rgb_std={std:.2f}, luma_std={luma:.2f}, "
                "span={span}, unique_sample={unique}".format(
                    width=stats["width"],
                    height=stats["height"],
                    std=stats["stddev_rgb"],
                    luma=stats["luma_stddev"],
                    span=stats["span"],
                    unique=stats["unique_sample"],
                )
            )
            warnings = stats.get("warnings") or []
            if warnings:
                lines.append(f"- warnings: {', '.join(warnings)}")
        failures = result.get("failures") or []
        if failures:
            lines.append(f"- failures: {', '.join(failures)}")
        image = Path(result["image"]).name
        if (out_dir / image).exists():
            lines.append("")
            lines.append(f"![{case['id']}]({image})")
        lines.append("")
    write_text(out_dir / "summary.md", "\n".join(lines))


def current_commit(repo: Path) -> str:
    proc = run(["git", "rev-parse", "--short", "HEAD"], repo, None, 10)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def execution_commit(args: argparse.Namespace) -> str:
    if not args.remote:
        return current_commit(args.repo)
    if args.dry_run:
        return "dry-run"
    cmd = f"cd {quote_remote_path(args.remote_repo)} && git rev-parse --short HEAD"
    proc = remote_shell(args.remote, cmd, 30)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def select_cases(args: argparse.Namespace) -> list[Case]:
    cases = list(DEFAULT_CASES)
    if args.include_1024 or args.case:
        cases.extend(LARGE_CASES)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.id in wanted]
        found = {case.id for case in cases}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit("unknown case(s): " + ", ".join(missing))
    elif args.quick:
        cases = cases[:2]
    if args.steps:
        cases = [dataclasses.replace(case, steps=args.steps) for case in cases]
    if not cases:
        raise SystemExit("no cases selected")
    return cases


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO, help="Local zdraw repo path")
    parser.add_argument("--weights", default=os.environ.get("KLEIN_SNAP", ""), help="Klein snapshot path")
    parser.add_argument("--zpack", default=os.environ.get("KLEIN_ZPACK", ""), help="Klein zpack sidecar path")
    parser.add_argument(
        "--model",
        default=os.environ.get("KLEIN_MODEL", DEFAULT_MODEL),
        choices=("flux2-klein-4b",), help="Supported Klein CLI model",
    )
    parser.add_argument("--out", type=Path, default=None, help="Local output directory")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per case in seconds")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Override every case's step count (the base model samples at 50); default keeps the cases' 4",
    )
    parser.add_argument("--quick", action="store_true", help="Run the first two 512px cases")
    parser.add_argument("--include-1024", action="store_true", help="Also run two 1024px safety cases")
    parser.add_argument("--case", action="append", help="Run one case id; may be repeated")
    parser.add_argument("--list", action="store_true", help="List selected cases and exit")
    parser.add_argument("--dry-run", action="store_true", help="Write no images; print commands into manifest")
    parser.add_argument("--no-build", action="store_true", help="Do not run zig build before the gate")
    parser.add_argument("--remote", default="", help="SSH host for remote M4 execution")
    parser.add_argument("--remote-repo", default="~/repos/zdraw", help="Remote repo path")
    parser.add_argument("--remote-out", default="", help="Remote output directory")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.repo = args.repo.resolve()
    cases = select_cases(args)
    if args.list:
        for case in cases:
            print(f"{case.id:24s} {case.width}x{case.height} seed={case.seed} {case.prompt}")
        return 0
    if not args.weights:
        raise SystemExit("missing --weights or KLEIN_SNAP")
    if not args.zpack:
        raise SystemExit("missing --zpack or KLEIN_ZPACK")
    out_dir = args.out or (args.repo / DEFAULT_OUT_ROOT / timestamp())
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_build and not args.dry_run:
        if args.remote:
            proc = remote_shell(args.remote, f"cd {quote_remote_path(args.remote_repo)} && zig build", args.timeout)
        else:
            proc = run(["zig", "build"], args.repo, None, args.timeout)
        write_text(out_dir / "build.log", proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"zig build failed; see {out_dir / 'build.log'}")

    remote_out = args.remote_out or f"{args.remote_repo.rstrip('/')}/runs/quality/klein/{timestamp()}"
    if args.remote and not args.dry_run:
        proc = remote_shell(args.remote, f"mkdir -p {quote_remote_path(remote_out)}", args.timeout)
        if proc.returncode != 0:
            raise SystemExit(proc.stdout + proc.stderr)

    results = []
    for case in cases:
        if args.remote:
            results.append(run_remote_case(args, case, out_dir, remote_out))
        else:
            results.append(run_local_case(args, case, out_dir))

    passed_cases = sum(1 for result in results if result["passed"])
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": execution_commit(args),
        "model": args.model,
        "host": args.remote or "local",
        "repo": str(args.repo),
        "remote_repo": args.remote_repo if args.remote else "",
        "remote_out": remote_out if args.remote else "",
        "weights": args.weights,
        "zpack": args.zpack,
        "case_count": len(results),
        "passed_cases": passed_cases,
        "passed": passed_cases == len(results),
        "results": results,
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_summary(out_dir, manifest)
    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
