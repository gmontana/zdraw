#!/usr/bin/env python3
"""Retain every image/log in a zdraw resolution quality or timing campaign.

Run timing mode through tools/perf_when_quiet.sh on an idle GPU. Timing uses
fresh processes with cached weight pages, three rounds in alternating order,
and the same external memory instrument as competitive_bench.py. Quality mode
adds two different prompts; its times are diagnostic, not benchmark evidence.
Both modes leave visual review open. Requires NumPy and Pillow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competitive_core import FOOTPRINT_RE, RSS_RE, ProcessResult, write_log
from content_check import check
from zimage_receipt import file_identity


SIZES = [(n, n) for n in (64, 128, 256, 512, 768, 1024, 1536, 2048)] + [
    (1024, 768), (768, 1024), (544, 800),
]
MODELS = {
    "flux2-klein-4b": ("FLUX.2-klein-4B", "zdraw-klein-w16.zpack", "ZDRAW_KLEIN_ZPACK"),
    "z-image-turbo": ("Z-Image-Turbo", "zdraw-w16.zpack", "ZDRAW_ZPACK"),
}
PROMPTS = {
    "fox": "a red fox sitting in deep snow, golden hour light",
    "portrait": "editorial portrait of a fashion designer in a bright studio",
    "market": "a crowded spice market stall, dozens of open sacks, fine texture",
}
REFERENCES = {
    "flux2-klein-4b": "afa4c8fb1ecbd1ecadee4042bfa62351eacbe3a841f67f71a3f2da65bbc505bb",
    "z-image-turbo": "c2ac10ec694ef04a01517fb11bd808c025e50c4e89a9a45eabe69f1109c59bb6",
}


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def measured(command: list[str], env: dict[str, str], timeout: int) -> ProcessResult:
    """Kill the whole owned process group on timeout/cancellation, including time's child."""
    started = time.perf_counter()
    proc = subprocess.Popen(
        ["/usr/bin/time", "-l", *command], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        stderr += "\nCAMPAIGN_TIMEOUT_OR_INTERRUPTION\n"
        # Keep the failed case's log; the caller aborts after recording it.
    wall = time.perf_counter() - started
    rss = RSS_RE.search(stderr)
    footprint = FOOTPRINT_RE.search(stderr)
    return ProcessResult(
        proc.returncode, wall, stdout, stderr,
        int(rss.group(1)) if rss else None,
        int(footprint.group(1)) if footprint else None,
    )


def image_record(path: Path, width: int, height: int) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        size = list(rgb.size)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    checker, std, problems = check(path)
    if size != [width, height]:
        problems.append(f"wrong dimensions: {size}")
    return {
        "file": file_identity(path), "size": size, "pixel_sha256": digest,
        "checker_score": checker, "luma_std": std, "problems": problems,
    }


def schedule(mode: str) -> list[tuple[str, int, int, str, str]]:
    cases = [(model, w, h) for w, h in SIZES for model in MODELS]
    if mode == "quality":
        return [(model, w, h, prompt, "quality")
                for model, w, h in cases for prompt in ("portrait", "market")]
    work = [(model, 1024, 1024, "fox", "warmup") for model in MODELS]
    orders = [cases, list(reversed(cases)), cases[len(cases)//2:] + cases[:len(cases)//2]]
    for number, ordered in enumerate(orders, 1):
        work.extend((model, w, h, "fox", f"round{number}") for model, w, h in ordered)
    return work


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "quality"), required=True)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/zdraw"))
    parser.add_argument("--models-home", type=Path, default=Path.home()/".zdraw/models")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("generation and /usr/bin/time -l require macOS")
    if args.timeout < 1:
        parser.error("timeout must be positive")
    binary = args.binary.resolve()
    metallib = binary.parent.parent/"lib/steel.metallib"
    if not metallib.is_file():
        metallib = binary.parent/"lib/steel.metallib"
    models = {}
    for model, (folder, pack_name, _) in MODELS.items():
        weights = (args.models_home/folder).resolve()
        models[model] = {"weights": str(weights), "pack": file_identity(weights/pack_name)}
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    work = schedule(args.mode)
    manifest = {
        "schema": "zdraw.resolution-sweep.v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "source_commit": capture(["git", "rev-parse", "HEAD"]),
        "harness": file_identity(Path(__file__)), "binary": file_identity(binary),
        "version": capture([str(binary), "--version"]), "metallib": file_identity(metallib),
        "host": {"name": platform.node(), "macos": platform.mac_ver()[0],
                 "chip": capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
                 "ram_bytes": int(capture(["sysctl", "-n", "hw.memsize"]))},
        "models": models, "steps": 4, "seed": 46, "prompts": PROMPTS,
        "sizes": SIZES, "planned": len(work), "completed": 0,
        "regime": "fresh processes with cached weight pages" if args.mode == "timing" else "functional only; no performance claims",
        "thermal_before": capture(["pmset", "-g", "therm"]),
        "swap_before": capture(["sysctl", "-n", "vm.swapusage"]),
        "status": "running", "visual_review": "not-reviewed", "results": [],
    }
    manifest_path = out/"manifest.json"
    write_manifest(manifest_path, manifest)
    # Exclude inherited experiment flags; explicitly record the whole render environment.
    base_env = {key: value for key, value in os.environ.items() if not key.startswith("ZDRAW_")}
    for number, (model, width, height, prompt, phase) in enumerate(work, 1):
        name = f"{number:03d}-{model}-{width}x{height}-{prompt}-{phase}"
        output = out/f"{name}.png"
        command = [str(binary), "generate", "--model", model, "--weights", models[model]["weights"],
                   "--prompt", PROMPTS[prompt], "--width", str(width), "--height", str(height),
                   "--steps", "4", "--seed", "46", "--out", str(output)]
        if model == "z-image-turbo":
            command += ["--profile", "product"]
        render_env = {"ZDRAW_PROGRESS": "quiet", "ZDRAW_METRICS": "1",  # memory comes from /usr/bin/time -l; ZDRAW_MEMTRACE would add its exact page walk to the wall time
                      "ZDRAW_REQUIRE_ZPACK": "1", MODELS[model][2]: models[model]["pack"]["path"]}
        print(f"START {number}/{len(work)} {name}", flush=True)
        load = list(os.getloadavg())
        proc = measured(command, base_env | render_env, args.timeout)
        log = out/f"{name}.log"
        write_log(log, command, [proc])
        row = {"case": name, "model": model, "width": width, "height": height,
               "prompt": prompt, "phase": phase, "command": command, "env": render_env,
               "load_before": load, "load_after": list(os.getloadavg()),
               "returncode": proc.returncode, "seconds": proc.wall_seconds,
               "peak_rss_bytes": proc.peak_rss_bytes, "peak_footprint_bytes": proc.peak_footprint_bytes,
               "log": file_identity(log), "problems": []}
        if proc.returncode != 0:
            row["problems"].append(f"generation exited {proc.returncode}")
        if not proc.peak_rss_bytes or not proc.peak_footprint_bytes:
            row["problems"].append("external memory measurement missing")
        try:
            row["image"] = image_record(output, width, height)
            row["problems"] += row["image"]["problems"]
            if prompt == "fox" and (width, height) == (1024, 1024):
                row["certified_match"] = row["image"]["pixel_sha256"] == REFERENCES[model]
                if not row["certified_match"]:
                    row["problems"].append("1024px certified reference mismatch")
        except (OSError, ValueError) as exc:
            row["problems"].append(f"image unavailable: {exc}")
        manifest["results"].append(row)
        manifest["completed"] = number
        write_manifest(manifest_path, manifest)
        print(f"DONE {name} {proc.wall_seconds:.2f}s problems={row['problems']}", flush=True)
        if "CAMPAIGN_TIMEOUT_OR_INTERRUPTION" in proc.stderr:
            break
    manifest["thermal_after"] = capture(["pmset", "-g", "therm"])
    manifest["swap_after"] = capture(["sysctl", "-n", "vm.swapusage"])
    failed = manifest["completed"] != manifest["planned"] or any(r["problems"] for r in manifest["results"])
    manifest["status"] = "automated-fail" if failed else "automated-pass; visual review pending"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print(manifest["status"], flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
