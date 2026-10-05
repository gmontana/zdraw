#!/usr/bin/env python3
"""Inventory and execution adapters for local diffusion engines."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from tools.competitive_core import (
        IRIS_RE,
        RUN_RE,
        Case,
        Inventory,
        ProcessResult,
        Result,
        finish_result,
        named_peak_footprint,
        timed_process,
    )
except ModuleNotFoundError:
    from competitive_core import (
        IRIS_RE,
        RUN_RE,
        Case,
        Inventory,
        ProcessResult,
        Result,
        finish_result,
        named_peak_footprint,
        timed_process,
    )

ENGINE_NAMES = (
    "zdraw",
    "drawthings",
    "drawthings-app",
    "iris",
    "stable-diffusion-cpp",
    "mflux",
    "diffusers",
    "ollama",
)
PRIMARY_COMPETITORS = (
    "drawthings",
    "drawthings-app",
    "iris",
    "stable-diffusion-cpp",
    "mflux",
    "ollama",
)


def command_path(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(value)


def existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def probe_output(command: list[str]) -> str:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout + "\n" + proc.stderr).strip()


def probe(command: list[str]) -> str:
    text = probe_output(command)
    return text.splitlines()[0] if text else ""


def git_version(binary: str) -> str:
    parent = Path(binary).resolve().parent
    value = probe(["git", "-C", str(parent), "rev-parse", "HEAD"])
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else ""


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def artifact_version(binary: str, reported: str = "") -> str:
    source = git_version(binary)
    labels = []
    if reported:
        labels.append(reported)
    if source:
        labels.append(f"repo-head:{source}")
    labels.append(f"sha256:{file_digest(binary)}")
    return "; ".join(labels)


def drawthings_app_version(app: Path) -> str:
    info_path = app / "Contents" / "Info.plist"
    executable = app / "Contents" / "MacOS" / "DrawThings"
    if not info_path.is_file() or not executable.is_file():
        return ""
    try:
        with info_path.open("rb") as stream:
            info = plistlib.load(stream)
    except (OSError, plistlib.InvalidFileException):
        return ""
    version = str(info.get("CFBundleShortVersionString", ""))
    return artifact_version(str(executable), version)


def drawthings_app_config(url: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=2) as response:
            value = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return value if isinstance(value, dict) else None


def zdraw_source_is_newer(binary: str) -> bool:
    root_text = probe(["git", "-C", str(Path(binary).parent), "rev-parse", "--show-toplevel"])
    if not root_text:
        return False
    root = Path(root_text)
    binary_mtime = Path(binary).stat().st_mtime_ns
    candidates = [root / "build.zig", root / "build.zig.zon"]
    source_dir = root / "src"
    if source_dir.is_dir():
        for path in source_dir.rglob("*"):
            if path.is_file():
                candidates.append(path)
    return any(path.is_file() and path.stat().st_mtime_ns > binary_mtime for path in candidates)


def python_version(python: str, package: str) -> str:
    code = (
        "import importlib.metadata as m; "
        f"print(m.version({package!r}))"
    )
    return probe([python, "-c", code])


def model_weights(args: argparse.Namespace) -> Path | None:
    value = args.zimage_weights if args.model == "z-image-turbo" else args.klein_weights
    return existing_path(value)


def effective_steps(engine: str, args: argparse.Namespace, case: Case) -> int:
    if args.settings == "matched" or case.model == "flux2-klein-4b":
        return case.steps
    return {
        "zdraw": 4,
        "mflux": 9,
        "iris": 9,
        "drawthings": 8,
        "drawthings-app": 8,
        "stable-diffusion-cpp": 8,
        "diffusers": 9,
        "ollama": 8,
    }[engine]


def drawthings_model(args: argparse.Namespace) -> str:
    if args.drawthings_model:
        return args.drawthings_model
    if args.model == "z-image-turbo":
        suffix = "f16" if args.tier == "strict" else "q6p"
        return f"z_image_turbo_1.0_{suffix}.ckpt"
    suffix = "f16" if args.tier == "strict" else "q6p"
    return f"flux_2_klein_4b_{suffix}.ckpt"


def discover(args: argparse.Namespace) -> dict[str, Inventory]:
    weights = model_weights(args)
    items: dict[str, Inventory] = {}

    zdraw = command_path(args.zdraw_bin)
    zdraw_stale = bool(zdraw and zdraw_source_is_newer(zdraw))
    if zdraw_stale:
        # A `zig build` install keeps the cached artifact's mtime, so a
        # fresh checkout looks newer than a correct binary. Rebuild once
        # (the binary digest is recorded on the row either way) and only
        # refuse if that build fails.
        root_text = probe(["git", "-C", str(Path(zdraw).parent), "rev-parse", "--show-toplevel"])
        built = subprocess.run(["zig", "build"], cwd=root_text or None, capture_output=True, text=True)
        if built.returncode == 0:
            Path(zdraw).touch()
            zdraw_stale = zdraw_source_is_newer(zdraw)
    items["zdraw"] = Inventory(
        "zdraw",
        zdraw is not None,
        zdraw is not None and weights is not None and not zdraw_stale,
        artifact_version(zdraw, probe([zdraw, "--version"])) if zdraw else "",
        (
            "binary is older than engine source; run `zig build`"
            if zdraw_stale
            else "" if zdraw and weights else "binary or selected model weights are missing"
        ),
    )

    dt = command_path(args.drawthings_bin)
    dt_dir = existing_path(args.drawthings_models_dir)
    dt_file = dt_dir / drawthings_model(args) if dt_dir else None
    items["drawthings"] = Inventory(
        "drawthings",
        dt is not None,
        dt is not None and dt_file is not None and dt_file.is_file(),
        artifact_version(dt, probe([dt, "--version"])) if dt else "",
        "" if dt and dt_file and dt_file.is_file() else "CLI or selected local checkpoint is missing",
    )

    dt_app = existing_path(args.drawthings_app)
    dt_app_config = drawthings_app_config(args.drawthings_app_url) if dt_app else None
    items["drawthings-app"] = Inventory(
        "drawthings-app",
        dt_app is not None,
        dt_app is not None and dt_app_config is not None,
        drawthings_app_version(dt_app) if dt_app else "",
        (
            f"release app is installed but its HTTP server is not reachable at "
            f"{args.drawthings_app_url}"
            if dt_app
            else "release app is not installed"
        )
        if dt_app_config is None
        else "",
    )

    iris = command_path(args.iris_bin)
    items["iris"] = Inventory(
        "iris",
        iris is not None,
        iris is not None and weights is not None,
        artifact_version(iris) if iris else "",
        "" if iris and weights else "Metal binary or selected model weights are missing",
    )

    sdcpp = command_path(args.sdcpp_bin)
    sd_assets = (
        existing_path(args.sdcpp_diffusion),
        existing_path(args.sdcpp_vae),
        existing_path(args.sdcpp_llm),
    )
    items["stable-diffusion-cpp"] = Inventory(
        "stable-diffusion-cpp",
        sdcpp is not None,
        sdcpp is not None and all(sd_assets),
        artifact_version(sdcpp, probe([sdcpp, "--version"])) if sdcpp else "",
        "" if sdcpp and all(sd_assets) else "CLI or one of diffusion/VAE/LLM assets is missing",
    )

    mflux_python = command_path(args.mflux_python)
    mflux_ver = python_version(mflux_python, "mflux") if mflux_python else ""
    items["mflux"] = Inventory(
        "mflux",
        bool(mflux_python and mflux_ver),
        bool(mflux_python and mflux_ver),
        mflux_ver,
        "" if mflux_python and mflux_ver else "Python environment does not contain mflux",
    )

    diff_python = command_path(args.diffusers_python)
    diff_ver = python_version(diff_python, "diffusers") if diff_python else ""
    items["diffusers"] = Inventory(
        "diffusers",
        bool(diff_python and diff_ver),
        bool(diff_python and diff_ver and weights),
        diff_ver,
        "" if diff_python and diff_ver and weights else "diffusers or selected model weights are missing",
    )

    ollama = command_path(args.ollama_bin)
    ollama_list = probe_output([ollama, "list"]) if ollama else ""
    ollama_model = args.ollama_model or (
        "x/z-image-turbo" if args.model == "z-image-turbo" else "x/flux2-klein"
    )
    has_ollama_model = bool(ollama_list and ollama_model in ollama_list)
    items["ollama"] = Inventory(
        "ollama",
        ollama is not None,
        bool(ollama and has_ollama_model),
        probe([ollama, "--version"]) if ollama else "",
        "" if ollama and has_ollama_model else "daemon is unavailable or selected image model is not local",
    )
    return items


def zdraw_command(args: argparse.Namespace, case: Case, output: Path, repeat: int) -> list[str]:
    weights = model_weights(args)
    assert weights is not None
    binary = command_path(args.zdraw_bin)
    assert binary is not None
    # --profile is a Z-Image execution tier; Klein has no strict tier and the
    # CLI rejects the flag at parse for it (ProfileUnsupported, 2026-08-10).
    profile = ["--profile", case.tier] if case.model == "z-image-turbo" else []
    return [
        binary,
        "generate",
        "--model",
        case.model,
        "--weights",
        str(weights),
        *profile,
        "--prompt",
        case.prompt,
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(effective_steps("zdraw", args, case)),
        "--seed",
        str(case.seed),
        "--repeat",
        str(repeat),
        "--out",
        str(output),
    ]


def run_zdraw(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    results = []
    env = os.environ.copy()
    # Memory comes from /usr/bin/time -l like every other arm; ZDRAW_MEMTRACE
    # would add its exact page walk to zdraw's wall time.
    env.update({"ZDRAW_PROGRESS": "quiet"})
    if args.zdraw_metrics:
        env["ZDRAW_METRICS"] = "1"
    vae_strip = getattr(args, "zdraw_vae_strip", None)
    if vae_strip is not None:
        env["ZDRAW_VAE_STRIP"] = str(vae_strip)
    zpack = existing_path(args.zdraw_zpack)
    pack_notes = []
    if zpack is not None:
        # ZDRAW_ZPACK reaches only the Z-Image loader (runtime_load.zig); Klein
        # reads ZDRAW_KLEIN_ZPACK (zflux2_run.zig), so setting the wrong one
        # leaves the loader to find a pack by discovery and the run is not
        # reproducible from its own command line.
        key = "ZDRAW_ZPACK" if case.model == "z-image-turbo" else "ZDRAW_KLEIN_ZPACK"
        env[key] = str(zpack)
        env["ZDRAW_REQUIRE_ZPACK"] = "1"
        pack_notes.append(f"{key}={zpack.name}")
    text_zpack = existing_path(args.zdraw_text_zpack)
    if text_zpack is not None:
        env["ZDRAW_KLEIN_TEXT_ZPACK"] = str(text_zpack)
        pack_notes.append(f"ZDRAW_KLEIN_TEXT_ZPACK={text_zpack.name}")
    if env.get("ZDRAW_COPY_WEIGHTS") == "1":
        pack_notes.append("ZDRAW_COPY_WEIGHTS=1 (weights copied, like-for-like RSS)")
    if args.protocol in ("warm", "both"):
        output = out_dir / "zdraw-warm.png"
        command = zdraw_command(args, case, output, case.warmups + case.runs)
        proc = timed_process(command, input_text=None, env=env, timeout=args.timeout)
        values = [float(value) for value in RUN_RE.findall(proc.stdout + proc.stderr)]
        values = values[case.warmups : case.warmups + case.runs]
        result = Result("zdraw", case.model, case.tier, "warm-session", "ok", item.version)
        result.notes.extend(pack_notes)
        if vae_strip is not None:
            result.notes.append(f"VAE strip rows: {vae_strip}")
        if not values and case.warmups == 0 and case.runs == 1:
            values = [proc.wall_seconds]
            result.notes.append("single-run timing includes process startup")
        result.seconds = values
        results.append(
            finish_result(result, [proc], command, output, out_dir / "zdraw-warm.log")
        )

    if args.protocol in ("cold", "both"):
        processes = []
        seconds = []
        output = out_dir / "zdraw-cold.png"
        command = zdraw_command(args, case, output, 1)
        for _ in range(case.runs):
            proc = timed_process(command, input_text=None, env=env, timeout=args.timeout)
            processes.append(proc)
            seconds.append(proc.wall_seconds)
        result = Result("zdraw", case.model, case.tier, "cold-process", "ok", item.version)
        result.notes.extend(pack_notes)
        if vae_strip is not None:
            result.notes.append(f"VAE strip rows: {vae_strip}")
        result.seconds = seconds
        results.append(
            finish_result(result, processes, command, output, out_dir / "zdraw-cold.log")
        )
    return results


def run_mflux(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    python = command_path(args.mflux_python)
    assert python is not None
    driver = Path(__file__).with_name("mflux_bench.py")
    base = [
        python,
        str(driver),
        "--model",
        case.model,
        "--prompt",
        case.prompt,
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(effective_steps("mflux", args, case)),
        "--seed",
        str(case.seed),
    ]
    if args.mflux_model_path:
        base.extend(("--model-path", str(Path(args.mflux_model_path).expanduser())))
    if case.tier == "product" and not args.mflux_prequantized and args.mflux_quantize:
        base.extend(("--quantize", str(args.mflux_quantize)))
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"

    results = []
    if args.protocol in ("warm", "both"):
        output = out_dir / "mflux-warm.png"
        command = [
            *base,
            "--warmups",
            str(case.warmups),
            "--runs",
            str(case.runs),
            "--output",
            str(output),
        ]
        proc = timed_process(command, input_text=None, env=env, timeout=args.timeout)
        values = []
        for line in (proc.stdout + proc.stderr).splitlines():
            if not line.startswith("ZBENCH "):
                continue
            record = json.loads(line.removeprefix("ZBENCH "))
            if record["measured"]:
                values.append(float(record["seconds"]))
        result = Result("mflux", case.model, case.tier, "warm-session", "ok", item.version)
        result.seconds = values
        if args.mflux_prequantized:
            result.notes.append(f"pre-quantized model: {args.mflux_model_path}")
        results.append(
            finish_result(result, [proc], command, output, out_dir / "mflux-warm.log")
        )

    if args.protocol in ("cold", "both"):
        processes = []
        seconds = []
        output = out_dir / "mflux-cold.png"
        command = [*base, "--warmups", "0", "--runs", "1", "--output", str(output)]
        for _ in range(case.runs):
            proc = timed_process(command, input_text=None, env=env, timeout=args.timeout)
            processes.append(proc)
            seconds.append(proc.wall_seconds)
        result = Result("mflux", case.model, case.tier, "cold-process", "ok", item.version)
        result.seconds = seconds
        if args.mflux_prequantized:
            result.notes.append(f"pre-quantized model: {args.mflux_model_path}")
        results.append(
            finish_result(result, processes, command, output, out_dir / "mflux-cold.log")
        )
    return results


def iris_command(
    args: argparse.Namespace,
    case: Case,
    output: Path,
) -> list[str]:
    binary = command_path(args.iris_bin)
    weights = Path(args.iris_weights) if getattr(args, "iris_weights", None) else model_weights(args)
    assert binary is not None and weights is not None
    return [
        binary,
        "-d",
        str(weights),
        "-p",
        case.prompt,
        "-o",
        str(output),
        "-W",
        str(case.width),
        "-H",
        str(case.height),
        "-s",
        str(effective_steps("iris", args, case)),
        "-S",
        str(case.seed),
        "--mmap",
    ]


def run_iris(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    binary = command_path(args.iris_bin)
    weights = Path(args.iris_weights) if getattr(args, "iris_weights", None) else model_weights(args)
    assert binary is not None and weights is not None
    results = []
    if args.protocol in ("warm", "both"):
        output = out_dir / "iris-warm.png"
        command = [binary, "-d", str(weights), "--mmap"]
        prompts = [case.prompt] * (case.warmups + case.runs)
        input_text = "\n".join(
            [
                f"!size {case.width}x{case.height}",
                f"!steps {effective_steps('iris', args, case)}",
                f"!seed {case.seed}",
                *prompts,
                f"!save {output}",
                "!quit",
                "",
            ]
        )
        env = os.environ.copy()
        for name in (
            "GHOSTTY_RESOURCES_DIR",
            "ITERM_SESSION_ID",
            "KITTY_WINDOW_ID",
            "KONSOLE_VERSION",
            "TERM_PROGRAM",
            "WEZTERM_PANE",
        ):
            env.pop(name, None)
        proc = timed_process(command, input_text=input_text, env=env, timeout=args.timeout)
        values = [float(value) for value in IRIS_RE.findall(proc.stdout + proc.stderr)]
        values = values[case.warmups : case.warmups + case.runs]
        result = Result("iris", case.model, case.tier, "warm-session", "ok", item.version)
        result.seconds = values
        result.notes.append("repeat prompts use iris's in-session embedding cache")
        results.append(
            finish_result(result, [proc], command, output, out_dir / "iris-warm.log")
        )

    if args.protocol in ("cold", "both"):
        processes = []
        seconds = []
        output = out_dir / "iris-cold.png"
        command = iris_command(args, case, output)
        for _ in range(case.runs):
            proc = timed_process(command, input_text=None, env=None, timeout=args.timeout)
            processes.append(proc)
            seconds.append(proc.wall_seconds)
        result = Result("iris", case.model, case.tier, "cold-process", "ok", item.version)
        result.seconds = seconds
        results.append(
            finish_result(result, processes, command, output, out_dir / "iris-cold.log")
        )
    return results


def run_drawthings(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    if args.protocol == "warm":
        return [
            Result(
                "drawthings",
                case.model,
                case.tier,
                "warm-session",
                "unsupported",
                item.version,
                "the public local CLI exposes one generation per process",
            )
        ]
    binary = command_path(args.drawthings_bin)
    models_dir = existing_path(args.drawthings_models_dir)
    assert binary is not None and models_dir is not None
    processes = []
    seconds = []
    output = out_dir / "drawthings-cold.png"
    command = [
        binary,
        "generate",
        "--models-dir",
        str(models_dir),
        "--model",
        drawthings_model(args),
        "--prompt",
        case.prompt,
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(effective_steps("drawthings", args, case)),
        "--seed",
        str(case.seed),
        "--offline",
        "--no-download-missing",
        "--disable-preview",
        "--output",
        str(output),
    ]
    for _ in range(case.runs):
        proc = timed_process(command, input_text=None, env=None, timeout=args.timeout)
        processes.append(proc)
        seconds.append(proc.wall_seconds)
    result = Result("drawthings", case.model, case.tier, "cold-process", "ok", item.version)
    result.seconds = seconds
    result.notes.append("wall timing includes model loading; public CLI has no repeat mode")
    return [
        finish_result(
            result,
            processes,
            command,
            output,
            out_dir / "drawthings-cold.log",
        )
    ]


def drawthings_app_payload(
    args: argparse.Namespace,
    case: Case,
) -> dict[str, object]:
    return {
        "model": drawthings_model(args),
        "prompt": case.prompt,
        "negative_prompt": "",
        "width": case.width,
        "height": case.height,
        "steps": effective_steps("drawthings-app", args, case),
        "seed": case.seed,
        "guidance_scale": 0.0 if case.model == "z-image-turbo" else 1.0,
        "batch_size": 1,
    }


def decode_drawthings_app_response(raw: bytes) -> bytes:
    value = json.loads(raw)
    images = value.get("images") if isinstance(value, dict) else None
    if not isinstance(images, list) or not images or not isinstance(images[0], str):
        raise ValueError("Draw Things response has no encoded image")
    return base64.b64decode(images[0], validate=True)


def drawthings_app_request(
    url: str,
    payload: dict[str, object],
    timeout: int,
) -> tuple[ProcessResult, bytes | None]:
    body = json.dumps(payload, sort_keys=True).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/sdapi/v1/txt2img",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        image_bytes = decode_drawthings_app_response(raw)
        status = f"HTTP 200; response_bytes={len(raw)}; image_bytes={len(image_bytes)}"
        return (
            ProcessResult(0, time.perf_counter() - started, status, "", None, None),
            image_bytes,
        )
    except (OSError, ValueError, urllib.error.URLError) as error:
        return (
            ProcessResult(
                1,
                time.perf_counter() - started,
                "",
                f"{type(error).__name__}: {error}",
                None,
                None,
            ),
            None,
        )


def run_drawthings_app(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    results = []
    if args.protocol in ("cold", "both"):
        results.append(
            Result(
                "drawthings-app",
                case.model,
                case.tier,
                "cold-process",
                "unsupported",
                item.version,
                "the release app API does not expose a reproducible model-unload operation",
            )
        )
    if args.protocol not in ("warm", "both"):
        return results

    payload = drawthings_app_payload(args, case)
    endpoint = args.drawthings_app_url.rstrip("/") + "/sdapi/v1/txt2img"
    command = ["POST", endpoint, json.dumps(payload, sort_keys=True)]
    output = out_dir / "drawthings-app-warm.png"
    processes = []
    seconds = []
    footprints = []
    total_requests = case.warmups + case.runs
    for index in range(total_requests):
        proc, image_bytes = drawthings_app_request(
            args.drawthings_app_url,
            payload,
            args.timeout,
        )
        processes.append(proc)
        if image_bytes is not None:
            output.write_bytes(image_bytes)
        footprints.append(named_peak_footprint(args.drawthings_process_name))
        if index >= case.warmups:
            seconds.append(proc.wall_seconds)

    result = Result(
        "drawthings-app",
        case.model,
        case.tier,
        "warm-session",
        "ok",
        item.version,
    )
    result.seconds = seconds
    result.memory_scope = (
        f"processes reported by footprint for {args.drawthings_process_name!r}"
    )
    result.notes.extend(
        (
            "requests use the shipping app's local HTTP API",
            "warmups precede measured requests; the app remains resident",
            "the adapter overrides model, prompt, dimensions, steps, seed, guidance, and batch size",
        )
    )
    result = finish_result(
        result,
        processes,
        command,
        output,
        out_dir / "drawthings-app-warm.log",
    )
    result.peak_footprint_bytes = max(
        (value for value in footprints if value is not None),
        default=None,
    )
    return [*results, result]


def run_sdcpp(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    if args.protocol == "warm":
        return [
            Result(
                "stable-diffusion-cpp",
                case.model,
                case.tier,
                "warm-session",
                "unsupported",
                item.version,
                "the CLI comparison currently uses one generation per process",
            )
        ]
    binary = command_path(args.sdcpp_bin)
    assert binary is not None
    output = out_dir / "sdcpp-cold.png"
    command = [
        binary,
        "--diffusion-model",
        str(existing_path(args.sdcpp_diffusion)),
        "--vae",
        str(existing_path(args.sdcpp_vae)),
        "--llm",
        str(existing_path(args.sdcpp_llm)),
        "--prompt",
        case.prompt,
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(effective_steps("stable-diffusion-cpp", args, case)),
        "--seed",
        str(case.seed),
        "--cfg-scale",
        "1.0",
        "--diffusion-fa",
        "--offload-to-cpu",
        "--output",
        str(output),
        "--verbose",
    ]
    processes = []
    seconds = []
    for _ in range(case.runs):
        proc = timed_process(command, input_text=None, env=None, timeout=args.timeout)
        processes.append(proc)
        seconds.append(proc.wall_seconds)
    result = Result(
        "stable-diffusion-cpp",
        case.model,
        case.tier,
        "cold-process",
        "ok",
        item.version,
    )
    result.seconds = seconds
    return [
        finish_result(result, processes, command, output, out_dir / "sdcpp-cold.log")
    ]


def run_diffusers(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    python = command_path(args.diffusers_python)
    weights = model_weights(args)
    assert python is not None and weights is not None
    driver = Path(__file__).with_name("diffusers_bench.py")
    base = [
        python,
        str(driver),
        "--model",
        case.model,
        "--weights",
        str(weights),
        "--prompt",
        case.prompt,
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(effective_steps("diffusers", args, case)),
        "--guidance",
        "0.0" if case.model == "z-image-turbo" else "1.0",
        "--seed",
        str(case.seed),
    ]
    results = []
    if args.protocol in ("warm", "both"):
        output = out_dir / "diffusers-warm.png"
        command = [
            *base,
            "--warmups",
            str(case.warmups),
            "--runs",
            str(case.runs),
            "--output",
            str(output),
        ]
        proc = timed_process(command, input_text=None, env=None, timeout=args.timeout)
        values = []
        for line in (proc.stdout + proc.stderr).splitlines():
            if not line.startswith("ZBENCH "):
                continue
            record = json.loads(line.removeprefix("ZBENCH "))
            if record["measured"]:
                values.append(float(record["seconds"]))
        result = Result(
            "diffusers",
            case.model,
            case.tier,
            "warm-session",
            "ok",
            item.version,
        )
        result.seconds = values
        result.notes.append("bf16 MPS reference; no product quantization applied")
        results.append(
            finish_result(
                result,
                [proc],
                command,
                output,
                out_dir / "diffusers-warm.log",
            )
        )

    if args.protocol in ("cold", "both"):
        output = out_dir / "diffusers-cold.png"
        command = [*base, "--warmups", "0", "--runs", "1", "--output", str(output)]
        processes = []
        seconds = []
        for _ in range(case.runs):
            proc = timed_process(command, input_text=None, env=None, timeout=args.timeout)
            processes.append(proc)
            seconds.append(proc.wall_seconds)
        result = Result(
            "diffusers",
            case.model,
            case.tier,
            "cold-process",
            "ok",
            item.version,
        )
        result.seconds = seconds
        result.notes.append("bf16 MPS reference; no product quantization applied")
        results.append(
            finish_result(
                result,
                processes,
                command,
                output,
                out_dir / "diffusers-cold.log",
            )
        )
    return results


def run_ollama(
    args: argparse.Namespace,
    case: Case,
    item: Inventory,
    out_dir: Path,
) -> list[Result]:
    binary = command_path(args.ollama_bin)
    assert binary is not None
    model = args.ollama_model or (
        "x/z-image-turbo:fp8"
        if case.model == "z-image-turbo"
        else "x/flux2-klein:4b"
    )
    base = [
        binary,
        "run",
        model,
        case.prompt,
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(effective_steps("ollama", args, case)),
        "--seed",
        str(case.seed),
        "--keepalive",
        "5m",
    ]

    def stop_model() -> None:
        subprocess.run(
            [binary, "stop", model],
            capture_output=True,
            text=True,
            check=False,
        )

    def generate() -> tuple[ProcessResult, Path | None]:
        before = set(out_dir.glob("*.png"))
        proc = timed_process(
            base,
            input_text=None,
            env=None,
            timeout=args.timeout,
            cwd=out_dir,
        )
        created = set(out_dir.glob("*.png")) - before
        output = max(created, key=lambda path: path.stat().st_mtime) if created else None
        return proc, output

    results = []
    if args.protocol in ("warm", "both"):
        for _ in range(case.warmups):
            generate()
        processes = []
        seconds = []
        output = None
        footprints = []
        for _ in range(case.runs):
            proc, generated = generate()
            processes.append(proc)
            seconds.append(proc.wall_seconds)
            output = generated or output
            footprints.append(named_peak_footprint("ollama"))
        result = Result("ollama", case.model, case.tier, "warm-session", "ok", item.version)
        result.seconds = seconds
        result.memory_scope = "Ollama daemon and resident runner processes"
        result.notes.append("latency is the CLI request against a warm local daemon")
        result = finish_result(
            result,
            processes,
            base,
            output or out_dir / "ollama-warm-missing.png",
            out_dir / "ollama-warm.log",
        )
        result.peak_footprint_bytes = max(
            (value for value in footprints if value is not None),
            default=None,
        )
        results.append(result)

    if args.protocol in ("cold", "both"):
        processes = []
        seconds = []
        output = None
        footprints = []
        for _ in range(case.runs):
            stop_model()
            proc, generated = generate()
            processes.append(proc)
            seconds.append(proc.wall_seconds)
            output = generated or output
            footprints.append(named_peak_footprint("ollama"))
        result = Result("ollama", case.model, case.tier, "cold-process", "ok", item.version)
        result.seconds = seconds
        result.memory_scope = "Ollama daemon and resident runner processes"
        result.notes.append("model was stopped before every measured request")
        result = finish_result(
            result,
            processes,
            base,
            output or out_dir / "ollama-cold-missing.png",
            out_dir / "ollama-cold.log",
        )
        result.peak_footprint_bytes = max(
            (value for value in footprints if value is not None),
            default=None,
        )
        results.append(result)
    stop_model()
    return results


def unsupported_runner(
    engine: str,
    case: Case,
    item: Inventory,
    reason: str,
) -> list[Result]:
    return [
        Result(
            engine,
            case.model,
            case.tier,
            "none",
            "unsupported",
            item.version,
            reason,
        )
    ]
