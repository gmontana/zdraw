#!/usr/bin/env python3
"""Reproducible local Apple-Silicon diffusion-engine comparison.

Cold-process and warm-session measurements stay separate, and unavailable
engines remain visible instead of silently shrinking the competitor set.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    from tools.competitive_core import (
        Case,
        Result,
        check_content,
        host_info,
        unavailable,
        verdict,
        write_reports,
    )
    from tools.competitive_engines import (
        ENGINE_NAMES,
        PRIMARY_COMPETITORS,
        command_path,
        discover,
        effective_steps,
        run_diffusers,
        run_drawthings,
        run_drawthings_app,
        run_iris,
        run_mflux,
        run_ollama,
        run_sdcpp,
        run_zdraw,
        unsupported_runner,
    )
except ModuleNotFoundError:
    from competitive_core import (
        Case,
        Result,
        check_content,
        host_info,
        unavailable,
        verdict,
        write_reports,
    )
    from competitive_engines import (
        ENGINE_NAMES,
        PRIMARY_COMPETITORS,
        command_path,
        discover,
        effective_steps,
        run_diffusers,
        run_drawthings,
        run_drawthings_app,
        run_iris,
        run_mflux,
        run_ollama,
        run_sdcpp,
        run_zdraw,
        unsupported_runner,
    )


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    default_dt_dir = (
        Path.home()
        / "Library"
        / "Containers"
        / "com.liuliu.draw-things"
        / "Data"
        / "Documents"
        / "Models"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("inventory", "run"), nargs="?", default="inventory")
    ap.add_argument("--model", choices=("z-image-turbo", "flux2-klein-4b"), default="z-image-turbo")
    ap.add_argument("--tier", choices=("strict", "product"), default="product")
    ap.add_argument("--settings", choices=("matched", "recommended"), default="matched")
    ap.add_argument("--prompt", default="a red fox in deep snow, golden hour light")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--protocol", choices=("cold", "warm", "both"), default="both")
    ap.add_argument("--engines", default=",".join(ENGINE_NAMES))
    ap.add_argument("--required", default=",".join(PRIMARY_COMPETITORS))
    ap.add_argument("--output-dir")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--zimage-weights", default=None, help="Z-Image-Turbo weights dir (required for the zdraw arm)")
    ap.add_argument("--klein-weights", default=None, help="FLUX.2 Klein snapshot dir (required for the zdraw arm)")
    ap.add_argument("--zdraw-bin", default=str(root / "zig-out" / "bin" / "zdraw"))
    ap.add_argument("--zdraw-zpack")
    ap.add_argument("--zdraw-text-zpack", help="Klein 4-bit text-encoder pack (ZDRAW_KLEIN_TEXT_ZPACK)")
    ap.add_argument("--zdraw-metrics", action="store_true")
    ap.add_argument("--zdraw-vae-strip", type=int)
    ap.add_argument("--drawthings-bin", default="draw-things-cli")
    ap.add_argument("--drawthings-app", default="/Applications/Draw Things.app")
    ap.add_argument("--drawthings-app-url", default="http://127.0.0.1:7860")
    ap.add_argument("--drawthings-process-name", default="DrawThings")
    ap.add_argument("--drawthings-model")
    ap.add_argument("--drawthings-models-dir", default=str(default_dt_dir))
    ap.add_argument("--iris-bin", default="iris")
    ap.add_argument("--iris-weights", help="iris.c model directory (its download_model.py output); defaults to the model weights")
    ap.add_argument("--sdcpp-bin", default="sd-cli")
    ap.add_argument("--sdcpp-diffusion")
    ap.add_argument("--sdcpp-vae")
    ap.add_argument("--sdcpp-llm")
    ap.add_argument("--mflux-python", default=sys.executable)
    ap.add_argument("--mflux-model-path")
    ap.add_argument("--mflux-prequantized", action="store_true")
    ap.add_argument("--mflux-quantize", type=int, choices=(4, 8), default=8)
    ap.add_argument("--diffusers-python", default=sys.executable)
    ap.add_argument("--ollama-bin", default="ollama")
    ap.add_argument("--ollama-model")
    return ap


def selected_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(ENGINE_NAMES))
    if unknown:
        raise SystemExit(f"unknown engines: {', '.join(unknown)}")
    return names


def main() -> int:
    args = parser().parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be positive and --warmups non-negative")
    if min(args.width, args.height, args.steps) < 1:
        raise SystemExit("width, height, and steps must be positive")
    if args.zdraw_vae_strip is not None and args.zdraw_vae_strip < 1:
        raise SystemExit("--zdraw-vae-strip must be positive")
    if args.mflux_prequantized and not args.mflux_model_path:
        raise SystemExit("--mflux-prequantized requires --mflux-model-path")

    inventory = discover(args)
    names = selected_names(args.engines)
    if args.action == "inventory":
        for name in names:
            item = inventory[name]
            state = "runnable" if item.runnable else "unavailable"
            detail = "; ".join(value for value in (item.version, item.reason) if value)
            print(f"{name:22} {state:11} {detail}")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("runs") / "competitive" / f"{args.model}-{args.tier}-{stamp}"
    )
    if out_dir.exists():
        raise SystemExit(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    case = Case(
        args.model,
        args.prompt,
        args.width,
        args.height,
        args.steps,
        args.seed,
        args.tier,
        args.settings,
        args.runs,
        args.warmups,
    )

    runners: dict[str, Callable[..., list[Result]]] = {
        "zdraw": run_zdraw,
        "drawthings": run_drawthings,
        "drawthings-app": run_drawthings_app,
        "iris": run_iris,
        "stable-diffusion-cpp": run_sdcpp,
        "mflux": run_mflux,
        "diffusers": run_diffusers,
        "ollama": run_ollama,
    }
    results: list[Result] = []
    for name in names:
        item = inventory[name]
        print(f"==> {name}: {'running' if item.runnable else item.reason}", flush=True)
        if not item.runnable:
            results.append(unavailable(name, case, item))
            continue
        runner = runners.get(name)
        if runner is None:
            results.extend(
                unsupported_runner(
                    name,
                    case,
                    item,
                    "inventory is implemented; a matched local generation adapter is not",
                )
            )
            continue
        results.extend(runner(args, case, item, out_dir))
        for result in results:
            if result.engine == name and result.effective_steps is None:
                result.effective_steps = effective_steps(name, args, case)
            if result.engine == name:
                check_content(
                    result,
                    Path(__file__).resolve().parents[1],
                    command_path(args.diffusers_python) or "python3",
                )

    required = set(selected_names(args.required))
    protocols = ("cold-process", "warm-session") if args.protocol == "both" else (
        f"{args.protocol}-process" if args.protocol == "cold" else "warm-session",
    )
    verdicts = [verdict(results, required, protocol) for protocol in protocols]
    write_reports(out_dir, case, inventory, results, verdicts, host_info())
    print(f"wrote {out_dir / 'summary.md'}")
    return (
        0
        if all(result.status not in ("failed", "invalid-output") for result in results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
