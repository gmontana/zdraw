#!/usr/bin/env python3
"""Run repeatable in-process mflux generations for competitive_bench.py."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("z-image-turbo", "flux2-klein-4b", "flux2-klein-9b"), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quantize", type=int, choices=(4, 8))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    kwargs = {"quantize": args.quantize, "model_path": args.model_path}
    if args.model == "z-image-turbo":
        from mflux.models.z_image.variants.z_image import ZImage

        return ZImage(**kwargs)

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    preset = getattr(ModelConfig, args.model.replace("-", "_"))
    return Flux2Klein(model_config=preset(), **kwargs)


def save_image(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = getattr(result, "image", result)
    image.save(path)


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.runs < 1:
        raise SystemExit("--warmups must be non-negative and --runs must be positive")

    model = load_model(args)
    total = args.warmups + args.runs
    for index in range(total):
        started = time.perf_counter()
        result = model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
        )
        elapsed = time.perf_counter() - started
        measured = index >= args.warmups
        print(
            "ZBENCH "
            + json.dumps(
                {
                    "run": index + 1,
                    "seconds": elapsed,
                    "measured": measured,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if index == total - 1:
            save_image(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
