#!/usr/bin/env python3
"""Run repeatable in-process diffusers MPS generations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("z-image-turbo", "flux2-klein-4b", "flux2-klein-base-4b", "flux2-klein-9b"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace):
    if args.model == "z-image-turbo":
        from diffusers import ZImagePipeline

        pipeline_type = ZImagePipeline
    else:
        from diffusers import Flux2KleinPipeline

        pipeline_type = Flux2KleinPipeline
    return pipeline_type.from_pretrained(
        args.weights,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("mps")


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.runs < 1:
        raise SystemExit("--warmups must be non-negative and --runs must be positive")

    pipeline = load_pipeline(args)
    total = args.warmups + args.runs
    for index in range(total):
        generator = torch.Generator("cpu").manual_seed(args.seed)
        started = time.perf_counter()
        result = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=generator,
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
