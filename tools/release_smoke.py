#!/usr/bin/env python3
"""Check a locally built release archive away from its checkout (no model weights)."""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from cli_smoke import check as check_cli


def check(archive: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="zdraw-release-smoke-") as raw:
        scratch = Path(raw).resolve()
        subprocess.run(["tar", "-xzf", str(archive.resolve()), "-C", str(scratch)], check=True)
        roots = list(scratch.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError("release archive must contain one top-level directory")
        root = roots[0]
        for name in (
            "zdraw", "lib/steel.metallib", "README.md", "LICENSE", "NOTICE",
            "vendor/mfa/LICENSE-MIT-philipturner", "vendor/steel/LICENSE",
            "licenses/stanza/LICENSE",
        ):
            if not (root / name).is_file():
                raise RuntimeError(f"release archive is missing {name}")

        # Ignore developer overrides and run outside both the checkout and package.
        env = {k: v for k, v in os.environ.items() if not k.startswith("ZDRAW_")}
        binary = str(root / "zdraw")
        subprocess.run([binary, "version"], cwd=scratch, env=env, check=True)
        result = subprocess.run(
            [binary, "doctor", "--json"], cwd=scratch, env=env,
            check=True, capture_output=True, text=True,
        )
        report = json.loads(result.stdout)
        if report["schema_version"] != 1 or report["device"] is None:
            raise RuntimeError("doctor did not report a supported schema and Metal device")
        routes = report["routes"]
        if not routes["steel"]["ok"]:
            raise RuntimeError(f"bundled Metal library is unusable: {routes['steel']}")
        if Path(routes["steel_path"]).resolve() != root / "lib/steel.metallib":
            raise RuntimeError("doctor resolved a Metal library outside the release archive")

        output = scratch / "preview.png"
        subprocess.run(
            [binary, "preview", "--prompt", "a red boat", "--width", "64",
             "--height", "64", "--out", str(output)],
            cwd=scratch, env=env, check=True,
        )
        header = output.read_bytes()[:24]
        if (header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR"
                or len(header) != 24 or struct.unpack(">II", header[16:24]) != (64, 64)):
            raise RuntimeError("preview did not write a 64x64 PNG header")
        check_cli(Path(binary))
        print("Release smoke passed: relocated CLI, bundled Metal library, preview PNG header.")
        print("No diffusion model was run; full-image quality gates remain separate.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="locally built .tar.gz release archive")
    check(parser.parse_args().archive)


if __name__ == "__main__":
    main()
