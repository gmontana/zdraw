#!/usr/bin/env python3
"""Check real Klein terminal previews and final-image parity. Requires W16 weights."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import struct
import subprocess
import termios
import time

ROOT = Path(__file__).resolve().parents[1]
LABEL = re.compile(rb"preview  ([1-4])/4")
TERMINAL_VARS = {
    "KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "WEZTERM_PANE",
    "KONSOLE_VERSION", "ITERM_SESSION_ID", "TERM_PROGRAM",
}


def run_pty(command, env):
    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 38, 112, 0, 0))
    before = termios.tcgetattr(slave)
    proc = None
    data = bytearray()
    try:
        proc = subprocess.Popen(command, env=env, stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        slave = None
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                data.extend(chunk)
                assert len(data) < 16 * 1024 * 1024, "terminal output limit exceeded"
            elif proc.poll() is not None:
                break
        after = termios.tcgetattr(master)
        os.close(master)
        master = None
        code = proc.wait(timeout=max(0, deadline - time.monotonic()))
        assert before == after, "terminal modes changed"
        return code, bytes(data)
    finally:
        if slave is not None:
            os.close(slave)
        if master is not None:
            os.close(master)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def check_case(args, reference, env, case):
    from PIL import Image

    name, extra, flags, size, piped, previews = case
    output = args.out_dir / (name + ".png")
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    command = [
        str(args.binary), "generate", "--model", "flux2-klein-4b",
        "--prompt", reference["prompt"], "--width", str(size), "--height", str(size),
        "--steps", "4", "--seed", "46", "--out", str(output), *flags,
    ]
    if args.weights:
        command += ["--weights", str(args.weights)]
    if piped:
        result = subprocess.run(command, env=env | extra, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        code, data = result.returncode, result.stdout
    else:
        code, data = run_pty(command, env | extra)
    (args.out_dir / (name + ".log")).write_bytes(data)
    assert code == 0, (name, code, data[-2000:])
    labels = LABEL.findall(data)
    assert labels == ([b"1", b"2", b"3", b"4"] if previews else []), (name, labels)
    if previews:
        assert data.rfind(b"preview  4/4") < data.rfind(b"wrote "), name
        assert b"\r  denoise " not in data[data.index(b"preview  1/4"):], name
    if name == "iterm":
        assert data.count(b"height=28;preserveAspectRatio=1") == 4
        assert 0 <= data.rfind(b"--- memory / round-trips ---") < data.rfind(b"\x1b]1337;File="), name
    if name == "kitty":
        assert data.count(b"a=T,f=100,q=2,C=1,i=2053407344") == 4
        assert data.count(b"a=d,d=I,i=2053407344") == 4
    with Image.open(output) as image:
        image = image.convert("RGB")
        assert image.size == (size, size), name
        digest = hashlib.sha256(image.tobytes()).hexdigest()
    if size == 1024:
        expected = next(row for row in reference["entries"]
                        if row["model"] == "flux2-klein-4b" and row["width"] == size)
        assert digest == expected["sha256"], (name, digest)
    return {"case": name, "previews": len(labels), "size": size, "pixel_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--zpack", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ZDRAW_") and k not in TERMINAL_VARS}
    env.update(TERM="xterm-256color", ZDRAW_PROGRESS="compact")
    if args.zpack:
        env["ZDRAW_KLEIN_ZPACK"] = str(args.zpack.resolve())
    reference = json.loads((ROOT / "certified/hashes.json").read_text())
    cases = [
        ("iterm", {"TERM_PROGRAM": "iTerm.app"}, ["--show"], 1024, False, True),
        ("kitty", {"KITTY_WINDOW_ID": "1"}, ["--show"], 512, False, True),
        ("ansi", {}, ["--show"], 512, False, True),
        ("final-only", {"TERM_PROGRAM": "iTerm.app"},
         ["--show", "--no-progressive"], 512, False, False),
        ("no-show", {"TERM_PROGRAM": "iTerm.app"}, [], 512, False, False),
        ("redirected", {"TERM_PROGRAM": "iTerm.app"}, ["--show"], 512, True, False),
    ]
    results = []
    for case in cases:
        result = check_case(args, reference, env, case)
        results.append(result)
        print(json.dumps(result), flush=True)
    assert len({r["pixel_sha256"] for r in results if r["size"] == 512}) == 1
    record = {"kind": "functional checks, not performance measurements",
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "results": results}
    (args.out_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Progressive CLI checks passed; final pixels agree and terminal modes are unchanged.")


if __name__ == "__main__":
    main()
