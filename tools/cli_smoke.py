#!/usr/bin/env python3
"""Exercise CLI errors and the Stanza session without downloading or running a model."""

import argparse
import errno
import fcntl
import os
from pathlib import Path
import re
import select
import struct
import subprocess
import tempfile
import termios
import time


ANSI = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def png(path: Path) -> None:
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    assert struct.unpack(">II", data[16:24]) == (64, 64), path


class Session:
    def __init__(self, binary: Path, cwd: Path, env: dict, output_dir: str | None = None):
        self.master, slave = os.openpty()
        self.before = termios.tcgetattr(slave)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))

        def terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        save_args = ["--out-dir", output_dir] if output_dir else ["--no-auto-save"]
        self.proc = subprocess.Popen(
            [str(binary), "session", "--preview", "--no-show", *save_args,
             "--width", "64", "--height", "64"],
            cwd=cwd, env=env, stdin=slave, stdout=slave, stderr=slave,
            preexec_fn=terminal,
        )
        os.close(slave)
        self.output = bytearray()

    def expect(self, expected: bytes, start: int = 0, after: int = 0):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            found = ANSI.sub(b"", self.output[start:]).find(expected, after)
            if found >= 0:
                return found + len(expected)
            readable, _, _ = select.select([self.master], [], [], 0.1)
            if readable:
                try:
                    data = os.read(self.master, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not data:
                    break
                self.output.extend(data)
            elif self.proc.poll() is not None:
                break
        raise AssertionError(f"missing {expected!r}\n{bytes(self.output)!r}")

    def send(self, keys: bytes, expected: bytes):
        start = len(self.output)
        os.write(self.master, keys)
        after = self.expect(expected, start)
        self.expect(b"zdraw ) ", start, after)

    def wait(self):
        # Drain output and release the master at EOF so macOS can reap the child.
        deadline = time.monotonic() + 15
        while self.proc.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.proc.args, 15)
            readable, _, _ = select.select([self.master], [], [], 0.1)
            if readable:
                try:
                    data = os.read(self.master, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not data:
                    break
                self.output.extend(data)
        self.after = termios.tcgetattr(self.master)
        os.close(self.master)
        self.master = None
        return self.proc.wait(timeout=max(0, deadline - time.monotonic()))

    def close(self):
        # Release the PTY before waiting, including after a failed expectation.
        if self.master is not None:
            os.close(self.master)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def check(binary: Path) -> None:
    binary = binary.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="zdraw-cli-smoke-") as raw:
        cwd = Path(raw)
        env = {k: v for k, v in os.environ.items() if not k.startswith("ZDRAW_")}
        env.update(HOME=raw, TERM="xterm-256color", NO_COLOR="1")
        for command in ("generate", "session", "fetch", "bench", "doctor",
                        "preview", "inspect", "version"):
            result = subprocess.run([str(binary), command, "--help"], cwd=cwd,
                                    env=env, capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, result.stderr
            assert f"Usage: zdraw {command}" in result.stdout, result.stdout
        for args, expected in (
            (["generate", "--prompt", "cat"], "provide --out"),
            (["generate", "--prompt", "cat", "--out", "x.png", "--width", "17"],
             "multiples of 16"),
            (["generate", "--model", "flux2-klein-4b", "--prompt", "cat", "--out", "x.png",
              "--width", "544", "--height", "800"], "area divisible by 8192"),
            (["session", "--model", "flux2-klein-4b", "--no-auto-save",
              "--width", "1536", "--height", "1536"], "up to 1,048,576"),
            (["generate", "--prompt", "cat", "--out", "x.png"], "zdraw fetch"),
            (["serve"], "unknown command"),
        ):
            result = subprocess.run([str(binary), *args], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=15)
            assert result.returncode == 1, result
            assert expected in result.stderr, result.stderr
        session = Session(binary, cwd, env)
        try:
            session.expect(b"zdraw ) ")
            session.send(b"he\t\r", b"Tab completes commands")
            session.send(b"/missing\r", b"unknown command")
            session.send(b"steps 0\r", b"steps must be > 0")
            session.send(b"save empty.png\r", b"nothing to save")
            session.send(b"seed 18446744073709551615\r", b"seed 18446744073709551615")
            session.send(b"new a red boat\r", b"result  ")
            session.send(b"save saved image.png\r", b"saved saved image.png")
            png(cwd / "saved image.png")
            session.send(b"clear\r", b"prompt cleared")
            session.send(b"prompt\r", b"prompt empty")
            session.send(b"\x1b[A\r", b"prompt empty")
            session.send(b"discard this\x03", b"\r\n")
            os.write(session.master, b"\x04")
            assert session.wait() == 0
            assert session.after == session.before, "terminal was not restored"
            assert (cwd / ".zdraw_history").is_file(), "history was not saved"
            assert not (cwd / "empty.png").exists()
        finally:
            session.close()
        images = cwd / "images"
        images.mkdir()
        original = images / "1.png"
        original.write_bytes(b"existing image: do not replace")
        for number in (2, 3):
            session = Session(binary, cwd, env, output_dir="images")
            try:
                session.expect(b"zdraw ) ")
                session.send(b"new a red boat\r", f"images/{number}.png".encode())
                png(images / f"{number}.png")
                assert original.read_bytes() == b"existing image: do not replace"
                os.write(session.master, b"quit\r")
                assert session.wait() == 0
                assert session.after == session.before
            finally:
                session.close()
    print("CLI smoke passed: help, errors, Stanza completion/history, preview save, Ctrl-C/D, terminal restore.")
    print("No diffusion model was run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    check(parser.parse_args().binary)
