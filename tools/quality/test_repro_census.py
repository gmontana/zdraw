#!/usr/bin/env python3
"""Portable census control-flow tests; all renders and content checks are stubs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("repro_census.sh")


class CensusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="zdraw-census-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scripts = self.root / "tools/quality"
        scripts.mkdir(parents=True)
        shutil.copy2(SCRIPT, scripts / SCRIPT.name)
        self.commands = self.root / "commands"
        self.commands.mkdir()
        (self.commands / "census-python").symlink_to(sys.executable)
        for command, output in (("git", "stub-commit"), ("date", "fixed-time")):
            self.executable(self.commands / command, f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
        self.executable(self.root / "zig-out/bin/zdraw", f"#!{sys.executable}\n" + '''
import os
from pathlib import Path
import sys

output = Path(sys.argv[sys.argv.index("--out") + 1])
mode = os.environ.get("STUB_RENDER", "ok")
if mode == "missing" and output.name == "r2.png":
    sys.exit(0)
output.write_bytes(output.name.encode() if mode == "different" else b"stub image")
print("Metal dispatches: stub")
if mode == "fail-all" or (mode == "fail-one" and output.name == "r2.png"):
    sys.exit(9)
''')
        (scripts / "content_check.py").write_text('''
import os
from pathlib import Path
import sys

assert all(Path(name).is_file() for name in sys.argv[1:])
print(f"stub content check: {len(sys.argv) - 1} images")
sys.exit(int(os.environ.get("STUB_CONTENT_RC", "0")))
''')
        self.env = dict(os.environ, PATH=f"{self.commands}{os.pathsep}{os.environ['PATH']}",
                        MFLUX_PY=sys.executable, STUB_RENDER="ok", STUB_CONTENT_RC="0")

    def executable(self, path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        path.chmod(0o755)

    def run_census(self, n="3", **env):
        return subprocess.run(
            ["bash", str(self.root / "tools/quality/repro_census.sh"),
             "stub weights", "stub pack", "flux2-klein-4b", "-", n],
            env=dict(self.env, **env), capture_output=True, text=True, timeout=15,
        )

    def artifacts(self):
        return sorted((self.root / "runs/repro-census").glob("*"))

    def test_complete_census_checks_every_image(self):
        result = self.run_census()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("n=3 completed=3 distinct=1", result.stdout)
        output, = self.artifacts()
        self.assertEqual(len((output / "sha256.txt").read_text().splitlines()), 3)
        self.assertIn("3 images", (output / "content.txt").read_text())

    def test_distinct_hashes_remain_a_reported_measurement(self):
        result = self.run_census(STUB_RENDER="different")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("n=3 completed=3 distinct=3", result.stdout)

    def test_failed_render_is_excluded_even_if_it_writes_an_image(self):
        result = self.run_census(STUB_RENDER="fail-one")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("n=3 completed=2 distinct=1", result.stdout)
        output, = self.artifacts()
        hashes = (output / "sha256.txt").read_text()
        self.assertEqual(len(hashes.splitlines()), 2)
        self.assertNotIn("r2.png", hashes)

    def test_success_without_an_image_fails(self):
        result = self.run_census(STUB_RENDER="missing")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("n=3 completed=2 distinct=1", result.stdout)

    def test_all_failed_renders_have_no_hashes(self):
        result = self.run_census(STUB_RENDER="fail-all")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("n=3 completed=0 distinct=0", result.stdout)
        output, = self.artifacts()
        self.assertEqual((output / "sha256.txt").read_text(), "")

    def test_content_failure_fails_the_census(self):
        result = self.run_census(STUB_CONTENT_RC="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        output, = self.artifacts()
        self.assertIn("3 images", (output / "content.txt").read_text())

    def test_python_command_is_resolved_on_path(self):
        result = self.run_census(MFLUX_PY="census-python")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output, = self.artifacts()
        self.assertIn("3 images", (output / "content.txt").read_text())

    def test_missing_checker_dependencies_fail_before_rendering(self):
        for name in ("missing", str(self.root / "missing-python")):
            with self.subTest(name=name):
                result = self.run_census(MFLUX_PY=name)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(self.artifacts())
        (self.root / "tools/quality/content_check.py").unlink()
        result = self.run_census()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.artifacts())

    def test_invalid_counts_fail_before_rendering(self):
        for n in ("0", "-1", "1.5", "invalid", "08"):
            with self.subTest(n=n):
                result = self.run_census(n=n)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(self.artifacts())

    def test_same_timestamp_runs_have_separate_artifacts(self):
        for _ in range(2):
            result = self.run_census()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.artifacts()), 2)

    def test_missing_binary_fails_before_rendering(self):
        (self.root / "zig-out/bin/zdraw").unlink()
        result = self.run_census()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.artifacts())

    def test_hashing_failure_cannot_report_success(self):
        self.executable(self.commands / "shasum", "#!/bin/sh\nexit 9\n")
        result = self.run_census()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("CENSUS n=", result.stdout)


if __name__ == "__main__":
    unittest.main()
