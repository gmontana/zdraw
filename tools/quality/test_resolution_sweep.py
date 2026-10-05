#!/usr/bin/env python3
"""Portable campaign checks; needs the same NumPy/Pillow environment as the sweep."""

from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import resolution_sweep as sweep


class SweepTests(unittest.TestCase):
    def test_timing_covers_each_case_three_times_and_keeps_warmups_separate(self):
        work = sweep.schedule("timing")
        measured = [row for row in work if row[-1] != "warmup"]
        expected = {(model, w, h) for model in sweep.MODELS for w, h in sweep.SIZES}
        counts = Counter(row[:3] for row in measured)
        self.assertEqual(set(counts), expected)
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(len(work) - len(measured), 2)
        rounds = [[r[:3] for r in measured if r[-1] == f"round{n}"] for n in (1, 2, 3)]
        self.assertEqual(rounds[1], list(reversed(rounds[0])))
        self.assertNotEqual(rounds[0], rounds[2])

    def test_quality_uses_both_additional_prompts_at_every_size(self):
        work = sweep.schedule("quality")
        self.assertEqual(len(work), len(sweep.MODELS)*len(sweep.SIZES)*2)
        self.assertEqual(len(set(work)), len(work))
        self.assertEqual({r[3] for r in work}, {"portrait", "market"})

    def test_rejects_flat_output_and_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"flat.png"
            Image.new("RGB", (64, 64), (120, 120, 120)).save(path)
            record = sweep.image_record(path, 128, 128)
            self.assertTrue(any("blank-like" in p for p in record["problems"]))
            self.assertTrue(any("dimensions" in p for p in record["problems"]))

    def test_rejects_checkerboard_corruption_with_high_variance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"checker.png"
            x = np.indices((128, 128))[1]
            Image.fromarray(((x//8 % 2)*255).astype(np.uint8)).convert("RGB").save(path)
            record = sweep.image_record(path, 128, 128)
            self.assertGreater(record["luma_std"], 4)
            self.assertTrue(any("checkerboard" in p for p in record["problems"]))

    def test_timeout_stops_descendant_and_preserves_failure(self):
        # Remove only the macOS-specific time wrapper, retaining real process-group behaviour.
        original_popen = subprocess.Popen
        def launch(command, **kwargs):
            return original_popen(command[2:], **kwargs)

        script = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(p.pid,flush=True); time.sleep(60)"
        )
        with patch.object(sweep.subprocess, "Popen", side_effect=launch):
            result = sweep.measured([sys.executable, "-c", script], os.environ.copy(), 0.5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CAMPAIGN_TIMEOUT_OR_INTERRUPTION", result.stderr)
        pid = result.stdout.strip()
        self.assertTrue(pid.isdigit(), result)
        for _ in range(100):
            state = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True)
            if state.returncode != 0 or state.stdout.strip().startswith("Z"):
                break
            time.sleep(0.01)
        else:
            self.fail(f"timed-out descendant {pid} is still running")


if __name__ == "__main__":
    unittest.main()
