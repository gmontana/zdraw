#!/usr/bin/env python3
"""Content sanity check for generated images. Exit 0 sane, 1 corrupt.

Catches the failure classes that per-pixel stats miss: the 2026-07-18
checkerboard corruption had high variance and full span, so blank-detection
passed it. Checks, per image:
  - blank/flat (span, stddev): the existing class
  - checkerboard/noise energy: mean |8px-shift difference| over the lower
    half. Calibration (2026-07-19, 74-image campaign): typical renders 4-15,
    legitimately busy content (neon rain, dense texture, signage) 31-35
    ACROSS ALL ENGINES, real corruption 48-53. Threshold 40 splits the
    distributions with headroom on both sides.

Usage: content_check.py IMG [IMG...]   (prints one line per image)
Needs numpy+PIL (dev-env tool, same class as quality_cert).
"""

import sys

import numpy as np
from PIL import Image


def check(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = img.mean(axis=2)
    span = float(img.max() - img.min())
    std = float(gray.std())
    lower = gray[gray.shape[0] // 2 :]
    checker = float(np.abs(lower[:, 8:] - lower[:, :-8]).mean())
    problems = []
    if span < 16 or std < 4:
        problems.append(f"blank-like span={span:.0f} std={std:.1f}")
    if checker > 40.0:
        problems.append(f"checkerboard-noise score={checker:.1f} (sane<40)")
    return checker, std, problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for path in sys.argv[1:]:
        try:
            checker, std, problems = check(path)
        except Exception as err:  # noqa: BLE001 - report and fail per image
            print(f"FAIL {path}: unreadable ({err})")
            bad += 1
            continue
        if problems:
            print(f"FAIL {path}: {'; '.join(problems)}")
            bad += 1
        else:
            print(f"ok   {path} (checker={checker:.1f} std={std:.1f})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
