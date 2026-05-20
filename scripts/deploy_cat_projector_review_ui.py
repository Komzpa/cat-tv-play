#!/usr/bin/env python3
"""Stage the Cat TV Play review/calibration UI for Home Assistant /local."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web"
LABEL_REVIEW_NAMESPACE = "cat_projector_label_review"

FILES = [
    (
        WEB_ROOT / "calibration-tools" / "projector-wall-calibrator.html",
        Path("cat-tv-learning/calibration-tools/projector-wall-calibrator.html"),
    ),
]


def deploy(www_root: Path) -> list[Path]:
    written: list[Path] = []
    for source, relative_target in FILES:
        if not source.exists():
            raise FileNotFoundError(source)
        target = www_root / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--www-root",
        type=Path,
        default=Path("/config/www"),
        help="Home Assistant www root exposed as /local.",
    )
    args = parser.parse_args()

    for path in deploy(args.www_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
