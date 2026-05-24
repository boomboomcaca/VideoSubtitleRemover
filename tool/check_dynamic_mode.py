"""
Check whether VSR Pro's experimental Dynamic Watermark Mode is usable.

The mode shells out to a sibling ``watermark_remover`` project's
bundled conda env for the SAM + DeAOT + ProPainter pipeline (see
``backend/dynamic`` for the wiring). This script verifies that the
sibling checkout is present and complete; it does not modify anything.

Exit codes:
    0  Ready.
    1  Sibling checkout not found.
    2  Sibling checkout found but incomplete (missing files or env).
    3  Other unexpected problem -- traceback printed.

Usage::

    python -m tool.check_dynamic_mode [--wm-path PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


REQUIRED_ROOT_FILES = (
    "SegTracker.py",
    "seg_track_anything.py",
    "model_args.py",
    "aot_tracker.py",
)
REQUIRED_NESTED_FILES = (
    "ProPainter/inference_propainter.py",
    "ProPainter/model/propainter.py",
    "ProPainter/RAFT/raft.py",
    "env/python.exe",
    "env/Library/bin/ffmpeg.exe",
    "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
    "sam",
)


def _candidates(explicit: str | None) -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit).expanduser().resolve())
    env = os.environ.get("VSR_WATERMARK_REMOVER_PATH")
    if env:
        cands.append(Path(env).expanduser().resolve())
    cands.append((repo_root.parent / "watermark_remover").resolve())
    cands.append(Path("D:/Repos/watermark_remover").resolve())
    seen = set()
    uniq: list[Path] = []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def check(explicit: str | None = None, verbose: bool = False) -> int:
    cands = _candidates(explicit)
    print("Looking for watermark_remover sibling project. Order:")
    for c in cands:
        marker = "(exists)" if c.is_dir() else "(missing)"
        print(f"  - {c}  {marker}")
    print()

    chosen = next((c for c in cands if c.is_dir()), None)
    if chosen is None:
        print("[FAIL] No candidate path is a directory.")
        print(
            "\nDynamic Watermark Mode needs a sibling 'watermark_remover'\n"
            "project checkout. Either:\n"
            "  - clone https://github.com/lzhbrian/watermark-remover (or your\n"
            "    fork) next to this repo, OR\n"
            "  - set the VSR_WATERMARK_REMOVER_PATH env var to its location.\n"
        )
        return 1

    print(f"[OK]  Using: {chosen}\n")

    missing: list[str] = []
    for rel in REQUIRED_ROOT_FILES + REQUIRED_NESTED_FILES:
        p = chosen / rel
        ok = p.is_file() if not rel.endswith("/") and "." in p.name else p.exists()
        status = "OK " if ok else "MISS"
        print(f"  [{status}] {rel}")
        if not ok:
            missing.append(rel)

    if missing:
        print(f"\n[FAIL] {len(missing)} required item(s) missing from the sibling project:")
        for m in missing:
            print(f"   - {m}")
        if any("env/python.exe" in m for m in missing):
            print(
                "\nThe bundled 'env/' Python is missing. This usually means\n"
                "you cloned the source but didn't run the project's installer\n"
                "(typically a start.bat that downloads a portable conda env).\n"
                "Run watermark_remover's setup before retrying.\n"
            )
        if any("ckpt/R50_DeAOTL" in m for m in missing):
            print(
                "\nThe DeAOT R50 checkpoint is missing. Download from the\n"
                "watermark_remover release page and place it in ckpt/.\n"
            )
        return 2

    print("\n[READY] Dynamic Watermark Mode is fully configured.")
    print("        Launch VSR Pro and click 'Watermark Mode' in the header.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_dynamic_mode",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wm-path", default=None,
        help="Explicit path to watermark_remover checkout (overrides env var "
             "and auto-discovery).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        return check(args.wm_path, args.verbose)
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
