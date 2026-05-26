"""
Experimental dynamic watermark removal CLI.

Usage
-----
    python -m tool.dynamic_inpaint_cli \
        --video in.mp4 \
        --points "320,240+;400,250+;500,100-" \
        --output out.mp4

Click-point format
------------------
Semicolon-separated ``x,y[+/-]`` entries on the **first frame**:

* ``x,y+`` -- positive click; tells SAM "this pixel IS the watermark"
* ``x,y-`` -- negative click; tells SAM "this pixel is NOT the watermark"
* ``x,y``  -- treated as positive (``+`` is the default)

Add a few positive clicks across the watermark for a tight mask, and
sprinkle negative clicks just outside it if SAM bleeds into background.

Notes
-----
This is the **phase-A MVP**. It depends on a sibling
``watermark_remover`` project checkout for the SAM + DeAOT + ProPainter
implementation. Resolve order:

1. ``--wm-path`` (this flag)
2. ``VSR_WATERMARK_REMOVER_PATH`` env var
3. ``../watermark_remover`` next to this repo
4. ``D:/Repos/watermark_remover``

Phase B will (a) vendor the code in-tree and (b) wire this into the
desktop UI as a "Dynamic watermark" mode. Until then, expect the CLI
to print ProPainter's own tqdm progress directly to your terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow ``python tool/dynamic_inpaint_cli.py ...`` as well as ``-m tool...``
# by ensuring the repo root is on sys.path.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.dynamic import (  # noqa: E402  (sys.path tweak above)
    parse_clicks,
    resolve_watermark_remover_path,
    run_dynamic_removal,
)


# ----------------------------------------------------------------- #
# Simple inline progress reporter
# ----------------------------------------------------------------- #

_PHASE_LABEL = {
    "loading":      "Loading SAM + DeAOT",
    "sam":          "First-frame segmentation",
    "deaot":        "Tracking watermark (slow)",
    "mask_cleanup": "Cleaning mask sequence",
    "bbox":         "Computing crop bbox",
    "crop":         "Cropping video + masks",
    "propainter":   "Inpainting (ProPainter)",
    "overlay":      "Compositing result",
    "done":         "Done",
}


def _make_progress_reporter():
    """Return a progress_callback that prints concise stderr updates.

    Only prints once per (phase, transition) to avoid log spam: 'starting'
    at value==0.0, 'done' at value==1.0. Intermediate values would only
    appear if a phase emits them (none do currently).
    """
    state = {"last_phase": None, "last_value": None}

    def _cb(phase: str, value: float, extra: str, overall: float) -> None:
        label = _PHASE_LABEL.get(phase, phase)
        last = (state["last_phase"], state["last_value"])
        if last == (phase, value):
            return
        state["last_phase"], state["last_value"] = phase, value
        if value <= 0.001:
            tag = "START"
        elif value >= 0.999:
            tag = "DONE "
        else:
            tag = f"{int(value*100):3d}% "
        extra_str = f"  ({extra})" if extra else ""
        bar_len = 24
        filled = int(overall * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"  [{bar}] {int(overall*100):3d}%  {tag}  {label}{extra_str}",
            file=sys.stderr,
            flush=True,
        )

    return _cb


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dynamic_inpaint_cli",
        description=(
            "Experimental dynamic watermark removal using SAM + DeAOT + "
            "ProPainter. Tracks a clicked watermark through the whole video "
            "and inpaints the moving region."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m tool.dynamic_inpaint_cli \\\n"
            "      --video sample.mp4 \\\n"
            "      --points '320,240+;400,250+;500,100-' \\\n"
            "      --output clean.mp4\n"
        ),
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Input video path.")
    p.add_argument("--points", required=True,
                   help=("First-frame click prompts. Format: 'x,y[+/-];...'. "
                         "+ means watermark, - means background. Default sign is +."))
    p.add_argument("--output", required=True, type=Path,
                   help="Output MP4 path.")
    p.add_argument("--wm-path", default=None,
                   help=("Path to watermark_remover project root. "
                         "Overrides VSR_WATERMARK_REMOVER_PATH and auto-discovery."))
    p.add_argument("--aot-model", default="r50_deaotl",
                   choices=("deaotb", "deaotl", "r50_deaotl"),
                   help="DeAOT model variant (default: r50_deaotl).")
    p.add_argument("--subvideo-length", type=int, default=80,
                   help="ProPainter --subvideo_length (default 80). "
                        "Auto-crop keeps the per-batch tensor small "
                        "enough that 80 fits a 12 GB GPU; if you pass "
                        "--no-auto-crop drop this to 20 or even 10.")
    crop = p.add_mutually_exclusive_group()
    crop.add_argument("--auto-crop", dest="auto_crop", action="store_true",
                      default=True,
                      help="(Default) Crop video+masks to the tracked "
                           "watermark's bounding box before running "
                           "ProPainter, then overlay the inpainted "
                           "result back. 5-10x faster on consumer GPUs "
                           "for small watermarks.")
    crop.add_argument("--no-auto-crop", dest="auto_crop", action="store_false",
                      help="Run ProPainter on the full frame. Slow on "
                           "consumer GPUs at 1080p; mainly for debugging.")
    p.add_argument("--crop-padding", type=int, default=96,
                   help="Pixels of context around the watermark bbox "
                        "when --auto-crop is on (default 96). More "
                        "padding = more surrounding texture for "
                        "ProPainter, but larger crop and slower run.")
    p.add_argument("--mask-dilate", type=int, default=12,
                   help="Pixels to grow each mask outward before saving "
                        "(default 12). 0 disables. Increases to 4-8 px "
                        "eliminate the faint outline ring that sharp-edge "
                        "watermarks leave in the inpaint output.")
    fp = p.add_mutually_exclusive_group()
    fp.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                    help="Use FP16 in ProPainter (default, halves VRAM).")
    fp.add_argument("--no-fp16", dest="fp16", action="store_false",
                    help="Disable FP16 in ProPainter.")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Preserve workspace (masks, chunked mp4s, "
                        "pre-encoded source) after a successful run "
                        "instead of deleting it. Default is to free "
                        "the 2-4 GB those intermediates occupy. "
                        "Failed runs ALWAYS preserve the workspace "
                        "regardless of this flag, so dynamic_resume_"
                        "from_masks.py can pick up from the crash.")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   default=True,
                   help="Suppress the inline phase-progress reporter.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose (DEBUG) logging.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("dynamic_inpaint_cli")

    try:
        clicks = parse_clicks(args.points)
    except ValueError as e:
        log.error("Invalid --points: %s", e)
        return 2

    try:
        wm_path = resolve_watermark_remover_path(args.wm_path)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 3
    log.info("watermark_remover root: %s", wm_path)
    log.info("Clicks: %s", clicks)

    try:
        result = run_dynamic_removal(
            video=args.video,
            clicks=clicks,
            output=args.output,
            wm_path=wm_path,
            fp16=args.fp16,
            subvideo_length=args.subvideo_length,
            aot_model=args.aot_model,
            auto_crop=args.auto_crop,
            crop_padding=args.crop_padding,
            mask_dilate=args.mask_dilate,
            keep_intermediates=args.keep_intermediates,
            progress_callback=_make_progress_reporter() if args.progress else None,
        )
    except FileNotFoundError as e:
        log.error("File missing: %s", e)
        return 4
    except RuntimeError as e:
        log.error("Pipeline failed: %s", e)
        return 5

    log.info("Success. Output: %s", result.output_video)
    log.info("Intermediates remain under: %s/output/%s/",
             result.wm_path, args.video.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
