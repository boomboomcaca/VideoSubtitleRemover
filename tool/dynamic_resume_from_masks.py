"""
Resume the dynamic-watermark pipeline from an existing DeAOT mask dump.

When the DeAOT phase completes successfully but a downstream stage
fails (mask cleanup, bbox, crop, ProPainter, overlay), the masks on
disk are still good. This script picks up from there, saving the
5-10 hours DeAOT would otherwise need to redo on a long video.

Usage::

    python -m tool.dynamic_resume_from_masks \
        --video "<original input.mp4>" \
        --masks "<path to existing *_masks/ dir>" \
        --output "<final_clean.mp4>"

By default reuses the auto-crop / fp16 / chunking defaults from the
main worker. Pass --no-auto-crop / --subvideo-length / etc. to override.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Bootstrap the worker imports the same way the spawned worker does --
# cwd is the watermark_remover root so SegTracker's ckpt/... resolves,
# even though we don't actually call any SegTracker code here.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from backend.dynamic.external_pipeline import resolve_watermark_remover_path
import backend.dynamic._worker as worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("resume")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path,
                   help="Original input video.")
    p.add_argument("--masks", required=True, type=Path,
                   help="Existing DeAOT mask directory.")
    p.add_argument("--output", required=True, type=Path,
                   help="Final output MP4 path.")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Scratch dir for crop/chunks/etc. "
                        "Defaults to the masks dir's parent.")
    p.add_argument("--auto-crop", dest="auto_crop", action="store_true", default=True)
    p.add_argument("--no-auto-crop", dest="auto_crop", action="store_false")
    p.add_argument("--crop-padding", type=int, default=96)
    p.add_argument("--subvideo-length", type=int, default=80)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = p.parse_args(argv)

    video = args.video.resolve()
    mask_dir = args.masks.resolve()
    output = args.output.resolve()
    workspace = (args.workspace or mask_dir.parent).resolve()

    if not video.is_file():
        log.error("Input video not found: %s", video)
        return 2
    if not mask_dir.is_dir():
        log.error("Mask dir not found: %s", mask_dir)
        return 2

    # The worker module needs WM_PATH set + cwd in wm_path so it can
    # find ffmpeg/ffprobe. We use the same resolver as the production
    # spawn does.
    wm_path = resolve_watermark_remover_path()
    log.info("watermark_remover root: %s", wm_path)
    os.chdir(wm_path)
    worker.WM_PATH = wm_path  # override what was set at import time

    ffmpeg = worker._find_ffmpeg()
    ffprobe = worker._find_ffprobe()
    log.info("ffmpeg: %s", ffmpeg)
    log.info("ffprobe: %s", ffprobe)

    # -- Probe video for dimensions --
    with worker.FfmpegVideoReader(video, ffmpeg, ffprobe) as r:
        frame_w, frame_h = r.width, r.height
    log.info("Source video: %dx%d", frame_w, frame_h)

    # -- Stage 3: mask cleanup (largest-CC by centroid) --
    log.info("=== Stage 3: mask cleanup (in-place) ===")
    n_masks, cached_union, n_with_content = worker._clean_segtracker_masks(mask_dir)
    log.info("Cleaned %d mask PNGs", n_masks)

    # -- Stage 4: auto-crop decision --
    log.info("=== Stage 4: bbox compute ===")
    bbox = None
    auto_crop = args.auto_crop
    if auto_crop:
        # Pass the cached union from cleanup so we skip the rescan.
        bbox = worker._compute_bbox(
            mask_dir, frame_w, frame_h, padding=args.crop_padding,
            cached_union=cached_union,
            cached_n_with_content=n_with_content,
            align=8,
        )
        if bbox is None:
            log.warning("Empty mask -- disabling auto-crop")
            auto_crop = False
        else:
            cx, cy, cw, ch = bbox
            coverage = (cw * ch) / (frame_w * frame_h)
            if coverage > 0.6:
                log.warning("Bbox %dx%d covers %.1f%% of frame -- "
                            "disabling auto-crop (not worth ffmpeg overhead)",
                            cw, ch, coverage * 100)
                auto_crop = False

    # -- Stage 5: crop + ProPainter --
    if auto_crop and bbox is not None:
        cx, cy, cw, ch = bbox
        log.info("=== Stage 5a: ffmpeg crop video + masks to %dx%d at (%d,%d) ===",
                 cw, ch, cx, cy)
        crop_workspace = workspace / "_crop"
        crop_workspace.mkdir(exist_ok=True)
        crop_video = crop_workspace / f"{video.stem}_crop.mp4"
        crop_masks = crop_workspace / "masks"
        crop_out = crop_workspace / "out"
        crop_out.mkdir(exist_ok=True)

        worker._crop_video(video, crop_video, cx, cy, cw, ch, ffmpeg)
        worker._crop_masks(mask_dir, crop_masks, cx, cy, cw, ch)

        log.info("=== Stage 5b: ProPainter (chunked) ===")
        inpainted_crop = worker._run_propainter(
            video=crop_video, mask_dir=crop_masks, output_dir=crop_out,
            fp16=args.fp16, subvideo_length=args.subvideo_length,
            ffmpeg=ffmpeg,
        )

        log.info("=== Stage 6: ffmpeg overlay onto original ===")
        output.parent.mkdir(parents=True, exist_ok=True)
        worker._overlay(video, inpainted_crop, output, cx, cy, ffmpeg)
    else:
        log.info("=== Stage 5: ProPainter on full frame (no auto-crop) ===")
        full_out = workspace / "_full_out"
        full_out.mkdir(exist_ok=True)
        inpainted = worker._run_propainter(
            video=video, mask_dir=mask_dir, output_dir=full_out,
            fp16=args.fp16, subvideo_length=args.subvideo_length,
            ffmpeg=ffmpeg,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(inpainted, output)

    size_mb = output.stat().st_size / 1e6
    log.info("=== DONE: %s (%.1f MB) ===", output, size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
