"""
Re-do the final overlay step of a dynamic-mode pipeline run, using the
feathered-alpha composite, WITHOUT re-running ProPainter.

When to use this:

  * Your previous run finished with a visible rectangular box at the
    bbox boundary in the bottom-right (or wherever the watermark was).
    That's the legacy hard-edge overlay's signature artifact -- the
    inpaint result is correct, the composite is what's wrong.
  * You don't want to wait 3-4 hours for ProPainter to redo work that
    already produced a perfectly good inpainted crop.

What this tool needs from the workspace:

  * ``<ws>/_source_clean.mp4`` -- the re-encoded source (decode base)
  * ``<ws>/_source_clean/_crop/_bbox.json`` -- bbox coordinates
  * ``<ws>/_source_clean/_crop/masks/00000.png ...`` -- cropped masks
    (used as the alpha channel for the new composite)
  * ``<ws>/_source_clean/_crop/out/_source_clean_crop/inpaint_out.mp4``
    -- the ProPainter output (the cropped inpaint result; created by
    ``_run_propainter_chunked`` after concatenating per-chunk outputs)

Workspaces created BEFORE the bbox-persistence patch landed will be
missing ``_bbox.json``; you can either supply the coords manually with
``--bbox x,y,w,h`` or just re-run the pipeline once (the rerun will
write the sidecar and will then be recomposite-able forever after).

Usage::

    python -m tool.dynamic_recomposite \\
        --workspace "<repo>/output/dynamic/<stem>_<hash>" \\
        --output "<somewhere>.mp4"

    # With a custom feather width (default 5 px; larger = softer edge)
    python -m tool.dynamic_recomposite --workspace ... --output ... \\
        --feather 8

    # When the bbox sidecar is missing (old workspace, pre-patch)
    python -m tool.dynamic_recomposite --workspace ... --output ... \\
        --bbox 1340,820,256,256
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from backend.dynamic.workspace import (  # noqa: E402
    DYNAMIC_ROOT,
    workspace_for_video,
)
from backend.dynamic._worker import (  # noqa: E402
    _find_ffmpeg,
    _overlay,
    _probe_fps,
)


_VIDEO_STEM = "_source_clean"
_CROP_STEM = "_source_clean_crop"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("dynamic_recomposite")


def _discover_paths(ws: Path):
    """Return (base, crop, masks, bbox_path) for *ws*.

    Raises FileNotFoundError with a useful message if any piece is
    missing -- this is the failure mode users will hit when they
    pointed the tool at the wrong directory.
    """
    base = ws / f"{_VIDEO_STEM}.mp4"
    crop_out = (ws / _VIDEO_STEM / "_crop" / "out" / _CROP_STEM
                / "inpaint_out.mp4")
    masks = ws / _VIDEO_STEM / "_crop" / "masks"
    bbox_path = ws / _VIDEO_STEM / "_crop" / "_bbox.json"

    missing = []
    if not base.is_file():
        missing.append(f"base video: {base}")
    if not crop_out.is_file():
        missing.append(f"ProPainter output: {crop_out}")
    if not masks.is_dir() or not (masks / "00000.png").is_file():
        missing.append(f"mask sequence dir: {masks}")
    if missing:
        raise FileNotFoundError(
            "Workspace is missing required artefacts:\n  - "
            + "\n  - ".join(missing)
            + "\n\nThis tool needs a workspace that finished through "
            "the ProPainter stage. If the workspace was wiped or "
            "never reached ProPainter, re-run the pipeline instead."
        )
    return base, crop_out, masks, bbox_path


def _resolve_bbox(bbox_path: Path, cli_bbox: str | None):
    """Pick the bbox from the sidecar JSON, falling back to a CLI
    string ``x,y,w,h`` if the sidecar is absent (old workspaces).

    Returns the 4-tuple ``(x, y, w, h)`` as ints, or raises.
    """
    if cli_bbox:
        try:
            parts = [int(v.strip()) for v in cli_bbox.split(",")]
        except ValueError as e:
            raise SystemExit(f"--bbox must be 'x,y,w,h' ints: {e}")
        if len(parts) != 4:
            raise SystemExit("--bbox must be exactly 4 ints: x,y,w,h")
        return tuple(parts)

    if not bbox_path.is_file():
        raise SystemExit(
            f"bbox sidecar missing: {bbox_path}\n"
            "Either pass --bbox x,y,w,h manually (look up the values "
            "from the original run log, search for 'Mask bbox:'), or "
            "re-run the pipeline once -- new runs write this sidecar "
            "automatically."
        )

    with open(bbox_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return (int(data["x"]), int(data["y"]),
            int(data["w"]), int(data["h"]))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--workspace", type=Path,
        help="Path to the dynamic workspace dir (the one ending in "
             "<stem>_<hash>).",
    )
    g.add_argument(
        "--video", type=Path,
        help="Original source video -- the workspace path is derived "
             "the same way the pipeline derives it.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the recomposited mp4.",
    )
    p.add_argument(
        "--feather", type=int, default=5,
        help="Mask-edge feather radius in pixels (default: 5). Larger "
             "values produce a softer transition at the watermark "
             "boundary; too small risks the inpaint/original colour "
             "step still being visible.",
    )
    p.add_argument(
        "--bbox", type=str, default=None,
        help="Override bbox as 'x,y,w,h' ints, for workspaces created "
             "before the bbox-sidecar patch.",
    )
    p.add_argument(
        "--no-mask", action="store_true",
        help="Skip the alpha-feathered composite and use the legacy "
             "hard-edge paste -- only useful for A/B comparison.",
    )
    args = p.parse_args(argv)

    if args.video:
        ws = workspace_for_video(args.video.resolve())
    else:
        ws = args.workspace.resolve()

    if not ws.is_dir():
        raise SystemExit(f"Workspace does not exist: {ws}")

    base, crop, masks, bbox_path = _discover_paths(ws)
    cx, cy, cw, ch = _resolve_bbox(bbox_path, args.bbox)

    ffmpeg = _find_ffmpeg()

    # ffprobe lives next to ffmpeg in the bundled toolchain
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe"))
    if not Path(ffprobe).is_file():
        ffprobe_unix = str(Path(ffmpeg).with_name("ffprobe"))
        ffprobe = ffprobe_unix if Path(ffprobe_unix).is_file() else ""

    fps = _probe_fps(base, ffprobe) if ffprobe else None
    if (not args.no_mask) and (not fps or fps <= 0):
        raise SystemExit(
            f"Could not probe fps from {base} via {ffprobe!r}. "
            "Pass --no-mask to fall back to the legacy hard-edge "
            "overlay, or fix the ffmpeg toolchain."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("Workspace : %s", ws)
    log.info("Base      : %s", base)
    log.info("Inpaint   : %s", crop)
    log.info("Masks     : %s (%d PNGs)",
             masks, sum(1 for _ in masks.glob("*.png")))
    log.info("Bbox      : x=%d y=%d w=%d h=%d", cx, cy, cw, ch)
    log.info("Output    : %s", args.output)
    log.info("Mode      : %s",
             "hard-edge (--no-mask)" if args.no_mask
             else f"alpha-feather radius={args.feather} fps={fps:.3f}")

    if args.no_mask:
        _overlay(base, crop, args.output, cx, cy, ffmpeg)
    else:
        _overlay(
            base, crop, args.output, cx, cy, ffmpeg,
            mask_seq_dir=masks, feather_px=args.feather, fps=fps,
        )

    log.info("Wrote %s (%.1f MB)",
             args.output, args.output.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
