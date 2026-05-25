"""
Workspace path resolution + introspection + wipe utilities.

Single source of truth for "where do dynamic-mode intermediates live"
and "what's currently in there". Both the CLI cleanup tool and the
GUI pre-launch confirmation dialog import from here so they can't
drift out of sync with the production worker's workspace layout.

Workspace key is ``<stem>_<sha1(abs_path)[:8]>`` to keep "same file
rerun" routing to the same dir (preserves _source_clean.mp4 reuse)
while making "same name, different file" route to a different dir.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_ROOT = _REPO_ROOT / "output" / "dynamic"

# These names mirror what `_worker.py` writes. If the worker ever
# renames an intermediate, update both places together.
_INTERMEDIATE_VIDEO_NAME = "_source_clean.mp4"
_VIDEO_STEM = "_source_clean"


@dataclass(frozen=True)
class WorkspaceState:
    """Structured snapshot of what stages a workspace has cached.

    Mirrors the same dirs the worker uses on a resume detection -- the
    field set is intentionally exhaustive so a UI can decide *which*
    stages would actually be re-run by a fresh invocation, not just
    "is there anything cached".
    """

    workspace: Path
    exists: bool
    intermediate_exists: bool
    n_masks: int                  # 0 if dir missing
    cleanup_sidecar_exists: bool
    crop_video_exists: bool
    n_crop_masks: int             # 0 if dir missing
    n_chunks_done: int            # ProPainter chunks with non-trivial output
    size_bytes: int

    @property
    def has_any_cache(self) -> bool:
        """True iff any stage's artefact is present. Drives the
        "show confirm dialog vs. just launch" decision in the GUI."""
        return (
            self.intermediate_exists
            or self.n_masks > 0
            or self.cleanup_sidecar_exists
            or self.crop_video_exists
            or self.n_crop_masks > 0
            or self.n_chunks_done > 0
        )

    def stage_summary(self) -> list[str]:
        """Human-readable bullets for a confirmation dialog body."""
        bullets = []
        if self.intermediate_exists:
            bullets.append("Reencoded source video (intermediate)")
        if self.n_masks > 0:
            bullets.append(f"{self.n_masks:,} DeAOT tracking masks")
        if self.cleanup_sidecar_exists:
            bullets.append("Mask cleanup summary cache")
        if self.crop_video_exists:
            bullets.append("Cropped video for ProPainter")
        if self.n_crop_masks > 0:
            bullets.append(f"{self.n_crop_masks:,} cropped masks")
        if self.n_chunks_done > 0:
            bullets.append(f"{self.n_chunks_done} completed ProPainter chunk(s)")
        return bullets


def workspace_for_video(video: Path) -> Path:
    """Return the canonical workspace dir for *video*.

    Does NOT create the directory; callers that want the actual
    on-disk dir should ``mkdir(parents=True, exist_ok=True)`` it.
    """
    path_hash = hashlib.sha1(
        str(video.resolve()).encode("utf-8")
    ).hexdigest()[:8]
    return DYNAMIC_ROOT / f"{video.stem}_{path_hash}"


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def describe_workspace(ws: Path) -> WorkspaceState:
    """Snapshot the on-disk state of *ws* without touching anything."""
    if not ws.is_dir():
        return WorkspaceState(
            workspace=ws, exists=False,
            intermediate_exists=False, n_masks=0,
            cleanup_sidecar_exists=False,
            crop_video_exists=False, n_crop_masks=0,
            n_chunks_done=0, size_bytes=0,
        )

    intermediate = ws / _INTERMEDIATE_VIDEO_NAME
    masks_dir = ws / _VIDEO_STEM / f"{_VIDEO_STEM}_masks"
    sidecar = ws / _VIDEO_STEM / f"{_VIDEO_STEM}_masks_cleanup.json"
    crop_video = ws / _VIDEO_STEM / "_crop" / f"{_VIDEO_STEM}_crop.mp4"
    crop_masks = ws / _VIDEO_STEM / "_crop" / "masks"
    chunks_dir = ws / _VIDEO_STEM / "_crop" / "out" / "_chunks"

    n_masks = (sum(1 for _ in masks_dir.glob("*.png"))
               if masks_dir.is_dir() else 0)
    n_crop_masks = (sum(1 for _ in crop_masks.glob("*.png"))
                    if crop_masks.is_dir() else 0)

    n_chunks_done = 0
    if chunks_dir.is_dir():
        for c in chunks_dir.iterdir():
            if not c.is_dir():
                continue
            out_mp4 = c / "out" / "in" / "inpaint_out.mp4"
            try:
                if out_mp4.is_file() and out_mp4.stat().st_size > 1024:
                    n_chunks_done += 1
            except OSError:
                pass

    return WorkspaceState(
        workspace=ws,
        exists=True,
        intermediate_exists=intermediate.is_file(),
        n_masks=n_masks,
        cleanup_sidecar_exists=sidecar.is_file(),
        crop_video_exists=crop_video.is_file(),
        n_crop_masks=n_crop_masks,
        n_chunks_done=n_chunks_done,
        size_bytes=_dir_size(ws),
    )


def wipe_workspace(
    ws: Path,
    *,
    keep_source_clean: bool = False,
    only_propainter: bool = False,
) -> bool:
    """Remove the cached artefacts under *ws*.

    Modes
    -----
    * ``only_propainter=True``  -- delete only the ProPainter chunks +
      final concat; keep DeAOT masks, cleanup sidecar, and the cropped
      inputs. Use when ProPainter output looks wrong but earlier stages
      are fine.
    * ``keep_source_clean=True`` -- delete everything except the
      reencoded intermediate (saves 1-3 min on the next run; ignored
      when ``only_propainter=True``).
    * Default -- nuke everything except the workspace dir itself,
      letting the next run reuse the empty container.

    Returns True iff every wipe target was successfully removed.
    """
    if not ws.is_dir():
        return True   # nothing to do is success

    chunks_dir = ws / _VIDEO_STEM / "_crop" / "out" / "_chunks"
    crop_dir = ws / _VIDEO_STEM / "_crop"
    intermediate = ws / _INTERMEDIATE_VIDEO_NAME
    deaot_root = ws / _VIDEO_STEM

    if only_propainter:
        ok_a = _safe_remove(chunks_dir)
        ok_b = _safe_remove(crop_dir / "out" / _VIDEO_STEM)
        return ok_a and ok_b

    targets = []
    if not keep_source_clean:
        targets.append(intermediate)
    targets.append(deaot_root)

    return all(_safe_remove(t) for t in targets)


def _safe_remove(target: Path) -> bool:
    if not target.exists():
        return True
    try:
        if target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target, ignore_errors=False)
        return True
    except OSError:
        return False


def format_size(n_bytes: int) -> str:
    """Compact byte-count formatter for UI labels (e.g. ``3.2 GB``)."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    val = n_bytes / 1024
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:,.1f} {unit}"
        val /= 1024
    return f"{n_bytes} B"  # unreachable, mollifies type checkers
