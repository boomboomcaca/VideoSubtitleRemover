"""
SAM + DeAOT + ProPainter pipeline wrapper.

Spawns the sibling ``watermark_remover`` project's bundled Python
environment (``env/python.exe``) to run a worker script that:

1. **SAM** -- click-driven first-frame segmentation.
2. **DeAOT (SegTracker)** -- propagates the mask through every frame.
3. **ProPainter** -- optical-flow-guided video inpainting.

Why subprocess to the bundled env (and not in-process)?
-------------------------------------------------------
The watermark_remover stack pulls in ``transformers``, ``groundingdino``,
``timm``, ``av``, ``einops`` etc. -- VSR Pro's venv intentionally
doesn't carry that load. The bundled ``env/`` is conda-style with
everything already installed, so we drive it as a worker. The contract
is small (JSON in, sentinel line out) and lets the parent process stay
in VSR Pro's venv with only stdlib + cv2.

Phase B may vendor the dependencies in-tree behind an optional install
extras; until then, MVP users keep the two repos side-by-side.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# On Windows, subprocess.Popen pops a console window for every child
# process by default. When the parent is a Tk GUI (pythonw.exe) those
# flashes look like errors. CREATE_NO_WINDOW (0x08000000) suppresses
# the console window without affecting stdin/stdout pipes -- the
# subprocess still captures them, we just don't see the cmd shell.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


# Phases the worker may emit, in canonical order. UIs can use this to
# render a stepper or to allocate ranges of an overall progress bar.
PHASES: Tuple[str, ...] = (
    "loading", "sam", "deaot", "mask_cleanup",
    "bbox", "crop", "propainter", "overlay", "done",
)

# Per-phase wall-time weights based on observed runs (RTX 3060 12GB,
# ~600 frames @ 1080p source with auto-crop on). The weights only need
# to be *relative*; the parent converts (phase, value 0..1) into an
# overall 0..1 fraction using these.
PHASE_WEIGHTS = {
    "loading": 1,
    "sam": 1,
    "deaot": 30,        # 11 min observed
    "mask_cleanup": 1,
    "bbox": 1,
    "crop": 1,
    "propainter": 30,   # 10 min observed with auto-crop
    "overlay": 1,
    "done": 0,
}
_TOTAL_WEIGHT = sum(PHASE_WEIGHTS.values())


def phase_to_overall(phase: str, value: float) -> float:
    """Convert a (phase, value) pair into an overall 0..1 progress fraction."""
    if phase not in PHASE_WEIGHTS:
        return 0.0
    completed = 0.0
    for p in PHASES:
        if p == phase:
            completed += PHASE_WEIGHTS[p] * max(0.0, min(1.0, value))
            break
        completed += PHASE_WEIGHTS[p]
    return completed / _TOTAL_WEIGHT


# Callable signature: (phase, value 0..1, extra string, overall 0..1).
# All four args are positional and required; pass a 4-arg lambda or
# a regular function. Extra is a free-form string (e.g. "596_clicks"
# or "384x384") -- never None.
ProgressCallback = Callable[[str, float, str, float], None]


# --------------------------------------------------------------------------- #
# Watermark-remover discovery
# --------------------------------------------------------------------------- #

_WM_REQUIRED = ("SegTracker.py", "seg_track_anything.py", "model_args.py")
_WM_REQUIRED_NESTED = ("ProPainter/inference_propainter.py", "env/python.exe")


def _looks_like_wm_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    if not all((p / f).is_file() for f in _WM_REQUIRED):
        return False
    if not all((p / f).is_file() for f in _WM_REQUIRED_NESTED):
        return False
    return True


def resolve_watermark_remover_path(explicit: Optional[str] = None) -> Path:
    """
    Locate a usable ``watermark_remover`` checkout (with its bundled env).

    Resolution order (first match wins):

    1. ``explicit`` (typically the ``--wm-path`` flag)
    2. ``VSR_WATERMARK_REMOVER_PATH`` env var
    3. ``../watermark_remover`` sibling
    4. ``D:/Repos/watermark_remover`` dev fallback
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]

    candidates: List[Optional[Path]] = [
        Path(explicit).expanduser() if explicit else None,
        Path(os.environ["VSR_WATERMARK_REMOVER_PATH"]).expanduser()
            if os.environ.get("VSR_WATERMARK_REMOVER_PATH") else None,
        repo_root.parent / "watermark_remover",
        Path("D:/Repos/watermark_remover"),
    ]

    tried = []
    for c in candidates:
        if c is None:
            continue
        c = c.resolve()
        tried.append(str(c))
        if _looks_like_wm_root(c):
            logger.debug("Resolved watermark_remover at %s", c)
            return c

    raise FileNotFoundError(
        "Could not locate a usable watermark_remover project. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\nA valid root must contain SegTracker.py, seg_track_anything.py, "
        "model_args.py, ProPainter/inference_propainter.py AND env/python.exe "
        "(the bundled conda env with all deps)."
    )


# --------------------------------------------------------------------------- #
# Click-prompt parsing
# --------------------------------------------------------------------------- #

ClickPoint = Tuple[int, int, int]  # (x, y, mode); mode: 1=positive, 0=negative


def parse_clicks(spec: str) -> List[ClickPoint]:
    """
    Parse ``--points`` into ``[(x, y, mode), ...]``.

    Format: semicolon-separated ``x,y[+/-]``. ``+`` (positive: mark
    watermark) is the default; ``-`` is negative (mark background).
    """
    points: List[ClickPoint] = []
    for raw in spec.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if raw[-1] in "+-":
            sign, body = raw[-1], raw[:-1]
        else:
            sign, body = "+", raw
        try:
            x_str, y_str = body.split(",", 1)
            x, y = int(x_str.strip()), int(y_str.strip())
        except ValueError as e:
            raise ValueError(f"Cannot parse click '{raw}': expected 'x,y[+/-]'") from e
        points.append((x, y, 1 if sign == "+" else 0))

    if not points:
        raise ValueError(f"No clicks parsed from spec: {spec!r}")
    return points


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@dataclass
class DynamicRemovalResult:
    output_video: Path
    wm_path: Path
    worker_returncode: int


_WORKER_REL = "backend/dynamic/_worker.py"
_RESULT_SENTINEL = "DYNAMIC_RESULT "
_PROGRESS_SENTINEL = "PROGRESS "

# SegTracker prints "processed frame N, obj_num X" (or "...obj_num X\r")
# once per video frame during DeAOT propagation. We capture these
# implicit progress signals on the parent side to fill the long silent
# gap that would otherwise sit at 3% for several minutes.
import re
_DEAOT_FRAME_RE = re.compile(r"^processed frame (\d+)")


def _parse_progress(line: str) -> Optional[Tuple[str, float, str]]:
    """Parse a ``PROGRESS phase value [extra]`` worker sentinel.

    Returns ``(phase, value, extra)`` or None if the line is not a
    progress sentinel or is malformed (silently ignored -- progress is
    advisory).
    """
    if not line.startswith(_PROGRESS_SENTINEL):
        return None
    body = line[len(_PROGRESS_SENTINEL):].strip()
    parts = body.split(maxsplit=2)
    if len(parts) < 2:
        return None
    try:
        value = float(parts[1])
    except ValueError:
        return None
    extra = parts[2] if len(parts) > 2 else ""
    return parts[0], value, extra


def _parse_deaot_frame(line: str) -> Optional[int]:
    """Return the frame index from a SegTracker ``processed frame N`` line."""
    m = _DEAOT_FRAME_RE.match(line)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def run_dynamic_removal(
    video: Path,
    clicks: Sequence[ClickPoint],
    output: Path,
    wm_path: Path,
    *,
    fp16: bool = True,
    subvideo_length: int = 80,
    aot_model: str = "r50_deaotl",
    auto_crop: bool = True,
    crop_padding: int = 96,
    keep_intermediates: bool = False,
    stream_to_stderr: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> DynamicRemovalResult:
    """
    Run the full SAM + DeAOT + ProPainter pipeline on *video*.

    The actual heavy lifting happens in a worker subprocess running under
    ``<wm_path>/env/python.exe``. The worker writes its log lines to
    *stderr*; this function tees those to our stderr if
    ``stream_to_stderr`` is True (default) so the user sees progress.

    Parameters
    ----------
    video, output
        Absolute paths. Output's parent is created if missing.
    clicks
        First-frame SAM prompts. Mix of positive and negative points.
    wm_path
        Result of :func:`resolve_watermark_remover_path`.
    fp16
        Pass ``--fp16`` to ProPainter (halves VRAM, recommended).
    subvideo_length
        ProPainter ``--subvideo_length``; lower for less VRAM. With
        ``auto_crop=True`` (default) the crop is small so 80 fits
        comfortably on a 12 GB GPU; without auto-crop, drop to 20 or
        even 10 on a 12 GB card processing 1080p input.
    aot_model
        Must be a key of ``seg_track_anything.aot_model2ckpt``.
    auto_crop
        If True (default), the worker computes the bounding box of the
        tracked mask across all frames, crops the video + masks to a
        small region (with padding), runs ProPainter on the crop, and
        overlays the result back onto the original. This is the
        difference between a 10-minute run and a 90-minute run on
        consumer GPUs -- the only reason to disable is debugging or if
        the watermark sweeps across most of the frame.
    crop_padding
        Pixels of context to leave around the tracked-mask bounding box
        (per side) when auto-cropping. Larger padding gives ProPainter
        more surrounding texture to learn from but increases VRAM /
        wall-time cost. Default 96 is a good balance for logo-sized
        watermarks.

    Returns
    -------
    DynamicRemovalResult
    """
    video = video.resolve()
    output = output.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")

    wm_python = wm_path / "env" / "python.exe"
    if not wm_python.is_file():
        raise FileNotFoundError(
            f"watermark_remover bundled python missing: {wm_python}"
        )

    # The worker lives in VSR Pro's tree, but we run it under wm_env --
    # so we pass its absolute path on the command line.
    repo_root = Path(__file__).resolve().parents[2]
    worker_path = repo_root / _WORKER_REL
    if not worker_path.is_file():
        raise FileNotFoundError(f"Worker script not found: {worker_path}")

    # Workspace for intermediates lives *inside* VSR Pro's tree, not in
    # the sibling watermark_remover repo. SegTracker insists on writing
    # its masks under <wm_path>/output (the path is hard-coded via
    # __file__), so the worker stages them there briefly and then
    # relocates everything into this workspace before running
    # ProPainter / ffmpeg.
    #
    # Workspace key is ``<stem>_<sha1(abs_path)[:8]>`` rather than just
    # ``<stem>``. Two different videos that happen to share a stem
    # (e.g. ~/dl/video.mp4 and ~/backup/video.mp4) would otherwise both
    # route to ``output/dynamic/video/`` and the second run would silently
    # reuse the first's ``_source_clean.mp4`` -- which was encoded from
    # the wrong source -- producing garbage output with no error.
    # Hashing the absolute path keeps "same file rerun" → "same key"
    # (preserves the _source_clean.mp4 reuse fast-path for retries)
    # while making "same name, different file" → "different key".
    repo_root = Path(__file__).resolve().parents[2]
    path_hash = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:8]
    workspace = repo_root / "output" / "dynamic" / f"{video.stem}_{path_hash}"
    workspace.mkdir(parents=True, exist_ok=True)
    logger.info("Dynamic intermediates workspace: %s", workspace)

    # Orphan detection: workspaces created before the hash-suffix rename
    # live at ``output/dynamic/<stem>/`` (no suffix). Auto-cleanup never
    # touches them because the cleanup path keys off the current workspace.
    # Warn so the user can delete manually; we don't auto-delete because
    # two different videos could have collided into the same legacy dir
    # back when the bug was live, and we have no way to know which one
    # is the rightful owner.
    legacy_workspace = repo_root / "output" / "dynamic" / video.stem
    if legacy_workspace.is_dir() and legacy_workspace != workspace:
        logger.warning(
            "Detected legacy (pre-hash) workspace at %s -- this is an "
            "orphan from before the workspace naming change and will not "
            "be auto-cleaned. Safe to delete manually after confirming "
            "any prior runs of this video produced good output.",
            legacy_workspace,
        )

    payload = {
        "video": str(video),
        "output": str(output),
        "clicks": [list(c) for c in clicks],
        "aot_model": aot_model,
        "subvideo_length": int(subvideo_length),
        "fp16": bool(fp16),
        "auto_crop": bool(auto_crop),
        "crop_padding": int(crop_padding),
        "workspace": str(workspace),
    }

    logger.info("Spawning worker under %s", wm_python)
    logger.info("Payload: %s", payload)

    # Run with cwd=wm_path so the worker's sys.path and SegTracker's
    # relative 'ckpt/...' paths resolve correctly.
    proc = subprocess.Popen(
        [str(wm_python), str(worker_path)],
        cwd=str(wm_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if stream_to_stderr else subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_NO_WINDOW,
    )

    # Feed JSON in and close stdin so the worker proceeds.
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload))
    proc.stdin.close()

    result_line: Optional[str] = None
    assert proc.stdout is not None

    # State for synthesising per-frame DeAOT progress. The worker emits
    # PROGRESS deaot 0.0 total=596 at the start of tracking; we
    # remember the total and then turn each "processed frame N" line
    # SegTracker prints into a synthetic (phase, value) update.
    deaot_total = 0
    last_deaot_frame_emitted = -1

    def _dispatch(phase: str, value: float, extra: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(phase, value, extra,
                              phase_to_overall(phase, value))
        except Exception:  # noqa: BLE001
            logger.exception("progress_callback raised; continuing")

    # Tee stdout (and merged stderr) line-by-line to our stderr so the
    # operator sees DeAOT + ProPainter progress in real time. Intercept
    # the two sentinel families: DYNAMIC_RESULT for the final path,
    # PROGRESS for phase-level progress (which we forward to the
    # caller's callback if provided).
    for raw_line in proc.stdout:
        # SegTracker uses ``end='\r'`` for its per-frame prints, so a
        # single "line" yielded by the iterator may actually contain
        # many \r-separated updates. Split them out so we don't miss any.
        for line in raw_line.replace("\r", "\n").split("\n"):
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(_RESULT_SENTINEL):
                result_line = line[len(_RESULT_SENTINEL):].strip()
                print(line, file=sys.stderr)
                continue

            parsed = _parse_progress(line)
            if parsed is not None:
                phase, value, extra = parsed
                # Capture ``total=N`` so we can render per-frame
                # DeAOT progress as SegTracker streams frames.
                if phase == "deaot" and extra.startswith("total="):
                    try:
                        deaot_total = int(extra.split("=", 1)[1])
                        last_deaot_frame_emitted = -1
                    except ValueError:
                        pass
                _dispatch(phase, value, extra)
                print(line, file=sys.stderr)
                continue

            # Synthesise DeAOT per-frame progress from SegTracker's
            # plain stdout chatter. Throttle to one emission per ~1%
            # of the video to keep callback rate sane on long videos.
            n = _parse_deaot_frame(line)
            if n is not None and deaot_total > 0:
                # Throttle: emit at most every max(1, total/100) frames
                # so a 60000-frame video gets ~100 updates not 60000.
                stride = max(1, deaot_total // 100)
                if n - last_deaot_frame_emitted >= stride or n + 1 == deaot_total:
                    last_deaot_frame_emitted = n
                    _dispatch("deaot", min(1.0, (n + 1) / deaot_total),
                              f"{n + 1}/{deaot_total}")

            print(line, file=sys.stderr)
    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"Dynamic worker failed with exit code {rc}. "
            "See the streamed log above for the failure point."
        )
    if not result_line:
        raise RuntimeError(
            "Worker exited 0 but did not print a DYNAMIC_RESULT sentinel."
        )

    out_path = Path(result_line)
    if not out_path.is_file():
        raise FileNotFoundError(
            f"Worker reported success but output file missing: {out_path}"
        )

    # Workspace lifecycle: this point is the only place we can prove the
    # run succeeded end-to-end (worker exited 0, sentinel parsed, final
    # mp4 on disk). Drop the 2-4 GB of intermediates unless the caller
    # opted in to keep them.
    #
    # Failure paths -- worker non-zero rc, missing sentinel, missing
    # output -- raise BEFORE reaching here, so the workspace is naturally
    # preserved for inspection / for tool/dynamic_resume_from_masks.py
    # to pick up. The workspace path was logged at start of run.
    if not keep_intermediates:
        try:
            shutil.rmtree(workspace, ignore_errors=True)
            logger.info("Cleaned up workspace: %s", workspace)
        except Exception:  # noqa: BLE001
            logger.warning("Workspace cleanup failed at %s",
                           workspace, exc_info=True)

    return DynamicRemovalResult(
        output_video=out_path,
        wm_path=wm_path,
        worker_returncode=rc,
    )


__all__ = [
    "ClickPoint",
    "DynamicRemovalResult",
    "parse_clicks",
    "resolve_watermark_remover_path",
    "run_dynamic_removal",
]
