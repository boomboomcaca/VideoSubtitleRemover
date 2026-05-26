"""
In-environment worker for dynamic watermark removal.

Runs inside ``<wm_path>/env/python.exe`` (the watermark_remover project's
bundled conda env), invoked as a subprocess by the parent VSR Pro
process. The parent stays in VSR Pro's venv with only stdlib + cv2; the
heavy deps (torch, transformers, groundingdino, av, einops, ...) live
exclusively in the wm_env.

Pipeline
--------
1. **SAM** -- click-driven first-frame segmentation
2. **DeAOT** -- propagate the mask through every frame
3. **Mask cleanup** -- drop the ``_new.png`` auxiliary files SegTracker
   emits every ``sam_gap`` frames, and rewrite each mask to a clean
   binary of object id == 1 only (otherwise SegTracker's "segment
   everything" mode tracks 10-20 extra objects by end of clip)
4. **Auto-crop** (if enabled) -- compute the union bounding box of all
   masks, crop video + masks to that region with padding. Drops a
   typical 1080p logo-removal job from ~90 min to ~10 min on a 12 GB
   GPU because RAFT correlation memory is O(H * W).
5. **ProPainter** -- optical-flow-guided inpainting on the (possibly
   cropped) clip
6. **Overlay** (if cropped in step 4) -- composite the inpainted crop
   back onto the original full-resolution video

Contract
--------
Input (stdin, JSON)::

    {
      "video": "<absolute path>",
      "output": "<absolute path>",
      "clicks": [[x, y, mode], ...],   # mode: 1=positive, 0=negative
      "aot_model": "r50_deaotl",
      "subvideo_length": 80,
      "fp16": true,
      "auto_crop": true,
      "crop_padding": 96
    }

Output (stdout, last line)::

    DYNAMIC_RESULT <absolute path to written MP4>

On failure: non-zero exit code; error to stderr.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

# Suppress console-window popups for our ffmpeg/ProPainter children.
# When the parent VSR Pro GUI runs under pythonw.exe (no console), the
# default behaviour on Windows is for each new console subprocess to
# allocate its own cmd window, which flashes to the user. The flag is
# a no-op on non-Windows / older Pythons.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

# -- bootstrap: the parent process sets cwd to the watermark_remover
# root before spawning us, so SegTracker's relative paths (ckpt/...)
# resolve correctly.
WM_PATH = Path.cwd().resolve()

sys.path.insert(0, str(WM_PATH))
sys.path.insert(0, str(WM_PATH / "sam"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER %(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("dynamic_worker")


# --------------------------------------------------------------------------- #
# Structured progress emission
# --------------------------------------------------------------------------- #
# The parent process scans our stdout for lines starting with "PROGRESS "
# and forwards them to its progress_callback. Keep the format stable:
#
#     PROGRESS <phase> <value 0..1> [optional extra string]
#
# Phases (in order): loading, sam, deaot, mask_cleanup, bbox, crop,
# propainter, overlay, done. Each phase value 0.0 = starting, 1.0 =
# finished. Intermediate values are emitted only where cheap to compute
# without parsing subprocess output -- DeAOT's per-frame counter and
# ProPainter's tqdm bar are NOT mirrored here, because that would
# require character-level (rather than line-level) stdout parsing in
# the parent. UIs that want a finer-grained bar should poll the mask
# directory file count for DeAOT and the output frame count for
# ProPainter.

def emit_progress(phase: str, value: float = 0.0, extra: str = "") -> None:
    """Write a structured progress sentinel to stdout."""
    # stdout, not stderr, because the parent reads stdout. Flush so
    # block-buffering doesn't hide our progress from the parent.
    msg = f"PROGRESS {phase} {value:.3f}"
    if extra:
        msg += f" {extra}"
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# ffmpeg discovery
# --------------------------------------------------------------------------- #

def _find_ffmpeg() -> str:
    """Return the absolute path to a usable ffmpeg binary.

    Prefer the watermark_remover bundled one (predictable version,
    available even when the user has no system ffmpeg). Fall back to
    PATH lookup.
    """
    bundled = WM_PATH / "env" / "Library" / "bin" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError(
        "ffmpeg not found. Expected at "
        f"{bundled} or on PATH."
    )


def _find_ffprobe() -> Optional[str]:
    """Locate ffprobe (companion to ffmpeg). Returns None if unavailable.

    Not strictly required: when missing we fall back to parsing
    ``ffmpeg -i`` stderr for stream metadata.
    """
    bundled = WM_PATH / "env" / "Library" / "bin" / "ffprobe.exe"
    if bundled.is_file():
        return str(bundled)
    return shutil.which("ffprobe")


# --------------------------------------------------------------------------- #
# ffmpeg-based video reader (cv2 replacement)
# --------------------------------------------------------------------------- #
#
# cv2.VideoCapture.read() can block FOREVER on malformed H.264 frames,
# unusual B-frame references, or container-level corruption. There is
# no timeout API and no way to interrupt it from Python (the call holds
# the GIL via libav internals on the C side). We hit this on a real
# 90-min user video: the read hung at frame 84918 and the worker
# became a 3-hour zombie before we noticed.
#
# Reading frames through an ffmpeg.exe subprocess + a stdout pipe
# moves the decode into a separate OS process. The parent reads raw
# uint8 BGR bytes with a per-frame watchdog timeout; if ffmpeg ever
# hangs we kill the subprocess and treat the read as EOF (DeAOT stops
# cleanly at that point, and the rest of the pipeline -- mask
# cleanup, ffmpeg slicing, ProPainter chunking -- still works on the
# truncated mask set).
#
# Performance overhead vs in-process cv2 is in the 5-10% range for
# raw uint8 BGR over a pipe (no per-frame encode/decode, just memcpy).
# Worth the cost trivially given the alternative is process death.

class FfmpegVideoReader:
    """Stream BGR frames from a video file via an ffmpeg subprocess.

    Drop-in for the cv2.VideoCapture pattern we used in this module --
    the API exposes ``width``, ``height``, ``fps``, ``n_frames``, and
    ``read() -> (ok, ndarray)``. Use as a context manager so the child
    process is always cleaned up.

    On any read that doesn't complete within ``timeout_sec`` (default 30s)
    the ffmpeg subprocess is killed and subsequent ``read()`` calls
    return ``(False, None)``. The class never raises from read errors --
    callers detect EOF / failure via the ``ok`` flag, same as cv2.
    """

    def __init__(
        self,
        video_path: Path,
        ffmpeg: str,
        ffprobe: Optional[str] = None,
        timeout_sec: float = 30.0,
    ):
        self._video = Path(video_path)
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe if ffprobe is not None else _find_ffprobe()
        self._timeout = float(timeout_sec)
        self._proc: Optional[subprocess.Popen] = None
        self._closed = False
        self._frames_read = 0

        # Probe metadata up front -- callers need width/height before
        # the first read to allocate buffers and crop coordinates.
        meta = self._probe()
        self.width: int = meta["width"]
        self.height: int = meta["height"]
        self.fps: float = meta["fps"]
        self.n_frames: int = meta["n_frames"]
        self._frame_bytes = self.width * self.height * 3

    # ----- metadata -----

    def _probe(self) -> dict:
        """Probe video dimensions and frame count via ffprobe or fallback."""
        if self._ffprobe:
            try:
                return self._probe_via_ffprobe()
            except Exception as e:  # noqa: BLE001
                log.warning("ffprobe failed (%s); falling back to ffmpeg -i", e)
        return self._probe_via_ffmpeg_stderr()

    def _probe_via_ffprobe(self) -> dict:
        import json as _json
        cmd = [
            self._ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "json",
            str(self._video),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=20, creationflags=_NO_WINDOW,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe rc={proc.returncode}: {proc.stderr}")
        data = _json.loads(proc.stdout)
        stream = data["streams"][0]
        # r_frame_rate is "num/den"
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
        n_frames_raw = stream.get("nb_frames")
        try:
            n_frames = int(n_frames_raw) if n_frames_raw not in (None, "N/A") else 0
        except ValueError:
            n_frames = 0
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "n_frames": n_frames,
        }

    def _probe_via_ffmpeg_stderr(self) -> dict:
        """Parse ``ffmpeg -i <file>`` stderr to extract stream info."""
        import re as _re
        cmd = [self._ffmpeg, "-i", str(self._video)]
        # ffmpeg exits with rc=1 when only -i is given (no output)
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=20, creationflags=_NO_WINDOW,
        )
        text = proc.stderr
        # e.g. "Video: h264 ..., 1920x1080 [SAR 1:1 DAR 16:9], 5990 kb/s, 60 fps"
        m_size = _re.search(r"(\d+)x(\d+)\b", text)
        m_fps = _re.search(r"([\d.]+)\s*fps", text)
        if not m_size:
            raise RuntimeError("Could not parse width/height from ffmpeg stderr")
        return {
            "width": int(m_size.group(1)),
            "height": int(m_size.group(2)),
            "fps": float(m_fps.group(1)) if m_fps else 30.0,
            "n_frames": 0,  # ffmpeg -i doesn't report frame count
        }

    # ----- lifecycle -----

    def __enter__(self) -> "FfmpegVideoReader":
        self._spawn()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _spawn(self) -> None:
        # ``-err_detect ignore_err`` is the lone "tolerate bad bitstream"
        # flag we need. We used to also pass ``-fflags +discardcorrupt``
        # AND ``-vsync passthrough`` for what felt like extra robustness,
        # but on a 27-min H.264 user video those flags caused ffmpeg to
        # emit a clean EOF after ~18726 frames (out of 98668), silently
        # truncating the dynamic-watermark pipeline to ~5 minutes of
        # output. ffmpeg's own ``-map 0:v:0 -f null -`` test on the
        # same source processed all 98668 frames once the two flags
        # were removed, so they were causing the early termination,
        # not protecting against it.
        # -hwaccel cuda offloads H.264 / HEVC decode to NVDEC (NVIDIA
        # hardware decoder). Frames are downloaded to CPU before we
        # convert to bgr24 (our consumer is numpy on CPU), but the
        # actual decode cost drops to near-zero. If the GPU isn't
        # NVIDIA / decoder unavailable, ffmpeg silently falls back to
        # software decode -- no behaviour change required.
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-err_detect", "ignore_err",
            "-hwaccel", "cuda",
            "-i", str(self._video),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-",
        ]
        log.debug("FfmpegVideoReader spawn: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
            # The default IO buffer (~8KB) is way too small for raw 1080p
            # frames (~6 MB each). Larger buffer reduces syscall churn.
            bufsize=self._frame_bytes * 2,
        )

    def read(self):
        """Read one frame. Returns ``(ok, frame_bgr_ndarray)``.

        Returns ``(False, None)`` on EOF, partial read, or watchdog
        timeout. After a failure the reader is permanently closed; the
        caller should not retry."""
        import numpy as np

        if self._closed or self._proc is None:
            return False, None

        buf = self._read_exact_with_timeout(self._frame_bytes, self._timeout)
        if buf is None or len(buf) < self._frame_bytes:
            self.close()
            return False, None

        frame = np.frombuffer(buf, dtype=np.uint8).reshape(
            self.height, self.width, 3,
        )
        self._frames_read += 1
        return True, frame

    def _read_exact_with_timeout(self, n: int, timeout: float):
        """Read exactly *n* bytes from ffmpeg stdout, with watchdog.

        Windows pipes don't support ``select()``, so we use a daemon
        reader thread joined with a wall-clock timeout. On timeout the
        ffmpeg subprocess is killed so the thread's blocking read
        unblocks; the thread then exits naturally as the process winds
        down."""
        import threading

        result_buf = bytearray()
        target = [n]
        done = threading.Event()
        err = [None]
        proc = self._proc

        def _reader():
            try:
                stdout = proc.stdout
                assert stdout is not None
                while len(result_buf) < target[0]:
                    chunk = stdout.read(target[0] - len(result_buf))
                    if not chunk:
                        break
                    result_buf.extend(chunk)
            except Exception as e:  # noqa: BLE001
                err[0] = e
            finally:
                done.set()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        if not done.wait(timeout):
            # Watchdog fire: ffmpeg hung. Killing it should unblock the
            # reader thread (Windows pipes EOF when the writer dies).
            log.warning(
                "ffmpeg read timed out after %.1fs at frame %d; killing.",
                timeout, self._frames_read,
            )
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            # Give the reader thread a brief grace period to notice
            done.wait(1.0)
            return None

        if err[0] is not None:
            log.warning("ffmpeg reader thread raised: %s", err[0])
            return None
        return bytes(result_buf) if len(result_buf) >= n else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None


# --------------------------------------------------------------------------- #
# Mask processing
# --------------------------------------------------------------------------- #

def _clean_one_mask_worker(
    args: Tuple[str, float, float, int],
) -> Tuple[int, bool, Optional[Tuple[int, int, int, int]]]:
    """Per-frame mask cleanup; runs inside a ProcessPoolExecutor child.

    Module-level so it pickles cleanly across processes on Windows. Re-
    imports its dependencies lazily because each child reaches this
    function before the parent's main() block has run any heavy imports.

    ``args`` is ``(png_path, ref_cx, ref_cy, dilate_px)`` where
    ``dilate_px`` is how many pixels to grow the cleaned mask outward
    before saving (handles anti-aliased / soft watermark edges that the
    raw SAM mask under-covers, leaving 1-3 px halos in the inpaint
    output). 0 disables dilation.

    Returns ``(max_object_id_seen, was_drifted, bbox)`` -- the parent
    aggregates these to keep the existing log line accurate.
    """
    png_path: str
    ref_cx: float
    ref_cy: float
    dilate_px: int
    png_path, ref_cx, ref_cy, dilate_px = args

    import numpy as np
    from PIL import Image

    # Prefer cv2 (~2-3x faster CC than scipy); fall back to scipy on
    # wm_envs that somehow lack cv2. ``backend == "none"`` skips
    # CC-based filtering entirely (matches the historical fallback).
    try:
        import cv2 as _cv2  # noqa: F401
        backend = "cv2"
    except ImportError:
        try:
            from scipy.ndimage import label as _cc_label, center_of_mass as _com  # noqa: F401
            backend = "scipy"
        except ImportError:
            backend = "none"

    arr = np.array(Image.open(png_path))
    max_obj = int(arr.max())
    # Tolerates {0,1} (fresh from DeAOT) or {0,255} (resume scenario).
    binary = (arr > 0).astype(np.uint8)
    drift = False

    if binary.any() and ref_cx is not None and backend != "none":
        if backend == "cv2":
            num_labels, labels, _stats, centroids = _cv2.connectedComponentsWithStats(
                binary, connectivity=4)
            if num_labels > 2:  # background + 2+ foreground CCs
                # cv2 returns centroids as (cx, cy); index 0 is bg.
                best_label = 1
                best_dist = float("inf")
                for i in range(1, num_labels):
                    cx_i, cy_i = centroids[i]
                    d = (cy_i - ref_cy) ** 2 + (cx_i - ref_cx) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_label = i
                kept = (labels == best_label).astype(np.uint8)
                if kept.sum() < binary.sum():
                    drift = True
                binary = kept
        else:  # scipy path -- identical semantics to the original loop
            labels, n_cc = _cc_label(binary)
            if n_cc > 1:
                centroids = _com(binary, labels, range(1, n_cc + 1))
                best_label = 1
                best_dist = float("inf")
                for i, (cy_i, cx_i) in enumerate(centroids, start=1):
                    d = (cy_i - ref_cy) ** 2 + (cx_i - ref_cx) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_label = i
                kept = (labels == best_label).astype(np.uint8)
                if kept.sum() < binary.sum():
                    drift = True
                binary = kept

    # Optional outward dilation (capped at 20px to prevent excessive growth).
    # Watermarks with crisp circular outlines or anti-aliased glyph edges
    # leak 1-3 px past whatever SAM segments as "the watermark", and
    # ProPainter has no chance of reconstructing those edge pixels because
    # they sit OUTSIDE the mask -- the user sees a faint halo / outline
    # ring in the cleaned video. Growing the mask by a handful of pixels
    # swallows the soft edge at the cost of asking ProPainter to fill a
    # slightly larger area (no measurable quality drop on the 4-12 px
    # range we care about).
    if dilate_px > 0 and binary.any():
        # Clamp to reasonable maximum to prevent pathological ksize values.
        safe_dilate = min(dilate_px, 20)
        ksize = 2 * safe_dilate + 1
        if backend == "cv2":
            kernel = _cv2.getStructuringElement(
                _cv2.MORPH_ELLIPSE, (ksize, ksize))
            binary = _cv2.dilate(binary, kernel, iterations=1)
        elif backend == "scipy":
            from scipy.ndimage import binary_dilation
            binary = binary_dilation(
                binary, iterations=safe_dilate).astype(np.uint8)
        # backend == "none" -> skip silently; warning already emitted
        # in the parent when cv2/scipy were both missing.

    # Compute the bbox of the final binary mask. The parent aggregates
    # these into a union bbox during the cleanup pass, eliminating the
    # need for a second full scan in _compute_bbox -- saves ~10-15 min
    # of I/O on a 98K-mask video. Returned as inclusive (x_min, y_min,
    # x_max, y_max) to match _compute_bbox's existing convention; None
    # signals "this frame has no mask content".
    if binary.any():
        ys, xs = np.where(binary)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()))
    else:
        bbox = None

    # Fast path: skip the rewrite when the input PNG is already in the
    # canonical {0, 255} binary form AND no drift CC was filtered AND
    # no dilation is requested (dilation always changes pixel data, so
    # we must rewrite). This is the common case on resume runs (e.g.
    # dynamic_resume_from_masks) where a previous pass already cleaned
    # the masks -- avoids re-zlib-encoding ~100K files for no
    # behavioural change.
    if not drift and max_obj == 255 and binary.any() and dilate_px == 0:
        # All non-zero pixels must be exactly 255 (no stray palette
        # indices like {0, 1, 255}); cheaper than np.unique.
        if int(arr[arr > 0].min()) == 255:
            return max_obj, drift, bbox

    # compress_level=1 (vs PIL's PNG default of 6) is 2-3x faster on
    # save; for binary masks the deflate output is nearly identical in
    # size at either level because long runs of zeros compress trivially.
    # Pixel values are bit-identical -- ProPainter sees the same masks.
    Image.fromarray(binary * 255).save(png_path, compress_level=1)
    return max_obj, drift, bbox


def _clean_segtracker_masks(mask_dir: Path, dilate_px: Optional[int] = None):
    """Sanitise the raw mask sequence SegTracker wrote.

    1. Move ``*_new.png`` files (auxiliary "newly discovered object"
       markers that SegTracker emits every sam_gap frames) into a
       sibling ``*_aux/`` directory. ProPainter naively counts every
       PNG in the mask dir, so leaving them in causes a 596-vs-655
       tensor shape mismatch right before inpainting starts.
    2. Rewrite each remaining mask to a binary mask of *object id == 1*
       only. SegTracker's default is "segment everything tracking" --
       it adds new object IDs (2, 3, ...) every sam_gap frames as SAM
       finds new things in the scene, and by the end of a typical clip
       99% of pixels are flagged as some tracked object. We only want
       the watermark the user clicked.
    3. Optionally dilate each cleaned mask outward by ``dilate_px``
       pixels (default 0). Watermarks with sharp circular outlines or
       anti-aliased glyphs leak ~1-3 px past whatever SAM segments;
       without dilation those ring pixels stay un-masked and survive
       inpainting as a faint halo. 4-8 px is the sweet spot for the
       common "logo with stroke" cases.

    Per-frame work runs in a ProcessPoolExecutor pool because each
    frame's cleanup is independent given the reference centroid; on a
    98K-mask, 8-core box this drops cleanup from ~60 min to ~7 min.

    Returns ``(count, union_bbox, n_with_content)`` where:
      * ``count`` -- number of mask PNGs in *mask_dir* after cleanup
      * ``union_bbox`` -- inclusive ``(x_min, y_min, x_max, y_max)`` of
        all non-empty masks, or ``None`` if every mask was empty.
        This is the raw (unpadded, unaligned) union; downstream
        ``_compute_bbox`` adds padding + alignment.
      * ``n_with_content`` -- number of frames whose mask was non-empty.
    """
    import json
    import numpy as np
    from PIL import Image
    from concurrent.futures import ProcessPoolExecutor

    # Normalise so callers can pass None for the default of 0.
    dilate_px = max(0, int(dilate_px)) if dilate_px is not None else 0

    # Sidecar fast-path: a previous successful cleanup writes a tiny
    # JSON summary next to the mask dir; if it's present and the PNG
    # count still matches, the entire cleanup pass (~5-10 min on a
    # 98K-mask resume) collapses to a single file read.
    sidecar = mask_dir.parent / f"{mask_dir.name}_cleanup.json"
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data.get("schema_version") == 1:
                cached_n = int(data["n_masks"])
                current_n = sum(1 for _ in mask_dir.glob("*.png"))
                cached_dilate = int(data.get("dilate_px", 0))
                if current_n == cached_n and current_n > 0 and cached_dilate == dilate_px:
                    bb = data.get("union_bbox")
                    bb = tuple(bb) if bb is not None else None
                    nwc = int(data.get("n_with_content", 0))
                    log.info(
                        "Cleanup sidecar matches (%d masks, dilate=%d); "
                        "SKIPPING the entire cleanup pass.",
                        cached_n, cached_dilate,
                    )
                    return cached_n, bb, nwc
                elif cached_dilate != dilate_px:
                    log.info(
                        "Cleanup sidecar present but dilate_px changed "
                        "(%d cached vs %d requested); re-running cleanup.",
                        cached_dilate, dilate_px,
                    )
                else:
                    log.info(
                        "Cleanup sidecar present but mask count drifted "
                        "(%d on disk vs %d cached); ignoring sidecar.",
                        current_n, cached_n,
                    )
        except (ValueError, KeyError, OSError) as e:
            log.warning("Could not read cleanup sidecar %s: %s; ignoring.",
                        sidecar, e)

    aux_masks = list(mask_dir.glob("*_new.png"))
    if aux_masks:
        aux_dir = mask_dir.parent / f"{mask_dir.name}_aux"
        aux_dir.mkdir(exist_ok=True)
        for f in aux_masks:
            f.replace(aux_dir / f.name)
        log.info("Moved %d auxiliary _new.png mask(s) to %s",
                 len(aux_masks), aux_dir)

    # Filter each frame to ONLY the connected component closest to
    # the watermark. DeAOT can drift over long videos -- object-id 1
    # occasionally leaks to unrelated regions during scene changes /
    # occlusion / memory bleed. On a real 90-min run this drift blew
    # the union bounding box from ~135x130 to 1386x1042 (69.6% of the
    # frame), tripping the "skip auto-crop if bbox > 60%" guard and
    # dropping the whole pipeline back to 1080p ProPainter (instant
    # OOM). We use the FIRST frame's mask centroid as the watermark
    # reference -- frame 0 is the SAM-clicked seed and is always
    # correct -- then on every subsequent frame keep the CC whose
    # centroid is closest to that reference. The watermark doesn't
    # teleport between frames, so this is robust.
    backend = "cv2"
    try:
        import cv2  # noqa: F401
    except ImportError:
        try:
            import scipy.ndimage  # noqa: F401
            backend = "scipy"
        except ImportError:
            backend = "none"
            log.warning("Neither cv2 nor scipy available -- skipping CC-based "
                        "mask cleanup; DeAOT drift artifacts may bloat the "
                        "auto-crop bbox")

    # Find the watermark reference centroid from frame 0. The mask
    # format is palette-PNG with values {0, 1} when fresh from DeAOT,
    # but the file may already be a binary 0/255 PNG if a previous
    # cleanup pass ran and we're being invoked again (e.g. via the
    # resume script). ``arr > 0`` handles both -- DO NOT compare to
    # the literal value 1, that bricks the masks on the second pass.
    ref_cy, ref_cx = None, None
    first_png = mask_dir / "00000.png"
    if backend != "none" and first_png.is_file():
        arr0 = np.array(Image.open(first_png))
        binary0 = (arr0 > 0).astype(np.uint8)
        if binary0.any():
            ys, xs = np.where(binary0)
            ref_cy = float(ys.mean())
            ref_cx = float(xs.mean())
            log.info("Watermark reference centroid (frame 0): (%.0f, %.0f)",
                     ref_cx, ref_cy)

    png_files = sorted(mask_dir.glob("*.png"))
    total = len(png_files)
    if total == 0:
        return 0, None, 0

    # Worker count: cap at 16 to avoid disk-thrash on long videos with
    # tens of thousands of small PNGs. chunksize=64 amortises the
    # spawn/pickle round-trip across a sensible batch.
    n_workers = max(2, min(16, (os.cpu_count() or 4)))
    args_iter = [(str(p), ref_cx, ref_cy, int(dilate_px)) for p in png_files]

    rewritten = 0
    max_obj_seen = 0
    drift_frames = 0
    n_with_content = 0
    union_bbox = None  # inclusive (x_min, y_min, x_max, y_max)
    # ~100 progress emissions across the whole run so the UI bar moves
    # smoothly without spamming PROGRESS lines.
    emit_step = max(1, total // 100)

    log.info("Cleaning %d masks with %d worker processes (backend=%s)...",
             total, n_workers, backend)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for max_obj, drift, frame_bbox in executor.map(
                _clean_one_mask_worker, args_iter, chunksize=64):
            max_obj_seen = max(max_obj_seen, max_obj)
            if drift:
                drift_frames += 1
            rewritten += 1
            if frame_bbox is not None:
                n_with_content += 1
                if union_bbox is None:
                    union_bbox = frame_bbox
                else:
                    union_bbox = (
                        min(union_bbox[0], frame_bbox[0]),
                        min(union_bbox[1], frame_bbox[1]),
                        max(union_bbox[2], frame_bbox[2]),
                        max(union_bbox[3], frame_bbox[3]),
                    )
            if (rewritten % emit_step) == 0 or rewritten == total:
                emit_progress("mask_cleanup", rewritten / total,
                              f"{rewritten}/{total}")

    log.info(
        "Rewrote %d mask PNGs to object-1-only binary; "
        "discarded DeAOT-drift artifacts in %d frame(s) "
        "(max tracked id observed across video: %d); "
        "%d/%d frames had non-empty mask",
        rewritten, drift_frames, max_obj_seen,
        n_with_content, rewritten,
    )

    # Persist the cleanup summary so a future re-run can short-circuit
    # straight to crop / ProPainter without rescanning 98K PNGs. Written
    # atomically (temp file + replace) so a kill mid-write can't leave
    # a half-formed sidecar that future runs would have to ignore.
    try:
        payload = {
            "schema_version": 1,
            "n_masks": rewritten,
            "union_bbox": list(union_bbox) if union_bbox is not None else None,
            "n_with_content": n_with_content,
            "dilate_px": dilate_px,
        }
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(sidecar)
    except OSError as e:
        log.warning("Failed to write cleanup sidecar %s: %s", sidecar, e)

    return rewritten, union_bbox, n_with_content


def _compute_bbox(
    mask_dir: Path,
    frame_w: int,
    frame_h: int,
    padding: int,
    align: int = 16,
    cached_union: Optional[Tuple[int, int, int, int]] = None,
    cached_n_with_content: Optional[int] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute the union bounding box of all nonzero pixels in *mask_dir*.

    Returns ``(x, y, w, h)`` clamped to the video frame and snapped so
    that ``w`` and ``h`` are multiples of *align* (ProPainter resizes
    inputs to multiples of 8 internally, but imageio's ffmpeg writer
    resizes outputs to multiples of 16 for H.264 macroblock compatibility;
    using 16-alignment prevents frame mismatch at the overlay boundary).

    Returns None if every mask is empty (no work to do).

    When *cached_union* is supplied (the inclusive ``(x_min, y_min,
    x_max, y_max)`` produced by the cleanup pass), the full mask-dir
    rescan is skipped -- the saved ~10-15 min on a 98K-mask job. Pass
    ``cached_union=None`` for the legacy behaviour (used by
    ``dynamic_resume_from_masks.py`` where cleanup didn't run in-process).
    """
    import numpy as np
    from PIL import Image

    if cached_union is not None:
        min_x, min_y, max_x, max_y = cached_union
        n_with_content = (
            cached_n_with_content
            if cached_n_with_content is not None
            else -1   # "not measured"; the log line just shows it as such
        )
    else:
        min_x = min_y = 10**9
        max_x = max_y = -1
        n_with_content = 0

        for png in sorted(mask_dir.glob("*.png")):
            arr = np.array(Image.open(png))
            ys, xs = np.where(arr > 0)
            if xs.size == 0:
                continue
            n_with_content += 1
            min_x = min(min_x, int(xs.min()))
            max_x = max(max_x, int(xs.max()))
            min_y = min(min_y, int(ys.min()))
            max_y = max(max_y, int(ys.max()))

        if n_with_content == 0:
            log.warning("Every mask is empty; nothing to inpaint.")
            return None

    # Pad
    x0 = max(0, min_x - padding)
    y0 = max(0, min_y - padding)
    x1 = min(frame_w, max_x + padding + 1)   # +1 because max is inclusive
    y1 = min(frame_h, max_y + padding + 1)

    # Snap dimensions to alignment, biased to keep the bbox inside the frame
    w = x1 - x0
    h = y1 - y0
    w_aligned = (w // align) * align
    h_aligned = (h // align) * align
    # If the snap shrank us, try to grow by shifting x0/y0 left/up
    if w_aligned < w and x0 + w_aligned + align <= frame_w:
        w_aligned += align
    if h_aligned < h and y0 + h_aligned + align <= frame_h:
        h_aligned += align

    log.info(
        "Mask bbox: union=(%d,%d)-(%d,%d), padded+aligned=%dx%d at (%d,%d) "
        "[%.1f%% of frame area, %s frame(s) had mask content]",
        min_x, min_y, max_x, max_y, w_aligned, h_aligned, x0, y0,
        100 * w_aligned * h_aligned / (frame_w * frame_h),
        "?" if n_with_content < 0 else str(n_with_content),
    )
    return x0, y0, w_aligned, h_aligned


def _crop_video(
    src: Path, dst: Path, x: int, y: int, w: int, h: int, ffmpeg: str,
) -> None:
    """Crop *src* to ``w x h`` at offset ``(x, y)``, write to *dst*.

    NVENC encode (~10-50x faster than libx264 on this hardware) with
    libx264 fallback if NVENC fails (e.g. driver issue / no NVIDIA GPU).
    """
    log.info("ffmpeg crop -> %s (%dx%d at %d,%d)", dst.name, w, h, x, y)
    # `-hwaccel cuda` enables NVDEC; ffmpeg silently falls back to software
    # decode for codecs CUDA can't handle, so adding this is safe.
    cmd_nvenc = [
        ffmpeg, "-y", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", str(src),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd_nvenc, creationflags=_NO_WINDOW)
    if rc == 0 and dst.is_file() and dst.stat().st_size > 1024:
        return
    log.warning("NVENC crop failed (rc=%d); falling back to libx264", rc)
    dst.unlink(missing_ok=True)
    cmd_x264 = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd_x264, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(f"ffmpeg crop failed (exit {rc})")


def _crop_one_mask_worker(args):
    """Per-frame mask crop; runs inside a ProcessPoolExecutor child.

    Module-level so it pickles cleanly across processes on Windows.
    """
    src, dst, x, y, w, h = args
    from PIL import Image
    img = Image.open(src).crop((x, y, x + w, y + h))
    # compress_level=1 mirrors _clean_one_mask_worker -- bit-identical
    # pixel data, ~2-3x faster save on binary masks.
    img.save(dst, compress_level=1)


def _crop_masks(
    src_dir: Path, dst_dir: Path, x: int, y: int, w: int, h: int,
) -> int:
    """Crop every PNG mask in *src_dir* and write to *dst_dir*.

    Parallelised across CPU cores -- a 98K-mask job drops from ~10 min
    serial to ~1-2 min on an 8-core box.
    """
    from concurrent.futures import ProcessPoolExecutor

    dst_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(src_dir.glob("*.png"))
    total = len(pngs)
    if total == 0:
        return 0

    args_iter = [
        (str(p), str(dst_dir / p.name), x, y, w, h) for p in pngs
    ]
    n_workers = max(2, min(16, (os.cpu_count() or 4)))

    n = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # consume() the iterator -- map raises if any worker raised.
        for _ in executor.map(_crop_one_mask_worker, args_iter, chunksize=64):
            n += 1
    log.info("Cropped %d masks -> %s", n, dst_dir)
    return n


def _build_overlay_filter(
    x: int, y: int, *, mask_input_idx: Optional[int], feather_px: int,
) -> str:
    """Build the ffmpeg ``-filter_complex`` graph for the overlay step.

    Two modes, both isolated here so the test suite can verify the
    exact graph without spawning ffmpeg.

    Hard-edge mode (``mask_input_idx is None``)
        Legacy behaviour: ``[0:v][1:v]overlay=x:y``. The entire bbox
        rectangle of inpainted output replaces the corresponding region
        of the base, producing a visible rectangular seam at the bbox
        edges because ProPainter slightly shifts colour/lighting on
        unmasked pixels inside the bbox.

    Alpha-feathered mode (``mask_input_idx`` is an integer >= 2)
        Use the mask sequence as the alpha channel: only pixels that
        the watermark actually covered get replaced, plus a
        ``feather_px``-wide ramp at the mask edges (gaussian-like via
        ``boxblur``) so the transition is sub-pixel and invisible.
    """
    if mask_input_idx is None:
        return f"[0:v][1:v]overlay={x}:{y}:eof_action=pass"

    # boxblur with luma_radius=feather_px and luma_power=1 gives a
    # soft 2*feather_px-wide ramp at the mask edge. format=gray
    # ensures the mask reads as luma regardless of how ffmpeg
    # decoded the PNG sequence (always grayscale in our case but
    # being explicit is cheap).
    return (
        f"[{mask_input_idx}:v]format=gray,boxblur={feather_px}:1[mfeath];"
        f"[1:v][mfeath]alphamerge[crop_a];"
        f"[0:v][crop_a]overlay={x}:{y}:eof_action=pass"
    )


def _probe_fps(video: Path, ffprobe: str) -> Optional[float]:
    """Average frame rate of *video* via ffprobe, or None on failure."""
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video)],
            capture_output=True, text=True, timeout=15,
            creationflags=_NO_WINDOW,
        )
        if proc.returncode != 0:
            return None
        frac = proc.stdout.strip()
        if "/" in frac:
            num, den = frac.split("/", 1)
            num_i = int(num)
            den_i = int(den)
            if den_i == 0:
                return None
            return num_i / den_i
        return float(frac) if frac else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _overlay(
    base: Path, crop: Path, out: Path, x: int, y: int, ffmpeg: str,
    *,
    mask_seq_dir: Optional[Path] = None,
    feather_px: int = 5,
    fps: Optional[float] = None,
) -> None:
    """Overlay *crop* on *base* at ``(x, y)`` -> *out*, preserving audio.

    When ``mask_seq_dir`` is provided (default in the main pipeline),
    the PNG mask sequence inside it is fed as the alpha channel for
    *crop* and a ``feather_px``-radius boxblur smooths the mask edge,
    so only the actual watermark pixels (with a soft ramp) replace the
    base. This eliminates the rectangular bbox seam that the hard-edge
    overlay used to leave at the auto-crop boundary.

    When ``mask_seq_dir`` is None we fall back to the hard-edge paste --
    kept for backward compatibility and as an escape hatch if a future
    issue with the alpha path appears (set the kwarg to None to
    bisect).

    Critical: NO ``-shortest`` flag. If the inpaint stream is shorter
    than the source (e.g., DeAOT processed fewer frames than the source
    actually has because the reader EOF'd early), we still want the
    output to span the full source duration -- the post-inpaint frames
    just pass through unmodified. ``-shortest`` would truncate the
    output to the shorter input, which previously cost a user the back
    22 minutes of a 27-minute video.
    """
    use_mask = mask_seq_dir is not None
    if use_mask:
        if not mask_seq_dir.is_dir():
            raise FileNotFoundError(
                f"mask_seq_dir does not exist: {mask_seq_dir}"
            )
        first_mask = mask_seq_dir / "00000.png"
        if not first_mask.is_file():
            raise FileNotFoundError(
                f"mask_seq_dir is empty (no 00000.png): {mask_seq_dir}"
            )
        if fps is None or fps <= 0:
            raise ValueError(
                "fps must be provided (and positive) when using "
                "mask_seq_dir"
            )

    log.info(
        "ffmpeg overlay %s onto %s at (%d,%d) -> %s (alpha=%s)",
        crop.name, base.name, x, y, out.name,
        f"feather={feather_px}" if use_mask else "hard-edge",
    )

    filter_complex = _build_overlay_filter(
        x, y,
        mask_input_idx=2 if use_mask else None,
        feather_px=feather_px,
    )

    inputs = ["-i", str(base), "-i", str(crop)]
    if use_mask:
        inputs += [
            "-framerate", f"{fps:.6f}",
            "-i", str(mask_seq_dir / "%05d.png"),
        ]

    def _build_cmd(codec_args):
        return [
            ffmpeg, "-y", "-loglevel", "error",
            *inputs,
            "-filter_complex", filter_complex,
            *codec_args,
            "-c:a", "copy",
            str(out),
        ]

    cmd_nvenc = _build_cmd([
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
    ])
    rc = subprocess.call(cmd_nvenc, creationflags=_NO_WINDOW)
    if rc == 0 and out.is_file() and out.stat().st_size > 1024:
        return
    log.warning("NVENC overlay failed (rc=%d); falling back to libx264", rc)
    out.unlink(missing_ok=True)
    cmd_x264 = _build_cmd([
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
    ])
    rc = subprocess.call(cmd_x264, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(f"ffmpeg overlay failed (exit {rc})")


# --------------------------------------------------------------------------- #
# ProPainter invocation
# --------------------------------------------------------------------------- #

# Hard cap on the number of frames a single ProPainter invocation handles.
# ProPainter loads ALL frames into RAM up-front (its RAFT pass needs the
# whole sequence) so even with auto-crop a long video OOMs immediately.
# 800 frames at ~400x400 fp16 fits comfortably on a 12 GB consumer GPU
# (empirically we ran 596 frames at 224x224 with VRAM headroom). When
# the clip is longer we slice with ffmpeg, ProPainter each PADDED slice
# independently, then concat the (un-padded) inpainted slices back. See
# PROPAINTER_CHUNK_PAD for why padding is mandatory: without it every
# chunk boundary is a visible colour/lighting seam.
PROPAINTER_CHUNK_FRAMES = 800
# Frames of overlap on EACH side between adjacent chunks. The padded
# frames are fed to ProPainter purely as temporal context and discarded
# before concat -- without this, every PROPAINTER_CHUNK_FRAMES boundary
# was a visible seam (colour/lighting jump) because ProPainter has no
# temporal context that crosses chunk boundaries. 80 frames at 24fps is
# ~3.3 sec of context; smaller risks the seam reappearing, larger costs
# more compute (~ 2*pad/chunk_frames extra GPU time per middle chunk).
PROPAINTER_CHUNK_PAD = 80


def _run_propainter(
    video: Path,
    mask_dir: Path,
    output_dir: Path,
    *,
    fp16: bool,
    subvideo_length: int,
    ffmpeg: Optional[str] = None,
    chunk_frames: int = PROPAINTER_CHUNK_FRAMES,
) -> Path:
    """Run ProPainter; return the path to the produced inpaint_out.mp4.

    Automatically temporally-chunks long videos. ``ffmpeg`` must be
    supplied for chunking to be possible; if omitted we always run in
    single-shot mode (which OOMs on long clips).
    """
    n_masks = sum(1 for _ in mask_dir.glob("*.png"))
    if n_masks <= chunk_frames or ffmpeg is None:
        return _run_propainter_single(
            video, mask_dir, output_dir,
            fp16=fp16, subvideo_length=subvideo_length,
        )

    log.info(
        "Video has %d frames > %d cap -- ProPainter will run in %d "
        "temporal chunks (each %d frames or fewer).",
        n_masks, chunk_frames,
        (n_masks + chunk_frames - 1) // chunk_frames, chunk_frames,
    )
    return _run_propainter_chunked(
        video, mask_dir, output_dir,
        fp16=fp16, subvideo_length=subvideo_length,
        ffmpeg=ffmpeg, chunk_frames=chunk_frames, total_frames=n_masks,
    )


def _run_propainter_single(
    video: Path,
    mask_dir: Path,
    output_dir: Path,
    *,
    fp16: bool,
    subvideo_length: int,
) -> Path:
    """Single ProPainter invocation on the full video + mask sequence."""
    propainter_dir = WM_PATH / "ProPainter"
    cmd = [
        sys.executable,
        str(propainter_dir / "inference_propainter.py"),
        "--video", str(video),
        "--mask", str(mask_dir),
        "--output", str(output_dir),
        "--subvideo_length", str(subvideo_length),
    ]
    if fp16:
        cmd.append("--fp16")
    log.info("Running ProPainter: %s", " ".join(cmd))

    prev_cwd = os.getcwd()
    os.chdir(propainter_dir)
    try:
        rc = subprocess.call(cmd, creationflags=_NO_WINDOW)
    finally:
        os.chdir(prev_cwd)
    if rc != 0:
        raise RuntimeError(f"ProPainter exited with code {rc}")

    produced = output_dir / video.stem / "inpaint_out.mp4"
    if not produced.is_file():
        raise FileNotFoundError(
            f"ProPainter completed but expected output missing: {produced}"
        )
    return produced


def _reencode_source_clean(
    src: Path, dst: Path, ffmpeg: str,
) -> None:
    """Re-encode *src* into a robust intermediate at *dst*.

    Some user videos -- even nominally-valid H.264 ones -- have decode
    quirks (corrupt packets, timestamp discontinuities, B-frame
    references to dropped I-frames) that don't surface when ffmpeg
    runs in standalone mode, but DO surface as early EOF when feeding
    raw bgr24 through a slow Python consumer pipe. The decoder gives
    up after a few thousand frames and our pipeline silently truncates.

    Fix: BEFORE running DeAOT, transcode the source into a clean
    intermediate that decodes end-to-end without back-pressure issues.
    Costs a few minutes up front but eliminates the silent truncation.

    Tries NVENC first (fast hardware encode on NVIDIA GPUs), falls back
    to libx264 ultrafast if NVENC is unavailable or fails. Quality
    setting is high (CQ/CRF 16-18) so the intermediate is visually
    indistinguishable from the source -- downstream DeAOT + ProPainter
    operate on this, and the final overlay also uses this as the base
    so user-perceived quality matches the intermediate.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # NVENC encode + NVDEC decode (`-hwaccel cuda`). The decode side falls
    # back to CPU automatically for codecs CUDA doesn't support, so adding
    # the flag is safe even on inputs that aren't H.264.
    cmd_nvenc = [
        ffmpeg, "-y", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-err_detect", "ignore_err",
        "-fflags", "+discardcorrupt",
        "-i", str(src),
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", "18",
        "-c:a", "copy",
        str(dst),
    ]
    log.info("Pre-encoding source to robust intermediate (NVENC)...")
    rc = subprocess.call(cmd_nvenc, creationflags=_NO_WINDOW)
    if rc == 0 and dst.is_file() and dst.stat().st_size > 1024:
        log.info("NVENC pre-encode succeeded -> %s (%.1f MB)",
                 dst.name, dst.stat().st_size / 1e6)
        return

    log.warning("NVENC failed (rc=%d), falling back to libx264 ultrafast...", rc)
    # Clean up partial NVENC output
    dst.unlink(missing_ok=True)
    cmd_x264 = [
        ffmpeg, "-y", "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-fflags", "+discardcorrupt",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "16",
        "-c:a", "copy",
        str(dst),
    ]
    rc = subprocess.call(cmd_x264, creationflags=_NO_WINDOW)
    if rc != 0 or not dst.is_file() or dst.stat().st_size < 1024:
        raise RuntimeError(
            f"Source pre-encode failed (libx264 rc={rc}, "
            f"output exists={dst.is_file()}). Cannot continue."
        )
    log.info("libx264 pre-encode succeeded -> %s (%.1f MB)",
             dst.name, dst.stat().st_size / 1e6)


def _ffmpeg_slice_video(
    src: Path, dst: Path, start_frame: int, n_frames: int, ffmpeg: str,
) -> None:
    """Extract ``[start_frame, start_frame + n_frames)`` from *src* into *dst*.

    Uses ``select=between(n,...)`` for exact frame alignment with the
    cv2/ProPainter view of the video, and ``setpts=PTS-STARTPTS`` to
    rebase timestamps so the chunk plays from t=0.
    """
    last = start_frame + n_frames - 1
    vf = (f"select=between(n\\,{start_frame}\\,{last}),"
          f"setpts=PTS-STARTPTS")
    # NVENC first -- consistent with the rest of the pipeline
    # (reencode, crop, overlay all NVENC). On a 62-chunk run the
    # cumulative slice cost drops from ~2 min libx264 -> ~10 s NVENC.
    cmd_nvenc = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-vsync", "vfr",
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd_nvenc, creationflags=_NO_WINDOW)
    if rc == 0 and dst.is_file() and dst.stat().st_size > 1024:
        return
    log.warning("NVENC slice failed (rc=%d); falling back to libx264", rc)
    dst.unlink(missing_ok=True)
    cmd_x264 = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-vsync", "vfr",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd_x264, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg slice failed (exit {rc}) for frames {start_frame}..{last}"
        )


def _compute_chunk_geometry(
    total_frames: int, chunk_frames: int, pad: int,
) -> list:
    """Plan padded ProPainter chunks.

    Returns a list of ``(chunk_start, chunk_n, keep_offset, keep_n)`` tuples
    describing how to slice a *total_frames*-long sequence into
    overlapping ProPainter chunks, where:

    - ``[chunk_start, chunk_start + chunk_n)`` is the frame range to
      feed ProPainter (with up to *pad* padding frames on each side, used
      purely as temporal context).
    - ``[chunk_start + keep_offset, chunk_start + keep_offset + keep_n)``
      is the slice of the inpainted output that should be retained;
      the rest is padding to be discarded by the caller before concat.

    The keep ranges of consecutive tuples are exactly contiguous and
    together cover ``[0, total_frames)`` with no gaps and no overlap, so
    concatenating the kept slices reproduces the full video length.

    Edge cases:
    - First chunk has ``keep_offset == 0`` (no left pad to discard).
    - Last chunk has ``keep_n == total_frames - keep_start`` and no
      right padding to discard.
    - If ``total_frames <= chunk_frames``, returns a single tuple with
      no padding (single-shot, fully covered by ProPainter's own
      subvideo windowing -- no seams to fix).
    """
    if total_frames <= 0:
        return []
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")

    n_chunks = (total_frames + chunk_frames - 1) // chunk_frames
    if n_chunks == 1:
        return [(0, total_frames, 0, total_frames)]

    plan = []
    for idx in range(n_chunks):
        keep_start = idx * chunk_frames
        keep_end = min(total_frames, keep_start + chunk_frames)

        pad_left = min(keep_start, pad)             # 0 for first chunk
        pad_right = min(total_frames - keep_end, pad)  # 0 for last chunk

        chunk_start = keep_start - pad_left
        chunk_end = keep_end + pad_right
        chunk_n = chunk_end - chunk_start
        keep_offset = pad_left
        keep_n = keep_end - keep_start

        plan.append((chunk_start, chunk_n, keep_offset, keep_n))
    return plan


def _ffmpeg_trim_frames(
    src: Path, dst: Path, start_idx: int, n_frames: int, ffmpeg: str,
) -> None:
    """Re-encode *src* keeping only ``[start_idx, start_idx + n_frames)``.

    Used to drop the context-padding frames from a ProPainter chunk
    before concat. Uses NVENC for speed with libx264 fallback. Stream
    copy with ``-ss`` is unsafe here because we need exact frame-level
    alignment between adjacent chunks; a single re-encode at CQ 18 is
    visually transparent.
    """
    last = start_idx + n_frames - 1
    vf = (f"select=between(n\\,{start_idx}\\,{last}),"
          f"setpts=PTS-STARTPTS")
    cmd_nvenc = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-vsync", "vfr",
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd_nvenc, creationflags=_NO_WINDOW)
    if rc == 0 and dst.is_file() and dst.stat().st_size > 1024:
        return
    log.warning("NVENC trim failed (rc=%d); falling back to libx264", rc)
    dst.unlink(missing_ok=True)
    cmd_x264 = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-vsync", "vfr",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd_x264, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg trim failed (exit {rc}) for frames {start_idx}..{last}"
        )


def _ffmpeg_concat(parts: list, dst: Path, ffmpeg: str) -> None:
    """Concatenate *parts* into *dst* losslessly with the concat demuxer."""
    list_file = dst.parent / "_concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            # Concat demuxer wants forward slashes + single-quote-escaped paths
            esc = str(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{esc}'\n")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(dst),
    ]
    rc = subprocess.call(cmd, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {rc}); list at {list_file}"
        )


def _run_propainter_chunked(
    video: Path,
    mask_dir: Path,
    output_dir: Path,
    *,
    fp16: bool,
    subvideo_length: int,
    ffmpeg: str,
    chunk_frames: int,
    total_frames: int,
    pad: int = PROPAINTER_CHUNK_PAD,
) -> Path:
    """Slice -> ProPainter each chunk (with context padding) -> trim ->
    ffmpeg concat the trimmed outputs.

    Each chunk is fed to ProPainter with up to *pad* extra frames on
    each side as temporal context. The padded frames are then dropped
    via ffmpeg before concat so the seam between chunks lands inside a
    region both neighbours saw, eliminating the colour/lighting jumps
    that were visible at every chunk boundary in the un-padded version.

    Per-chunk overhead vs un-padded: ``2 * pad / chunk_frames`` (e.g.
    ~20% at pad=80 / chunk_frames=800 for the middle chunks; first and
    last chunks have only single-side padding).
    """
    import json as _json
    import shutil as _shutil

    chunks_dir = output_dir / "_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    plan = _compute_chunk_geometry(total_frames, chunk_frames, pad)
    n_chunks = len(plan)
    chunk_outputs = []

    n_skipped = 0
    for idx, (chunk_start, chunk_n, keep_offset, keep_n) in enumerate(plan):
        emit_progress("propainter", idx / max(n_chunks, 1),
                      f"chunk_{idx + 1}/{n_chunks}")

        cdir = chunks_dir / f"c{idx:04d}"
        cdir.mkdir(exist_ok=True)
        cvideo = cdir / "in.mp4"
        cmask = cdir / "masks"
        cout = cdir / "out"
        trimmed = cdir / "trimmed.mp4"
        geom_path = cdir / "geometry.json"
        expected_geom = {
            "version": 2,
            "chunk_start": chunk_start,
            "chunk_n": chunk_n,
            "keep_offset": keep_offset,
            "keep_n": keep_n,
            "pad": pad,
        }

        # Resume: only reuse if BOTH the trimmed mp4 exists AND the
        # sidecar geometry exactly matches what we'd produce now. This
        # invalidates any chunks computed by an older code path (no
        # padding, or different pad value), since they have either no
        # trimmed.mp4 or a mismatching geometry.json.
        if trimmed.is_file() and trimmed.stat().st_size > 1024 \
                and geom_path.is_file():
            try:
                with open(geom_path, "r", encoding="utf-8") as fh:
                    cached = _json.load(fh)
            except Exception:  # noqa: BLE001
                cached = None
            if cached == expected_geom:
                log.info(
                    "Chunk %d/%d: frames %d..%d (keep %d) -- REUSING "
                    "trimmed output (%.1f MB)",
                    idx + 1, n_chunks, chunk_start, chunk_start + chunk_n - 1,
                    keep_n, trimmed.stat().st_size / 1e6,
                )
                chunk_outputs.append(trimmed)
                n_skipped += 1
                continue
            else:
                log.info(
                    "Chunk %d/%d: geometry mismatch (cached=%r, expected=%r)"
                    " -- recomputing",
                    idx + 1, n_chunks, cached, expected_geom,
                )

        log.info(
            "Chunk %d/%d: feeding ProPainter frames %d..%d "
            "(%d frames, keep %d offset %d)",
            idx + 1, n_chunks,
            chunk_start, chunk_start + chunk_n - 1,
            chunk_n, keep_n, keep_offset,
        )

        cmask.mkdir(exist_ok=True)
        cout.mkdir(exist_ok=True)

        # ffmpeg slice the source video to this chunk's PADDED frame range
        _ffmpeg_slice_video(video, cvideo, chunk_start, chunk_n, ffmpeg)

        # Copy the matching mask subset (renamed to start from 00000)
        for i in range(chunk_start, chunk_start + chunk_n):
            src_mask = mask_dir / f"{i:05d}.png"
            if not src_mask.is_file():
                raise FileNotFoundError(
                    f"Expected mask missing: {src_mask}"
                )
            dst_mask = cmask / f"{i - chunk_start:05d}.png"
            _shutil.copy(src_mask, dst_mask)

        # ProPainter on the padded slice -- single-shot since it's small
        produced = _run_propainter_single(
            cvideo, cmask, cout,
            fp16=fp16, subvideo_length=subvideo_length,
        )

        # Trim the padding off so concat boundaries land in the region
        # both neighbours saw (no seam). For first/last chunks with no
        # padding to drop, skip the re-encode and use the raw output.
        if keep_offset == 0 and keep_n == chunk_n:
            _shutil.copy(produced, trimmed)
        else:
            _ffmpeg_trim_frames(produced, trimmed, keep_offset, keep_n,
                                ffmpeg)

        with open(geom_path, "w", encoding="utf-8") as fh:
            _json.dump(expected_geom, fh)

        chunk_outputs.append(trimmed)

    if n_skipped:
        log.info("ProPainter resume: reused %d/%d existing chunk outputs",
                 n_skipped, n_chunks)

    # Concat the chunk outputs into one final video at the conventional
    # path our caller expects (output_dir/<video_stem>/inpaint_out.mp4).
    final_dir = output_dir / video.stem
    final_dir.mkdir(parents=True, exist_ok=True)
    final = final_dir / "inpaint_out.mp4"
    _ffmpeg_concat(chunk_outputs, final, ffmpeg)
    log.info("Concatenated %d chunks -> %s", len(chunk_outputs), final)
    return final


# --------------------------------------------------------------------------- #
# DeAOT resume detection
# --------------------------------------------------------------------------- #

def _resume_threshold_met(
    n_existing: int, n_expected: int, threshold: float = 0.95,
) -> bool:
    """Pure helper -- decide whether *n_existing* masks are "enough" to
    treat as a successful prior DeAOT run.

    A 5% shortfall is tolerated because ffprobe's ``nb_frames`` can be
    slightly off for variable-fps containers / streams without an
    accurate header. Anything below that is treated as an interrupted
    prior run; we'll re-run DeAOT from scratch rather than feed a
    truncated mask set into ProPainter (which would silently produce a
    shorter output video than the source).
    """
    if n_existing <= 0 or n_expected <= 0:
        return False
    return n_existing >= n_expected * threshold


def _try_resume_from_workspace_masks(
    decode_source: Path,
    workspace: Optional[Path],
    ffmpeg: str,
    ffprobe: Optional[str],
):
    """Detect a substantially-complete mask dump in *workspace* so we
    can skip the SAM+DeAOT stage entirely on a re-run.

    Background: SegTracker hard-codes its output to
    ``<wm_path>/output/<stem>/<stem>_masks/``, which the main worker
    then moves into ``<workspace>/<stem>/<stem>_masks/`` once DeAOT
    finishes. If a previous run completed DeAOT but crashed in a
    downstream stage (ProPainter, overlay, ...), the full mask set is
    still on disk in the workspace -- but the next run wipes the
    SegTracker staging dir and re-runs DeAOT from frame 0, costing
    hours on long videos. This helper closes that gap.

    Returns ``(mask_dir, out_root, frame_w, frame_h)`` on resume, or
    ``None`` if no usable mask dump exists / the probe failed.
    """
    if workspace is None:
        return None
    expected_out = workspace / decode_source.stem
    expected_masks = expected_out / f"{decode_source.stem}_masks"
    if not expected_masks.is_dir():
        return None
    n_existing = sum(1 for _ in expected_masks.glob("*.png"))
    if n_existing == 0:
        return None

    # Probe the intermediate for expected frame count + dimensions.
    try:
        with FfmpegVideoReader(decode_source, ffmpeg, ffprobe=ffprobe) as r:
            frame_w, frame_h = r.width, r.height
            n_expected = r.n_frames or 0
    except Exception as e:  # noqa: BLE001
        log.warning("Resume probe failed (%s); will re-run DeAOT", e)
        return None

    if n_expected == 0:
        log.warning(
            "ffprobe couldn't determine frame count for %s; cannot "
            "safely resume from %d existing mask(s) -- re-running DeAOT.",
            decode_source.name, n_existing,
        )
        return None

    if not _resume_threshold_met(n_existing, n_expected):
        log.info(
            "Existing mask set has %d PNGs vs ~%d expected frames -- "
            "looks like an interrupted DeAOT, re-running from scratch.",
            n_existing, n_expected,
        )
        return None

    log.info(
        "Resume: found %d masks in workspace (%d expected); "
        "SKIPPING SAM+DeAOT and feeding existing masks straight into "
        "cleanup.", n_existing, n_expected,
    )
    return expected_masks, expected_out, frame_w, frame_h


# --------------------------------------------------------------------------- #
# DeAOT tracking
# --------------------------------------------------------------------------- #

def _propagate_masks_streaming(tracker, video_path, mask_dir, ffmpeg,
                               ffprobe=None, frame_total_hint=0):
    """Run DeAOT through *video_path*, writing one palette mask PNG per frame.

    Reads frames via :class:`FfmpegVideoReader` (a hardened
    ffmpeg-subprocess wrapper with a per-frame watchdog) -- the
    in-process cv2 path historically hung indefinitely on certain
    malformed H.264 frames in real user videos, which made the worker a
    zombie after 3 hours of no progress.

    Replaces watermark_remover's ``tracking_objects_in_video`` which
    additionally did a second pass over the video to render a debug
    overlay video (~50 GB of unused PNGs on a 90-min 1080p clip). Cutting
    that pass roughly halves DeAOT wall-time with zero quality impact.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from aot_tracker import _palette  # type: ignore

    mask_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    with FfmpegVideoReader(video_path, ffmpeg, ffprobe=ffprobe) as cap:
        with torch.cuda.amp.autocast():
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if frame_idx == 0:
                    pred_mask = tracker.first_frame_mask
                else:
                    # sam_gap is set to 10**9 by the caller, so the
                    # periodic SAM re-detection branch is intentionally
                    # never hit -- pure DeAOT propagation only.
                    pred_mask = tracker.track(frame_rgb, update_memory=True)

                # Save as palette PNG (binary-mask format ProPainter expects).
                save_mask = Image.fromarray(pred_mask.astype(np.uint8))
                save_mask = save_mask.convert(mode="P")
                save_mask.putpalette(_palette)
                save_mask.save(mask_dir / f"{frame_idx:05d}.png")

                # Parent's stdout parser turns these into PROGRESS deaot updates.
                obj_num = tracker.get_obj_num() if hasattr(tracker, "get_obj_num") else 1
                print(f"processed frame {frame_idx}, obj_num {obj_num}", flush=True)

                frame_idx += 1

                # Periodic VRAM cleanup. The upstream code did this every
                # frame which is wasteful CPU. 50-frame stride keeps
                # memory pressure low without dominating wall-time.
                if frame_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    log.info("DeAOT propagation finished: %d frames", frame_idx)
    return frame_idx


def _run_sam_and_deaot(
    video: Path,
    clicks: list,
    aot_model: str,
    ffmpeg: str,
    ffprobe: Optional[str] = None,
):
    """Run SAM on the first frame then DeAOT through the video.

    Returns the path to the mask directory and (W, H) of the source
    video. The tracker is *deleted* before this function returns and
    cuda cache is emptied -- callers don't get a tracker reference
    because keeping it alive would steal GPU memory from ProPainter
    (which spawns next and shows up as a confusing
    ``CUDNN_STATUS_MAPPING_ERROR`` at the first optical-flow call).
    """
    import cv2
    import numpy as np
    import torch
    from model_args import segtracker_args, sam_args, aot_args  # type: ignore
    from SegTracker import SegTracker  # type: ignore
    from seg_track_anything import aot_model2ckpt  # type: ignore

    if aot_model not in aot_model2ckpt:
        raise ValueError(
            f"Unknown AOT model {aot_model!r}; available: {list(aot_model2ckpt)}"
        )
    aot_args = dict(aot_args)
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]

    # Disable SegTracker's periodic "segment everything" re-detection.
    # Defaults to sam_gap=10 -- every 10 frames it re-runs SAM in
    # automatic-everything mode and merges discovered objects via
    # ``pred_mask = track_mask + new_obj_mask`` (integer addition,
    # see watermark_remover/seg_track_anything.py). For single-watermark
    # tracking this is actively harmful: the addition can corrupt the
    # object-1 ID at the sam_gap boundary frames, causing scattered
    # spurious "object 1" pixels far from the actual watermark
    # (we observed obj=1 mask spanning 65% of the frame on those
    # boundaries vs. ~0.7% on clean frames). Setting sam_gap larger
    # than the longest expected video makes DeAOT track *only* the
    # objects registered at frame 0 -- exactly what we want.
    segtracker_args = dict(segtracker_args)
    segtracker_args["sam_gap"] = 10**9

    emit_progress("loading", 0.0)
    log.info("Loading SAM + DeAOT (model=%s, sam_gap disabled)...", aot_model)
    tracker = SegTracker(segtracker_args, sam_args, aot_args)
    tracker.restart_tracker()
    emit_progress("loading", 1.0)

    log.info("Reading first frame (via ffmpeg subprocess)...")
    with FfmpegVideoReader(video, ffmpeg, ffprobe=ffprobe) as cap:
        frame_w, frame_h = cap.width, cap.height
        ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise IOError(f"Cannot read first frame of {video}")
    first_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    coords = np.array([[c[0], c[1]] for c in clicks])
    modes = np.array([c[2] for c in clicks])
    emit_progress("sam", 0.0, f"{len(clicks)}_clicks")
    log.info("Running SAM with %d click(s): %s", len(clicks), clicks)
    predicted_mask, _ = tracker.seg_acc_click(
        origin_frame=first_frame,
        coords=coords,
        modes=modes,
        multimask="True",
    )
    log.info("Initial mask: %d positive pixels of %d total",
             int((predicted_mask > 0).sum()), predicted_mask.size)
    emit_progress("sam", 1.0)

    with torch.cuda.amp.autocast():
        tracker.restart_tracker()
        tracker.add_reference(first_frame, predicted_mask, 0)
        tracker.first_frame_mask = predicted_mask

    # Estimate frame count for the per-frame DeAOT progress bar via
    # ffprobe (instant). The progress bar tolerates a wrong total
    # because the worker also emits PROGRESS deaot 1.0 at the end,
    # snapping the bar to 100% regardless. Counting by decoding every
    # frame was the original approach but takes HOURS on long videos.
    try:
        _probe_reader = FfmpegVideoReader(video, ffmpeg, ffprobe=ffprobe)
        _n_frames = _probe_reader.n_frames or 0
        # No need to .close() -- we never spawned the streaming subprocess
    except Exception:  # noqa: BLE001
        _n_frames = 0
    log.info("Video frame count (from ffprobe): %d", _n_frames)
    emit_progress("deaot", 0.0, f"total={_n_frames}" if _n_frames else "")
    log.info("Propagating mask through video with DeAOT (this is slow)...")

    # Use our streaming variant (no Pass 2 + ffmpeg-subprocess reader)
    # -- writes the same mask PNGs to the same path
    # tracking_objects_in_video would have, then returns without the
    # visualisation pass that doubled the wall-time and wrote ~50 GB of
    # unused PNGs on long clips. The ffmpeg reader can't hang on bad
    # frames the way cv2.VideoCapture does.
    video_stem = video.stem
    out_root = WM_PATH / "output" / video_stem
    mask_dir = out_root / f"{video_stem}_masks"
    # Wipe any stale state from an interrupted prior run before DeAOT
    # starts writing. Otherwise: prior run crashed at frame 50000 (left
    # masks 00000..49999.png in here, not yet moved to workspace), this
    # run crashes at frame 30000, masks 30000..49999 are stale tail
    # from prior run. Downstream _compute_bbox globs *.png and would
    # union the stale-tail mask positions into the bbox -- typically
    # bloating it past the 60% auto-crop guard and forcing full-frame
    # ProPainter (instant OOM on consumer GPUs).
    if out_root.exists():
        log.info("Wiping stale DeAOT staging dir before fresh run: %s",
                 out_root)
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    _propagate_masks_streaming(
        tracker, video, mask_dir,
        ffmpeg=ffmpeg, ffprobe=ffprobe,
    )
    emit_progress("deaot", 1.0)

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"DeAOT did not produce mask directory {mask_dir}"
        )

    # Free GPU memory before returning -- ProPainter runs next and needs
    # the whole 12 GB on consumer cards.
    log.info("Releasing GPU memory before ProPainter spawn...")
    del tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    return mask_dir, out_root, frame_w, frame_h


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        log.error("No JSON payload on stdin")
        return 2
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Invalid JSON: %s", e)
        return 2

    video = Path(cfg["video"]).resolve()
    output = Path(cfg["output"]).resolve()
    clicks = cfg["clicks"]
    aot_model = cfg.get("aot_model", "r50_deaotl")
    subvideo_length = int(cfg.get("subvideo_length", 80))
    fp16 = bool(cfg.get("fp16", True))
    auto_crop = bool(cfg.get("auto_crop", True))
    crop_padding = int(cfg.get("crop_padding", 96))
    # Pixels to grow each cleaned mask outward before saving. 0 = legacy
    # behaviour (mask matches what SAM/DeAOT segmented exactly). 4-8 px
    # eliminates the 1-3 px halo / outline ring that crisp-edged
    # watermarks (rounded logos, stroked glyphs) leave behind in the
    # inpaint output. >12 px bloats the auto-crop bbox without measurable
    # quality gain.
    mask_dilate = max(0, int(cfg.get("mask_dilate", 12)))
    workspace_str = cfg.get("workspace")
    workspace = Path(workspace_str).resolve() if workspace_str else None

    if not video.is_file():
        log.error("Input video not found: %s", video)
        return 3
    if not clicks:
        log.error("No clicks provided")
        return 3

    ffmpeg = _find_ffmpeg()
    ffprobe = _find_ffprobe()

    # Stage 0: pre-encode the source into a robust intermediate.
    # Without this, slow Python consumption of ffmpeg's raw bgr24 pipe
    # causes ffmpeg to give up after a few thousand to ~18k frames on
    # certain source videos (we observed this on a real user 27-min
    # H.264 clip). The intermediate is decoded clean end-to-end by
    # subsequent ffmpeg subprocesses, eliminating the back-pressure
    # truncation. All downstream stages -- DeAOT input, ffmpeg crop,
    # ffmpeg overlay base -- use the intermediate, not the original.
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        intermediate = workspace / "_source_clean.mp4"
    else:
        intermediate = video.with_suffix(".clean.mp4")
    # Re-use the intermediate IF AND ONLY IF it parses end-to-end --
    # a previous run that crashed mid-encode left a partial file with
    # no moov atom that fooled the size-only check and caused v4 to
    # fail with "Invalid data found when processing input".
    intermediate_ok = False
    if intermediate.exists() and intermediate.stat().st_size > 1024:
        # Trust the file only if ffprobe can read its container metadata.
        if ffprobe:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries",
                 "stream=width,height", "-of", "csv", str(intermediate)],
                capture_output=True, text=True, timeout=15,
                creationflags=_NO_WINDOW,
            )
            intermediate_ok = probe.returncode == 0 and "stream" in probe.stdout
        else:
            intermediate_ok = True  # best effort if no ffprobe
    if intermediate_ok:
        log.info("Re-using existing intermediate %s (%.1f MB)",
                 intermediate.name, intermediate.stat().st_size / 1e6)
    else:
        intermediate.unlink(missing_ok=True)
        emit_progress("reencode", 0.0, str(video.stat().st_size))
        _reencode_source_clean(video, intermediate, ffmpeg)
        emit_progress("reencode", 1.0)
    decode_source = intermediate

    # Stage 1+2: SAM first-frame + DeAOT propagation. SegTracker hard-
    # codes its mask output to <wm_path>/output/<stem>; we relocate
    # the whole thing immediately afterwards so subsequent stages and
    # any leftover files live inside VSR Pro's tree, not in the
    # sibling watermark_remover repo.
    #
    # First, see if a prior run already produced a complete mask set in
    # the workspace. If so, skip the (multi-hour on long videos) DeAOT
    # propagation and re-use those masks. The cleanup pass + fast-path
    # rewrite-skip will then make a second cleanup essentially free.
    resumed = _try_resume_from_workspace_masks(
        decode_source, workspace, ffmpeg, ffprobe,
    )
    if resumed is not None:
        mask_dir, out_root, frame_w, frame_h = resumed
        # Mirror the progress emissions the SAM/DeAOT path would have
        # made so the UI doesn't appear stuck at "loading 0%".
        emit_progress("loading", 1.0, "resumed")
        emit_progress("sam", 1.0, "resumed")
        emit_progress("deaot", 1.0, "resumed")
    else:
        mask_dir, out_root, frame_w, frame_h = _run_sam_and_deaot(
            video=decode_source, clicks=clicks, aot_model=aot_model,
            ffmpeg=ffmpeg, ffprobe=ffprobe,
        )

        if workspace is not None:
            workspace.mkdir(parents=True, exist_ok=True)
            # Move (or copy + delete) the entire DeAOT output folder into
            # the VSR-Pro-owned workspace, then re-anchor mask_dir /
            # out_root to the relocated copy.
            target = workspace / out_root.name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            try:
                shutil.move(str(out_root), str(workspace))
            except Exception as e:  # noqa: BLE001
                # Fallback to copy on cross-volume / file-busy situations
                log.warning("Direct move failed (%s); copying instead", e)
                shutil.copytree(str(out_root), str(target))
                shutil.rmtree(str(out_root), ignore_errors=True)
            out_root = target
            mask_dir = out_root / mask_dir.name
            log.info("Relocated DeAOT outputs -> %s", out_root)

    # Stage 3: mask sanity pass. The cleanup pass now also returns the
    # raw union bbox + count of non-empty frames, so Stage 4 can skip
    # its full mask-dir rescan -- saving ~10-15 min on a 98K-mask video.
    emit_progress("mask_cleanup", 0.0)
    if mask_dilate > 0:
        log.info("Mask cleanup will dilate each mask outward by %d px "
                 "to swallow anti-aliased / outline edges", mask_dilate)
    n_masks, cached_union, n_with_content = _clean_segtracker_masks(
        mask_dir, dilate_px=mask_dilate)
    log.info("Cleaned mask directory: %d PNGs ready for ProPainter", n_masks)
    emit_progress("mask_cleanup", 1.0, str(n_masks))

    # Stage 4: auto-crop decision
    bbox = None
    if auto_crop:
        emit_progress("bbox", 0.0)
        if cached_union is None:
            # Every frame was empty -- mirror legacy _compute_bbox warning.
            log.warning("Disabling auto-crop because mask is empty.")
            auto_crop = False
        else:
            bbox = _compute_bbox(
                mask_dir, frame_w, frame_h, padding=crop_padding,
                cached_union=cached_union,
                cached_n_with_content=n_with_content,
            )
            if bbox is None:
                log.warning("Disabling auto-crop because mask is empty.")
                auto_crop = False
            else:
                emit_progress("bbox", 1.0, f"{bbox[2]}x{bbox[3]}")

    if auto_crop and bbox is not None:
        cx, cy, cw, ch = bbox

        # Skip cropping if the bbox is already most of the frame -- the
        # ffmpeg roundtrip would cost more than it saves.
        coverage = (cw * ch) / (frame_w * frame_h)
        if coverage > 0.6:
            log.info(
                "Bbox covers %.0f%% of frame; auto-crop disabled "
                "(not worth the ffmpeg overhead).", coverage * 100,
            )
            auto_crop = False

    if auto_crop and bbox is not None:
        cx, cy, cw, ch = bbox
        crop_workspace = out_root / "_crop"
        crop_workspace.mkdir(exist_ok=True)
        crop_video = crop_workspace / f"{video.stem}_crop.mp4"
        crop_masks = crop_workspace / "masks"
        crop_out = crop_workspace / "out"
        crop_out.mkdir(exist_ok=True)

        # Persist bbox + source frame dims so a later recomposite pass
        # (tool/dynamic_recomposite.py) can re-do the overlay step
        # without re-running ProPainter. Cheap (~50 bytes) and protects
        # against re-deriving the padded+aligned bbox from masks, which
        # would be off by a few pixels.
        import json as _json
        bbox_sidecar = crop_workspace / "_bbox.json"
        try:
            with open(bbox_sidecar, "w", encoding="utf-8") as _fh:
                _json.dump({
                    "schema_version": 1,
                    "x": int(cx), "y": int(cy),
                    "w": int(cw), "h": int(ch),
                    "frame_w": int(frame_w), "frame_h": int(frame_h),
                }, _fh)
        except OSError as _e:
            log.warning("Failed to write bbox sidecar %s: %s",
                        bbox_sidecar, _e)

        # 4a. crop the cleaned intermediate -- skip when a prior run
        # already produced a probe-parseable file at the same path.
        # Trust ffprobe over size alone because NVENC can leave a
        # ~half-meg partial file with no moov atom on a kill.
        emit_progress("crop", 0.0)
        crop_video_ok = False
        if crop_video.is_file() and crop_video.stat().st_size > 1024:
            if ffprobe:
                probe = subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries",
                     "stream=width,height", "-of", "csv", str(crop_video)],
                    capture_output=True, text=True, timeout=15,
                    creationflags=_NO_WINDOW,
                )
                crop_video_ok = (probe.returncode == 0 and
                                 "stream" in probe.stdout)
            else:
                crop_video_ok = True  # best effort if no ffprobe
        if crop_video_ok:
            log.info("Re-using existing crop video %s (%.1f MB)",
                     crop_video.name, crop_video.stat().st_size / 1e6)
        else:
            crop_video.unlink(missing_ok=True)
            _crop_video(decode_source, crop_video, cx, cy, cw, ch, ffmpeg)

        # 4b. crop the masks -- skip when count matches the cleaned set
        # AND the dest is non-empty. The crop is deterministic given
        # the same bbox, so this is safe.
        expected_n = sum(1 for _ in mask_dir.glob("*.png"))
        existing_n = (sum(1 for _ in crop_masks.glob("*.png"))
                      if crop_masks.is_dir() else 0)
        if existing_n == expected_n and expected_n > 0:
            log.info("Re-using %d existing cropped masks in %s",
                     existing_n, crop_masks)
        else:
            if crop_masks.is_dir() and existing_n != expected_n:
                log.info("Crop-mask count drifted (%d on disk vs %d "
                         "expected); regenerating.", existing_n, expected_n)
                shutil.rmtree(crop_masks, ignore_errors=True)
            _crop_masks(mask_dir, crop_masks, cx, cy, cw, ch)
        emit_progress("crop", 1.0)
        # 5. ProPainter on the crop (chunks internally for long videos)
        emit_progress("propainter", 0.0, f"{cw}x{ch}")
        inpainted_crop = _run_propainter(
            video=crop_video,
            mask_dir=crop_masks,
            output_dir=crop_out,
            fp16=fp16,
            subvideo_length=subvideo_length,
            ffmpeg=ffmpeg,
        )
        emit_progress("propainter", 1.0)
        # 6. overlay back onto the cleaned base (cleaned has same length
        # as original; using cleaned as base avoids re-introducing the
        # decode quirks we paid to fix in Stage 0).
        #
        # Use the cropped masks as a feathered alpha channel so only
        # the actual watermark pixels (plus a 5px ramp) get replaced.
        # A hard-edge paste of the full bbox produces a visible
        # rectangular seam at the bbox boundary because ProPainter
        # slightly nudges colour/lighting on unmasked pixels inside
        # the bbox to maintain temporal consistency.
        emit_progress("overlay", 0.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = _probe_fps(decode_source, ffprobe) if ffprobe else None
        if fps and fps > 0:
            _overlay(
                decode_source, inpainted_crop, output, cx, cy, ffmpeg,
                mask_seq_dir=crop_masks, feather_px=5, fps=fps,
            )
        else:
            log.warning(
                "ffprobe could not determine source fps (%r); falling "
                "back to hard-edge overlay -- the bbox boundary will "
                "be visible in the output.", fps,
            )
            _overlay(decode_source, inpainted_crop, output, cx, cy, ffmpeg)
        emit_progress("overlay", 1.0)

    else:
        # No-crop path: ProPainter on the full frame, copy result out.
        emit_progress("propainter", 0.0, f"{frame_w}x{frame_h}")
        inpainted = _run_propainter(
            video=decode_source,
            mask_dir=mask_dir,
            output_dir=out_root,
            fp16=fp16,
            subvideo_length=subvideo_length,
            ffmpeg=ffmpeg,
        )
        emit_progress("propainter", 1.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(inpainted, output)

    log.info("Wrote %s (%.1f MB)", output, output.stat().st_size / 1e6)
    emit_progress("done", 1.0, str(output))

    # Sentinel for the parent to parse
    print(f"DYNAMIC_RESULT {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as e:
        log.error("%s", e)
        sys.exit(4)
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(6)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(99)
