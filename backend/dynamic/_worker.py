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
        # We let ffmpeg quietly skip undecodable packets so a single
        # corrupted frame doesn't take the whole pipe down. The cost
        # is a frame count that may be slightly less than container
        # metadata claims (which is fine -- DeAOT just stops there).
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-i", str(self._video),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-vsync", "passthrough",
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

def _clean_segtracker_masks(mask_dir: Path) -> int:
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

    Returns the count of remaining mask PNGs after cleanup.
    """
    import numpy as np
    from PIL import Image

    aux_masks = list(mask_dir.glob("*_new.png"))
    if aux_masks:
        aux_dir = mask_dir.parent / f"{mask_dir.name}_aux"
        aux_dir.mkdir(exist_ok=True)
        for f in aux_masks:
            f.replace(aux_dir / f.name)
        log.info("Moved %d auxiliary _new.png mask(s) to %s",
                 len(aux_masks), aux_dir)

    rewritten = 0
    max_obj_seen = 0
    for png in sorted(mask_dir.glob("*.png")):
        arr = np.array(Image.open(png))
        max_obj_seen = max(max_obj_seen, int(arr.max()))
        binary = ((arr == 1).astype(np.uint8) * 255)
        Image.fromarray(binary).save(png)
        rewritten += 1
    log.info(
        "Rewrote %d mask PNGs to object-1-only binary "
        "(max tracked id observed across video: %d)",
        rewritten, max_obj_seen,
    )
    return rewritten


def _compute_bbox(
    mask_dir: Path,
    frame_w: int,
    frame_h: int,
    padding: int,
    align: int = 8,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute the union bounding box of all nonzero pixels in *mask_dir*.

    Returns ``(x, y, w, h)`` clamped to the video frame and snapped so
    that ``w`` and ``h`` are multiples of *align* (ProPainter resizes
    inputs to multiples of 8 internally, so giving it an already-aligned
    crop avoids a one-pixel resize artifact at the overlay boundary).

    Returns None if every mask is empty (no work to do).
    """
    import numpy as np
    from PIL import Image

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
        "[%.1f%% of frame area, %d of %d frames had mask content]",
        min_x, min_y, max_x, max_y, w_aligned, h_aligned, x0, y0,
        100 * w_aligned * h_aligned / (frame_w * frame_h),
        n_with_content, len(list(mask_dir.glob("*.png"))),
    )
    return x0, y0, w_aligned, h_aligned


def _crop_video(
    src: Path, dst: Path, x: int, y: int, w: int, h: int, ffmpeg: str,
) -> None:
    """Crop *src* to ``w x h`` at offset ``(x, y)``, write to *dst*."""
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-an",
        str(dst),
    ]
    log.info("ffmpeg crop -> %s (%dx%d at %d,%d)", dst.name, w, h, x, y)
    rc = subprocess.call(cmd, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(f"ffmpeg crop failed (exit {rc})")


def _crop_masks(
    src_dir: Path, dst_dir: Path, x: int, y: int, w: int, h: int,
) -> int:
    """Crop every PNG mask in *src_dir* and write to *dst_dir*."""
    from PIL import Image
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(src_dir.glob("*.png")):
        img = Image.open(png).crop((x, y, x + w, y + h))
        img.save(dst_dir / png.name)
        n += 1
    log.info("Cropped %d masks -> %s", n, dst_dir)
    return n


def _overlay(
    base: Path, crop: Path, out: Path, x: int, y: int, ffmpeg: str,
) -> None:
    """Overlay *crop* on *base* at ``(x, y)`` -> *out*, preserving audio."""
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(base),
        "-i", str(crop),
        "-filter_complex", f"[0:v][1:v]overlay={x}:{y}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy", "-shortest",
        str(out),
    ]
    log.info("ffmpeg overlay %s onto %s at (%d,%d) -> %s",
             crop.name, base.name, x, y, out.name)
    rc = subprocess.call(cmd, creationflags=_NO_WINDOW)
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
# the clip is longer we slice with ffmpeg, ProPainter each slice
# independently, then concat the inpainted slices back. Boundaries are
# visible only if the watermark drifts dramatically inside a single
# slice -- for typical fixed-position logos the seams are invisible.
PROPAINTER_CHUNK_FRAMES = 800


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


def _ffmpeg_slice_video(
    src: Path, dst: Path, start_frame: int, n_frames: int, ffmpeg: str,
) -> None:
    """Extract ``[start_frame, start_frame + n_frames)`` from *src* into *dst*.

    Uses ``select=between(n,...)`` for exact frame alignment with the
    cv2/ProPainter view of the video, and ``setpts=PTS-STARTPTS`` to
    rebase timestamps so the chunk plays from t=0.
    """
    last = start_frame + n_frames - 1
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"select=between(n\\,{start_frame}\\,{last}),"
               f"setpts=PTS-STARTPTS",
        "-vsync", "vfr",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-an",
        str(dst),
    ]
    rc = subprocess.call(cmd, creationflags=_NO_WINDOW)
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg slice failed (exit {rc}) for frames {start_frame}..{last}"
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
) -> Path:
    """Slice -> ProPainter each chunk -> ffmpeg concat the outputs."""
    import shutil as _shutil

    chunks_dir = output_dir / "_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (total_frames + chunk_frames - 1) // chunk_frames
    chunk_outputs = []

    for idx in range(n_chunks):
        start = idx * chunk_frames
        n = min(chunk_frames, total_frames - start)
        last = start + n - 1
        log.info("Chunk %d/%d: frames %d..%d (%d frames)",
                 idx + 1, n_chunks, start, last, n)
        emit_progress("propainter", (idx) / n_chunks,
                      f"chunk_{idx + 1}/{n_chunks}")

        cdir = chunks_dir / f"c{idx:04d}"
        cdir.mkdir(exist_ok=True)
        cvideo = cdir / "in.mp4"
        cmask = cdir / "masks"
        cmask.mkdir(exist_ok=True)
        cout = cdir / "out"
        cout.mkdir(exist_ok=True)

        # ffmpeg slice the source video to this chunk's frame range
        _ffmpeg_slice_video(video, cvideo, start, n, ffmpeg)

        # Copy the matching mask subset (renamed to start from 00000)
        for i in range(start, start + n):
            src_mask = mask_dir / f"{i:05d}.png"
            if not src_mask.is_file():
                raise FileNotFoundError(
                    f"Expected mask missing: {src_mask}"
                )
            dst_mask = cmask / f"{i - start:05d}.png"
            _shutil.copy(src_mask, dst_mask)

        # ProPainter on this chunk -- single-shot since it's now small
        produced = _run_propainter_single(
            cvideo, cmask, cout,
            fp16=fp16, subvideo_length=subvideo_length,
        )
        chunk_outputs.append(produced)

    # Concat the chunk outputs into one final video at the conventional
    # path our caller expects (output_dir/<video_stem>/inpaint_out.mp4).
    final_dir = output_dir / video.stem
    final_dir.mkdir(parents=True, exist_ok=True)
    final = final_dir / "inpaint_out.mp4"
    _ffmpeg_concat(chunk_outputs, final, ffmpeg)
    log.info("Concatenated %d chunks -> %s", len(chunk_outputs), final)
    return final


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
    workspace_str = cfg.get("workspace")
    workspace = Path(workspace_str).resolve() if workspace_str else None

    if not video.is_file():
        log.error("Input video not found: %s", video)
        return 3
    if not clicks:
        log.error("No clicks provided")
        return 3

    ffmpeg = _find_ffmpeg()

    # Stage 1+2: SAM first-frame + DeAOT propagation. SegTracker hard-
    # codes its mask output to <wm_path>/output/<stem>; we relocate
    # the whole thing immediately afterwards so subsequent stages and
    # any leftover files live inside VSR Pro's tree, not in the
    # sibling watermark_remover repo.
    ffprobe = _find_ffprobe()
    mask_dir, out_root, frame_w, frame_h = _run_sam_and_deaot(
        video=video, clicks=clicks, aot_model=aot_model,
        ffmpeg=ffmpeg, ffprobe=ffprobe,
    )

    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        # Move (or copy + delete) the entire DeAOT output folder into
        # the VSR-Pro-owned workspace, then re-anchor mask_dir / out_root
        # to the relocated copy.
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

    # Stage 3: mask sanity pass
    emit_progress("mask_cleanup", 0.0)
    n_masks = _clean_segtracker_masks(mask_dir)
    log.info("Cleaned mask directory: %d PNGs ready for ProPainter", n_masks)
    emit_progress("mask_cleanup", 1.0, str(n_masks))

    # Stage 4: auto-crop decision
    bbox = None
    if auto_crop:
        emit_progress("bbox", 0.0)
        bbox = _compute_bbox(mask_dir, frame_w, frame_h, padding=crop_padding)
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

        # 4a. crop the video
        emit_progress("crop", 0.0)
        _crop_video(video, crop_video, cx, cy, cw, ch, ffmpeg)
        # 4b. crop the masks
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
        # 6. overlay back onto the original
        emit_progress("overlay", 0.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        _overlay(video, inpainted_crop, output, cx, cy, ffmpeg)
        emit_progress("overlay", 1.0)

    else:
        # No-crop path: ProPainter on the full frame, copy result out.
        emit_progress("propainter", 0.0, f"{frame_w}x{frame_h}")
        inpainted = _run_propainter(
            video=video,
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
