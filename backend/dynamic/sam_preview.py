"""
Client-side wrapper around the persistent SAM preview worker.

Spawns ``backend/dynamic/_sam_worker.py`` under the watermark_remover
project's bundled ``env/python.exe`` (the only Python on the machine
that has SAM + torch installed), and exposes a small API the UI can
call from a background thread:

* :py:meth:`SamPreviewClient.set_image(frame_bgr)` -- give the worker
  a new frame; it caches the SAM image features so subsequent
  predict() calls are fast.
* :py:meth:`SamPreviewClient.predict(coords, modes)` -- run SAM with
  the given clicks; returns the mask as a ``uint8`` numpy array
  (255 = inside, 0 = outside) or None on failure.
* :py:meth:`SamPreviewClient.close()` -- shut the worker down cleanly.

The class is thread-safe (an internal lock serialises stdin/stdout
exchanges with the worker), so the UI can fire predicts from a
background thread while the GUI thread keeps painting.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

# Match _worker.py's discovery: the parent process is expected to have
# already resolved the watermark_remover path and to pass it in.
_WORKER_REL = "backend/dynamic/_sam_worker.py"


class SamPreviewClient:
    """Persistent SAM worker driver.

    Lifecycle: lazy -- the subprocess is not spawned until the first
    ``set_image`` or ``predict`` call. ``close()`` is safe to call
    multiple times and a no-op when the worker was never started.
    """

    def __init__(self, wm_path: Path):
        self._wm_path = Path(wm_path).resolve()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="sam_preview_"))
        self._image_path = self._tmpdir / "frame.png"
        self._mask_path = self._tmpdir / "mask.png"
        self._closed = False

    # ----- lifecycle -----

    def _ensure_proc(self) -> None:
        """Spawn the worker subprocess if it isn't already running.

        The very first byte the worker writes is a JSON readiness
        sentinel; we block here until we see it so callers don't race
        the model load.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._closed:
            raise RuntimeError("SamPreviewClient already closed")

        wm_python = self._wm_path / "env" / "python.exe"
        if not wm_python.is_file():
            raise FileNotFoundError(
                f"watermark_remover bundled python missing: {wm_python}"
            )
        repo_root = Path(__file__).resolve().parents[2]
        worker_path = repo_root / _WORKER_REL
        if not worker_path.is_file():
            raise FileNotFoundError(f"SAM worker script missing: {worker_path}")

        logger.info("Spawning SAM worker under %s", wm_python)
        self._proc = subprocess.Popen(
            [str(wm_python), str(worker_path)],
            cwd=str(self._wm_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # keep stderr separate so it doesn't
                                     # corrupt the JSON stream on stdout
            text=True, bufsize=1,
            encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW,
        )
        # Wait for ready sentinel
        line = self._proc.stdout.readline() if self._proc.stdout else ""
        try:
            ready = json.loads(line)
        except json.JSONDecodeError:
            self._kill()
            raise RuntimeError(
                f"SAM worker did not emit a ready sentinel; got {line!r}"
            )
        if not ready.get("ok") or not ready.get("ready"):
            err = ready.get("error", "unknown")
            self._kill()
            raise RuntimeError(f"SAM worker failed to initialise: {err}")
        logger.info("SAM worker ready.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                with self._lock:
                    self._send_locked({"type": "shutdown"})
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                logger.warning("SAM worker did not shut down cleanly; killing")
                self._kill()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _kill(self):
        try:
            if self._proc is not None:
                self._proc.kill()
                self._proc.wait(timeout=2)
        except Exception:
            pass

    def __del__(self):  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ----- API -----

    def set_image(self, frame_bgr) -> None:
        """Cache the SAM image features for *frame_bgr* (OpenCV BGR ndarray)."""
        import cv2
        with self._lock:
            self._ensure_proc()
            cv2.imwrite(str(self._image_path), frame_bgr)
            resp = self._send_locked({
                "type": "set_image",
                "path": str(self._image_path),
            })
            if not resp.get("ok"):
                raise RuntimeError(
                    f"SAM set_image failed: {resp.get('error')}"
                )

    def predict(
        self,
        coords: Sequence[Tuple[int, int]],
        modes: Sequence[int],
    ):
        """Run SAM with the given clicks; return ``uint8`` mask or None."""
        import cv2
        with self._lock:
            self._ensure_proc()
            resp = self._send_locked({
                "type": "predict",
                "coords": [[int(x), int(y)] for x, y in coords],
                "modes": [int(m) for m in modes],
                "out_path": str(self._mask_path),
            })
            if not resp.get("ok"):
                logger.warning("SAM predict failed: %s", resp.get("error"))
                return None
            mask = cv2.imread(resp["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning("Cannot reload mask from %s",
                               resp.get("mask_path"))
                return None
            return mask

    # ----- low-level IPC -----

    def _send_locked(self, req: dict) -> dict:
        """Send one request, return parsed response. Caller holds _lock."""
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            # Worker died -- drain stderr for diagnostics
            err = ""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read() or ""
                except Exception:  # noqa: BLE001
                    pass
            raise RuntimeError(
                f"SAM worker exited unexpectedly. stderr tail: {err[-500:]!r}"
            )
        return json.loads(line)


# --------------------------------------------------------------------------- #
# Async wrapper: debounced background predictor for UI use
# --------------------------------------------------------------------------- #

class DebouncedSamPreview:
    """Fire predicts on a background thread, dropping intermediate requests.

    Click events stream in faster than SAM can predict (~300-800 ms per
    call on a 3060). Without debouncing the UI queues up obsolete
    predictions and lags behind the user. We:

    * On request: stash the latest (coords, modes) as the *pending* job.
    * Worker thread loops: if there's a pending job, snapshot it, run
      predict, deliver via callback. While running it can't see new
      requests; when it returns it picks up whatever the latest pending
      job became (could be the same job, could be a newer one).
    * The callback runs on the worker thread; the UI should ``after(0,
      ...)`` into Tk's mainloop.
    """

    def __init__(self, client: SamPreviewClient, on_mask):
        self._client = client
        self._on_mask = on_mask
        self._pending: Optional[Tuple[List[Tuple[int, int]], List[int]]] = None
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def request(self, coords: Sequence[Tuple[int, int]],
                modes: Sequence[int]) -> None:
        with self._cond:
            self._pending = (list(coords), list(modes))
            self._cond.notify()

    def clear(self) -> None:
        with self._cond:
            self._pending = None
            self._cond.notify()
        try:
            self._on_mask(None)
        except Exception:  # noqa: BLE001
            logger.exception("on_mask callback raised on clear")

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job = self._pending
                self._pending = None
            coords, modes = job
            if not coords:
                continue
            try:
                mask = self._client.predict(coords, modes)
            except Exception:  # noqa: BLE001
                logger.exception("SAM predict raised")
                mask = None
            try:
                self._on_mask(mask)
            except Exception:  # noqa: BLE001
                logger.exception("on_mask callback raised")


__all__ = ["SamPreviewClient", "DebouncedSamPreview"]
