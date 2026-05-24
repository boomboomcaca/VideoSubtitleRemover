"""
Long-lived SAM-only worker for interactive click preview.

The dynamic-watermark window spawns one of these the first time the
user loads a video and keeps it alive for the rest of the session.
We use SAM directly (no SegTracker / DeAOT overhead) so each predict
call is cheap once the image features are cached.

Why a persistent worker?
------------------------
VSR Pro's venv doesn't carry the SAM / torch stack -- that lives in
the sibling watermark_remover project's bundled conda env. Every
click triggers a SAM forward pass; spawning a fresh subprocess each
time would mean ~5-10 s of process + model load overhead per click,
which destroys the interactive feel. Loading once and replying to a
stream of JSON requests gives sub-second predicts.

Protocol
--------
Read JSON requests one per stdin line; write JSON responses one per
stdout line. The first stdout line after startup is always
``{"ok": true, "ready": true}`` so the parent knows model loading
finished.

Request shapes::

    {"type": "set_image", "path": "<absolute path>"}
        -> {"ok": true}                  (also caches SAM features)

    {"type": "predict",
     "coords": [[x, y], ...],
     "modes":  [1, 0, ...],              # 1 = positive, 0 = negative
     "out_path": "<absolute png path>"}
        -> {"ok": true, "mask_path": "<out_path>", "score": 0.87}

    {"type": "shutdown"}
        -> {"ok": true} then exit 0

Errors:
        -> {"ok": false, "error": "<message>"}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

WM_PATH = Path.cwd().resolve()
sys.path.insert(0, str(WM_PATH))
sys.path.insert(0, str(WM_PATH / "sam"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SAM-WORKER %(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("sam_worker")


def _emit(obj) -> None:
    """Write one JSON response to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        import cv2
        import numpy as np
        import torch
        from sam.segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as e:  # noqa: BLE001
        _emit({"ok": False, "ready": False, "error": f"import failed: {e}"})
        traceback.print_exc(file=sys.stderr)
        return 2

    ckpt = WM_PATH / "ckpt" / "sam_vit_b_01ec64.pth"
    if not ckpt.is_file():
        _emit({"ok": False, "ready": False,
               "error": f"SAM checkpoint missing: {ckpt}"})
        return 3

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("Loading SAM vit_b from %s onto %s...", ckpt, device)
    try:
        sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
        sam.to(device)
        predictor = SamPredictor(sam)
    except Exception as e:  # noqa: BLE001
        _emit({"ok": False, "ready": False,
               "error": f"SAM init failed: {e}"})
        traceback.print_exc(file=sys.stderr)
        return 4
    log.info("SAM ready.")
    _emit({"ok": True, "ready": True})

    has_image = False
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            _emit({"ok": False, "error": f"invalid JSON: {e}"})
            continue

        t = req.get("type")
        try:
            if t == "shutdown":
                _emit({"ok": True})
                return 0

            elif t == "set_image":
                img = cv2.imread(req["path"])
                if img is None:
                    _emit({"ok": False, "error": f"cannot read {req['path']}"})
                    continue
                # SAM expects RGB
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                predictor.set_image(rgb)
                has_image = True
                _emit({"ok": True, "size": [rgb.shape[1], rgb.shape[0]]})

            elif t == "predict":
                if not has_image:
                    _emit({"ok": False, "error": "no image set"})
                    continue
                coords = np.array(req["coords"], dtype=np.float32)
                modes = np.array(req["modes"], dtype=np.int32)
                if coords.size == 0:
                    _emit({"ok": False, "error": "no clicks"})
                    continue
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=modes,
                    multimask_output=True,
                )
                best_idx = int(scores.argmax())
                mask = (masks[best_idx].astype(np.uint8)) * 255
                out_path = req["out_path"]
                cv2.imwrite(out_path, mask)
                _emit({
                    "ok": True,
                    "mask_path": out_path,
                    "score": float(scores[best_idx]),
                    "nonzero_px": int((masks[best_idx] > 0).sum()),
                })

            else:
                _emit({"ok": False, "error": f"unknown type: {t!r}"})

        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
