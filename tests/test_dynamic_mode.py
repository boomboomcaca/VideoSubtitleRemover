"""
Smoke + unit tests for the experimental Dynamic Watermark Mode wiring.

These tests deliberately exercise *only* the deterministic, low-cost
pieces of ``backend.dynamic``:

* click-spec parsing
* phase-to-overall-progress math
* progress-sentinel parsing
* bounding-box computation on synthetic masks
* the Tk UI module imports + builds without crashing (headless)

The actual SAM / DeAOT / ProPainter pipeline is *not* exercised here --
it needs the sibling watermark_remover project's bundled conda env
plus a GPU, neither of which belong in CI. Run
``python -m tool.check_dynamic_mode`` to verify the full stack
end-to-end on a developer machine.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


def _has_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# --------------------------------------------------------------------------- #

class TestParseClicks(unittest.TestCase):
    def test_basic_positive(self):
        from backend.dynamic import parse_clicks
        self.assertEqual(parse_clicks("100,200+"), [(100, 200, 1)])

    def test_default_sign_is_positive(self):
        from backend.dynamic import parse_clicks
        self.assertEqual(parse_clicks("50,50"), [(50, 50, 1)])

    def test_mixed_signs(self):
        from backend.dynamic import parse_clicks
        self.assertEqual(
            parse_clicks("100,200+;300,400-;500,600+"),
            [(100, 200, 1), (300, 400, 0), (500, 600, 1)],
        )

    def test_whitespace_tolerated(self):
        from backend.dynamic import parse_clicks
        self.assertEqual(
            parse_clicks("  100 , 200 + ;  300 , 400 -  "),
            [(100, 200, 1), (300, 400, 0)],
        )

    def test_rejects_empty(self):
        from backend.dynamic import parse_clicks
        with self.assertRaises(ValueError):
            parse_clicks("")
        with self.assertRaises(ValueError):
            parse_clicks(";;;")

    def test_rejects_malformed(self):
        from backend.dynamic import parse_clicks
        with self.assertRaises(ValueError):
            parse_clicks("notacoord")
        with self.assertRaises(ValueError):
            parse_clicks("100;200+")


class TestPhaseToOverall(unittest.TestCase):
    def test_endpoints(self):
        from backend.dynamic import phase_to_overall
        self.assertEqual(phase_to_overall("loading", 0.0), 0.0)
        self.assertEqual(phase_to_overall("done", 1.0), 1.0)

    def test_monotonic_across_phases(self):
        from backend.dynamic import phase_to_overall, PHASES
        last = -1.0
        for p in PHASES:
            v = phase_to_overall(p, 1.0)
            self.assertGreaterEqual(v, last, f"non-monotonic at {p}")
            last = v

    def test_clamps_out_of_range(self):
        from backend.dynamic import phase_to_overall
        self.assertEqual(phase_to_overall("deaot", -0.5), phase_to_overall("deaot", 0.0))
        self.assertEqual(phase_to_overall("deaot", 1.5), phase_to_overall("deaot", 1.0))

    def test_unknown_phase_is_zero(self):
        from backend.dynamic import phase_to_overall
        self.assertEqual(phase_to_overall("nonexistent", 0.5), 0.0)


class TestParseProgress(unittest.TestCase):
    def test_well_formed_two_field(self):
        from backend.dynamic.external_pipeline import _parse_progress
        self.assertEqual(_parse_progress("PROGRESS deaot 0.500"),
                         ("deaot", 0.5, ""))

    def test_well_formed_with_extra(self):
        from backend.dynamic.external_pipeline import _parse_progress
        self.assertEqual(_parse_progress("PROGRESS bbox 1.0 384x384"),
                         ("bbox", 1.0, "384x384"))
        self.assertEqual(_parse_progress("PROGRESS sam 0.0 3_clicks"),
                         ("sam", 0.0, "3_clicks"))

    def test_unrelated_lines_return_none(self):
        from backend.dynamic.external_pipeline import _parse_progress
        self.assertIsNone(_parse_progress("some other log line"))
        self.assertIsNone(_parse_progress("DYNAMIC_RESULT /tmp/out.mp4"))
        self.assertIsNone(_parse_progress(""))

    def test_malformed_returns_none(self):
        from backend.dynamic.external_pipeline import _parse_progress
        self.assertIsNone(_parse_progress("PROGRESS deaot"))
        self.assertIsNone(_parse_progress("PROGRESS deaot notanumber"))


class TestMaskCleanupIdempotent(unittest.TestCase):
    """Regression: when _clean_segtracker_masks runs a second time on
    its own output (which is 0/255 binary), the threshold check must
    NOT compare to literal 1 -- doing so wipes every mask to all-zero
    and silently destroys hours of DeAOT work. This actually shipped
    once: a 5-hour 98668-frame DeAOT run was lost. Test prevents it."""

    def setUp(self):
        try:
            from PIL import Image  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("PIL/numpy unavailable")
        import shutil as _shutil
        self.tmp = Path(tempfile.mkdtemp(prefix="vsr_idem_test_"))
        self._addCleanup = lambda: _shutil.rmtree(self.tmp, ignore_errors=True)

    def tearDown(self):
        self._addCleanup()

    def test_second_pass_does_not_zero_masks(self):
        from PIL import Image
        import numpy as np
        from backend.dynamic._worker import _clean_segtracker_masks
        mask_dir = self.tmp / "masks"
        mask_dir.mkdir()
        # Write a few PRE-CLEANED masks (0 or 255, simulating what the
        # bug-era cleanup produced)
        for i in range(5):
            arr = np.zeros((100, 100), dtype=np.uint8)
            # A small filled circle as the watermark
            arr[40:60, 40:60] = 255
            Image.fromarray(arr).save(mask_dir / f"{i:05d}.png")
        # Run cleanup -- must NOT zero them out
        n = _clean_segtracker_masks(mask_dir)
        self.assertEqual(n, 5)
        # Sample a frame, confirm pixels survived
        out = np.array(Image.open(mask_dir / "00002.png"))
        self.assertGreater((out > 0).sum(), 0,
            "Cleanup zeroed all pixels -- the 0/255 vs 0/1 bug regressed!")
        # Specifically expect roughly the original 400 px (20x20 square)
        self.assertEqual((out > 0).sum(), 400)


class TestParseDeaotFrame(unittest.TestCase):
    """Regression: parent must turn SegTracker's plain-stdout
    'processed frame N' chatter into per-frame DeAOT progress updates
    so the UI doesn't sit at 3% for minutes."""

    def test_extracts_simple(self):
        from backend.dynamic.external_pipeline import _parse_deaot_frame
        self.assertEqual(_parse_deaot_frame("processed frame 0"), 0)
        self.assertEqual(_parse_deaot_frame("processed frame 234, obj_num 1"), 234)
        self.assertEqual(_parse_deaot_frame("processed frame 9999, obj_num 17"), 9999)

    def test_ignores_unrelated(self):
        from backend.dynamic.external_pipeline import _parse_deaot_frame
        self.assertIsNone(_parse_deaot_frame("PROGRESS deaot 0.5"))
        self.assertIsNone(_parse_deaot_frame("DYNAMIC_RESULT /tmp/out.mp4"))
        self.assertIsNone(_parse_deaot_frame("All results saved"))
        self.assertIsNone(_parse_deaot_frame(""))

    def test_handles_partial_match(self):
        from backend.dynamic.external_pipeline import _parse_deaot_frame
        # Lines that contain "processed frame" but not at the start
        self.assertIsNone(_parse_deaot_frame("  processed frame 50"))
        self.assertIsNone(_parse_deaot_frame("re-processed frame 50"))


class TestComputeBbox(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("PIL or numpy unavailable")
        self.tmp = Path(tempfile.mkdtemp(prefix="vsr_bbox_test_"))
        self._write_synthetic_masks(self.tmp / "masks")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_synthetic_masks(self, d: Path):
        from PIL import Image
        import numpy as np
        d.mkdir()
        for i in range(5):
            arr = np.zeros((480, 640), dtype=np.uint8)
            x0, y0 = 100 + i * 20, 80 + i * 15
            arr[y0:y0 + 40, x0:x0 + 40] = 255
            Image.fromarray(arr).save(d / f"{i:05d}.png")

    def test_unions_motion(self):
        from backend.dynamic._worker import _compute_bbox
        bbox = _compute_bbox(self.tmp / "masks", frame_w=640, frame_h=480, padding=0)
        # 5 squares, each 40x40, moving by (+20, +15) per frame:
        #   x0 in {100,120,140,160,180}, square spans [x0, x0+40)
        #   y0 in {80,95,110,125,140},   square spans [y0, y0+40)
        # Pixel-inclusive union extent:
        #   x: 100..219 (width = 120)
        #   y: 80..179  (height = 100)
        # Worker snaps to multiples of 8:
        #   width 120 already 0 mod 8, stays 120.
        #   height 100 -> floor(100/8)*8 = 96; grow to 104 if it still
        #   fits inside the frame (it does), so final 104.
        self.assertEqual(bbox, (100, 80, 120, 104))

    def test_pads_and_aligns(self):
        from backend.dynamic._worker import _compute_bbox
        bbox = _compute_bbox(self.tmp / "masks", frame_w=640, frame_h=480, padding=10)
        self.assertIsNotNone(bbox)
        x, y, w, h = bbox
        self.assertEqual(w % 8, 0)
        self.assertEqual(h % 8, 0)
        # Padded bbox must contain the unpadded inclusive extent
        # x in [100, 220), y in [80, 180).
        self.assertLessEqual(x, 100)
        self.assertLessEqual(y, 80)
        self.assertGreaterEqual(x + w, 220)
        self.assertGreaterEqual(y + h, 180)

    def test_clamps_to_frame(self):
        from backend.dynamic._worker import _compute_bbox
        bbox = _compute_bbox(self.tmp / "masks", frame_w=640, frame_h=480, padding=10000)
        self.assertIsNotNone(bbox)
        x, y, w, h = bbox
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 480)

    def test_empty_masks_return_none(self):
        from PIL import Image
        import numpy as np
        from backend.dynamic._worker import _compute_bbox
        d = self.tmp / "empty_masks"
        d.mkdir()
        Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).save(d / "00000.png")
        self.assertIsNone(_compute_bbox(d, 100, 100, padding=4))


class TestFontNormalizer(unittest.TestCase):
    """Regression: the host app's Theme stores F_* as bare ints, not
    tuples; an earlier version of the UI module crashed with
    ``TypeError: unsupported operand type(s) for +: 'int' and 'tuple'``
    when opening the window with the real Theme. _font() must handle
    int, tuple, and str shapes uniformly."""

    def test_int_size(self):
        from backend.dynamic.ui import _font
        # Mimic VSR Pro's Theme: F_TITLE = 12 (just a size)
        self.assertEqual(_font(12, "bold"), ("Segoe UI", 12, "bold"))

    def test_tuple_full_spec(self):
        from backend.dynamic.ui import _font
        self.assertEqual(_font(("Arial", 14), "bold"),
                         ("Arial", 14, "bold"))

    def test_string_family(self):
        from backend.dynamic.ui import _font
        result = _font("Verdana", "italic")
        self.assertEqual(result[0], "Verdana")
        self.assertIn("italic", result)

    def test_no_modifiers(self):
        from backend.dynamic.ui import _font
        self.assertEqual(_font(10), ("Segoe UI", 10))

    def test_with_host_int_theme_does_not_crash(self):
        """Build the actual window with an int-style theme (the failure
        mode that ate the production launch)."""
        if not _has_display():
            self.skipTest("No display available")
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("PIL unavailable")
        import tkinter as tk

        class IntTheme:
            BG_DARK = "#000"; BG_SECONDARY = "#111"; BG_TERTIARY = "#222"
            BG_RAISED = "#333"; TEXT_PRIMARY = "#fff"; TEXT_SECONDARY = "#ccc"
            TEXT_MUTED = "#888"; ACCENT = "#08f"; SUCCESS = "#0f0"
            WARNING = "#fa0"; ERROR = "#f00"
            S_XS = 4; S_SM = 8; S_MD = 12; S_LG = 16; S_XL = 24
            F_TITLE = 16; F_HEADING = 12; F_BODY = 10; F_META = 9  # ints!

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk cannot initialise")
        root.withdraw()
        try:
            from backend.dynamic.ui import DynamicWatermarkWindow
            win = DynamicWatermarkWindow(root, theme=IntTheme)
            win.update_idletasks()
            win.destroy()
        finally:
            root.destroy()

    def test_with_minimal_theme_falls_back_for_missing_attrs(self):
        """Regression: VSR Pro's Theme is missing ACCENT/SUCCESS/etc.,
        which previously crashed window construction. The _ThemeAdapter
        must transparently fall back to _DefaultTheme for any attribute
        the host theme doesn't define."""
        if not _has_display():
            self.skipTest("No display available")
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("PIL unavailable")
        import tkinter as tk

        # Pathological host theme: only defines a *few* attributes;
        # everything else must come from the fallback.
        class MinimalHostTheme:
            BG_DARK = "#001122"
            TEXT_PRIMARY = "#eeeeff"
            F_BODY = 11
            # intentionally no ACCENT, SUCCESS, WARNING, ERROR,
            # BG_RAISED, BG_TERTIARY, S_*, F_TITLE, etc.

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk cannot initialise")
        root.withdraw()
        try:
            from backend.dynamic.ui import DynamicWatermarkWindow
            win = DynamicWatermarkWindow(root, theme=MinimalHostTheme)
            win.update_idletasks()
            # Confirm we successfully used the host attr where present
            # and fell back where not.
            assert win._theme.BG_DARK == "#001122", "host attr should win"
            assert win._theme.ACCENT, "missing attr should fall back"
            win.destroy()
        finally:
            root.destroy()

    def test_theme_adapter_proxies_correctly(self):
        from backend.dynamic.ui import _ThemeAdapter, _DefaultTheme

        class Host:
            BG_DARK = "#hostbg"
            # everything else missing

        a = _ThemeAdapter(Host)
        self.assertEqual(a.BG_DARK, "#hostbg")  # host wins
        self.assertEqual(a.ACCENT, _DefaultTheme.ACCENT)  # fall back
        # None host -> pure default
        b = _ThemeAdapter(None)
        self.assertEqual(b.BG_DARK, _DefaultTheme.BG_DARK)


class TestUiSmoke(unittest.TestCase):
    def setUp(self):
        if not _has_display():
            self.skipTest("No display available")
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("PIL unavailable")

    def test_window_builds_without_error(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk cannot initialise")
        root.withdraw()
        try:
            from backend.dynamic.ui import DynamicWatermarkWindow
            win = DynamicWatermarkWindow(root)
            win.update_idletasks()
            win.destroy()
        finally:
            root.destroy()

    def test_click_coordinate_translation(self):
        import tkinter as tk
        import numpy as np
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk cannot initialise")
        root.withdraw()
        try:
            from backend.dynamic.ui import ClickPointsCanvas
            c = ClickPointsCanvas(root)
            c.pack()
            c.update_idletasks()
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            c.set_frame(frame)
            c.update_idletasks()
            cw, ch = c.winfo_width(), c.winfo_height()
            if cw < 10 or ch < 10:
                self.skipTest("Canvas did not realize size in headless env")
            c._add_click(cw // 2, ch // 2, mode=1)
            self.assertEqual(len(c.clicks), 1)
            x, y, mode = c.clicks[0]
            self.assertEqual(mode, 1)
            # Centre of canvas should map to roughly centre of original frame
            self.assertTrue(900 < x < 1020, f"x={x} not centre-ish")
            self.assertTrue(500 < y < 580, f"y={y} not centre-ish")
        finally:
            root.destroy()


class TestSamPreviewClient(unittest.TestCase):
    """SamPreviewClient with a fake subprocess -- verifies the JSON
    protocol without needing the wm_env / SAM stack."""

    def _make_client_with_fake_proc(self, responses):
        """Build a client whose _ensure_proc populates a fake Popen-like
        object that scripts canned stdin reads."""
        from backend.dynamic.sam_preview import SamPreviewClient
        import io

        class FakeProc:
            def __init__(self, resp_lines):
                self.stdin = io.StringIO()
                # Prepend the ready sentinel that _ensure_proc waits for
                payload = '{"ok": true, "ready": true}\n' + "".join(
                    json.dumps(r) + "\n" for r in resp_lines
                )
                self.stdout = io.StringIO(payload)
                self.stderr = io.StringIO("")
                self._alive = True

            def poll(self):
                return None if self._alive else 0

            def wait(self, timeout=None):
                self._alive = False

            def kill(self):
                self._alive = False

        client = SamPreviewClient.__new__(SamPreviewClient)
        # Hand-init only the bits _ensure_proc would otherwise set up
        import tempfile
        from pathlib import Path
        client._wm_path = Path("/fake/wm")
        client._proc = FakeProc(responses)
        client._lock = __import__("threading").Lock()
        client._tmpdir = Path(tempfile.mkdtemp(prefix="sam_preview_test_"))
        client._image_path = client._tmpdir / "frame.png"
        client._mask_path = client._tmpdir / "mask.png"
        client._closed = False
        # Drain the ready sentinel as _ensure_proc would
        client._proc.stdout.readline()
        return client

    def test_predict_sends_correct_request(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("PIL/numpy unavailable")
        # We need to know the mask_path *before* building the fake
        # responses (the response must echo the real on-disk path so
        # cv2.imread succeeds when the client tries to load it).
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="sam_preview_test_"))
        mask_path = tmpdir / "mask.png"
        Image.fromarray(np.array([[255]], dtype=np.uint8)).save(mask_path)

        client = self._make_client_with_fake_proc([
            {"ok": True, "mask_path": str(mask_path)},
        ])
        client._mask_path = mask_path  # match what the response points to
        mask = client.predict([(10, 20), (30, 40)], [1, 0])
        # Check request that was written to fake stdin
        sent = client._proc.stdin.getvalue().strip()
        req = json.loads(sent)
        self.assertEqual(req["type"], "predict")
        self.assertEqual(req["coords"], [[10, 20], [30, 40]])
        self.assertEqual(req["modes"], [1, 0])
        self.assertIsNotNone(mask)

    def test_predict_failure_returns_none(self):
        client = self._make_client_with_fake_proc([
            {"ok": False, "error": "no image set"},
        ])
        mask = client.predict([(10, 20)], [1])
        self.assertIsNone(mask)

    def test_close_is_idempotent(self):
        from backend.dynamic.sam_preview import SamPreviewClient
        c = SamPreviewClient(Path("/fake/wm"))
        c.close()
        c.close()  # second call must not crash


class TestDebouncedSamPreview(unittest.TestCase):
    """DebouncedSamPreview against a stub client that records calls."""

    def test_collapse_to_latest_request(self):
        from backend.dynamic.sam_preview import DebouncedSamPreview
        import threading

        seen_predicts = []
        masks_delivered = []
        gate = threading.Event()

        class StubClient:
            def predict(self_inner, coords, modes):
                seen_predicts.append((list(coords), list(modes)))
                gate.wait(timeout=2)  # block until the test releases
                return f"mask-{len(seen_predicts)}"

        def on_mask(m):
            masks_delivered.append(m)

        deb = DebouncedSamPreview(StubClient(), on_mask=on_mask)
        try:
            deb.request([(1, 1)], [1])  # job A starts (blocks)
            # While A is running, queue many requests
            for i in range(5):
                deb.request([(i, i)], [1])  # only the last survives
            gate.set()  # release A
            # Wait for delivery of A then of the LAST pending request
            for _ in range(30):
                if len(masks_delivered) >= 2:
                    break
                import time
                time.sleep(0.05)
        finally:
            deb.stop()

        # We must have collapsed to AT MOST 2 predicts (A + final)
        self.assertLessEqual(len(seen_predicts), 2,
                             f"too many predicts: {seen_predicts}")
        self.assertGreaterEqual(len(masks_delivered), 1)
        # The last predict must be the LAST request
        if len(seen_predicts) >= 2:
            self.assertEqual(seen_predicts[-1], ([(4, 4)], [1]))


class TestSamWorkerScriptIsValid(unittest.TestCase):
    """The SAM worker runs in a different env so we can't import it
    here, but we can verify it parses as valid Python."""

    def test_parses(self):
        import ast
        worker_path = Path(__file__).resolve().parents[1] / \
            "backend" / "dynamic" / "_sam_worker.py"
        src = worker_path.read_text(encoding="utf-8")
        ast.parse(src)  # raises SyntaxError on failure


if __name__ == "__main__":
    import json  # used inside test methods above
    unittest.main()
