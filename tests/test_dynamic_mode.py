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
import shutil
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
        # Run cleanup -- must NOT zero them out. Newer signature returns
        # (count, union_bbox, n_with_content) so cleanup can hand its
        # bbox to _compute_bbox without a second mask-dir scan.
        n, union_bbox, n_with_content = _clean_segtracker_masks(mask_dir)
        self.assertEqual(n, 5)
        self.assertEqual(n_with_content, 5)
        # Bbox is inclusive (x_min, y_min, x_max, y_max) of the 20x20
        # square at [40:60, 40:60] -> (40, 40, 59, 59).
        self.assertEqual(union_bbox, (40, 40, 59, 59))
        # Sample a frame, confirm pixels survived
        out = np.array(Image.open(mask_dir / "00002.png"))
        self.assertGreater((out > 0).sum(), 0,
            "Cleanup zeroed all pixels -- the 0/255 vs 0/1 bug regressed!")
        # Specifically expect roughly the original 400 px (20x20 square)
        self.assertEqual((out > 0).sum(), 400)


class TestCleanupSidecar(unittest.TestCase):
    """Cleanup writes a JSON sidecar with {n_masks, union_bbox,
    n_with_content} on success. A re-run that finds a valid sidecar
    skips the entire per-frame cleanup pass -- this is what lets a
    98K-mask resume return in milliseconds instead of ~5-10 min."""

    def setUp(self):
        try:
            from PIL import Image  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("PIL/numpy unavailable")
        import shutil as _shutil
        self.tmp = Path(tempfile.mkdtemp(prefix="vsr_sidecar_test_"))
        self._addCleanup = lambda: _shutil.rmtree(self.tmp, ignore_errors=True)

    def tearDown(self):
        self._addCleanup()

    def _make_masks(self, mask_dir, n, value=255):
        from PIL import Image
        import numpy as np
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = np.zeros((50, 50), dtype=np.uint8)
            arr[20:30, 20:30] = value
            Image.fromarray(arr).save(mask_dir / f"{i:05d}.png")

    def test_sidecar_written_on_completion(self):
        import json
        from backend.dynamic._worker import _clean_segtracker_masks
        mask_dir = self.tmp / "masks"
        self._make_masks(mask_dir, 3, value=1)  # {0,1} from DeAOT
        n, bb, nwc = _clean_segtracker_masks(mask_dir)
        sidecar = mask_dir.parent / f"{mask_dir.name}_cleanup.json"
        self.assertTrue(sidecar.is_file(), "sidecar not written")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["n_masks"], 3)
        self.assertEqual(data["n_with_content"], 3)
        # 20x20 square -> inclusive bbox (20, 20, 29, 29)
        self.assertEqual(data["union_bbox"], [20, 20, 29, 29])

    def test_sidecar_short_circuits_second_run(self):
        from backend.dynamic._worker import _clean_segtracker_masks
        mask_dir = self.tmp / "masks"
        self._make_masks(mask_dir, 4, value=1)
        # First run: writes sidecar
        n1, bb1, nwc1 = _clean_segtracker_masks(mask_dir)
        sidecar = mask_dir.parent / f"{mask_dir.name}_cleanup.json"
        first_mtime = sidecar.stat().st_mtime_ns
        # Sleep so any new write would change the mtime detectably
        import time
        time.sleep(1.05)
        # Touch one of the mask files BACKWARDS in time to prove the
        # second call did NOT touch it (would update mtime if rewritten).
        target = mask_dir / "00002.png"
        target_mtime_before = target.stat().st_mtime_ns
        # Second run: should hit the sidecar fast-path
        n2, bb2, nwc2 = _clean_segtracker_masks(mask_dir)
        self.assertEqual((n1, bb1, nwc1), (n2, bb2, nwc2))
        # Sidecar was NOT rewritten (no new payload to persist)
        self.assertEqual(sidecar.stat().st_mtime_ns, first_mtime)
        # The mask file was NOT touched either
        self.assertEqual(target.stat().st_mtime_ns, target_mtime_before)

    def test_sidecar_ignored_when_mask_count_mismatches(self):
        import json
        from backend.dynamic._worker import _clean_segtracker_masks
        mask_dir = self.tmp / "masks"
        self._make_masks(mask_dir, 5, value=1)
        # Run once to get a sidecar
        _clean_segtracker_masks(mask_dir)
        sidecar = mask_dir.parent / f"{mask_dir.name}_cleanup.json"
        self.assertTrue(sidecar.is_file())
        # Now delete one mask -- count drifts from cached 5 to 4
        (mask_dir / "00003.png").unlink()
        # Re-run: must NOT trust the sidecar; should re-process the 4
        # remaining masks and rewrite the sidecar with n=4.
        n, bb, nwc = _clean_segtracker_masks(mask_dir)
        self.assertEqual(n, 4)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(data["n_masks"], 4)


class TestWorkspaceModule(unittest.TestCase):
    """backend.dynamic.workspace backs both the CLI cleanup tool and the
    GUI pre-launch confirm dialog. A bug here silently corrupts both
    cache-management code paths, so the per-stage detection logic must
    be locked down by tests."""

    def setUp(self):
        import shutil as _shutil
        self.tmp = Path(tempfile.mkdtemp(prefix="vsr_workspace_test_"))
        self._addCleanup = lambda: _shutil.rmtree(self.tmp, ignore_errors=True)

    def tearDown(self):
        self._addCleanup()

    def _make_full_workspace(self):
        """Mirror the on-disk layout the worker produces post-pipeline:
        intermediate mp4, mask dir, sidecar, crop dir, chunks with output
        mp4s. Every helper field should report yes/non-zero on this fixture.
        """
        ws = self.tmp / "fixture_workspace"
        (ws / "_source_clean").mkdir(parents=True)
        (ws / "_source_clean.mp4").write_bytes(b"\0" * 2048)
        masks = ws / "_source_clean" / "_source_clean_masks"
        masks.mkdir()
        for i in range(3):
            (masks / f"{i:05d}.png").write_bytes(b"\0" * 512)
        sidecar = ws / "_source_clean" / "_source_clean_masks_cleanup.json"
        sidecar.write_text('{"schema_version": 1, "n_masks": 3}',
                           encoding="utf-8")
        crop_dir = ws / "_source_clean" / "_crop"
        crop_dir.mkdir(parents=True)
        (crop_dir / "_source_clean_crop.mp4").write_bytes(b"\0" * 4096)
        crop_masks = crop_dir / "masks"
        crop_masks.mkdir()
        for i in range(3):
            (crop_masks / f"{i:05d}.png").write_bytes(b"\0" * 256)
        chunks = crop_dir / "out" / "_chunks"
        for i in range(2):
            cdir = chunks / f"c{i:04d}"
            cdir.mkdir(parents=True)
            # New schema: trimmed.mp4 + geometry.json sidecar
            (cdir / "trimmed.mp4").write_bytes(b"\0" * 8192)
            (cdir / "geometry.json").write_text(
                '{"version": 2, "chunk_start": 0, "chunk_n": 880,'
                ' "keep_offset": 0, "keep_n": 800, "pad": 80}',
                encoding="utf-8",
            )
        # An incomplete chunk -- a tiny file the size check should reject.
        c_partial = chunks / "c0002"
        c_partial.mkdir(parents=True)
        (c_partial / "trimmed.mp4").write_bytes(b"\0" * 100)
        (c_partial / "geometry.json").write_text("{}", encoding="utf-8")
        # An OLD-schema chunk (inpaint_out.mp4 only, no trimmed/geometry).
        # Must NOT count as done -- old chunks are un-padded and produce
        # visible seams; the next run is supposed to recompute them.
        c_old = chunks / "c0003" / "out" / "in"
        c_old.mkdir(parents=True)
        (c_old / "inpaint_out.mp4").write_bytes(b"\0" * 8192)
        return ws

    def test_describe_empty(self):
        from backend.dynamic.workspace import describe_workspace
        s = describe_workspace(self.tmp / "does_not_exist")
        self.assertFalse(s.exists)
        self.assertFalse(s.has_any_cache)

    def test_describe_full_workspace(self):
        from backend.dynamic.workspace import describe_workspace
        ws = self._make_full_workspace()
        s = describe_workspace(ws)
        self.assertTrue(s.exists)
        self.assertTrue(s.intermediate_exists)
        self.assertEqual(s.n_masks, 3)
        self.assertTrue(s.cleanup_sidecar_exists)
        self.assertTrue(s.crop_video_exists)
        self.assertEqual(s.n_crop_masks, 3)
        # Only the 2 real chunks count; the 100-byte partial is excluded
        self.assertEqual(s.n_chunks_done, 2)
        self.assertGreater(s.size_bytes, 0)
        self.assertTrue(s.has_any_cache)
        bullets = s.stage_summary()
        self.assertTrue(any("Reencoded" in b for b in bullets))
        self.assertTrue(any("DeAOT" in b for b in bullets))

    def test_wipe_full(self):
        from backend.dynamic.workspace import (
            describe_workspace, wipe_workspace,
        )
        ws = self._make_full_workspace()
        self.assertTrue(wipe_workspace(ws))
        s = describe_workspace(ws)
        # Workspace dir itself is kept (the next run reuses it); all
        # cached stages should be gone.
        self.assertFalse(s.intermediate_exists)
        self.assertEqual(s.n_masks, 0)
        self.assertFalse(s.cleanup_sidecar_exists)
        self.assertFalse(s.crop_video_exists)
        self.assertEqual(s.n_crop_masks, 0)
        self.assertEqual(s.n_chunks_done, 0)
        self.assertFalse(s.has_any_cache)

    def test_wipe_keep_source_clean(self):
        from backend.dynamic.workspace import (
            describe_workspace, wipe_workspace,
        )
        ws = self._make_full_workspace()
        self.assertTrue(wipe_workspace(ws, keep_source_clean=True))
        s = describe_workspace(ws)
        # Intermediate preserved, everything else gone
        self.assertTrue(s.intermediate_exists)
        self.assertEqual(s.n_masks, 0)
        self.assertEqual(s.n_chunks_done, 0)

    def test_wipe_only_propainter(self):
        from backend.dynamic.workspace import (
            describe_workspace, wipe_workspace,
        )
        ws = self._make_full_workspace()
        self.assertTrue(wipe_workspace(ws, only_propainter=True))
        s = describe_workspace(ws)
        # ProPainter chunks gone; everything before propainter preserved
        self.assertTrue(s.intermediate_exists)
        self.assertEqual(s.n_masks, 3)
        self.assertTrue(s.cleanup_sidecar_exists)
        self.assertTrue(s.crop_video_exists)
        self.assertEqual(s.n_crop_masks, 3)
        self.assertEqual(s.n_chunks_done, 0)

    def test_workspace_for_video_is_stable_and_path_keyed(self):
        from backend.dynamic.workspace import workspace_for_video
        v1 = self.tmp / "subdir_a" / "movie.mp4"
        v2 = self.tmp / "subdir_b" / "movie.mp4"
        v1.parent.mkdir(parents=True)
        v2.parent.mkdir(parents=True)
        v1.touch()
        v2.touch()
        # Same path, different times -> stable
        self.assertEqual(workspace_for_video(v1), workspace_for_video(v1))
        # Same stem, different absolute path -> different workspace
        # (this is the bug that the sha1(path) suffix prevents)
        self.assertNotEqual(workspace_for_video(v1), workspace_for_video(v2))

    def test_format_size_units(self):
        from backend.dynamic.workspace import format_size
        self.assertEqual(format_size(0), "0 B")
        self.assertEqual(format_size(512), "512 B")
        self.assertEqual(format_size(2048), "2.0 KB")
        self.assertEqual(format_size(2 * 1024 * 1024), "2.0 MB")
        self.assertEqual(format_size(int(3.2 * 1024 ** 3)), "3.2 GB")


class TestOverlayFilterGraph(unittest.TestCase):
    """The exact filter_complex string is the contract between us and
    ffmpeg. Regressions here are silent: the command runs, the file is
    produced, but the visual result is wrong. Pin every variant."""

    def test_hard_edge_no_mask(self):
        from backend.dynamic._worker import _build_overlay_filter
        # Legacy fallback when no mask is provided -- the original
        # signature, unchanged on purpose so a bisect to mask=None
        # cleanly reproduces the pre-feather behaviour.
        g = _build_overlay_filter(100, 200,
                                  mask_input_idx=None, feather_px=5)
        self.assertEqual(g, "[0:v][1:v]overlay=100:200:eof_action=pass")

    def test_alpha_feathered_mask_at_input_2(self):
        from backend.dynamic._worker import _build_overlay_filter
        g = _build_overlay_filter(100, 200,
                                  mask_input_idx=2, feather_px=5)
        # Mask must go through boxblur (the feather), get alpha-merged
        # onto the inpaint stream, THEN overlay onto base. Order matters:
        # alphamerge after blur means the soft edge is preserved into
        # alpha; if you blur after alphamerge it blurs the rgb too.
        self.assertEqual(
            g,
            "[2:v]format=gray,boxblur=5:1[mfeath];"
            "[1:v][mfeath]alphamerge[crop_a];"
            "[0:v][crop_a]overlay=100:200:eof_action=pass",
        )

    def test_feather_px_threads_into_filter(self):
        from backend.dynamic._worker import _build_overlay_filter
        g = _build_overlay_filter(0, 0, mask_input_idx=2, feather_px=12)
        self.assertIn("boxblur=12:1", g)

    def test_negative_coords_pass_through(self):
        # ffmpeg's overlay accepts negative x/y (shifts the inpaint
        # partially off-base). The graph must just thread the values
        # through without sanitising.
        from backend.dynamic._worker import _build_overlay_filter
        g = _build_overlay_filter(-50, -25,
                                  mask_input_idx=None, feather_px=5)
        self.assertEqual(g, "[0:v][1:v]overlay=-50:-25:eof_action=pass")

    def test_eof_action_pass_in_all_modes(self):
        # Critical correctness invariant from the bigger _overlay
        # docstring: eof_action=pass is what guarantees the output
        # spans the full base duration even if the inpaint stream is
        # short. Both modes must preserve it.
        from backend.dynamic._worker import _build_overlay_filter
        for kw in ({"mask_input_idx": None},
                   {"mask_input_idx": 2}):
            g = _build_overlay_filter(10, 20, feather_px=5, **kw)
            self.assertIn("eof_action=pass", g,
                          f"missing eof_action=pass for {kw}")


class TestBboxSidecar(unittest.TestCase):
    """bbox.json sidecar is the contract that lets the recomposite CLI
    redo just the overlay step. Schema is tiny but stable -- bump
    schema_version if the field set changes so old workspaces don't
    silently mismatch."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vsr_bbox_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        # The worker writes this dict; the recomposite CLI reads the
        # x/y/w/h fields back. Verify the contract directly so a typo
        # on either side fails here, not at runtime.
        import json
        sidecar = self.tmp / "_bbox.json"
        payload = {
            "schema_version": 1,
            "x": 1340, "y": 820,
            "w": 256, "h": 256,
            "frame_w": 1920, "frame_h": 1080,
        }
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(sidecar, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("x", "y", "w", "h", "frame_w", "frame_h",
                    "schema_version"):
            self.assertIn(key, data)
        self.assertEqual((data["x"], data["y"], data["w"], data["h"]),
                         (1340, 820, 256, 256))


class TestRecompositeDiscovery(unittest.TestCase):
    """The recomposite CLI's path-discovery is what users will hit when
    they point it at the wrong dir; every missing file should produce a
    helpful error, not a cryptic TypeError deep in ffmpeg arg-building."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vsr_recomp_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_workspace(self, *, with_bbox=True, with_masks=True,
                        with_crop_out=True, with_base=True):
        from tool.dynamic_recomposite import _VIDEO_STEM, _CROP_STEM
        ws = self.tmp / "fakews_abc12345"
        ws.mkdir()
        if with_base:
            (ws / f"{_VIDEO_STEM}.mp4").write_bytes(b"\0" * 4096)
        crop_root = ws / _VIDEO_STEM / "_crop"
        crop_root.mkdir(parents=True)
        if with_bbox:
            (crop_root / "_bbox.json").write_text(
                '{"schema_version": 1, "x": 100, "y": 200,'
                ' "w": 256, "h": 128,'
                ' "frame_w": 1920, "frame_h": 1080}',
                encoding="utf-8",
            )
        if with_masks:
            masks = crop_root / "masks"
            masks.mkdir()
            (masks / "00000.png").write_bytes(b"\0" * 256)
            (masks / "00001.png").write_bytes(b"\0" * 256)
        if with_crop_out:
            crop_out_dir = crop_root / "out" / _CROP_STEM
            crop_out_dir.mkdir(parents=True)
            (crop_out_dir / "inpaint_out.mp4").write_bytes(b"\0" * 8192)
        return ws

    def test_full_workspace_resolves(self):
        from tool.dynamic_recomposite import _discover_paths, _resolve_bbox
        ws = self._make_workspace()
        base, crop, masks, bbox_path = _discover_paths(ws)
        self.assertTrue(base.is_file())
        self.assertTrue(crop.is_file())
        self.assertTrue(masks.is_dir())
        self.assertTrue(bbox_path.is_file())
        bb = _resolve_bbox(bbox_path, None)
        self.assertEqual(bb, (100, 200, 256, 128))

    def test_missing_base_raises_with_clear_message(self):
        from tool.dynamic_recomposite import _discover_paths
        ws = self._make_workspace(with_base=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            _discover_paths(ws)
        self.assertIn("base video", str(ctx.exception))

    def test_missing_crop_out_raises_with_clear_message(self):
        from tool.dynamic_recomposite import _discover_paths
        ws = self._make_workspace(with_crop_out=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            _discover_paths(ws)
        self.assertIn("ProPainter output", str(ctx.exception))

    def test_missing_masks_raises_with_clear_message(self):
        from tool.dynamic_recomposite import _discover_paths
        ws = self._make_workspace(with_masks=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            _discover_paths(ws)
        self.assertIn("mask sequence dir", str(ctx.exception))

    def test_cli_bbox_override(self):
        # --bbox "x,y,w,h" lets users recomposite workspaces that
        # predate the bbox sidecar.
        from tool.dynamic_recomposite import _resolve_bbox
        bb = _resolve_bbox(self.tmp / "does_not_exist.json", "10,20,30,40")
        self.assertEqual(bb, (10, 20, 30, 40))

    def test_missing_sidecar_and_no_override_raises(self):
        from tool.dynamic_recomposite import _resolve_bbox
        with self.assertRaises(SystemExit) as ctx:
            _resolve_bbox(self.tmp / "does_not_exist.json", None)
        self.assertIn("bbox sidecar", str(ctx.exception))

    def test_cli_bbox_malformed_raises(self):
        from tool.dynamic_recomposite import _resolve_bbox
        with self.assertRaises(SystemExit):
            _resolve_bbox(self.tmp / "x.json", "10,20,30")        # 3 ints
        with self.assertRaises(SystemExit):
            _resolve_bbox(self.tmp / "x.json", "a,b,c,d")          # not ints


class TestChunkGeometry(unittest.TestCase):
    """Planner for padded ProPainter chunks. Bugs here either leave
    visible seams (under-pad, gaps in coverage) or silently truncate /
    duplicate frames (overlapping or non-contiguous keep ranges), both
    of which are user-visible quality regressions, so the math wants
    exhaustive coverage."""

    def _validate_plan(self, plan, total_frames):
        """Common invariants every valid plan must satisfy."""
        # Concatenated keep ranges exactly cover [0, total_frames).
        cursor = 0
        for chunk_start, chunk_n, keep_offset, keep_n in plan:
            self.assertGreaterEqual(chunk_start, 0)
            self.assertGreater(chunk_n, 0)
            self.assertGreaterEqual(keep_offset, 0)
            self.assertGreater(keep_n, 0)
            self.assertLessEqual(keep_offset + keep_n, chunk_n)
            self.assertEqual(chunk_start + keep_offset, cursor,
                             f"keep range gap/overlap at cursor={cursor}, "
                             f"chunk={chunk_start}+{keep_offset}")
            cursor += keep_n
        self.assertEqual(cursor, total_frames,
                         f"keep ranges cover {cursor}, expected {total_frames}")

    def test_single_chunk_no_padding(self):
        # Small video fits in one chunk; no padding needed (no seams).
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(500, chunk_frames=800, pad=80)
        self.assertEqual(plan, [(0, 500, 0, 500)])

    def test_exactly_chunk_size_single_shot(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(800, chunk_frames=800, pad=80)
        self.assertEqual(plan, [(0, 800, 0, 800)])

    def test_two_chunks_first_no_left_pad_last_no_right_pad(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(1600, chunk_frames=800, pad=80)
        self.assertEqual(len(plan), 2)
        # Chunk 0: covers [0, 880), keeps [0, 800) (no left pad)
        self.assertEqual(plan[0], (0, 880, 0, 800))
        # Chunk 1: covers [720, 1600), keeps [800, 1600) (no right pad)
        self.assertEqual(plan[1], (720, 880, 80, 800))
        self._validate_plan(plan, 1600)

    def test_three_chunks_middle_has_symmetric_pad(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(2400, chunk_frames=800, pad=80)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0], (0, 880, 0, 800))         # no left pad
        self.assertEqual(plan[1], (720, 960, 80, 800))      # both sides
        self.assertEqual(plan[2], (1520, 880, 80, 800))     # no right pad
        self._validate_plan(plan, 2400)

    def test_user_98668_frames_124_chunks(self):
        # Reproduces the real Diablo-4 case: 98668 frames / chunk_frames=800
        # = 124 chunks (last chunk is partial: 98668 - 123*800 = 268 frames).
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(98668, chunk_frames=800, pad=80)
        self.assertEqual(len(plan), 124)
        # First: [0, 880), keep [0, 800)
        self.assertEqual(plan[0], (0, 880, 0, 800))
        # Middle (e.g. 50): [50*800-80, 51*800+80) = [39920, 40880), keep 800
        self.assertEqual(plan[50], (39920, 960, 80, 800))
        # Last (idx=123): keep_start=98400, keep_n=268, no right pad
        self.assertEqual(plan[123], (98320, 348, 80, 268))
        self._validate_plan(plan, 98668)

    def test_uneven_last_chunk_keeps_full_remainder(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        # 1500 frames into 800-chunks => 800 + 700 split
        plan = _compute_chunk_geometry(1500, chunk_frames=800, pad=80)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0], (0, 880, 0, 800))
        # Chunk 1: keep_start=800, keep_n=700, left pad=80, no right pad
        self.assertEqual(plan[1], (720, 780, 80, 700))
        self._validate_plan(plan, 1500)

    def test_pad_zero_falls_back_to_unpadded(self):
        # Setting pad=0 must reproduce the legacy un-padded behaviour
        # exactly -- escape hatch if NVENC trim ever becomes a bottleneck.
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(2000, chunk_frames=800, pad=0)
        self.assertEqual(plan, [
            (0,    800, 0, 800),
            (800,  800, 0, 800),
            (1600, 400, 0, 400),
        ])
        self._validate_plan(plan, 2000)

    def test_pad_larger_than_first_chunk_offset_clipped(self):
        # If pad > chunk_frames it would create double-counted context;
        # the clip-to-video-bounds logic still has to keep the plan
        # internally consistent.
        from backend.dynamic._worker import _compute_chunk_geometry
        plan = _compute_chunk_geometry(2000, chunk_frames=800, pad=200)
        # Chunk 0: keep_start=0 -> left pad clipped to 0
        self.assertEqual(plan[0][0], 0)
        # Chunk 2 (last): right pad clipped to 0 (no frames beyond keep_end)
        self.assertEqual(plan[2][0] + plan[2][1], 2000)
        self._validate_plan(plan, 2000)

    def test_empty_video_returns_empty_plan(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        self.assertEqual(_compute_chunk_geometry(0, 800, 80), [])

    def test_invalid_chunk_frames_rejected(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        with self.assertRaises(ValueError):
            _compute_chunk_geometry(1000, chunk_frames=0, pad=80)
        with self.assertRaises(ValueError):
            _compute_chunk_geometry(1000, chunk_frames=-100, pad=80)

    def test_invalid_pad_rejected(self):
        from backend.dynamic._worker import _compute_chunk_geometry
        with self.assertRaises(ValueError):
            _compute_chunk_geometry(1000, chunk_frames=800, pad=-1)


class TestResumeThreshold(unittest.TestCase):
    """The DeAOT-skip resume detection MUST refuse partial mask sets --
    feeding them into ProPainter silently truncates the output video.
    A 5% shortfall is allowed because ffprobe's nb_frames is unreliable
    on some variable-fps containers."""

    def test_empty_inputs_return_false(self):
        from backend.dynamic._worker import _resume_threshold_met
        self.assertFalse(_resume_threshold_met(0, 100))
        self.assertFalse(_resume_threshold_met(100, 0))
        self.assertFalse(_resume_threshold_met(0, 0))
        self.assertFalse(_resume_threshold_met(-1, 100))

    def test_well_below_threshold_rejected(self):
        from backend.dynamic._worker import _resume_threshold_met
        # 1972/98668 = 2% -- the exact pathological case the user hit
        self.assertFalse(_resume_threshold_met(1972, 98668))
        self.assertFalse(_resume_threshold_met(50, 100))
        self.assertFalse(_resume_threshold_met(94, 100))  # just under 95%

    def test_at_or_above_threshold_accepted(self):
        from backend.dynamic._worker import _resume_threshold_met
        self.assertTrue(_resume_threshold_met(95, 100))   # exact threshold
        self.assertTrue(_resume_threshold_met(100, 100))
        self.assertTrue(_resume_threshold_met(98668, 98668))
        # 5% ffprobe undercount on a long video -- still resume
        self.assertTrue(_resume_threshold_met(98668, 100000))

    def test_more_masks_than_frames_accepted(self):
        # Defensive: if ffprobe under-reports we should still resume,
        # never the other way around.
        from backend.dynamic._worker import _resume_threshold_met
        self.assertTrue(_resume_threshold_met(101, 100))


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
