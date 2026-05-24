"""
Tk UI for the dynamic watermark removal pipeline.

The main VSR Pro app spawns ``DynamicWatermarkWindow`` as a ``Toplevel``
when the user clicks the "Watermark Mode" button. The window is fully
self-contained: file picker, first-frame click canvas, run button,
threaded pipeline execution, and progress display. It reuses the host
app's ``Theme`` and modern widgets when available (passed in via the
``theme`` and ``widgets`` constructor args), but falls back to plain
``tk.Button`` / ``tk.Frame`` so this module can be imported and
smoke-tested in isolation.

Public API
----------
``DynamicWatermarkWindow(master, *, theme=None, widgets=None,
                         initial_video=None, wm_path=None)``

The window manages its own pipeline thread; only one removal job can be
in flight at a time per window. Closing the window while a job is
running asks for confirmation.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from .external_pipeline import (
    DynamicRemovalResult,
    parse_clicks,
    resolve_watermark_remover_path,
    run_dynamic_removal,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Theme adapter
# --------------------------------------------------------------------------- #
# We accept an optional ``theme`` namespace so the panel matches the host
# app's styling. The fallback values keep the panel usable when imported
# standalone (e.g. ``python -m backend.dynamic.ui``).

class _DefaultTheme:
    BG_DARK = "#1a1d24"
    BG_SECONDARY = "#22262f"
    BG_TERTIARY = "#2a2f3a"
    BG_RAISED = "#2f3441"
    TEXT_PRIMARY = "#e8eaed"
    TEXT_SECONDARY = "#b0b6c1"
    TEXT_MUTED = "#7a8090"
    ACCENT = "#58a6ff"
    SUCCESS = "#4ade80"
    WARNING = "#fbbf24"
    ERROR = "#f87171"
    S_XS = 4
    S_SM = 8
    S_MD = 12
    S_LG = 16
    S_XL = 24
    # Some host themes (notably VSR Pro's main Theme class) store font
    # entries as bare integer sizes; others as ``(family, size)`` tuples.
    # We use ints here so the _font() normalizer takes the same path
    # regardless of who supplied the theme.
    F_TITLE = 16
    F_HEADING = 12
    F_BODY = 10
    F_META = 9


class _ThemeAdapter:
    """Attribute proxy that falls back to ``_DefaultTheme`` on misses.

    The host app's ``Theme`` class doesn't necessarily expose every
    attribute we reference (colours like ``ACCENT`` / ``SUCCESS``,
    spacing tokens like ``S_XS``...). Wrapping it lets us use a
    consistent ``t.FOO`` syntax in ``_build_layout()`` and silently
    fall back to our local default when the host omits an attribute,
    instead of crashing with ``AttributeError`` at window-open time.
    """

    def __init__(self, host=None):
        self._host = host

    def __getattr__(self, name):
        # __getattr__ is only invoked on misses for normal attribute
        # lookup, so this code path is the fallback chain itself.
        if self._host is not None:
            try:
                return getattr(self._host, name)
            except AttributeError:
                pass
        return getattr(_DefaultTheme, name)


def _font(attr, *modifiers) -> tuple:
    """Normalise a theme font attribute into a Tk-compatible font tuple.

    The host app may set the ``F_*`` theme attributes as either:

    * an **int** (just a point size; we supply a default family), or
    * a **tuple** like ``("Segoe UI", 12)`` (use as-is), or
    * a **string** font family name (we supply a default size).

    Any positional modifiers (``"bold"``, ``"italic"``, ...) are
    appended. This is the moral equivalent of VSR Pro's own ``f()``
    helper, kept local so this module stays self-contained.
    """
    if isinstance(attr, int):
        return ("Segoe UI", attr) + modifiers
    if isinstance(attr, tuple):
        return attr + modifiers
    if isinstance(attr, str):
        return (attr, 10) + modifiers
    return ("TkDefaultFont", 10) + modifiers


# --------------------------------------------------------------------------- #
# Clickable first-frame canvas
# --------------------------------------------------------------------------- #

class ClickPointsCanvas(tk.Canvas):
    """Canvas that displays the first video frame and collects SAM clicks.

    Left-click adds a positive point (green dot), right-click adds a
    negative point (red dot). The display is letterboxed -- the image
    is rendered at the largest size that fits inside the canvas, and
    click coordinates are translated back to original video pixel
    coordinates before being stored.

    State:
        clicks: list of (x, y, mode) tuples in *original* video
                coordinates. mode: 1=positive, 0=negative.
    """

    POS_COLOR = "#4ade80"  # green
    NEG_COLOR = "#f87171"  # red
    POS_OUTLINE = "#0d2818"
    NEG_OUTLINE = "#3a0808"

    def __init__(self, master, theme=None, on_change=None, **kw):
        theme = _ThemeAdapter(theme)
        super().__init__(
            master,
            bg=theme.BG_DARK,
            highlightthickness=1,
            highlightbackground=theme.BG_TERTIARY,
            cursor="crosshair",
            **kw,
        )
        self._theme = theme
        self._on_change = on_change  # called whenever click list changes
        self._frame_img = None       # PIL.Image of original first frame
        self._mask_img = None        # PIL.Image (L mode) of latest SAM mask at orig resolution
        self._photo = None           # ImageTk.PhotoImage (kept alive)
        self._draw_box = (0, 0, 1, 1)  # (x0, y0, w, h) on canvas where image is drawn
        self._orig_size = (1, 1)     # (W, H) of original frame
        self.clicks: List[Tuple[int, int, int]] = []  # (x, y, mode) in original coords

        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", lambda e: self._add_click(e.x, e.y, mode=1))
        self.bind("<Button-3>", lambda e: self._add_click(e.x, e.y, mode=0))

    # -- public --

    def set_frame(self, frame_bgr):
        """Set the displayed frame from an opencv BGR ndarray."""
        try:
            from PIL import Image, ImageTk
            import cv2
        except ImportError:
            logger.exception("PIL/cv2 unavailable; cannot display frame")
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_img = Image.fromarray(rgb)
        self._orig_size = (self._frame_img.width, self._frame_img.height)
        self.clicks = []
        self._mask_img = None
        self._render()
        if self._on_change:
            self._on_change(self.clicks)

    def clear_clicks(self):
        self.clicks = []
        self._mask_img = None
        self._render()
        if self._on_change:
            self._on_change(self.clicks)

    def set_mask(self, mask_uint8):
        """Show a SAM-predicted mask as a semi-transparent green overlay.

        Pass ``None`` to clear. ``mask_uint8`` is a 2-D numpy array
        (any size; we resize to original frame dimensions) where >0
        means inside the watermark region.
        """
        if mask_uint8 is None:
            self._mask_img = None
        else:
            try:
                from PIL import Image
                import numpy as np
            except ImportError:
                return
            arr = mask_uint8
            if arr.dtype != np.uint8:
                arr = (arr > 0).astype(np.uint8) * 255
            self._mask_img = Image.fromarray(arr, mode="L")
            # Resize mask up to original frame resolution if SAM returned
            # at a different size (it shouldn't, but be defensive).
            if self._mask_img.size != self._orig_size:
                self._mask_img = self._mask_img.resize(
                    self._orig_size, Image.NEAREST,
                )
        self._render()

    def undo_click(self):
        if not self.clicks:
            return
        self.clicks.pop()
        self._render()
        if self._on_change:
            self._on_change(self.clicks)

    # -- internals --

    def _on_resize(self, _event):
        self._render()

    def _add_click(self, cx, cy, mode):
        if self._frame_img is None:
            return
        # Translate canvas coords to original-image coords
        x0, y0, dw, dh = self._draw_box
        if not (x0 <= cx <= x0 + dw and y0 <= cy <= y0 + dh):
            return  # click outside the letterboxed image area
        ow, oh = self._orig_size
        ox = int((cx - x0) * ow / dw)
        oy = int((cy - y0) * oh / dh)
        self.clicks.append((ox, oy, mode))
        self._render()
        if self._on_change:
            self._on_change(self.clicks)

    def _render(self):
        self.delete("all")
        if self._frame_img is None:
            return
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        ow, oh = self._orig_size
        scale = min(cw / ow, ch / oh)
        dw, dh = max(1, int(ow * scale)), max(1, int(oh * scale))
        x0, y0 = (cw - dw) // 2, (ch - dh) // 2
        self._draw_box = (x0, y0, dw, dh)

        try:
            from PIL import Image, ImageTk
        except ImportError:
            return

        # Composite the SAM mask (if any) onto the frame BEFORE resizing
        # to display dimensions so the green stays sharp at the boundary.
        # Tint: same green as the positive-click dots, ~45% alpha.
        if self._mask_img is not None:
            base_rgba = self._frame_img.convert("RGBA")
            green = Image.new("RGBA", base_rgba.size, (74, 222, 128, 115))
            transparent = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
            overlay = Image.composite(green, transparent, self._mask_img)
            composed = Image.alpha_composite(base_rgba, overlay).convert("RGB")
        else:
            composed = self._frame_img

        resized = composed.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.create_image(x0, y0, anchor="nw", image=self._photo)

        # Draw click dots
        for ox, oy, mode in self.clicks:
            cx = x0 + ox * dw / ow
            cy = y0 + oy * dh / oh
            color = self.POS_COLOR if mode == 1 else self.NEG_COLOR
            outline = self.POS_OUTLINE if mode == 1 else self.NEG_OUTLINE
            r = 6
            self.create_oval(cx - r, cy - r, cx + r, cy + r,
                             fill=color, outline=outline, width=2)
            # tiny + / - marker
            if mode == 1:
                self.create_line(cx - 3, cy, cx + 3, cy, fill=outline, width=2)
                self.create_line(cx, cy - 3, cx, cy + 3, fill=outline, width=2)
            else:
                self.create_line(cx - 3, cy, cx + 3, cy, fill=outline, width=2)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class DynamicWatermarkWindow(tk.Toplevel):
    """Self-contained dynamic-watermark-removal workflow window."""

    def __init__(
        self,
        master=None,
        *,
        theme=None,
        initial_video: Optional[Path] = None,
        wm_path: Optional[str] = None,
    ):
        super().__init__(master)
        # Always wrap, even when the host passed something -- the host's
        # Theme class is not guaranteed to expose every attribute we use.
        self._theme = _ThemeAdapter(theme)
        self.title("Dynamic Watermark Removal (Experimental)")
        self.geometry("1100x780")
        self.minsize(900, 640)
        self.configure(bg=self._theme.BG_DARK)

        # State
        self._video_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._wm_path_override = wm_path
        self._wm_path: Optional[Path] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        # SAM preview wiring (lazy: created on first video load so the
        # process spawn cost isn't paid until the user actually needs it)
        self._sam_client = None        # SamPreviewClient
        self._sam_preview = None       # DebouncedSamPreview
        self._sam_setting_image = False  # guard against re-entrant set_image

        self._build_layout()

        # Initial wm_path resolution (non-fatal; status will warn)
        try:
            self._wm_path = resolve_watermark_remover_path(self._wm_path_override)
            self._set_status(f"watermark_remover: {self._wm_path}", tone="info")
        except FileNotFoundError as e:
            self._set_status(str(e), tone="error")

        if initial_video is not None:
            self._load_video(initial_video)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- layout ----------------

    def _build_layout(self):
        t = self._theme
        root_frame = tk.Frame(self, bg=t.BG_DARK)
        root_frame.pack(fill="both", expand=True, padx=t.S_LG, pady=t.S_LG)

        # Header
        header = tk.Frame(root_frame, bg=t.BG_DARK)
        header.pack(fill="x", pady=(0, t.S_MD))
        tk.Label(header, text="Dynamic Watermark Removal",
                 font=_font(t.F_TITLE, "bold"), bg=t.BG_DARK,
                 fg=t.TEXT_PRIMARY).pack(side="left")
        tk.Label(header, text="experimental",
                 font=_font(t.F_META, "italic"), bg=t.BG_DARK,
                 fg=t.TEXT_MUTED).pack(side="left", padx=(t.S_SM, 0), pady=(6, 0))

        # File row
        file_row = tk.Frame(root_frame, bg=t.BG_DARK)
        file_row.pack(fill="x", pady=(0, t.S_MD))
        tk.Label(file_row, text="Video:", font=_font(t.F_BODY), bg=t.BG_DARK,
                 fg=t.TEXT_SECONDARY, width=8, anchor="w").pack(side="left")
        self._video_label = tk.Label(
            file_row, text="(no file selected)", font=_font(t.F_BODY),
            bg=t.BG_TERTIARY, fg=t.TEXT_PRIMARY, anchor="w",
            padx=t.S_SM, pady=4,
        )
        self._video_label.pack(side="left", fill="x", expand=True, padx=t.S_SM)
        tk.Button(file_row, text="Browse...", font=_font(t.F_BODY),
                  command=self._pick_video,
                  bg=t.BG_RAISED, fg=t.TEXT_PRIMARY,
                  activebackground=t.ACCENT, activeforeground="#fff",
                  relief="flat", padx=t.S_MD, pady=4).pack(side="left")

        # Main split: left = canvas, right = controls
        body = tk.Frame(root_frame, bg=t.BG_DARK)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1, minsize=300)
        body.rowconfigure(0, weight=1)

        # Canvas pane
        canvas_pane = tk.Frame(body, bg=t.BG_DARK)
        canvas_pane.grid(row=0, column=0, sticky="nsew", padx=(0, t.S_MD))
        canvas_pane.rowconfigure(1, weight=1)
        canvas_pane.columnconfigure(0, weight=1)

        instructions = tk.Label(
            canvas_pane,
            text=(
                "Left-click = positive (this IS the watermark) - green\n"
                "Right-click = negative (this is NOT the watermark) - red\n"
                "Add 1-3 positives in the centre of the watermark; add a "
                "negative just outside if SAM bleeds into the background."
            ),
            font=_font(t.F_META), bg=t.BG_DARK, fg=t.TEXT_MUTED, justify="left",
        )
        instructions.grid(row=0, column=0, sticky="ew", pady=(0, t.S_SM))

        self._canvas = ClickPointsCanvas(
            canvas_pane, theme=t, on_change=self._on_clicks_changed,
        )
        self._canvas.grid(row=1, column=0, sticky="nsew")

        # Control pane
        ctrl = tk.Frame(body, bg=t.BG_SECONDARY)
        ctrl.grid(row=0, column=1, sticky="nsew")

        ctrl_inner = tk.Frame(ctrl, bg=t.BG_SECONDARY)
        ctrl_inner.pack(fill="both", expand=True, padx=t.S_MD, pady=t.S_MD)

        tk.Label(ctrl_inner, text="Clicks", font=_font(t.F_HEADING, "bold"),
                 bg=t.BG_SECONDARY, fg=t.TEXT_PRIMARY).pack(anchor="w")
        self._clicks_label = tk.Label(
            ctrl_inner, text="None yet", font=_font(t.F_BODY),
            bg=t.BG_SECONDARY, fg=t.TEXT_SECONDARY, anchor="w",
            justify="left", wraplength=260,
        )
        self._clicks_label.pack(anchor="w", pady=(t.S_XS, t.S_SM), fill="x")

        click_btn_row = tk.Frame(ctrl_inner, bg=t.BG_SECONDARY)
        click_btn_row.pack(fill="x", pady=(0, t.S_MD))
        tk.Button(click_btn_row, text="Undo", font=_font(t.F_BODY),
                  command=self._canvas.undo_click,
                  bg=t.BG_RAISED, fg=t.TEXT_PRIMARY, relief="flat",
                  padx=t.S_SM, pady=2).pack(side="left", padx=(0, t.S_XS))
        tk.Button(click_btn_row, text="Clear all", font=_font(t.F_BODY),
                  command=self._canvas.clear_clicks,
                  bg=t.BG_RAISED, fg=t.TEXT_PRIMARY, relief="flat",
                  padx=t.S_SM, pady=2).pack(side="left")

        # Output controls
        tk.Label(ctrl_inner, text="Output", font=_font(t.F_HEADING, "bold"),
                 bg=t.BG_SECONDARY, fg=t.TEXT_PRIMARY).pack(anchor="w",
                                                            pady=(t.S_SM, 0))
        out_row = tk.Frame(ctrl_inner, bg=t.BG_SECONDARY)
        out_row.pack(fill="x", pady=t.S_XS)
        self._output_label = tk.Label(
            out_row, text="(auto: alongside source)", font=_font(t.F_META),
            bg=t.BG_TERTIARY, fg=t.TEXT_PRIMARY, anchor="w",
            padx=t.S_SM, pady=4,
        )
        self._output_label.pack(side="left", fill="x", expand=True)
        tk.Button(out_row, text="...", font=_font(t.F_BODY), command=self._pick_output,
                  bg=t.BG_RAISED, fg=t.TEXT_PRIMARY, relief="flat",
                  padx=t.S_SM, pady=2).pack(side="left", padx=(t.S_XS, 0))

        # Options
        tk.Label(ctrl_inner, text="Options", font=_font(t.F_HEADING, "bold"),
                 bg=t.BG_SECONDARY, fg=t.TEXT_PRIMARY).pack(anchor="w",
                                                            pady=(t.S_MD, t.S_XS))
        self._auto_crop_var = tk.BooleanVar(value=True)
        self._fp16_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            ctrl_inner, text="Auto-crop (5-10x faster)",
            variable=self._auto_crop_var, font=_font(t.F_BODY),
            bg=t.BG_SECONDARY, fg=t.TEXT_PRIMARY,
            activebackground=t.BG_SECONDARY, activeforeground=t.TEXT_PRIMARY,
            selectcolor=t.BG_TERTIARY, anchor="w",
        ).pack(anchor="w", fill="x")
        tk.Checkbutton(
            ctrl_inner, text="FP16 (halve VRAM)",
            variable=self._fp16_var, font=_font(t.F_BODY),
            bg=t.BG_SECONDARY, fg=t.TEXT_PRIMARY,
            activebackground=t.BG_SECONDARY, activeforeground=t.TEXT_PRIMARY,
            selectcolor=t.BG_TERTIARY, anchor="w",
        ).pack(anchor="w", fill="x")

        # Run button -- bottom of control pane, big and obvious
        ctrl_inner.pack_configure(fill="both", expand=True)

        run_row = tk.Frame(ctrl_inner, bg=t.BG_SECONDARY)
        run_row.pack(side="bottom", fill="x", pady=(t.S_LG, 0))
        self._run_btn = tk.Button(
            run_row, text="Run", font=_font(t.F_HEADING, "bold"),
            command=self._on_run, bg=t.ACCENT, fg="#ffffff",
            activebackground=t.SUCCESS, activeforeground="#ffffff",
            relief="flat", padx=t.S_LG, pady=t.S_SM, state="disabled",
        )
        self._run_btn.pack(fill="x")
        self._reveal_btn = tk.Button(
            run_row, text="Open output folder", font=_font(t.F_BODY),
            command=self._reveal_output, bg=t.BG_RAISED, fg=t.TEXT_PRIMARY,
            relief="flat", padx=t.S_MD, pady=4, state="disabled",
        )
        self._reveal_btn.pack(fill="x", pady=(t.S_XS, 0))

        # Status + progress
        status_row = tk.Frame(root_frame, bg=t.BG_DARK)
        status_row.pack(fill="x", pady=(t.S_MD, 0))
        self._status_label = tk.Label(
            status_row, text="Select a video to begin.", font=_font(t.F_BODY),
            bg=t.BG_DARK, fg=t.TEXT_SECONDARY, anchor="w",
        )
        self._status_label.pack(fill="x")

        self._progress = ProgressBar(root_frame, theme=t)
        self._progress.pack(fill="x", pady=(t.S_SM, 0))

    # ---------------- file pickers ----------------

    def _pick_video(self):
        path = tk.filedialog.askopenfilename(
            parent=self,
            title="Choose video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                       ("All files", "*.*")],
        )
        if path:
            self._load_video(Path(path))

    def _pick_output(self):
        if self._video_path is None:
            return
        path = tk.filedialog.asksaveasfilename(
            parent=self,
            title="Save cleaned video",
            initialdir=str(self._video_path.parent),
            initialfile=f"{self._video_path.stem}_clean.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        if path:
            self._output_path = Path(path)
            self._output_label.config(text=str(self._output_path))

    def _load_video(self, path: Path):
        try:
            import cv2
        except ImportError:
            self._set_status("opencv not installed; cannot preview frame.",
                             tone="error")
            return
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            self._set_status(f"Cannot read first frame of {path.name}",
                             tone="error")
            return
        self._video_path = path
        self._video_label.config(text=str(path))
        self._canvas.set_frame(frame)
        self._set_status(
            f"Loaded {path.name} -- click on the watermark to mark it.",
            tone="info",
        )
        # Refresh default output path
        self._output_path = None
        self._output_label.config(text="(auto: alongside source)")

        # Kick SAM preview on a background thread: spawn the worker
        # (if not already) and feed it the first frame so subsequent
        # click predictions are sub-second.
        if self._wm_path is not None:
            self._kick_sam_preview_for_frame(frame)

    def _kick_sam_preview_for_frame(self, frame_bgr):
        """Spawn (or reuse) the SAM preview worker and hand it the new frame.

        Runs the spawn + set_image in a background thread so the GUI
        stays responsive across the ~3-5 s SAM startup cost. Sets a
        guard so reloading another video while the first hand-off is
        still in flight doesn't race."""
        if self._sam_setting_image:
            return
        self._sam_setting_image = True

        def _bg():
            try:
                if self._sam_client is None:
                    from .sam_preview import (
                        SamPreviewClient, DebouncedSamPreview,
                    )
                    self._sam_client = SamPreviewClient(self._wm_path)
                    self._sam_preview = DebouncedSamPreview(
                        self._sam_client,
                        on_mask=self._on_sam_mask_ready,
                    )
                    self.after(0, lambda: self._set_status(
                        "Loading SAM (one-time, ~5s)...", tone="info"))
                self._sam_client.set_image(frame_bgr)
                self.after(0, lambda: self._set_status(
                    "SAM ready -- click to preview the mask.", tone="info"))
            except Exception as e:  # noqa: BLE001
                logger.exception("SAM preview init failed")
                self.after(0, lambda: self._set_status(
                    f"SAM preview unavailable: {e}", tone="warn"))
            finally:
                self._sam_setting_image = False

        threading.Thread(target=_bg, daemon=True).start()

    def _on_sam_mask_ready(self, mask):
        """Callback from DebouncedSamPreview (runs on its worker thread)."""
        # Marshal into Tk thread before touching the canvas
        self.after(0, lambda: self._canvas.set_mask(mask))

    # ---------------- click handling ----------------

    def _on_clicks_changed(self, clicks):
        n_pos = sum(1 for c in clicks if c[2] == 1)
        n_neg = sum(1 for c in clicks if c[2] == 0)
        if not clicks:
            self._clicks_label.config(text="None yet")
        else:
            preview = "; ".join(
                f"({x},{y}){'+' if m == 1 else '-'}" for x, y, m in clicks[-3:]
            )
            self._clicks_label.config(
                text=f"{n_pos} positive, {n_neg} negative\nlast: {preview}",
            )
        # Enable Run when at least one positive click + video loaded + wm path resolved
        ready = (n_pos >= 1 and self._video_path is not None
                 and self._wm_path is not None
                 and self._worker_thread is None)
        self._run_btn.config(state="normal" if ready else "disabled")

        # Ask SAM for a fresh mask preview. If no positives, clear the
        # overlay (SAM needs at least one positive point to be meaningful).
        if self._sam_preview is not None:
            if n_pos >= 1:
                coords = [(c[0], c[1]) for c in clicks]
                modes = [c[2] for c in clicks]
                self._sam_preview.request(coords, modes)
            else:
                self._sam_preview.clear()

    # ---------------- run pipeline ----------------

    def _on_run(self):
        if self._worker_thread is not None:
            return
        if self._video_path is None or self._wm_path is None:
            return
        clicks = list(self._canvas.clicks)
        if not any(c[2] == 1 for c in clicks):
            self._set_status("Need at least one positive (left) click.",
                             tone="warn")
            return

        out_path = self._output_path or self._video_path.with_name(
            f"{self._video_path.stem}_clean.mp4"
        )
        self._output_path = out_path

        self._run_btn.config(state="disabled")
        self._reveal_btn.config(state="disabled")
        self._set_status("Starting pipeline...", tone="info")
        self._cancel_event.clear()

        self._worker_thread = threading.Thread(
            target=self._worker_main,
            args=(self._video_path, clicks, out_path, self._wm_path,
                  bool(self._auto_crop_var.get()),
                  bool(self._fp16_var.get())),
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_main(self, video, clicks, output, wm_path, auto_crop, fp16):
        try:
            def cb(phase, value, extra, overall):
                # Marshall back to UI thread
                self.after(0, lambda: self._progress.set(overall, phase, extra))

            result = run_dynamic_removal(
                video=video,
                clicks=clicks,
                output=output,
                wm_path=wm_path,
                auto_crop=auto_crop,
                fp16=fp16,
                progress_callback=cb,
            )
            self.after(0, lambda: self._on_success(result))
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            self.after(0, lambda: self._on_failure(e, tb))

    def _on_success(self, result: DynamicRemovalResult):
        self._worker_thread = None
        self._set_status(
            f"Done -> {result.output_video.name} "
            f"({result.output_video.stat().st_size / 1e6:.1f} MB)",
            tone="success",
        )
        self._progress.set(1.0, "done", "")
        self._reveal_btn.config(state="normal")
        # Re-evaluate run button readiness
        self._on_clicks_changed(self._canvas.clicks)

    def _on_failure(self, exc: Exception, tb: str):
        self._worker_thread = None
        logger.error("Dynamic watermark removal failed:\n%s", tb)
        self._set_status(f"Failed: {exc}", tone="error")
        tk.messagebox.showerror(
            "Dynamic watermark removal failed",
            f"{type(exc).__name__}: {exc}\n\nSee log for full traceback.",
            parent=self,
        )
        # Re-evaluate run button readiness
        self._on_clicks_changed(self._canvas.clicks)

    # ---------------- misc ----------------

    def _reveal_output(self):
        if self._output_path is None or not self._output_path.exists():
            return
        import subprocess
        subprocess.Popen(["explorer", "/select,", str(self._output_path)])

    def _set_status(self, text: str, tone: str = "info"):
        color = {
            "info":    self._theme.TEXT_SECONDARY,
            "success": self._theme.SUCCESS,
            "warn":    self._theme.WARNING,
            "error":   self._theme.ERROR,
        }.get(tone, self._theme.TEXT_SECONDARY)
        self._status_label.config(text=text, fg=color)

    def _on_close(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            ok = tk.messagebox.askyesno(
                "Cancel running job?",
                "A dynamic watermark removal job is still running. "
                "Closing the window will NOT stop the worker subprocess "
                "(it will keep running in the background until it finishes "
                "or you kill it manually). Close anyway?",
                parent=self,
            )
            if not ok:
                return
        # Tear down SAM preview worker
        try:
            if self._sam_preview is not None:
                self._sam_preview.stop()
            if self._sam_client is not None:
                self._sam_client.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error tearing down SAM preview")
        self.destroy()


# --------------------------------------------------------------------------- #
# Simple progress bar (avoids dependency on host app widgets)
# --------------------------------------------------------------------------- #

class ProgressBar(tk.Frame):
    """Two-line progress display: bar + textual phase label."""

    def __init__(self, master, theme=None, **kw):
        theme = _ThemeAdapter(theme)
        super().__init__(master, bg=theme.BG_DARK, **kw)
        self._theme = theme

        self._canvas = tk.Canvas(
            self, bg=theme.BG_TERTIARY, height=8,
            highlightthickness=0,
        )
        self._canvas.pack(fill="x")
        self._bar_id = self._canvas.create_rectangle(
            0, 0, 0, 8, fill=theme.ACCENT, width=0,
        )

        self._label = tk.Label(
            self, text="", font=_font(theme.F_META), bg=theme.BG_DARK,
            fg=theme.TEXT_MUTED, anchor="w",
        )
        self._label.pack(fill="x", pady=(2, 0))

        self._value = 0.0
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, value: float, phase: str = "", extra: str = ""):
        self._value = max(0.0, min(1.0, value))
        text = f"{int(self._value * 100):3d}%   {phase}"
        if extra:
            text += f"  ({extra})"
        self._label.config(text=text)
        self._redraw()

    def _redraw(self):
        w = max(1, self._canvas.winfo_width())
        self._canvas.coords(self._bar_id, 0, 0, int(self._value * w), 8)


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    root = tk.Tk()
    root.withdraw()
    win = DynamicWatermarkWindow(root)
    win.mainloop()
