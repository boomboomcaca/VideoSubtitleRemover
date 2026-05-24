"""
Experimental dynamic watermark removal.

This subpackage adds support for *moving* / *deforming* watermarks
(logos, overlays, tracked objects) on top of the existing static-region
subtitle pipeline. It is intentionally additive -- nothing in
``backend/processor.py`` is changed by importing this package.

MVP (phase A) design
--------------------
We piggy-back on the sibling ``watermark_remover`` project's
SAM + DeAOT + ProPainter implementation rather than vendoring the
code. The wrapper module ``external_pipeline`` is the single point of
contact; phase B can swap it for an in-tree port without touching
the CLI or any future UI integration.
"""

from .external_pipeline import (
    PHASES,
    PHASE_WEIGHTS,
    ClickPoint,
    DynamicRemovalResult,
    ProgressCallback,
    parse_clicks,
    phase_to_overall,
    resolve_watermark_remover_path,
    run_dynamic_removal,
)

__all__ = [
    "PHASES",
    "PHASE_WEIGHTS",
    "ClickPoint",
    "DynamicRemovalResult",
    "ProgressCallback",
    "parse_clicks",
    "phase_to_overall",
    "resolve_watermark_remover_path",
    "run_dynamic_removal",
]
