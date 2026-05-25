"""
Wipe a dynamic-mode workspace so the next run redoes every stage from
scratch. Useful when:

  * Click points changed -- the cached DeAOT masks are now stale
  * The cached state looks suspicious / corrupted
  * You want to verify the full pipeline end-to-end without resume

Workspace path is derived the same way the production pipeline derives
it (``<repo>/output/dynamic/<stem>_<sha1(abs_path)[:8]>``), so passing
``--video`` is equivalent to looking up the workspace by hand.

The same helpers used here back the GUI's pre-launch cache-confirm
dialog -- both code paths import :mod:`backend.dynamic.workspace`.

Usage::

    # List all dynamic-mode workspaces with size + stage state
    python -m tool.dynamic_clean_workspace --list

    # Wipe everything for a specific source video (asks before deleting)
    python -m tool.dynamic_clean_workspace --video "<original.mp4>"

    # Wipe a workspace by its directory path
    python -m tool.dynamic_clean_workspace --workspace "<dir>"

    # Skip the y/n prompt
    python -m tool.dynamic_clean_workspace --video "<...>" --yes

    # Preserve the reencoded intermediate (saves 1-3 min on next run)
    python -m tool.dynamic_clean_workspace --video "<...>" \\
        --keep-source-clean --yes

    # Only nuke ProPainter chunks so ProPainter redoes them but the
    # earlier stages (DeAOT, cleanup, crop) stay cached
    python -m tool.dynamic_clean_workspace --video "<...>" \\
        --only-propainter --yes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repo root on sys.path so `backend.dynamic.workspace` is importable
# when the tool is run as ``python tool/dynamic_clean_workspace.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from backend.dynamic.workspace import (  # noqa: E402
    DYNAMIC_ROOT,
    describe_workspace,
    format_size,
    wipe_workspace,
    workspace_for_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("clean")


def _format_state_line(ws: Path) -> str:
    s = describe_workspace(ws)
    if not s.exists:
        return f"{ws.name}: (missing)"
    parts = [
        f"intermediate={'yes' if s.intermediate_exists else 'no'}",
        f"masks={s.n_masks}",
        f"sidecar={'yes' if s.cleanup_sidecar_exists else 'no'}",
        f"crop_mp4={'yes' if s.crop_video_exists else 'no'}",
        f"crop_masks={s.n_crop_masks}",
        f"chunks_done={s.n_chunks_done}",
    ]
    return f"{ws.name}: [{', '.join(parts)}] {format_size(s.size_bytes)}"


def _list_workspaces() -> int:
    if not DYNAMIC_ROOT.is_dir():
        log.info("No dynamic workspaces directory found at %s", DYNAMIC_ROOT)
        return 0
    rows = sorted(
        (p for p in DYNAMIC_ROOT.iterdir()
         if p.is_dir() and not p.name.startswith("_")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not rows:
        log.info("No workspaces under %s", DYNAMIC_ROOT)
        return 0
    print(f"Workspaces in {DYNAMIC_ROOT}:")
    for ws in rows:
        print(f"  {_format_state_line(ws)}")
    return 0


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--video", type=Path,
                   help="Source video; the workspace is derived from its path hash.")
    g.add_argument("--workspace", type=Path,
                   help="Workspace directory to wipe.")
    g.add_argument("--list", action="store_true",
                   help="List all dynamic-mode workspaces and exit.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the y/n confirmation prompt.")
    p.add_argument("--keep-source-clean", action="store_true",
                   help="Preserve the reencoded intermediate; saves 1-3 min "
                        "on the next run. Ignored with --only-propainter.")
    p.add_argument("--only-propainter", action="store_true",
                   help="Only wipe ProPainter chunks + final concat; keep "
                        "DeAOT masks + crop. Use when ProPainter output "
                        "looks wrong but earlier stages are fine.")
    args = p.parse_args(argv)

    if args.list:
        return _list_workspaces()

    if args.video is not None:
        if not args.video.is_file():
            log.error("Video not found: %s", args.video)
            return 2
        ws = workspace_for_video(args.video)
    elif args.workspace is not None:
        ws = args.workspace.resolve()
    else:
        p.print_help()
        return 1

    if not ws.is_dir():
        log.error("Workspace does not exist: %s", ws)
        return 2

    print("About to wipe workspace:")
    print(f"  {_format_state_line(ws)}")
    if args.only_propainter:
        print("  (only ProPainter chunks + final concat)")
    elif args.keep_source_clean:
        print("  (keeping _source_clean.mp4)")
    else:
        print("  (full wipe)")

    if not args.yes and not _confirm("Continue?"):
        log.info("Aborted.")
        return 0

    ok = wipe_workspace(
        ws,
        keep_source_clean=args.keep_source_clean,
        only_propainter=args.only_propainter,
    )
    if ok:
        log.info("Wiped %s", ws)
        return 0
    log.error("Wipe completed with errors; some artefacts may remain in %s", ws)
    return 1


if __name__ == "__main__":
    sys.exit(main())
