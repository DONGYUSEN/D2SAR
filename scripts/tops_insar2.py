#!/usr/bin/env python3
"""Sentinel-1 TOPS InSAR processor — ISCE3-native, burst-first."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Step 1: sys.modules poison — blocks strip backends from any import path ──
BLOCK_GUARDS: frozenset[str] = frozenset({
    "strip_insar", "strip_insar2",
    "scripts.strip_insar", "scripts.strip_insar2",
    "tops_insar",
})
for _name in BLOCK_GUARDS:
    sys.modules[_name] = type(sys)("blocked")  # pragma: no cover


# ── Step 2: AST check — verify no tops_*.py imports a strip backend ────────────
# Legacy files (tops_insar.py) that already exist in the repo may import strip
# backends; they are excluded from this scan.  Only NEW tops_insar2-*.py modules
# must comply with the zero-import constraint.
_LEGACY_EXCLUDED = frozenset({"tops_insar.py"})


def _check_no_forbidden_imports() -> None:
    import ast
    for path in sorted(Path("scripts").glob("tops_*.py")):
        if path.name in _LEGACY_EXCLUDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in BLOCK_GUARDS:
                        raise AssertionError(
                            f"{path}: 'import {alias.name}' is forbidden"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level  # 0 = absolute, 1+ = relative
                if module in BLOCK_GUARDS or module.startswith("strip"):
                    raise AssertionError(
                        f"{path}: 'from {module} import ...' is forbidden"
                    )
                # Relative import inside a package — resolve "from .strip_…" style
                if level > 0:
                    raise AssertionError(
                        f"{path}: relative import 'from .{module} import ...' is forbidden"
                    )


STAGE_SEQUENCE: list[str] = [
    "check",
    "preprocess",
    "common_bursts",
    "topo",
    "subset_overlaps",
    "coarse_resamp",
    "overlap_ifg",
    "prep_esd",
    "esd",
    "range_coreg",
    "fine_resamp",
    "burst_ifg",
    "merge_bursts",
    "filter",
    "unwrap",
    "geocode",
    "publish",
]


def main(argv: list[str] | None = None) -> int:
    _check_no_forbidden_imports()

    parser = argparse.ArgumentParser(
        description="Sentinel-1 TOPS InSAR — ISCE3-native, burst-first processor",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("master_safe_or_manifest", type=Path)
    parser.add_argument("slave_safe_or_manifest", type=Path)
    parser.add_argument(
        "--swath",
        default="all",
        help="Comma-separated IW swaths (IW1,IW2,IW3) or 'all' (default: all)",
    )
    parser.add_argument(
        "--start-stage",
        default="check",
        choices=STAGE_SEQUENCE,
        help="First pipeline stage to run (default: check)",
    )
    parser.add_argument(
        "--end-stage",
        default="publish",
        choices=STAGE_SEQUENCE,
        help="Last pipeline stage to run (default: publish)",
    )
    parser.add_argument("--dem", type=Path)
    parser.add_argument(
        "--resolution-meters", type=float, default=20.0,
        help="Output ground resolution in metres (default: 20.0)",
    )
    parser.add_argument(
        "--range-looks", type=int, default=1,
        help="Number of range looks for multi-looking (default: 1)",
    )
    parser.add_argument(
        "--azimuth-looks", type=int, default=1,
        help="Number of azimuth looks for multi-looking (default: 1)",
    )
    parser.add_argument(
        "--unwrap-method",
        default="icu",
        choices=["icu", "snaphu", "dolphin"],
        help="Unwrapping method (default: icu)",
    )
    parser.add_argument(
        "--extra-esd-cycles", type=float, default=0.0,
        help="Extra integer phase cycles to add to ESD offset (default: 0.0)",
    )
    parser.add_argument(
        "--esd-coherence-threshold", type=float, default=0.85,
        help="Coherence mask threshold for ESD estimation (default: 0.85)",
    )
    parser.add_argument(
        "--do-ionospheric-correction", action="store_true",
        help="Enable split-band ionospheric phase correction",
    )
    parser.add_argument(
        "--gpu-mode",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="GPU acceleration mode (default: auto)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    log = logging.getLogger("tops_insar2")

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    stages = _build_stage_sequence(args.start_stage, args.end_stage)
    swaths = _resolve_swaths(args.swath)

    log.info("tops_insar2 starting — output=%s master=%s slave=%s",
             args.output_dir, args.master_safe_or_manifest,
             args.slave_safe_or_manifest)
    log.info("Stages: %s | Swaths: %s", stages, swaths)

    for swath in swaths:
        log.info("Processing swath %s", swath)
        _run_swath(args, swath, stages)

    log.info("tops_insar2 complete: %s", args.output_dir)
    return 0


def _build_stage_sequence(start: str, end: str) -> list[str]:
    """Return the inclusive sub-sequence of stages from start to end."""
    s_idx = STAGE_SEQUENCE.index(start)
    e_idx = STAGE_SEQUENCE.index(end)
    return STAGE_SEQUENCE[s_idx:e_idx + 1]


def _resolve_swaths(sel: str) -> list[str]:
    """Return the list of IW swaths to process."""
    if sel == "all":
        return ["IW1", "IW2", "IW3"]
    return sel.split(",")


def _run_swath(args, swath: str, stages: list[str]) -> None:
    """Run the full pipeline for one swath.

    Placeholder — each stage is implemented in a dedicated tops_*.py module.
    """
    raise NotImplementedError(
        f"Swath runner not yet implemented (swath={swath}, stages={stages})"
    )


if __name__ == "__main__":
    sys.exit(main())
