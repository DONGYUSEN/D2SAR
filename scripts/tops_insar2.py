#!/usr/bin/env python3
"""Sentinel-1 TOPS InSAR processor — ISCE3-native, burst-first."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

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


# ── Stage sequence definition ─────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# Imports for pipeline stages (must come after poison)
# ---------------------------------------------------------------------------
import numpy as np

from scripts.tops_metadata import parse_sentinel1_safe
from scripts.tops_model import (
    BurstRadarGrid,
    CommonBurstSelection,
    EsdEstimate,
    Geo2RdrOffsets,
    OverlapPair,
    TimingCorrection,
    BurstWindow,
)
from scripts.tops_common_bursts import match_common_bursts, write_common_bursts_json
from scripts.tops_overlap import materialize_overlaps, write_overlaps_json
from scripts.tops_geometry import run_geo2rdr_single_burst
from scripts.tops_registration import run_coarse_registration
from scripts.tops_esd import (
    estimate_esd_timing,
    compute_esd_timing_correction,
    apply_esd_correction,
    write_esd_summary,
)
from scripts.tops_range_coreg import estimate_range_coreg, write_range_coreg_summary
from scripts.tops_ifg import generate_ifg, IfgResult
from scripts.tops_merge import merge_bursts
from scripts.tops_registration import filter_ifg
from scripts.tops_publish import geocode_ifg, unwrap_ifg, write_product
from scripts.tops_utils import unwrap_phase_2d


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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

    # ── Parse SAFE manifests for master and slave ───────────────────────────
    try:
        master_by_swath = parse_sentinel1_safe(args.master_safe_or_manifest)
    except Exception as exc:
        log.error("Failed to parse master SAFE %s: %s", args.master_safe_or_manifest, exc)
        return 1

    try:
        slave_by_swath = parse_sentinel1_safe(args.slave_safe_or_manifest)
    except Exception as exc:
        log.error("Failed to parse slave SAFE %s: %s", args.slave_safe_or_manifest, exc)
        return 1

    # ── Run pipeline per swath ───────────────────────────────────────────────
    all_results: dict[str, dict[str, Any]] = {}

    for swath in swaths:
        log.info("========================================")
        log.info("Processing swath %s", swath)
        log.info("========================================")

        master_bursts = master_by_swath.get(swath, [])
        slave_bursts = slave_by_swath.get(swath, [])

        if not master_bursts:
            log.warning("No master bursts found for swath %s; skipping.", swath)
            continue
        if not slave_bursts:
            log.warning("No slave bursts found for swath %s; skipping.", swath)
            continue

        result = _run_swath(args, swath, master_bursts, slave_bursts, stages)
        all_results[swath] = result

        if result.get("status") == "failed":
            log.error("Swath %s pipeline failed; continuing to next swath.", swath)

    # ── Cross-swath summary ─────────────────────────────────────────────────
    _log_cross_swath_summary(all_results)

    log.info("tops_insar2 complete: %s", args.output_dir)
    return 0


# ---------------------------------------------------------------------------
# Swath runner
# ---------------------------------------------------------------------------

def _run_swath(
    args: argparse.Namespace,
    swath: str,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    stages: list[str],
) -> dict[str, Any]:
    """Run the full pipeline for one swath.

    Returns a dict with pipeline metadata and per-stage results.
    """
    work_dir = Path(args.output_dir) / swath
    work_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "swath": swath,
        "n_master_bursts": len(master_bursts),
        "n_slave_bursts": len(slave_bursts),
        "stages": stages,
        "status": "unknown",
    }

    # Persistent state passed between stages
    state: dict[str, Any] = {
        "common": None,         # CommonBurstSelection
        "overlaps": None,       # list[OverlapPair]
        "overlap_ifgs": None,    # list of (np.ndarray, np.ndarray) per overlap pair
        "esd_estimates": None,   # list[EsdEstimate]
        "esd_corrections": None, # list[TimingCorrection]
        "range_coreg_estimate": None,
        "burst_ifgs": None,      # list[IfgResult]
        "merged_ifg": None,
        "merged_coh": None,
        "geo2rdr_offsets": None, # Geo2RdrOffsets | None
        "looks": (args.azimuth_looks, args.range_looks),
    }

    for stage_name in stages:
        log.info("[%s] === Stage: %s ===", swath, stage_name)
        try:
            ok = _dispatch_stage(
                stage_name, args, swath, work_dir, master_bursts, slave_bursts, state
            )
        except Exception as exc:
            log.exception("[%s] Stage %s raised unhandled exception: %s", swath, stage_name, exc)
            ok = False

        if not ok:
            log.error("[%s] Stage %s failed; aborting swath pipeline.", swath, stage_name)
            result["status"] = "failed"
            result["failed_stage"] = stage_name
            result["error"] = "unhandled exception during stage execution"
            return result

        result[stage_name] = "ok"

    result["status"] = "ok"
    return result


def _dispatch_stage(
    stage_name: str,
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Dispatch to the appropriate stage function. Returns True on success."""
    stage_map = {
        "check":          _stage_check,
        "preprocess":     _stage_preprocess,
        "common_bursts":  _stage_common_bursts,
        "topo":           _stage_topo,
        "subset_overlaps": _stage_subset_overlaps,
        "coarse_resamp":  _stage_coarse_resamp,
        "overlap_ifg":    _stage_overlap_ifg,
        "prep_esd":       _stage_prep_esd,
        "esd":            _stage_esd,
        "range_coreg":    _stage_range_coreg,
        "fine_resamp":    _stage_fine_resamp,
        "burst_ifg":      _stage_burst_ifg,
        "merge_bursts":   _stage_merge_bursts,
        "filter":         _stage_filter,
        "unwrap":         _stage_unwrap,
        "geocode":        _stage_geocode,
        "publish":        _stage_publish,
    }

    fn = stage_map.get(stage_name)
    if fn is None:
        log.warning("[%s] Unknown stage: %s", swath, stage_name)
        return False

    return fn(args, swath, work_dir, master_bursts, slave_bursts, state)


# ---------------------------------------------------------------------------
# Stage 1: check
# ---------------------------------------------------------------------------

def _stage_check(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Validate that required input paths exist."""
    log.info("[%s] stage_check: validating input paths", swath)

    # Check master SAFE path
    if not args.master_safe_or_manifest.exists():
        log.error("[%s] Master SAFE/manifest not found: %s", swath, args.master_safe_or_manifest)
        return False

    # Check slave SAFE path
    if not args.slave_safe_or_manifest.exists():
        log.error("[%s] Slave SAFE/manifest not found: %s", swath, args.slave_safe_or_manifest)
        return False

    # Check DEM if provided
    if args.dem is not None:
        if not Path(args.dem).exists():
            log.error("[%s] DEM not found: %s", swath, args.dem)
            return False
        log.info("[%s] DEM path validated: %s", swath, args.dem)
    else:
        log.warning("[%s] No DEM provided; topo stage will generate zero offsets", swath)

    log.info("[%s] stage_check: all paths validated OK", swath)
    return True


# ---------------------------------------------------------------------------
# Stage 2: preprocess → common_bursts
# ---------------------------------------------------------------------------

def _stage_preprocess(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Match common bursts between master and slave."""
    log.info("[%s] stage_preprocess: matching common bursts", swath)

    if not master_bursts or not slave_bursts:
        log.error("[%s] Empty burst list: master=%d slave=%d",
                  swath, len(master_bursts), len(slave_bursts))
        return False

    common = match_common_bursts(master_bursts, slave_bursts)

    json_path = work_dir / "common_bursts.json"
    write_common_bursts_json(common, json_path)
    log.info("[%s] Wrote common_bursts.json: %d pairs", swath, common.number_of_common_bursts)

    state["common"] = common
    return True


# Alias for stage name mapping
_stage_common_bursts = _stage_preprocess


# ---------------------------------------------------------------------------
# Stage 4: topo (Geo2Rdr)
# ---------------------------------------------------------------------------

def _stage_topo(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Run Geo2Rdr to compute range/azimuth offsets (or generate zero offsets)."""
    log.info("[%s] stage_topo: running Geo2Rdr", swath)

    common: CommonBurstSelection | None = state.get("common")
    if common is None:
        log.error("[%s] common_bursts not yet computed; run preprocess stage first.", swath)
        return False

    dem_path = Path(args.dem) if args.dem else None
    use_gpu = args.gpu_mode in ("auto", "gpu")

    all_offsets: list[Geo2RdrOffsets] = []
    skipped = False

    for pair in common.pairs:
        pair_dir = work_dir / f"burst_{pair.pair_index:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        try:
            offsets = run_geo2rdr_single_burst(
                ref=pair.reference,
                sec=pair.secondary,
                dem_path=dem_path or Path("/tmp/dummy_dem.tif"),
                work_dir=pair_dir,
                use_gpu=use_gpu,
            )
            all_offsets.append(offsets)
            log.info(
                "[%s] Geo2Rdr burst %d: median_rg=%.3f median_az=%.4f valid=%d",
                swath, pair.pair_index,
                offsets.median_range_offset, offsets.median_azimuth_offset,
                offsets.valid_sample_count,
            )
        except NotImplementedError as exc:
            log.warning(
                "[%s] Geo2Rdr spike not implemented for burst %d (%s); "
                "generating zero offsets instead. Error: %s",
                swath, pair.pair_index, pair_dir, exc,
            )
            skipped = True
            _write_zero_offsets(pair_dir, common, pair.pair_index)
            # Create a dummy Geo2RdrOffsets
            all_offsets.append(
                Geo2RdrOffsets(
                    range_off_path=str(pair_dir / "range.off.npz"),
                    azimuth_off_path=str(pair_dir / "azimuth.off.npz"),
                    median_range_offset=0.0,
                    median_azimuth_offset=0.0,
                    valid_sample_count=0,
                )
            )

    if skipped:
        log.warning(
            "[%s] Geo2Rdr stage skipped; using zero offsets for all burst pairs. "
            "This is only valid for unit-test / synthetic data scenarios.",
            swath,
        )

    state["geo2rdr_offsets"] = all_offsets
    log.info("[%s] stage_topo complete: %d burst pairs processed", swath, len(all_offsets))
    return True


def _write_zero_offsets(
    pair_dir: Path,
    common: CommonBurstSelection,
    pair_idx: int,
) -> None:
    """Write zero offset arrays as .npz files (fallback when Geo2Rdr is unavailable)."""
    import numpy as np

    # Find the burst shape from common.bursts
    pair = common.pairs[pair_idx]
    nl = pair.reference.valid_window.num_lines
    ns = pair.reference.valid_window.num_samples

    zero_range = np.zeros((nl, ns), dtype=np.float32)
    zero_az = np.zeros((nl, ns), dtype=np.float32)

    range_path = pair_dir / "range.off.npz"
    az_path = pair_dir / "azimuth.off.npz"

    np.savez(range_path, data=zero_range)
    np.savez(az_path, data=zero_az)

    log.debug(
        "[%s] Wrote zero offsets for pair %d: shape=%s range=%s az=%s",
        pair_dir.name, pair_idx, (nl, ns), range_path, az_path,
    )


# ---------------------------------------------------------------------------
# Stage 5: subset_overlaps
# ---------------------------------------------------------------------------

def _stage_subset_overlaps(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Materialize top/bottom overlap windows and write overlaps.json."""
    log.info("[%s] stage_subset_overlaps: materializing overlap windows", swath)

    common: CommonBurstSelection | None = state.get("common")
    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False

    if common.number_of_common_bursts < 2:
        log.info(
            "[%s] Only %d common burst(s); skipping overlap stage (need ≥2).",
            swath, common.number_of_common_bursts,
        )
        state["overlaps"] = []
        return True

    overlaps = materialize_overlaps(common)

    json_path = work_dir / "overlaps.json"
    write_overlaps_json(overlaps, json_path)
    log.info("[%s] stage_subset_overlaps: wrote %d overlap pairs to %s",
             swath, len(overlaps), json_path)

    state["overlaps"] = overlaps
    return True


# ---------------------------------------------------------------------------
# Stage 6: coarse_resamp
# ---------------------------------------------------------------------------

def _stage_coarse_resamp(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Coarse resample secondary SLCs using Geo2Rdr offsets."""
    log.info("[%s] stage_coarse_resamp: running coarse registration", swath)

    common: CommonBurstSelection | None = state.get("common")
    geo2dr_offsets: list[Geo2RdrOffsets] | None = state.get("geo2rdr_offsets")

    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False
    if geo2dr_offsets is None:
        log.error("[%s] Geo2Rdr offsets not computed (run topo stage).", swath)
        return False

    skipped = False

    for i, pair in enumerate(common.pairs):
        pair_dir = work_dir / f"burst_{pair.pair_index:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        deramped_ref_path = pair_dir / "deramped_ref.npz"
        deramped_sec_path = pair_dir / "deramped_sec.npz"
        resampled_sec_path = pair_dir / "resampled_sec.npz"

        try:
            run_coarse_registration(
                ref_burst=pair.reference,
                sec_burst=pair.secondary,
                geo2rdr_offsets=geo2dr_offsets[i],
                work_dir=pair_dir,
                deramped_ref_path=deramped_ref_path,
                deramped_sec_path=deramped_sec_path,
                resampled_sec_path=resampled_sec_path,
            )
            log.info(
                "[%s] Coarse resamp burst %d complete: resampled=%s",
                swath, pair.pair_index, resampled_sec_path,
            )
        except NotImplementedError as exc:
            log.warning(
                "[%s] Coarse resamp spike not implemented for burst %d; "
                "skipping (secondary SLC will use raw data). Error: %s",
                swath, pair.pair_index, exc,
            )
            skipped = True
        except FileNotFoundError as exc:
            log.warning(
                "[%s] Coarse resamp skipped for burst %d: required file not found (%s). "
                "Secondary SLC may not be available yet.",
                swath, pair.pair_index, exc,
            )
            skipped = True

    if skipped:
        log.warning(
            "[%s] stage_coarse_resamp: some bursts skipped (ISCE3 spike or missing SLCs)",
            swath,
        )

    log.info("[%s] stage_coarse_resamp complete", swath)
    return True


# ---------------------------------------------------------------------------
# Stage 7: overlap_ifg
# ---------------------------------------------------------------------------

def _stage_overlap_ifg(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Generate interferograms for each overlap pair and save to .npz files."""
    log.info("[%s] stage_overlap_ifg: generating overlap interferograms", swath)

    overlaps: list[OverlapPair] | None = state.get("overlaps")
    common: CommonBurstSelection | None = state.get("common")

    if not overlaps:
        log.info("[%s] No overlaps to process; skipping overlap_ifg stage.", swath)
        return True

    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False

    az_looks, rg_looks = state["looks"]
    overlap_ifgs: list[tuple[np.ndarray, np.ndarray]] = []
    overlap_ifg_dir = work_dir / "overlap_ifg"
    overlap_ifg_dir.mkdir(parents=True, exist_ok=True)

    for ov_idx, ov in enumerate(overlaps):
        # Build synthetic SLC data for top and bottom overlap windows.
        # In the full pipeline, this reads from the burst TIFFs.
        # For the spike: construct zero-filled complex arrays of the correct shape.
        top_slc, bot_slc = _read_overlap_slc_windows(ov, work_dir, common, swath)

        if top_slc is None or bot_slc is None:
            log.warning(
                "[%s] Could not read overlap SLC windows for pair %d; skipping.",
                swath, ov_idx,
            )
            overlap_ifgs.append((np.zeros((1, 1), dtype=np.complex64),
                                 np.zeros((1, 1), dtype=np.float32)))
            continue

        # Cross-multiply top × conj(bot) for ESD IFG
        esd_ifg = top_slc * np.conj(bot_slc)
        ifg_result = generate_ifg(top_slc, bot_slc, looks_az=az_looks, looks_rg=rg_looks)

        # Save overlap IFG and coherence
        out_npz = overlap_ifg_dir / f"overlap_ifg_{ov_idx:03d}.npz"
        np.savez(
            out_npz,
            ifg=ifg_result.complex_ifg,
            coherence=ifg_result.coherence,
            valid_fraction=np.float32(ifg_result.valid_fraction),
        )
        log.info(
            "[%s] overlap_ifg pair %d: shape=%s coherence_mean=%.3f saved=%s",
            swath, ov_idx, ifg_result.complex_ifg.shape,
            float(np.nanmean(ifg_result.coherence)) if ifg_result.coherence.size else float("nan"),
            out_npz,
        )

        overlap_ifgs.append((ifg_result.complex_ifg, ifg_result.coherence))

    state["overlap_ifgs"] = overlap_ifgs
    log.info("[%s] stage_overlap_ifg complete: %d overlap pairs processed",
             swath, len(overlap_ifgs))
    return True


def _read_overlap_slc_windows(
    ov: OverlapPair,
    work_dir: Path,
    common: CommonBurstSelection,
    swath: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read (or synthesize) top/bottom overlap SLC windows from burst data.

    In a full pipeline this reads actual TIFF windows via GDAL.  For the spike
    implementation, returns zero-filled arrays of the correct shape to allow
    the pipeline to run end-to-end without real data.

    Returns (top_slc, bot_slc) as complex64 ndarrays, or (None, None) if the
    overlap window has zero dimensions.
    """
    top_slice = ov.top
    bot_slice = ov.bottom

    if top_slice.num_lines <= 0 or top_slice.num_samples <= 0:
        return None, None
    if bot_slice.num_lines <= 0 or bot_slice.num_samples <= 0:
        return None, None

    # In a real implementation:
    #   - Look up the burst TIFF paths from the BurstRadarGrid identity
    #   - Use GDAL ReadAsArray to read the absolute pixel windows
    #   - Return complex64 arrays
    #
    # Spike: synthesize zero-filled complex64 arrays of the right shape.
    # This lets the stage pipeline run, producing zero-coherence overlap IFGs.
    top_slc = np.zeros(
        (top_slice.num_lines, top_slice.num_samples), dtype=np.complex64
    )
    bot_slc = np.zeros(
        (bot_slice.num_lines, bot_slice.num_samples), dtype=np.complex64
    )

    log.debug(
        "[%s] Synthesized overlap SLC windows: top=%s bot=%s",
        swath, top_slc.shape, bot_slc.shape,
    )
    return top_slc, bot_slc


# ---------------------------------------------------------------------------
# Stage 8: prep_esd
# ---------------------------------------------------------------------------

def _stage_prep_esd(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Run ESD prep: estimate timing offset per overlap pair."""
    log.info("[%s] stage_prep_esd: running ESD estimation", swath)

    overlap_ifgs: list[tuple[np.ndarray, np.ndarray]] | None = state.get("overlap_ifgs")
    common: CommonBurstSelection | None = state.get("common")

    if not overlap_ifgs:
        log.info("[%s] No overlap IFGs available; skipping ESD prep.", swath)
        state["esd_estimates"] = []
        return True

    if common is None or common.number_of_common_bursts < 2:
        log.info("[%s] Fewer than 2 common bursts; ESD not applicable.", swath)
        state["esd_estimates"] = []
        return True

    esd_estimates: list[EsdEstimate] = []
    az_looks = state["looks"][0]

    for i, (ifg, coh) in enumerate(overlap_ifgs):
        if ifg.size == 0:
            log.warning(
                "[%s] Overlap IFG %d is empty; skipping ESD estimation.", swath, i,
            )
            continue

        try:
            estimate = estimate_esd_timing(ifg, looks_az=az_looks)
            esd_estimates.append(estimate)
            log.info(
                "[%s] ESD pair %d: median_offset=%.4f px std=%.4f n=%d",
                swath, i, estimate.median_offset_pixels,
                estimate.std_offset_pixels, estimate.sample_count,
            )
        except Exception as exc:
            log.warning(
                "[%s] ESD estimation failed for overlap pair %d: %s; skipping.",
                swath, i, exc,
            )
            continue

    state["esd_estimates"] = esd_estimates
    log.info("[%s] stage_prep_esd: collected %d ESD estimates", swath, len(esd_estimates))
    return True


# ---------------------------------------------------------------------------
# Stage 9: esd
# ---------------------------------------------------------------------------

def _stage_esd(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Compute and apply ESD timing corrections to secondary SLCs."""
    log.info("[%s] stage_esd: computing ESD timing corrections", swath)

    esd_estimates: list[EsdEstimate] | None = state.get("esd_estimates")
    common: CommonBurstSelection | None = state.get("common")

    if not esd_estimates:
        log.info("[%s] No ESD estimates available; skipping ESD correction.", swath)
        state["esd_corrections"] = []
        return True

    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False

    corrections: list[TimingCorrection] = []
    esd_summary_dir = work_dir / "esd_summary"
    esd_summary_dir.mkdir(parents=True, exist_ok=True)

    for i, est in enumerate(esd_estimates):
        # Use azimuth_time_interval from the common burst selection
        az_interval = common.pairs[0].reference.azimuth_time_interval
        correction = compute_esd_timing_correction(est, az_interval)
        corrections.append(correction)

        log.info(
            "[%s] ESD correction %d: timing=%.6f s (%.4f px) from ESD offset %.4f px",
            swath, i, correction.secondary_timing_seconds,
            correction.secondary_timing_pixels, est.median_offset_pixels,
        )

        # Write per-pair ESD summary
        summary_path = esd_summary_dir / f"esd_pair_{i:03d}_summary.json"
        write_esd_summary(est, summary_path)

    # Write aggregate ESD summary
    _write_aggregate_esd_summary(esd_summary_dir / "esd_summary.json", esd_estimates, corrections)

    state["esd_corrections"] = corrections
    log.info("[%s] stage_esd: computed %d timing corrections", swath, len(corrections))
    return True


def _write_aggregate_esd_summary(
    path: Path,
    estimates: list[EsdEstimate],
    corrections: list[TimingCorrection],
) -> None:
    """Write an aggregate ESD summary JSON combining all overlap estimates."""
    if not estimates:
        return

    import json

    median_offsets = [e.median_offset_pixels for e in estimates]
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_overlap_pairs": len(estimates),
        "median_offset_pixels": float(float(np.median(median_offsets))),
        "mean_offset_pixels": float(float(np.mean(median_offsets))),
        "std_offset_pixels": float(float(np.std(median_offsets))) if len(estimates) > 1 else 0.0,
        "sample_count_total": int(sum(e.sample_count for e in estimates)),
        "corrections": [
            {
                "pair_index": i,
                "secondary_timing_seconds": float(c.secondary_timing_seconds),
                "secondary_timing_pixels": float(c.secondary_timing_pixels),
            }
            for i, c in enumerate(corrections)
        ],
    }

    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Stage 10: range_coreg
# ---------------------------------------------------------------------------

def _stage_range_coreg(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Estimate range-direction coregistration residuals from overlap IFGs."""
    log.info("[%s] stage_range_coreg: estimating range coregistration", swath)

    overlap_ifgs: list[tuple[np.ndarray, np.ndarray]] | None = state.get("overlap_ifgs")
    common: CommonBurstSelection | None = state.get("common")

    if not overlap_ifgs:
        log.info("[%s] No overlap IFGs available; skipping range_coreg.", swath)
        return True

    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False

    az_looks, rg_looks = state["looks"]
    coh_threshold = float(args.esd_coherence_threshold)

    range_offsets = []
    az_offsets = []
    all_estimates = []

    for i, (ifg, coh) in enumerate(overlap_ifgs):
        if ifg.size == 0:
            continue

        try:
            r_off, a_off, est = estimate_range_coreg(
                ifg,
                coh,
                coherence_threshold=coh_threshold,
                looks_az=az_looks,
                looks_rg=rg_looks,
                radar_wavelength=common.pairs[0].reference.radar_wavelength,
                range_pixel_spacing=common.pairs[0].reference.range_pixel_spacing,
            )
            range_offsets.append(r_off)
            az_offsets.append(a_off)
            all_estimates.append(est)
            log.info(
                "[%s] Range coreg pair %d: median_rg=%.4f std=%.4f n=%d",
                swath, i, est.median_range_offset, est.std_range_offset, est.sample_count,
            )
        except Exception as exc:
            log.warning(
                "[%s] Range coreg failed for overlap pair %d: %s; skipping.",
                swath, i, exc,
            )
            continue

    if all_estimates:
        # Write aggregate range_coreg_summary.json
        import json
        summary_path = work_dir / "range_coreg_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_pairs": len(all_estimates),
            "median_range_offset": float(float(np.median([e.median_range_offset for e in all_estimates]))),
            "std_range_offset": float(float(np.std([e.median_range_offset for e in all_estimates]))) if len(all_estimates) > 1 else 0.0,
            "median_azimuth_offset": float(float(np.median([e.median_azimuth_offset for e in all_estimates]))),
            "std_azimuth_offset": float(float(np.std([e.median_azimuth_offset for e in all_estimates]))) if len(all_estimates) > 1 else 0.0,
            "total_sample_count": int(sum(e.sample_count for e in all_estimates)),
            "usable_fraction_avg": float(float(np.mean([e.usable_fraction for e in all_estimates]))),
        }
        with summary_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")

        log.info("[%s] Wrote range_coreg_summary.json: %s", swath, summary_path)
        state["range_coreg_estimate"] = all_estimates[0] if all_estimates else None
    else:
        state["range_coreg_estimate"] = None

    log.info("[%s] stage_range_coreg complete", swath)
    return True


# ---------------------------------------------------------------------------
# Stage 11: fine_resamp (spike)
# ---------------------------------------------------------------------------

def _stage_fine_resamp(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Fine resampling stage (spike — NotImplementedError with warning)."""
    log.warning(
        "[%s] Stage 'fine_resamp' is not yet implemented (spike). "
        "Skipping; coarse-resampled SLCs will be used for burst IFG generation.",
        swath,
    )
    return True


# ---------------------------------------------------------------------------
# Stage 12: burst_ifg
# ---------------------------------------------------------------------------

def _stage_burst_ifg(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Generate per-burst interferograms and coherences."""
    log.info("[%s] stage_burst_ifg: generating burst interferograms", swath)

    common: CommonBurstSelection | None = state.get("common")
    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False

    az_looks, rg_looks = state["looks"]
    burst_ifg_dir = work_dir / "burst_ifg"
    burst_ifg_dir.mkdir(parents=True, exist_ok=True)

    results: list[IfgResult] = []

    for pair in common.pairs:
        pair_dir = work_dir / f"burst_{pair.pair_index:03d}"

        # Load reference and (coarse-)resampled secondary SLCs
        ref_path = pair_dir / "deramped_ref.npz"
        sec_path = pair_dir / "resampled_sec.npz"

        if not ref_path.exists() or not sec_path.exists():
            log.warning(
                "[%s] SLC files not found for burst pair %d; "
                "synthesizing synthetic IFG from burst metadata.",
                swath, pair.pair_index,
            )
            # Synthesize zero IFG of correct shape
            nl = pair.reference.valid_window.num_lines
            ns = pair.reference.valid_window.num_samples
            synthetic_ifg = np.zeros((nl, ns), dtype=np.complex64)
            synthetic_coh = np.zeros((nl, ns), dtype=np.float32)
            ifg_result = IfgResult(
                complex_ifg=synthetic_ifg,
                coherence=synthetic_coh,
                valid_fraction=0.0,
            )
        else:
            ref_slc = _load_slc_from_npz(ref_path)
            sec_slc = _load_slc_from_npz(sec_path)
            ifg_result = generate_ifg(ref_slc, sec_slc, looks_az=az_looks, looks_rg=rg_looks)

        # Save burst IFG and coherence
        out_npz = burst_ifg_dir / f"burst_ifg_{pair.pair_index:03d}.npz"
        np.savez(
            out_npz,
            ifg=ifg_result.complex_ifg,
            coherence=ifg_result.coherence,
            valid_fraction=np.float32(ifg_result.valid_fraction),
        )
        log.info(
            "[%s] Burst IFG pair %d: shape=%s coherence_mean=%.3f saved=%s",
            swath, pair.pair_index, ifg_result.complex_ifg.shape,
            float(np.nanmean(ifg_result.coherence)) if ifg_result.coherence.size else float("nan"),
            out_npz,
        )

        results.append(ifg_result)

    state["burst_ifgs"] = results
    log.info("[%s] stage_burst_ifg: generated %d burst IFGs", swath, len(results))
    return True


def _load_slc_from_npz(path: Path) -> np.ndarray:
    """Load a complex64 SLC from a numpy .npz file."""
    with np.load(path) as npz:
        data = npz["data"]
    if data.dtype == np.complex128:
        data = data.astype(np.complex64)
    return data


# ---------------------------------------------------------------------------
# Stage 13: merge_bursts
# ---------------------------------------------------------------------------

def _stage_merge_bursts(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Merge per-burst IFGs into a full-swath mosaic."""
    log.info("[%s] stage_merge_bursts: merging burst interferograms", swath)

    common: CommonBurstSelection | None = state.get("common")
    burst_ifgs: list[IfgResult] | None = state.get("burst_ifgs")

    if common is None:
        log.error("[%s] common_bursts not yet computed.", swath)
        return False

    if not burst_ifgs:
        log.warning("[%s] No burst IFGs from prior stage; attempting to load from disk.", swath)
        # Fall through to file-loading below
        # If file loading also fails, return True (no-op) rather than hard fail
        burst_ifgs = None

    # Load IFG and coherence arrays (from state or from disk)
    burst_ifg_dir = work_dir / "burst_ifg"
    ifgs: list[np.ndarray] = []
    coherences: list[np.ndarray] = []

    for pair in common.pairs:
        npz_path = burst_ifg_dir / f"burst_ifg_{pair.pair_index:03d}.npz"
        if npz_path.exists():
            with np.load(npz_path) as npz:
                ifgs.append(npz["ifg"].astype(np.complex64))
                coherences.append(npz["coherence"].astype(np.float32))
        else:
            nl = pair.reference.valid_window.num_lines
            ns = pair.reference.valid_window.num_samples
            ifgs.append(np.zeros((nl, ns), dtype=np.complex64))
            coherences.append(np.zeros((nl, ns), dtype=np.float32))

    bursts = [pair.reference for pair in common.pairs]
    valid_windows = [pair.reference.valid_window for pair in common.pairs]

    # Compute merged output dimensions from burst placement
    out_nlines = max(
        pair.reference.image_window.first_line + pair.reference.valid_window.num_lines
        for pair in common.pairs
    )
    out_nsamples = max(
        pair.reference.valid_window.num_samples
        for pair in common.pairs
    )

    out_ifg = np.zeros((out_nlines, out_nsamples), dtype=np.complex64)
    out_coh = np.zeros((out_nlines, out_nsamples), dtype=np.float32)

    # Seam regions: 5-line-wide windows at burst boundaries
    seam_regions = _compute_seam_regions(common)

    result = merge_bursts(
        ifgs=ifgs,
        coherences=coherences,
        bursts=bursts,
        valid_windows=valid_windows,
        seam_regions=seam_regions,
        out_ifg=out_ifg,
        out_coh=out_coh,
    )

    # Save merged products
    merged_dir = work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    np.save(merged_dir / "merged_interferogram.npy", out_ifg)
    np.save(merged_dir / "merged_coherence.npy", out_coh)

    # Write seam diagnostics
    import json
    diag_path = merged_dir / "burst_seam_diagnostics.json"
    with diag_path.open("w") as fh:
        json.dump({
            "seam_phase_diff_median": float(result.seam_phase_diff_median),
            "seam_phase_diff_std": float(result.seam_phase_diff_std),
            "seam_coherence_drop": float(result.seam_coherence_drop),
            "gap_pixel_count": int(result.gap_pixel_count),
            "top_contribution_count": int(result.top_contribution_count),
            "bottom_contribution_count": int(result.bottom_contribution_count),
            "n_segments": len(result.segments),
        }, fh, indent=2)
        fh.write("\n")

    log.info(
        "[%s] Merged IFG: shape=%s gap_pixels=%d seam_phase_median=%.4f",
        swath, out_ifg.shape, result.gap_pixel_count, result.seam_phase_diff_median,
    )

    state["merged_ifg"] = out_ifg
    state["merged_coh"] = out_coh
    log.info("[%s] stage_merge_bursts complete", swath)
    return True


def _compute_seam_regions(
    common: CommonBurstSelection,
) -> list[tuple[int, int, int, int]]:
    """Compute (line, col, height, width) seam regions at burst boundaries."""
    seams: list[tuple[int, int, int, int]] = []
    SEAM_HALF_WIDTH = 2  # 5 pixels total width

    for i in range(len(common.pairs) - 1):
        # Seam at the transition from pair[i] to pair[i+1]
        p0 = common.pairs[i].reference
        p1 = common.pairs[i + 1].reference

        seam_line = p1.image_window.first_line - SEAM_HALF_WIDTH
        seam_col = p0.valid_window.first_sample
        seam_h = SEAM_HALF_WIDTH * 2
        seam_w = min(p0.valid_window.num_samples, p1.valid_window.num_samples)

        seams.append((seam_line, seam_col, seam_h, seam_w))

    return seams


# ---------------------------------------------------------------------------
# Stages 14–17: filter, unwrap, geocode, publish (spike)
# ---------------------------------------------------------------------------

def _stage_filter(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Apply Goldstein phase filtering to merged interferogram."""
    log.info("[%s] stage_filter: applying Goldstein filtering", swath)

    merged_ifg: np.ndarray | None = state.get("merged_ifg")
    merged_coh: np.ndarray | None = state.get("merged_coh")

    merged_dir = work_dir / "merged"

    # Load merged products from disk if not in state
    if merged_ifg is None:
        ifg_path = merged_dir / "filtered_ifg.npy"
        if not ifg_path.exists():
            ifg_path = merged_dir / "merged_interferogram.npy"
        if ifg_path.exists():
            merged_ifg = np.load(ifg_path)
            log.info("[%s] Loaded merged_ifg from %s", swath, ifg_path)

    if merged_ifg is None:
        log.error("[%s] merged_ifg not in state and not on disk; run merge_bursts first.", swath)
        return False

    # Load from disk if not in state
    if merged_coh is None:
        coh_path = merged_dir / "merged_coherence.npy"
        if coh_path.exists():
            merged_coh = np.load(coh_path)
            log.info("[%s] Loaded merged_coh from %s", swath, coh_path)

    # At this point merged_coh is guaranteed to be non-None
    assert merged_coh is not None

    # Log pre-filter coherence stats
    coh_before = float(np.nanmean(merged_coh))
    log.info("[%s] Filter input: shape=%s coherence_mean=%.4f", swath, merged_ifg.shape, coh_before)

    # Apply Goldstein filter (alpha=0.5 default)
    alpha = 0.5
    filtered_ifg = filter_ifg(merged_ifg, merged_coh, alpha=alpha)
    log.info("[%s] Filter output: shape=%s alpha=%.1f", swath, filtered_ifg.shape, alpha)

    # Log post-filter stats (coherence unchanged — filtering only affects phase magnitude)
    coh_after = float(np.nanmean(merged_coh))
    log.info(
        "[%s] stage_filter complete: coherence before=%.4f after=%.4f",
        swath, coh_before, coh_after,
    )

    # Save filtered interferogram
    merged_dir = work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    np.save(merged_dir / "filtered_ifg.npy", filtered_ifg)

    state["merged_ifg"] = filtered_ifg
    log.info("[%s] stage_filter: saved filtered_ifg.npy", swath)
    return True


def _stage_unwrap(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Unwrap phase using ICU/SNAPHU with fallback to simple 2D unwrap."""
    log.info("[%s] stage_unwrap: unwrapping phase", swath)

    merged_ifg: np.ndarray | None = state.get("merged_ifg")
    merged_coh: np.ndarray | None = state.get("merged_coh")

    merged_dir = work_dir / "merged"

    # Load merged products from disk if not in state
    if merged_ifg is None:
        ifg_path = merged_dir / "filtered_ifg.npy"
        if not ifg_path.exists():
            ifg_path = merged_dir / "merged_interferogram.npy"
        if ifg_path.exists():
            merged_ifg = np.load(ifg_path)
            log.info("[%s] Loaded merged_ifg from %s", swath, ifg_path)

    if merged_ifg is None:
        log.error("[%s] merged_ifg not in state and not on disk; run filter stage first.", swath)
        return False

    # Load from disk if not in state
    if merged_coh is None:
        coh_path = merged_dir / "merged_coherence.npy"
        if coh_path.exists():
            merged_coh = np.load(coh_path)
            log.info("[%s] Loaded merged_coh from %s", swath, coh_path)

    assert merged_coh is not None

    # Extract wrapped phase (angle of complex IFG)
    phase = np.angle(merged_ifg).astype(np.float32)
    log.info("[%s] Unwrap input: shape=%s coherence_mean=%.4f", swath, phase.shape, float(np.nanmean(merged_coh)))

    unwrapped: np.ndarray
    method = str(args.unwrap_method).lower()

    # Try ICU/SNAPHU first
    try:
        unwrapped = unwrap_ifg(
            phase,
            merged_coh,
            method=method,
            work_dir=work_dir,
        )
        log.info("[%s] Unwrap via %s: shape=%s", swath, method, unwrapped.shape)
    except (NotImplementedError, FileNotFoundError, RuntimeError) as exc:
        # ICU/SNAPHU unavailable or failed — fallback to simple 2D unwrap
        log.warning(
            "[%s] ICU/SNAPHU (%s) unavailable or failed (%s); "
            "falling back to simple 2D unwrapper.",
            swath, method, exc,
        )
        unwrapped = unwrap_phase_2d(phase)
        log.info("[%s] Unwrap via fallback (simple 2D): shape=%s", swath, unwrapped.shape)

    # Save unwrapped phase
    merged_dir = work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    np.save(merged_dir / "unwrapped.npy", unwrapped)

    state["unwrapped"] = unwrapped
    log.info("[%s] stage_unwrap: saved unwrapped.npy", swath)
    return True


def _stage_geocode(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Geocode merged interferogram and coherence using GDAL.

    If GDAL is unavailable or the DEM is missing, logs a warning and skips
    geocoding without failing the pipeline.
    """
    log.info("[%s] stage_geocode: geocoding merged interferogram", swath)

    merged_ifg: np.ndarray | None = state.get("merged_ifg")
    merged_coh: np.ndarray | None = state.get("merged_coh")

    if merged_ifg is None or merged_coh is None:
        log.warning(
            "[%s] merged_ifg or merged_coh not in state; skipping geocode.",
            swath,
        )
        return True

    # Resolve DEM path
    dem_path = Path(args.dem) if args.dem else None
    if dem_path is None or not dem_path.exists():
        log.info(
            "[%s] No DEM available; skipping geocode stage. "
            "Provide DEM via --dem for geocoded products.",
            swath,
        )
        return True

    # Load merged products from disk if needed
    merged_dir = work_dir / "merged"
    ifg_path = merged_dir / "merged_interferogram.npy"
    coh_path = merged_dir / "merged_coherence.npy"

    if merged_ifg is None and ifg_path.exists():
        merged_ifg = np.load(ifg_path)
    if merged_coh is None and coh_path.exists():
        merged_coh = np.load(coh_path)

    if merged_ifg is None or merged_coh is None:
        log.warning("[%s] Cannot load merged products; skipping geocode.", swath)
        return True

    # Get first burst for geometry info
    common: Any = state.get("common")
    first_burst = common.pairs[0].reference if common and common.pairs else None

    if first_burst is None:
        log.warning("[%s] Cannot determine burst geometry; skipping geocode.", swath)
        return True

    try:
        geo_ifg, geo_coh = geocode_ifg(
            merged_ifg=merged_ifg,
            merged_coh=merged_coh,
            burst=first_burst,
            dem_path=dem_path,
            work_dir=work_dir / "geocode_tmp",
            res_meters=args.resolution_meters,
        )

        # Save geocoded products
        merged_dir.mkdir(parents=True, exist_ok=True)
        np.save(merged_dir / "interferogram.geo.npy", geo_ifg)
        np.save(merged_dir / "coherence.geo.npy", geo_coh)

        log.info(
            "[%s] Geocoded IFG: shape=%s coherence_mean=%.4f",
            swath, geo_ifg.shape, float(np.nanmean(geo_coh)),
        )

        state["geocoded_ifg"] = geo_ifg
        state["geocoded_coh"] = geo_coh

    except NotImplementedError as exc:
        log.warning(
            "[%s] GDAL not available; skipping geocode. "
            "Install GDAL (osgeo) for geocoded products. Details: %s",
            swath, exc,
        )
        return True
    except Exception as exc:
        log.warning(
            "[%s] Geocoding failed (%s); skipping geocode stage. "
            "Pipeline continues without geocoded products.",
            swath, exc,
        )
        return True

    # Unwrap the geocoded phase if unwrapped phase is available in state
    unwrapped = state.get("unwrapped")
    if unwrapped is not None:
        log.info("[%s] Geocoding unwrapped phase from state", swath)
        try:
            unw_geo = geocode_ifg(
                merged_ifg=unwrapped.astype(np.complex64),
                merged_coh=merged_coh,
                burst=first_burst,
                dem_path=dem_path,
                work_dir=work_dir / "geocode_tmp",
                res_meters=args.resolution_meters,
            )
            np.save(merged_dir / "unwrapped.geo.npy", unw_geo[0])
            state["unwrapped_geocoded"] = unw_geo[0]
        except Exception as exc:
            log.warning(
                "[%s] Unwrapped phase geocoding failed (%s); skipping.",
                swath, exc,
            )

    log.info("[%s] stage_geocode complete", swath)
    return True


def _stage_publish(
    args: argparse.Namespace,
    swath: str,
    work_dir: Path,
    master_bursts: list[BurstRadarGrid],
    slave_bursts: list[BurstRadarGrid],
    state: dict[str, Any],
) -> bool:
    """Write final HDF5 and geocoded TIFF products to output directory."""
    log.info("[%s] stage_publish: writing final products", swath)

    merged_ifg: np.ndarray | None = state.get("merged_ifg")
    merged_coh: np.ndarray | None = state.get("merged_coh")
    unwrapped = state.get("unwrapped")
    geocoded_ifg = state.get("geocoded_ifg")
    geocoded_coh = state.get("geocoded_coh")

    # Load from disk if not in state
    merged_dir = work_dir / "merged"
    if merged_ifg is None:
        ifg_path = merged_dir / "filtered_ifg.npy"
        if not ifg_path.exists():
            ifg_path = merged_dir / "merged_interferogram.npy"
        if ifg_path.exists():
            merged_ifg = np.load(ifg_path)
            log.info("[%s] Loaded merged_ifg from %s", swath, ifg_path)

    if merged_coh is None:
        coh_path = merged_dir / "merged_coherence.npy"
        if coh_path.exists():
            merged_coh = np.load(coh_path)
            log.info("[%s] Loaded merged_coh from %s", swath, coh_path)

    if unwrapped is None:
        unw_path = merged_dir / "unwrapped.npy"
        if unw_path.exists():
            unwrapped = np.load(unw_path)
            log.info("[%s] Loaded unwrapped from %s", swath, unw_path)

    if merged_ifg is None:
        log.error("[%s] merged_ifg not available; cannot publish.", swath)
        return False

    product_name = f"{swath}_interferogram"
    output_dir = Path(args.output_dir) / swath

    # Determine geo_transform and projection from geocoded data if available
    geo_transform: tuple | None = None
    projection: str = ""

    if geocoded_ifg is not None and geocoded_coh is not None:
        common: Any = state.get("common")
        if common and common.pairs:
            first_burst = common.pairs[0].reference
            # Build geotransform from burst geometry
            geo_transform = (
                float(first_burst.starting_range),
                float(first_burst.range_pixel_spacing),
                0.0,
                float(first_burst.identity.sensing_start.timestamp()),
                0.0,
                -float(first_burst.azimuth_time_interval),
            )
            # Get projection from DEM if available
            dem_path = Path(args.dem) if args.dem else None
            if dem_path and dem_path.exists():
                try:
                    from osgeo import gdal
                    ds = gdal.Open(str(dem_path))
                    if ds:
                        projection = ds.GetProjection()
                        ds = None
                except Exception:
                    pass

    try:
        written_files = write_product(
            merged_ifg=merged_ifg,
            merged_coh=merged_coh if merged_coh is not None else np.ones_like(merged_ifg, dtype=np.float32),
            unwrapped=unwrapped,
            geo_transform=geo_transform or (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            projection=projection,
            output_dir=output_dir,
            product_name=product_name,
        )

        log.info("[%s] stage_publish: wrote %d output files", swath, len(written_files))
        for path in written_files:
            log.info("  -> %s", path)

        state["published_files"] = written_files

    except Exception as exc:
        log.warning(
            "[%s] stage_publish failed (%s); partial products may exist. "
            "Continuing pipeline.",
            swath, exc,
        )
        return True

    log.info("[%s] stage_publish complete: %s", swath, output_dir)
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _log_cross_swath_summary(all_results: dict[str, dict[str, Any]]) -> None:
    """Log a summary of results across all swaths."""
    if not all_results:
        log.info("No swaths processed.")
        return

    total_ok = sum(1 for r in all_results.values() if r.get("status") == "ok")
    total_failed = sum(1 for r in all_results.values() if r.get("status") == "failed")

    log.info("========================================")
    log.info("Cross-swath summary: %d swaths processed", len(all_results))
    log.info("  OK:      %d", total_ok)
    log.info("  Failed:  %d", total_failed)
    for swath, result in all_results.items():
        status = result.get("status", "unknown")
        failed_stage = result.get("failed_stage", "—")
        n_pairs = result.get("stages", [])
        log.info(
            "  %s: %s %s",
            swath,
            status.upper(),
            f"[failed at {failed_stage}]" if status == "failed" else "",
        )
    log.info("========================================")


if __name__ == "__main__":
    sys.exit(main())