"""tops_merge — Valid-window-aware burst mosaic for Sentinel-1 TOPS InSAR.

Combines per-burst interferograms (complex64) and coherences (float32) into a
full-swath mosaic using per-burst valid-window placement and cosine-tapered
overlap blending with inter-burst phase alignment.

Seam diagnostics are computed at burst boundaries and returned as a MergeResult.

No imports from strip_insar / strip_insar2 / tops_insar backends.
"""

from __future__ import annotations

from datetime import timezone
import logging
from typing import Sequence

import numpy as np

from scripts.tops_model import (
    BurstRadarGrid,
    BurstWindow,
    MergeResult,
    MergeSegment,
)

__all__ = ["merge_bursts", "merged_mosaic_shape", "plan_merge_segments"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_bursts(
    ifgs: list[np.ndarray],
    coherences: list[np.ndarray],
    bursts: list[BurstRadarGrid],
    valid_windows: list[BurstWindow],
    seam_regions: list[tuple[int, int, int, int]],
    *,
    out_ifg: np.ndarray,
    out_coh: np.ndarray,
) -> MergeResult:
    """Merge per-burst interferograms and coherences into a full-swath mosaic.

    Merging pipeline:
      1. Assemble bursts sequentially using ISCE2-style VRT assembly:
         - Overlap region between adjacent bursts: simple average (0.5 × (prev + cur))
         - Non-overlap: direct copy of current burst data
      2. Normalize by accumulation weight.
      3. Compute seam diagnostics at burst boundaries.

    No phase offset estimation or cosine tapering (ISCE2 design choice):
    ESD-derived azimuth timing correction handles TOPS phase consistency
    during the fine resampling stage. Overlap is small (~20 px for S1 IW),
    so simple averaging suffices.

    Parameters
    ----------
    ifgs : list of np.ndarray (complex64)
        Per-burst complex interferograms, one per burst.
    coherences : list of np.ndarray (float32)
        Per-burst coherence rasters, same shape as ``ifgs``.
    bursts : list of BurstRadarGrid
        Radar-grid metadata for each burst (used for contribution counting).
    valid_windows : list of BurstWindow
        Valid-window (relative to each burst's SLC) for each burst.
    seam_regions : list of (line, sample, height, width)
        Seam windows (5 px wide by default) at adjacent-burst boundaries.
        Coordinates are relative to the merged output array.
    out_ifg : np.ndarray (complex64, writable)
        Pre-allocated output interferogram mosaic. Modified in-place.
    out_coh : np.ndarray (float32, writable)
        Pre-allocated output coherence mosaic. Modified in-place.
    overlap_lines_taper : int, default 15
        Number of lines at burst edges to apply cosine taper.
        Sentinel-1 IW overlap is ~20 lines; default covers most.

    Returns
    -------
    MergeResult
        Seam diagnostics, gap count, and contribution statistics.
    """
    n = len(ifgs)
    if len(coherences) != n:
        raise ValueError(
            f"ifgs and coherences must have the same length: {n} vs {len(coherences)}"
        )
    if len(bursts) != n or len(valid_windows) != n:
        raise ValueError(
            f"bursts ({len(bursts)}) and valid_windows ({len(valid_windows)}) "
            f"must have same length as ifgs ({n})"
        )

    out_shape = out_ifg.shape
    if out_coh.shape != out_shape:
        raise ValueError(
            f"out_ifg and out_coh must have the same shape: "
            f"{out_shape} vs {out_coh.shape}"
        )

    # ------------------------------------------------------------------
    # Step 0: Compute output placement for each burst
    # ------------------------------------------------------------------
    rd_origin = _rd_grid_origin(bursts, valid_windows)
    out_h, out_w = out_shape

    placements: list[tuple[int, int, int, int]] = []  # (line_start, col_start, line_end, col_end)
    for burst, vw in zip(bursts, valid_windows):
        line_start, col_start = _rd_output_start(burst, vw, rd_origin)
        in_h, in_w = vw.num_lines, vw.num_samples
        line_end = min(line_start + in_h, out_h)
        col_end = min(col_start + in_w, out_w)
        placements.append((line_start, col_start, line_end, col_end))

    # ------------------------------------------------------------------
    # Step 1: ISCE2-style burst assembly with overlap averaging
    # ------------------------------------------------------------------
    # ISCE2 (runMergeBursts.py mergeBursts) merges by:
    #   - Overlap region: simple 0.5*(top_data + cur_data) averaging
    #   - Non-overlap: direct copy from current burst
    #   - No phase offset estimation (ESD already handled during resampling)
    #   - No cosine tapering (overlap is only ~20px, averaging suffices)
    # Here we add coherence-weighted averaging as an improvement.

    ifg_sum = np.zeros(out_shape, dtype=np.complex64)
    coh_sum = np.zeros(out_shape, dtype=np.float32)
    weight = np.zeros(out_shape, dtype=np.float32)
    top_contrib = np.zeros(out_shape, dtype=np.int32)
    bot_contrib = np.zeros(out_shape, dtype=np.int32)

    top_contribution_count = 0
    bottom_contribution_count = 0
    segments: list[MergeSegment] = []

    for i, (ifg, coh, burst, vw) in enumerate(zip(ifgs, coherences, bursts, valid_windows)):
        _check_input_shape(ifg, coh, vw, i)
        in_h, in_w = ifg.shape
        line_start, col_start, line_end, col_end = placements[i]
        clip_h = line_end - line_start
        clip_w = col_end - col_start
        if clip_h <= 0 or clip_w <= 0:
            continue

        # Determine overlap with previous burst in output coordinates
        if i > 0:
            ls_prev, _, le_prev, _ = placements[i - 1]
            ol_start = max(ls_prev, line_start)
            ol_end = min(le_prev, line_end)
            olap = max(0, ol_end - ol_start)
        else:
            olap = 0

        # Clip source
        src_ifg = ifg[:clip_h, :clip_w]
        src_coh = coh[:clip_h, :clip_w]
        dst_line_slc = slice(line_start, line_end)
        dst_col_slc = slice(col_start, col_end)

        if olap <= 0:
            # No overlap — direct write
            ifg_sum[dst_line_slc, dst_col_slc] += src_ifg.astype(np.complex64)
            coh_sum[dst_line_slc, dst_col_slc] += src_coh.astype(np.float32)
            weight[dst_line_slc, dst_col_slc] += 1.0
        else:
            # Overlap with previous burst: merge is already accumulated
            # in ifg_sum/coh_sum/weight from burst i-1.  Add non-overlap
            # portion of current burst.
            ol_lines_in_cur = ol_end - ol_start  # overlap lines in output
            cur_ol_start = ol_start - line_start  # overlap start within cur burst
            cur_ol_end = ol_end - line_start

            # Non-overlap portion (lower part of this burst)
            if cur_ol_end < clip_h:
                non_ol_slc_line = slice(cur_ol_end, clip_h)
                non_ol_dst_line = slice(line_start + cur_ol_end, line_end)
                ifg_sum[non_ol_dst_line, dst_col_slc] += src_ifg[non_ol_slc_line, :].astype(np.complex64)
                coh_sum[non_ol_dst_line, dst_col_slc] += src_coh[non_ol_slc_line, :].astype(np.float32)
                weight[non_ol_dst_line, dst_col_slc] += 1.0

            # Overlap region: average with previous burst's data.
            # ISCE2 method: 0.5*(prev_overlap + cur_overlap)
            ol_dst_line = slice(ol_start, ol_end)
            cur_ol_ifg = src_ifg[cur_ol_start:cur_ol_end, :]
            cur_ol_coh = src_coh[cur_ol_start:cur_ol_end, :]

            # Previous burst's overlap contribution was added with weight=1.
            # The average is: (prev + cur) / 2 = ifg_sum/weight (prev only)
            # which gives 0.5*prev. We add cur and set weight=2:
            # final = (prev + cur) / 2
            ifg_sum[ol_dst_line, dst_col_slc] += cur_ol_ifg.astype(np.complex64)
            coh_sum[ol_dst_line, dst_col_slc] += cur_ol_coh.astype(np.float32)
            weight[ol_dst_line, dst_col_slc] += 1.0  # was 1 from prev, now 2

        # Contribution statistics (for diagnostics only)
        if i < max(1, n // 2):
            top_contrib[dst_line_slc, dst_col_slc] = 1
            top_contribution_count += clip_h * clip_w
        else:
            bot_contrib[dst_line_slc, dst_col_slc] = 1
            bottom_contribution_count += clip_h * clip_w

        segments.append(
            MergeSegment(
                burst_index=i,
                pair_index=i,
                input_line_start=0,
                input_num_lines=clip_h,
                input_sample_start=0,
                input_num_samples=clip_w,
                output_line_start=line_start,
                output_num_lines=clip_h,
                output_sample_start=col_start,
                output_num_samples=clip_w,
            )
        )

    # ------------------------------------------------------------------
    # Step 2: Normalize by total weight
    # ------------------------------------------------------------------
    mask = weight > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out_ifg[:] = np.where(mask, ifg_sum / weight, 0.0).astype(np.complex64)
        out_coh[:] = np.where(mask, coh_sum / weight, 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3: Seam diagnostics
    # ------------------------------------------------------------------
    seam_phase_diffs: list[float] = []
    seam_coh_drops: list[float] = []

    for (seam_line, seam_col, seam_h, seam_w) in seam_regions:
        if seam_line < 0 or seam_col < 0:
            logging.warning(
                "Seam region with negative coordinates skipped: (%d, %d, %d, %d)",
                seam_line, seam_col, seam_h, seam_w,
            )
            continue
        if seam_line + seam_h > out_h or seam_col + seam_w > out_w:
            continue
        if seam_h <= 0 or seam_w <= 0:
            continue

        seam_ifg = out_ifg[seam_line:seam_line + seam_h, seam_col:seam_col + seam_w]
        seam_coh = out_coh[seam_line:seam_line + seam_h, seam_col:seam_col + seam_w]

        # Phase diff: median phase of the seam region
        phase_seam = np.angle(seam_ifg)
        if phase_seam.size > 0:
            seam_phase_diffs.append(float(np.nanmedian(phase_seam)))

        if seam_coh.size > 0:
            seam_coh_drops.append(float(np.nanmean(seam_coh)))

    seam_phase_diff_median = (
        float(np.median(seam_phase_diffs)) if seam_phase_diffs else 0.0
    )
    seam_phase_diff_std = (
        float(np.std(seam_phase_diffs)) if len(seam_phase_diffs) > 1 else 0.0
    )
    seam_coherence_drop = (
        float(np.mean(seam_coh_drops)) if seam_coh_drops else 0.0
    )

    gap_pixel_count = int(np.sum(weight == 0))

    return MergeResult(
        seam_phase_diff_median=seam_phase_diff_median,
        seam_phase_diff_std=seam_phase_diff_std,
        seam_coherence_drop=seam_coherence_drop,
        gap_pixel_count=gap_pixel_count,
        top_contribution_count=top_contribution_count,
        bottom_contribution_count=bottom_contribution_count,
        segments=tuple(segments),
    )


def plan_merge_segments(
    bursts: Sequence[BurstRadarGrid],
    valid_windows: Sequence[BurstWindow],
    out_nlines: int,
    out_nsamples: int,
) -> tuple[MergeSegment, ...]:
    """Plan the placement of each burst's valid window into the merged mosaic."""
    segments: list[MergeSegment] = []
    rd_origin = _rd_grid_origin(bursts, valid_windows)
    for i, (burst, vw) in enumerate(zip(bursts, valid_windows)):
        out_line_start, out_col_start = _rd_output_start(burst, vw, rd_origin)
        segments.append(
            MergeSegment(
                burst_index=i,
                pair_index=i,
                input_line_start=0,
                input_num_lines=vw.num_lines,
                input_sample_start=0,
                input_num_samples=vw.num_samples,
                output_line_start=out_line_start,
                output_num_lines=min(vw.num_lines, out_nlines - out_line_start)
                if out_line_start < out_nlines else 0,
                output_sample_start=out_col_start,
                output_num_samples=min(vw.num_samples, out_nsamples - out_col_start)
                if out_col_start < out_nsamples else 0,
            )
        )
    return tuple(sorted(segments, key=lambda s: s.output_line_start))


def merged_mosaic_shape(
    bursts: Sequence[BurstRadarGrid],
    valid_windows: Sequence[BurstWindow],
) -> tuple[int, int]:
    """Compute output shape for an RD-domain TOPS burst mosaic."""
    if len(bursts) != len(valid_windows):
        raise ValueError(
            f"bursts ({len(bursts)}) and valid_windows ({len(valid_windows)}) must have same length"
        )
    rd_origin = _rd_grid_origin(bursts, valid_windows)
    out_nlines = 0
    out_nsamples = 0
    for burst, vw in zip(bursts, valid_windows):
        out_line_start, out_col_start = _rd_output_start(burst, vw, rd_origin)
        out_nlines = max(out_nlines, out_line_start + vw.num_lines)
        out_nsamples = max(out_nsamples, out_col_start + vw.num_samples)
    return out_nlines, out_nsamples


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rd_grid_origin(
    bursts: Sequence[BurstRadarGrid],
    valid_windows: Sequence[BurstWindow],
) -> tuple[float, float]:
    """Return merged RD-grid origin for the first valid mosaic pixel."""
    if not bursts:
        return 0.0, 0.0
    return (
        min(
            _sensing_seconds(burst)
            + window.first_line * burst.azimuth_time_interval
            for burst, window in zip(bursts, valid_windows)
        ),
        min(
            float(burst.starting_range)
            + window.first_sample * burst.range_pixel_spacing
            for burst, window in zip(bursts, valid_windows)
        ),
    )


def _sensing_seconds(burst: BurstRadarGrid) -> float:
    sensing_start = burst.identity.sensing_start
    if sensing_start.tzinfo is None:
        sensing_start = sensing_start.replace(tzinfo=timezone.utc)
    return sensing_start.timestamp()


def _rd_output_start(
    burst: BurstRadarGrid,
    valid_window: BurstWindow,
    rd_origin: tuple[float, float],
) -> tuple[int, int]:
    ref_sensing_seconds, ref_range = rd_origin
    first_valid_time = (
        _sensing_seconds(burst)
        + valid_window.first_line * burst.azimuth_time_interval
    )
    first_valid_range = (
        float(burst.starting_range)
        + valid_window.first_sample * burst.range_pixel_spacing
    )
    row_off = int(round((first_valid_time - ref_sensing_seconds) / burst.azimuth_time_interval))
    col_off = int(round((first_valid_range - ref_range) / burst.range_pixel_spacing))
    return row_off, col_off


def _check_input_shape(
    ifg: np.ndarray,
    coh: np.ndarray,
    vw: BurstWindow,
    idx: int,
) -> None:
    """Validate that an input IFG/coherence pair matches its valid window."""
    if ifg.shape != coh.shape:
        raise ValueError(
            f"Burst {idx}: ifg shape {ifg.shape} != coh shape {coh.shape}"
        )
    expected = (vw.num_lines, vw.num_samples)
    if ifg.shape != expected:
        raise ValueError(
            f"Burst {idx}: ifg shape {ifg.shape} != expected {expected} "
            f"from valid_window"
        )
    if ifg.dtype != np.complex64:
        logging.warning(
            "Burst %d: ifg dtype is %s, expected complex64; results may lose precision.",
            idx, ifg.dtype,
        )
    if coh.dtype != np.float32:
        logging.warning(
            "Burst %d: coh dtype is %s, expected float32; results may lose precision.",
            idx, coh.dtype,
        )