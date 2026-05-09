"""tops_merge — Valid-window-aware burst mosaic for Sentinel-1 TOPS InSAR.

Combines per-burst interferograms (complex64) and coherences (float32) into a
full-swath mosaic using per-burst valid-window placement.  Seam diagnostics
are computed at burst boundaries and returned as a MergeResult.

No imports from strip_insar / strip_insar2 / tops_insar backends.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from scripts.tops_model import (
    BurstRadarGrid,
    BurstWindow,
    MergeResult,
    MergeSegment,
)

__all__ = ["merge_bursts", "plan_merge_segments"]


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

    Returns
    -------
    MergeResult
        Seam diagnostics, gap count, and contribution statistics.

    Raises
    ------
    ValueError
        If the number of input arrays does not match, or if any input shape
        is incompatible with its corresponding valid_window / burst metadata.
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

    # Accumulation arrays for weighted average at seams
    ifg_sum = np.zeros(out_shape, dtype=np.complex64)
    coh_sum = np.zeros(out_shape, dtype=np.float32)
    weight = np.zeros(out_shape, dtype=np.float32)   # contribution count per pixel
    top_contrib = np.zeros(out_shape, dtype=np.int32)
    bot_contrib = np.zeros(out_shape, dtype=np.int32)

    top_contribution_count = 0
    bottom_contribution_count = 0
    segments: list[MergeSegment] = []

    out_h, out_w = out_shape

    for i, (ifg, coh, burst, vw) in enumerate(zip(ifgs, coherences, bursts, valid_windows)):

        # Validate input shape matches valid_window
        _check_input_shape(ifg, coh, vw, i)

        # Compute absolute placement in output: valid_window is relative to burst,
        # so we use image_window.first_line + valid_window.first_line as the line offset.
        # (This mirrors BurstRadarGrid.valid_line_start semantics.)
        out_line_start = burst.image_window.first_line + vw.first_line
        out_col_start = vw.first_sample          # sample coordinate is absolute already

        in_h, in_w = ifg.shape
        out_h, out_w = out_shape

        # Clamp to output bounds
        out_line_end = min(out_line_start + in_h, out_h)
        out_col_end = min(out_col_start + in_w, out_w)
        clip_h = out_line_end - out_line_start
        clip_w = out_col_end - out_col_start

        if clip_h <= 0 or clip_w <= 0:
            logging.warning(
                "Burst %d valid window completely outside output bounds; skipping.", i
            )
            continue

        # Source slice within the burst's valid window
        src_line = 0
        src_col = 0

        # Destination slice within output
        dst_line_slice = slice(out_line_start, out_line_end)
        dst_col_slice = slice(out_col_start, out_col_end)

        # Accumulate (for averaging seam regions)
        ifg_sum[dst_line_slice, dst_col_slice] += ifg[src_line:src_line + clip_h,
                                                      src_col:src_col + clip_w]
        coh_sum[dst_line_slice, dst_col_slice] += coh[src_line:src_line + clip_h,
                                                      src_col:src_col + clip_w]
        weight[dst_line_slice, dst_col_slice] += 1.0

        # Contribution split: top bursts are i=0, odd i; bottom bursts are even i>0
        # We define "top" as strictly above the mosaic centre, "bottom" as below.
        # Use output line position as proxy: bursts with earlier output_line_start are top.
        if out_line_start < out_h // 2:
            top_contrib[dst_line_slice, dst_col_slice] = 1
            top_contribution_count += clip_h * clip_w
        else:
            bot_contrib[dst_line_slice, dst_col_slice] = 1
            bottom_contribution_count += clip_h * clip_w

        segments.append(
            MergeSegment(
                burst_index=i,
                pair_index=i,
                input_line_start=src_line,
                input_num_lines=clip_h,
                input_sample_start=src_col,
                input_num_samples=clip_w,
                output_line_start=out_line_start,
                output_num_lines=clip_h,
                output_sample_start=out_col_start,
                output_num_samples=clip_w,
            )
        )

    # Final merge: divide accumulated sums by weights
    mask = weight > 0
    # Use np.where to avoid division by zero on non-contributing pixels (leave zeros)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_ifg[:] = np.where(mask, ifg_sum / weight, 0.0).astype(np.complex64)
        out_coh[:] = np.where(mask, coh_sum / weight, 0.0).astype(np.float32)

    # Seam diagnostics
    seam_phase_diffs: list[float] = []
    seam_coh_drops: list[float] = []

    for (seam_line, seam_col, seam_h, seam_w) in seam_regions:
        # Skip out-of-bounds seam regions
        if seam_line < 0 or seam_col < 0:
            logging.warning(
                "Seam region with negative coordinates skipped: (%d, %d, %d, %d)",
                seam_line, seam_col, seam_h, seam_w,
            )
            continue
        if seam_line + seam_h > out_h or seam_col + seam_w > out_w:
            logging.warning(
                "Seam region out of output bounds skipped: "
                "(line=%d, col=%d, h=%d, w=%d) out_shape=%s",
                seam_line, seam_col, seam_h, seam_w, out_shape,
            )
            continue
        if seam_h <= 0 or seam_w <= 0:
            continue

        seam_ifg = out_ifg[seam_line:seam_line + seam_h, seam_col:seam_col + seam_w]
        seam_coh = out_coh[seam_line:seam_line + seam_h, seam_col:seam_col + seam_w]

        # Phase diff: median of phase difference across seam (here: median phase of seam region)
        phase_seam = np.angle(seam_ifg)
        if phase_seam.size > 0:
            seam_phase_diffs.append(float(np.nanmedian(phase_seam)))

        # Coherence drop: mean coherence at seam
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

    # Gap detection: count zero-pixels in output (where weight == 0)
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
    """Plan the placement of each burst's valid window into the merged mosaic.

    Parameters
    ----------
    bursts : sequence of BurstRadarGrid
    valid_windows : sequence of BurstWindow (relative to each burst's SLC)
    out_nlines : int
        Number of lines in the output mosaic.
    out_nsamples : int
        Number of samples in the output mosaic.

    Returns
    -------
    tuple of MergeSegment
        One segment per burst, ordered by increasing ``output_line_start``.
    """
    segments: list[MergeSegment] = []
    for i, (burst, vw) in enumerate(zip(bursts, valid_windows)):
        out_line_start = burst.image_window.first_line + vw.first_line
        out_col_start = vw.first_sample
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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