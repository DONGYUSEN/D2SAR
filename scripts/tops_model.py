"""tops_model — Immutable dataclass definitions for Sentinel-1 TOPS InSAR burst processing.

All objects are frozen (immutable) to prevent accidental mutation in the
pipeline.  No external dependencies (ISCE3, numpy, etc.) — pure stdlib only.
"""

from __future__ import annotations

__all__ = [
    "BurstIdentity",
    "BurstWindow",
    "BurstRadarGrid",
    "CommonBurstPair",
    "CommonBurstSelection",
    "Geo2RdrOffsets",
    "OverlapSlice",
    "OverlapPair",
    "EsdEstimate",
    "TimingCorrection",
    "MergeSegment",
    "MergeResult",
    "RangeCoregEstimate",
]

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Tuple


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Identity / Window primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BurstIdentity:
    """Uniquely identifies a single Sentinel-1 TOPS burst."""
    swath: str                       # "IW1" | "IW2" | "IW3"
    burst_index: int                 # 0-based index within the swath
    sensing_start: datetime          # timezone-aware UTC
    sensing_stop: datetime            # timezone-aware UTC
    polarization: str                # "VV" | "VH" | "HH" | "HV"
    orbit_direction: str             # "ascending" | "descending"
    azimuth_steering_rate: float     # rad/s (expected ≈ ±0.0018 rad/s for TOPS)

    def __post_init__(self) -> None:
        if not isinstance(self.swath, str) or not self.swath:
            raise ValueError("swath must be a non-empty string")
        if self.burst_index < 0:
            raise ValueError(f"burst_index must be non-negative, got {self.burst_index}")
        if self.sensing_stop <= self.sensing_start:
            raise ValueError("sensing_stop must be strictly after sensing_start")


@dataclass(frozen=True)
class BurstWindow:
    """Pixel-coordinate window (relative/absolute semantics determined by caller)."""
    first_line: int
    num_lines: int
    first_sample: int
    num_samples: int

    def __post_init__(self) -> None:
        if self.num_lines < 0:
            raise ValueError(f"num_lines must be non-negative, got {self.num_lines}")
        if self.num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {self.num_samples}")

    @property
    def line_stop(self) -> int:
        """One-past-the-last line (first_line + num_lines)."""
        return self.first_line + self.num_lines

    @property
    def sample_stop(self) -> int:
        """One-past-the-last sample (first_sample + num_samples)."""
        return self.first_sample + self.num_samples


# ---------------------------------------------------------------------------
# Radar-grid container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BurstRadarGrid:
    """Complete burst-level radar geometry parameters.

    This is the primary "burst object" carried through all pipeline stages.
    The ``identity`` field carries the burst's identity; ``image_window``
    describes the full burst image extent; ``valid_window`` describes the
    sub-region of non-zero padding within the burst.
    """
    identity: BurstIdentity
    image_window: BurstWindow        # full burst image window
    valid_window: BurstWindow       # non-padded SLC region inside image_window
    line_offset: int              # line offset of this burst within the full measurement image
    azimuth_time_interval: float   # seconds per line (≈ 1/PRF)
    range_pixel_spacing: float    # metres
    starting_range: float          # metres
    radar_wavelength: float        # metres
    doppler_coefficients: Tuple[float, ...]
    azimuth_fm_rate_coefficients: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.azimuth_time_interval <= 0.0:
            raise ValueError(
                f"azimuth_time_interval must be positive, got {self.azimuth_time_interval}"
            )
        if self.range_pixel_spacing <= 0.0:
            raise ValueError(
                f"range_pixel_spacing must be positive, got {self.range_pixel_spacing}"
            )
        if self.starting_range <= 0.0:
            raise ValueError(
                f"starting_range must be positive, got {self.starting_range}"
            )
        if self.radar_wavelength <= 0.0:
            raise ValueError(
                f"radar_wavelength must be positive, got {self.radar_wavelength}"
            )

    # ------------------------------------------------------------------ properties

    @property
    def prf(self) -> float:
        """Pulse Repetion Frequency (Hz)."""
        return 1.0 / self.azimuth_time_interval

    @property
    def duration(self) -> float:
        """Burst duration in seconds."""
        delta: timedelta = self.identity.sensing_stop - self.identity.sensing_start
        return delta.total_seconds()

    @property
    def valid_line_start(self) -> int:
        """Absolute line of first valid pixel in the full measurement image."""
        return self.image_window.first_line + self.valid_window.first_line

    @property
    def valid_line_stop(self) -> int:
        """One-past-the-last valid line in the full measurement image."""
        return self.valid_line_start + self.valid_window.num_lines

    def slant_range_at(self, sample: int) -> float:
        """Slant range in metres at the given sample index."""
        return self.starting_range + sample * self.range_pixel_spacing

    def azimuth_time_at_line(self, line: int) -> datetime:
        """UTC datetime of the given line within this burst."""
        delta_seconds = line * self.azimuth_time_interval
        return self.identity.sensing_start + timedelta(seconds=delta_seconds)


# ---------------------------------------------------------------------------
# Common-burst selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonBurstPair:
    """One pair of reference / secondary bursts at the same scene location."""
    pair_index: int          # 0-based index within the common-burst selection
    reference: BurstRadarGrid
    secondary: BurstRadarGrid
    burst_offset: int       # secondary_index = reference_index + burst_offset

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ValueError(
                f"pair_index must be non-negative, got {self.pair_index}"
            )


@dataclass(frozen=True)
class CommonBurstSelection:
    """All common-burst pairs for one swath.

    The pairs are stored in order of increasing burst index.
    """
    swath: str
    reference_start_index: int
    secondary_start_index: int
    number_of_common_bursts: int
    pairs: Tuple[CommonBurstPair, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.number_of_common_bursts < 0:
            raise ValueError(
                f"number_of_common_bursts must be non-negative, "
                f"got {self.number_of_common_bursts}"
            )


# ---------------------------------------------------------------------------
# Overlap slices
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlapSlice:
    """One overlap region (top or bottom half) between two adjacent bursts.

    Coordinates are absolute (relative to the full measurement image) unless
    documented otherwise.  ``is_top`` is True when this slice belongs to the
    earlier (top) burst in the pair.
    """
    burst_pair: CommonBurstPair
    is_top: bool
    first_line: int         # absolute line in full measurement image
    num_lines: int
    first_sample: int       # absolute sample
    num_samples: int
    sensing_start: datetime
    sensing_stop: datetime

    def __post_init__(self) -> None:
        if self.num_lines < 0:
            raise ValueError(f"num_lines must be non-negative, got {self.num_lines}")
        if self.num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {self.num_samples}")
        if self.sensing_stop <= self.sensing_start:
            raise ValueError("sensing_stop must be strictly after sensing_start")


@dataclass(frozen=True)
class OverlapPair:
    """Top + bottom overlap slices for one adjacent-burst pair."""
    pair_index: int
    top: OverlapSlice       # overlap region from the top (earlier) burst
    bottom: OverlapSlice   # overlap region from the bottom (later) burst

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ValueError(f"pair_index must be non-negative, got {self.pair_index}")


# ---------------------------------------------------------------------------
# Geo2Rdr / range coregistration results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geo2RdrOffsets:
    """Range and azimuth offsets from the Geo2Rdr coregistration step."""
    range_off_path: str                 # path to range offset raster on disk
    azimuth_off_path: str              # path to azimuth offset raster on disk
    median_range_offset: float         # median range offset (metres or pixels — caller decides)
    median_azimuth_offset: float      # median azimuth offset (metres or pixels)
    valid_sample_count: int           # number of valid (non-NaN) samples used

    def __post_init__(self) -> None:
        if self.valid_sample_count < 0:
            raise ValueError(
                f"valid_sample_count must be non-negative, got {self.valid_sample_count}"
            )


@dataclass(frozen=True)
class RangeCoregEstimate:
    """Range (and azimuth) residual coregistration estimate from overlap interferogram.

    Attributes
    ----------
    median_range_offset : float
        Robust median of range-direction offset estimates (pixels).
    std_range_offset : float
        Robust standard deviation of range-direction offset estimates (pixels).
    median_azimuth_offset : float
        Robust median of azimuth-direction offset estimates (pixels).
    std_azimuth_offset : float
        Robust standard deviation of azimuth-direction offset estimates (pixels).
    sample_count : int
        Number of valid (coherent, in-bounds) pixels used in the estimate.
    usable_fraction : float
        Fraction of coherent pixels that passed outlier rejection (0–1).
    """
    median_range_offset: float
    std_range_offset: float
    median_azimuth_offset: float
    std_azimuth_offset: float
    sample_count: int
    usable_fraction: float

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError(
                f"sample_count must be non-negative, got {self.sample_count}"
            )
        if not (0.0 <= self.usable_fraction <= 1.0):
            raise ValueError(
                f"usable_fraction must be in [0,1], got {self.usable_fraction}"
            )


# ---------------------------------------------------------------------------
# ESD timing correction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EsdEstimate:
    """Estimate of azimuth misregistration in pixels from overlap interferogram ESD."""
    median_offset_pixels: float
    mean_offset_pixels: float
    std_offset_pixels: float
    sample_count: int
    azimuth_time_interval: float       # copy of the burst azimuth_time_interval (s/line)
    mean_coherence: float = 1.0        # mean coherence over the ESD overlap window

    def __post_init__(self) -> None:
        if not (-1.0 <= self.mean_coherence <= 1.0):
            raise ValueError(
                f"mean_coherence must be in [-1, 1], got {self.mean_coherence}"
            )
        if self.sample_count < 0:
            raise ValueError(
                f"sample_count must be non-negative, got {self.sample_count}"
            )
        if self.azimuth_time_interval <= 0.0:
            raise ValueError(
                f"azimuth_time_interval must be positive, got {self.azimuth_time_interval}"
            )


@dataclass(frozen=True)
class TimingCorrection:
    """Secondary timing correction in time and pixel units."""
    secondary_timing_seconds: float   # correction in seconds (to add to secondary timing)
    secondary_timing_pixels: float   # correction in pixels (median_offset * azimuth_interval)
    esd_estimate: EsdEstimate


# ---------------------------------------------------------------------------
# Merge / mosaic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeSegment:
    """Coordinates for placing one burst's valid window into the merged mosaic.

    ``input_*`` fields are relative to the full measurement image.
    ``output_*`` fields are relative to the merged output array.
    """
    burst_index: int
    pair_index: int
    input_line_start: int
    input_num_lines: int
    input_sample_start: int
    input_num_samples: int
    output_line_start: int
    output_num_lines: int
    output_sample_start: int
    output_num_samples: int

    def __post_init__(self) -> None:
        for name, val in [
            ("input_num_lines", self.input_num_lines),
            ("input_num_samples", self.input_num_samples),
            ("output_num_lines", self.output_num_lines),
            ("output_num_samples", self.output_num_samples),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")


@dataclass(frozen=True)
class MergeResult:
    """Result of merging per-burst interferograms into a full-swath mosaic."""
    seam_phase_diff_median: float
    seam_phase_diff_std: float
    seam_coherence_drop: float
    gap_pixel_count: int
    top_contribution_count: int       # total pixels contributed by top-burst regions
    bottom_contribution_count: int   # total pixels contributed by bottom-burst regions
    segments: Tuple[MergeSegment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.gap_pixel_count < 0:
            raise ValueError(
                f"gap_pixel_count must be non-negative, got {self.gap_pixel_count}"
            )
