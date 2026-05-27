"""strip_insar_types — Dataclasses, defaults, and constants for strip_insar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Unwrap defaults
# ---------------------------------------------------------------------------

ICU_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "buffer_lines": 3700,
    "overlap_lines": 200,
    "use_phase_gradient_neutron": False,
    "use_intensity_neutron": False,
    "phase_gradient_window_size": 5,
    "neutron_phase_gradient_threshold": 3.0,
    "neutron_intensity_threshold": 8.0,
    "max_intensity_correlation_threshold": 0.8,
    "trees_number": 7,
    "max_branch_length": 64,
    "pixel_spacing_ratio": 1.0,
    "initial_correlation_threshold": 0.1,
    "max_correlation_threshold": 0.9,
    "correlation_threshold_increments": 0.1,
    "min_tile_area": 0.003125,
    "bootstrap_lines": 16,
    "min_overlap_area": 16,
    "phase_variance_threshold": 8.0,
}

ICU_RELAXED_OVERRIDES: dict[str, Any] = {
    "overlap_lines": 400,
    "use_phase_gradient_neutron": True,
    "phase_gradient_window_size": 7,
    "neutron_phase_gradient_threshold": 1.5,
    "trees_number": 12,
    "max_branch_length": 128,
    "initial_correlation_threshold": 0.015,
    "max_correlation_threshold": 0.55,
    "correlation_threshold_increments": 0.03,
    "min_tile_area": 0.0005,
    "bootstrap_lines": 12,
    "min_overlap_area": 8,
    "phase_variance_threshold": 15.0,
}

ICU_MIN_VALID_FRACTION_FOR_RELAXED_RETRY: float = 0.01

SNAPHU_DEFAULTS: dict[str, Any] = {
    "wavelength": None,
    "earth_radius": 6378137.0,
    "altitude": 6878137.0,
    "max_components": 20,
    "connected_component_cost_threshold": 300,
    "nlooks": 1.0,
    "cost_mode": "defo",
    "initialization_method": "mst",
    "defomax_cycle": 4.0,
    "ntiles": "auto",
    "tile_overlap": (512, 512),
    "nproc": 4,
    "row_overlap": 0,
    "col_overlap": 0,
    "min_conncomp_frac": 0.01,
    "tile_cost_thresh": 300,
    "min_region_size": 300,
    "phase_grad_window": (5, 5),
    "single_tile_reoptimize": False,
    "regrow_conncomps": True,
    "corr_thresh": 0.12,
    "auto_tile_max_pixels": 4_000_000,
}

SNAPHU_RETRY_PROFILES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("default", {}),
    (
        "relaxed",
        {
            "cost_mode": "defo",
            "min_conncomp_frac": 0.001,
            "min_region_size": 100,
            "phase_grad_window": (5, 5),
            "single_tile_reoptimize": False,
        },
    ),
)


# ---------------------------------------------------------------------------
# TOPS burst merge constants
# ---------------------------------------------------------------------------

TOPS_VALID_EDGE_THRESHOLD_LINES: int = 2
TOPS_VALID_EDGE_THRESHOLD_SAMPLES: int = 2

ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT: int = 2000
ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT: int = 1024

GEO2RDR_OFFSET_NODATA: float = -999999.0
GEO2RDR_OFFSET_INVALID_LOW: float = -1.0e5
NISAR_OFFSET_INVALID_VALUE: float = -1.0e6


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TopsBurstInfo:
    """Container for TOPS burst metadata."""
    burst_index: int
    line_offset: int
    number_of_lines: int
    number_of_samples: int
    first_valid_line: int
    num_valid_lines: int
    first_valid_sample: int
    num_valid_samples: int
    sensing_start: str | None = None
    azimuth_time_interval: float | None = None
    radar_wavelength: float | None = None


@dataclass
class TopsOverlapInfo:
    """Container for TOPS burst overlap metadata."""
    previous_burst_index: int
    next_burst_index: int
    estimated_overlap_lines: int
    overlap_start_line: int | None = None


@dataclass
class PairContext:
    """Runtime context for a single master-slave pair."""
    master_manifest_path: Path
    slave_manifest_path: Path
    master_manifest: dict
    slave_manifest: dict
    master_orbit_data: dict
    slave_orbit_data: dict
    master_acq_data: dict
    slave_acq_data: dict
    master_rg_data: dict
    slave_rg_data: dict
    master_dop_data: dict
    slave_dop_data: dict
    output_root: Path
    pair_name: str
    pair_dir: Path
    output_paths: dict[str, str]
    resolved_dem: str
    orbit_interp: str
    wavelength: float
    effective_crop_window: dict | None = None


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_SEQUENCE: tuple[str, ...] = (
    "check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6",
)

STAGE_DIR_NAMES: dict[str, str] = {
    "check": "check",
    "prep": "prep",
    "crop": "crop",
    "p0": "p0_burst_topo",
    "p1": "p1_burst_geo2rdr",
    "p2": "p2_burst_ifg",
    "p3": "p3_burst_merge_unwrap",
    "p4": "p4_geocode",
    "p5": "p5_hdf",
    "p6": "p6_publish",
}

STAGE_LOG_LABELS: dict[str, str] = {
    "p0": "rdr2geo/topo",
    "p1": "resample/registration",
    "p2": "burst-ifg/overlap-ifg",
    "p3": "unwrap",
    "p4": "los",
    "p5": "product/hdf",
    "p6": "export/publish",
}

SUPPORTED_UNWRAP_METHODS: tuple[str, ...] = ("snaphu", "icu", "phass", "dolphin")