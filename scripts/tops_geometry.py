"""tops_geometry — ISCE3 burst geometry adapters.

Converts Sentinel-1 burst metadata (tops_model.BurstRadarGrid) into
ISCE3 C++ bindings: RadarGridParameters, Orbit, Doppler LUT2d, and
Geo2Rdr offsets.

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.tops_model import BurstRadarGrid, Geo2RdrOffsets

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Radar-grid adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class S1RadarGrid:
    """ISCE3-compatible radar grid parameters for one Sentinel-1 TOPS burst.

    Attributes
    ----------
    sensing_start : datetime
        UTC sensing start of the first line.
    wavelength : float
        Radar wavelength in metres (positive, validated).
    prf : float
        Pulse Repetition Frequency in Hz (positive, validated).
    starting_range : float
        Slant range of the first sample in metres.
    range_pixel_spacing : float
        Slant-range spacing between adjacent range samples in metres
        (positive, validated).
    number_of_lines : int
        Number of azimuth (line) samples in the valid window
        (positive, validated).
    number_of_samples : int
        Number of range samples in the valid window
        (positive, validated).
    look_side : str
        Look direction; "right" for Sentinel-1 TOPS (default).

    Methods
    -------
    slant_range_at(sample: int) -> float
        Slant range in metres at the given range sample index.
        Formula: starting_range + sample * range_pixel_spacing
    azimuth_time_at_line(line: int) -> datetime
        UTC datetime of the given azimuth line index.
        Formula: sensing_start + line / prf  (seconds)
    """

    sensing_start: datetime
    wavelength: float            # metres  (> 0)
    prf: float                   # Hz      (> 0)
    starting_range: float       # metres
    range_pixel_spacing: float  # metres  (> 0)
    number_of_lines: int        #         (> 0)
    number_of_samples: int      #         (> 0)
    look_side: str = "right"

    def __post_init__(self) -> None:
        if not isinstance(self.sensing_start, datetime):
            raise TypeError("sensing_start must be a datetime instance")
        if self.wavelength <= 0.0:
            raise ValueError(f"wavelength must be positive, got {self.wavelength}")
        if self.prf <= 0.0:
            raise ValueError(f"prf must be positive, got {self.prf}")
        if self.range_pixel_spacing <= 0.0:
            raise ValueError(f"range_pixel_spacing must be positive, got {self.range_pixel_spacing}")
        if self.number_of_lines <= 0:
            raise ValueError(f"number_of_lines must be positive, got {self.number_of_lines}")
        if self.number_of_samples <= 0:
            raise ValueError(f"number_of_samples must be positive, got {self.number_of_samples}")

    def slant_range_at(self, sample: int) -> float:
        """Slant range in metres at range sample index ``sample``.

        Formula:  slant_range = starting_range + sample * range_pixel_spacing
        """
        return self.starting_range + sample * self.range_pixel_spacing

    def azimuth_time_at_line(self, line: int) -> datetime:
        """UTC datetime of azimuth line index ``line``.

        Formula:  t(line) = sensing_start + line / prf  (seconds)
        """
        return self.sensing_start + timedelta(seconds=line / self.prf)


def burst_to_radar_grid(burst: BurstRadarGrid) -> S1RadarGrid:
    """Convert a tops_model.BurstRadarGrid to an ISCE3-compatible S1RadarGrid.

    Parameters
    ----------
    burst : BurstRadarGrid
        Burst metadata parsed from Sentinel-1 annotation XML.

    Returns
    -------
    S1RadarGrid
        ISCE3-compatible radar-grid parameter object.

    Raises
    ------
    ValueError
        If azimuth_time_interval ≤ 0 (prf would be non-positive).

    Key derivations
    ---------------
    - ``prf = 1.0 / azimuth_time_interval``
    - ``number_of_lines`` and ``number_of_samples`` are taken from
      ``burst.valid_window`` to represent the ISCE3 RadarGridParameters
      valid extent.
    """
    if burst.azimuth_time_interval <= 0.0:
        raise ValueError(
            f"azimuth_time_interval must be positive for prf computation, "
            f"got {burst.azimuth_time_interval}"
        )
    return S1RadarGrid(
        sensing_start=burst.identity.sensing_start,
        wavelength=burst.radar_wavelength,
        prf=1.0 / burst.azimuth_time_interval,
        starting_range=burst.starting_range,
        range_pixel_spacing=burst.range_pixel_spacing,
        number_of_lines=burst.valid_window.num_lines,
        number_of_samples=burst.valid_window.num_samples,
        look_side="right",
    )


# ---------------------------------------------------------------------------
# Orbit builder  (Spike — NotImplementedError with ISCE3 calling path)
# ---------------------------------------------------------------------------

def build_isce3_orbit_from_safe(
    safe_path: Path,
    t0: datetime,
    t1: datetime,
) -> Any:
    """Build an ISCE3 core.Orbit from Sentinel-1 orbit state vectors in a SAFE.

    Parameters
    ----------
    safe_path : Path
        Path to the Sentinel-1 SAFE directory (or unpacked ZIP root).
    t0 : datetime
        Sensing start of the burst window (UTC).
    t1 : datetime
        Sensing stop of the burst window (UTC).

    Returns
    -------
    isce3.core.Orbit
        Interpolated orbit covering [t0, t1] with margin.

    Raises
    ------
    NotImplementedError
        This is a spike stub.  Full implementation steps:

    **ISCE3 calling path**::

        from isce3.core import Orbit, StateVector, DateTime
        from os import path as os_path

        # 1. Locate the POD orbit file inside the SAFE.
        #    Sentinel-1 Level-1 SLC orbit data lives in:
        #    {safe_path}/aux/POEORB*.EOF   (precise orbit)
        #    {safe_path}/aux/_RESORB*.EOF  (restituted orbit, fallback)
        orbit_files = sorted(safe_path.glob("aux/POEORB*.EOF"))
        if not orbit_files:
            orbit_files = sorted(safe_path.glob("aux/RESORB*.EOF"))
        orbit_file = orbit_files[-1]  # most recent

        # 2. Parse the EOF into a list of StateVector tuples:
        #    [(datetime, x, y, z, vx, vy, vz), ...]
        #    Use D2SAR/scripts/sentinel_orbit.py _parse_eof() as reference.
        state_vectors = _parse_eof(orbit_file)  # stub

        # 3. Convert datetime → isce3.core.DateTime.
        isce3_orbit = Orbit()
        for dt, x, y, z, vx, vy, vz in state_vectors:
            t_isce3 = DateTime(dt.year, dt.month, dt.day,
                               dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
            sv = StateVector(t_isce3, [x, y, z], [vx, vy, vz])
            isce3_orbit.append(sv)

        # 4. Trim/or extend to cover [t0 - margin, t1 + margin].
        #    isce3.core.Orbit does not support trim; slice by index or
        #    rebuild filtered orbit.
        return isce3_orbit

    **Dependencies**: ``isce3.core.Orbit``, ``isce3.core.StateVector``,
    ``isce3.core.DateTime``.
    """
    raise NotImplementedError(
        "build_isce3_orbit_from_safe is a spike stub.\n"
        "Implement by parsing Sentinel-1 POEORB/RESORB EOF files into "
        "isce3.core.Orbit objects.  See ISCE3 calling path in docstring."
    )


# ---------------------------------------------------------------------------
# Doppler LUT builder  (Spike — NotImplementedError with ISCE3 calling path)
# ---------------------------------------------------------------------------

def build_doppler_lut(burst: BurstRadarGrid) -> Any:
    """Build an ISCE3 core.LUT2d from Sentinel-1 annotation Doppler coefficients.

    Parameters
    ----------
    burst : BurstRadarGrid
        Burst metadata containing ``doppler_coefficients`` and
        ``radar_wavelength``.

    Returns
    -------
    isce3.core.LUT2d
        2-D Doppler look-up table: f_D(range_time, azimuth_time).

    Raises
    ------
    NotImplementedError
        This is a spike stub.  Full implementation steps:

    **ISCE3 calling path**::

        from isce3.core import LUT2d, DateTime, UnixTime
        import numpy as np

        # 1. Extract coefficients from BurstRadarGrid.
        #    Sentinel-1 annotation XML provides a 1-D Doppler polynomial:
        #        f_D(s_rg) = c0 + c1*s_rg + c2*s_rg^2 + ...
        #    where s_rg is slant-range distance (m) relative to starting_range.
        coeffs = burst.doppler_coefficients   # Tuple[float, ...]
        wavelength = burst.radar_wavelength   # metres

        # 2. Convert to ISCE3 LUT2d.  ISCE3 expects f_D in Hz at
        #    reference epoch (first sample, sensing_start).
        #    Build a regular grid in (range, azimuth) and evaluate the poly.
        ref_slant_range = burst.starting_range
        slant_step = burst.range_pixel_spacing
        n_rg = burst.valid_window.num_samples
        n_az = burst.valid_window.num_lines

        range_vec = np.array([
            ref_slant_range + s * slant_step for s in range(n_rg)
        ])

        # Evaluate polynomial: f_D(s_rg) in Hz
        f_doppler_vec = np.polyval(coeffs[::-1], range_vec)

        # 3. ISCE3 LUT2d is indexed by (range_time, azimuth_time).
        #    For a 1-D polynomial the azimuth dependence is zero
        #    (constant across lines), so build a 2-D LUT as:
        #    lut.set_values(f_doppler_vec.reshape(1, -1), rgrid_time, az_time)
        #    where rgrid_time = (range_vec - ref_slant_range) / c * 2
        #    and c = speed of light ≈ 299792458 m/s.
        c = 299792458.0
        rgrid_time = (range_vec - ref_slant_range) * 2.0 / c  # two-way time

        t0 = burst.identity.sensing_start
        az_time_vec = np.array([
            (t0 + timedelta(seconds=l / (1.0 / burst.azimuth_time_interval)))
            .timestamp() for l in range(n_az)
        ])

        lut = LUT2d()
        lut.set_values(f_doppler_vec.reshape(1, -1), rgrid_time, az_time_vec)
        return lut

    **Dependencies**: ``isce3.core.LUT2d``, ``isce3.core.DateTime``.
    """
    raise NotImplementedError(
        "build_doppler_lut is a spike stub.\n"
        "Implement by converting BurstRadarGrid.doppler_coefficients to "
        "isce3.core.LUT2d.  See ISCE3 calling path in docstring."
    )


# ---------------------------------------------------------------------------
# Geo2Rdr single-burst  (Spike — NotImplementedError with ISCE3 calling path)
# ---------------------------------------------------------------------------

def run_geo2rdr_single_burst(
    ref: BurstRadarGrid,
    sec: BurstRadarGrid,
    dem_path: Path,
    work_dir: Path,
    *,
    use_gpu: bool = False,
) -> Geo2RdrOffsets:
    """Run ISCE3 Geo2Rdr for one reference / secondary burst pair.

    Parameters
    ----------
    ref : BurstRadarGrid
        Reference (master) burst radar grid.
    sec : BurstRadarGrid
        Secondary (slave) burst radar grid.
    dem_path : Path
        Path to the reference DEM GeoTIFF.
    work_dir : Path
        Working directory where offset files are written.
    use_gpu : bool, default False
        Whether to attempt GPU-accelerated Geo2Rdr (isce3.cuda.geometry.Geo2Rdr).

    Returns
    -------
    Geo2RdrOffsets
        Range and azimuth offsets, plus median statistics and valid-sample count.

    Raises
    ------
    NotImplementedError
        This is a spike stub.  Full implementation steps:

    **ISCE3 calling path**::

        from isce3.geometry import Geo2Rdr
        from isce3.core import LUT2d
        from nisar.workflows.helpers import get_radar_grid_cpp_args
        from os import path as os_path

        # 1. Build ISCE3 RadarGridParameters for both bursts.
        #    Use the existing NISAR helper or call the C++ extension directly:
        ref_radar_grid = get_radar_grid_cpp_args(ref_burst_s1radargrid)
        sec_radar_grid = get_radar_grid_cpp_args(sec_burst_s1radargrid)

        # 2. Build Orbit objects.
        #    (delegate to build_isce3_orbit_from_safe once spike is resolved)
        ref_orbit = build_isce3_orbit_from_safe(safe_path, ref.identity.sensing_start,
                                                ref.identity.sensing_stop)
        sec_orbit = build_isce3_orbit_from_safe(safe_path, sec.identity.sensing_start,
                                                sec.identity.sensing_stop)

        # 3. Build Doppler LUT2d.
        #    (delegate to build_doppler_lut once spike is resolved)
        ref_doppler = build_doppler_lut(ref)
        sec_doppler = build_doppler_lut(sec)

        # 4. Open DEM.
        dem_raster = isce3.io.Raster(str(dem_path))

        # 5. Configure Geo2Rdr.
        geo2rdr = Geo2Rdr()
        geo2rdr.dem_raster = dem_raster
        geo2rdr.orbit_ref = ref_orbit
        geo2rdr.orbit_sec = sec_orbit
        geo2rdr.doppler_ref = ref_doppler
        geo2rdr.doppler_sec = sec_doppler
        geo2rdr.threshold_geo2rdr = 1e-4      # ISCE3 default
        geo2rdr.num_iter = 100                # ISCE3 default
        geo2rdr.lines_per_block = 500
        geo2rdr.use_gpu = use_gpu

        # 6. Prepare output rasters.
        work_dir.mkdir(parents=True, exist_ok=True)
        range_off_path = work_dir / "range.off"
        azimuth_off_path = work_dir / "azimuth.off"
        range_off_raster = isce3.io.Raster(str(range_off_path),
                                            ref_radar_grid.number_of_samples,
                                            ref_radar_grid.number_of_lines, 1,
                                            isce3.io.gdal.GDT_Float32, "ENVI")
        azimuth_off_raster = isce3.io.Raster(str(azimuth_off_path),
                                              ref_radar_grid.number_of_samples,
                                              ref_radar_grid.number_of_lines, 1,
                                              isce3.io.gdal.GDT_Float32, "ENVI")

        # 7. Run coregistration.
        geo2rdr.offset_hdr.set_range_off_path(str(range_off_path))
        geo2rdr.offset_hdr.set_azimuth_off_path(str(azimuth_off_path))
        geo2rdr.coregister(ref_radar_grid, sec_radar_grid,
                            range_off_raster, azimuth_off_raster)

        # 8. Compute median offsets (skip NaN).
        range_off_arr = range_off_raster.data().astype(float)
        az_off_arr = azimuth_off_raster.data().astype(float)
        valid_mask = np.isfinite(range_off_arr) & np.isfinite(az_off_arr)
        median_range = float(np.nanmedian(range_off_arr[valid_mask]))
        median_az = float(np.nanmedian(az_off_arr[valid_mask]))
        valid_count = int(valid_mask.sum())

        return Geo2RdrOffsets(
            range_off_path=str(range_off_path),
            azimuth_off_path=str(azimuth_off_path),
            median_range_offset=median_range,
            median_azimuth_offset=median_az,
            valid_sample_count=valid_count,
        )

    **Output files written to work_dir**:
    - ``range.off``   — range direction offset grid (metres or pixels, ISCE3 convention)
    - ``azimuth.off`` — azimuth direction offset grid (seconds or pixels)

    **Validation**: median_range_offset and median_azimuth_offset must be finite
    and within reasonable bounds (e.g. |median_range| < 1000 m, |median_az| < 10 s).

    **Dependencies**: ``isce3.geometry.Geo2Rdr``, ``isce3.core.LUT2d``,
    ``isce3.core.Orbit``, ``isce3.io.Raster``.
    """
    raise NotImplementedError(
        "run_geo2rdr_single_burst is a spike stub.\n"
        "Implement by calling isce3.geometry.Geo2Rdr.coregister(). "
        "See ISCE3 calling path in docstring."
    )
