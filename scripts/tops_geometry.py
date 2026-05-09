"""tops_geometry — ISCE3 burst geometry adapters.

Converts Sentinel-1 burst metadata (tops_model.BurstRadarGrid) into
ISCE3 C++ bindings: RadarGridParameters, Orbit, Doppler LUT2d, and
Geo2Rdr offsets.

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.tops_model import BurstRadarGrid, Geo2RdrOffsets

UTC = timezone.utc
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISCE3 lazy import helper
# ---------------------------------------------------------------------------

def _get_isce3():
    """Try to import isce3 with lazy path injection.

    Returns
    -------
    module
        The isce3 module.

    Raises
    ------
    NotImplementedError
        If isce3 C++ bindings are not available.
    """
    try:
        import isce3  # noqa: F401
        # Check if C++ bindings are available by trying to access a core class
        from isce3.core import DateTime  # noqa: F401
        return isce3
    except (ImportError, AttributeError):
        pass

    # Try injecting the ISCE3 python path
    isce3_path = Path(__file__).parents[1] / "isce3" / "python"
    if str(isce3_path) not in sys.path:
        sys.path.insert(0, str(isce3_path))

    try:
        import isce3  # noqa: F401
        from isce3.core import DateTime  # noqa: F401
        return isce3
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "isce3 C++ bindings are not available. "
            "Build ISCE3 with pybind11 extensions first. "
            f"Import error: {exc}"
        ) from exc


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
# Orbit builder
# ---------------------------------------------------------------------------

# IMPLEMENTED: build_isce3_orbit_from_safe


def _parse_eof_datetime(text: str) -> datetime:
    """Parse EOF datetime string like 'UTC=2023-06-24T22:59:42.000000'.

    Parameters
    ----------
    text : str
        EOF datetime string with 'UTC=' prefix.

    Returns
    -------
    datetime
        Parsed UTC datetime.
    """
    # Strip UTC= prefix and parse
    value = text.strip()
    if value.startswith("UTC="):
        value = value[4:]
    elif value.startswith("TAI="):
        value = value[4:]
    # Parse with microseconds
    return datetime.fromisoformat(value)


def _parse_eof_state_vectors(eof_path: Path) -> list[tuple[datetime, float, float, float, float, float, float]]:
    """Parse Sentinel-1 EOF orbit file into state vector tuples.

    EOF format (Sentinel-1 POEORB/RESORB)::
        <OSV>
          <UTC>...</UTC>
          <X>...</X><Y>...</Y><Z>...</Z>
          <VX>...</VX><VY>...</VY><VZ>...</VZ>
          <Quality>...</Quality>
        </OSV>

    Parameters
    ----------
    eof_path : Path
        Path to the EOF orbit file.

    Returns
    -------
    list of tuple
        List of (datetime, x, y, z, vx, vy, vz) state vectors sorted by time.
    """
    tree = ET.parse(eof_path)
    root = tree.getroot()

    state_vectors = []
    for osv in root.iter("OSV"):
        utc_elem = osv.find("UTC")
        x_elem = osv.find("X")
        y_elem = osv.find("Y")
        z_elem = osv.find("Z")
        vx_elem = osv.find("VX")
        vy_elem = osv.find("VY")
        vz_elem = osv.find("VZ")

        utc_text = utc_elem.text if utc_elem is not None else "UTC=1970-01-01T00:00:00.000000"
        dt = _parse_eof_datetime(utc_text)  # type: ignore[arg-type]
        x = float(x_elem.text) if x_elem is not None else 0.0  # type: ignore[arg-type]
        y = float(y_elem.text) if y_elem is not None else 0.0  # type: ignore[arg-type]
        z = float(z_elem.text) if z_elem is not None else 0.0  # type: ignore[arg-type]
        vx = float(vx_elem.text) if vx_elem is not None else 0.0  # type: ignore[arg-type]
        vy = float(vy_elem.text) if vy_elem is not None else 0.0  # type: ignore[arg-type]
        vz = float(vz_elem.text) if vz_elem is not None else 0.0  # type: ignore[arg-type]
        state_vectors.append((dt, x, y, z, vx, vy, vz))

    # Sort by datetime
    state_vectors.sort(key=lambda sv: sv[0])
    return state_vectors


def _find_eof_file(safe_path: Path) -> Path | None:
    """Find the best available EOF orbit file for a SAFE product.

    Searches for POEORB first (precise orbit), then falls back to RESORB
    (restituted orbit). Returns the most recent file by modification time.

    Parameters
    ----------
    safe_path : Path
        Path to the Sentinel-1 SAFE directory.

    Returns
    -------
    Path or None
        Path to the EOF file, or None if not found.
    """
    # Try POEORB first (precise orbit)
    poeorb_files = sorted(safe_path.glob("aux/*POEORB*.EOF"))
    if poeorb_files:
        return max(poeorb_files, key=lambda p: p.stat().st_mtime)

    # Fall back to RESORB (restituted orbit)
    resorb_files = sorted(safe_path.glob("aux/*RESORB*.EOF"))
    if resorb_files:
        return max(resorb_files, key=lambda p: p.stat().st_mtime)

    return None


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
        Interpolated orbit covering [t0, t1] with ±120s margin.

    Raises
    ------
    NotImplementedError
        If ISCE3 C++ bindings are not available.
    FileNotFoundError
        If no EOF orbit file is found in the SAFE.
    """
    isce3 = _get_isce3()
    from isce3.core import DateTime, StateVector, Orbit, OrbitInterpMethod

    # Step 1: Find the best available EOF file
    eof_path = _find_eof_file(safe_path)
    if eof_path is None:
        raise FileNotFoundError(
            f"No POEORB or RESORB EOF file found in {safe_path}/aux/"
        )

    LOG.debug("Using orbit file: %s", eof_path)

    # Step 2: Parse EOF into state vectors
    state_vectors = _parse_eof_state_vectors(eof_path)
    if not state_vectors:
        raise ValueError(f"No state vectors found in EOF file: {eof_path}")

    # Step 3: Convert to ISCE3 Orbit
    # Reference epoch is the first state vector datetime
    first_dt = state_vectors[0][0]
    ref_epoch = DateTime(
        first_dt.year, first_dt.month, first_dt.day,
        first_dt.hour, first_dt.minute,
        first_dt.second + first_dt.microsecond * 1e-6
    )

    orbit = Orbit([], ref_epoch, OrbitInterpMethod.SCHEDULE_EXT_ENDPOINT)

    for dt, x, y, z, vx, vy, vz in state_vectors:
        t_isce3 = DateTime(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute,
            dt.second + dt.microsecond * 1e-6
        )
        sv = StateVector(t_isce3, [x, y, z], [vx, vy, vz])
        orbit.append(sv)

    # Step 4: The orbit already covers a time range from the EOF file.
    # We don't need explicit trimming since the Geo2Rdr will use
    # orbit.interpolate() for any time within the orbit's span.
    LOG.debug(
        "Built orbit with %d state vectors, ref_epoch=%s",
        orbit.size, ref_epoch
    )

    return orbit


# ---------------------------------------------------------------------------
# Doppler LUT builder
# ---------------------------------------------------------------------------

# IMPLEMENTED: build_doppler_lut


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
        If ISCE3 C++ bindings are not available.
    """
    isce3 = _get_isce3()
    from isce3.core import DateTime, LUT2d, TimeDelta

    # Step 1: Extract coefficients from BurstRadarGrid.
    # Sentinel-1 annotation XML provides a 1-D Doppler polynomial:
    #     f_D(s_rg) = c0 + c1*s_rg + c2*s_rg^2 + ...
    # where s_rg is slant-range distance (m) relative to starting_range.
    coeffs = burst.doppler_coefficients  # Tuple[float, ...]

    # Step 2: Build slant range vector
    ref_slant_range = burst.starting_range
    slant_step = burst.range_pixel_spacing
    n_rg = burst.valid_window.num_samples
    n_az = burst.valid_window.num_lines

    range_vec = np.array([
        ref_slant_range + s * slant_step for s in range(n_rg)
    ])

    # Evaluate polynomial: f_D(s_rg) in Hz
    # numpy.polyval expects coefficients in descending order
    if coeffs:
        f_doppler_vec = np.polyval(coeffs[::-1], range_vec)
    else:
        f_doppler_vec = np.zeros(n_rg)

    # Step 3: Build ISCE3 LUT2d indexed by (range_time, azimuth_time)
    # For a 1-D polynomial the azimuth dependence is zero
    # (constant across lines), so build a 2-D LUT as:
    # lut.set_values(f_doppler_vec.reshape(1, -1), rgrid_time, az_time)
    # where rgrid_time = (range_vec - ref_slant_range) / c * 2
    # and c = speed of light ≈ 299792458 m/s.
    c = 299792458.0  # speed of light in m/s
    rgrid_time = (range_vec - ref_slant_range) * 2.0 / c  # two-way time in seconds

    t0 = burst.identity.sensing_start
    prf = 1.0 / burst.azimuth_time_interval
    az_time_vec = np.array([
        (t0 + timedelta(seconds=l / prf)).timestamp()
        for l in range(n_az)
    ])

    # Reshape f_doppler to 2D: (n_az, n_rg) where values are constant across azimuth
    f_doppler_2d = np.tile(f_doppler_vec, (n_az, 1))

    # Create LUT2d: slant range (1st arg), azimuth time (2nd arg), values (3rd arg)
    lut = LUT2d(rgrid_time, az_time_vec, f_doppler_2d)

    LOG.debug(
        "Built Doppler LUT2d: shape=%s, range=[%.2f, %.2f] s, "
        "azimuth=[%.2f, %.2f] s",
        f_doppler_2d.shape, rgrid_time[0], rgrid_time[-1],
        az_time_vec[0], az_time_vec[-1]
    )

    return lut


# ---------------------------------------------------------------------------
# Geo2Rdr single-burst
# ---------------------------------------------------------------------------

# IMPLEMENTED: run_geo2rdr_single_burst


def run_geo2rdr_single_burst(
    ref: BurstRadarGrid,
    sec: BurstRadarGrid,
    dem_path: Path,
    work_dir: Path,
    safe_path: Path | None = None,
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
    safe_path : Path, optional
        Path to the Sentinel-1 SAFE directory for orbit file lookup.
        If not provided, attempts to find orbit files in standard locations.
    use_gpu : bool, default False
        Whether to attempt GPU-accelerated Geo2Rdr (isce3.cuda.geometry.Geo2Rdr).

    Returns
    -------
    Geo2RdrOffsets
        Range and azimuth offsets, plus median statistics and valid-sample count.

    Raises
    ------
    NotImplementedError
        If ISCE3 C++ bindings are not available.
    FileNotFoundError
        If required orbit or DEM files are not found.
    RuntimeError
        If Geo2Rdr processing fails.

    **Output files written to work_dir**:
    - ``range.off``   — range direction offset grid
    - ``azimuth.off`` — azimuth direction offset grid

    **Dependencies**: ``isce3.geometry.Geo2Rdr``, ``isce3.core.LUT2d``,
    ``isce3.core.Orbit``, ``isce3.io.Raster``.
    """
    isce3 = _get_isce3()
    from isce3.core import DateTime, LookSide, Orbit, LUT2d
    from isce3.product import RadarGridParameters
    from isce3.geometry import Geo2Rdr

    # Step 1: Build ISCE3 RadarGridParameters for both bursts
    # Reference epoch is the sensing start of the reference burst
    ref_epoch = DateTime(
        ref.identity.sensing_start.year,
        ref.identity.sensing_start.month,
        ref.identity.sensing_start.day,
        ref.identity.sensing_start.hour,
        ref.identity.sensing_start.minute,
        ref.identity.sensing_start.second + ref.identity.sensing_start.microsecond * 1e-6
    )

    prf = 1.0 / ref.azimuth_time_interval

    ref_radar_grid = RadarGridParameters(
        0.0,  # sensingStart_seconds_since_epoch (relative to ref_epoch)
        ref.radar_wavelength,
        prf,
        ref.starting_range,
        ref.range_pixel_spacing,
        LookSide.Right,
        ref.valid_window.num_lines,
        ref.valid_window.num_samples,
        ref_epoch
    )

    sec_radar_grid = RadarGridParameters(
        (sec.identity.sensing_start - ref.identity.sensing_start).total_seconds(),
        sec.radar_wavelength,
        prf,
        sec.starting_range,
        sec.range_pixel_spacing,
        LookSide.Right,
        sec.valid_window.num_lines,
        sec.valid_window.num_samples,
        ref_epoch
    )

    # Step 2: Build Orbit objects for both bursts
    if safe_path is None:
        # Try to find the SAFE path from the work directory or common locations
        safe_path = work_dir.parent  # Default fallback

    t0_ref = ref.identity.sensing_start
    t1_ref = ref.identity.sensing_stop
    t0_sec = sec.identity.sensing_start
    t1_sec = sec.identity.sensing_stop

    ref_orbit = build_isce3_orbit_from_safe(safe_path, t0_ref, t1_ref)
    sec_orbit = build_isce3_orbit_from_safe(safe_path, t0_sec, t1_sec)

    # Step 3: Build Doppler LUT2d for both bursts
    ref_doppler = build_doppler_lut(ref)
    sec_doppler = build_doppler_lut(sec)

    # Step 4: Open DEM and get ellipsoid
    dem_raster = isce3.io.Raster(str(dem_path))
    epsg = dem_raster.get_GEOProjectionEPSG()
    projection = isce3.core.make_projection(epsg)
    ellipsoid = projection.ellipsoid

    # Step 5: Configure Geo2Rdr
    geo2rdr = Geo2Rdr(
        radargrid=ref_radar_grid,
        orbit=ref_orbit,
        ellipsoid=ellipsoid,
        doppler=ref_doppler,
        threshold=1e-4,
        numiter=100,
        lines_per_block=500
    )

    # Step 6: Prepare output directory
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check for topo file from rdr2geo stage
    topo_path = work_dir / "rdr2geo" / "topo.vrt"
    if not topo_path.exists():
        topo_path = work_dir / "topo.vrt"

    if not topo_path.exists():
        raise FileNotFoundError(
            f"Topo file not found at {topo_path}. "
            "Ensure rdr2geo has been run first."
        )

    # Step 7: Run Geo2Rdr
    LOG.info("Running Geo2Rdr for burst pair, output to %s", work_dir)
    geo2rdr.geo2rdr(str(topo_path), str(work_dir))

    # Step 8: Read offset files and compute statistics
    range_off_path = work_dir / "range.off"
    azimuth_off_path = work_dir / "azimuth.off"

    if not range_off_path.exists():
        raise FileNotFoundError(f"Range offset file not found: {range_off_path}")
    if not azimuth_off_path.exists():
        raise FileNotFoundError(f"Azimuth offset file not found: {azimuth_off_path}")

    # Read offset files using GDAL
    try:
        from osgeo import gdal
    except ImportError:
        import osgeo.gdal as gdal  # type: ignore

    range_ds = gdal.Open(str(range_off_path), gdal.GA_ReadOnly)
    azimuth_ds = gdal.Open(str(azimuth_off_path), gdal.GA_ReadOnly)

    if range_ds is None or azimuth_ds is None:
        raise RuntimeError("Failed to open offset raster files")

    range_off_arr = range_ds.ReadAsArray()
    azimuth_off_arr = azimuth_ds.ReadAsArray()

    range_ds = None
    azimuth_ds = None

    if range_off_arr is None or azimuth_off_arr is None:
        raise RuntimeError("Failed to read offset data from rasters")

    # Convert to float and compute valid mask
    range_off_arr = range_off_arr.astype(float)
    azimuth_off_arr = azimuth_off_arr.astype(float)

    valid_mask = np.isfinite(range_off_arr) & np.isfinite(azimuth_off_arr)
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        LOG.warning("No valid offset samples found")
        median_range = 0.0
        median_az = 0.0
    else:
        median_range = float(np.nanmedian(range_off_arr[valid_mask]))
        median_az = float(np.nanmedian(azimuth_off_arr[valid_mask]))

    LOG.info(
        "Geo2Rdr complete: median_range=%.3f, median_az=%.3f, valid=%d/%d",
        median_range, median_az, valid_count, range_off_arr.size
    )

    return Geo2RdrOffsets(
        range_off_path=str(range_off_path),
        azimuth_off_path=str(azimuth_off_path),
        median_range_offset=median_range,
        median_azimuth_offset=median_az,
        valid_sample_count=valid_count,
    )
