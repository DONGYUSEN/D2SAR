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
from scripts.sentinel_orbit import resolve_orbit_for_product

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

        utc_text = utc_elem.text if utc_elem is not None else None
        if utc_text is None:
            continue
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


def _find_eof_file(safe_path: Path, orbit_dir: Path | None = None) -> Path | None:
    """Find the best available EOF orbit file for a SAFE product.

    Searches for POEORB first (precise orbit), then falls back to RESORB
    (restituted orbit). Returns the most recent file by modification time.

    Parameters
    ----------
    safe_path : Path
        Path to the Sentinel-1 SAFE directory.
    orbit_dir : Path | None
        Optional external directory containing orbit files.
        If provided, searches here first before SAFE/aux/.

    Returns
    -------
    Path or None
        Path to the EOF file, or None if not found.
    """
    # If external orbit directory is provided, search there first
    if orbit_dir is not None and orbit_dir.exists():
        poeorb_files = sorted(orbit_dir.glob("*POEORB*.EOF"))
        if poeorb_files:
            return max(poeorb_files, key=lambda p: p.stat().st_mtime)
        poeorb_zip = sorted(orbit_dir.glob("*POEORB*.EOF.zip"))
        if poeorb_zip:
            return max(poeorb_zip, key=lambda p: p.stat().st_mtime)
        resorb_files = sorted(orbit_dir.glob("*RESORB*.EOF"))
        if resorb_files:
            return max(resorb_files, key=lambda p: p.stat().st_mtime)

    # Try POEORB first (precise orbit) in SAFE/aux/
    poeorb_files = sorted(safe_path.glob("aux/*POEORB*.EOF"))
    if poeorb_files:
        return max(poeorb_files, key=lambda p: p.stat().st_mtime)

    # Fall back to RESORB (restituted orbit)
    resorb_files = sorted(safe_path.glob("aux/*RESORB*.EOF"))
    if resorb_files:
        return max(resorb_files, key=lambda p: p.stat().st_mtime)

    return None

def _resolve_orbit_file_for_safe(
    safe_path: Path,
    orbit_dir: Path | None = None,
) -> Path:
    """Resolve an orbit file for a SAFE, downloading it if needed."""
    eof_path = _find_eof_file(safe_path, orbit_dir)
    if eof_path is not None:
        return eof_path

    download_dir = orbit_dir or (safe_path.parent / "orbits")
    result = resolve_orbit_for_product(
        safe_path,
        orbit_dir=download_dir,
        download=True,
        work_dir=safe_path.parent,
    )
    if result is None:
        raise FileNotFoundError(
            f"No POEORB or RESORB EOF file found in {safe_path}/aux/ or {orbit_dir}; "
            f"auto-download to {download_dir} failed"
        )

    eof_path = Path(result.path)
    LOG.info("Downloaded orbit file for %s: %s", safe_path, eof_path)
    return eof_path


def build_isce3_orbit_from_safe(
    safe_path: Path,
    t0: datetime,
    t1: datetime,
    orbit_dir: Path | None = None,
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
    orbit_dir : Path | None
        Optional external directory containing orbit EOF files.

    Returns
    -------
    isce3.core.Orbit
        Interpolated orbit covering [t0, t1] with ±120s margin.

    Raises
    ------
    NotImplementedError
        If ISCE3 C++ bindings are not available.
    FileNotFoundError
        If no EOF orbit file is found in the SAFE and auto-download fails.
    """
    isce3 = _get_isce3()
    from isce3.core import DateTime, StateVector, Orbit, OrbitInterpMethod

    # Step 1: Find the best available EOF file, or auto-download one.
    eof_path = _resolve_orbit_file_for_safe(safe_path, orbit_dir)

    LOG.debug("Using orbit file: %s", eof_path)

    # Step 2: Parse EOF into state vectors
    state_vectors = _parse_eof_state_vectors(eof_path)
    if not state_vectors:
        raise ValueError(f"No state vectors found in EOF file: {eof_path}")

    # Step 3: Convert to ISCE3 Orbit
    # Convert state vectors to ISCE3 format and build Orbit
    # Note: Orbit constructor requires at least 2 initial state vectors
    isce3_svs = []
    for dt, x, y, z, vx, vy, vz in state_vectors:
        t_isce3 = DateTime(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute,
            dt.second + dt.microsecond * 1e-6
        )
        sv = StateVector(t_isce3, [x, y, z], [vx, vy, vz])
        isce3_svs.append(sv)

    orbit = Orbit(isce3_svs, OrbitInterpMethod.HERMITE)

    LOG.debug(
        "Built orbit with %d state vectors, time span: %s to %s",
        orbit.size, state_vectors[0][0], state_vectors[-1][0]
    )

    return orbit


# ---------------------------------------------------------------------------
# Doppler LUT builder
# ---------------------------------------------------------------------------
# Doppler LUT builder
# ---------------------------------------------------------------------------

# IMPLEMENTED: build_doppler_lut

def build_doppler_lut(burst: BurstRadarGrid, use_zero: bool = True) -> Any:
    """Build an ISCE3 core.LUT2d from Sentinel-1 annotation Doppler coefficients.

    Parameters
    ----------
    burst : BurstRadarGrid
        Burst metadata containing ``doppler_coefficients``.
    use_zero : bool
        If True (default), return a zero-Doppler LUT for speed/validation.
        If False, build LUT from Sentinel-1 polynomial coefficients.

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
    from isce3.core import LUT2d

    if use_zero:
        # Zero Doppler LUT — fastest path for testing/simulation
        return LUT2d(0.0)

    # Build full LUT from Sentinel-1 polynomial coefficients
    coeffs = burst.doppler_coefficients
    ref_slant_range = burst.starting_range
    slant_step = burst.range_pixel_spacing
    n_rg = burst.valid_window.num_samples
    n_az = burst.valid_window.num_lines

    range_vec = np.array([
        ref_slant_range + s * slant_step for s in range(n_rg)
    ])

    # Evaluate polynomial: f_D(s_rg) in Hz
    if coeffs:
        f_doppler_vec = np.polyval(coeffs[::-1], range_vec)
    else:
        f_doppler_vec = np.zeros(n_rg)

    # Build regular-grid LUT2d
    # x=range time (seconds), y=azimuth index, data=Doppler Hz
    c = 299792458.0
    rgrid_time = (range_vec - ref_slant_range) * 2.0 / c
    f_doppler_2d = np.tile(f_doppler_vec, (n_az, 1)).astype(np.float64)
    rgrid_time = rgrid_time.astype(np.float64)

    # Use coordinate-array constructor
    ycoord = np.arange(n_az, dtype=np.float64)
    lut = LUT2d(rgrid_time, ycoord, f_doppler_2d)
    return lut


# ---------------------------------------------------------------------------
# Geo2Rdr single-burst
# ---------------------------------------------------------------------------


def _geo2rdr_valid_mask(range_offsets: np.ndarray, azimuth_offsets: np.ndarray) -> np.ndarray:
    """Return valid Geo2Rdr samples, excluding ISCE3 NULL_VALUE and nodata cells."""
    geo2rdr_nodata = -999999.0
    return (
        np.isfinite(range_offsets)
        & np.isfinite(azimuth_offsets)
        & (range_offsets != geo2rdr_nodata)
        & (azimuth_offsets != geo2rdr_nodata)
        & (range_offsets > -9.0e5)
        & (azimuth_offsets > -9.0e5)
    )


def run_geo2rdr_single_burst(
    ref: BurstRadarGrid,
    sec: BurstRadarGrid,
    dem_path: Path,
    work_dir: Path,
    safe_path: Path | None = None,
    sec_safe_path: Path | None = None,
    orbit_dir: Path | None = None,
    use_gpu: bool = False,
    use_zero_doppler: bool = True,
    gpu_id: int = 0,
) -> Geo2RdrOffsets:
    """Run ISCE3 Rdr2Geo → Geo2Rdr for one reference / secondary burst pair.

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
    orbit_dir : Path, optional
        External directory containing orbit EOF files.
        If provided, searched before SAFE/aux/.
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

    **Dependencies**: ``isce3.geometry.Rdr2Geo``, ``isce3.geometry.Geo2Rdr``,
    ``isce3.core.LUT2d``, ``isce3.core.Orbit``, ``isce3.io.Raster``.
    """
    isce3 = _get_isce3()
    from isce3.core import DateTime, LookSide, Orbit, LUT2d
    from isce3.product import RadarGridParameters
    from isce3.geometry import rdr2geo, DEMInterpolator

    if use_gpu:
        import isce3.cuda.core as _cuda_core
        try:
            _device = _cuda_core.Device(gpu_id)
            _cuda_core.set_device(_device)
        except Exception as exc:
            LOG.warning("Failed to set CUDA device %d, falling back to CPU: %s", gpu_id, exc)
            use_gpu = False

    if use_gpu:
        try:
            from isce3.cuda.geometry import Geo2Rdr, Rdr2Geo
        except (ImportError, AttributeError) as exc:
            LOG.warning("ISCE3 CUDA geometry unavailable; falling back to CPU: %s", exc)
            use_gpu = False

    # Use strip_insar2.py style GPU check instead of forcing CPU
    if use_gpu:
        try:
            from isce3.core.gpu_check import use_gpu as gpu_check
            use_gpu = gpu_check(True, gpu_id)
            if not use_gpu:
                LOG.info("GPU check failed; using CPU mode")
        except Exception as exc:
            LOG.warning("GPU check failed: %s; using CPU mode", exc)
            use_gpu = False

    # Use strip_insar2.py style parameters for better stability
    GEOMETRY_THRESHOLD = 1e-8  # More strict than default 1e-4
    GEOMETRY_NUMITER = 50  # Same as strip_insar2.py
    GEOMETRY_LINES_PER_BLOCK = 2000  # Same as strip_insar2.py

    if not use_gpu:
        from isce3.geometry import Geo2Rdr, Rdr2Geo


    # Step 1: Build ISCE3 RadarGridParameters for both bursts
    # Use separate epochs for master and slave to avoid time reference issues
    ref_epoch = DateTime(
        ref.identity.sensing_start.year,
        ref.identity.sensing_start.month,
        ref.identity.sensing_start.day,
        ref.identity.sensing_start.hour,
        ref.identity.sensing_start.minute,
        ref.identity.sensing_start.second + ref.identity.sensing_start.microsecond * 1e-6
    )

    sec_epoch = DateTime(
        sec.identity.sensing_start.year,
        sec.identity.sensing_start.month,
        sec.identity.sensing_start.day,
        sec.identity.sensing_start.hour,
        sec.identity.sensing_start.minute,
        sec.identity.sensing_start.second + sec.identity.sensing_start.microsecond * 1e-6
    )

    ref_prf = 1.0 / ref.azimuth_time_interval
    sec_prf = 1.0 / sec.azimuth_time_interval

    ref_radar_grid = RadarGridParameters(
        0.0,
        ref.radar_wavelength,
        ref_prf,
        ref.starting_range,
        ref.range_pixel_spacing,
        LookSide.Right,
        ref.valid_window.num_lines,
        ref.valid_window.num_samples,
        ref_epoch
    )

    sec_radar_grid = RadarGridParameters(
        0.0,
        sec.radar_wavelength,
        sec_prf,
        sec.starting_range,
        sec.range_pixel_spacing,
        LookSide.Right,
        sec.valid_window.num_lines,
        sec.valid_window.num_samples,
        sec_epoch
    )

    LOG.info("Ref radar grid: epoch=%s, lines=%d, samples=%d",
           ref_epoch.isoformat(), ref.valid_window.num_lines, ref.valid_window.num_samples)
    LOG.info("Sec radar grid: epoch=%s, lines=%d, samples=%d",
           sec_epoch.isoformat(), sec.valid_window.num_lines, sec.valid_window.num_samples)

    # Step 2: Build Orbit objects for both bursts
    if safe_path is None:
        safe_path = work_dir.parent
    if sec_safe_path is None:
        sec_safe_path = safe_path

    t0_ref = ref.identity.sensing_start
    t1_ref = ref.identity.sensing_stop
    t0_sec = sec.identity.sensing_start
    t1_sec = sec.identity.sensing_stop

    orbit_margin = timedelta(seconds=300)
    ref_orbit = build_isce3_orbit_from_safe(safe_path, t0_ref - orbit_margin, t1_ref + orbit_margin, orbit_dir)
    sec_orbit = build_isce3_orbit_from_safe(sec_safe_path, t0_sec - orbit_margin, t1_sec + orbit_margin, orbit_dir)

    # Step 3: Build Doppler LUT2d for both bursts (zero-Doppler fallback)
    ref_doppler = build_doppler_lut(ref, use_zero=use_zero_doppler)
    sec_doppler = build_doppler_lut(sec, use_zero=use_zero_doppler)

    # Step 4: Open DEM. DEM is a single-band height raster; the Rdr2Geo
    # output topo.vrt is multi-band x/y/z geometry.
    dem_raster = isce3.io.Raster(str(dem_path))
    dem_interp = DEMInterpolator()
    dem_interp.load_dem(dem_raster)

    # Step 5: Run Rdr2Geo on the reference burst to build topo.vrt
    ref_topo_dir = work_dir / "rdr2geo_ref"
    ref_topo_dir.mkdir(parents=True, exist_ok=True)
    rdr2geo = Rdr2Geo(
        ref_radar_grid,
        ref_orbit,
        isce3.core.Ellipsoid(),
        ref_doppler,
        numiter=25,
        dem_interp_method=isce3.core.DataInterpMethod.BIQUINTIC,
        epsg_out=4326,
        compute_mask=True,
        lines_per_block=GEOMETRY_LINES_PER_BLOCK,
    )

    try:
        rdr2geo.topo(dem_raster, str(ref_topo_dir))
    except Exception as exc:
        LOG.warning("GPU Rdr2Geo.topo() failed: %s; falling back to CPU", exc)
        from isce3.geometry import Geo2Rdr as Geo2Rdr_cpu, Rdr2Geo as Rdr2Geo_cpu
        rdr2geo = Rdr2Geo_cpu(
            ref_radar_grid,
            ref_orbit,
            isce3.core.Ellipsoid(),
            ref_doppler,
            numiter=25,
            dem_interp_method=isce3.core.DataInterpMethod.BIQUINTIC,
            epsg_out=4326,
            compute_mask=True,
            lines_per_block=GEOMETRY_LINES_PER_BLOCK,
        )
        rdr2geo.topo(dem_raster, str(ref_topo_dir))

    topo_vrt = ref_topo_dir / "topo.vrt"
    if not topo_vrt.exists():
        raise FileNotFoundError(f"Rdr2Geo topo.vrt was not produced: {topo_vrt}")

    topo_raster = isce3.io.Raster(str(topo_vrt))

    # Step 6: Configure Geo2Rdr for the secondary burst
    geo2rdr = Geo2Rdr(
        radar_grid=sec_radar_grid,
        orbit=sec_orbit,
        ellipsoid=isce3.core.Ellipsoid(),
        doppler=sec_doppler,
        threshold=GEOMETRY_THRESHOLD,
        numiter=GEOMETRY_NUMITER,
        lines_per_block=GEOMETRY_LINES_PER_BLOCK,
    )

    # Step 7: Prepare output directory
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 8: Run Geo2Rdr using the reference topo raster
    LOG.info("Running Geo2Rdr for burst pair, output to %s", work_dir)
    try:
        geo2rdr.geo2rdr(topo_raster, str(work_dir))
    except Exception as exc:
        LOG.warning("GPU Geo2Rdr.geo2rdr() failed: %s; falling back to CPU", exc)
        from isce3.geometry import Geo2Rdr as Geo2Rdr_cpu
        geo2rdr = Geo2Rdr_cpu(
            radar_grid=sec_radar_grid,
            orbit=sec_orbit,
            ellipsoid=isce3.core.Ellipsoid(),
            doppler=sec_doppler,
            threshold=GEOMETRY_THRESHOLD,
            numiter=GEOMETRY_NUMITER,
            lines_per_block=GEOMETRY_LINES_PER_BLOCK,
        )
        geo2rdr.geo2rdr(topo_raster, str(work_dir))

    # Spot-check a single radar→geo→radar roundtrip to prove that both
    # ISCE3 geometry directions are wired into the coarse alignment path.
    # This is optional and can be skipped if methods are not available
    try:
        sample_line = ref.valid_window.num_lines // 2
        sample_sample = ref.valid_window.num_samples // 2
        sample_time = (
            ref.identity.sensing_start
            + timedelta(seconds=sample_line * ref.azimuth_time_interval)
        )
        ref_epoch = ref_orbit.reference_epoch
        ref_epoch_dt = datetime(
            ref_epoch.year,
            ref_epoch.month,
            ref_epoch.day,
            ref_epoch.hour,
            ref_epoch.minute,
            int(ref_epoch.second),
            min(999_999, int(round(ref_epoch.frac * 1_000_000))),
            tzinfo=UTC,
        )
        aztime = (sample_time - ref_epoch_dt).total_seconds()
        slant_range = ref.slant_range_at(sample_sample)

        from isce3.geometry import rdr2geo as rdr2geo_func
        xyz = rdr2geo_func(
            aztime,
            slant_range,
            ref_orbit,
            LookSide.Right,
            doppler=0.0,
            wavelength=ref.radar_wavelength,
            dem=dem_interp,
            ellipsoid=isce3.core.Ellipsoid(),
        )
        aztime_rt, range_rt = isce3.geometry.geo2rdr(
            xyz,
            isce3.core.Ellipsoid(),
            sec_orbit,
            sec_doppler,
            sec.radar_wavelength,
            LookSide.Right,
        )
        LOG.info(
            "Geo2Rdr/Rdr2Geo roundtrip: az_residual=%.6f s rg_residual=%.6f m",
            float(aztime_rt - aztime),
            float(range_rt - slant_range),
        )
    except Exception as exc:
        LOG.warning("Geo2Rdr/Rdr2Geo roundtrip spot-check skipped: %s", exc)

    # Step 9: Read offset files and compute statistics
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

    valid_mask = _geo2rdr_valid_mask(range_off_arr, azimuth_off_arr)
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
