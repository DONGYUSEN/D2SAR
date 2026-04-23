# strip_insar.py — Unified InSAR Processor Design

## Date: 2026-04-18

## Overview

Unified InSAR processor for Tianyi/Lutan strip-mode SAR satellites. Mirrors `strip_rtc.py` architecture: two manifests (master + slave) → 6-stage interferometry pipeline → hybrid GPU/CPU with honest fallback → HDF5 + GeoTIFF + PNG output.

## CLI

```bash
python3 scripts/strip_insar.py <master_manifest> <slave_manifest> <output_dir> \
  [--dem PATH] \
  [--dem-cache-dir PATH] \
  [--dem-margin-deg FLOAT] \
  [--unwrap-method icu|snaphu] \
  [--resolution METERS] \
  [--block-rows INT] \
  [--gpu-mode auto|cpu|gpu] \
  [--gpu-id INT]
```

### Arguments
| Argument | Default | Description |
|---|---|---|
| `master_manifest` | required | Master SLC manifest.json path |
| `slave_manifest` | required | Slave SLC manifest.json path |
| `output_dir` | required | Output directory |
| `--dem` | None | Override DEM path |
| `--dem-cache-dir` | None | DEM cache directory |
| `--dem-margin-deg` | 0.05 | DEM margin in degrees |
| `--unwrap-method` | `icu` | Phase unwrapping method (icu/snaphu) |
| `--resolution` | None | Output ground resolution meters (default: 2x max range/azimuth) |
| `--block-rows` | 256 | CPU block rows for HDF5 I/O |
| `--gpu-mode` | `auto` | GPU preference (auto/cpu/gpu) |
| `--gpu-id` | 0 | CUDA device ID |

## Processing Chain (P0-P5)

| Phase | Stage | GPU Module | CPU Fallback |
|---|---|---|---|
| P0 | Geo2Rdr coarse coregistration | `isce3.cuda.geometry.Geo2Rdr` | `isce3.geometry.Geo2Rdr` |
| P1 | PyCuAmpcor dense matching | `isce3.cuda.matchtemplate.PyCuAmpcor` | **None** (skip if GPU unavailable) |
| P2 | Crossmul interferogram + coherence | `isce3.cuda.signal.Crossmul` | `isce3.signal.CrossMultiply` |
| P3 | Phase unwrapping | `PhaseUnwrapper` interface (ICU/SNAPHU) | always CPU |
| P4 | Geocode products | `isce3.cuda.geocode.Geocode` | `isce3.geocode.Geocode` |
| P5 | Multi-look output + GeoTIFF + PNG | CPU post-processing | — |

## GPU Strategy

**Option A + Honest Fallback**

- Geo2Rdr → GPU with CPU fallback
- Crossmul → GPU with CPU fallback
- PyCuAmpcor → GPU-only, no fallback; skip dense matching if GPU unavailable
- Geocode → GPU with CPU fallback
- ICU unwrapping → always CPU (no GPU option in ISCE3)

On OOM in GPU stages: retry with halved block_rows (min 64), then fall back to CPU.

## HDF5 Output

**File:** `interferogram_fullres.h5`

| Dataset | Type | Description |
|---|---|---|
| `avg_amplitude` | float32 | `(master_amplitude + slave_amplitude) / 2` |
| `interferogram` | complex64 | Complex interferogram `master × conj(slave)` |
| `coherence` | float32 | Interferometric coherence [0, 1] |
| `unwrapped_phase` | float32 | Unwrapped phase in radians |
| `los_displacement` | float32 | LOS displacement in meters: `unwrapped_phase × wavelength / (4π)` |
| `longitude` | float32 | Longitude (WGS84) |
| `latitude` | float32 | Latitude (WGS84) |
| `height` | float32 | Height above ellipsoid (m) |
| `utm_x` | float32 | UTM X coordinate |
| `utm_y` | float32 | UTM Y coordinate |

**Attributes:**
- `product_type`: `insar_interferogram`
- `coordinate_system`: `EPSG:4326` (lon/lat/height), `EPSG:XXXX` (UTM)
- `utm_epsg`: integer
- `wavelength`: meters
- `unwrap_method`: `icu` or `snaphu`
- `master_manifest`: master manifest path
- `slave_manifest`: slave manifest path

## GeoTIFF / PNG Outputs

| File | Description |
|---|---|
| `interferogram_utm_geocoded.tif` | Geocoded interferogram (phase, float32) |
| `coherence_utm_geocoded.tif` | Geocoded coherence (float32) |
| `unwrapped_phase_utm_geocoded.tif` | Geocoded unwrapped phase (float32, radians) |
| `los_displacement_utm_geocoded.tif` | Geocoded LOS displacement (float32, meters) |
| `interferogram_utm_geocoded.png` | Wrapped phase preview PNG |

## PhaseUnwrapper Interface

```python
class PhaseUnwrapper(ABC):
    @abstractmethod
    def unwrap(self, interferogram: np.ndarray,
               coherence: np.ndarray,
               radar_grid, orbit, dem_path: str) -> np.ndarray:
        """Input: complex64 interferogram (rows, cols), coherence (rows, cols)
           Output: unwrapped phase in radians (rows, cols)"""
        pass

class ICUUnwrapper(PhaseUnwrapper):
    """ISCE3 ICU - default CPU implementation"""
    pass

class SNAPHUUnwrapper(PhaseUnwrapper):
    """SNAPHU wrapper - optional external dependency"""
    pass
```

Selection via `--unwrap-method` flag. Default: `icu`.

## LOS Displacement Calculation

```
los_displacement (m) = unwrapped_phase (rad) × wavelength (m) / (4 × π)
```

## JSON Result

```json
{
  "sensor": "tianyi",
  "master_manifest": "/path/to/master/manifest.json",
  "slave_manifest": "/path/to/slave/manifest.json",
  "backend_requested": "gpu",
  "backend_used": "gpu",
  "pipeline_mode": "hybrid",
  "stage_backends": {
    "geo2rdr": "gpu",
    "pycuampcor": "gpu",
    "crossmul": "gpu",
    "unwrap": "cpu",
    "geocode": "gpu",
    "output": "cpu"
  },
  "unwrap_method": "icu",
  "dem": "/path/to/dem.tif",
  "orbit_interp": "Hermite",
  "resolution_meters": 5.0,
  "topo_block_rows": 1024,
  "interferogram_h5": "/path/to/interferogram_fullres.h5",
  "interferogram_tif": "/path/to/interferogram_utm_geocoded.tif",
  "coherence_tif": "/path/to/coherence_utm_geocoded.tif",
  "unwrapped_phase_tif": "/path/to/unwrapped_phase_utm_geocoded.tif",
  "los_displacement_tif": "/path/to/los_displacement_utm_geocoded.tif",
  "interferogram_png": "/path/to/interferogram_utm_geocoded.png"
}
```

## Architecture (mirrors strip_rtc.py)

```
scripts/
  strip_insar.py      — CLI entrypoint + main dispatcher
  common_processing.py — shared utilities (reused from strip_rtc.py)

tests/
  test_strip_insar.py — unit tests for strip_insar.py
```

**Key functions in strip_insar.py:**
- `load_manifest()`, `detect_sensor_from_manifest()` — same as strip_rtc.py
- `build_output_paths()` — InSAR output paths
- `select_processing_backend()` — GPU/CPU selection
- `query_gpu_memory_info()`, `choose_gpu_topo_block_rows()`, `_is_gpu_memory_error()`, `_halve_topo_block_rows()` — GPU memory management (reused from strip_rtc.py)
- `PhaseUnwrapper`, `ICUUnwrapper`, `SNAPHUUnwrapper` — unwrapper interface
- `process_strip_insar()` — main dispatcher
- `_process_insar_gpu()`, `_process_insar_cpu()` — stage execution
- `compute_los_displacement()` — phase to LOS conversion

## Reused from strip_rtc.py / common_processing.py

- GPU memory management functions (unchanged)
- `construct_orbit()`, `construct_doppler_lut2d()`, `construct_radar_grid()`
- `resolve_manifest_data_path()`, `resolve_manifest_metadata_path()`
- `append_topo_coordinates_hdf()`, `append_utm_coordinates_hdf()`
- `compute_utm_output_shape()`, `accumulate_utm_grid()`
- `write_geocoded_geotiff()`, `write_geocoded_png()`
- `load_scene_corners_with_fallback()`, `resolve_dem_for_scene()`

## Notes

- Master SLC is the geometric reference (DEM coregistration uses master radar grid)
- PyCuAmpcor is GPU-only in ISCE3; if GPU unavailable, dense matching is skipped and coarse coregistration is used
- LOS displacement sign convention: positive = motion toward satellite
- Antimeridian handling: reuse `dem_manager.py` bbox logic
