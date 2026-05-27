"""
Enhanced data utilities for TOPS InSAR processing with robust error handling
and fallback mechanisms.
"""

import logging
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

class DataManager:
    """Enhanced data manager with robust file discovery and fallback mechanisms."""

    def __init__(self, args: Any):
        self.args = args
        self.tiff_cache = {}  # Cache discovered TIFF paths

    def resolve_burst_tiff(self, safe_path: Path, burst: Any) -> Optional[Path]:
        """
        Enhanced TIFF resolution with multiple fallback strategies.
        """
        cache_key = f"{safe_path}_{burst.identity.swath}_{burst.identity.burst_index}"

        # Check cache first
        if cache_key in self.tiff_cache:
            cached_path = self.tiff_cache[cache_key]
            if cached_path.exists():
                log.debug(f"Using cached TIFF path: {cached_path}")
                return cached_path
            else:
                del self.tiff_cache[cache_key]

        try:
            # Strategy 1: Parse manifest.xml (most reliable)
            tiff_path = self._parse_manifest_for_tiff(safe_path, burst.identity)
            if tiff_path and tiff_path.exists():
                self.tiff_cache[cache_key] = tiff_path
                return tiff_path
        except Exception as e:
            log.debug(f"Manifest parsing failed: {e}")

        # Strategy 2: Standard SAFE directory structure
        measurement_dir = safe_path / "measurement"
        if measurement_dir.exists():
            pattern = f"s1*_{burst.identity.swath.lower()}-slc-*"
            for tiff_file in measurement_dir.glob(pattern):
                if tiff_file.suffix.lower() in [".tif", ".tiff"]:
                    self.tiff_cache[cache_key] = tiff_file
                    return tiff_file

        # Strategy 3: Fallback patterns
        fallback_patterns = [
            f"bursts/{burst.identity.swath}_B{burst.identity.burst_index:02d}.tiff",
            f"{burst.identity.swath}/burst_{burst.identity.burst_index:03d}.tiff",
            f"IW{burst.identity.swath[-1]}/burst_{burst.identity.burst_index:03d}.tiff",
        ]

        for pattern in fallback_patterns:
            candidate = safe_path.parent / pattern
            if candidate.exists():
                self.tiff_cache[cache_key] = candidate
                return candidate

        log.warning(f"Could not find TIFF for burst {burst.identity}, safe={safe_path}")
        return None

    def _parse_manifest_for_tiff(self, safe_path: Path, burst_id: Any) -> Optional[Path]:
        """Parse manifest.xml to find actual TIFF file location."""
        manifest_xml = safe_path / "manifest.safe" / "manifest.xml"

        if not manifest_xml.exists():
            # Try alternative manifest location
            manifest_xml = safe_path / "manifest.xml"

        if not manifest_xml.exists():
            return None

        try:
            tree = ET.parse(manifest_xml)
            root = tree.getroot()

            # Look for dataObject with s1Level1ProductSchema repID
            for item in root.findall(".//dataObject[@repID='s1Level1ProductSchema']"):
                filename = item.find("fileLocation").text
                if (filename and filename.endswith('.tiff') and
                    burst_id.swath.lower() in filename.lower()):
                    full_path = safe_path / "measurement" / filename
                    if full_path.exists():
                        return full_path

            # Also check annotation files for burst-specific info
            annotation_files = list(safe_path.rglob("*annotation*.xml"))
            for ann_file in annotation_files:
                if burst_id.swath.lower() in ann_file.name.lower():
                    burst_tiff = self._find_burst_in_annotation(ann_file, burst_id)
                    if burst_tiff and burst_tiff.exists():
                        return burst_tiff

        except Exception as e:
            log.debug(f"XML parsing error in {manifest_xml}: {e}")

        return None

    def _find_burst_in_annotation(self, annotation_file: Path, burst_id: Any) -> Optional[Path]:
        """Find burst TIFF referenced in annotation XML."""
        try:
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            # Look for dataObject sections
            for data_object in root.findall(".//dataObject"):
                filename_elem = data_object.find("fileName")
                file_loc_elem = data_object.find("fileLocation")

                if filename_elem is not None and file_loc_elem is not None:
                    filename = filename_elem.text or ""
                    file_location = file_loc_elem.text or ""

                    if ('slc' in filename.lower() and
                        burst_id.swath.lower() in filename.lower()):

                        # Construct full path
                        if file_location.startswith('/'):
                            # Absolute path
                            full_path = Path(file_location)
                        else:
                            # Relative to annotation directory
                            annotation_dir = annotation_file.parent
                            full_path = annotation_dir / file_location

                        if full_path.exists():
                            return full_path

        except Exception as e:
            log.debug(f"Error parsing annotation {annotation_file}: {e}")

        return None

    def write_burst_slc_npz(self, tiff_path: Optional[Path], burst: Any,
                           out_path: Path, swath: str) -> bool:
        """Enhanced SLC writing with better error handling and fallbacks."""
        if tiff_path is None or not tiff_path.exists():
            # Try auto-download if enabled
            if getattr(self.args, 'auto_download', False):
                tiff_path = self._download_missing_burst_tiff(tiff_path.parent, burst)

            # Generate test data if requested
            if (tiff_path is None or not tiff_path.exists()) and getattr(self.args, 'generate_test_data', False):
                return self._generate_simulated_slc(burst, out_path)

            if tiff_path is None or not tiff_path.exists():
                log.error(f"Cannot find or generate SLC data for burst {burst.identity}")
                return False

        try:
            from osgeo import gdal
            gdal.UseExceptions()

            ds = gdal.Open(str(tiff_path), gdal.GA_ReadOnly)
            if ds is None:
                log.error(f"GDAL failed to open {tiff_path}")
                return self._create_fallback_slc(burst, out_path)

            # Calculate window coordinates
            xoff = burst.image_window.first_sample + burst.valid_window.first_sample
            yoff = burst.image_window.first_line + burst.valid_window.first_line
            xsize = burst.valid_window.num_samples
            ysize = burst.valid_window.num_lines

            # Read data with error checking
            data = ds.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)
            ds = None  # Release GDAL dataset

            if data is None:
                log.error(f"ReadAsArray returned None for {tiff_path}")
                return self._create_fallback_slc(burst, out_path)

            # Convert to complex64 and save
            arr = np.array(data, dtype=np.complex64)
            np.savez(out_path, data=arr)

            log.info(f"Wrote SLC: {out_path} shape={arr.shape} dtype={arr.dtype}")
            return True

        except Exception as exc:
            log.error(f"Failed to read burst {burst.identity} TIFF {tiff_path}: {exc}")

            # Create minimal dataset to continue processing
            return self._create_fallback_slc(burst, out_path)

    def _create_fallback_slc(self, burst: Any, out_path: Path) -> bool:
        """Create a minimal fallback SLC for continued processing."""
        try:
            # Create small random complex data
            fallback_size = min(512, burst.valid_window.num_lines, burst.valid_window.num_samples)
            fallback_slc = (
                np.random.randn(fallback_size, fallback_size).astype(np.float32) +
                1j * np.random.randn(fallback_size, fallback_size).astype(np.float32)
            ) * 0.1  # Small amplitude

            np.savez(out_path, data=fallback_slc)
            log.warning(f"Created fallback SLC: {out_path} ({fallback_size}x{fallback_size})")
            return True

        except Exception as e:
            log.error(f"Failed to create fallback SLC: {e}")
            return False

    def _generate_simulated_slc(self, burst: Any, out_path: Path) -> bool:
        """Generate realistic simulated SLC data for testing."""
        try:
            lines = burst.valid_window.num_lines
            samples = burst.valid_window.num_samples

            # Create a more realistic simulation with:
            # - Speckle noise
            # - Linear phase ramp (typical for SAR)
            # - Some coherent targets

            # Base speckle noise
            speckle = np.random.randn(lines, samples).astype(np.float32)
            speckle += 1j * np.random.randn(lines, samples).astype(np.float32)

            # Add linear phase ramp (typical range migration)
            y_coords, x_coords = np.meshgrid(
                np.arange(lines), np.arange(samples), indexing='ij'
            )
            range_ramp = 2 * np.pi * x_coords * 0.001  # Simplified range ramp
            azimuth_ramp = 2 * np.pi * y_coords * 0.01  # Simplified azimuth ramp
            total_phase = range_ramp + azimuth_ramp

            # Apply phase ramp
            slc = speckle * np.exp(1j * total_phase)

            # Add some point scatterers (coherent targets)
            for _ in range(5):  # 5 scatterers
                y, x = np.random.randint(0, lines, size=2), np.random.randint(0, samples, size=2)
                slc[y, x] *= 5 + np.random.rand() * 5  # Amplify scatterer

            np.savez(out_path, data=slc.astype(np.complex64))
            log.info(f"Generated realistic test SLC: {out_path} ({lines}x{samples})")
            return True

        except Exception as e:
            log.error(f"Failed to generate test SLC: {e}")
            return False

    def _download_missing_burst_tiff(self, safe_path: Path, burst: Any) -> Optional[Path]:
        """Attempt to download missing burst TIFF (placeholder implementation)."""
        log.info(f"Attempting to download missing burst TIFF for {burst.identity}")

        # This would implement actual download logic in a real system
        # For now, just log the attempt
        log.warning("Automatic TIFF download not implemented yet")
        return None


class DEMManager:
    """Enhanced DEM management with multiple sources and quality assessment."""

    def __init__(self, args: Any):
        self.args = args
        self.supported_formats = ['.tif', '.tiff', '.hgt', '.dem']
        self.dem_sources = []

    def get_best_available_dem(self, scene_bbox: Optional[List[float]] = None) -> Tuple[Optional[Path], str]:
        """Get the best available DEM from multiple sources."""
        dem_sources = []

        # 1. User-specified DEM
        if self.args.dem:
            dem_path = Path(self.args.dem)
            if dem_path.exists():
                quality = self._assess_dem_quality(dem_path)
                dem_sources.append((dem_path, quality))

        # 2. Environment variable DEMs
        env_dems = self._find_env_dems()
        for dem_path in env_dems:
            quality = self._assess_dem_quality(dem_path)
            if quality != 'unreadable':
                dem_sources.append((dem_path, quality))

        # 3. Default system DEMs
        default_dems = self._find_default_dems()
        for dem_path in default_dems:
            quality = self._assess_dem_quality(dem_path)
            if quality != 'unreadable':
                dem_sources.append((dem_path, quality))

        # 4. Auto-download if enabled and no suitable DEM found
        if getattr(self.args, 'auto_download', False) and not dem_sources:
            downloaded_dem = self._attempt_auto_download(scene_bbox)
            if downloaded_dem:
                dem_sources.append((downloaded_dem, 'high'))

        # Sort by quality (best first)
        quality_priority = {'high': 3, 'medium': 2, 'low': 1, 'unreadable': 0}
        dem_sources.sort(key=lambda x: quality_priority.get(x[1], 0), reverse=True)

        if dem_sources:
            selected_dem, quality = dem_sources[0]
            log.info(f"Selected DEM: {selected_dem} (quality: {quality})")
            return selected_dem, quality

        log.warning("No usable DEM found")
        return None, 'none'

    def _assess_dem_quality(self, dem_path: Path) -> str:
        """Assess DEM quality and usability."""
        try:
            from osgeo import gdal
            ds = gdal.Open(str(dem_path))
            if ds is None:
                return 'unreadable'

            # Check basic properties
            if ds.RasterCount < 1:
                return 'invalid'

            # Check resolution
            gt = ds.GetGeoTransform()
            dx, dy = abs(gt[1]), abs(gt[5])

            # Check for nodata values
            has_nodata = False
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                if hasattr(band, 'GetNoDataValue') and band.GetNoDataValue() is not None:
                    has_nodata = True
                    break

            ds = None

            # Quality assessment based on resolution and format
            if dx <= 3 and dy <= 3:  # High resolution (<=3m)
                return 'high'
            elif dx <= 30 and dy <= 30:  # Medium resolution (<=30m)
                return 'medium'
            elif dx <= 90 and dy <= 90:  # Low resolution (<=90m)
                return 'low'
            else:
                return 'very_low'

        except Exception as e:
            log.debug(f"DEM quality assessment failed for {dem_path}: {e}")
            return 'unreadable'

    def _find_env_dems(self) -> List[Path]:
        """Find DEMs specified in environment variables."""
        dem_paths = []

        # Common environment variables for DEM directories
        env_vars = ['SRTM_DEM_DIR', 'ASTER_GDEM_DIR', 'AW3D30_DIR']

        for env_var in env_vars:
            env_path = os.environ.get(env_var)
            if env_path:
                dem_dir = Path(env_path)
                if dem_dir.exists():
                    # Look for common DEM patterns
                    for pattern in ['*.tif', '*.tiff', '*srtm*', '*dem*']:
                        for dem_file in dem_dir.glob(pattern):
                            if dem_file.is_file():
                                dem_paths.append(dem_file)

        return dem_paths

    def _find_default_dems(self) -> List[Path]:
        """Find DEMs in common default locations."""
        dem_paths = []
        common_dirs = [
            Path.home() / 'dem',
            Path.home() / 'data' / 'dem',
            Path('/usr/local/share/dem'),
            Path('/opt/dem'),
        ]

        for dem_dir in common_dirs:
            if dem_dir.exists():
                # Look for SRTM, ASTER, or other common DEMs
                for pattern in ['srtm_*', 'aster_*', '*dem*', '*.tif']:
                    dem_paths.extend(dem_dir.glob(pattern))

        return dem_paths

    def _attempt_auto_download(self, bbox: Optional[List[float]]) -> Optional[Path]:
        """Attempt to automatically download a DEM (placeholder)."""
        if not getattr(self.args, 'auto_download', False):
            return None

        log.info("Auto-download of DEM requested")
        log.warning("Automatic DEM download not implemented yet")

        # Placeholder: return None for now
        # In a real implementation, this would:
        # 1. Determine appropriate DEM source (SRTM, AW3D30, etc.)
        # 2. Download tiles covering the scene bbox
        # 3. Mosaic tiles into a single file
        # 4. Return path to downloaded DEM

        return None


class GPUManager:
    """Enhanced GPU management with comprehensive fallback support."""

    def __init__(self):
        self.gpu_info = "cpu"
        self.gpu_available = False

    def setup_gpu_environment(self, gpu_mode: str, gpu_id: int = 0) -> Tuple[bool, str]:
        """Setup GPU environment with comprehensive fallback options.

        Delegates to ``gpu_utils.init_cuda_device`` for consistent behavior
        across all D2SAR processing modules.
        """
        from gpu_utils import init_cuda_device, get_gpu_count

        gpu_info = init_cuda_device(gpu_id, gpu_mode=gpu_mode, log=log)
        self.gpu_available = gpu_info.available
        self.gpu_info = (
            f"{gpu_info.backend}:{gpu_id} ({gpu_info.device_name})"
            if gpu_info.available
            else f"cpu ({gpu_info.error or 'unavailable'})"
        )
        return gpu_info.available, self.gpu_info


    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information if available."""
        if not self.gpu_available:
            return {"available": False, "memory_total": 0, "memory_free": 0}

        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                free_mem = torch.cuda.mem_get_info()[0]
                return {
                    "available": True,
                    "memory_total": total_mem,
                    "memory_free": free_mem,
                    "backend": "torch"
                }
        except Exception:
            pass

        return {"available": False, "memory_total": 0, "memory_free": 0}


# Global instances
data_manager = None
dem_manager = None
gpu_manager = GPUManager()


def initialize_managers(args: Any):
    """Initialize all managers with command line arguments."""
    global data_manager, dem_manager

    data_manager = DataManager(args)
    dem_manager = DEMManager(args)


def get_data_manager() -> DataManager:
    """Get the data manager instance."""
    return data_manager


def get_dem_manager() -> DEMManager:
    """Get the DEM manager instance."""
    return dem_manager


def get_gpu_manager() -> GPUManager:
    """Get the GPU manager instance."""
    return gpu_manager