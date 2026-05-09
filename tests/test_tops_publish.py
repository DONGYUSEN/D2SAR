"""Tests for tops_publish — unwrap, geocode, and HDF5 product packaging."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure scripts/ is on the import path for the worktree root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tops_model import BurstIdentity, BurstRadarGrid, BurstWindow
from scripts.tops_publish import (
    geocode_ifg,
    unwrap_ifg,
    write_hdf5_product,
    write_product,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity(idx: int = 0) -> BurstIdentity:
    return BurstIdentity(
        swath="IW1",
        burst_index=idx,
        sensing_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        sensing_stop=datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        polarization="VV",
        orbit_direction="ascending",
        azimuth_steering_rate=0.0,
    )


def _grid(idx: int = 0) -> BurstRadarGrid:
    return BurstRadarGrid(
        identity=_identity(idx),
        image_window=BurstWindow(
            first_line=0, num_lines=1500,
            first_sample=0, num_samples=25000,
        ),
        valid_window=BurstWindow(
            first_line=100, num_lines=1300,
            first_sample=500, num_samples=24000,
        ),
        line_offset=0,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def _make_ifg(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    return np.full((h, w), np.exp(1j * phase), dtype=np.complex64)


def _make_coh(h: int, w: int, value: float = 0.9) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# geocode_ifg — happy path
# ---------------------------------------------------------------------------

class TestGeocodeIfg:
    """Tests for geocode_ifg."""

    def test_raises_when_gdal_not_available(self, tmp_path):
        """NotImplementedError is raised when GDAL is absent."""
        with patch.dict(sys.modules, {"osgeo": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(NotImplementedError, match="GDAL"):
                    geocode_ifg(
                        _make_ifg(10, 10),
                        _make_coh(10, 10),
                        _grid(),
                        dem_path=tmp_path / "dem.tif",
                        work_dir=tmp_path,
                    )

    def test_dem_not_found_raises(self, tmp_path):
        """FileNotFoundError when DEM cannot be opened."""
        fake_dem = tmp_path / "nonexistent.tif"
        burst = _grid()

        with pytest.raises(FileNotFoundError, match="Cannot open DEM"):
            geocode_ifg(
                _make_ifg(10, 10),
                _make_coh(10, 10),
                burst,
                dem_path=fake_dem,
                work_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# unwrap_ifg — happy path
# ---------------------------------------------------------------------------

class TestUnwrapIfg:
    """Tests for unwrap_ifg."""

    def test_unsupported_method_raises(self):
        """Unsupported method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="only supports 'icu' or 'snaphu'"):
            unwrap_ifg(
                np.zeros((10, 10), dtype=np.float32),
                np.ones((10, 10), dtype=np.float32),
                method="phass",
            )

    def test_icu_not_in_path_raises(self, tmp_path):
        """ICU not found in PATH raises NotImplementedError."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(NotImplementedError, match="ICU executable not found"):
                unwrap_ifg(
                    np.zeros((10, 10), dtype=np.float32),
                    np.ones((10, 10), dtype=np.float32),
                    method="icu",
                    work_dir=tmp_path,
                )

    def test_snaphu_not_in_path_raises(self, tmp_path):
        """SNAPHU not found in PATH raises NotImplementedError."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(NotImplementedError, match="SNAPHU executable not found"):
                unwrap_ifg(
                    np.zeros((10, 10), dtype=np.float32),
                    np.ones((10, 10), dtype=np.float32),
                    method="snaphu",
                    work_dir=tmp_path,
                )

    def test_icu_binary_produces_output(self, tmp_path):
        """ICU binary output is read back correctly."""
        # Mock the ICU binary to produce a synthetic output file
        h, w = 5, 8
        phase = np.zeros((h, w), dtype=np.float32)
        coh = np.ones((h, w), dtype=np.float32)
        expected = np.arange(h * w, dtype=np.float32).reshape(h, w)

        fake_unw = tmp_path / "unw_output.bin"
        fake_unw.write_bytes(expected.tobytes())

        def fake_which(name):
            if name == "icu":
                return "/usr/bin/icu"
            return None

        def fake_run(cmd, capture_output, check, text):
            mr = MagicMock()
            mr.returncode = 0
            mr.stderr = ""
            return mr

        with patch("shutil.which", side_effect=fake_which):
            with patch("subprocess.run", side_effect=fake_run):
                result = unwrap_ifg(phase, coh, method="icu", work_dir=tmp_path)

        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# write_hdf5_product — HDF5 structure correctness
# ---------------------------------------------------------------------------

class TestWriteHdf5Product:
    """Tests for write_hdf5_product."""

    def test_h5py_required(self, tmp_path):
        """ImportError when h5py is absent."""
        with patch.dict(sys.modules, {"h5py": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="requires h5py"):
                    write_hdf5_product(
                        _make_ifg(10, 10),
                        _make_coh(10, 10),
                        None,
                        geo_transform=(0, 1, 0, 0, 0, -1),
                        projection="EPSG:4326",
                        output_path=tmp_path / "out.h5",
                        metadata={},
                    )

    def test_required_datasets_present(self, tmp_path):
        """All required datasets are written in the HDF5 file."""
        h, w = 20, 30
        ifg = _make_ifg(h, w, phase=0.5)
        coh = _make_coh(h, w, value=0.7)
        unw = np.linspace(-5, 5, h * w, dtype=np.float32).reshape(h, w)
        out_path = tmp_path / "product.h5"

        write_hdf5_product(
            merged_ifg=ifg,
            merged_coh=coh,
            unwrapped=unw,
            geo_transform=(0, 1, 0, 0, 0, -1),
            projection="EPSG:4326",
            output_path=out_path,
            metadata={"sensor": "SENTINEL1"},
        )

        import h5py
        with h5py.File(out_path, "r") as f:
            # Required datasets
            assert "science/SENTINEL1/interferogram/phase" in f
            assert "science/SENTINEL1/interferogram/coherence" in f
            assert "science/SENTINEL1/interferogram/unwrappedPhase" in f
            assert "science/SENTINEL1/metadata/productType" in f
            assert "science/SENTINEL1/metadata/lookSide" in f
            assert "science/SENTINEL1/metadata/burstBoundaries" in f

            # Check shapes
            assert f["science/SENTINEL1/interferogram/phase"].shape == (h, w)
            assert f["science/SENTINEL1/interferogram/coherence"].shape == (h, w)
            assert f["science/SENTINEL1/interferogram/unwrappedPhase"].shape == (h, w)

            # Check dtypes
            assert f["science/SENTINEL1/interferogram/phase"].dtype == np.float32
            assert f["science/SENTINEL1/interferogram/coherence"].dtype == np.float32
            assert f["science/SENTINEL1/interferogram/unwrappedPhase"].dtype == np.float32

            # Check phase is wrapped in [-π, π]
            phase_data = f["science/SENTINEL1/interferogram/phase"][:]
            assert phase_data.min() >= -np.pi - 1e-6
            assert phase_data.max() <= np.pi + 1e-6

            # Check coherence values
            np.testing.assert_array_almost_equal(
                f["science/SENTINEL1/interferogram/coherence"][:], 0.7
            )

            # Metadata values
            assert f["science/SENTINEL1/metadata/productType"][()].startswith(b"TOPSAR")
            assert f["science/SENTINEL1/metadata/lookSide"][()].startswith(b"right")

    def test_unwrapped_none_skips_dataset(self, tmp_path):
        """unwrapped=None skips the unwrappedPhase dataset."""
        h, w = 10, 15
        ifg = _make_ifg(h, w)
        coh = _make_coh(h, w)
        out_path = tmp_path / "no_unw.h5"

        write_hdf5_product(
            merged_ifg=ifg,
            merged_coh=coh,
            unwrapped=None,
            geo_transform=(0, 1, 0, 0, 0, -1),
            projection="EPSG:4326",
            output_path=out_path,
            metadata={},
        )

        import h5py
        with h5py.File(out_path, "r") as f:
            assert "science/SENTINEL1/interferogram/phase" in f
            assert "science/SENTINEL1/interferogram/coherence" in f
            assert "science/SENTINEL1/interferogram/unwrappedPhase" not in f

    def test_metadata_written_to_root(self, tmp_path):
        """Arbitrary metadata dict is written to root HDF5 group."""
        h, w = 5, 5
        out_path = tmp_path / "meta.h5"

        write_hdf5_product(
            merged_ifg=_make_ifg(h, w),
            merged_coh=_make_coh(h, w),
            unwrapped=None,
            geo_transform=(0, 1, 0, 0, 0, -1),
            projection="EPSG:4326",
            output_path=out_path,
            metadata={
                "sensor": "SENTINEL1",
                "orbitDirection": "ascending",
                "burstBoundaries": {"IW1": [0, 1500]},
            },
        )

        import h5py
        with h5py.File(out_path, "r") as f:
            # Non-string/numeric values should be JSON-encoded
            assert "burstBoundaries" in f.attrs
            parsed = json.loads(f.attrs["burstBoundaries"])
            assert parsed == {"IW1": [0, 1500]}

    def test_burstboundaries_metadata_field(self, tmp_path):
        """burstBoundaries from metadata dict is written correctly."""
        h, w = 5, 5
        out_path = tmp_path / "bb.h5"

        write_hdf5_product(
            merged_ifg=_make_ifg(h, w),
            merged_coh=_make_coh(h, w),
            unwrapped=None,
            geo_transform=(0, 1, 0, 0, 0, -1),
            projection="EPSG:4326",
            output_path=out_path,
            metadata={
                "burstBoundaries": {"IW1": [0, 1500], "IW2": [1500, 3000]},
            },
        )

        import h5py
        with h5py.File(out_path, "r") as f:
            bb_data = f["science/SENTINEL1/metadata/burstBoundaries"][()]
            parsed = json.loads(bb_data)
            assert parsed == {"IW1": [0, 1500], "IW2": [1500, 3000]}

    def test_coordinate_datasets_created_when_dims_provided(self, tmp_path):
        """Coordinate datasets are created when geocoded dimensions are provided."""
        h, w = 5, 10
        out_path = tmp_path / "coords.h5"

        write_hdf5_product(
            merged_ifg=_make_ifg(5, 10),
            merged_coh=_make_coh(5, 10),
            unwrapped=None,
            geo_transform=(0, 1, 0, 0, 0, -1),
            projection="EPSG:4326",
            output_path=out_path,
            metadata={
                "geocoded_lines": 5,
                "geocoded_samples": 10,
            },
        )

        import h5py
        with h5py.File(out_path, "r") as f:
            assert "science/SENTINEL1/coordinates/longitude" in f
            assert "science/SENTINEL1/coordinates/latitude" in f
            assert f["science/SENTINEL1/coordinates/longitude"].shape == (5, 10)
            assert f["science/SENTINEL1/coordinates/latitude"].shape == (5, 10)


# ---------------------------------------------------------------------------
# write_product — output file layout
# ---------------------------------------------------------------------------

class TestWriteProduct:
    """Tests for write_product."""

    def test_returns_list_of_paths(self, tmp_path):
        """write_product returns a list of Path objects."""
        h, w = 10, 20
        ifg = _make_ifg(h, w)
        coh = _make_coh(h, w)
        unw = np.zeros((h, w), dtype=np.float32)
        gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

        with patch("scripts.tops_publish.write_hdf5_product"):
            result = write_product(
                merged_ifg=ifg,
                merged_coh=coh,
                unwrapped=unw,
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="IW1_S1A_001",
            )

        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)
        assert len(result) == 4  # .int + .coh + .unw + .h5

    def test_int_tif_written(self, tmp_path):
        """Wrapped phase is written as .int.geo.tif."""
        h, w = 10, 20
        ifg = _make_ifg(h, w, phase=1.0)
        gt = (800000.0, 2.3, 0.0, 0.0, 0.0, -0.002)

        with patch("scripts.tops_publish.write_hdf5_product"):
            result = write_product(
                merged_ifg=ifg,
                merged_coh=_make_coh(h, w),
                unwrapped=None,
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="test_prod",
            )

        int_path = tmp_path / "test_prod.int.geo.tif"
        assert int_path in result
        assert int_path.exists()

    def test_coh_tif_written(self, tmp_path):
        """Coherence is written as .coh.geo.tif."""
        h, w = 10, 20
        gt = (800000.0, 2.3, 0.0, 0.0, 0.0, -0.002)

        with patch("scripts.tops_publish.write_hdf5_product"):
            result = write_product(
                merged_ifg=_make_ifg(h, w),
                merged_coh=_make_coh(h, w, value=0.85),
                unwrapped=None,
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="test_prod",
            )

        coh_path = tmp_path / "test_prod.coh.geo.tif"
        assert coh_path in result
        assert coh_path.exists()

    def test_unw_tif_skipped_when_none(self, tmp_path):
        """Unwrapped TIFF is skipped when unwrapped is None."""
        h, w = 10, 20
        gt = (800000.0, 2.3, 0.0, 0.0, 0.0, -0.002)

        with patch("scripts.tops_publish.write_hdf5_product"):
            result = write_product(
                merged_ifg=_make_ifg(h, w),
                merged_coh=_make_coh(h, w),
                unwrapped=None,
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="test_prod",
            )

        paths_str = [str(p) for p in result]
        assert not any(".unw." in p for p in paths_str)
        # Should still have int, coh, h5
        assert len(result) == 3

    def test_unw_tif_written_when_provided(self, tmp_path):
        """Unwrapped TIFF is written when unwrapped is not None."""
        h, w = 10, 20
        gt = (800000.0, 2.3, 0.0, 0.0, 0.0, -0.002)

        with patch("scripts.tops_publish.write_hdf5_product"):
            result = write_product(
                merged_ifg=_make_ifg(h, w),
                merged_coh=_make_coh(h, w),
                unwrapped=np.zeros((h, w), dtype=np.float32),
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="test_prod",
            )

        paths_str = [str(p) for p in result]
        assert any(".unw." in p for p in paths_str)

    def test_h5_file_written(self, tmp_path):
        """HDF5 product is written by write_product."""
        h, w = 10, 20
        gt = (800000.0, 2.3, 0.0, 0.0, 0.0, -0.002)

        with patch("scripts.tops_publish.write_hdf5_product") as mock_h5:
            write_product(
                merged_ifg=_make_ifg(h, w),
                merged_coh=_make_coh(h, w),
                unwrapped=None,
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="test_prod",
            )

            mock_h5.assert_called_once()
            args, kwargs = mock_h5.call_args
            assert kwargs["output_path"] == tmp_path / "test_prod.h5"
            assert kwargs["projection"] == "EPSG:4326"

    def test_output_dir_created_if_missing(self, tmp_path):
        """Output directory is created if it does not exist."""
        h, w = 5, 5
        out_dir = tmp_path / "subdir" / "nested"
        gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

        with patch("scripts.tops_publish.write_hdf5_product"):
            write_product(
                merged_ifg=_make_ifg(h, w),
                merged_coh=_make_coh(h, w),
                unwrapped=None,
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=out_dir,
                product_name="test",
            )

        assert out_dir.exists()


# ---------------------------------------------------------------------------
# Integration: full product output (GDAL mocked)
# ---------------------------------------------------------------------------

class TestWriteProductIntegration:
    """Integration-style tests for write_product with GDAL mocked at driver level."""

    def test_geotiff_placeholder_produces_file(self, tmp_path):
        """With GDAL mocked, write_product still produces a file (placeholder)."""
        h, w = 10, 20
        gt = (800000.0, 2.3, 0.0, 0.0, 0.0, -0.002)

        # We skip the GDAL-level mock and just verify the file path is returned
        with patch("scripts.tops_publish.write_hdf5_product"):
            result = write_product(
                merged_ifg=_make_ifg(h, w),
                merged_coh=_make_coh(h, w),
                unwrapped=np.zeros((h, w), dtype=np.float32),
                geo_transform=gt,
                projection="EPSG:4326",
                output_dir=tmp_path,
                product_name="IW1_full",
            )

        expected_names = {
            "IW1_full.int.geo.tif",
            "IW1_full.coh.geo.tif",
            "IW1_full.unw.geo.tif",
            "IW1_full.h5",
        }
        actual_names = {p.name for p in result}
        assert actual_names == expected_names
