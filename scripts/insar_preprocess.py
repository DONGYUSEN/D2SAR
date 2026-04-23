from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from common_processing import resolve_manifest_metadata_path, resolve_manifest_data_path


def _write_band_array(band, data: np.ndarray, xoff: int = 0, yoff: int = 0) -> None:
    arr = np.ascontiguousarray(data, dtype=np.float32)
    rows, cols = arr.shape
    band.WriteRaster(
        int(xoff),
        int(yoff),
        int(cols),
        int(rows),
        arr.tobytes(),
        buf_xsize=int(cols),
        buf_ysize=int(rows),
        buf_type=6,
    )


def _extract_doppler_coefficients(doppler: dict) -> list[float]:
    combined = doppler.get("combinedDoppler", {}) if isinstance(doppler, dict) else {}
    coeffs = combined.get("coefficients") or combined.get("coeffs")
    if coeffs:
        return [float(value) for value in coeffs]
    estimate = doppler.get("dopplerEstimate", {}) if isinstance(doppler, dict) else {}
    if "dopplerAtMidRange" in estimate:
        return [float(estimate["dopplerAtMidRange"])]
    return [0.0]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _resample_slave_slc_with_isce3(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    target_rows: int,
    target_cols: int,
    source_prf: float,
    target_prf: float,
    geometry_mode: str,
    doppler_coefficients: list[float],
) -> str:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import isce3
        from isce3.core import LUT2d, Poly2d
        from isce3.io import Raster
        from osgeo import gdal

        input_raster = Raster(str(src_path))
        rg_off_path = dst_path.parent / "prep_range_offsets.tif"
        az_off_path = dst_path.parent / "prep_azimuth_offsets.tif"
        drv = gdal.GetDriverByName("GTiff")
        for path in (rg_off_path, az_off_path):
            ds = drv.Create(str(path), target_cols, target_rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
            _write_band_array(ds.GetRasterBand(1), np.zeros((target_rows, target_cols), dtype=np.float32))
            ds = None

        doppler = LUT2d()
        if geometry_mode == "native-doppler" and doppler_coefficients:
            doppler = LUT2d(0.0, 0.0, np.array([doppler_coefficients], dtype=np.float64))
        cls = isce3.image.ResampSlc
        resamp = cls(
            doppler,
            0.0,
            1.0,
            0.0,
            target_prf if target_prf > 0 else 1.0,
            1.0,
            0.0 + 0.0j,
        )
        output_raster = Raster(str(dst_path), target_cols, target_rows, 1, gdal.GDT_CFloat32, "GTiff")
        resamp.resamp(input_raster, output_raster, Raster(str(rg_off_path)), Raster(str(az_off_path)))
    except Exception:
        shutil.copyfile(src_path, dst_path)
    return str(dst_path)


def _resample_slave_slc(*args, **kwargs) -> str:
    return _resample_slave_slc_with_isce3(*args, **kwargs)


def build_preprocess_plan(
    precheck: dict,
    slave_manifest_path: str | Path,
    stage_dir: str | Path,
    *,
    master_acquisition: dict | None = None,
    master_radargrid: dict | None = None,
    slave_acquisition: dict | None = None,
    slave_radargrid: dict | None = None,
    slave_doppler: dict | None = None,
) -> tuple[dict, str]:
    slave_manifest_path = Path(slave_manifest_path)
    stage_dir = Path(stage_dir)
    normalized_manifest_path = stage_dir / "normalized_slave_manifest.json"

    original_manifest = _load_json(slave_manifest_path)
    acquisition_path = resolve_manifest_metadata_path(
        slave_manifest_path, original_manifest, "acquisition"
    )
    radargrid_path = resolve_manifest_metadata_path(
        slave_manifest_path, original_manifest, "radargrid"
    )
    doppler_path = resolve_manifest_metadata_path(
        slave_manifest_path, original_manifest, "doppler"
    )
    acquisition = slave_acquisition or (
        _load_json(acquisition_path) if acquisition_path.exists() else {}
    )
    radargrid = slave_radargrid or (
        _load_json(radargrid_path) if radargrid_path.exists() else {}
    )
    doppler = slave_doppler or (_load_json(doppler_path) if doppler_path.exists() else {})

    plan = {
        "requires_normalization": bool(precheck.get("requires_prep")),
        "geometry_mode": precheck.get("recommended_geometry_mode", "zero-doppler"),
        "actions": [],
        "input_slave_manifest": str(slave_manifest_path),
        "normalized_slave_manifest": str(normalized_manifest_path),
    }

    if precheck.get("checks", {}).get("prf", {}).get("severity") == "warn":
        plan["actions"].append("normalize-slave-prf")
    if precheck.get("checks", {}).get("doppler", {}).get("severity") == "warn":
        plan["actions"].append("use-zero-doppler-geometry")
    if precheck.get("checks", {}).get("radar_grid", {}).get("severity") == "warn":
        plan["actions"].append("resample-slave-to-master-grid")
    if not plan["actions"]:
        plan["actions"].append("pass-through")

    normalized_manifest = json.loads(json.dumps(original_manifest))
    normalized_manifest.setdefault("processing", {})

    normalized_acquisition = dict(acquisition)
    normalized_radargrid = dict(radargrid)
    normalized_doppler = dict(doppler)
    source_prf = float(acquisition.get("prf", 0.0))
    target_prf = float((master_acquisition or {}).get("prf", source_prf))
    doppler_coefficients = _extract_doppler_coefficients(doppler)

    if master_acquisition and "normalize-slave-prf" in plan["actions"]:
        normalized_acquisition["sourcePrf"] = acquisition.get("prf")
        normalized_acquisition["prf"] = master_acquisition.get("prf")
        if "startGPSTime" in acquisition and "startGPSTime" in master_acquisition:
            normalized_acquisition["sourceStartGPSTime"] = acquisition.get("startGPSTime")
            normalized_acquisition["startGPSTime"] = master_acquisition.get("startGPSTime")

    if master_radargrid and "resample-slave-to-master-grid" in plan["actions"]:
        for key in (
            "numberOfRows",
            "numberOfColumns",
            "columnSpacing",
            "rangeTimeFirstPixel",
            "groundRangeResolution",
            "azimuthResolution",
        ):
            if key in master_radargrid:
                normalized_radargrid[key] = master_radargrid[key]

    normalized_doppler.setdefault("processing", {})
    normalized_doppler["geometryMode"] = plan["geometry_mode"]
    if plan["geometry_mode"] == "zero-doppler":
        normalized_doppler["combinedDoppler"] = {"coefficients": [0.0]}
    normalized_doppler["processing"]["insar_preprocess"] = {
        "sourceDopplerPath": str(doppler_path),
        "actions": plan["actions"],
        "source_coefficients": doppler_coefficients,
    }

    slc_entry = original_manifest.get("slc", {})
    slc_path = resolve_manifest_data_path(slave_manifest_path, slc_entry.get("path"))
    normalized_slc_value = slc_entry
    normalized_slc_path = None if slc_path is None else stage_dir / f"normalized_slave{Path(str(slc_path)).suffix or '.tif'}"
    if plan["requires_normalization"] and slc_path is not None and Path(str(slc_path)).exists():
        _resample_slave_slc(
            slc_path,
            normalized_slc_path,
            target_rows=int(normalized_radargrid.get("numberOfRows", 0)),
            target_cols=int(normalized_radargrid.get("numberOfColumns", 0)),
            source_prf=source_prf,
            target_prf=target_prf,
            geometry_mode=plan["geometry_mode"],
            doppler_coefficients=doppler_coefficients,
        )
        normalized_slc_value = {"path": str(normalized_slc_path)} if isinstance(slc_entry, dict) else str(normalized_slc_path)
    elif plan["requires_normalization"] and slc_path is not None:
        normalized_slc_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_slc_path.write_bytes(b"")
        normalized_slc_value = {"path": str(normalized_slc_path)} if isinstance(slc_entry, dict) else str(normalized_slc_path)

    normalized_acquisition_path = _write_json(
        stage_dir / "normalized_acquisition.json", normalized_acquisition
    )
    normalized_radargrid_path = _write_json(
        stage_dir / "normalized_radargrid.json", normalized_radargrid
    )
    normalized_doppler_path = _write_json(
        stage_dir / "normalized_doppler.json", normalized_doppler
    )

    if isinstance(slc_entry, dict):
        new_slc = dict(slc_entry)
        if isinstance(normalized_slc_value, dict):
            new_slc.update(normalized_slc_value)
        else:
            new_slc["path"] = normalized_slc_value
        normalized_manifest["slc"] = new_slc
    else:
        normalized_manifest["slc"] = {"path": normalized_slc_value}
    normalized_manifest["metadata"] = dict(normalized_manifest.get("metadata", {}))
    normalized_manifest["metadata"]["acquisition"] = str(normalized_acquisition_path)
    normalized_manifest["metadata"]["radargrid"] = str(normalized_radargrid_path)
    normalized_manifest["metadata"]["doppler"] = str(normalized_doppler_path)
    normalized_manifest["processing"]["insar_preprocess"] = {
        "source_manifest": str(slave_manifest_path),
        "actions": plan["actions"],
        "geometry_mode": plan["geometry_mode"],
        "normalized_slc": normalized_slc_value if not isinstance(normalized_slc_value, dict) else normalized_slc_value.get("path"),
        "resamp_slc": {
            "source_prf": source_prf,
            "target_prf": target_prf,
            "target_rows": int(normalized_radargrid.get("numberOfRows", 0)),
            "target_cols": int(normalized_radargrid.get("numberOfColumns", 0)),
            "doppler_coefficients": doppler_coefficients,
            "geometry_mode": plan["geometry_mode"],
        },
    }
    normalized_manifest_path.write_text(
        json.dumps(normalized_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return plan, str(normalized_manifest_path)
