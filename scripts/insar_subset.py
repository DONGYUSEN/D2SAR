from __future__ import annotations

import json
import shutil
from pathlib import Path

from osgeo import gdal

from common_processing import (
    manifest_relative_path,
    resolve_manifest_data_path,
    resolve_manifest_metadata_path,
)


def _is_vsi_path(path: str) -> bool:
    return str(path).startswith("/vsizip/") or str(path).startswith("/vsitar/")


def _copy_or_subset_raster(src_path: str, dst_path: Path, window: tuple[int, int, int, int] | None) -> str:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not _is_vsi_path(src_path) and not Path(src_path).exists():
        dst_path.write_bytes(b"")
        return str(dst_path)
    if window is None:
        shutil.copyfile(src_path, dst_path)
        return str(dst_path)
    try:
        ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    except Exception:
        dst_path.write_bytes(b"")
        return str(dst_path)
    if ds is None:
        dst_path.write_bytes(b"")
        return str(dst_path)
    row0, col0, rows, cols = window
    translated = gdal.Translate(
        str(dst_path),
        ds,
        srcWin=[col0, row0, cols, rows],
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )
    ds = None
    if translated is None:
        dst_path.write_bytes(b"")
        return str(dst_path)
    translated = None
    return str(dst_path)


def _build_subset_vrt(src_path: str, dst_path: Path, window: tuple[int, int, int, int]) -> str:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    except Exception:
        dst_path.write_text("", encoding="utf-8")
        return str(dst_path)
    if ds is None:
        dst_path.write_text("", encoding="utf-8")
        return str(dst_path)
    row0, col0, rows, cols = window
    translated = gdal.Translate(
        str(dst_path),
        ds,
        srcWin=[col0, row0, cols, rows],
        format="VRT",
    )
    ds = None
    if translated is None:
        dst_path.write_text("", encoding="utf-8")
        return str(dst_path)
    translated = None
    return str(dst_path)


def _subset_metadata(acquisition: dict, radargrid: dict, window: tuple[int, int, int, int] | None) -> tuple[dict, dict]:
    new_acq = dict(acquisition)
    new_rg = dict(radargrid)
    if window is None:
        return new_acq, new_rg
    row0, col0, rows, cols = window
    new_rg["numberOfRows"] = rows
    new_rg["numberOfColumns"] = cols
    if "rangeTimeFirstPixel" in radargrid and "columnSpacing" in radargrid:
        new_rg["rangeTimeFirstPixel"] = float(radargrid["rangeTimeFirstPixel"]) + (
            2.0 * col0 * float(radargrid["columnSpacing"]) / 299792458.0
        )
    if "startGPSTime" in acquisition and acquisition.get("prf"):
        new_acq["startGPSTime"] = float(acquisition["startGPSTime"]) + row0 / float(acquisition["prf"])
    return new_acq, new_rg


def _write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def build_cropped_manifest(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    output_name: str,
    window: tuple[int, int, int, int] | None,
) -> str:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_stem = output_name.replace("_", ".")
    slc_path = resolve_manifest_data_path(manifest_path, manifest["slc"]["path"])
    acquisition_path = resolve_manifest_metadata_path(manifest_path, manifest, "acquisition")
    radargrid_path = resolve_manifest_metadata_path(manifest_path, manifest, "radargrid")
    doppler_path = resolve_manifest_metadata_path(manifest_path, manifest, "doppler")
    scene_path = resolve_manifest_metadata_path(manifest_path, manifest, "scene")
    orbit_path = resolve_manifest_metadata_path(manifest_path, manifest, "orbit")

    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8")) if acquisition_path.exists() else {}
    radargrid = json.loads(radargrid_path.read_text(encoding="utf-8")) if radargrid_path.exists() else {}
    doppler = json.loads(doppler_path.read_text(encoding="utf-8")) if doppler_path.exists() else {}
    scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.exists() else {}
    orbit = json.loads(orbit_path.read_text(encoding="utf-8")) if orbit_path.exists() else {}

    new_acq, new_rg = _subset_metadata(acquisition, radargrid, window)
    manifest_out = output_dir / f"{output_stem}.json"
    acq_out = output_dir / f"{output_stem}.acquisition.json"
    rg_out = output_dir / f"{output_stem}.radargrid.json"
    dop_out = output_dir / f"{output_stem}.doppler.json"
    scene_out = output_dir / f"{output_stem}.scene.json"
    orbit_out = output_dir / f"{output_stem}.orbit.json"

    slc_entry_out = manifest.get("slc", {})
    if window is None:
        slc_out_value = slc_entry_out
    else:
        slc_tif_out = output_dir / f"{output_stem}.tif"
        slc_vrt_out = output_dir / f"{output_stem}.vrt"
        _copy_or_subset_raster(slc_path, slc_tif_out, window)
        _build_subset_vrt(slc_path, slc_vrt_out, window)
        slc_rel = manifest_relative_path(output_dir, slc_tif_out)
        slc_out_value = {"path": slc_rel} if isinstance(slc_entry_out, dict) else slc_rel

    _write_json(acq_out, new_acq)
    _write_json(rg_out, new_rg)
    _write_json(dop_out, doppler)
    _write_json(scene_out, scene)
    _write_json(orbit_out, orbit)

    new_manifest = json.loads(json.dumps(manifest))
    if isinstance(slc_entry_out, dict):
        new_slc = dict(slc_entry_out)
        if isinstance(slc_out_value, dict):
            new_slc.update(slc_out_value)
        else:
            new_slc["path"] = slc_out_value
        new_slc["rows"] = int(new_rg.get("numberOfRows", new_slc.get("rows", 0)))
        new_slc["columns"] = int(new_rg.get("numberOfColumns", new_slc.get("columns", 0)))
        new_manifest["slc"] = new_slc
    else:
        new_manifest["slc"] = {"path": slc_out_value}
    new_manifest["metadata"] = dict(new_manifest.get("metadata", {}))
    new_manifest["metadata"].update(
        {
            "acquisition": manifest_relative_path(output_dir, acq_out),
            "radargrid": manifest_relative_path(output_dir, rg_out),
            "doppler": manifest_relative_path(output_dir, dop_out),
            "scene": manifest_relative_path(output_dir, scene_out),
            "orbit": manifest_relative_path(output_dir, orbit_out),
        }
    )
    processing = dict(new_manifest.get("processing", {}))
    processing.pop("insar_preprocess", None)
    processing["insar_crop"] = {
        "source_manifest": str(manifest_path),
        "output_manifest": str(manifest_out),
        "master_window": None
        if window is None
        else {
            "row0": int(window[0]),
            "col0": int(window[1]),
            "rows": int(window[2]),
            "cols": int(window[3]),
        },
        "slc_tif": None if window is None else manifest_relative_path(output_dir, slc_tif_out),
        "slc_vrt": None if window is None else manifest_relative_path(output_dir, slc_vrt_out),
    }
    new_manifest["processing"] = processing
    manifest_out.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(manifest_out)
