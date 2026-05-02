from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from osgeo import gdal, osr

from common_processing import (
    append_topo_coordinates_hdf,
    compute_rtc_factor,
    compute_utm_output_shape,
    point2epsg,
    resolve_manifest_data_path,
    resolve_manifest_metadata_path,
    write_geocoded_geotiff,
    write_geocoded_png,
)
from tops_geometry import iter_burst_radar_grids, load_tops_metadata, select_burst_doppler


def prepare_tops_rtc(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_tops_metadata(manifest_path)
    manifest = metadata["manifest"]
    if str(manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("tops_rtc requires a sentinel-1 manifest")

    slc_path = resolve_manifest_data_path(manifest_path, manifest.get("slc", {}).get("path"))
    if not slc_path:
        raise ValueError("manifest.slc.path is required")

    burst_metadata = metadata["tops"].get("bursts", [])
    burst_grids = list(iter_burst_radar_grids(manifest_path))
    if len(burst_grids) != len(burst_metadata):
        raise ValueError("burst radar grid count does not match TOPS burst metadata count")

    bursts = []
    for burst_grid, burst in zip(burst_grids, burst_metadata):
        burst_index = int(burst_grid["burstIndex"])
        bursts.append(
            {
                "burstIndex": burst_index,
                "radargrid": burst_grid,
                "doppler": select_burst_doppler(burst),
                "slcWindow": _build_slc_window(burst_grid),
                "outputs": _build_burst_outputs(output_dir, burst_index),
            }
        )

    plan = {
        "version": "1.0",
        "mode": "prepare-only",
        "sensor": manifest.get("sensor"),
        "swath": metadata["tops"].get("swath"),
        "polarisation": metadata["tops"].get("polarisation"),
        "burst_count": len(bursts),
        "input_manifest": str(manifest_path),
        "slc_path": slc_path,
        "bursts": bursts,
    }
    plan_path = output_dir / "tops_rtc_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"plan_path": str(plan_path), "burst_count": len(bursts)}


def _build_slc_window(burst_grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "xoff": 0,
        "yoff": int(burst_grid["lineOffset"]),
        "xsize": int(burst_grid["numberOfColumns"]),
        "ysize": int(burst_grid["numberOfRows"]),
        "validWindow": {
            "xoff": int(burst_grid["firstValidSample"]),
            "yoff": int(burst_grid["firstValidLine"]),
            "xsize": int(burst_grid["numValidSamples"]),
            "ysize": int(burst_grid["numValidLines"]),
        },
    }


def _build_burst_outputs(output_dir: Path, burst_index: int) -> dict[str, str]:
    burst_dir = output_dir / f"burst_{burst_index:03d}"
    return {
        "directory": str(burst_dir),
        "amplitude_h5": str(burst_dir / "amplitude_fullres.h5"),
        "rtc_factor_tif": str(burst_dir / "rtc_factor.tif"),
        "topo_h5": str(burst_dir / "topo.h5"),
        "amplitude_utm_tif": str(burst_dir / "amplitude_utm_geocoded.tif"),
        "amplitude_utm_png": str(burst_dir / "amplitude_utm_geocoded.png"),
        "metadata_json": str(burst_dir / "metadata.json"),
    }


def materialize_tops_rtc_plan(
    plan_path: str | Path,
    *,
    burst_limit: int | None = None,
    block_rows: int = 256,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    plan["plan_path"] = str(plan_path)

    bursts = plan.get("bursts", [])
    if burst_limit is not None:
        bursts = bursts[: max(0, int(burst_limit))]

    materialized = []
    for burst in bursts:
        output_h5 = burst["outputs"]["amplitude_h5"]
        write_burst_amplitude_hdf(plan["slc_path"], burst, output_h5, block_rows=block_rows)
        materialized.append({"burstIndex": burst["burstIndex"], "amplitude_h5": output_h5})
    return {"plan_path": str(plan_path), "burst_count": len(materialized), "bursts": materialized}


def write_burst_amplitude_hdf(
    slc_path: str,
    burst_plan: dict[str, Any],
    output_h5: str | Path,
    *,
    block_rows: int = 256,
) -> str:
    slc_ds = gdal.Open(slc_path)
    if slc_ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")

    slc_window = burst_plan["slcWindow"]
    xoff = int(slc_window["xoff"])
    yoff = int(slc_window["yoff"])
    width = int(slc_window["xsize"])
    length = int(slc_window["ysize"])
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        f.attrs["product_type"] = "sentinel_tops_burst_amplitude"
        f.attrs["burst_index"] = int(burst_plan["burstIndex"])
        f.attrs["source_slc"] = slc_path
        f.attrs["slc_window_xoff"] = xoff
        f.attrs["slc_window_yoff"] = yoff
        f.attrs["radiometry"] = "amplitude"
        f.attrs["value_domain"] = "linear"
        valid_window = slc_window["validWindow"]
        for key, value in valid_window.items():
            f.attrs[f"valid_window_{key}"] = int(value)

        chunk_rows = max(1, min(int(block_rows), length))
        chunk_cols = max(1, min(1024, width))
        d_amp = f.create_dataset(
            "slc_amplitude",
            shape=(length, width),
            dtype="f4",
            chunks=(chunk_rows, chunk_cols),
            compression="gzip",
            shuffle=True,
        )
        d_mask = f.create_dataset(
            "valid_mask",
            shape=(length, width),
            dtype="u1",
            chunks=(chunk_rows, chunk_cols),
            compression="gzip",
            shuffle=True,
        )
        d_mask[:, :] = 0
        valid_x = int(valid_window["xoff"])
        valid_y = int(valid_window["yoff"])
        valid_w = int(valid_window["xsize"])
        valid_h = int(valid_window["ysize"])
        d_mask[valid_y : valid_y + valid_h, valid_x : valid_x + valid_w] = 1

        band1 = slc_ds.GetRasterBand(1)
        band2 = slc_ds.GetRasterBand(2) if slc_ds.RasterCount >= 2 else None
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            block1 = band1.ReadAsArray(xoff, yoff + row0, width, rows)
            if block1 is None:
                raise RuntimeError("failed to read SLC block")
            if band2 is not None:
                block2 = band2.ReadAsArray(xoff, yoff + row0, width, rows)
                if block2 is None:
                    raise RuntimeError("failed to read SLC block")
                amplitude = np.sqrt(block1.astype(np.float32) ** 2 + block2.astype(np.float32) ** 2)
            else:
                amplitude = np.abs(block1).astype(np.float32)
            d_amp[row0 : row0 + rows, :] = amplitude.astype(np.float32)
    return str(output_h5)


def compute_burst_rtc_factor(
    plan_path: str | Path,
    dem_path: str | Path,
    *,
    burst_limit: int = 1,
    orbit_interp: str = "Legendre",
    compute_func=compute_rtc_factor,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    plan["plan_path"] = str(plan_path)

    bursts = plan.get("bursts", [])[: max(0, int(burst_limit))]
    if len(bursts) > 1:
        return compute_merged_rtc_factor(
            plan,
            bursts,
            dem_path,
            orbit_interp=orbit_interp,
            compute_func=compute_func,
        )

    outputs = []
    work_dir = plan_path.parent / "burst_manifests"
    for burst in bursts:
        burst_manifest = write_burst_metadata_manifest(plan, burst, work_dir)
        output_path = burst["outputs"]["rtc_factor_tif"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        compute_func(str(burst_manifest), str(dem_path), output_path, orbit_interp=orbit_interp)
        outputs.append(
            {
                "burstIndex": burst["burstIndex"],
                "rtc_factor_tif": output_path,
                "burst_manifest": str(burst_manifest),
            }
        )
    return {"plan_path": str(plan_path), "burst_count": len(outputs), "bursts": outputs}


def _merged_radar_grid_json(bursts: list[dict[str, Any]]) -> dict[str, Any]:
    if not bursts:
        raise ValueError("at least one burst is required")
    grids = [burst["radargrid"] for burst in bursts]
    dt = float(grids[0].get("rowSpacing", 0.0))
    dr = float(grids[0].get("columnSpacing", 0.0))
    if dt <= 0.0 or dr <= 0.0:
        raise ValueError("burst radar grid rowSpacing and columnSpacing must be positive")
    ref_start = min(float(grid["sensingStartGPSTime"]) for grid in grids)
    ref_range = min(_burst_range_start(grid) for grid in grids)
    out_rows = 0
    out_cols = 0
    for grid in grids:
        row_off = int(round((float(grid["sensingStartGPSTime"]) - ref_start) / dt))
        col_off = int(round((_burst_range_start(grid) - ref_range) / dr))
        first_line = int(grid.get("firstValidLine", 0))
        num_lines = int(grid.get("numValidLines", grid.get("numberOfRows", 0)))
        first_sample = int(grid.get("firstValidSample", 0))
        num_samples = int(grid.get("numValidSamples", grid.get("numberOfColumns", 0)))
        number_rows = int(grid.get("numberOfRows", first_line + num_lines))
        number_cols = int(grid.get("numberOfColumns", first_sample + num_samples))
        out_rows = max(out_rows, row_off + number_rows)
        out_cols = max(out_cols, col_off + number_cols)

    merged = dict(grids[0])
    merged.update(
        {
            "source": "sentinel-1-merged-tops-bursts",
            "burstCount": len(bursts),
            "numberOfRows": int(out_rows),
            "numberOfColumns": int(out_cols),
            "sensingStartGPSTime": float(ref_start),
            "startingRange": float(ref_range),
            "firstValidLine": 0,
            "numValidLines": int(out_rows),
            "firstValidSample": 0,
            "numValidSamples": int(out_cols),
        }
    )
    if "rangeTimeFirstPixel" in merged:
        import isce3.core

        merged["rangeTimeFirstPixel"] = 2.0 * float(ref_range) / isce3.core.speed_of_light
    return merged


def write_merged_metadata_manifest(
    plan: dict[str, Any],
    bursts: list[dict[str, Any]],
    work_dir: str | Path,
) -> Path:
    work_dir = Path(work_dir) / "mosaic"
    metadata_dir = work_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = Path(plan["input_manifest"])
    with source_manifest_path.open(encoding="utf-8") as f:
        source_manifest = json.load(f)

    metadata = _load_source_metadata(source_manifest_path, source_manifest)
    radargrid_path = metadata_dir / "radargrid.json"
    orbit_path = metadata_dir / "orbit.json"
    acquisition_path = metadata_dir / "acquisition.json"
    doppler_path = metadata_dir / "doppler.json"
    scene_path = metadata_dir / "scene.json"

    radargrid_path.write_text(json.dumps(_merged_radar_grid_json(bursts), indent=2), encoding="utf-8")
    orbit_path.write_text(json.dumps(metadata["orbit"], indent=2), encoding="utf-8")
    acquisition_path.write_text(json.dumps(metadata["acquisition"], indent=2), encoding="utf-8")
    doppler_path.write_text(json.dumps(_burst_doppler_as_combined(bursts[0]["doppler"]), indent=2), encoding="utf-8")
    scene_path.write_text(json.dumps(metadata["scene"], indent=2), encoding="utf-8")

    manifest_path = work_dir / "manifest.json"
    manifest = {
        "version": "1.0",
        "sensor": "sentinel-1",
        "productType": source_manifest.get("productType", "SLC"),
        "platform": source_manifest.get("platform"),
        "polarisation": plan.get("polarisation"),
        "slc": {"path": plan.get("slc_path")},
        "metadata": {
            "acquisition": str(acquisition_path),
            "orbit": str(orbit_path),
            "radargrid": str(radargrid_path),
            "doppler": str(doppler_path),
            "scene": str(scene_path),
        },
        "tops": {
            "mode": "IW",
            "swath": plan.get("swath"),
            "burst_count": len(bursts),
            "merged_burst_indices": [int(burst["burstIndex"]) for burst in bursts],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def compute_merged_rtc_factor(
    plan: dict[str, Any],
    bursts: list[dict[str, Any]],
    dem_path: str | Path,
    *,
    orbit_interp: str = "Legendre",
    compute_func=compute_rtc_factor,
) -> dict[str, Any]:
    plan_path = Path(plan.get("plan_path") or Path(plan["bursts"][0]["outputs"]["directory"]).parent / "tops_rtc_plan.json")
    work_dir = plan_path.parent / "burst_manifests"
    merged_manifest = write_merged_metadata_manifest(plan, bursts, work_dir)
    output_path = plan_path.parent / "mosaic_rtc_factor.tif"
    compute_func(str(merged_manifest), str(dem_path), str(output_path), orbit_interp=orbit_interp)
    return {
        "plan_path": str(plan_path),
        "processing_mode": "radar_mosaic",
        "burst_count": len(bursts),
        "mosaic": {
            "rtc_factor_tif": str(output_path),
            "manifest": str(merged_manifest),
        },
    }


def write_burst_metadata_manifest(plan: dict[str, Any], burst: dict[str, Any], work_dir: str | Path) -> Path:
    work_dir = Path(work_dir) / f"burst_{int(burst['burstIndex']):03d}"
    metadata_dir = work_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = Path(plan["input_manifest"])
    with source_manifest_path.open(encoding="utf-8") as f:
        source_manifest = json.load(f)

    metadata = _load_source_metadata(source_manifest_path, source_manifest)
    radargrid_path = metadata_dir / "radargrid.json"
    orbit_path = metadata_dir / "orbit.json"
    acquisition_path = metadata_dir / "acquisition.json"
    doppler_path = metadata_dir / "doppler.json"
    scene_path = metadata_dir / "scene.json"

    radargrid_path.write_text(json.dumps(burst["radargrid"], indent=2), encoding="utf-8")
    orbit_path.write_text(json.dumps(metadata["orbit"], indent=2), encoding="utf-8")
    acquisition_path.write_text(json.dumps(metadata["acquisition"], indent=2), encoding="utf-8")
    doppler_path.write_text(json.dumps(_burst_doppler_as_combined(burst["doppler"]), indent=2), encoding="utf-8")
    scene_path.write_text(json.dumps(metadata["scene"], indent=2), encoding="utf-8")

    burst_manifest = {
        "version": "1.0",
        "sensor": "sentinel-1",
        "productType": source_manifest.get("productType", "SLC"),
        "platform": source_manifest.get("platform"),
        "polarisation": plan.get("polarisation"),
        "slc": {"path": plan["slc_path"]},
        "metadata": {
            "acquisition": str(acquisition_path),
            "orbit": str(orbit_path),
            "radargrid": str(radargrid_path),
            "doppler": str(doppler_path),
            "scene": str(scene_path),
        },
        "tops": {
            "mode": "IW",
            "swath": plan.get("swath"),
            "burst_index": burst["burstIndex"],
            "burst_count": 1,
        },
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(burst_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def compute_burst_topo(
    plan_path: str | Path,
    dem_path: str | Path,
    *,
    burst_limit: int = 1,
    block_rows: int = 256,
    orbit_interp: str | None = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
    topo_func=append_topo_coordinates_hdf,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)

    bursts = plan.get("bursts", [])[: max(0, int(burst_limit))]
    work_dir = plan_path.parent / "burst_manifests"
    outputs = []
    for burst in bursts:
        burst_manifest = write_burst_metadata_manifest(plan, burst, work_dir)
        output_h5 = burst["outputs"]["amplitude_h5"]
        if not Path(output_h5).is_file():
            write_burst_amplitude_hdf(plan["slc_path"], burst, output_h5, block_rows=block_rows)
        topo_func(
            str(burst_manifest),
            str(dem_path),
            output_h5,
            block_rows=block_rows,
            orbit_interp=orbit_interp,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        outputs.append(
            {
                "burstIndex": burst["burstIndex"],
                "amplitude_h5": output_h5,
                "topo_h5": output_h5,
                "burst_manifest": str(burst_manifest),
            }
        )
    return {"plan_path": str(plan_path), "burst_count": len(outputs), "bursts": outputs}


def _require_topo_datasets(output_h5: str | Path, amplitude_dataset: str = "slc_amplitude") -> None:
    with h5py.File(output_h5, "r") as f:
        missing = [name for name in (amplitude_dataset, "longitude", "latitude", "height") if name not in f]
    if missing:
        raise ValueError(f"required datasets missing from {output_h5}: {', '.join(missing)}")


def _load_source_metadata(source_manifest_path: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    loaded = {}
    for key in ("acquisition", "orbit", "scene"):
        with open(resolve_manifest_metadata_path(source_manifest_path, source_manifest, key), encoding="utf-8") as f:
            loaded[key] = json.load(f)
    return loaded


def _burst_doppler_as_combined(doppler: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "sentinel-1-burst",
        "combinedDoppler": {
            "polynomialDegree": max(len(doppler.get("coefficients", [])) - 1, 0),
            "referencePoint": doppler.get("t0", 0.0),
            "coefficients": doppler.get("coefficients", []),
        },
    }


def apply_burst_rtc_factor(
    amplitude_h5: str | Path,
    rtc_factor_tif: str | Path,
    output_h5: str | Path,
) -> str:
    amp_path = Path(amplitude_h5)
    factor_path = Path(rtc_factor_tif)
    out_path = Path(output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    factor_ds = gdal.Open(str(factor_path))
    if factor_ds is None:
        raise RuntimeError(f"failed to open RTC factor: {factor_path}")
    factor_band = factor_ds.GetRasterBand(1)
    with h5py.File(amp_path, "r") as f_in:
        amp = f_in["slc_amplitude"][:]
        valid = f_in["valid_mask"][:]
        shape = amp.shape

        factor_data = factor_band.ReadAsArray(0, 0, shape[1], shape[0])
        if factor_data is None:
            raise RuntimeError("failed to read RTC factor block")

        factor_data = np.where(np.isfinite(factor_data) & (factor_data > 0), factor_data, np.nan)

        rtc_amp = np.zeros_like(amp)
        valid_mask = (valid == 1) & np.isfinite(factor_data) & (factor_data > 0)
        rtc_amp[valid_mask] = (amp[valid_mask] / np.sqrt(factor_data[valid_mask])).astype(np.float32)

        with h5py.File(out_path, "w") as f_out:
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
            f_out.attrs["product_type"] = "sentinel_tops_burst_rtc_amplitude"
            f_out.attrs["rtc_factor_source"] = str(factor_path.resolve())

            for ds_name in ("slc_amplitude", "valid_mask", "longitude", "latitude", "height"):
                if ds_name in f_in:
                    f_in.copy(ds_name, f_out)

            chunk_rows = max(1, min(256, shape[0]))
            chunk_cols = max(1, min(1024, shape[1]))
            d_rtc = f_out.create_dataset(
                "rtc_amplitude",
                shape=shape,
                dtype=np.float32,
                chunks=(chunk_rows, chunk_cols),
                compression="gzip",
                shuffle=True,
            )
            d_rtc[:] = rtc_amp.astype(np.float32)

    factor_ds = None
    return str(out_path)


def apply_burst_rtc(
    plan_path: str | Path,
    *,
    burst_limit: int = 1,
    dem_path: str | Path | None = None,
    block_rows: int = 256,
    orbit_interp: str | None = None,
    resolution_meters: float = 20.0,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    plan["plan_path"] = str(plan_path)

    bursts = plan.get("bursts", [])[: max(0, int(burst_limit))]
    if len(bursts) > 1:
        return apply_merged_rtc(
            plan,
            bursts,
            dem_path=dem_path,
            block_rows=block_rows,
            orbit_interp=orbit_interp,
            resolution_meters=resolution_meters,
        )

    outputs = []
    for burst in bursts:
        amplitude_h5 = burst["outputs"]["amplitude_h5"]
        rtc_factor_tif = burst["outputs"]["rtc_factor_tif"]
        rtc_h5 = str(Path(burst["outputs"]["directory"]) / "amplitude_rtc.h5")
        result_path = apply_burst_rtc_factor(amplitude_h5, rtc_factor_tif, rtc_h5)
        outputs.append(
            {
                "burstIndex": burst["burstIndex"],
                "amplitude_h5": amplitude_h5,
                "rtc_factor_tif": rtc_factor_tif,
                "rtc_h5": result_path,
            }
        )
    return {"plan_path": str(plan_path), "burst_count": len(outputs), "bursts": outputs}


def apply_merged_rtc(
    plan: dict[str, Any],
    bursts: list[dict[str, Any]],
    *,
    dem_path: str | Path | None = None,
    block_rows: int = 256,
    orbit_interp: str | None = None,
    resolution_meters: float = 20.0,
) -> dict[str, Any]:
    plan_path = Path(plan["plan_path"])
    merged_h5 = plan_path.parent / "mosaic_slc_amplitude_radar.h5"
    factor_path = plan_path.parent / "mosaic_rtc_factor.tif"
    rtc_h5 = plan_path.parent / "mosaic_rtc_amplitude_radar.h5"
    merge_result = merge_bursts_radar_grid(
        bursts,
        "slc_amplitude",
        merged_h5,
        include_geometry=False,
        overlap_method="top",
    )
    topo_manifest = None
    if dem_path is not None:
        topo_manifest = write_merged_metadata_manifest(
            plan,
            bursts,
            plan_path.parent / "burst_manifests",
        )
        append_topo_coordinates_hdf(
            str(topo_manifest),
            str(dem_path),
            str(merged_h5),
            block_rows=block_rows,
            orbit_interp=orbit_interp,
        )
    result_path = apply_burst_rtc_factor(merged_h5, factor_path, rtc_h5)
    if dem_path is not None:
        with h5py.File(merged_h5, "r") as f_in, h5py.File(result_path, "a") as f_out:
            for ds_name in ("longitude", "latitude", "height"):
                if ds_name in f_in and ds_name not in f_out:
                    f_in.copy(ds_name, f_out)
        slc_geocoded = export_radar_hdf_geocoded(
            merged_h5,
            "slc_amplitude",
            plan_path.parent / "mosaic_slc_amplitude_geocoded",
            resolution_meters=resolution_meters,
            block_rows=block_rows,
        )
        rtc_geocoded = export_radar_hdf_geocoded(
            result_path,
            "rtc_amplitude",
            plan_path.parent / "mosaic_rtc_amplitude_geocoded",
            resolution_meters=resolution_meters,
            block_rows=block_rows,
        )
    else:
        slc_geocoded = None
        rtc_geocoded = None
    return {
        "plan_path": str(plan_path),
        "processing_mode": "radar_mosaic",
        "burst_count": len(bursts),
        "mosaic": {
            "amplitude_h5": str(merged_h5),
            "rtc_factor_tif": str(factor_path),
            "rtc_h5": result_path,
            "merge": merge_result,
            "topo_manifest": str(topo_manifest) if topo_manifest is not None else None,
            "slc_geocoded": slc_geocoded,
            "rtc_geocoded": rtc_geocoded,
        },
    }


def _burst_range_start(grid: dict[str, Any]) -> float:
    if "startingRange" in grid:
        return float(grid["startingRange"])
    if "rangeTimeFirstPixel" in grid:
        return float(grid["rangeTimeFirstPixel"]) * 299792458.0 / 2.0
    return 0.0


def _can_merge_bursts_radar_grid(bursts: list[dict[str, Any]], dataset_name: str) -> bool:
    if not bursts:
        return False
    for burst in bursts:
        if "radargrid" not in burst or "outputs" not in burst:
            return False
        if dataset_name == "rtc_amplitude":
            data_h5 = Path(burst["outputs"]["directory"]) / "amplitude_rtc.h5"
        else:
            data_h5 = Path(burst["outputs"]["amplitude_h5"])
        if not data_h5.is_file():
            return False
        try:
            with h5py.File(data_h5, "r") as f:
                required = (dataset_name, "valid_mask", "longitude", "latitude", "height")
                if any(name not in f for name in required):
                    return False
        except OSError:
            return False
    return True


def _burst_data_h5_path(burst: dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "rtc_amplitude":
        return str(Path(burst["outputs"]["directory"]) / "amplitude_rtc.h5")
    return burst["outputs"]["amplitude_h5"]

def merge_bursts_radar_grid(
    bursts: list[dict[str, Any]],
    dataset_name: str,
    output_h5: str | Path,
    *,
    block_rows: int = 256,
    overlap_method: str = "top",
    include_geometry: bool = True,
) -> dict[str, Any]:
    """Merge TOPS bursts in radar geometry, following ISCE2 valid-window stitching."""

    if overlap_method not in {"top", "bot", "avg"}:
        raise ValueError("overlap_method must be one of: top, bot, avg")
    if not bursts:
        raise ValueError("at least one burst is required")

    grids = [burst["radargrid"] for burst in bursts]
    dt = float(grids[0].get("rowSpacing", 0.0))
    dr = float(grids[0].get("columnSpacing", 0.0))
    if dt <= 0.0 or dr <= 0.0:
        raise ValueError("burst radar grid rowSpacing and columnSpacing must be positive")

    ref_start = min(float(grid["sensingStartGPSTime"]) for grid in grids)
    ref_range = min(_burst_range_start(grid) for grid in grids)

    placements = []
    out_rows = 0
    out_cols = 0
    for burst, grid in zip(bursts, grids):
        data_h5 = _burst_data_h5_path(burst, dataset_name)
        row_off = int(round((float(grid["sensingStartGPSTime"]) - ref_start) / dt))
        col_off = int(round((_burst_range_start(grid) - ref_range) / dr))
        first_line = int(grid.get("firstValidLine", 0))
        num_lines = int(grid.get("numValidLines", grid.get("numberOfRows", 0)))
        first_sample = int(grid.get("firstValidSample", 0))
        num_samples = int(grid.get("numValidSamples", grid.get("numberOfColumns", 0)))
        if num_lines <= 0 or num_samples <= 0:
            raise ValueError(f"burst {burst.get('burstIndex')} has an empty valid window")
        number_rows = int(grid.get("numberOfRows", first_line + num_lines))
        number_cols = int(grid.get("numberOfColumns", first_sample + num_samples))
        dst_row = row_off + first_line
        dst_col = col_off + first_sample
        out_rows = max(out_rows, row_off + number_rows)
        out_cols = max(out_cols, col_off + number_cols)
        placements.append(
            {
                "burst": burst,
                "data_h5": data_h5,
                "first_line": first_line,
                "num_lines": num_lines,
                "first_sample": first_sample,
                "num_samples": num_samples,
                "dst_row": dst_row,
                "dst_col": dst_col,
            }
        )

    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    chunk_rows = max(1, min(int(block_rows), out_rows))
    chunk_cols = max(1, min(1024, out_cols))

    with h5py.File(output_h5, "w") as f_out:
        f_out.attrs["product_type"] = "sentinel_tops_radar_mosaic"
        f_out.attrs["source_dataset"] = dataset_name
        f_out.attrs["mosaic_source"] = "isce2_style_radar_grid"
        f_out.attrs["overlap_method"] = overlap_method
        f_out.attrs["sensingStartGPSTime"] = ref_start
        f_out.attrs["rowSpacing"] = dt
        f_out.attrs["rangeStart"] = ref_range
        f_out.attrs["columnSpacing"] = dr

        data_out = f_out.create_dataset(
            dataset_name,
            shape=(out_rows, out_cols),
            dtype=np.float32,
            chunks=(chunk_rows, chunk_cols),
            compression="gzip",
            shuffle=True,
            fillvalue=0.0,
        )
        mask_out = f_out.create_dataset(
            "valid_mask",
            shape=(out_rows, out_cols),
            dtype=np.uint8,
            chunks=(chunk_rows, chunk_cols),
            compression="gzip",
            shuffle=True,
            fillvalue=0,
        )
        coord_out = {}
        if include_geometry:
            for name in ("longitude", "latitude", "height"):
                coord_out[name] = f_out.create_dataset(
                    name,
                    shape=(out_rows, out_cols),
                    dtype=np.float32,
                    chunks=(chunk_rows, chunk_cols),
                    compression="gzip",
                    shuffle=True,
                    fillvalue=np.nan,
                )

        for placement in placements:
            src_row = placement["first_line"]
            rows = placement["num_lines"]
            src_col = placement["first_sample"]
            cols = placement["num_samples"]
            dst_row = placement["dst_row"]
            dst_col = placement["dst_col"]
            with h5py.File(placement["data_h5"], "r") as f_in:
                src_valid = f_in["valid_mask"][src_row : src_row + rows, src_col : src_col + cols] == 1
                src_data = f_in[dataset_name][src_row : src_row + rows, src_col : src_col + cols].astype(np.float32)
                src_valid &= np.isfinite(src_data)
                if dataset_name in {"slc_amplitude", "rtc_amplitude"}:
                    src_valid &= src_data > 0.0

                target_slice = np.s_[dst_row : dst_row + rows, dst_col : dst_col + cols]
                existing_valid = mask_out[target_slice][:] == 1
                if overlap_method == "top":
                    write_mask = src_valid & ~existing_valid
                elif overlap_method == "bot":
                    write_mask = src_valid
                else:
                    write_mask = src_valid & ~existing_valid
                    avg_mask = src_valid & existing_valid
                    if np.any(avg_mask):
                        current = data_out[target_slice][:]
                        current[avg_mask] = 0.5 * (current[avg_mask] + src_data[avg_mask])
                        data_out[target_slice] = current

                if np.any(write_mask):
                    current = data_out[target_slice][:]
                    current_mask = mask_out[target_slice][:]
                    current[write_mask] = src_data[write_mask]
                    current_mask[write_mask] = 1
                    data_out[target_slice] = current
                    mask_out[target_slice] = current_mask
                    if include_geometry:
                        for name in ("longitude", "latitude", "height"):
                            src_coord = f_in[name][src_row : src_row + rows, src_col : src_col + cols].astype(np.float32)
                            current_coord = coord_out[name][target_slice][:]
                            current_coord[write_mask] = src_coord[write_mask]
                            coord_out[name][target_slice] = current_coord

        if include_geometry and "coordinate_system" not in f_out.attrs:
            f_out.attrs["coordinate_system"] = "EPSG:4326"
            f_out.attrs["longitude_units"] = "degrees_east"
            f_out.attrs["latitude_units"] = "degrees_north"
            f_out.attrs["height_units"] = "meters"
            f_out.attrs["coordinate_source"] = "isce2_style_radar_burst_mosaic"

    return {
        "burst_count": len(bursts),
        "mosaic_source": "radar",
        "mosaic_shape": (out_rows, out_cols),
        "radar_h5": str(output_h5),
        "overlap_method": overlap_method,
        "geometry_source": "per_burst" if include_geometry else "merged_topo_required",
    }




def _append_utm_coordinates_from_hdf_lonlat(input_h5: str | Path, block_rows: int = 64) -> str:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "a") as f:
        lon_ds = f["longitude"]
        lat_ds = f["latitude"]
        length, width = lon_ds.shape

        lon_sum = 0.0
        lat_sum = 0.0
        count = 0
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            lon = lon_ds[row0 : row0 + rows, :]
            lat = lat_ds[row0 : row0 + rows, :]
            valid = (
                np.isfinite(lon)
                & np.isfinite(lat)
                & (lon >= -180.0)
                & (lon <= 180.0)
                & (lat >= -90.0)
                & (lat <= 90.0)
            )
            if np.any(valid):
                lon_sum += float(np.sum(lon[valid], dtype=np.float64))
                lat_sum += float(np.sum(lat[valid], dtype=np.float64))
                count += int(valid.sum())
        if count == 0:
            raise ValueError(f"no valid lon/lat coordinates found in {input_h5}")
        epsg = point2epsg(lon_sum / count, lat_sum / count)

        src = osr.SpatialReference()
        src.ImportFromEPSG(4326)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(epsg)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(src, dst)

        for name in ("utm_x", "utm_y"):
            if name in f:
                del f[name]
        utm_x = f.create_dataset(
            "utm_x",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )
        utm_y = f.create_dataset(
            "utm_y",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )
        f.attrs["utm_epsg"] = epsg
        f.attrs["utm_coordinate_source"] = "transformed_from_mosaic_lonlat"
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            lon = lon_ds[row0 : row0 + rows, :]
            lat = lat_ds[row0 : row0 + rows, :]
            x_block = np.full((rows, width), np.nan, dtype=np.float32)
            y_block = np.full((rows, width), np.nan, dtype=np.float32)
            valid = (
                np.isfinite(lon)
                & np.isfinite(lat)
                & (lon >= -180.0)
                & (lon <= 180.0)
                & (lat >= -90.0)
                & (lat <= 90.0)
            )
            if np.any(valid):
                pts = np.column_stack([lon[valid], lat[valid]])
                transformed = np.asarray(transform.TransformPoints(pts[:, :2]), dtype=np.float64)
                x_block[valid] = transformed[:, 0].astype(np.float32)
                y_block[valid] = transformed[:, 1].astype(np.float32)
            utm_x[row0 : row0 + rows, :] = x_block
            utm_y[row0 : row0 + rows, :] = y_block
    return str(input_h5)


def export_radar_hdf_geocoded(
    input_h5: str | Path,
    dataset_name: str,
    output_prefix: str | Path,
    *,
    resolution_meters: float = 20.0,
    block_rows: int = 64,
) -> dict[str, Any]:
    export_block_rows = min(int(block_rows), 64)
    output_prefix = Path(output_prefix)
    geotiff = output_prefix.with_suffix(".tif")
    png = output_prefix.with_suffix(".png")
    _require_topo_datasets(input_h5, dataset_name)
    _append_utm_coordinates_from_hdf_lonlat(input_h5, block_rows=export_block_rows)
    target_width, target_height = compute_utm_output_shape(
        str(input_h5),
        resolution_meters,
        block_rows=export_block_rows,
    )
    write_geocoded_geotiff(
        str(input_h5),
        str(geotiff),
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=export_block_rows,
    )
    write_geocoded_png(
        str(input_h5),
        str(png),
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=export_block_rows,
    )
    return {
        "input_h5": str(input_h5),
        "dataset": dataset_name,
        "geotiff": str(geotiff),
        "png": str(png),
        "target_width": target_width,
        "target_height": target_height,
        "resolution_meters": float(resolution_meters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Sentinel TOPS per-burst RTC execution plan")
    parser.add_argument("manifest", help="Path to Sentinel importer manifest.json")
    parser.add_argument("output_dir", help="Output directory for tops_rtc_plan.json")
    parser.add_argument("--materialize", action="store_true", help="Write per-burst amplitude HDF5 files")
    parser.add_argument("--burst-limit", type=int, default=None, help="Limit materialization to first N bursts")
    parser.add_argument("--block-rows", type=int, default=256, help="Rows per SLC read block")
    parser.add_argument("--compute-rtc-factor", action="store_true", help="Compute per-burst RTC factor TIFFs")
    parser.add_argument("--dem", help="DEM path for RTC factor computation")
    parser.add_argument("--orbit-interp", default="Legendre", help="Orbit interpolation method")
    parser.add_argument("--compute-topo", action="store_true", help="Append per-burst topo coordinates to amplitude HDF5 files")
    parser.add_argument("--topo-gpu", action="store_true", help="Use ISCE3 CUDA Rdr2Geo for topo")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for topo when --topo-gpu is set")
    parser.add_argument("--apply-rtc", action="store_true", help="Apply RTC factor to amplitude to produce RTC amplitude HDF5")
    parser.add_argument("--resolution", type=float, default=20.0, help="Geocoded output resolution in meters")
    args = parser.parse_args()
    result = prepare_tops_rtc(args.manifest, args.output_dir)
    if args.materialize:
        result["materialized"] = materialize_tops_rtc_plan(
            result["plan_path"], burst_limit=args.burst_limit, block_rows=args.block_rows
        )
    if args.compute_rtc_factor:
        if not args.dem:
            raise SystemExit("--dem is required with --compute-rtc-factor")
        result["rtc_factor"] = compute_burst_rtc_factor(
            result["plan_path"],
            args.dem,
            burst_limit=args.burst_limit or 1,
            orbit_interp=args.orbit_interp,
        )
    if args.compute_topo:
        if not args.dem:
            raise SystemExit("--dem is required with --compute-topo")
        result["topo"] = compute_burst_topo(
            result["plan_path"],
            args.dem,
            burst_limit=args.burst_limit or 1,
            block_rows=args.block_rows,
            orbit_interp=args.orbit_interp,
            use_gpu=args.topo_gpu,
            gpu_id=args.gpu_id,
        )
    if args.apply_rtc:
        if (args.burst_limit or 1) > 1 and not args.dem:
            raise SystemExit("--dem is required with --apply-rtc when processing multiple bursts")
        result["rtc_applied"] = apply_burst_rtc(
            result["plan_path"],
            burst_limit=args.burst_limit or 1,
            dem_path=args.dem,
            block_rows=args.block_rows,
            orbit_interp=args.orbit_interp,
            resolution_meters=args.resolution,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
