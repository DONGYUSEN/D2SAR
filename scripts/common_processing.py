"""Shared processing utilities for D2SAR sensor workflows.

This module exposes the stable/common parts of the current custom processing
stack so different sensor pipelines can call the same functions directly.
Sensor-specific importers and experimental diagnostic scripts remain separate.
"""

from __future__ import annotations

import json
import os
import tempfile
import shutil
import colorsys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal, osr
from PIL import Image

from dem_manager import (
    normalize_scene_corners,
    scene_bbox_from_corners,
    dem_covers_scene_corners,
    fetch_dem,
    resolve_dem_for_scene,
)
from orbit_smooth import orbit_smooth, smooth_orbit_from_json, robust_scale

__all__ = [
    "gps_to_datetime",
    "choose_orbit_interp",
    "load_scene_corners_with_fallback",
    "construct_orbit",
    "construct_doppler_lut2d",
    "construct_radar_grid",
    "compute_rtc_factor",
    "write_rtc_hdf",
    "write_geocoded_tif_hdf",
    "append_topo_coordinates_hdf",
    "append_geogrid_coordinates_hdf",
    "append_corner_interpolated_coordinates_hdf",
    "point2epsg",
    "append_utm_coordinates_hdf",
    "accumulate_utm_grid",
    "prepare_display_grid",
    "write_geocoded_geotiff",
    "write_geocoded_png",
    "write_wrapped_phase_geotiff",
    "write_wrapped_phase_png",
    "normalize_scene_corners",
    "scene_bbox_from_corners",
    "dem_covers_scene_corners",
    "fetch_dem",
    "resolve_dem_for_scene",
    "orbit_smooth",
    "smooth_orbit_from_json",
    "robust_scale",
    "verify_look_direction_from_corner",
    "verify_look_direction_from_center",
    "get_local_look_vector",
    "verify_look_direction_gmtsar_style",
    "verify_and_correct_look_direction",
    "resolve_manifest_data_path",
    "resolve_manifest_metadata_path",
    "manifest_relative_path",
]


def _gdal_dtype_to_numpy(gdal_dtype: int) -> np.dtype:
    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
        gdal.GDT_CInt16: np.complex64,
        gdal.GDT_CInt32: np.complex64,
        gdal.GDT_CFloat32: np.complex64,
        gdal.GDT_CFloat64: np.complex128,
    }
    return np.dtype(mapping.get(gdal_dtype, np.float32))


def _numpy_dtype_to_gdal(np_dtype: np.dtype | type) -> int:
    dtype = np.dtype(np_dtype)
    if dtype == np.dtype(np.uint8):
        return gdal.GDT_Byte
    if dtype == np.dtype(np.float64):
        return gdal.GDT_Float64
    if dtype == np.dtype(np.complex64):
        return gdal.GDT_CFloat32
    if dtype == np.dtype(np.complex128):
        return gdal.GDT_CFloat64
    return gdal.GDT_Float32


def _read_band_array(
    band,
    xoff: int = 0,
    yoff: int = 0,
    xsize: int | None = None,
    ysize: int | None = None,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    if xsize is None:
        xsize = band.XSize
    if ysize is None:
        ysize = band.YSize
    np_dtype = np.dtype(dtype) if dtype is not None else _gdal_dtype_to_numpy(band.DataType)
    buf_type = _numpy_dtype_to_gdal(np_dtype)
    if buf_type == gdal.GDT_Float32 and np_dtype != np.dtype(np.float32):
        np_dtype = np.dtype(np.float32)
    raw = band.ReadRaster(
        int(xoff),
        int(yoff),
        int(xsize),
        int(ysize),
        buf_xsize=int(xsize),
        buf_ysize=int(ysize),
        buf_type=buf_type,
    )
    if raw is None:
        raise RuntimeError("failed to read raster block")
    return np.frombuffer(raw, dtype=np_dtype).reshape(int(ysize), int(xsize)).copy()


def _write_band_array(band, data: np.ndarray, xoff: int = 0, yoff: int = 0) -> None:
    arr = np.ascontiguousarray(data)
    buf_type = _numpy_dtype_to_gdal(arr.dtype)
    if buf_type == gdal.GDT_Float32 and arr.dtype != np.dtype(np.float32):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    rows, cols = arr.shape
    band.WriteRaster(
        int(xoff),
        int(yoff),
        int(cols),
        int(rows),
        arr.tobytes(),
        buf_xsize=int(cols),
        buf_ysize=int(rows),
        buf_type=buf_type,
    )


def manifest_relative_path(manifest_dir: str | Path, target_path: str | Path) -> str:
    manifest_dir = Path(manifest_dir)
    target_path = Path(target_path)
    if not target_path.is_absolute():
        return str(target_path)
    return os.path.relpath(str(target_path), str(manifest_dir.resolve()))


def resolve_manifest_data_path(manifest_path: str | Path, entry) -> str | None:
    def _remap_legacy_absolute_path(resolved: str) -> str:
        if not resolved.startswith("/"):
            return resolved
        candidates = []
        if resolved.startswith("/results/"):
            candidates.append(Path.cwd() / resolved.lstrip("/"))
        if resolved.startswith("/work/results/"):
            candidates.append(Path.cwd() / resolved[len("/work/"):].lstrip("/"))
        if resolved.startswith("/tmp/"):
            candidates.append(Path("/home/ysdong/Temp") / resolved[5:])
        if resolved.startswith("/temp/"):
            candidates.append(Path("/home/ysdong/Temp") / resolved[6:])
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return str(candidate)
        return resolved

    def _find_archive_candidate(path_str: str) -> str:
        path_obj = Path(path_str)
        candidates: list[Path] = [path_obj]
        name = path_obj.name
        if name:
            candidates.extend(
                [
                    manifest_dir / name,
                    manifest_dir / "temp" / name,
                    manifest_dir.parent / name,
                    manifest_dir.parent / "temp" / name,
                    Path.cwd() / name,
                    Path("/results") / name,
                    Path("/work") / name,
                    Path("/tmp") / name,
                    Path("/temp") / name,
                ]
            )
        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate.resolve())
            except Exception:
                continue
        return path_str

    def _normalize_vsi_archive_path(vsi_path: str) -> str:
        def _split_archive_member(path: str, storage: str) -> tuple[str, str] | None:
            lower = path.lower()
            if storage == "tar":
                markers = (".tar.gz/", ".tgz/", ".tar/")
            else:
                markers = (".zip/",)
            for marker in markers:
                idx = lower.find(marker)
                if idx < 0:
                    continue
                archive_end = idx + len(marker) - 1
                archive = path[:archive_end]
                member = path[archive_end + 1 :]
                if archive and member:
                    return archive, member
            return None

        if vsi_path.startswith("/vsitar/"):
            prefix = "/vsitar/"
            storage = "tar"
        elif vsi_path.startswith("/vsizip/"):
            prefix = "/vsizip/"
            storage = "zip"
        else:
            return vsi_path

        remainder = vsi_path[len(prefix) :]
        split = _split_archive_member(remainder, storage)
        if split is None:
            return vsi_path
        archive_part, member = split

        archive_path = Path(archive_part)
        if archive_path.is_absolute():
            resolved_archive = str(archive_path)
        else:
            candidate = (manifest_dir / archive_path).resolve()
            if candidate.exists():
                resolved_archive = str(candidate)
            else:
                rooted_candidate = _remap_legacy_absolute_path("/" + archive_part.lstrip("/"))
                if Path(rooted_candidate).exists():
                    resolved_archive = rooted_candidate
                else:
                    resolved_archive = str(candidate)
        if not Path(resolved_archive).exists():
            resolved_archive = _remap_legacy_absolute_path(resolved_archive)
        if not Path(resolved_archive).exists():
            resolved_archive = _find_archive_candidate(resolved_archive)
        return f"{prefix}{resolved_archive}/{member}"

    if entry is None:
        return None
    manifest_dir = Path(manifest_path).parent.resolve()
    if isinstance(entry, dict):
        path_value = entry.get("path")
        if path_value is None:
            return None
        base_path = Path(path_value)
        resolved = str(
            base_path if base_path.is_absolute() else (manifest_dir / base_path)
        )
        if not Path(resolved).exists():
            resolved = _remap_legacy_absolute_path(resolved)
        if entry.get("storage") == "tar":
            member = entry.get("member")
            if not member:
                raise ValueError(f"tar manifest entry missing member: {entry}")
            if not Path(resolved).exists():
                resolved = _find_archive_candidate(resolved)
            return f"/vsitar/{resolved}/{member}"
        if entry.get("storage") == "zip":
            member = entry.get("member")
            if not member:
                raise ValueError(f"zip manifest entry missing member: {entry}")
            if not Path(resolved).exists():
                resolved = _find_archive_candidate(resolved)
            resolved_path = Path(resolved)
            if resolved_path.suffix == ".zip":
                zip_stem = resolved_path.stem
                if member.startswith(zip_stem + ".zip/"):
                    member = zip_stem + "/" + member[len(zip_stem) + 5:]
            return f"/vsizip/{resolved}/{member}"
        return resolved
    entry = str(entry)
    if entry.startswith("/vsitar/"):
        return _normalize_vsi_archive_path(entry)
    if entry.startswith("/vsizip/"):
        return _normalize_vsi_archive_path(entry)
    entry_path = Path(entry)
    resolved = str(entry_path if entry_path.is_absolute() else (manifest_dir / entry_path))
    if not Path(resolved).exists():
        return _remap_legacy_absolute_path(resolved)
    return resolved


def resolve_manifest_metadata_path(
    manifest_path: str | Path,
    manifest: dict,
    key: str,
) -> Path:
    manifest_path = Path(manifest_path)
    metadata = manifest.get("metadata", {})
    entry = metadata.get(key)
    fallback_path = manifest_path.parent / "metadata" / f"{key}.json"
    if entry is not None:
        resolved = resolve_manifest_data_path(manifest_path, entry)
        if resolved is None:
            raise FileNotFoundError(f"metadata entry '{key}' is null in manifest")
        resolved_path = Path(resolved)
        if resolved_path.exists() or not fallback_path.exists():
            return resolved_path
        return fallback_path
    return fallback_path


def gps_to_datetime(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    elif not (ts[-6:-5] in ("+", "-") and ts[-6:].count(":") == 3):
        ts = ts + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def choose_orbit_interp(orbit_json: dict, acquisition_json: dict | None = None) -> str:
    """Choose the safest orbit interpolation method from orbit metadata.

    Prefer Hermite only when orbit timing and supplied velocities appear
    self-consistent; otherwise fall back to Legendre.
    """

    try:
        state_vectors = orbit_json.get("stateVectors", [])
        if len(state_vectors) < 8:
            return "Legendre"

        times = []
        pos = []
        vel = []
        for sv in state_vectors:
            if "gpsTime" in sv and sv["gpsTime"] is not None:
                times.append(float(sv["gpsTime"]))
            else:
                gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
                times.append(
                    (gps_to_datetime(sv["timeUTC"]) - gps_epoch).total_seconds()
                )
            pos.append([sv["posX"], sv["posY"], sv["posZ"]])
            vel.append([sv["velX"], sv["velY"], sv["velZ"]])

        t = np.asarray(times, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)

        if not (
            np.all(np.isfinite(t))
            and np.all(np.isfinite(pos))
            and np.all(np.isfinite(vel))
        ):
            return "Legendre"

        dt = np.diff(t)
        if len(dt) == 0 or np.any(dt <= 0):
            return "Legendre"

        median_dt = float(np.median(dt))
        if median_dt <= 0:
            return "Legendre"

        max_dev = float(np.max(np.abs(dt - median_dt)))
        if max_dev > 1e-3:
            return "Legendre"

        if acquisition_json is not None:
            start_gps = acquisition_json.get("startGPSTime")
            stop_gps = acquisition_json.get("stopGPSTime")
            if start_gps is None or stop_gps is None:
                return "Legendre"
            start_gps = float(start_gps)
            stop_gps = float(stop_gps)
            margin = 2.0 * median_dt
            if start_gps < t[0] + margin or stop_gps > t[-1] - margin:
                return "Legendre"

        fd_vel = (pos[2:] - pos[:-2]) / (t[2:, None] - t[:-2, None])
        vel_mid = vel[1:-1]
        denom = np.linalg.norm(fd_vel, axis=1)
        valid = denom > 0
        if not np.any(valid):
            return "Legendre"
        rel_err = np.linalg.norm(vel_mid[valid] - fd_vel[valid], axis=1) / denom[valid]
        if len(rel_err) == 0:
            return "Legendre"

        if (
            float(np.median(rel_err)) <= 1e-5
            and float(np.percentile(rel_err, 95)) <= 1e-4
        ):
            return "Hermite"
        return "Legendre"
    except Exception:
        return "Legendre"


def load_scene_corners_with_fallback(
    manifest_path: str | Path,
    manifest: dict,
    ancillary_override=None,
):
    manifest_path = Path(manifest_path)
    scene_json = resolve_manifest_metadata_path(manifest_path, manifest, "scene")
    if scene_json.exists():
        with open(scene_json, encoding="utf-8") as f:
            scene = json.load(f)
        corners = scene.get("sceneCorners") or []
        if corners:
            return corners

    ancillary = ancillary_override or manifest.get("ancillary", {})

    annotation = ancillary.get("annotationXML")
    if annotation:
        from tianyi_importer import TianyiImporter

        if isinstance(annotation, dict) and annotation.get("storage") in ("zip", "tar"):
            archive_path = resolve_manifest_data_path(
                manifest_path, {"path": annotation["path"]}
            )
            importer = TianyiImporter(archive_path)
            root = importer.parse_xml_root(annotation["member"])
        else:
            resolved_annotation = resolve_manifest_data_path(manifest_path, annotation)
            importer = TianyiImporter(str(Path(resolved_annotation).parent.parent))
            root = importer.parse_xml_root(resolved_annotation)
        scene_info = importer.extract_scene_info(root)
        return scene_info["sceneCorners"]

    meta_xml = ancillary.get("metaXML")
    if meta_xml:
        from lutan_importer import LutanImporter

        if isinstance(meta_xml, dict) and meta_xml.get("storage") in ("zip", "tar"):
            archive_path = resolve_manifest_data_path(
                manifest_path, {"path": meta_xml["path"]}
            )
            importer = LutanImporter(archive_path)
            root = importer.parse_meta_xml(meta_xml["member"])
        else:
            resolved_meta_xml = resolve_manifest_data_path(manifest_path, meta_xml)
            importer = LutanImporter(str(Path(resolved_meta_xml).parent))
            root = importer.parse_meta_xml(resolved_meta_xml)
        scene_info = importer.extract_scene_info(root)
        return list(scene_info["sceneCorners"].values())

    raise FileNotFoundError(
        "scene.json not found and no recoverable ancillary metadata available. "
        "Expected metadata/scene.json or ancillary.annotationXML/metaXML."
    )


def construct_orbit(orbit_json: dict, interp_method: str = "Hermite"):
    import isce3.core

    raw_datetimes = [
        gps_to_datetime(sv["timeUTC"]) for sv in orbit_json["stateVectors"]
    ]
    if len(raw_datetimes) >= 3:
        raw_seconds = np.array(
            [dt.timestamp() for dt in raw_datetimes], dtype=np.float64
        )
        dt_seconds = np.diff(raw_seconds)
        uniform_spacing = float(np.median(dt_seconds))
        if np.max(np.abs(dt_seconds - uniform_spacing)) < 1e-3:
            spacing_us = int(round(uniform_spacing * 1_000_000))
            raw_datetimes = [
                raw_datetimes[0] + timedelta(microseconds=i * spacing_us)
                for i in range(len(raw_datetimes))
            ]

    state_vectors = []
    for i, sv in enumerate(orbit_json["stateVectors"]):
        dt = isce3.core.DateTime(raw_datetimes[i])
        pos = np.array([sv["posX"], sv["posY"], sv["posZ"]], dtype=np.float64)
        vel = np.array([sv["velX"], sv["velY"], sv["velZ"]], dtype=np.float64)
        state_vectors.append(isce3.core.StateVector(dt, pos, vel))

    ref_dt = isce3.core.DateTime(
        gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
    )
    method_map = {
        "Hermite": isce3.core.OrbitInterpMethod.HERMITE,
        "Legendre": isce3.core.OrbitInterpMethod.LEGENDRE,
    }
    return isce3.core.Orbit(state_vectors, ref_dt, method_map[interp_method])


def construct_doppler_lut2d(
    doppler_json: dict,
    *,
    radargrid_json: dict,
    acquisition_json: dict,
    orbit_json: dict,
):
    import isce3.core

    combined = doppler_json["combinedDoppler"]
    degree = combined["polynomialDegree"]
    coeffs = combined["coefficients"]
    ref_range_time = float(combined["referencePoint"])

    starting_range = (
        isce3.core.speed_of_light * float(radargrid_json["rangeTimeFirstPixel"]) / 2.0
    )
    range_pixel_spacing = float(radargrid_json["columnSpacing"])
    width = int(radargrid_json["numberOfColumns"])
    if width <= 0:
        raise ValueError("radargrid_json.numberOfColumns must be positive")
    # Pad one extra sample at far range so ISCE3 boundary evaluations at the
    # last pixel center remain inside the LUT domain under floating-point error.
    x_coord = starting_range + range_pixel_spacing * np.arange(width + 1, dtype=np.float64)

    orbit_ref_dt = gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (orbit_ref_dt - gps_epoch).total_seconds()
    sensing_start = float(acquisition_json["startGPSTime"]) - orbit_ref_gps
    length = int(radargrid_json["numberOfRows"])
    if length <= 0:
        raise ValueError("radargrid_json.numberOfRows must be positive")
    prf = float(acquisition_json["prf"])
    if prf <= 0.0:
        raise ValueError("acquisition_json.prf must be positive")
    # Pad one extra azimuth sample so end-of-swath evaluations remain inside
    # the LUT domain under floating-point error.
    sensing_end = sensing_start + length / prf
    y_coord = np.array([sensing_start, sensing_end], dtype=np.float64)

    data = np.zeros((len(y_coord), len(x_coord)), dtype=np.float64)
    range_times = 2.0 * x_coord / isce3.core.speed_of_light
    row = np.zeros(len(x_coord), dtype=np.float64)
    for j, c in enumerate(coeffs):
        row += float(c) * np.power(range_times - ref_range_time, j)
    data[:] = row[None, :]

    return isce3.core.LUT2d(
        xstart=float(x_coord[0]),
        ystart=float(y_coord[0]),
        dx=(float(x_coord[-1]) - float(x_coord[0])) / max(len(x_coord) - 1, 1),
        dy=(float(y_coord[-1]) - float(y_coord[0])) / max(len(y_coord) - 1, 1),
        data=data,
        method="bilinear",
        b_error=True,
    )


def construct_radar_grid(
    radargrid_json: dict,
    acquisition_json: dict,
    orbit_json: dict,
):
    import isce3.core
    import isce3.product

    sensing_start_abs_gps = acquisition_json["startGPSTime"]
    orbit_ref_dt = gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (orbit_ref_dt - gps_epoch).total_seconds()
    sensing_start_rel = sensing_start_abs_gps - orbit_ref_gps

    wavelength = isce3.core.speed_of_light / acquisition_json["centerFrequency"]
    prf = acquisition_json["prf"]
    r0 = isce3.core.speed_of_light * radargrid_json["rangeTimeFirstPixel"] / 2.0
    range_pixel_spacing = radargrid_json["columnSpacing"]

    look_raw = acquisition_json.get("lookDirection", "RIGHT").strip().upper()
    look_side = (
        isce3.core.LookSide.Left if look_raw == "LEFT" else isce3.core.LookSide.Right
    )

    return isce3.product.RadarGridParameters(
        sensing_start=sensing_start_rel,
        wavelength=wavelength,
        prf=prf,
        starting_range=r0,
        range_pixel_spacing=range_pixel_spacing,
        lookside=look_side,
        length=radargrid_json["numberOfRows"],
        width=radargrid_json["numberOfColumns"],
        ref_epoch=isce3.core.DateTime(orbit_ref_dt),
    )


def compute_rtc_factor(
    manifest_path: str,
    dem_path: str,
    rtc_factor_path: str,
    orbit_interp: str | None = None,
    zero_doppler: bool = True,
) -> str:
    import isce3.core
    import isce3.geometry
    import isce3.io

    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "orbit"),
        encoding="utf-8",
    ) as f:
        orbit_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "radargrid"),
        encoding="utf-8",
    ) as f:
        radargrid_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "acquisition"),
        encoding="utf-8",
    ) as f:
        acquisition_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "doppler"),
        encoding="utf-8",
    ) as f:
        doppler_data = json.load(f)

    if orbit_interp is None:
        orbit_interp = choose_orbit_interp(orbit_data, acquisition_data)

    orbit = construct_orbit(orbit_data, orbit_interp)
    radar_grid = construct_radar_grid(
        radargrid_data,
        acquisition_data,
        orbit_data,
    )
    dem_raster = isce3.io.Raster(str(dem_path))
    out = isce3.io.Raster(
        str(rtc_factor_path), radar_grid.width, radar_grid.length, 1, 6, "GTiff"
    )
    doppler = (
        isce3.core.LUT2d()
        if zero_doppler
        else construct_doppler_lut2d(
            doppler_data,
            radargrid_json=radargrid_data,
            acquisition_json=acquisition_data,
            orbit_json=orbit_data,
        )
    )
    isce3.geometry.compute_rtc(radar_grid, orbit, doppler, dem_raster, out)
    dem_raster.close_dataset()
    out.close_dataset()
    return str(rtc_factor_path)


def write_rtc_hdf(
    slc_path: str,
    rtc_factor_path: str,
    output_h5: str,
    block_rows: int = 256,
) -> str:
    slc_ds = gdal.Open(slc_path)
    if slc_ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    rtc_ds = gdal.Open(rtc_factor_path)
    if rtc_ds is None:
        raise RuntimeError(f"failed to open RTC factor: {rtc_factor_path}")

    # Use RTC TIFF dimensions for HDF5 dataset and RTC reads
    # SLC dimensions are only used for validation
    rtc_width = rtc_ds.RasterXSize
    rtc_length = rtc_ds.RasterYSize
    slc_width = slc_ds.RasterXSize
    slc_length = slc_ds.RasterYSize
    slc_band_count = slc_ds.RasterCount

    # Validate SLC and RTC have matching dimensions
    if slc_width != rtc_width or slc_length != rtc_length:
        raise ValueError(
            f"SLC and RTC factor dimensions mismatch: "
            f"SLC=({slc_length}, {slc_width}), RTC=({rtc_length}, {rtc_width})"
        )

    output_h5 = str(Path(output_h5))

    with h5py.File(output_h5, "w") as f:
        f.attrs["product_type"] = "slc_amplitude_fullres"
        f.attrs["width"] = rtc_width
        f.attrs["length"] = rtc_length
        f.attrs["source_slc"] = slc_path
        f.attrs["rtc_factor_raster"] = rtc_factor_path
        f.attrs["source_dataset"] = "slc_amplitude"
        f.attrs["radiometry"] = "amplitude"
        f.attrs["value_domain"] = "linear"
        f.attrs["amplitude_definition"] = "sqrt(I^2 + Q^2)"

        d_amp = f.create_dataset(
            "slc_amplitude",
            shape=(rtc_length, rtc_width),
            dtype="f4",
            chunks=(min(block_rows, rtc_length), min(1024, rtc_width)),
            compression="gzip",
            shuffle=True,
        )
        d_amp.attrs["long_name"] = "slc_amplitude"
        d_amp.attrs["description"] = "Linear SLC amplitude computed as sqrt(I^2 + Q^2)"
        d_amp.attrs["units"] = "linear_amplitude"
        d_factor = f.create_dataset(
            "rtc_factor",
            shape=(rtc_length, rtc_width),
            dtype="f4",
            chunks=(min(block_rows, rtc_length), min(1024, rtc_width)),
            compression="gzip",
            shuffle=True,
        )
        d_factor.attrs["long_name"] = "rtc_factor"
        d_factor.attrs["description"] = (
            "RTC area normalization factor retained as auxiliary data"
        )
        d_factor.attrs["units"] = "unitless"

        slc_band_1 = slc_ds.GetRasterBand(1)
        slc_band_2 = slc_ds.GetRasterBand(2) if slc_band_count >= 2 else None
        rtc_band = rtc_ds.GetRasterBand(1)
        for row0 in range(0, rtc_length, block_rows):
            rows = min(block_rows, rtc_length - row0)
            slc_block_1 = _read_band_array(slc_band_1, 0, row0, rtc_width, rows)
            rtc_block = _read_band_array(rtc_band, 0, row0, rtc_width, rows).astype(
                np.float32
            )
            # Ensure rtc_block has correct shape (rows, rtc_width)
            if rtc_block.shape != (rows, rtc_width):
                rtc_block = rtc_block.T
            if slc_band_2 is not None:
                slc_block_2 = _read_band_array(slc_band_2, 0, row0, rtc_width, rows)
                if slc_block_1.shape != (rows, rtc_width):
                    slc_block_1 = slc_block_1.T
                if slc_block_2.shape != (rows, rtc_width):
                    slc_block_2 = slc_block_2.T
                amplitude = np.sqrt(
                    slc_block_1.astype(np.float32) ** 2
                    + slc_block_2.astype(np.float32) ** 2
                ).astype(np.float32)
            else:
                if slc_block_1.shape != (rows, rtc_width):
                    slc_block_1 = slc_block_1.T
                amplitude = np.sqrt(
                    slc_block_1.real.astype(np.float32) ** 2
                    + slc_block_1.imag.astype(np.float32) ** 2
                ).astype(np.float32)
            d_amp[row0 : row0 + rows, :] = amplitude
            d_factor[row0 : row0 + rows, :] = rtc_block
    return output_h5


def write_geocoded_tif_hdf(
    geocoded_tif_path: str,
    output_h5: str,
    rtc_factor_path: str | None = None,
    rtc_factor_value: float | None = None,
    block_rows: int = 256,
) -> str:
    rtc_ds = gdal.Open(geocoded_tif_path)
    if rtc_ds is None:
        raise RuntimeError(f"failed to open geocoded RTC TIFF: {geocoded_tif_path}")

    factor_ds = None
    if rtc_factor_path is not None:
        factor_ds = gdal.Open(rtc_factor_path)
        if factor_ds is None:
            raise RuntimeError(f"failed to open rtc factor raster: {rtc_factor_path}")

    width = rtc_ds.RasterXSize
    length = rtc_ds.RasterYSize
    output_h5 = str(Path(output_h5))

    with h5py.File(output_h5, "w") as f:
        f.attrs["product_type"] = "slc_amplitude_geocoded_fullres"
        f.attrs["width"] = width
        f.attrs["length"] = length
        f.attrs["source_geocoded_tif"] = geocoded_tif_path
        if rtc_factor_path is not None:
            f.attrs["rtc_factor_raster"] = rtc_factor_path
        if rtc_factor_value is not None:
            f.attrs["rtc_factor_constant_value"] = float(rtc_factor_value)
        f.attrs["source_dataset"] = "slc_amplitude"
        f.attrs["radiometry"] = "amplitude"
        f.attrs["value_domain"] = "linear"
        f.attrs["amplitude_definition"] = "sqrt(I^2 + Q^2)"

        d_amp = f.create_dataset(
            "slc_amplitude",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )
        d_amp.attrs["long_name"] = "slc_amplitude"
        d_amp.attrs["description"] = "Linear SLC amplitude on the geocoded grid"
        d_amp.attrs["units"] = "linear_amplitude"
        d_factor = f.create_dataset(
            "rtc_factor",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )
        d_factor.attrs["long_name"] = "rtc_factor"
        d_factor.attrs["description"] = (
            "RTC area normalization factor retained as auxiliary data"
        )
        d_factor.attrs["units"] = "unitless"

        rtc_band = rtc_ds.GetRasterBand(1)
        factor_band = factor_ds.GetRasterBand(1) if factor_ds is not None else None

        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            amplitude_block = _read_band_array(rtc_band, 0, row0, width, rows).astype(
                np.float32
            )
            d_amp[row0 : row0 + rows, :] = amplitude_block
            if factor_band is not None:
                factor_block = _read_band_array(factor_band, 0, row0, width, rows).astype(
                    np.float32
                )
            elif rtc_factor_value is not None:
                factor_block = np.full(
                    (rows, width), rtc_factor_value, dtype=np.float32
                )
            else:
                factor_block = np.full((rows, width), np.nan, dtype=np.float32)
            d_factor[row0 : row0 + rows, :] = factor_block

    return output_h5


def append_geogrid_coordinates_hdf(
    geocoded_tif_path: str,
    output_h5: str,
    dem_path: str | None = None,
    block_rows: int = 256,
) -> str:
    src_ds = gdal.Open(geocoded_tif_path)
    if src_ds is None:
        raise RuntimeError(f"failed to open geocoded TIFF: {geocoded_tif_path}")

    width = src_ds.RasterXSize
    length = src_ds.RasterYSize
    gt = src_ds.GetGeoTransform()
    proj = src_ds.GetProjection()

    dem_ds = None
    if dem_path is not None:
        tmp_dem = (
            Path(tempfile.mkdtemp(prefix="d2sar_dem_resample_", dir="/tmp"))
            / "height.tif"
        )
        dem_ds = gdal.Warp(
            str(tmp_dem),
            dem_path,
            format="GTiff",
            width=width,
            height=length,
            outputBounds=(gt[0], gt[3] + length * gt[5], gt[0] + width * gt[1], gt[3]),
            dstSRS=proj,
            resampleAlg="bilinear",
            dstNodata=np.nan,
        )
        if dem_ds is None:
            raise RuntimeError("failed to resample DEM onto geocoded grid")
        dem_band = dem_ds.GetRasterBand(1)
    else:
        dem_band = None

    x_idx = np.arange(width, dtype=np.float64)
    x_coords = gt[0] + x_idx * gt[1]
    try:
        with h5py.File(output_h5, "a") as f:
            for name in ("longitude", "latitude", "height"):
                if name in f:
                    del f[name]
            d_lon = f.create_dataset(
                "longitude",
                shape=(length, width),
                dtype="f4",
                chunks=(min(block_rows, length), min(1024, width)),
                compression="gzip",
                shuffle=True,
            )
            d_lat = f.create_dataset(
                "latitude",
                shape=(length, width),
                dtype="f4",
                chunks=(min(block_rows, length), min(1024, width)),
                compression="gzip",
                shuffle=True,
            )
            d_hgt = f.create_dataset(
                "height",
                shape=(length, width),
                dtype="f4",
                chunks=(min(block_rows, length), min(1024, width)),
                compression="gzip",
                shuffle=True,
            )

            f.attrs["coordinate_system"] = "EPSG:4326"
            f.attrs["longitude_units"] = "degrees_east"
            f.attrs["latitude_units"] = "degrees_north"
            f.attrs["height_units"] = "meters"
            f.attrs["coordinate_source"] = (
                "geocoded_grid_geotransform_with_validated_dem"
            )

            for row0 in range(0, length, block_rows):
                rows = min(block_rows, length - row0)
                y_idx = np.arange(row0, row0 + rows, dtype=np.float64)
                y_coords = gt[3] + y_idx * gt[5]
                lon_block = np.broadcast_to(x_coords[None, :], (rows, width)).astype(
                    np.float32
                )
                lat_block = np.broadcast_to(y_coords[:, None], (rows, width)).astype(
                    np.float32
                )
                if dem_band is not None:
                    hgt_block = _read_band_array(dem_band, 0, row0, width, rows).astype(
                        np.float32
                    )
                else:
                    hgt_block = np.full((rows, width), np.nan, dtype=np.float32)
                d_lon[row0 : row0 + rows, :] = lon_block
                d_lat[row0 : row0 + rows, :] = lat_block
                d_hgt[row0 : row0 + rows, :] = hgt_block
    finally:
        if dem_ds is not None:
            dem_ds = None
            shutil.rmtree(tmp_dem.parent, ignore_errors=True)

    return str(output_h5)


def append_corner_interpolated_coordinates_hdf(
    scene_json_path: str,
    dem_path: str,
    output_h5: str,
    shape: tuple[int, int],
    block_rows: int = 256,
) -> str:
    scene_json_path = Path(scene_json_path)
    output_h5 = Path(output_h5)
    with open(scene_json_path, encoding="utf-8") as f:
        scene = json.load(f)

    corners = scene["sceneCorners"]
    length, width = shape

    top = sorted(
        [c for c in corners if c["refRow"] == min(pt["refRow"] for pt in corners)],
        key=lambda c: c["refColumn"],
    )
    bottom = sorted(
        [c for c in corners if c["refRow"] == max(pt["refRow"] for pt in corners)],
        key=lambda c: c["refColumn"],
    )
    if len(top) != 2 or len(bottom) != 2:
        raise RuntimeError("Expected exactly 4 scene corners for Lutan interpolation")
    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]

    x_frac = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_frac = np.linspace(0.0, 1.0, length, dtype=np.float32)

    top_lon = tl["lon"] + (tr["lon"] - tl["lon"]) * x_frac
    bottom_lon = bl["lon"] + (br["lon"] - bl["lon"]) * x_frac
    top_lat = tl["lat"] + (tr["lat"] - tl["lat"]) * x_frac
    bottom_lat = bl["lat"] + (br["lat"] - bl["lat"]) * x_frac

    dem_ds = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
    if dem_ds is None:
        raise RuntimeError(
            f"failed to open DEM for corner interpolation sampling: {dem_path}"
        )
    dem_arr = _read_band_array(dem_ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
    dem_gt = dem_ds.GetGeoTransform()
    dem_nodata = dem_ds.GetRasterBand(1).GetNoDataValue()
    dem_width = dem_ds.RasterXSize
    dem_length = dem_ds.RasterYSize

    with h5py.File(output_h5, "a") as f:
        for name in ("longitude", "latitude", "height"):
            if name in f:
                del f[name]
        d_lon = f.create_dataset(
            "longitude",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )
        d_lat = f.create_dataset(
            "latitude",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )
        d_hgt = f.create_dataset(
            "height",
            shape=(length, width),
            dtype="f4",
            chunks=(min(block_rows, length), min(1024, width)),
            compression="gzip",
            shuffle=True,
        )

        f.attrs["coordinate_system"] = "EPSG:4326"
        f.attrs["longitude_units"] = "degrees_east"
        f.attrs["latitude_units"] = "degrees_north"
        f.attrs["height_units"] = "meters"
        f.attrs["coordinate_source"] = (
            "bilinear_interpolation_from_scene_corners_with_validated_dem"
        )

        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            yf = y_frac[row0 : row0 + rows][:, None]
            lon_block = top_lon[None, :] + (bottom_lon - top_lon)[None, :] * yf
            lat_block = top_lat[None, :] + (bottom_lat - top_lat)[None, :] * yf

            dem_x = np.rint((lon_block - dem_gt[0]) / dem_gt[1]).astype(np.int32)
            dem_y = np.rint((lat_block - dem_gt[3]) / dem_gt[5]).astype(np.int32)
            valid = (
                (dem_x >= 0) & (dem_x < dem_width) & (dem_y >= 0) & (dem_y < dem_length)
            )
            hgt_block = np.full((rows, width), np.nan, dtype=np.float32)
            hgt_block[valid] = dem_arr[dem_y[valid], dem_x[valid]]
            if dem_nodata is not None:
                hgt_block[np.isclose(hgt_block, dem_nodata)] = np.nan

            d_lon[row0 : row0 + rows, :] = lon_block.astype(np.float32)
            d_lat[row0 : row0 + rows, :] = lat_block.astype(np.float32)
            d_hgt[row0 : row0 + rows, :] = hgt_block

    dem_ds = None
    return str(output_h5)


def append_topo_coordinates_hdf(
    manifest_path: str,
    dem_path: str,
    output_h5: str,
    block_rows: int = 256,
    orbit_interp: str | None = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> str:
    import isce3.core
    import isce3.io

    manifest_path = Path(manifest_path)
    output_h5 = Path(output_h5)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "radargrid"),
        encoding="utf-8",
    ) as f:
        radargrid_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "orbit"),
        encoding="utf-8",
    ) as f:
        orbit_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "acquisition"),
        encoding="utf-8",
    ) as f:
        acquisition_data = json.load(f)

    width = radargrid_data["numberOfColumns"]
    length = radargrid_data["numberOfRows"]
    print(
        f"[DEBUG] append_topo_coordinates_hdf: radargrid w={width}, l={length}, dem={dem_path}"
    )

    if orbit_interp is None:
        orbit_interp = choose_orbit_interp(orbit_data, acquisition_data)

    orbit = construct_orbit(orbit_data, orbit_interp)
    radar_grid = construct_radar_grid(
        radargrid_data,
        acquisition_data,
        orbit_data,
    )
    dem_raster = isce3.io.Raster(str(dem_path))

    if use_gpu:
        import isce3.cuda.core
        import isce3.cuda.geometry

        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)
        rdr2geo_cls = isce3.cuda.geometry.Rdr2Geo
    else:
        import isce3.geometry

        rdr2geo_cls = isce3.geometry.Rdr2Geo

    topo = rdr2geo_cls(
        radar_grid,
        orbit,
        isce3.core.Ellipsoid(),
        isce3.core.LUT2d(),
        epsg_out=4326,
        compute_mask=True,
        lines_per_block=block_rows,
    )

    tmp_parent = output_h5.parent / ".topo_tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="d2sar_topo_gtiff_", dir=str(tmp_parent)))
    try:

        def make_raster(name: str, dtype: int = gdal.GDT_Float32):
            return isce3.io.Raster(
                str(workdir / f"{name}.tif"),
                radar_grid.width,
                radar_grid.length,
                1,
                dtype,
                "GTiff",
            )

        x_raster = make_raster("x", gdal.GDT_Float64)
        y_raster = make_raster("y", gdal.GDT_Float64)
        z_raster = make_raster("z", gdal.GDT_Float64)
        inc_raster = make_raster("inc")
        hdg_raster = make_raster("hdg")
        local_inc_raster = make_raster("localInc")
        local_psi_raster = make_raster("localPsi")
        simamp_raster = make_raster("simamp")
        layover_raster = make_raster("layoverShadowMask", gdal.GDT_Byte)
        los_e_raster = make_raster("los_east")
        los_n_raster = make_raster("los_north")

        topo.topo(
            dem_raster,
            x_raster,
            y_raster,
            z_raster,
            inc_raster,
            hdg_raster,
            local_inc_raster,
            local_psi_raster,
            simamp_raster,
            layover_raster,
            los_e_raster,
            los_n_raster,
        )

        for raster in [
            x_raster,
            y_raster,
            z_raster,
            inc_raster,
            hdg_raster,
            local_inc_raster,
            local_psi_raster,
            simamp_raster,
            layover_raster,
            los_e_raster,
            los_n_raster,
        ]:
            raster.close_dataset()

        lon_ds = gdal.Open(str(workdir / "x.tif"))
        lat_ds = gdal.Open(str(workdir / "y.tif"))
        hgt_ds = gdal.Open(str(workdir / "z.tif"))
        if lon_ds is None or lat_ds is None or hgt_ds is None:
            raise RuntimeError("failed to reopen topo GTiff rasters")

        with h5py.File(output_h5, "a") as f:
            for name in ("longitude", "latitude", "height"):
                if name in f:
                    del f[name]
            d_lon = f.create_dataset(
                "longitude",
                shape=(length, width),
                dtype="f4",
                chunks=(min(block_rows, length), min(1024, width)),
                compression="gzip",
                shuffle=True,
            )
            d_lat = f.create_dataset(
                "latitude",
                shape=(length, width),
                dtype="f4",
                chunks=(min(block_rows, length), min(1024, width)),
                compression="gzip",
                shuffle=True,
            )
            d_hgt = f.create_dataset(
                "height",
                shape=(length, width),
                dtype="f4",
                chunks=(min(block_rows, length), min(1024, width)),
                compression="gzip",
                shuffle=True,
            )
            f.attrs["coordinate_system"] = "EPSG:4326"
            f.attrs["longitude_units"] = "degrees_east"
            f.attrs["latitude_units"] = "degrees_north"
            f.attrs["height_units"] = "meters"
            f.attrs["coordinate_source"] = "rdr2geo_topo_with_validated_dem"

            lon_band = lon_ds.GetRasterBand(1)
            lat_band = lat_ds.GetRasterBand(1)
            hgt_band = hgt_ds.GetRasterBand(1)
            for row0 in range(0, length, block_rows):
                rows = min(block_rows, length - row0)
                d_lon[row0 : row0 + rows, :] = _read_band_array(
                    lon_band, 0, row0, width, rows
                ).astype(np.float32)
                d_lat[row0 : row0 + rows, :] = _read_band_array(
                    lat_band, 0, row0, width, rows
                ).astype(np.float32)
                d_hgt[row0 : row0 + rows, :] = _read_band_array(
                    hgt_band, 0, row0, width, rows
                ).astype(np.float32)
        return str(output_h5)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if tmp_parent.exists() and not any(tmp_parent.iterdir()):
            tmp_parent.rmdir()


def point2epsg(lon: float, lat: float) -> int:
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 75.0:
        return 3413
    if lat <= -75.0:
        return 3031
    if lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    if lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(f"Could not determine projection for {lat},{lon}")


def append_utm_coordinates_hdf(
    output_h5: str, manifest_path: str, block_rows: int = 32
) -> str:
    output_h5 = Path(output_h5)
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "scene"),
        encoding="utf-8",
    ) as f:
        scene = json.load(f)

    corners = scene["sceneCorners"]
    center_lon = sum(pt["lon"] for pt in corners) / len(corners)
    center_lat = sum(pt["lat"] for pt in corners) / len(corners)
    epsg = point2epsg(center_lon, center_lat)

    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)

    with h5py.File(output_h5, "a") as f:
        lon_ds = f["longitude"]
        lat_ds = f["latitude"]
        length, width = lon_ds.shape
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
        f.attrs["utm_coordinate_source"] = "transformed_from_topo_driven_lonlat"
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            lon_block = lon_ds[row0 : row0 + rows, :]
            lat_block = lat_ds[row0 : row0 + rows, :]
            x_block = np.full((rows, width), np.nan, dtype=np.float32)
            y_block = np.full((rows, width), np.nan, dtype=np.float32)
            valid = (
                np.isfinite(lon_block)
                & np.isfinite(lat_block)
                & (lon_block >= -180.0)
                & (lon_block <= 180.0)
                & (lat_block >= -90.0)
                & (lat_block <= 90.0)
            )
            if np.any(valid):
                pts = np.column_stack([lon_block[valid], lat_block[valid]])
                transformed = np.asarray(
                    transform.TransformPoints(pts[:, :2]), dtype=np.float64
                )
                x_block[valid] = transformed[:, 0].astype(np.float32)
                y_block[valid] = transformed[:, 1].astype(np.float32)
            utm_x[row0 : row0 + rows, :] = x_block
            utm_y[row0 : row0 + rows, :] = y_block
    return str(output_h5)


def compute_utm_output_shape(
    input_h5: str, resolution_meters: float, block_rows: int = 64
) -> tuple[int, int]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        length, width = x_ds.shape
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
    target_width = max(1, int(round((x_max - x_min) / resolution_meters)))
    target_height = max(1, int(round((y_max - y_min) / resolution_meters)))
    return target_width, target_height


def accumulate_utm_grid(
    input_h5: str,
    dataset_name: str = "slc_amplitude",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
) -> tuple[np.ndarray, dict]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        amp_ds = f[dataset_name]
        if np.issubdtype(amp_ds.dtype, np.complexfloating):
            raise TypeError(
                f"Dataset '{dataset_name}' is complex; use wrapped-phase export helpers"
            )
        length, width = amp_ds.shape
        utm_epsg = int(f.attrs["utm_epsg"])
        positive_only = dataset_name in {"slc_amplitude", "avg_amplitude", "coherence"}
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
        aspect = (y_max - y_min) / max(x_max - x_min, 1e-9)
        if target_height is None:
            target_height = max(1, int(round(target_width * aspect)))
        else:
            target_height = max(1, target_height)
        sums = np.zeros((target_height, target_width), dtype=np.float64)
        counts = np.zeros((target_height, target_width), dtype=np.uint32)
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            amplitude = amp_ds[row0 : row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(amplitude)
            if positive_only:
                valid &= amplitude > 0
            if not np.any(valid):
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            amplitude_valid = amplitude[valid]
            col = np.clip(
                (
                    (x_valid - x_min) / max(x_max - x_min, 1e-9) * (target_width - 1)
                ).astype(np.int32),
                0,
                target_width - 1,
            )
            row = np.clip(
                (
                    (y_max - y_valid) / max(y_max - y_min, 1e-9) * (target_height - 1)
                ).astype(np.int32),
                0,
                target_height - 1,
            )
            flat_idx = row * target_width + col
            np.add.at(sums.ravel(), flat_idx, amplitude_valid.astype(np.float64))
            np.add.at(counts.ravel(), flat_idx, 1)

    out = np.full((target_height, target_width), np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    meta = {
        "utm_epsg": utm_epsg,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "target_width": target_width,
        "target_height": target_height,
        "x_res": (x_max - x_min) / max(target_width - 1, 1),
        "y_res": (y_max - y_min) / max(target_height - 1, 1),
    }
    return out, meta


def prepare_display_grid(
    out: np.ndarray, confidence_interval_pct: float | None = 98.0
) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(out) & (out > 0)
    cleaned = np.zeros(out.shape, dtype=np.float64)
    info = {
        "confidence_interval_pct": confidence_interval_pct,
        "clip_db_lower": None,
        "clip_db_upper": None,
        "clip_linear_lower": None,
        "clip_linear_upper": None,
        "nan_fill_value": 0.0,
    }
    if not np.any(valid) or confidence_interval_pct is None:
        cleaned[valid] = out[valid]
        return cleaned, info
    tail_pct = max(0.0, (100.0 - confidence_interval_pct) / 2.0)
    lower_pct = tail_pct
    upper_pct = 100.0 - tail_pct
    vals_db = 10.0 * np.log10(out[valid])
    lower_db = float(np.percentile(vals_db, lower_pct))
    upper_db = float(np.percentile(vals_db, upper_pct))
    lower_linear = float(10 ** (lower_db / 10.0))
    upper_linear = float(10 ** (upper_db / 10.0))
    cleaned[valid] = np.clip(out[valid], lower_linear, upper_linear)
    info["clip_db_lower"] = lower_db
    info["clip_db_upper"] = upper_db
    info["clip_linear_lower"] = lower_linear
    info["clip_linear_upper"] = upper_linear
    return cleaned, info


def _stretch_source_grid_for_png(
    input_h5: str,
    dataset_name: str,
    stretch_percent: float = 5.0,
) -> np.ndarray:
    with h5py.File(input_h5, "r") as f:
        data = f[dataset_name][:]
    data = np.asarray(data, dtype=np.float32)
    display = np.full(data.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(data)
    if dataset_name in {"slc_amplitude", "avg_amplitude", "coherence"}:
        valid &= data > 0
    if not np.any(valid):
        return display
    lower_pct = float(np.clip(stretch_percent, 0.0, 50.0))
    upper_pct = 100.0 - lower_pct
    vals = data[valid].astype(np.float64)
    lower = float(np.percentile(vals, lower_pct))
    upper = float(np.percentile(vals, upper_pct))
    if upper <= lower:
        display[valid] = 255.0
        return display
    scaled = np.clip((data[valid] - lower) / (upper - lower), 0.0, 1.0)
    display[valid] = (scaled * 255.0).astype(np.float32)
    return display


def _accumulate_source_grid_to_utm(
    input_h5: str,
    source_grid: np.ndarray,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> tuple[np.ndarray, dict]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        length, width = source_grid.shape
        if x_ds.shape != source_grid.shape or y_ds.shape != source_grid.shape:
            raise RuntimeError(
                f"source grid shape {source_grid.shape} does not match UTM coordinate grids"
            )
        utm_epsg = int(f.attrs["utm_epsg"])
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
        aspect = (y_max - y_min) / max(x_max - x_min, 1e-9)
        if target_height is None:
            target_height = max(1, int(round(target_width * aspect)))
        else:
            target_height = max(1, target_height)
        sums = np.zeros((target_height, target_width), dtype=np.float64)
        counts = np.zeros((target_height, target_width), dtype=np.uint32)
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            data = source_grid[row0 : row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(data)
            if not np.any(valid):
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            vals = data[valid].astype(np.float64)
            col = np.clip(
                (
                    (x_valid - x_min)
                    / max(x_max - x_min, 1e-9)
                    * (target_width - 1)
                ).astype(np.int32),
                0,
                target_width - 1,
            )
            row = np.clip(
                (
                    (y_max - y_valid)
                    / max(y_max - y_min, 1e-9)
                    * (target_height - 1)
                ).astype(np.int32),
                0,
                target_height - 1,
            )
            flat_idx = row * target_width + col
            np.add.at(sums.ravel(), flat_idx, vals)
            np.add.at(counts.ravel(), flat_idx, 1)

    out = np.full((target_height, target_width), np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    meta = {
        "utm_epsg": utm_epsg,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "target_width": target_width,
        "target_height": target_height,
        "x_res": (x_max - x_min) / max(target_width - 1, 1),
        "y_res": (y_max - y_min) / max(target_height - 1, 1),
    }
    return out, meta


def write_geocoded_geotiff(
    input_h5: str,
    output_tif: str,
    dataset_name: str = "slc_amplitude",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
    apply_clip: bool = False,
    confidence_interval_pct: float | None = 98.0,
) -> str:
    output_tif = Path(output_tif)
    out, meta = accumulate_utm_grid(
        input_h5,
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=block_rows,
    )

    clip_info = {
        "confidence_interval_pct": None,
        "nan_fill_value": 0.0,
        "clip_db_lower": None,
        "clip_db_upper": None,
        "clip_linear_lower": None,
        "clip_linear_upper": None,
    }

    if apply_clip and confidence_interval_pct is not None:
        out, clip_info = prepare_display_grid(
            out, confidence_interval_pct=confidence_interval_pct
        )
    else:
        # 使用原始数据，仅填充NaN为0
        out = np.nan_to_num(out, nan=0.0)
        clip_info["confidence_interval_pct"] = None

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(output_tif),
        meta["target_width"],
        meta["target_height"],
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(
        [meta["x_min"], meta["x_res"], 0.0, meta["y_max"], 0.0, -meta["y_res"]]
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(meta["utm_epsg"])
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadataItem(
        "CONFIDENCE_INTERVAL_PCT", str(clip_info["confidence_interval_pct"])
    )
    ds.SetMetadataItem("SOURCE_DATASET", dataset_name)
    if dataset_name in {"slc_amplitude", "avg_amplitude"}:
        ds.SetMetadataItem("RADIOMETRY", "amplitude")
        ds.SetMetadataItem("VALUE_DOMAIN", "linear_amplitude")
        ds.SetMetadataItem("AMPLITUDE_DEFINITION", "sqrt(I^2 + Q^2)")
    elif dataset_name == "coherence":
        ds.SetMetadataItem("RADIOMETRY", "coherence")
        ds.SetMetadataItem("VALUE_DOMAIN", "unit_interval")
    elif dataset_name == "unwrapped_phase":
        ds.SetMetadataItem("RADIOMETRY", "phase")
        ds.SetMetadataItem("VALUE_DOMAIN", "radians")
    elif dataset_name == "los_displacement":
        ds.SetMetadataItem("RADIOMETRY", "displacement")
        ds.SetMetadataItem("VALUE_DOMAIN", "meters")
    ds.SetMetadataItem("NAN_FILL_VALUE", str(clip_info["nan_fill_value"]))
    if clip_info["clip_db_lower"] is not None:
        ds.SetMetadataItem("CLIP_DB_LOWER", str(clip_info["clip_db_lower"]))
        ds.SetMetadataItem("CLIP_DB_UPPER", str(clip_info["clip_db_upper"]))
        ds.SetMetadataItem("CLIP_LINEAR_LOWER", str(clip_info["clip_linear_lower"]))
        ds.SetMetadataItem("CLIP_LINEAR_UPPER", str(clip_info["clip_linear_upper"]))
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0.0)
    _write_band_array(band, out.astype(np.float32))
    band.FlushCache()
    ds.FlushCache()
    band = None
    ds = None
    return str(output_tif)


def write_geocoded_png(
    input_h5: str,
    output_png: str,
    dataset_name: str = "slc_amplitude",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
    confidence_interval_pct: float | None = 98.0,
) -> str:
    output_png = Path(output_png)
    if dataset_name == "avg_amplitude":
        source_display = _stretch_source_grid_for_png(
            input_h5, dataset_name, stretch_percent=5.0
        )
        out, _ = _accumulate_source_grid_to_utm(
            input_h5,
            source_display,
            target_width=target_width,
            target_height=target_height,
            block_rows=block_rows,
        )
        img = np.zeros(out.shape, dtype=np.uint8)
        valid = np.isfinite(out)
        if np.any(valid):
            img[valid] = np.rint(np.clip(out[valid], 0.0, 255.0)).astype(np.uint8)
        Image.fromarray(img, mode="L").save(output_png)
        return str(output_png)

    out, _ = accumulate_utm_grid(
        input_h5,
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=block_rows,
    )
    img = np.zeros(out.shape, dtype=np.uint8)
    if dataset_name in {"slc_amplitude", "avg_amplitude", "coherence"}:
        out, _ = prepare_display_grid(
            out, confidence_interval_pct=confidence_interval_pct
        )
        valid = np.isfinite(out) & (out > 0)
    else:
        valid = np.isfinite(out)
    if np.any(valid):
        if dataset_name in {"slc_amplitude", "avg_amplitude", "coherence"}:
            vals = 10.0 * np.log10(out[valid])
        else:
            vals = out[valid]
        p2 = np.percentile(vals, 2)
        p98 = np.percentile(vals, 98)
        scaled = np.clip((vals - p2) / (p98 - p2 + 1e-9), 0, 1)
        img[valid] = (scaled * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(output_png)
    return str(output_png)


def _accumulate_wrapped_phase_grid(
    input_h5: str,
    dataset_name: str = "interferogram",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
) -> tuple[np.ndarray, dict]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        ifg_ds = f[dataset_name]
        length, width = ifg_ds.shape
        utm_epsg = int(f.attrs["utm_epsg"])
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
        aspect = (y_max - y_min) / max(x_max - x_min, 1e-9)
        if target_height is None:
            target_height = max(1, int(round(target_width * aspect)))
        else:
            target_height = max(1, target_height)

        strongest = np.zeros((target_height, target_width), dtype=np.complex64)
        strongest_amp = np.zeros((target_height, target_width), dtype=np.float32)
        counts = np.zeros((target_height, target_width), dtype=np.uint32)
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0 : row0 + rows, :]
            y = y_ds[row0 : row0 + rows, :]
            ifg = ifg_ds[row0 : row0 + rows, :]
            valid = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(ifg.real)
                & np.isfinite(ifg.imag)
            )
            if not np.any(valid):
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            ifg_valid = ifg[valid]
            col = np.clip(
                (
                    (x_valid - x_min) / max(x_max - x_min, 1e-9) * (target_width - 1)
                ).astype(np.int32),
                0,
                target_width - 1,
            )
            row = np.clip(
                (
                    (y_max - y_valid) / max(y_max - y_min, 1e-9) * (target_height - 1)
                ).astype(np.int32),
                0,
                target_height - 1,
            )
            flat_idx = row * target_width + col
            amps = np.abs(ifg_valid).astype(np.float32)
            strongest_flat = strongest.ravel()
            strongest_amp_flat = strongest_amp.ravel()
            for idx, amp, val in zip(flat_idx, amps, ifg_valid, strict=False):
                if amp > strongest_amp_flat[idx]:
                    strongest_amp_flat[idx] = amp
                    strongest_flat[idx] = np.complex64(val)
            np.add.at(counts.ravel(), flat_idx, 1)

    out = np.full((target_height, target_width), np.nan, dtype=np.float32)
    mask = counts > 0
    out[mask] = np.angle(strongest[mask]).astype(np.float32)
    meta = {
        "utm_epsg": utm_epsg,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "target_width": target_width,
        "target_height": target_height,
        "x_res": (x_max - x_min) / max(target_width - 1, 1),
        "y_res": (y_max - y_min) / max(target_height - 1, 1),
    }
    return out, meta


def write_wrapped_phase_geotiff(
    input_h5: str,
    output_tif: str,
    dataset_name: str = "interferogram",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    output_tif = Path(output_tif)
    out, meta = _accumulate_wrapped_phase_grid(
        input_h5,
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=block_rows,
    )
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(output_tif),
        meta["target_width"],
        meta["target_height"],
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(
        [meta["x_min"], meta["x_res"], 0.0, meta["y_max"], 0.0, -meta["y_res"]]
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(meta["utm_epsg"])
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadataItem("SOURCE_DATASET", dataset_name)
    ds.SetMetadataItem("RADIOMETRY", "wrapped_phase")
    ds.SetMetadataItem("VALUE_DOMAIN", "radians")
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    _write_band_array(band, out.astype(np.float32))
    band.FlushCache()
    ds.FlushCache()
    band = None
    ds = None
    return str(output_tif)


def write_wrapped_phase_png(
    input_h5: str,
    output_png: str,
    dataset_name: str = "interferogram",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    output_png = Path(output_png)
    phase, _ = _accumulate_wrapped_phase_grid(
        input_h5,
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=block_rows,
    )
    rgb = np.zeros((*phase.shape, 3), dtype=np.uint8)
    value = np.ones(phase.shape, dtype=np.float64)
    try:
        avg_amplitude, _ = accumulate_utm_grid(
            input_h5,
            dataset_name="avg_amplitude",
            target_width=phase.shape[1],
            target_height=phase.shape[0],
            block_rows=block_rows,
        )
        avg_amplitude, _ = prepare_display_grid(
            avg_amplitude,
            confidence_interval_pct=98.0,
        )
        amp_valid = np.isfinite(avg_amplitude) & (avg_amplitude > 0)
        if np.any(amp_valid):
            amp_db = 10.0 * np.log10(avg_amplitude[amp_valid])
            p2 = float(np.percentile(amp_db, 2))
            p98 = float(np.percentile(amp_db, 98))
            scaled = np.clip((amp_db - p2) / (p98 - p2 + 1.0e-9), 0.0, 1.0)
            value[:] = 0.15
            value[amp_valid] = np.clip(scaled, 0.15, 1.0)
    except Exception:
        pass
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = ((phase[valid] + np.pi) / (2.0 * np.pi)).astype(np.float64)
        colors = np.array(
            [
                colorsys.hsv_to_rgb(float(h), 1.0, float(v))
                for h, v in zip(hue, value[valid], strict=False)
            ]
        )
        rgb[valid] = (colors * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_png)
    return str(output_png)


def write_unwrapped_phase_png(
    input_h5: str,
    output_png: str,
    dataset_name: str = "unwrapped_phase",
    target_width: int = 2048,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    output_png = Path(output_png)
    phase, _ = accumulate_utm_grid(
        input_h5,
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=block_rows,
    )
    phase = np.mod(np.asarray(phase, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi
    rgb = np.zeros((*phase.shape, 3), dtype=np.uint8)
    value = np.ones(phase.shape, dtype=np.float64)
    try:
        avg_amplitude, _ = accumulate_utm_grid(
            input_h5,
            dataset_name="avg_amplitude",
            target_width=phase.shape[1],
            target_height=phase.shape[0],
            block_rows=block_rows,
        )
        avg_amplitude, _ = prepare_display_grid(
            avg_amplitude,
            confidence_interval_pct=98.0,
        )
        amp_valid = np.isfinite(avg_amplitude) & (avg_amplitude > 0)
        if np.any(amp_valid):
            amp_db = 10.0 * np.log10(avg_amplitude[amp_valid])
            p2 = float(np.percentile(amp_db, 2))
            p98 = float(np.percentile(amp_db, 98))
            scaled = np.clip((amp_db - p2) / (p98 - p2 + 1.0e-9), 0.0, 1.0)
            value[:] = 0.15
            value[amp_valid] = np.clip(scaled, 0.15, 1.0)
    except Exception:
        pass
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = ((phase[valid] + np.pi) / (2.0 * np.pi)).astype(np.float64)
        colors = np.array(
            [
                colorsys.hsv_to_rgb(float(h), 1.0, float(v))
                for h, v in zip(hue, value[valid], strict=False)
            ]
        )
        rgb[valid] = (colors * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_png)
    return str(output_png)


def llh_to_xyz(lon_deg: float, lat_deg: float, h_m: float = 0.0) -> np.ndarray:
    """Convert WGS84 longitude/latitude/height to ECEF XYZ."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + h_m) * np.cos(lat) * np.cos(lon),
            (n + h_m) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + h_m) * np.sin(lat),
        ],
        dtype=np.float64,
    )


def verify_look_direction_from_corner(
    orbit_data: dict,
    corner: dict,
    dem_height: float = 0.0,
) -> tuple[str, float]:
    import isce3.core

    orbit = construct_orbit(orbit_data, "Hermite")

    corner_lon = corner["lon"]
    corner_lat = corner["lat"]
    corner_h = dem_height

    target_xyz = llh_to_xyz(corner_lon, corner_lat, corner_h)

    if "azimuthTimeUTC" in corner:
        az_time_str = corner["azimuthTimeUTC"]
    elif "timeUTC" in corner:
        az_time_str = corner["timeUTC"]
    else:
        raise ValueError("Corner must have azimuthTimeUTC or timeUTC field")

    ref_dt = gps_to_datetime(orbit_data["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (ref_dt - gps_epoch).total_seconds()
    az_time_dt = gps_to_datetime(az_time_str)
    az_time_gps = (az_time_dt - gps_epoch).total_seconds()
    az_time_rel = az_time_gps - orbit_ref_gps

    pos, vel = orbit.interpolate(az_time_rel)

    range_vec = target_xyz - pos
    slant_range = np.linalg.norm(range_vec)
    range_unit = range_vec / slant_range

    cross = np.cross(range_unit, vel)
    cross_z = cross[2]

    if cross_z < 0:
        look_direction = "RIGHT"
    else:
        look_direction = "LEFT"

    return look_direction, float(cross_z)


def verify_look_direction_from_center(
    orbit_data: dict,
    scene_corners: list,
    dem_height: float = 0.0,
) -> tuple[str, float]:
    import isce3.core

    if len(scene_corners) < 4:
        raise ValueError("Need at least 4 corners to compute center")

    # 计算四个角点的平均坐标
    lons = []
    lats = []
    for corner in scene_corners:
        lons.append(corner["lon"])
        lats.append(corner["lat"])
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)

    # 计算中心点的时间（使用中间时刻）
    times = []
    for corner in scene_corners:
        if "azimuthTimeUTC" in corner:
            times.append(gps_to_datetime(corner["azimuthTimeUTC"]))
        elif "timeUTC" in corner:
            times.append(gps_to_datetime(corner["timeUTC"]))
    if not times:
        raise ValueError("No time information found in corners")
    center_time = min(times) + (max(times) - min(times)) / 2

    orbit = construct_orbit(orbit_data, "Hermite")
    target_xyz = llh_to_xyz(center_lon, center_lat, dem_height)

    ref_dt = gps_to_datetime(orbit_data["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (ref_dt - gps_epoch).total_seconds()
    center_time_gps = (center_time - gps_epoch).total_seconds()
    center_time_rel = center_time_gps - orbit_ref_gps

    pos, vel = orbit.interpolate(center_time_rel)

    range_vec = target_xyz - pos
    slant_range = np.linalg.norm(range_vec)
    range_unit = range_vec / slant_range

    cross = np.cross(range_unit, vel)
    cross_z = cross[2]

    if cross_z < 0:
        look_direction = "RIGHT"
    else:
        look_direction = "LEFT"

    return look_direction, float(cross_z)


def get_local_look_vector(
    pos: np.ndarray, target_xyz: np.ndarray, lon: float, lat: float
) -> np.ndarray:
    """计算本地坐标系下的观测向量

    Args:
        pos: 卫星位置 (ECEF)
        target_xyz: 目标点位置 (ECEF)
        lon: 目标点经度 (度)
        lat: 目标点纬度 (度)

    Returns:
        本地坐标系下的观测向量 [东, 北, 天顶]
    """
    # 计算观测向量（从目标指向卫星）
    look_vec = pos - target_xyz
    look_vec_unit = look_vec / np.linalg.norm(look_vec)

    # 构建旋转矩阵：全局 ECEF → 本地（东-北-天顶）
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # 旋转矩阵
    R = np.array(
        [
            [-np.sin(lon_rad), np.cos(lon_rad), 0],
            [
                -np.sin(lat_rad) * np.cos(lon_rad),
                -np.sin(lat_rad) * np.sin(lon_rad),
                np.cos(lat_rad),
            ],
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ],
        ]
    )

    # 转换到本地坐标系
    local_look = R @ look_vec_unit
    return local_look


def verify_look_direction_gmtsar_style(
    orbit_data: dict,
    scene_corners: list,
    dem_height: float = 0.0,
) -> tuple[str, float]:
    """GMTSAR 风格的侧视方向判断

    基于本地坐标系下的观测向量东向分量判断

    Args:
        orbit_data: 轨道数据
        scene_corners: 场景角点列表
        dem_height: 高程 (米)

    Returns:
        (look_direction, east_component)
    """
    import isce3.core

    if len(scene_corners) < 4:
        # 没有足够的角点，返回默认值
        print(
            "[LOOK_DIRECTION] Warning: Insufficient corners for GMTSAR-style look direction calculation. Using default RIGHT."
        )
        return "RIGHT", 0.0

    # 计算四个角点的平均坐标
    lons = []
    lats = []
    for corner in scene_corners:
        lons.append(corner["lon"])
        lats.append(corner["lat"])
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)

    # 计算中心点的时间（使用中间时刻）
    times = []
    for corner in scene_corners:
        if "azimuthTimeUTC" in corner:
            times.append(gps_to_datetime(corner["azimuthTimeUTC"]))
        elif "timeUTC" in corner:
            times.append(gps_to_datetime(corner["timeUTC"]))
    if not times:
        # 没有时间信息，返回默认值
        print(
            "[LOOK_DIRECTION] Warning: No time information in corners. Using default RIGHT."
        )
        return "RIGHT", 0.0
    center_time = min(times) + (max(times) - min(times)) / 2

    orbit = construct_orbit(orbit_data, "Hermite")
    target_xyz = llh_to_xyz(center_lon, center_lat, dem_height)

    ref_dt = gps_to_datetime(orbit_data["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (ref_dt - gps_epoch).total_seconds()
    center_time_gps = (center_time - gps_epoch).total_seconds()
    center_time_rel = center_time_gps - orbit_ref_gps

    pos, _ = orbit.interpolate(center_time_rel)

    # 计算本地观测向量
    local_look = get_local_look_vector(pos, target_xyz, center_lon, center_lat)
    east_component = local_look[0]

    # 判断侧视方向
    if east_component > 0:
        look_direction = "RIGHT"
    else:
        look_direction = "LEFT"

    return look_direction, float(east_component)


def verify_and_correct_look_direction(
    orbit_data: dict,
    scene_corners: list,
    xml_look_direction: str,
    dem_height: float = 0.0,
) -> tuple[str, bool]:
    if not scene_corners:
        print(
            "[LOOK_DIRECTION] Warning: No scene corners provided. Using XML look direction."
        )
        return xml_look_direction, False

    # 使用 GMTSAR 风格的方法计算侧视方向
    computed_direction, east_component = verify_look_direction_gmtsar_style(
        orbit_data, scene_corners, dem_height
    )

    xml_direction = xml_look_direction.strip().upper()
    is_correct = computed_direction == xml_direction

    if not is_correct:
        print(
            f"[LOOK_DIRECTION] Warning: XML says '{xml_direction}' "
            f"but geometric calculation gives '{computed_direction}' "
            f"(east_component={east_component:.3f}). Using computed direction."
        )
        return computed_direction, True

    return xml_direction, False
