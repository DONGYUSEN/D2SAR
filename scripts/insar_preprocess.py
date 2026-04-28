from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from common_processing import resolve_manifest_metadata_path, resolve_manifest_data_path


def _normalize_vsi_source_path(path: str | Path) -> str:
    path_str = str(path)
    for prefix, markers in (
        ("/vsitar/", (".tar.gz/", ".tgz/", ".tar/")),
        ("/vsizip/", (".zip/",)),
    ):
        if not path_str.startswith(prefix):
            continue
        remainder = path_str[len(prefix) :]
        lower = remainder.lower()
        for marker in markers:
            idx = lower.find(marker)
            if idx < 0:
                continue
            archive_end = idx + len(marker) - 1
            archive = remainder[:archive_end]
            member = remainder[archive_end + 1 :]
            if archive.startswith("/"):
                return path_str
            return f"{prefix}/{archive}/{member.lstrip('/')}"
    return path_str


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


def _doppler_reference_point(doppler: dict, radargrid: dict) -> float:
    combined = doppler.get("combinedDoppler", {}) if isinstance(doppler, dict) else {}
    try:
        return float(combined.get("referencePoint", radargrid.get("rangeTimeFirstPixel", 0.0)))
    except Exception:
        return float(radargrid.get("rangeTimeFirstPixel", 0.0) or 0.0)


def _evaluate_doppler_hz(doppler: dict, radargrid: dict, width: int) -> np.ndarray:
    coeffs = _extract_doppler_coefficients(doppler)
    width = max(int(width), 1)
    sample_count = min(max(width, 2), 8192)
    cols = np.linspace(0.0, float(width - 1), num=sample_count, dtype=np.float64)
    range_time0 = float(radargrid.get("rangeTimeFirstPixel", 0.0) or 0.0)
    spacing = float(radargrid.get("columnSpacing", 1.0) or 1.0)
    range_times = range_time0 + (2.0 * spacing * cols / 299792458.0)
    ref = _doppler_reference_point(doppler, radargrid)
    values = np.zeros_like(range_times)
    for order, coeff in enumerate(coeffs):
        values += float(coeff) * np.power(range_times - ref, order)
    return values


def _infer_azimuth_bandwidth_hz(acquisition: dict | None, prf: float) -> tuple[float, str]:
    acquisition = acquisition or {}
    for key in (
        "totalProcessedAzimuthBandwidth",
        "azimuthLookBandwidth",
        "processedAzimuthBandwidth",
    ):
        try:
            value = float(acquisition.get(key, 0.0))
        except Exception:
            value = 0.0
        if value > 0.0:
            return value, f"acquisition:{key}"
    return max(float(prf) * 0.55, 1.0), "fallback:0.55*prf"


def _build_dc_policy(
    *,
    master_acquisition: dict | None,
    master_radargrid: dict | None,
    master_doppler: dict | None,
    slave_acquisition: dict,
    slave_radargrid: dict,
    slave_doppler: dict,
    prf_affects: bool,
) -> dict:
    ref_prf = max(
        float((master_acquisition or {}).get("prf", slave_acquisition.get("prf", 1.0)) or 1.0),
        1.0e-6,
    )
    sec_prf = max(float(slave_acquisition.get("prf", ref_prf) or ref_prf), 1.0e-6)
    width = int(
        min(
            int((master_radargrid or {}).get("numberOfColumns", slave_radargrid.get("numberOfColumns", 1)) or 1),
            int(slave_radargrid.get("numberOfColumns", 1) or 1),
        )
    )
    if not master_doppler:
        return {
            "available": False,
            "regime": "unavailable",
            "actions": [],
            "reason": "master_doppler_unavailable",
        }

    ref_hz = _evaluate_doppler_hz(master_doppler, master_radargrid or slave_radargrid, width)
    sec_hz = _evaluate_doppler_hz(slave_doppler, slave_radargrid, width)
    max_abs_delta_hz = float(np.max(np.abs(sec_hz - ref_hz))) if ref_hz.size else 0.0
    max_df_over_prf = float(max_abs_delta_hz / ref_prf)
    ref_bw, ref_bw_source = _infer_azimuth_bandwidth_hz(master_acquisition, ref_prf)
    sec_bw, sec_bw_source = _infer_azimuth_bandwidth_hz(slave_acquisition, sec_prf)
    common_baz_hz = max(min(ref_bw, sec_bw), 1.0e-6)
    overlap = max(0.0, min(1.0, 1.0 - max_abs_delta_hz / common_baz_hz))

    small_threshold = max(float(os.environ.get("D2SAR_DC_SMALL_NORM_THRESHOLD", 0.02)), 0.0)
    medium_threshold = max(float(os.environ.get("D2SAR_DC_MEDIUM_NORM_THRESHOLD", 0.10)), small_threshold)
    commonband_overlap = min(max(float(os.environ.get("D2SAR_DC_OVERLAP_FORCE_COMMONBAND", 0.70)), 0.0), 1.0)

    if max_df_over_prf < small_threshold and overlap >= commonband_overlap:
        regime = "small"
    elif max_df_over_prf < medium_threshold and overlap >= commonband_overlap:
        regime = "medium"
    else:
        regime = "large"

    actions: list[str] = []
    if regime in {"medium", "large"}:
        actions.append("dc-deramp-reramp")
    if regime == "large":
        actions.append("dc-commonband")
    if max_abs_delta_hz > 0.0 or prf_affects or regime in {"medium", "large"}:
        actions.append("harmonize-doppler-to-master")

    return {
        "available": True,
        "regime": regime,
        "actions": actions,
        "max_abs_delta_hz": max_abs_delta_hz,
        "max_abs_delta_over_prf": max_df_over_prf,
        "ref_azimuth_bandwidth_hz": ref_bw,
        "ref_azimuth_bandwidth_source": ref_bw_source,
        "slave_azimuth_bandwidth_hz": sec_bw,
        "slave_azimuth_bandwidth_source": sec_bw_source,
        "common_azimuth_bandwidth_hz": common_baz_hz,
        "overlap": overlap,
        "thresholds": {
            "small": small_threshold,
            "medium": medium_threshold,
            "commonband_overlap": commonband_overlap,
        },
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _requires_slc_materialization(actions: list[str]) -> bool:
    return any(
        action
        in {
            "normalize-slave-prf",
            "resample-slave-to-master-grid",
            "dc-deramp-reramp",
            "dc-commonband",
        }
        for action in actions
    )


def _requires_sampling_materialization(actions: list[str]) -> bool:
    return any(
        action in {"normalize-slave-prf", "resample-slave-to-master-grid"}
        for action in actions
    )


def _requires_dc_materialization(actions: list[str]) -> bool:
    return any(action in {"dc-deramp-reramp", "dc-commonband"} for action in actions)


def _is_gdal_virtual_path(path: str | Path | None) -> bool:
    if path is None:
        return False
    return str(path).startswith("/vsi")


def _copy_raster_with_gdal(src_path: str | Path, dst_path: str | Path) -> None:
    from osgeo import gdal

    src_ds = gdal.Open(_normalize_vsi_source_path(src_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"failed to open source raster: {src_path}")
    try:
        rows = int(src_ds.RasterYSize)
        cols = int(src_ds.RasterXSize)
        b1 = src_ds.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
        if b1 is None:
            raise RuntimeError(f"failed reading source raster: {src_path}")
        if np.iscomplexobj(b1):
            complex_arr = np.asarray(b1, dtype=np.complex64)
        elif int(src_ds.RasterCount) >= 2:
            b2 = src_ds.GetRasterBand(2).ReadAsArray(0, 0, cols, rows)
            if b2 is None:
                b2 = np.zeros_like(b1, dtype=np.float32)
            complex_arr = np.asarray(b1, dtype=np.float32) + 1j * np.asarray(b2, dtype=np.float32)
            complex_arr = complex_arr.astype(np.complex64)
        else:
            complex_arr = np.asarray(b1, dtype=np.float32).astype(np.complex64)

        drv = gdal.GetDriverByName("GTiff")
        dst_ds = drv.Create(
            str(dst_path),
            cols,
            rows,
            1,
            gdal.GDT_CFloat32,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        if dst_ds is None:
            raise RuntimeError(f"failed to create destination raster: {dst_path}")
        gt = src_ds.GetGeoTransform(can_return_null=True)
        if gt is not None:
            dst_ds.SetGeoTransform(gt)
        proj = src_ds.GetProjectionRef()
        if proj:
            dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(complex_arr)
        dst_ds.FlushCache()
        dst_ds = None
    finally:
        src_ds = None


def _read_complex_block(ds, xoff: int, yoff: int, cols: int, rows: int) -> np.ndarray:
    b1 = ds.GetRasterBand(1).ReadAsArray(int(xoff), int(yoff), int(cols), int(rows))
    if b1 is None:
        raise RuntimeError("failed reading SLC block")
    if np.iscomplexobj(b1):
        return np.asarray(b1, dtype=np.complex64)
    if int(ds.RasterCount) >= 2:
        b2 = ds.GetRasterBand(2).ReadAsArray(int(xoff), int(yoff), int(cols), int(rows))
        if b2 is None:
            b2 = np.zeros_like(b1, dtype=np.float32)
        return (np.asarray(b1, dtype=np.float32) + 1j * np.asarray(b2, dtype=np.float32)).astype(
            np.complex64
        )
    return np.asarray(b1, dtype=np.float32).astype(np.complex64)


def _create_cfloat32_like(src_ds, dst_path: str | Path):
    from osgeo import gdal

    rows = int(src_ds.RasterYSize)
    cols = int(src_ds.RasterXSize)
    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(
        str(dst_path),
        cols,
        rows,
        1,
        gdal.GDT_CFloat32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if out is None:
        raise RuntimeError(f"failed to create destination raster: {dst_path}")
    gt = src_ds.GetGeoTransform(can_return_null=True)
    if gt is not None:
        out.SetGeoTransform(gt)
    proj = src_ds.GetProjectionRef()
    if proj:
        out.SetProjection(proj)
    return out


def _doppler_vector_for_width(doppler: dict, radargrid: dict, width: int) -> np.ndarray:
    coeffs = _extract_doppler_coefficients(doppler)
    cols = np.arange(max(int(width), 1), dtype=np.float64)
    range_time0 = float(radargrid.get("rangeTimeFirstPixel", 0.0) or 0.0)
    spacing = float(radargrid.get("columnSpacing", 1.0) or 1.0)
    range_times = range_time0 + (2.0 * spacing * cols / 299792458.0)
    ref = _doppler_reference_point(doppler, radargrid)
    values = np.zeros_like(range_times)
    for order, coeff in enumerate(coeffs):
        values += float(coeff) * np.power(range_times - ref, order)
    return values


def _apply_dc_deramp_reramp(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    delta_hz: np.ndarray,
    prf: float,
    chunk_lines: int = 256,
) -> None:
    from osgeo import gdal

    src = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"failed to open SLC for DC deramp/reramp: {src_path}")
    out = _create_cfloat32_like(src, dst_path)
    try:
        rows = int(src.RasterYSize)
        cols = int(src.RasterXSize)
        delta = np.asarray(delta_hz, dtype=np.float64)
        if delta.size != cols:
            raise RuntimeError(f"DC vector width mismatch: expected {cols}, got {delta.size}")
        two_pi_over_prf = 2.0 * np.pi / max(float(prf), 1.0e-6)
        chunk_lines = max(int(chunk_lines), 1)
        for row0 in range(0, rows, chunk_lines):
            nrows = min(chunk_lines, rows - row0)
            block = _read_complex_block(src, 0, row0, cols, nrows)
            lines = np.arange(row0, row0 + nrows, dtype=np.float64)[:, None]
            phase = np.exp(-1j * two_pi_over_prf * (lines * delta[None, :])).astype(np.complex64)
            out.GetRasterBand(1).WriteArray((block * phase).astype(np.complex64), 0, row0)
        out.FlushCache()
    finally:
        out = None
        src = None


def _build_commonband_window(length: int, prf: float, bandwidth_hz: float) -> np.ndarray:
    length = max(int(length), 1)
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    half_bw = 0.5 * max(float(bandwidth_hz), 0.0)
    taper_hz = max(float(os.environ.get("D2SAR_DC_COMMONBAND_TAPER_HZ", min(0.1 * half_bw, 25.0))), 0.0)
    taper_hz = min(taper_hz, half_bw)
    freqs = np.fft.fftfreq(length, d=1.0 / max(float(prf), 1.0e-6))
    af = np.abs(freqs)
    if half_bw <= 0.0:
        return np.zeros((length,), dtype=np.float32)
    if taper_hz <= 0.0:
        return (af <= half_bw).astype(np.float32)
    pass_edge = max(half_bw - taper_hz, 0.0)
    win = np.zeros((length,), dtype=np.float32)
    inner = af <= pass_edge
    trans = (af > pass_edge) & (af <= half_bw)
    win[inner] = 1.0
    if np.any(trans):
        xi = (af[trans] - pass_edge) / max(taper_hz, 1.0e-6)
        win[trans] = 0.5 * (1.0 + np.cos(np.pi * xi))
    return win


def _apply_dc_commonband(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    source_hz: np.ndarray,
    target_hz: np.ndarray,
    prf: float,
    bandwidth_hz: float,
    chunk_cols: int = 256,
) -> None:
    from osgeo import gdal

    src = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"failed to open SLC for DC common-band filtering: {src_path}")
    out = _create_cfloat32_like(src, dst_path)
    try:
        rows = int(src.RasterYSize)
        cols = int(src.RasterXSize)
        source_hz = np.asarray(source_hz, dtype=np.float64)
        target_hz = np.asarray(target_hz, dtype=np.float64)
        if source_hz.size != cols or target_hz.size != cols:
            raise RuntimeError(
                f"common-band doppler vector width mismatch: width={cols}, "
                f"source={source_hz.size}, target={target_hz.size}"
            )
        win = _build_commonband_window(rows, prf, bandwidth_hz).astype(np.complex64)
        two_pi_over_prf = 2.0 * np.pi / max(float(prf), 1.0e-6)
        lines = np.arange(rows, dtype=np.float64)[:, None]
        chunk_cols = max(int(chunk_cols), 1)
        for col0 in range(0, cols, chunk_cols):
            ncols = min(chunk_cols, cols - col0)
            block = _read_complex_block(src, col0, 0, ncols, rows)
            src_hz = source_hz[col0 : col0 + ncols][None, :]
            tgt_hz = target_hz[col0 : col0 + ncols][None, :]
            deramp = np.exp(-1j * two_pi_over_prf * (lines * src_hz)).astype(np.complex64)
            reramp = np.exp(1j * two_pi_over_prf * (lines * tgt_hz)).astype(np.complex64)
            spectrum = np.fft.fft(block * deramp, axis=0)
            spectrum *= win[:, None]
            filtered = np.fft.ifft(spectrum, axis=0).astype(np.complex64)
            out.GetRasterBand(1).WriteArray((filtered * reramp).astype(np.complex64), col0, 0)
        out.FlushCache()
    finally:
        out = None
        src = None


def _apply_dc_signal_processing(
    slc_path: str | Path,
    *,
    radargrid: dict,
    acquisition: dict,
    source_doppler: dict,
    target_doppler: dict,
    dc_policy: dict,
) -> None:
    actions = set(dc_policy.get("actions") or [])
    if not actions.intersection({"dc-deramp-reramp", "dc-commonband"}):
        return
    path = Path(slc_path)
    prf = float(acquisition.get("prf", 1.0) or 1.0)
    width = int(radargrid.get("numberOfColumns", 0) or 0)
    source_hz = _doppler_vector_for_width(source_doppler, radargrid, width)
    target_hz = _doppler_vector_for_width(target_doppler, radargrid, width)
    current = path
    if "dc-deramp-reramp" in actions:
        shifted = path.with_suffix(path.suffix + ".dcshift.tif")
        _apply_dc_deramp_reramp(
            current,
            shifted,
            delta_hz=source_hz - target_hz,
            prf=prf,
        )
        shutil.move(str(shifted), str(path))
        current = path
    if "dc-commonband" in actions:
        cband = path.with_suffix(path.suffix + ".cband.tif")
        _apply_dc_commonband(
            current,
            cband,
            source_hz=source_hz,
            target_hz=target_hz,
            prf=prf,
            bandwidth_hz=float(dc_policy.get("common_azimuth_bandwidth_hz", prf * 0.5)),
        )
        shutil.move(str(cband), str(path))


def _close_isce3_rasters(rasters: list[object]) -> None:
    import gc

    for raster in reversed(rasters):
        close_dataset = getattr(raster, "close_dataset", None)
        if callable(close_dataset):
            close_dataset()
    gc.collect()


@contextlib.contextmanager
def _suppress_native_stdout():
    """Suppress noisy native-library progress writes while preserving stderr."""
    sys.stdout.flush()
    saved_stdout_fd = None
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            stdout_fd = sys.__stdout__.fileno()
            saved_stdout_fd = os.dup(stdout_fd)
            os.dup2(devnull.fileno(), stdout_fd)
        except Exception:
            stdout_fd = None
            saved_stdout_fd = None
        try:
            with contextlib.redirect_stdout(devnull):
                yield
        finally:
            sys.stdout.flush()
            try:
                ctypes.CDLL(None).fflush(None)
            except Exception:
                pass
            if stdout_fd is not None and saved_stdout_fd is not None:
                os.dup2(saved_stdout_fd, stdout_fd)
                os.close(saved_stdout_fd)


def _compute_normalization_scaling(
    source_prf: float,
    target_prf: float,
    source_rsr: float,
    target_rsr: float,
    source_rows: int,
    source_cols: int,
) -> tuple[float, float, int, int, float, float]:
    """
    Compute normalization scaling factors and output dimensions.

    Following ISCE2's normalizeSecondarySampling approach:
    - alpha = source_prf / target_prf (azimuth scaling)
    - beta = source_rsr / target_rsr (range scaling)
    - output_rows = floor((source_rows - 1) / alpha) + 1
    - output_cols = floor((source_cols - 1) / beta) + 1

    Returns:
        alpha, beta, out_rows, out_cols, out_prf, out_rsr
    """
    eps = 1.0e-6
    alpha = max(abs(source_prf) / max(abs(target_prf), eps), eps)
    beta = max(abs(source_rsr) / max(abs(target_rsr), eps), eps)

    out_rows = int(math.floor((max(source_rows, 1) - 1) / alpha)) + 1
    out_cols = int(math.floor((max(source_cols, 1) - 1) / beta)) + 1
    out_rows = max(out_rows, 1)
    out_cols = max(out_cols, 1)

    out_prf = target_prf
    out_rsr = target_rsr

    return alpha, beta, out_rows, out_cols, out_prf, out_rsr


def _resample_slave_slc_with_isce3(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    source_rows: int,
    source_cols: int,
    source_prf: float,
    target_prf: float,
    source_rsr: float,
    target_rsr: float,
    geometry_mode: str,
    doppler_coefficients: list[float],
) -> tuple[str, int, int, float, float]:
    """
    Resample slave SLC to match reference PRF/RSR using ISCE3 ResampSlc.

    Following ISCE2's stretch-based resampling approach:
    - az_offset = (alpha - 1) * out_row  where alpha = source_prf / target_prf
    - rg_offset = (beta - 1) * out_col   where beta = source_rsr / target_rsr

    Returns:
        (output_path, output_rows, output_cols, alpha, beta)
    """
    src_path_str = _normalize_vsi_source_path(src_path)
    src_path = Path(src_path_str) if not str(src_path_str).startswith("/vsi") else src_path_str
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    isce3_rasters: list[object] = []

    alpha, beta, out_rows, out_cols, out_prf, out_rsr = _compute_normalization_scaling(
        source_prf, target_prf, source_rsr, target_rsr, source_rows, source_cols
    )

    try:
        import isce3
        from isce3.core import LUT2d
        from isce3.io import Raster
        from osgeo import gdal

        input_raster = Raster(str(src_path))
        isce3_rasters.append(input_raster)

        rg_off_path = dst_path.parent / "prep_range_offsets.tif"
        az_off_path = dst_path.parent / "prep_azimuth_offsets.tif"
        drv = gdal.GetDriverByName("GTiff")

        range_slope = float(beta - 1.0)
        azimuth_slope = float(alpha - 1.0)

        doppler = LUT2d()
        if geometry_mode == "native-doppler" and doppler_coefficients:
            doppler = LUT2d(0.0, 0.0, np.array([doppler_coefficients], dtype=np.float64))

        cls = isce3.image.ResampSlc
        resamp = cls(
            doppler,
            0.0,
            1.0,
            0.0,
            out_prf if out_prf > 0 else 1.0,
            1.0,
            0.0 + 0.0j,
        )

        output_raster = Raster(str(dst_path), out_cols, out_rows, 1, gdal.GDT_CFloat32, "GTiff")
        isce3_rasters.append(output_raster)

        offset_block_rows = max(int(os.environ.get("D2SAR_NORMALIZE_OFFSET_BLOCK_ROWS", 512)), 1)
        range_values = (np.arange(out_cols, dtype=np.float32) * np.float32(range_slope))[None, :]
        for path, mode in ((rg_off_path, "range"), (az_off_path, "azimuth")):
            ds = drv.Create(str(path), out_cols, out_rows, 1, gdal.GDT_Float32)
            if ds is None:
                raise RuntimeError(f"failed to create normalization offset raster: {path}")
            band = ds.GetRasterBand(1)
            for row0 in range(0, out_rows, offset_block_rows):
                nrows = min(offset_block_rows, out_rows - row0)
                if mode == "range":
                    data = np.broadcast_to(range_values, (nrows, out_cols)).astype(
                        np.float32,
                        copy=True,
                    )
                else:
                    rows_vec = np.arange(row0, row0 + nrows, dtype=np.float32)[:, None]
                    data = (rows_vec * np.float32(azimuth_slope)).astype(np.float32, copy=False)
                    data = np.broadcast_to(data, (nrows, out_cols)).astype(np.float32, copy=True)
                band.WriteArray(data, 0, row0)
            ds.FlushCache()
            ds = None

        for raster in isce3_rasters:
            if hasattr(raster, 'close_dataset'):
                raster.close_dataset()
        isce3_rasters.clear()

        range_offset_raster2 = Raster(str(rg_off_path))
        azimuth_offset_raster2 = Raster(str(az_off_path))

        input_raster2 = Raster(str(src_path))
        output_raster2 = Raster(str(dst_path), out_cols, out_rows, 1, gdal.GDT_CFloat32, "GTiff")

        with _suppress_native_stdout():
            resamp.resamp(input_raster2, output_raster2, range_offset_raster2, azimuth_offset_raster2)

        for raster in [input_raster2, output_raster2, range_offset_raster2, azimuth_offset_raster2]:
            if hasattr(raster, 'close_dataset'):
                raster.close_dataset()

    except Exception as exc:
        for raster in isce3_rasters:
            if hasattr(raster, 'close_dataset'):
                try:
                    raster.close_dataset()
                except Exception:
                    pass
        raise RuntimeError(f"slave normalization resample failed: {exc}") from exc

    return str(dst_path), out_rows, out_cols, alpha, beta


def _resample_slave_slc(*args, **kwargs) -> tuple[str, int, int, float, float]:
    return _resample_slave_slc_with_isce3(*args, **kwargs)


def build_preprocess_plan(
    precheck: dict,
    slave_manifest_path: str | Path,
    stage_dir: str | Path,
    *,
    master_acquisition: dict | None = None,
    master_radargrid: dict | None = None,
    master_doppler: dict | None = None,
    slave_acquisition: dict | None = None,
    slave_radargrid: dict | None = None,
    slave_doppler: dict | None = None,
) -> tuple[dict, str]:
    slave_manifest_path = Path(slave_manifest_path).resolve()
    stage_dir = Path(stage_dir).resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
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
    orbit_path = resolve_manifest_metadata_path(
        slave_manifest_path, original_manifest, "orbit"
    )
    scene_path = resolve_manifest_metadata_path(
        slave_manifest_path, original_manifest, "scene"
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

    if precheck.get("checks", {}).get("prf", {}).get("severity") in {"warn", "fatal"}:
        plan["actions"].append("normalize-slave-prf")
    if precheck.get("checks", {}).get("doppler", {}).get("severity") in {"warn", "fatal"}:
        plan["actions"].append("use-zero-doppler-geometry")
    if precheck.get("checks", {}).get("radar_grid", {}).get("severity") in {"warn", "fatal"}:
        plan["actions"].append("resample-slave-to-master-grid")

    normalized_manifest = json.loads(json.dumps(original_manifest))
    normalized_manifest.setdefault("processing", {})

    normalized_acquisition = dict(acquisition)
    normalized_radargrid = dict(radargrid)
    normalized_doppler = dict(doppler)
    source_prf = float(acquisition.get("prf", 0.0))
    target_prf = float((master_acquisition or {}).get("prf", source_prf))
    doppler_coefficients = _extract_doppler_coefficients(doppler)

    source_rows = int(radargrid.get("numberOfRows", 0))
    source_cols = int(radargrid.get("numberOfColumns", 0))
    source_rsr = float(radargrid.get("columnSpacing", 0.0))
    if source_rsr <= 0:
        source_rsr = 1.0
    target_rsr = float(master_radargrid.get("columnSpacing", source_rsr)) if master_radargrid else source_rsr
    if target_rsr <= 0:
        target_rsr = source_rsr

    prf_affects = "normalize-slave-prf" in plan["actions"]
    dc_policy = _build_dc_policy(
        master_acquisition=master_acquisition,
        master_radargrid=master_radargrid,
        master_doppler=master_doppler,
        slave_acquisition=acquisition,
        slave_radargrid=radargrid,
        slave_doppler=doppler,
        prf_affects=prf_affects,
    )
    for action in dc_policy.get("actions", []):
        if action not in plan["actions"]:
            plan["actions"].append(action)
    if not plan["actions"]:
        plan["actions"].append("pass-through")
    plan["dc_policy"] = dc_policy

    if master_acquisition and "normalize-slave-prf" in plan["actions"]:
        normalized_acquisition["sourcePrf"] = acquisition.get("prf")
        normalized_acquisition["prf"] = master_acquisition.get("prf")
        if "startGPSTime" in acquisition and "startGPSTime" in master_acquisition:
            normalized_acquisition["sourceStartGPSTime"] = acquisition.get("startGPSTime")
    if "prf" in normalized_acquisition:
        normalized_radargrid["prf"] = normalized_acquisition["prf"]

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
    if "harmonize-doppler-to-master" in plan["actions"] and master_doppler:
        normalized_doppler["signalDopplerAfterNormalization"] = master_doppler.get(
            "combinedDoppler",
            {},
        )
    normalized_doppler["processing"]["insar_preprocess"] = {
        "sourceDopplerPath": str(doppler_path),
        "actions": plan["actions"],
        "source_coefficients": doppler_coefficients,
        "dc_policy": dc_policy,
    }

    slc_entry = original_manifest.get("slc", {})
    slc_path = resolve_manifest_data_path(slave_manifest_path, slc_entry.get("path"))
    if slc_path is not None:
        slc_path = _normalize_vsi_source_path(slc_path)
    normalized_slc_value = slc_entry
    normalized_slc_path = None if slc_path is None else stage_dir / f"normalized_slave{Path(str(slc_path)).suffix or '.tif'}"
    should_materialize_slc = plan["requires_normalization"] and _requires_slc_materialization(plan["actions"])
    sampling_materialization_required = plan["requires_normalization"] and _requires_sampling_materialization(plan["actions"])
    dc_materialization_required = plan["requires_normalization"] and _requires_dc_materialization(plan["actions"])
    slc_source_available = slc_path is not None and (
        _is_gdal_virtual_path(slc_path) or Path(str(slc_path)).exists()
    )
    dc_signal_processing = {
        "requested": bool(dc_materialization_required),
        "performed": False,
        "skipped_reason": None,
    }
    if should_materialize_slc and slc_source_available:
        if sampling_materialization_required:
            resamp_result = _resample_slave_slc(
                slc_path,
                normalized_slc_path,
                source_rows=source_rows,
                source_cols=source_cols,
                source_prf=source_prf,
                target_prf=target_prf,
                source_rsr=source_rsr,
                target_rsr=target_rsr,
                geometry_mode=plan["geometry_mode"],
                doppler_coefficients=doppler_coefficients,
            )
            output_path, output_rows, output_cols, alpha, beta = resamp_result
            normalized_radargrid["numberOfRows"] = output_rows
            normalized_radargrid["numberOfColumns"] = output_cols
            normalized_radargrid["prf"] = normalized_acquisition.get("prf", target_prf)
            if output_rows != source_rows or output_cols != source_cols:
                if "rangeTimeFirstPixel" in radargrid:
                    normalized_radargrid["rangeTimeFirstPixel"] = float(radargrid["rangeTimeFirstPixel"])
                if "startGPSTime" in acquisition:
                    normalized_acquisition["startGPSTime"] = float(acquisition["startGPSTime"])
        elif dc_materialization_required:
            _copy_raster_with_gdal(slc_path, normalized_slc_path)
            output_rows = source_rows
            output_cols = source_cols
        if dc_materialization_required and normalized_slc_path is not None:
            _apply_dc_signal_processing(
                normalized_slc_path,
                radargrid=normalized_radargrid,
                acquisition=normalized_acquisition,
                source_doppler=doppler,
                target_doppler=master_doppler or {"combinedDoppler": {"coefficients": [0.0]}},
                dc_policy=dc_policy,
            )
            dc_signal_processing["performed"] = True
        normalized_slc_value = {"path": str(normalized_slc_path)} if isinstance(slc_entry, dict) else str(normalized_slc_path)
    elif sampling_materialization_required and slc_path is not None:
        raise FileNotFoundError(f"slave SLC not found for normalization: {slc_path}")
    elif dc_materialization_required:
        dc_signal_processing["skipped_reason"] = "source_slc_unavailable"

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
        if "numberOfRows" in normalized_radargrid:
            new_slc["rows"] = int(normalized_radargrid["numberOfRows"])
        if "numberOfColumns" in normalized_radargrid:
            new_slc["columns"] = int(normalized_radargrid["numberOfColumns"])
        if normalized_slc_path is not None:
            new_slc["sample_format"] = "cfloat32"
            new_slc["storage_layout"] = "single_band_complex"
            new_slc["complex_band_count"] = 1
            new_slc.pop("band_mapping", None)
            new_slc["processing_format"] = "single_band_cfloat32"
        normalized_manifest["slc"] = new_slc
    else:
        normalized_manifest["slc"] = {
            "path": normalized_slc_value,
            "rows": int(normalized_radargrid.get("numberOfRows", 0)),
            "columns": int(normalized_radargrid.get("numberOfColumns", 0)),
        }
        if normalized_slc_path is not None:
            normalized_manifest["slc"].update(
                {
                    "sample_format": "cfloat32",
                    "storage_layout": "single_band_complex",
                    "complex_band_count": 1,
                    "processing_format": "single_band_cfloat32",
                }
            )
    normalized_manifest["metadata"] = dict(normalized_manifest.get("metadata", {}))
    normalized_manifest["metadata"]["acquisition"] = str(normalized_acquisition_path)
    normalized_manifest["metadata"]["radargrid"] = str(normalized_radargrid_path)
    normalized_manifest["metadata"]["doppler"] = str(normalized_doppler_path)
    if orbit_path.exists():
        normalized_manifest["metadata"]["orbit"] = str(orbit_path)
    if scene_path.exists():
        normalized_manifest["metadata"]["scene"] = str(scene_path)
    normalized_manifest["processing"]["insar_preprocess"] = {
        "source_manifest": str(slave_manifest_path),
        "actions": plan["actions"],
        "geometry_mode": plan["geometry_mode"],
        "dc_policy": dc_policy,
        "dc_signal_processing": dc_signal_processing,
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
