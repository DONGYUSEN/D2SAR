from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timedelta
from dataclasses import dataclass, replace
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from osgeo import gdal, osr

import strip_insar2


VALID_SWATHS = ("IW1", "IW2", "IW3")
STAGE_SEQUENCE = ("check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6")


class IonosphericParams:
    """电离层校正参数，参考 ISCE2 runIon.py 的 ionParam。"""

    def __init__(self) -> None:
        self.all_steps = ["subband", "rawion", "grd2ion", "filt_gaussian", "ionosphere_shift", "ion2grd", "esd"]
        self.do_ion = False
        self.ion_height = 200.0
        self.ion_fit = True
        self.ion_filtering_winsize_max = 200
        self.ion_filtering_winsize_min = 100
        self.ionshift_filtering_winsize_max = 150
        self.ionshift_filtering_winsize_min = 75
        self.azshift_flag = 1
        self.number_azimuth_looks = 50
        self.number_range_looks = 200
        self.number_azimuth_looks0 = 10
        self.number_range_looks0 = 40
        self.ion_dirname = "ion"
        self.lower_dirname = "lower"
        self.upper_dirname = "upper"
        self.ioncal_dirname = "ion_cal"
        self.ion_burst_dirname = "ion_burst"
        self.fine_ifg_dirname = "fine_interferogram"
        self.merged_dirname = "merged"
        self.radar_wavelength = 0.0
        self.rg_bandwidth_for_split = 40.0 * 10**6
        self.rg_bandwidth_sub = self.rg_bandwidth_for_split / 3.0
        self.radar_wavelength_lower = 0.0
        self.radar_wavelength_upper = 0.0
        self.pass_direction = ""
        self.cal_ion_with_merged = False
        self.ramp_removel = 0


def _ensure_2d_array(arr: Any) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data[np.newaxis, :]
    return data


def _weighted_gaussian_smooth(values: np.ndarray, weights: np.ndarray, winsize_max: int, winsize_min: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    values = np.nan_to_num(values, nan=0.0)
    weights = np.nan_to_num(weights, nan=0.0)
    span = max(int(winsize_min), int(winsize_max))
    kernel_size = max(3, min(31, span // 10 * 2 + 1))
    half = kernel_size // 2
    sigma = max(1.0, kernel_size / 6.0)
    grid = np.arange(kernel_size, dtype=np.float64) - half
    kernel_1d = np.exp(-(grid ** 2) / (2.0 * sigma ** 2))
    kernel_1d /= np.sum(kernel_1d)
    kernel = np.outer(kernel_1d, kernel_1d)
    pad = ((half, half), (half, half))
    numerator = np.pad(values * weights, pad, mode="edge")
    denominator = np.pad(weights, pad, mode="edge")
    out = np.zeros_like(values, dtype=np.float64)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            block_num = numerator[i : i + kernel_size, j : j + kernel_size]
            block_den = denominator[i : i + kernel_size, j : j + kernel_size]
            w = kernel * block_den
            denom = np.sum(w)
            out[i, j] = float(np.sum(block_num * kernel) / denom) if denom > 0 else float(np.mean(block_num))
    return out


def _fit_weighted_plane(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(mask) < 3:
        return values, {"a": 0.0, "b": 0.0, "c": float(np.nanmedian(values)) if np.isfinite(np.nanmedian(values)) else 0.0}
    a = np.column_stack((xx[mask], yy[mask], np.ones(np.count_nonzero(mask), dtype=np.float64)))
    w = np.sqrt(np.clip(weights[mask], 0.0, None))
    aw = a * w[:, None]
    bw = values[mask] * w
    coeffs, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return values - plane, {"a": float(coeffs[0]), "b": float(coeffs[1]), "c": float(coeffs[2])}


def _split_subband(
    slc: np.ndarray,
    radar_wavelength: float,
    rg_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """将 SLC 分裂为上下子带。"""
    data = np.asarray(slc)
    if data.ndim == 0:
        data = data.reshape(1, 1)
    axis = data.ndim - 1
    n = data.shape[axis]
    spectrum = np.fft.fft(data, axis=axis)
    freq = np.fft.fftfreq(n)
    width = 1.0 / 3.0
    sigma = max(width / 4.0, 1e-3)
    lower_mask = np.exp(-((freq + width / 3.0) ** 2) / (2.0 * sigma ** 2))
    upper_mask = np.exp(-((freq - width / 3.0) ** 2) / (2.0 * sigma ** 2))
    reshape = [1] * data.ndim
    reshape[axis] = n
    lower = np.fft.ifft(spectrum * lower_mask.reshape(reshape), axis=axis)
    upper = np.fft.ifft(spectrum * upper_mask.reshape(reshape), axis=axis)
    c = 299792458.0
    wavelength_lower = c / (c / float(radar_wavelength) - float(rg_bandwidth) / 3.0)
    wavelength_upper = c / (c / float(radar_wavelength) + float(rg_bandwidth) / 3.0)
    return lower, upper, float(wavelength_lower), float(wavelength_upper)


def _estimate_raw_ionosphere(
    lower_ifg: np.ndarray,
    upper_ifg: np.ndarray,
    coherence: np.ndarray,
    wavelength_lower: float,
    wavelength_upper: float,
    ion_height: float = 200.0,
) -> np.ndarray:
    """基于上下子带干涉图差异估算原始电离层相位。"""
    lower = np.asarray(lower_ifg)
    upper = np.asarray(upper_ifg)
    coh = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
    phase_diff = np.angle(lower * np.conj(upper))
    delta_wl = max(abs(float(wavelength_upper) - float(wavelength_lower)), 1e-12)
    scale = (float(wavelength_lower) * float(wavelength_upper)) / (4.0 * np.pi * delta_wl)
    ion_phase = phase_diff * scale * coh
    if ion_height > 0:
        ion_phase = ion_phase / float(ion_height) * float(ion_height)
    return ion_phase.astype(np.float64, copy=False)


def _grd2ion(ion_phase: np.ndarray, reference_grid: dict[str, Any] | None = None) -> dict[str, Any]:
    """占位版 grd2ion：保留地理参考信息与电离层相位。"""
    return {
        "ion_phase": np.asarray(ion_phase, dtype=np.float64),
        "grid": dict(reference_grid or {}),
    }


def _filter_ionosphere(
    ion_phase: np.ndarray,
    coherence: np.ndarray,
    winsize_max: int = 200,
    winsize_min: int = 100,
    fit: bool = True,
) -> np.ndarray:
    """对电离层相位进行高斯滤波。"""
    phase = _ensure_2d_array(ion_phase).astype(np.float64, copy=False)
    coh = _ensure_2d_array(coherence).astype(np.float64, copy=False)
    weights = np.clip(coh, 0.0, 1.0)
    working = phase
    if fit:
        working, _ = _fit_weighted_plane(working, weights)
    filtered = _weighted_gaussian_smooth(working, weights, winsize_max, winsize_min)
    if fit:
        filtered = filtered + np.nanmedian(phase)
    return filtered


def _compute_ionosphere_shift(
    ion_phase: np.ndarray,
    wavelength: float,
    ion_height: float = 200.0,
    azshift_flag: int = 1,
) -> np.ndarray:
    """将电离层相位转换为方位向偏移量。"""
    phase = np.asarray(ion_phase, dtype=np.float64)
    if azshift_flag <= 0:
        return np.zeros_like(phase, dtype=np.float64)
    shift_m = phase * float(wavelength) / (4.0 * np.pi)
    if ion_height > 0:
        shift_px = shift_m / float(ion_height)
    else:
        shift_px = shift_m
    if azshift_flag == 1:
        shift_px = shift_px - np.nanmean(shift_px)
    return shift_px.astype(np.float64, copy=False)


def _ion2grd(ion_data: dict[str, Any], reference_grid: dict[str, Any] | None = None) -> dict[str, Any]:
    """占位版 ion2grd：输出与网格对齐的电离层产品。"""
    payload = dict(ion_data)
    if reference_grid is not None:
        payload["reference_grid"] = dict(reference_grid)
    return payload


def _esd(ion_shift: np.ndarray, burst_metadata: list[Any] | None = None) -> dict[str, Any]:
    """占位版 ESD：汇总电离层方位偏移统计。"""
    data = np.asarray(ion_shift, dtype=np.float64)
    valid = data[np.isfinite(data)]
    return {
        "burst_count": len(burst_metadata or []),
        "valid_count": int(valid.size),
        "mean_shift": float(np.mean(valid)) if valid.size else 0.0,
        "median_shift": float(np.median(valid)) if valid.size else 0.0,
        "std_shift": float(np.std(valid)) if valid.size else 0.0,
    }


def _run_ionospheric_correction(
    plan: dict[str, Any],
    context: Any,
    master_bursts: list[Any],
    slave_bursts: list[Any],
    master_slc_path: str,
    slave_slc_path: str,
    ion_params: IonosphericParams | None = None,
) -> dict[str, Any]:
    """电离层校正主流程框架，参考 ISCE2 runIon.py 的 runIon/ionSwathBySwath。"""
    params = ion_params or IonosphericParams()
    if not params.do_ion:
        return {"status": "skipped", "reason": "do_ion=False", "steps": []}

    master_infos = [_coerce_tops_burst_info(item) for item in master_bursts]
    slave_infos = [_coerce_tops_burst_info(item) for item in slave_bursts]
    burst_count = min(len(master_infos), len(slave_infos))
    if burst_count <= 0:
        return {"status": "skipped", "reason": "no_bursts", "steps": []}
    summary: dict[str, Any] = {
        "status": "ok",
        "steps": [],
        "burst_count": burst_count,
        "parameters": {
            "all_steps": list(params.all_steps),
            "ion_height": float(params.ion_height),
            "ion_fit": bool(params.ion_fit),
            "azshift_flag": int(params.azshift_flag),
        },
    }

    for step in params.all_steps:
        summary["steps"].append({"step": step, "status": "planned"})

    overlap_records: list[dict[str, Any]] = []
    burst_records: list[dict[str, Any]] = []
    wavelength_fallback = float(
        _plan_option(plan, "radar_wavelength", 0.0)
        or getattr(master_infos[0], "radar_wavelength", 0.0)
        or getattr(slave_infos[0], "radar_wavelength", 0.0)
        or 0.05546576
    )
    rg_bandwidth = float(_plan_option(plan, "rg_bandwidth_for_split", params.rg_bandwidth_for_split) or params.rg_bandwidth_for_split)

    def _burst_ionosphere_products(master_burst: Any, slave_burst: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        master_win = _read_complex_slc_valid_window(master_slc_path, adjustValidLineSample(replace(master_burst), replace(slave_burst)))
        slave_win = _read_complex_slc_valid_window(slave_slc_path, adjustValidLineSample(replace(slave_burst), replace(master_burst)))
        rows = min(master_win.shape[0], slave_win.shape[0])
        cols = min(master_win.shape[1], slave_win.shape[1])
        if rows <= 0 or cols <= 0:
            return (
                np.zeros((0, 0), dtype=np.complex64),
                np.zeros((0, 0), dtype=np.complex64),
                np.zeros((0, 0), dtype=np.float32),
                0.0,
                0.0,
            )
        master_win = np.asarray(master_win[:rows, :cols], dtype=np.complex64)
        slave_win = np.asarray(slave_win[:rows, :cols], dtype=np.complex64)
        lower_master, upper_master, wl_lower, wl_upper = _split_subband(master_win, wavelength_fallback, rg_bandwidth)
        lower_slave, upper_slave, _, _ = _split_subband(slave_win, wavelength_fallback, rg_bandwidth)
        lower_ifg = (lower_master * np.conj(lower_slave)).astype(np.complex64)
        upper_ifg = (upper_master * np.conj(upper_slave)).astype(np.complex64)
        coherence = _estimate_burst_coherence(master_win, slave_win)
        return lower_ifg, upper_ifg, coherence, wl_lower, wl_upper

    for burst in master_infos[:burst_count]:
        slave_burst = slave_infos[min(int(getattr(burst, "burst_index", 1)) - 1, len(slave_infos) - 1)]
        lower_ifg, upper_ifg, coherence, wl_lower, wl_upper = _burst_ionosphere_products(burst, slave_burst)
        if lower_ifg.size == 0 or upper_ifg.size == 0:
            continue
        raw_ionosphere = _estimate_raw_ionosphere(lower_ifg, upper_ifg, coherence, wl_lower, wl_upper, params.ion_height)
        filtered_ionosphere = _filter_ionosphere(
            raw_ionosphere,
            coherence,
            winsize_max=params.ion_filtering_winsize_max,
            winsize_min=params.ion_filtering_winsize_min,
            fit=params.ion_fit,
        )
        ion_shift = _compute_ionosphere_shift(
            filtered_ionosphere,
            wavelength_fallback,
            ion_height=params.ion_height,
            azshift_flag=params.azshift_flag,
        )
        burst_index = int(getattr(burst, "burst_index", 0))
        burst_records.append(
            {
                "burst_index": burst_index,
                "lower_wavelength": float(wl_lower),
                "upper_wavelength": float(wl_upper),
                "raw_ionosphere": _grd2ion(raw_ionosphere),
                "filtered_ionosphere": _ion2grd({"ion_phase": filtered_ionosphere}),
                "ionosphere_shift": _esd(ion_shift, [burst]),
            }
        )
        summary.setdefault("ion_shift", {})[str(burst_index)] = np.asarray(ion_shift, dtype=np.float64).tolist()

    for idx in range(max(0, burst_count - 1)):
        lower_ifg, upper_ifg, coherence, wl_lower, wl_upper = _burst_ionosphere_products(master_infos[idx], slave_infos[idx])
        if lower_ifg.size == 0 or upper_ifg.size == 0:
            continue
        raw_ionosphere = _estimate_raw_ionosphere(lower_ifg, upper_ifg, coherence, wl_lower, wl_upper, params.ion_height)
        filtered_ionosphere = _filter_ionosphere(
            raw_ionosphere,
            coherence,
            winsize_max=params.ion_filtering_winsize_max,
            winsize_min=params.ion_filtering_winsize_min,
            fit=params.ion_fit,
        )
        ion_shift = _compute_ionosphere_shift(
            filtered_ionosphere,
            wavelength_fallback,
            ion_height=params.ion_height,
            azshift_flag=params.azshift_flag,
        )
        overlap_records.append(
            {
                "pair": [idx + 1, idx + 2],
                "operation": "generate_overlap_subband_ifgs",
                "products": {
                    "lower_subband_ifg": f"ion_burst/overlap_{idx + 1:03d}_{idx + 2:03d}/lower.int",
                    "upper_subband_ifg": f"ion_burst/overlap_{idx + 1:03d}_{idx + 2:03d}/upper.int",
                    "raw_ionosphere": f"ion_burst/overlap_{idx + 1:03d}_{idx + 2:03d}/raw.ion",
                },
                "raw_ionosphere_stats": _esd(raw_ionosphere, []),
                "filtered_ionosphere_stats": _esd(filtered_ionosphere, []),
                "ionosphere_shift_stats": _esd(ion_shift, []),
            }
        )

    summary["context"] = {
        "type": type(context).__name__ if context is not None else None,
        "swaths": list(plan.get("swaths", [])),
    }
    summary["products"] = {
        "ion_dirname": params.ion_dirname,
        "lower_dirname": params.lower_dirname,
        "upper_dirname": params.upper_dirname,
        "ioncal_dirname": params.ioncal_dirname,
        "ion_burst_dirname": params.ion_burst_dirname,
    }
    summary["overlap_records"] = overlap_records
    summary["burst_records"] = burst_records
    summary["steps"] = [{"step": step, "status": "done"} for step in params.all_steps]
    return summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _host_path_fallback(path_str: str) -> str:
    p = Path(path_str)
    if p.is_file():
        return str(p)
    if p.is_absolute():
        text = str(p)
        if text.startswith("/results/") or text == "/results":
            mapped = _repo_root() / "results" / text.removeprefix("/results/").removeprefix("/results")
            if mapped.is_file():
                return str(mapped)
        if text.startswith("/work/") or text == "/work":
            mapped = _repo_root() / text.removeprefix("/work/").removeprefix("/work")
            if mapped.is_file():
                return str(mapped)
    return path_str


def parse_swath_selector(swath: str) -> tuple[list[str], list[str]]:
    text = str(swath).strip().upper()
    if text == "ALL":
        return list(VALID_SWATHS), []

    if "," in text:
        parts = [item.strip() for item in text.split(",") if item.strip()]
    else:
        parts = [text]

    if not parts:
        raise ValueError("empty --swath selector")
    if any(item not in VALID_SWATHS for item in parts):
        raise ValueError(
            "--swath must be one of: IW1, IW2, IW3, IW1,IW2, IW2,IW3, IW1,IW3, all"
        )
    # keep order and deduplicate
    swaths = list(dict.fromkeys(parts))
    warnings: list[str] = []
    if set(swaths) == {"IW1", "IW3"} and len(swaths) == 2:
        warnings.append("IW1,IW3 are non-adjacent; skip cross-swath merge.")
    return swaths, warnings


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest_metadata_path(manifest_path: str | Path, manifest: dict[str, Any], key: str) -> str:
    ref = manifest.get("metadata", {}).get(key)
    if not ref:
        raise ValueError(f"manifest metadata.{key} missing")
    ref_path = Path(str(ref))
    if ref_path.is_absolute():
        return _host_path_fallback(str(ref_path))
    return _host_path_fallback(str(Path(manifest_path).resolve().parent / ref_path))


def _resolve_manifest_data_path(manifest_path: str | Path, ref: Any) -> str:
    if isinstance(ref, dict):
        path_val = str(ref.get("path", ""))
    elif isinstance(ref, str):
        path_val = ref
    else:
        path_val = ""
    if not path_val:
        return ""
    if path_val.startswith("/vsizip/") or path_val.startswith("/vsitar/"):
        return path_val
    p = Path(path_val)
    if p.is_absolute():
        return _host_path_fallback(str(p))
    return _host_path_fallback(str((Path(manifest_path).resolve().parent / p).resolve()))


def _load_tops_bundle(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if str(manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("tops_insar requires sentinel-1 manifest")
    required = ("acquisition", "radargrid", "tops")
    loaded: dict[str, Any] = {"manifest": manifest}
    for key in required:
        meta_path = _resolve_manifest_metadata_path(manifest_path, manifest, key)
        with open(meta_path, encoding="utf-8") as f:
            loaded[key] = json.load(f)
    return loaded


def _write_plan(plan: dict[str, Any], output_dir: str | Path) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "tops_insar_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(plan_path)


def _stage_range(start_stage: str, end_stage: str) -> list[str]:
    start = str(start_stage).strip()
    end = str(end_stage).strip()
    if start not in STAGE_SEQUENCE:
        raise ValueError(f"invalid --start-stage: {start}")
    if end not in STAGE_SEQUENCE:
        raise ValueError(f"invalid --end-stage: {end}")
    i0 = STAGE_SEQUENCE.index(start)
    i1 = STAGE_SEQUENCE.index(end)
    if i0 > i1:
        raise ValueError("--start-stage must not be after --end-stage")
    return list(STAGE_SEQUENCE[i0 : i1 + 1])


def _run_stage(stage: str, plan: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    stage_dir = out / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    marker = stage_dir / "done.json"
    if stage == "check":
        record = _run_stage_check(plan)
    elif stage == "prep":
        record = _run_stage_prep(plan)
    elif stage == "crop":
        record = _run_stage_crop(plan)
    elif stage == "p0":
        record = _run_stage_p0(plan)
    elif stage == "p1":
        record = _run_stage_p1(plan)
    elif stage == "p2":
        record = _run_stage_p2(plan)
    elif stage == "p3":
        record = _run_stage_p3(plan)
    elif stage == "p4":
        record = _run_stage_p4(plan)
    elif stage == "p5":
        record = _run_stage_p5(plan)
    elif stage == "p6":
        record = _run_stage_p6(plan)
    else:
        record = {
            "stage": stage,
            "status": "ok",
            "message": "stage runner placeholder (not implemented yet)",
            "swaths": plan.get("swaths", []),
        }
    marker.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _execute_stage_window(plan: dict[str, Any], output_dir: str | Path, start_stage: str, end_stage: str, resume: bool) -> dict[str, Any]:
    stages_to_run = _stage_range(start_stage, end_stage)
    stage_status = dict(plan.get("stage_status", {}))
    plan["stage_status"] = stage_status
    out_dir = Path(output_dir)
    for stage in stages_to_run:
        if resume and stage_status.get(stage, {}).get("status") == "ok":
            continue
        stage_status[stage] = _run_stage(stage, plan, output_dir)
        plan["stage_status"] = stage_status
        _write_plan(plan, out_dir)
        if stage_status[stage].get("status") != "ok":
            raise RuntimeError(f"stage {stage} failed: {stage_status[stage].get('message', 'unknown error')}")
    plan["stage_status"] = stage_status
    plan["executed_stages"] = [stage for stage in stages_to_run if stage_status.get(stage, {}).get("status") == "ok"]
    _write_plan(plan, out_dir)
    return plan


def _load_existing_plan(output_dir: str | Path) -> dict[str, Any]:
    plan_path = Path(output_dir) / "tops_insar_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"--resume requested but plan not found: {plan_path}")
    with plan_path.open(encoding="utf-8") as f:
        return json.load(f)


def _parse_utc_timestamp(text: str | None) -> datetime | None:
    if not text:
        return None
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    elif "+" not in value:
        value = value + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _extract_manifest_pair_for_swath(plan: dict[str, Any], swath: str) -> tuple[str, str]:
    if plan.get("mode") == "product":
        swath_inputs = plan.get("swath_inputs", {}).get(swath)
        if not isinstance(swath_inputs, dict):
            raise ValueError(f"missing swath_inputs for {swath}")
        return str(swath_inputs["master_manifest"]), str(swath_inputs["slave_manifest"])
    return str(plan["master_manifest"]), str(plan["slave_manifest"])


def _run_stage_check(plan: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
        master_meta = _load_tops_bundle(master_manifest_path)
        slave_meta = _load_tops_bundle(slave_manifest_path)
        master_swath = str(master_meta.get("tops", {}).get("swath", "")).upper()
        slave_swath = str(slave_meta.get("tops", {}).get("swath", "")).upper()
        master_burst_count = int(len(master_meta.get("tops", {}).get("bursts", [])))
        slave_burst_count = int(len(slave_meta.get("tops", {}).get("bursts", [])))
        if master_swath and master_swath != swath:
            raise ValueError(f"master swath mismatch for {swath}: manifest has {master_swath}")
        if slave_swath and slave_swath != swath:
            raise ValueError(f"slave swath mismatch for {swath}: manifest has {slave_swath}")
        if master_burst_count <= 0 or slave_burst_count <= 0:
            raise ValueError(f"empty burst list for swath {swath}")

        master_start = _parse_utc_timestamp(master_meta.get("acquisition", {}).get("startTimeUTC"))
        slave_start = _parse_utc_timestamp(slave_meta.get("acquisition", {}).get("startTimeUTC"))
        temporal_order = "unknown"
        if master_start and slave_start:
            temporal_order = "master_before_slave" if master_start <= slave_start else "master_after_slave"

        checks.append(
            {
                "swath": swath,
                "master_manifest": master_manifest_path,
                "slave_manifest": slave_manifest_path,
                "master_burst_count": master_burst_count,
                "slave_burst_count": slave_burst_count,
                "temporal_order": temporal_order,
            }
        )
    return {"stage": "check", "status": "ok", "swaths": checks}


def _run_stage_prep(plan: dict[str, Any]) -> dict[str, Any]:
    swath_records: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
        master_meta = _load_tops_bundle(master_manifest_path)
        slave_meta = _load_tops_bundle(slave_manifest_path)
        master_manifest = master_meta["manifest"]
        slave_manifest = slave_meta["manifest"]
        master_slc = _resolve_manifest_data_path(master_manifest_path, master_manifest.get("slc", {}).get("path"))
        slave_slc = _resolve_manifest_data_path(slave_manifest_path, slave_manifest.get("slc", {}).get("path"))
        if not master_slc or not slave_slc:
            raise ValueError(f"missing slc path for swath {swath}")
        swath_records.append(
            {
                "swath": swath,
                "master_slc_path": master_slc,
                "slave_slc_path": slave_slc,
                "master_burst_count": len(master_meta.get("tops", {}).get("bursts", [])),
                "slave_burst_count": len(slave_meta.get("tops", {}).get("bursts", [])),
            }
        )
    return {"stage": "prep", "status": "ok", "swaths": swath_records}


def _run_stage_p0(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p0")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p0", "status": "ok", "swaths": out}

    tasks: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
        master_meta = _load_tops_bundle(master_manifest_path)
        slave_meta = _load_tops_bundle(slave_manifest_path)
        master_grids = master_meta.get("tops", {}).get("bursts", [])
        slave_grids = slave_meta.get("tops", {}).get("bursts", [])
        common = min(len(master_grids), len(slave_grids))
        burst_tasks = []
        for idx in range(common):
            mg = master_grids[idx]
            sg = slave_grids[idx]
            burst_tasks.append(
                {
                    "burst_index": int(mg.get("burstIndex", idx + 1)),
                    "master": {
                        "rows": int(mg.get("numberOfLines", mg.get("numberOfRows", 0))),
                        "cols": int(mg.get("numberOfSamples", mg.get("numberOfColumns", 0))),
                        "line_offset": int(mg.get("lineOffset", 0)),
                    },
                    "slave": {
                        "rows": int(sg.get("numberOfLines", sg.get("numberOfRows", 0))),
                        "cols": int(sg.get("numberOfSamples", sg.get("numberOfColumns", 0))),
                        "line_offset": int(sg.get("lineOffset", 0)),
                    },
                }
            )
        tasks.append(
            {
                "swath": swath,
                "common_burst_count": common,
                "burst_tasks": burst_tasks,
                "topo_gpu_enabled": bool(plan.get("options", {}).get("topo_gpu", False)),
            }
        )
    return {"stage": "p0", "status": "ok", "swaths": tasks}


def _stage_swath_index(plan: dict[str, Any], stage: str) -> dict[str, dict[str, Any]]:
    swath_entries = plan.get("stage_status", {}).get(stage, {}).get("swaths", [])
    index: dict[str, dict[str, Any]] = {}
    if isinstance(swath_entries, list):
        for item in swath_entries:
            if isinstance(item, dict) and item.get("swath"):
                index[str(item["swath"])] = item
    return index


def _run_stage_crop(plan: dict[str, Any]) -> dict[str, Any]:
    check_by_swath = _stage_swath_index(plan, "check")
    burst_limit = plan.get("options", {}).get("burst_limit")
    selected: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        check = check_by_swath.get(swath)
        if not check:
            raise ValueError(f"check result missing for swath {swath}")
        burst_count = int(min(check.get("master_burst_count", 0), check.get("slave_burst_count", 0)))
        burst_indices = list(range(1, burst_count + 1))
        if burst_limit is not None:
            burst_indices = burst_indices[: max(0, int(burst_limit))]
        selected.append(
            {
                "swath": swath,
                "selected_burst_count": len(burst_indices),
                "selected_burst_indices": burst_indices,
            }
        )
    return {"stage": "crop", "status": "ok", "swaths": selected}


def _run_stage_p1(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p1")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p1", "status": "ok", "swaths": out}

    p0_by_swath = _stage_swath_index(plan, "p0")
    crop_by_swath = _stage_swath_index(plan, "crop")
    out: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        p0 = p0_by_swath.get(swath)
        crop = crop_by_swath.get(swath)
        if not p0 or not crop:
            raise ValueError(f"crop/p0 result missing for swath {swath}")
        selected = set(int(v) for v in crop.get("selected_burst_indices", []))
        registration_tasks: list[dict[str, Any]] = []
        overlap_tasks: list[dict[str, Any]] = []
        esd_tasks: list[dict[str, Any]] = []
        for item in p0.get("burst_tasks", []):
            idx = int(item.get("burst_index", 0))
            if idx not in selected:
                continue
            registration_tasks.append(
                {
                    "burst_index": idx,
                    "operation": "coregister_slave_to_master",
                    "inputs": {
                        "master_rows": int(item.get("master", {}).get("rows", 0)),
                        "master_cols": int(item.get("master", {}).get("cols", 0)),
                        "slave_rows": int(item.get("slave", {}).get("rows", 0)),
                        "slave_cols": int(item.get("slave", {}).get("cols", 0)),
                    },
                }
            )
        selected_sorted = sorted(selected)
        for prev_idx, next_idx in zip(selected_sorted[:-1], selected_sorted[1:]):
            if next_idx != prev_idx + 1:
                continue
            overlap_tasks.append(
                {
                    "pair": [prev_idx, next_idx],
                    "operation": "extract_burst_overlap",
                    "products": {
                        "master_overlap_slc": f"overlap_{prev_idx:03d}_{next_idx:03d}/master_overlap.slc",
                        "slave_overlap_slc": f"overlap_{prev_idx:03d}_{next_idx:03d}/slave_overlap.slc",
                    },
                }
            )
            esd_tasks.append(
                {
                    "pair": [prev_idx, next_idx],
                    "operation": "estimate_esd_azimuth_misregistration",
                    "inputs": {
                        "master_overlap_slc": f"overlap_{prev_idx:03d}_{next_idx:03d}/master_overlap.slc",
                        "slave_overlap_slc": f"overlap_{prev_idx:03d}_{next_idx:03d}/slave_overlap.slc",
                    },
                    "products": {
                        "esd_phase": f"overlap_{prev_idx:03d}_{next_idx:03d}/esd_phase.int",
                        "esd_azimuth_offset": f"overlap_{prev_idx:03d}_{next_idx:03d}/esd_azimuth_offset.txt",
                    },
                }
            )
        coreg_strategy = "geometry_only" if len(overlap_tasks) == 0 else "geometry_plus_esd"
        out.append(
            {
                "swath": swath,
                "task_count": len(registration_tasks),
                "tasks": registration_tasks,
                "overlap_task_count": len(overlap_tasks),
                "overlap_tasks": overlap_tasks,
                "esd_task_count": len(esd_tasks),
                "esd_tasks": esd_tasks,
                "coreg_strategy": coreg_strategy,
            }
        )
    if bool(plan.get("options", {}).get("execute_backend", False)):
        for swath_item in out:
            if swath_item.get("overlap_task_count", 0) > 0:
                swath_item["overlap_backend_execution"] = _run_overlap_esd_backend_for_swath(
                    plan, swath_item["swath"], swath_item["overlap_tasks"], swath_item["esd_tasks"]
                )
    return {"stage": "p1", "status": "ok", "swaths": out}


def _run_overlap_esd_backend_for_swath(
    plan: dict[str, Any],
    swath: str,
    overlap_tasks: list[dict[str, Any]],
    esd_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    from osgeo import gdal

    master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
    master_bundle = _load_tops_bundle(master_manifest_path)
    slave_bundle = _load_tops_bundle(slave_manifest_path)
    master_slc = _resolve_manifest_data_path(
        master_manifest_path, master_bundle["manifest"].get("slc", {}).get("path")
    )
    slave_slc = _resolve_manifest_data_path(
        slave_manifest_path, slave_bundle["manifest"].get("slc", {}).get("path")
    )
    if not master_slc or not slave_slc:
        raise ValueError(f"missing slc path for swath {swath}")

    master_ds = gdal.Open(master_slc)
    slave_ds = gdal.Open(slave_slc)
    if master_ds is None or slave_ds is None:
        raise RuntimeError(f"failed to open master/slave slc for swath {swath}")

    master_bursts = {
        int(b.get("burstIndex", i + 1)): b for i, b in enumerate(master_bundle.get("tops", {}).get("bursts", []))
    }
    slave_bursts = {
        int(b.get("burstIndex", i + 1)): b for i, b in enumerate(slave_bundle.get("tops", {}).get("bursts", []))
    }

    master_acq = master_bundle.get("acquisition", {})
    master_prf = float(master_acq.get("prf", 0.0) or 0.0)
    wavelength = float(master_acq.get("wavelength", 0.0) or 0.0)
    
    if master_prf <= 0 or wavelength <= 0:
        raise ValueError(f"Invalid PRF or wavelength for ESD computation")

    az_pixel_spacing = wavelength * master_prf / 2.0

    out_root = Path(plan["plan_path"]).parent / swath / "esd_backend"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    all_valid_offsets = []
    
    extra_offset_cycles = float(plan.get("options", {}).get("extra_esd_cycles", 0.0))
    esd_coherence_threshold = float(plan.get("options", {}).get("esd_coherence_threshold", 0.85))
    extra_offset_rad = extra_offset_cycles * np.pi * 2

    for ov_task, esd_task in zip(overlap_tasks, esd_tasks):
        prev_idx, next_idx = [int(v) for v in ov_task.get("pair", [0, 0])]
        if prev_idx <= 0 or next_idx <= 0:
            continue
        mb_prev = master_bursts.get(prev_idx)
        mb_next = master_bursts.get(next_idx)
        sb_prev = slave_bursts.get(prev_idx)
        sb_next = slave_bursts.get(next_idx)
        if mb_prev is None or mb_next is None or sb_prev is None or sb_next is None:
            continue
        assert mb_prev is not None and mb_next is not None and sb_prev is not None and sb_next is not None
        # 计算真实 overlap 窗口（使用 firstValidLine/numValidLines）
        m_prev_first_valid = int(mb_prev.get("firstValidLine", 0))
        m_prev_num_valid = int(mb_prev.get("numValidLines", mb_prev.get("numberOfLines", 0)))
        m_next_first_valid = int(mb_next.get("firstValidLine", 0))
        m_next_num_valid = int(mb_next.get("numValidLines", mb_next.get("numberOfLines", 0)))
        s_prev_first_valid = int(sb_prev.get("firstValidLine", 0))
        s_prev_num_valid = int(sb_prev.get("numValidLines", sb_prev.get("numberOfLines", 0)))
        s_next_first_valid = int(sb_next.get("firstValidLine", 0))
        s_next_num_valid = int(sb_next.get("numValidLines", sb_next.get("numberOfLines", 0)))

        # burst 有效数据窗口
        m_prev_valid_start = int(mb_prev.get("lineOffset", 0)) + m_prev_first_valid
        m_prev_valid_end = m_prev_valid_start + m_prev_num_valid
        m_next_valid_start = int(mb_next.get("lineOffset", 0)) + m_next_first_valid
        m_next_valid_end = m_next_valid_start + m_next_num_valid
        s_prev_valid_start = int(sb_prev.get("lineOffset", 0)) + s_prev_first_valid
        s_prev_valid_end = s_prev_valid_start + s_prev_num_valid
        s_next_valid_start = int(sb_next.get("lineOffset", 0)) + s_next_first_valid
        s_next_valid_end = s_next_valid_start + s_next_num_valid

        # overlap 区域为两个 burst 有效数据的交集
        m_overlap_start = max(m_prev_valid_start, m_next_valid_start)
        m_overlap_end = min(m_prev_valid_end, m_next_valid_end)
        s_overlap_start = max(s_prev_valid_start, s_next_valid_start)
        s_overlap_end = min(s_prev_valid_end, s_next_valid_end)

        lines = int(m_overlap_end - m_overlap_start) if m_overlap_end > m_overlap_start else 0
        overlap_cols = int(min(mb_prev.get("numberOfSamples", 0), sb_prev.get("numberOfSamples", 0)))
        if lines <= 0 or overlap_cols <= 0:
            raise RuntimeError(f"invalid overlap window for swath {swath} pair {prev_idx}-{next_idx}")

        # 使用 overlap 区域的起始行
        m_prev_row = int(m_overlap_start)
        m_next_row = int(m_next_valid_start)
        s_prev_row = int(s_overlap_start)
        s_next_row = int(s_next_valid_start)

        mb1 = master_ds.GetRasterBand(1).ReadAsArray(0, m_prev_row, overlap_cols, lines)
        mb2 = master_ds.GetRasterBand(1).ReadAsArray(0, m_next_row, overlap_cols, lines)
        sb1 = slave_ds.GetRasterBand(1).ReadAsArray(0, s_prev_row, overlap_cols, lines)
        sb2 = slave_ds.GetRasterBand(1).ReadAsArray(0, s_next_row, overlap_cols, lines)
        if any(arr is None for arr in (mb1, mb2, sb1, sb2)):
            raise RuntimeError(f"failed to read overlap data for swath {swath} pair {prev_idx}-{next_idx}")

        ifg_prev = mb1.astype(np.complex64) * np.conj(sb1.astype(np.complex64))
        ifg_next = mb2.astype(np.complex64) * np.conj(sb2.astype(np.complex64))
        esd_ifg = ifg_prev * np.conj(ifg_next)

        burst_time_interval = float(mb_prev.get("azimuthTimeInterval", 0.0) or 0.0)
        if burst_time_interval <= 0:
            burst_time_interval = 1.0 / master_prf

        freq = _estimate_esd_local_frequency(esd_ifg).astype(np.float32) / (2.0 * np.pi)
        if freq.shape != (lines, overlap_cols):
            freq = np.asarray(freq[:lines, :overlap_cols], dtype=np.float32)

        ifg = esd_ifg.astype(np.complex64)
        
        denom = freq
        valid_freq = denom != 0
        off = np.zeros((lines, overlap_cols), dtype=np.float32)
        off[valid_freq] = (np.angle(ifg[valid_freq]) + extra_offset_rad) / denom[valid_freq]

        phasor = np.exp(1j * np.angle(ifg))
        phasor_col_mean = np.mean(phasor, axis=1, keepdims=True)
        cor = np.abs(phasor_col_mean)
        
        mask = (np.abs(ifg) > 0) & (cor > esd_coherence_threshold)
        
        if np.any(mask):
            vali = off[mask]
            all_valid_offsets.extend(vali.tolist())
            est_az_offset_pixels = float(np.median(vali))
        else:
            est_az_offset_pixels = 0.0
            vali = np.array([])

        results.append(
            {
                "pair": [prev_idx, next_idx],
                "estimated_azimuth_offset_pixels": est_az_offset_pixels,
                "overlap_lines": lines,
                "overlap_cols": overlap_cols,
                "median_offset": float(np.median(vali)) if np.any(mask) else 0.0,
                "mean_offset": float(np.mean(vali)) if np.any(mask) else 0.0,
                "std_offset": float(np.std(vali)) if np.any(mask) else 0.0,
                "n_valid": int(np.sum(mask)) if np.any(mask) else 0,
                "coherence_threshold": esd_coherence_threshold,
            }
        )

    summary = out_root / "esd_summary.json"
    summary_data = {
        "swath": swath,
        "pairs": results,
        "azimuth_pixel_spacing": az_pixel_spacing,
        "total_valid_points": len(all_valid_offsets),
    }
    
    if not all_valid_offsets:
        raise Exception('Coherence threshold too strict. No points left for reliable ESD estimate')
    
    all_offsets_arr = np.array(all_valid_offsets, dtype=np.float64)
    summary_data["overall_median_offset_pixels"] = float(np.median(all_offsets_arr))
    summary_data["overall_mean_offset_pixels"] = float(np.mean(all_offsets_arr))
    summary_data["overall_std_offset_pixels"] = float(np.std(all_offsets_arr))
    secondary_seconds = _store_secondary_timing_correction(
        plan,
        swath,
        float(summary_data["overall_median_offset_pixels"]),
        [master_bursts[k] for k in sorted(master_bursts)],
    )
    summary_data["secondary_timing_correction_seconds"] = float(secondary_seconds)
    summary.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return {"summary_json": str(summary), "pair_count": len(results), "pairs": results, "total_valid_points": len(all_valid_offsets)}


def _run_stage_p2(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p2")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p2", "status": "ok", "swaths": out}

    p1_by_swath = _stage_swath_index(plan, "p1")
    p0_by_swath = _stage_swath_index(plan, "p0")
    out: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        p1 = p1_by_swath.get(swath)
        p0 = p0_by_swath.get(swath)
        if not p1:
            raise ValueError(f"p1 result missing for swath {swath}")
        if not p0:
            raise ValueError(f"p0 result missing for swath {swath}")
        overlap_ifg_tasks = []
        burst_ifg_tasks = []
        esd_refine_tasks = []
        for task in p1.get("tasks", []):
            burst_index = int(task.get("burst_index", 0))
            burst_ifg_tasks.append(
                {
                    "burst_index": burst_index,
                    "operation": "burst_interferogram_and_coherence",
                    "products": {
                        "interferogram": f"burst_{burst_index:03d}/interferogram.int",
                        "coherence": f"burst_{burst_index:03d}/coherence.cor",
                    },
                }
            )
        for task in p1.get("overlap_tasks", []):
            prev_idx, next_idx = task.get("pair", [0, 0])
            overlap_ifg_tasks.append(
                {
                    "pair": [prev_idx, next_idx],
                    "operation": "overlap_interferogram_and_coherence",
                    "products": {
                        "overlap_interferogram": f"overlap_{prev_idx:03d}_{next_idx:03d}/overlap_ifg.int",
                        "overlap_coherence": f"overlap_{prev_idx:03d}_{next_idx:03d}/overlap_coh.cor",
                    },
                }
            )
            esd_refine_tasks.append(
                {
                    "pair": [prev_idx, next_idx],
                    "operation": "apply_esd_refinement_to_coreg_offsets",
                    "inputs": {
                        "overlap_interferogram": f"overlap_{prev_idx:03d}_{next_idx:03d}/overlap_ifg.int",
                        "esd_azimuth_offset": f"overlap_{prev_idx:03d}_{next_idx:03d}/esd_azimuth_offset.txt",
                    },
                }
            )

        total_task_count = len(burst_ifg_tasks) + len(overlap_ifg_tasks) + len(esd_refine_tasks)
        swath_record: dict[str, Any] = {
            "swath": swath,
            "task_count": total_task_count,
            "burst_ifg_task_count": len(burst_ifg_tasks),
            "overlap_ifg_task_count": len(overlap_ifg_tasks),
            "esd_refine_task_count": len(esd_refine_tasks),
            "tasks": burst_ifg_tasks,
            "overlap_tasks": overlap_ifg_tasks,
            "esd_refine_tasks": esd_refine_tasks,
        }
        ion_shift_map: dict[int, np.ndarray] | None = None
        ion_enabled = bool(_plan_option(plan, "do_ionospheric_correction", False))
        if ion_enabled:
            master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
            master_bundle = _load_tops_bundle(master_manifest_path)
            slave_bundle = _load_tops_bundle(slave_manifest_path)
            selected_indices = _selected_burst_indices_for_swath(plan, swath, len(master_bundle.get("tops", {}).get("bursts", [])))
            ion_params = IonosphericParams()
            ion_params.do_ion = True
            ion_result = _run_ionospheric_correction(
                plan,
                context=swath_record,
                master_bursts=_burst_infos_from_bundle(master_bundle, selected_indices),
                slave_bursts=_burst_infos_from_bundle(slave_bundle, selected_indices),
                master_slc_path=_resolve_manifest_data_path(master_manifest_path, master_bundle["manifest"].get("slc", {}).get("path")),
                slave_slc_path=_resolve_manifest_data_path(slave_manifest_path, slave_bundle["manifest"].get("slc", {}).get("path")),
                ion_params=ion_params,
            )
            ion_shift_raw = ion_result.pop("ion_shift", {})
            if isinstance(ion_shift_raw, dict):
                ion_shift_map = {int(k): np.asarray(v, dtype=np.float64) for k, v in ion_shift_raw.items()}
            swath_record["ionospheric_correction"] = ion_result
        if _should_execute_stages(plan) and burst_ifg_tasks:
            burst_ifg_paths = _compute_burst_interferograms(plan, swath, ion_shift_map=ion_shift_map)
            swath_record["burst_ifg_paths"] = [str(path) for path in burst_ifg_paths]
            swath_record["burst_count"] = len(burst_ifg_paths)
        out.append(swath_record)
    return {"stage": "p2", "status": "ok", "swaths": out}


def _run_local_tops_backend_for_swath(
    plan: dict[str, Any],
    swath: str,
    *,
    stop_after_stage: str,
) -> dict[str, Any]:
    master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
    master_bundle = _load_tops_bundle(master_manifest_path)
    slave_bundle = _load_tops_bundle(slave_manifest_path)
    selected_indices = _selected_burst_indices_for_swath(
        plan,
        swath,
        len(master_bundle.get("tops", {}).get("bursts", [])),
    )
    master_bursts = _burst_infos_from_bundle(master_bundle, selected_indices, zero_based_index=True)
    slave_bursts = _burst_infos_from_bundle(slave_bundle, selected_indices, zero_based_index=True)
    overlaps = _overlap_infos_from_bundle(master_bundle, selected_indices)
    options = plan.get("options", {})
    output_root = Path(plan["plan_path"]).parent / str(swath)
    result = strip_insar2.process_strip_insar2(
        str(master_manifest_path),
        str(slave_manifest_path),
        output_root=str(output_root),
        gpu_mode=str(options.get("gpu_mode", "auto") or "auto"),
        gpu_id=int(options.get("gpu_id", 0) or 0),
        unwrap_method=str(options.get("unwrap_method", "icu") or "icu"),
        range_looks=int(options.get("range_looks", 1) or 1),
        azimuth_looks=int(options.get("azimuth_looks", 1) or 1),
        resolution_meters=float(options.get("resolution", 20.0) or 20.0),
        block_rows=options.get("block_rows"),
        dem_path=options.get("dem"),
        dem_cache_dir=options.get("dem_cache_dir"),
        dem_margin_deg=float(options.get("dem_margin_deg", 0.2) or 0.2),
        no_kml=bool(options.get("no_kml", False)),
        stop_after_stage=stop_after_stage,
        tops_mode=True,
        master_bursts=master_bursts,
        slave_bursts=slave_bursts,
        overlaps=overlaps,
        use_topo_flattening=True,
        extra_esd_cycles=float(options.get("extra_esd_cycles", 0.0) or 0.0),
        esd_coherence_threshold=float(options.get("esd_coherence_threshold", 0.85) or 0.85),
        do_burst_seam_repair=True,
    )
    return {
        "swath": swath,
        "status": "ok",
        "backend": "strip_insar2-local-tops",
        "stop_after_stage": stop_after_stage,
        "master_manifest": str(master_manifest_path),
        "slave_manifest": str(slave_manifest_path),
        "burst_count": len(master_bursts),
        "overlap_count": len(overlaps),
        "pair_name": result.get("pair_name"),
        "pair_dir": result.get("pair_dir"),
        "exports": result.get("exports", {}),
        "stage_backends": result.get("stage_backends", {}),
        "fallback_reasons": result.get("fallback_reasons", {}),
        "output_paths": result.get("output_paths", {}),
        "stopped_after_stage": result.get("stopped_after_stage"),
    }


def _estimate_esd_offset_for_swath(p1_swath: dict[str, Any]) -> float:
    backend = p1_swath.get("overlap_backend_execution", {})
    pairs = backend.get("pairs", []) if isinstance(backend, dict) else []
    vals: list[float] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        try:
            v = float(item.get("estimated_azimuth_offset_pixels", 0.0))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return 0.0
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _burst_window_from_tasks(p0_swath: dict[str, Any], selected_indices: set[int]) -> tuple[int, int, int, int]:
    selected = [b for b in p0_swath.get("burst_tasks", []) if int(b.get("burst_index", 0)) in selected_indices]
    if not selected:
        raise ValueError("no selected burst tasks for backend execution")
    row0 = min(int(b["master"]["line_offset"]) for b in selected)
    col0 = 0
    cols = int(selected[0]["master"]["cols"])
    row1 = max(int(b["master"]["line_offset"]) + int(b["master"]["rows"]) for b in selected)
    rows = max(1, row1 - row0)
    return row0, col0, rows, cols


def _run_strip_backend_for_swath(
    plan: dict[str, Any],
    swath: str,
    p0_swath: dict[str, Any],
    ifg_tasks: list[dict[str, Any]],
    esd_azimuth_offset_px: float = 0.0,
) -> dict[str, Any]:
    master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
    selected = set(int(t.get("burst_index", 0)) for t in ifg_tasks)
    row0, col0, rows, cols = _burst_window_from_tasks(p0_swath, selected)
    output_root = Path(plan["plan_path"]).parent / swath / "strip_backend"
    output_root.mkdir(parents=True, exist_ok=True)

    unwrap_method = str(plan.get("options", {}).get("unwrap_method", "icu")).lower()
    if unwrap_method == "dolphin":
        unwrap_method = "phass"
    gpu_mode = str(plan.get("options", {}).get("gpu_mode", "auto")).lower()
    if gpu_mode == "auto":
        gpu_mode = "gpu"
    if gpu_mode not in {"gpu", "cpu"}:
        gpu_mode = "gpu"
    resolution = float(plan.get("options", {}).get("resolution", 20.0))
    dem = plan.get("options", {}).get("dem")
    cmd = [
        "python3",
        str(_repo_root() / "scripts" / "strip_insar2.py"),
        str(master_manifest_path),
        str(slave_manifest_path),
        "--out",
        str(output_root),
        "--gpu",
        gpu_mode,
        "--gid",
        str(int(plan.get("options", {}).get("gpu_id", 0))),
        "--unwrap",
        unwrap_method,
        "--res",
        str(resolution),
        "--stop-after",
        "p2",
        "--esd-azimuth-offset-px",
        str(float(esd_azimuth_offset_px)),
        "--window",
        str(row0),
        str(col0),
        str(rows),
        str(cols),
    ]
    if dem:
        cmd.extend(["--dem", str(dem)])

    backend_timeout_s = int(plan.get("options", {}).get("backend_timeout_seconds", 1800))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(60, backend_timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        record = {
            "command": cmd,
            "window": {"row0": row0, "col0": col0, "rows": rows, "cols": cols},
            "returncode": -1,
            "output_root": str(output_root),
            "error": f"strip_insar2 backend timeout after {max(60, backend_timeout_s)} seconds",
        }
        if exc.stdout:
            record["stdout_tail"] = str(exc.stdout)[-4000:]
        if exc.stderr:
            record["stderr_tail"] = str(exc.stderr)[-4000:]
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    record: dict[str, Any] = {
        "command": cmd,
        "window": {"row0": row0, "col0": col0, "rows": rows, "cols": cols},
        "returncode": int(proc.returncode),
        "output_root": str(output_root),
    }
    if proc.stdout.strip():
        record["stdout_tail"] = proc.stdout.strip()[-4000:]
    if proc.stderr.strip():
        record["stderr_tail"] = proc.stderr.strip()[-4000:]
    if proc.returncode != 0:
        raise RuntimeError(f"strip_insar2 backend failed for swath {swath}")
    record["result"] = _extract_last_json_object(proc.stdout)
    if record["result"] is None:
        record["result"] = _load_latest_json_from_dir(output_root)
    return record


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    src = (text or "").strip()
    if not src:
        return None
    try:
        parsed = json.loads(src)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    end = src.rfind("}")
    if end < 0:
        return None
    depth = 0
    start = -1
    for i in range(end, -1, -1):
        ch = src[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        return None
    candidate = src[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _load_latest_json_from_dir(root: Path) -> dict[str, Any] | None:
    json_files = []
    for path in root.rglob("*.json"):
        try:
            mtime = path.stat().st_mtime
        except Exception:
            continue
        json_files.append((mtime, path))
    if not json_files:
        return None
    json_files.sort(reverse=True)
    for _, path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["_source_json"] = str(path)
                return data
        except Exception:
            continue
    return None


def _should_execute_stages(plan: dict[str, Any]) -> bool:
    return bool(plan.get("options", {}).get("execute_stages", True))


def _swath_root(plan: dict[str, Any], swath: str) -> Path:
    return Path(plan["plan_path"]).parent / swath


def _stage_output_dir(plan: dict[str, Any], swath: str, stage: str) -> Path:
    out = _swath_root(plan, swath) / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plan_option(plan: dict[str, Any], key: str, default: Any = None) -> Any:
    return plan.get("options", {}).get(key, default)


def _burst_indices_from_selection(bundle: dict[str, Any], selected_indices: set[int] | None = None) -> list[Any]:
    return _burst_infos_from_bundle(bundle, selected_indices)


def _read_complex_slc_window(slc_path: str, burst: Any) -> np.ndarray:
    slc = strip_insar2._open_slc_as_complex(slc_path)
    row0 = int(getattr(burst, "line_offset", 0)) + int(getattr(burst, "first_valid_line", 0))
    row1 = row0 + int(getattr(burst, "num_valid_lines", getattr(burst, "number_of_lines", 0)))
    col0 = int(getattr(burst, "first_valid_sample", 0))
    col1 = col0 + int(getattr(burst, "num_valid_samples", getattr(burst, "number_of_samples", 0)))
    row0 = max(0, row0)
    col0 = max(0, col0)
    row1 = min(slc.shape[0], max(row0, row1))
    col1 = min(slc.shape[1], max(col0, col1))
    return np.asarray(slc[row0:row1, col0:col1], dtype=np.complex64)


def _read_float_slc_window(slc_path: str, burst: Any) -> np.ndarray:
    data = _read_complex_slc_window(slc_path, burst)
    return np.abs(data).astype(np.float32)


def _read_complex_slc_valid_window(slc_path: str, burst: Any) -> np.ndarray:
    ds = gdal.Open(str(slc_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    try:
        row0, row1, col0, col1 = _burst_valid_window_from_info(burst)
        row0 = max(0, row0)
        col0 = max(0, col0)
        rows = max(0, row1 - row0)
        cols = max(0, col1 - col0)
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0), dtype=np.complex64)
        band1 = ds.GetRasterBand(1)
        if ds.RasterCount >= 2:
            band2 = ds.GetRasterBand(2)
            real = band1.ReadAsArray(col0, row0, cols, rows)
            imag = band2.ReadAsArray(col0, row0, cols, rows)
            if real is None or imag is None:
                return np.zeros((0, 0), dtype=np.complex64)
            return (np.asarray(real, dtype=np.float32) + 1j * np.asarray(imag, dtype=np.float32)).astype(np.complex64)
        arr = band1.ReadAsArray(col0, row0, cols, rows)
        if arr is None:
            return np.zeros((0, 0), dtype=np.complex64)
        return np.asarray(arr, dtype=np.complex64)
    finally:
        ds = None


def _read_complex_slc_bounds(slc_path: str, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    ds = gdal.Open(str(slc_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    try:
        row0 = max(0, int(row0))
        col0 = max(0, int(col0))
        row1 = max(row0, int(row1))
        col1 = max(col0, int(col1))
        rows = min(int(ds.RasterYSize) - row0, row1 - row0)
        cols = min(int(ds.RasterXSize) - col0, col1 - col0)
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0), dtype=np.complex64)
        band1 = ds.GetRasterBand(1)
        if ds.RasterCount >= 2:
            band2 = ds.GetRasterBand(2)
            real = band1.ReadAsArray(col0, row0, cols, rows)
            imag = band2.ReadAsArray(col0, row0, cols, rows)
            if real is None or imag is None:
                return np.zeros((0, 0), dtype=np.complex64)
            return (np.asarray(real, dtype=np.float32) + 1j * np.asarray(imag, dtype=np.float32)).astype(np.complex64)
        arr = band1.ReadAsArray(col0, row0, cols, rows)
        if arr is None:
            return np.zeros((0, 0), dtype=np.complex64)
        return np.asarray(arr, dtype=np.complex64)
    finally:
        ds = None


def adjustValidLineSample(reference: Any, secondary: Any) -> Any:
    reference_lastValidLine = int(getattr(reference, "first_valid_line", getattr(reference, "firstValidLine", 0))) + int(
        getattr(reference, "num_valid_lines", getattr(reference, "numValidLines", getattr(reference, "number_of_lines", 0)))
    ) - 1
    reference_lastValidSample = int(getattr(reference, "first_valid_sample", getattr(reference, "firstValidSample", 0))) + int(
        getattr(reference, "num_valid_samples", getattr(reference, "numValidSamples", getattr(reference, "number_of_samples", 0)))
    ) - 1
    secondary_lastValidLine = int(getattr(secondary, "first_valid_line", getattr(secondary, "firstValidLine", 0))) + int(
        getattr(secondary, "num_valid_lines", getattr(secondary, "numValidLines", getattr(secondary, "number_of_lines", 0)))
    ) - 1
    secondary_lastValidSample = int(getattr(secondary, "first_valid_sample", getattr(secondary, "firstValidSample", 0))) + int(
        getattr(secondary, "num_valid_samples", getattr(secondary, "numValidSamples", getattr(secondary, "number_of_samples", 0)))
    ) - 1

    igram_lastValidLine = min(reference_lastValidLine, secondary_lastValidLine)
    igram_lastValidSample = min(reference_lastValidSample, secondary_lastValidSample)

    first_line = max(
        int(getattr(reference, "first_valid_line", getattr(reference, "firstValidLine", 0))),
        int(getattr(secondary, "first_valid_line", getattr(secondary, "firstValidLine", 0))),
    )
    first_sample = max(
        int(getattr(reference, "first_valid_sample", getattr(reference, "firstValidSample", 0))),
        int(getattr(secondary, "first_valid_sample", getattr(secondary, "firstValidSample", 0))),
    )

    if hasattr(reference, "first_valid_line"):
        reference.first_valid_line = first_line
        reference.first_valid_sample = first_sample
        reference.num_valid_lines = igram_lastValidLine - first_line + 1
        reference.num_valid_samples = igram_lastValidSample - first_sample + 1
    else:
        reference.firstValidLine = first_line
        reference.firstValidSample = first_sample
        reference.numValidLines = igram_lastValidLine - first_line + 1
        reference.numValidSamples = igram_lastValidSample - first_sample + 1
    return reference


def _load_float_offset_raster(path: Path) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open offset raster: {path}")
    try:
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        if arr is None:
            raise RuntimeError(f"failed to read offset raster: {path}")
        return np.asarray(arr, dtype=np.float32)
    finally:
        ds = None


def _find_coregdir(plan: dict[str, Any], swath: str) -> Path | None:
    candidates: list[Path] = []
    configured = _plan_option(plan, "coregdir", None) or _plan_option(plan, "coreg_dir", None)
    if configured:
        candidates.append(Path(str(configured)) / swath)
        candidates.append(Path(str(configured)))
    swath_root = _swath_root(plan, swath)
    candidates.extend([
        swath_root / "p1",
        swath_root / "p1_dense_match",
        swath_root / "p1_geo2rdr_offsets",
        swath_root,
    ])
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("range_*.off")) and any(cand.glob("azimuth_*.off")):
            return cand
    return None


def _load_coreg_offsets(coregdir: Path | None, burst_index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if coregdir is None:
        return None, None
    range_candidates = [
        coregdir / f"range_{burst_index:02d}.off",
        coregdir / f"range_{burst_index:03d}.off",
        coregdir / f"range_{burst_index}.off",
    ]
    az_candidates = [
        coregdir / f"azimuth_{burst_index:02d}.off",
        coregdir / f"azimuth_{burst_index:03d}.off",
        coregdir / f"azimuth_{burst_index}.off",
    ]
    range_path = next((p for p in range_candidates if p.is_file()), None)
    az_path = next((p for p in az_candidates if p.is_file()), None)
    range_off = _load_float_offset_raster(range_path) if range_path else None
    az_off = _load_float_offset_raster(az_path) if az_path else None
    return range_off, az_off


def _resample_complex_with_offsets(slave: np.ndarray, range_off: np.ndarray | None, az_off: np.ndarray | None) -> np.ndarray:
    src = np.asarray(slave, dtype=np.complex64)
    rows, cols = src.shape[:2]
    if rows <= 0 or cols <= 0:
        return src.copy()
    if range_off is None and az_off is None:
        return src.copy()

    rg = np.zeros((rows, cols), dtype=np.float32) if range_off is None else np.asarray(range_off, dtype=np.float32)
    az = np.zeros((rows, cols), dtype=np.float32) if az_off is None else np.asarray(az_off, dtype=np.float32)
    rows = min(rows, rg.shape[0], az.shape[0])
    cols = min(cols, rg.shape[1], az.shape[1])
    src = src[:rows, :cols]
    rg = rg[:rows, :cols]
    az = az[:rows, :cols]

    yy, xx = np.meshgrid(np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij")
    src_y = yy - az
    src_x = xx - rg
    finite = np.isfinite(src_y) & np.isfinite(src_x)
    y0 = np.zeros((rows, cols), dtype=np.int32)
    x0 = np.zeros((rows, cols), dtype=np.int32)
    y0[finite] = np.floor(src_y[finite]).astype(np.int32)
    x0[finite] = np.floor(src_x[finite]).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (src_y - y0).astype(np.float32)
    wx = (src_x - x0).astype(np.float32)

    valid = finite & (y0 >= 0) & (x0 >= 0) & (y1 < rows) & (x1 < cols)
    out = np.zeros((rows, cols), dtype=np.complex64)
    if not np.any(valid):
        return out

    flat = src.reshape(-1)
    idx00 = y0[valid] * cols + x0[valid]
    idx01 = y0[valid] * cols + x1[valid]
    idx10 = y1[valid] * cols + x0[valid]
    idx11 = y1[valid] * cols + x1[valid]
    w00 = (1.0 - wy[valid]) * (1.0 - wx[valid])
    w01 = (1.0 - wy[valid]) * wx[valid]
    w10 = wy[valid] * (1.0 - wx[valid])
    w11 = wy[valid] * wx[valid]
    out[valid] = (
        flat[idx00] * w00 + flat[idx01] * w01 + flat[idx10] * w10 + flat[idx11] * w11
    ).astype(np.complex64)
    return out

    flat = src.reshape(-1)
    idx00 = y0[valid] * cols + x0[valid]
    idx01 = y0[valid] * cols + x1[valid]
    idx10 = y1[valid] * cols + x0[valid]
    idx11 = y1[valid] * cols + x1[valid]
    w00 = (1.0 - wy[valid]) * (1.0 - wx[valid])
    w01 = (1.0 - wy[valid]) * wx[valid]
    w10 = wy[valid] * (1.0 - wx[valid])
    w11 = wy[valid] * wx[valid]
    out[valid] = (
        flat[idx00] * w00 + flat[idx01] * w01 + flat[idx10] * w10 + flat[idx11] * w11
    ).astype(np.complex64)
    return out


def _load_burst_products_for_swath(
    plan: dict[str, Any],
    swath: str,
    selected_indices: set[int],
) -> tuple[list[Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
    master_bundle = _load_tops_bundle(master_manifest_path)
    slave_bundle = _load_tops_bundle(slave_manifest_path)
    master_slc = _resolve_manifest_data_path(master_manifest_path, master_bundle["manifest"].get("slc", {}).get("path"))
    slave_slc = _resolve_manifest_data_path(slave_manifest_path, slave_bundle["manifest"].get("slc", {}).get("path"))
    if not master_slc or not slave_slc:
        raise ValueError(f"missing SLC path for swath {swath}")

    bursts = _burst_infos_from_bundle(master_bundle, selected_indices)
    burst_ifgs: list[np.ndarray] = []
    burst_cohs: list[np.ndarray] = []
    burst_amps: list[np.ndarray] = []
    for burst in bursts:
        m_win = _read_complex_slc_window(master_slc, burst)
        s_win = _read_complex_slc_window(slave_slc, burst)
        rows = min(m_win.shape[0], s_win.shape[0])
        cols = min(m_win.shape[1], s_win.shape[1])
        if rows <= 0 or cols <= 0:
            raise RuntimeError(f"empty burst window for swath {swath}")
        m_win = m_win[:rows, :cols]
        s_win = s_win[:rows, :cols]
        ifg = (m_win * np.conj(s_win)).astype(np.complex64)
        amp = (0.5 * (np.abs(m_win) + np.abs(s_win))).astype(np.float32)
        coh = np.clip(np.abs(ifg) / (np.percentile(np.abs(ifg), 95) + 1.0e-6), 0.0, 1.0).astype(np.float32)
        burst_ifgs.append(ifg)
        burst_cohs.append(coh)
        burst_amps.append(amp)
    return bursts, burst_ifgs, burst_cohs, burst_amps


def _burst_valid_window_from_info(burst: Any) -> tuple[int, int, int, int]:
    row0 = int(getattr(burst, "line_offset", 0)) + int(getattr(burst, "first_valid_line", 0))
    row1 = row0 + int(getattr(burst, "num_valid_lines", getattr(burst, "number_of_lines", 0)))
    col0 = int(getattr(burst, "first_valid_sample", 0))
    col1 = col0 + int(getattr(burst, "num_valid_samples", getattr(burst, "number_of_samples", 0)))
    return row0, row1, col0, col1


def _write_float_envi(path: Path, data: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = data.shape
    ds = gdal.GetDriverByName("ENVI").Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    if ds is None:
        raise RuntimeError(f"failed to create ENVI raster: {path}")
    ds.GetRasterBand(1).WriteArray(np.asarray(data, dtype=np.float32))
    ds.FlushCache()
    ds = None
    return str(path)


def _boxcar_mean2d(array: np.ndarray, window_size: int = 5) -> np.ndarray:
    size = max(1, int(window_size))
    if size == 1:
        return np.asarray(array)
    if size % 2 == 0:
        size += 1
    arr = np.asarray(array)
    pad = size // 2
    if np.iscomplexobj(arr):
        real = _boxcar_mean2d(np.asarray(np.real(arr), dtype=np.float32), size)
        imag = _boxcar_mean2d(np.asarray(np.imag(arr), dtype=np.float32), size)
        return real.astype(np.float32) + 1j * imag.astype(np.float32)
    padded = np.pad(np.asarray(arr, dtype=np.float32), ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    summed = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return (summed / float(size * size)).astype(np.float32)


def _multilook_mean_isce2_style(arr: np.ndarray, azimuth_looks: int, range_looks: int) -> np.ndarray:
    az = max(1, int(azimuth_looks or 1))
    rg = max(1, int(range_looks or 1))
    data = np.asarray(arr)
    if az == 1 and rg == 1:
        return data
    rows_out = data.shape[0] // az
    cols_out = data.shape[1] // rg
    if rows_out < 1 or cols_out < 1:
        raise ValueError(f"multilook factors too large for shape {data.shape}: azimuth_looks={az}, range_looks={rg}")
    trimmed = data[: rows_out * az, : cols_out * rg]
    reshaped = trimmed.reshape(rows_out, az, cols_out, rg, *trimmed.shape[2:])
    return reshaped.mean(axis=(1, 3))


def _shift_complex_azimuth(arr: np.ndarray, az_shift_px: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.complex64)
    shift = float(az_shift_px or 0.0)
    if abs(shift) < 1.0e-6:
        return data
    rows = data.shape[0]
    if rows <= 1:
        return data.copy()
    y = np.arange(rows, dtype=np.float32)
    src_y = y - np.float32(shift)
    src0 = np.floor(src_y).astype(np.int32)
    frac = (src_y - src0).astype(np.float32)
    src1 = src0 + 1
    valid = (src0 >= 0) & (src1 < rows)
    out = np.zeros_like(data)
    if np.any(valid):
        idx0 = src0[valid]
        idx1 = src1[valid]
        w = frac[valid][:, None].astype(np.float32)
        out[valid, :] = (1.0 - w) * data[idx0, :] + w * data[idx1, :]
    return out


def _estimate_esd_local_frequency(ifg: np.ndarray, deramped: bool = False) -> np.ndarray:
    """估算 ESD 局部频率。

    Args:
        ifg: 输入干涉图
        deramped: 是否已经进行了 deramp 处理。如果是 True，则直接计算相位梯度；
                 如果是 False，需要先去除 TOPS deramp 项才能得到正确的频率估计。

    Returns:
        频率估计（弧度/像素）
    """
    phase = np.unwrap(np.angle(np.asarray(ifg, dtype=np.complex64)), axis=0)
    if phase.shape[0] <= 1:
        return np.zeros_like(phase, dtype=np.float32)
    freq = np.gradient(phase, axis=0).astype(np.float32)
    try:
        freq = _boxcar_mean2d(freq, 5).astype(np.float32)
    except Exception:
        pass
    freq[np.abs(freq) < 1.0e-6] = np.nan
    return freq


def _apply_tops_deramp(slc: np.ndarray, doppler: np.ndarray | None = None, frequency: float = 0.0) -> np.ndarray:
    """对 TOPS SLC 进行 deramp 处理。

    当前为简化实现，仅返回原始数据。完整的 TOPS deramp 需要：
    1. 提取 burst 中心时刻的 Doppler 频率
    2. 计算 azimuth 线性相位项 exp(-j * 2pi * fDoppler * t)
    3. 在整个 burst 窗口内应用该相位项

    Args:
        slc: 输入的复数 SLC 数据
        doppler: Doppler 频率数组（可选），如果为 None 则使用零多普勒
        frequency: 雷达中心频率（Hz），用于计算相位

    Returns:
        deramp 后的 SLC
    """
    # 当前简化实现：直接返回原始数据
    # TODO: 实现完整的 TOPS deramp/reramp
    #       需要从 burst metadata 中提取：
    #       - burst 中心时刻的 Doppler Centroid
    #       - azimuth FM Rate
    #       - 然后构建线性相位项 exp(-j * 2pi * (fdc + fdot * t) * t)
    return np.asarray(slc, dtype=np.complex64)


def _apply_tops_reramp(slc: np.ndarray, doppler: np.ndarray | None = None, frequency: float = 0.0) -> np.ndarray:
    """对 TOPS SLC 进行 reramp 处理（deramp 的逆操作）。

    Args:
        slc: 输入的复数 SLC 数据
        doppler: Doppler 频率数组（可选），如果为 None 则使用零多普勒
        frequency: 雷达中心频率（Hz），用于计算相位

    Returns:
        reramp 后的 SLC
    """
    # 当前简化实现：直接返回原始数据
    return np.asarray(slc, dtype=np.complex64)


def _estimate_burst_coherence(master_win: np.ndarray, slave_win: np.ndarray, window_size: int = 5) -> np.ndarray:
    master = np.asarray(master_win, dtype=np.complex64)
    slave = np.asarray(slave_win, dtype=np.complex64)
    if master.shape != slave.shape:
        rows = min(master.shape[0], slave.shape[0])
        cols = min(master.shape[1], slave.shape[1])
        master = master[:rows, :cols]
        slave = slave[:rows, :cols]
    cross = master * np.conj(slave)
    cross_mean = _boxcar_mean2d(cross, window_size)
    pwr_m = _boxcar_mean2d(np.abs(master) ** 2, window_size).astype(np.float32)
    pwr_s = _boxcar_mean2d(np.abs(slave) ** 2, window_size).astype(np.float32)
    denom = np.sqrt(np.maximum(pwr_m * pwr_s, 1.0e-9))
    coh = np.clip(np.abs(cross_mean) / denom, 0.0, 1.0)
    coh[~np.isfinite(coh)] = 0.0
    return coh.astype(np.float32)


def _compute_esd_spectral_diversity(
    overlap_ifgs: list[np.ndarray],
    overlap_cohs: list[np.ndarray],
    *,
    azimuth_looks: int = 5,
    range_looks: int = 15,
    coherence_threshold: float = 0.85,
    extra_esd_cycles: float = 0.0,
) -> tuple[float, float, float, np.ndarray]:
    """基于频谱多样性估算 ESD 偏移量。"""
    if len(overlap_ifgs) != len(overlap_cohs):
        raise ValueError("overlap_ifgs 和 overlap_cohs 长度不一致")

    extra_offset = float(extra_esd_cycles) * 2.0 * np.pi
    all_offsets: list[np.ndarray] = []

    for ifg, cor in zip(overlap_ifgs, overlap_cohs, strict=False):
        ifg_arr = np.asarray(ifg, dtype=np.complex64)
        cor_arr = np.asarray(cor, dtype=np.float32)
        rows = min(ifg_arr.shape[0], cor_arr.shape[0])
        cols = min(ifg_arr.shape[1], cor_arr.shape[1])
        if rows <= 0 or cols <= 0:
            continue
        ifg_arr = ifg_arr[:rows, :cols]
        cor_arr = cor_arr[:rows, :cols]

        if azimuth_looks > 1 or range_looks > 1:
            ifg_arr = _multilook_mean_isce2_style(ifg_arr, azimuth_looks, range_looks).astype(np.complex64)
            cor_arr = _multilook_mean_isce2_style(cor_arr, azimuth_looks, range_looks).astype(np.float32)

        rows, cols = ifg_arr.shape[:2]
        if rows <= 0 or cols <= 0:
            continue

        freq = _estimate_esd_local_frequency(ifg_arr)

        phase = np.angle(ifg_arr).astype(np.float32)
        off = np.full((rows, cols), np.nan, dtype=np.float32)
        valid_freq = np.isfinite(freq) & (np.abs(freq) > 1.0e-6)
        off[valid_freq] = (phase[valid_freq] + extra_offset) / freq[valid_freq]

        mask = (np.abs(ifg_arr) > 0) & (cor_arr > float(coherence_threshold)) & np.isfinite(off)
        if np.any(mask):
            all_offsets.append(off[mask].astype(np.float32))

    if not all_offsets:
        raise Exception("Coherence threshold too strict. No points left for reliable ESD estimate")

    offsets = np.concatenate(all_offsets).astype(np.float64)
    return (
        float(np.median(offsets)),
        float(np.mean(offsets)),
        float(np.std(offsets)),
        offsets.astype(np.float32),
    )


def _compute_az_reference_offsets(bursts: list[Any]) -> list[list[int]]:
    max_index = max((int(getattr(burst, "burst_index", i + 1)) for i, burst in enumerate(bursts)), default=0)
    offsets: list[list[int]] = [[0, 0] for _ in range(max_index + 1)]
    if not bursts:
        return offsets

    tstart = _parse_utc_timestamp(getattr(bursts[0], "sensing_start", None))
    if tstart is None:
        for burst in bursts:
            burst_index = int(getattr(burst, "burst_index", 0))
            if burst_index <= 0 or burst_index >= len(offsets):
                continue
            first_valid_line = int(getattr(burst, "first_valid_line", 0))
            num_valid_lines = int(getattr(burst, "num_valid_lines", getattr(burst, "number_of_lines", 0)))
            offsets[burst_index] = [first_valid_line, first_valid_line + num_valid_lines]
        return offsets

    for burst in bursts:
        burst_index = int(getattr(burst, "burst_index", 0))
        if burst_index <= 0 or burst_index >= len(offsets):
            continue
        sensing_start = _parse_utc_timestamp(getattr(burst, "sensing_start", None))
        dt = float(getattr(burst, "azimuth_time_interval", 0.0) or 0.0)
        if sensing_start is None or dt <= 0.0:
            first_valid_line = int(getattr(burst, "first_valid_line", 0))
            num_valid_lines = int(getattr(burst, "num_valid_lines", getattr(burst, "number_of_lines", 0)))
            offsets[burst_index] = [first_valid_line, first_valid_line + num_valid_lines]
            continue
        soff = sensing_start + timedelta(seconds=float(getattr(burst, "first_valid_line", 0)) * dt)
        start = int(np.round((soff - tstart).total_seconds() / dt))
        end = start + int(getattr(burst, "num_valid_lines", getattr(burst, "number_of_lines", 0)))
        offsets[burst_index] = [start, end]
    return offsets


def _store_secondary_timing_correction(plan: dict[str, Any], swath: str, median_offset_px: float, bursts: list[Any]) -> float:
    if not bursts:
        return 0.0
    dt = float(getattr(bursts[0], "azimuth_time_interval", 0.0) or 0.0)
    correction_s = float(median_offset_px) * dt if dt > 0.0 else 0.0
    corrections = dict(plan.get("secondary_timing_correction", {}))
    corrections[str(swath)] = correction_s
    plan["secondary_timing_correction"] = corrections
    return correction_s


def _secondary_timing_correction_pixels(plan: dict[str, Any], swath: str) -> float:
    return float(dict(plan.get("secondary_timing_correction", {})).get(str(swath), 0.0) or 0.0)


def _load_burst_ifg_and_coh(burst_ifg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ifg = np.asarray(_load_array(burst_ifg_path), dtype=np.complex64)
    coh_path = burst_ifg_path.with_name("coherence.cor")
    coh = np.asarray(_load_array(coh_path), dtype=np.float32) if coh_path.is_file() else np.ones(ifg.shape, dtype=np.float32)
    rows = min(ifg.shape[0], coh.shape[0])
    cols = min(ifg.shape[1], coh.shape[1])
    return ifg[:rows, :cols].copy(), coh[:rows, :cols].copy()


def _radargrid_reference_from_bundle(bundle: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    radargrid = bundle.get("radargrid", {})
    acquisition = bundle.get("acquisition", {})
    c = 299792458.0
    starting_range = radargrid.get("startingRange", radargrid.get("starting_range"))
    if starting_range is None:
        range_time_first = radargrid.get("rangeTimeFirstPixel")
        starting_range = 0.5 * c * float(range_time_first) if range_time_first is not None else 0.0
    range_pixel_spacing = radargrid.get("columnSpacing", radargrid.get("rangePixelSpacing", 1.0))
    az_time_interval = acquisition.get("azimuthTimeInterval", radargrid.get("azimuthTimeInterval", 1.0))
    projection = radargrid.get("projection", "LOCAL_CS[\"RadarGrid\"]")
    return {
        "geotransform": (float(starting_range), float(range_pixel_spacing), 0.0, 0.0, 0.0, -float(az_time_interval)),
        "projection": projection,
        "width": int(shape[1]),
        "height": int(shape[0]),
    }


def _compute_burst_interferograms(
    plan: dict[str, Any],
    swath: str,
    ion_shift_map: dict[int, np.ndarray] | None = None,
) -> list[Path]:
    """逐 burst 生成干涉图和相干度，保存到 p2/burst_XXX。"""
    master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
    master_bundle = _load_tops_bundle(master_manifest_path)
    slave_bundle = _load_tops_bundle(slave_manifest_path)
    master_slc = _resolve_manifest_data_path(master_manifest_path, master_bundle["manifest"].get("slc", {}).get("path"))
    slave_slc = _resolve_manifest_data_path(slave_manifest_path, slave_bundle["manifest"].get("slc", {}).get("path"))
    if not master_slc or not slave_slc:
        raise ValueError(f"missing SLC path for swath {swath}")

    selected_indices = _selected_burst_indices_for_swath(
        plan,
        swath,
        len(master_bundle.get("tops", {}).get("bursts", [])),
    )
    bursts = _burst_infos_from_bundle(master_bundle, selected_indices)
    slave_bursts = {int(b.burst_index): b for b in _burst_infos_from_bundle(slave_bundle, selected_indices)}
    master_raw_bursts = {
        int(item.get("burstIndex", idx + 1)): item for idx, item in enumerate(master_bundle.get("tops", {}).get("bursts", []))
    }
    slave_raw_bursts = {
        int(item.get("burstIndex", idx + 1)): item for idx, item in enumerate(slave_bundle.get("tops", {}).get("bursts", []))
    }
    if not bursts:
        raise RuntimeError(f"swath {swath} has no selected bursts")

    stage_dir = _stage_output_dir(plan, swath, "p2")
    burst_ifg_paths: list[Path] = []
    coherence_window = int(_plan_option(plan, "burst_coherence_window", 5) or 5)
    range_looks = int(_plan_option(plan, "range_looks", 1) or 1)
    azimuth_looks = int(_plan_option(plan, "azimuth_looks", 1) or 1)
    coregdir = _find_coregdir(plan, swath)
    for burst in bursts:
        slave_burst = slave_bursts.get(int(burst.burst_index), burst)
        master_win = _read_complex_slc_valid_window(master_slc, adjustValidLineSample(replace(burst), replace(slave_burst)))
        slave_win = _read_complex_slc_valid_window(slave_slc, adjustValidLineSample(replace(slave_burst), replace(burst)))
        range_off, az_off = _load_coreg_offsets(coregdir, int(burst.burst_index))
        if range_off is not None or az_off is not None:
            slave_win = _resample_complex_with_offsets(slave_win, range_off, az_off)
        timing_correction_s = float(dict(plan.get("secondary_timing_correction", {})).get(swath, 0.0) or 0.0)
        burst_dt = float(getattr(burst, "azimuth_time_interval", 0.0) or getattr(slave_burst, "azimuth_time_interval", 0.0) or 0.0)
        if timing_correction_s != 0.0 and burst_dt > 0.0:
            slave_win = _shift_complex_azimuth(slave_win, timing_correction_s / burst_dt)
        if ion_shift_map:
            ion_shift = ion_shift_map.get(int(burst.burst_index))
            if ion_shift is not None:
                valid_shift = np.asarray(ion_shift, dtype=np.float64)
                valid_shift = valid_shift[np.isfinite(valid_shift)]
                if valid_shift.size:
                    slave_win = _shift_complex_azimuth(slave_win, float(np.nanmedian(valid_shift)))
        rows = min(master_win.shape[0], slave_win.shape[0])
        cols = min(master_win.shape[1], slave_win.shape[1])
        if rows <= 0 or cols <= 0:
            raise RuntimeError(f"empty burst window for swath {swath}")
        master_win = np.asarray(master_win[:rows, :cols], dtype=np.complex64)
        slave_win = np.asarray(slave_win[:rows, :cols], dtype=np.complex64)
        ifg = (master_win * np.conj(slave_win)).astype(np.complex64)
        slave_raw = slave_raw_bursts.get(int(burst.burst_index), {})
        rg_pix = float(slave_raw.get("rangePixelSize", master_bundle.get("acquisition", {}).get("rangePixelSize", 0.0)) or 0.0)
        wl = float(slave_raw.get("radarWavelength", slave_bundle.get("acquisition", {}).get("radarWavelength", 0.0)) or 0.0)
        if range_off is not None and rg_pix > 0.0 and wl > 0.0:
            fact = 4.0 * np.pi * float(rg_pix) / float(wl)
            phs = np.exp((-1j * fact) * np.asarray(range_off[:rows, :cols], dtype=np.float32))
            ifg *= phs.astype(np.complex64)
        coh = _estimate_burst_coherence(master_win, slave_win, coherence_window)
        if range_looks > 1 or azimuth_looks > 1:
            ifg = _multilook_mean_isce2_style(ifg, azimuth_looks, range_looks).astype(np.complex64)
            coh = _multilook_mean_isce2_style(coh, azimuth_looks, range_looks).astype(np.float32)

        burst_dir = stage_dir / f"burst_{int(getattr(burst, 'burst_index', 0)):03d}"
        burst_dir.mkdir(parents=True, exist_ok=True)
        ifg_path = burst_dir / "interferogram.int"
        coh_path = burst_dir / "coherence.cor"
        strip_insar2._write_complex_envi(ifg_path, ifg)
        _write_float_envi(coh_path, coh)
        (burst_dir / "burst_metadata.json").write_text(
            json.dumps(
                {
                    "burst_index": int(getattr(burst, "burst_index", 0)),
                    "valid_window": {
                        "row0": int(getattr(burst, "line_offset", 0)) + int(getattr(burst, "first_valid_line", 0)),
                        "row1": int(getattr(burst, "line_offset", 0)) + int(getattr(burst, "first_valid_line", 0)) + int(
                            getattr(burst, "num_valid_lines", getattr(burst, "number_of_lines", 0))
                        ),
                        "col0": int(getattr(burst, "first_valid_sample", 0)),
                        "col1": int(getattr(burst, "first_valid_sample", 0)) + int(
                            getattr(burst, "num_valid_samples", getattr(burst, "number_of_samples", 0))
                        ),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        burst_ifg_paths.append(ifg_path)
    return burst_ifg_paths


def _merge_bursts_isce2_style(
    burst_ifg_paths: list[Path],
    bursts: list[Any],
    method: str = "avg",
    overlap_pairs: list[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按 ISCE2 的 burst 级合并方式拼接干涉图。"""
    if not burst_ifg_paths or not bursts:
        raise ValueError("empty burst interferograms or metadata")
    if len(burst_ifg_paths) != len(bursts):
        raise ValueError("burst interferogram count does not match burst metadata")

    burst_ifgs: list[np.ndarray] = []
    burst_cohs: list[np.ndarray] = []
    for path in burst_ifg_paths:
        ifg, coh = _load_burst_ifg_and_coh(Path(path))
        burst_ifgs.append(ifg)
        burst_cohs.append(coh)

    az_reference_offsets = _compute_az_reference_offsets(bursts)
    merged_ifg, merged_coh = strip_insar2._merge_tops_burst_interferograms(
        burst_ifgs,
        burst_cohs,
        bursts,
        overlap_pairs=[
            (
                int(item.get("previous_burst_index", 0)),
                int(item.get("next_burst_index", 0)),
                int(item.get("estimated_overlap_lines", 0)),
            )
            for item in (overlap_pairs or [])
        ] or None,
        method=method if method in {"top", "bot", "avg"} else "avg",
        az_reference_offsets=az_reference_offsets,
    )
    if plan is not None:
        range_looks = int(_plan_option(plan, "range_looks", 1) or 1)
        azimuth_looks = int(_plan_option(plan, "azimuth_looks", 1) or 1)
        if range_looks > 1 or azimuth_looks > 1:
            merged_ifg = _multilook_mean_isce2_style(np.asarray(merged_ifg, dtype=np.complex64), azimuth_looks, range_looks)
            merged_coh = _multilook_mean_isce2_style(np.asarray(merged_coh, dtype=np.float32), azimuth_looks, range_looks)
    return np.asarray(merged_ifg, dtype=np.complex64), np.asarray(merged_coh, dtype=np.float32)


def _estimate_esd_offsets_from_overlaps(
    bursts: list[Any],
    overlap_pairs: list[dict[str, Any]],
    burst_ifgs: list[np.ndarray],
    extra_cycles: float = 0.0,
) -> list[float]:
    offsets = [0.0] * len(bursts)
    if len(bursts) <= 1 or not overlap_pairs:
        return offsets

    burst_index_to_pos = {int(getattr(b, "burst_index", i + 1)): i for i, b in enumerate(bursts)}
    cumulative = 0.0
    pair_offsets: dict[tuple[int, int], float] = {}
    for pair in overlap_pairs:
        prev_idx = int(pair.get("previous_burst_index", 0))
        next_idx = int(pair.get("next_burst_index", 0))
        if prev_idx <= 0 or next_idx <= 0:
            continue
        prev_pos = burst_index_to_pos.get(prev_idx)
        next_pos = burst_index_to_pos.get(next_idx)
        if prev_pos is None or next_pos is None:
            continue
        ifg_prev = burst_ifgs[prev_pos]
        ifg_next = burst_ifgs[next_pos]
        rows = min(ifg_prev.shape[0], ifg_next.shape[0])
        cols = min(ifg_prev.shape[1], ifg_next.shape[1])
        if rows <= 0 or cols <= 0:
            continue
        overlap = ifg_prev[:rows, :cols] * np.conj(ifg_next[:rows, :cols])
        phase = np.angle(np.mean(overlap[np.isfinite(overlap)])) if np.any(np.isfinite(overlap)) else 0.0
        pair_offsets[(prev_idx, next_idx)] = float((phase + (extra_cycles * 2.0 * np.pi)) / (2.0 * np.pi))

    if not pair_offsets:
        return offsets
    for pos, burst in enumerate(bursts[1:], start=1):
        prev_idx = int(getattr(bursts[pos - 1], "burst_index", pos))
        next_idx = int(getattr(burst, "burst_index", pos + 1))
        if (prev_idx, next_idx) in pair_offsets:
            cumulative += pair_offsets[(prev_idx, next_idx)]
        offsets[pos] = cumulative
    return offsets


def _merge_burst_float_layers(
    burst_layers: list[np.ndarray],
    bursts: list[Any],
    overlap_pairs: list[dict[str, Any]] | None = None,
    *,
    method: str = "avg",
) -> np.ndarray:
    if not burst_layers or not bursts:
        raise ValueError("empty burst layers")
    if len(burst_layers) != len(bursts):
        raise ValueError("burst layers and metadata lengths do not match")
    merged_ifg, _ = strip_insar2._merge_tops_burst_interferograms(
        [np.asarray(layer, dtype=np.complex64) for layer in burst_layers],
        [np.ones_like(layer, dtype=np.float32) for layer in burst_layers],
        bursts,
        overlap_pairs=[
            (int(item.get("previous_burst_index", 0)), int(item.get("next_burst_index", 0)), int(item.get("estimated_overlap_lines", 0)))
            for item in (overlap_pairs or [])
        ],
        method=method,
    )
    return np.asarray(np.real(merged_ifg), dtype=np.float32)


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(array)
    out_rows, out_cols = int(shape[0]), int(shape[1])
    if arr.shape == (out_rows, out_cols):
        return arr.copy()
    row_idx = np.clip(np.rint(np.linspace(0, max(0, arr.shape[0] - 1), out_rows)).astype(int), 0, max(0, arr.shape[0] - 1))
    col_idx = np.clip(np.rint(np.linspace(0, max(0, arr.shape[1] - 1), out_cols)).astype(int), 0, max(0, arr.shape[1] - 1))
    return arr[np.ix_(row_idx, col_idx)]


def _geocode_to_reference(
    array: np.ndarray,
    reference: dict[str, Any],
    output_path: Path,
) -> str:
    rows = int(reference.get("height", array.shape[0]))
    cols = int(reference.get("width", array.shape[1]))
    resampled = _resize_nearest(np.asarray(array, dtype=np.float32), (rows, cols)).astype(np.float32)
    return _write_simple_geotiff(output_path, resampled, reference)


def _safe_load_stage_outputs(plan: dict[str, Any], stage: str, swath: str) -> dict[str, Any]:
    return _stage_swath_index(plan, stage).get(swath, {})


def _burst_infos_from_bundle(
    bundle: dict[str, Any],
    selected_indices: set[int] | None = None,
    *,
    zero_based_index: bool = False,
) -> list[Any]:
    bursts: list[Any] = []
    for idx, burst in enumerate(bundle.get("tops", {}).get("bursts", []), start=1):
        burst_index = int(burst.get("burstIndex", idx))
        if selected_indices is not None and burst_index not in selected_indices:
            continue
        bursts.append(
            strip_insar2.TopsBurstInfo(
                burst_index=(burst_index - 1 if zero_based_index else burst_index),
                line_offset=int(burst.get("lineOffset", 0)),
                number_of_lines=int(burst.get("numberOfLines", burst.get("numberOfRows", 0))),
                number_of_samples=int(burst.get("numberOfSamples", burst.get("numberOfColumns", 0))),
                first_valid_line=int(burst.get("firstValidLine", 0)),
                num_valid_lines=int(burst.get("numValidLines", burst.get("numberOfLines", 0))),
                first_valid_sample=int(burst.get("firstValidSample", 0)),
                num_valid_samples=int(burst.get("numValidSamples", burst.get("numberOfSamples", 0))),
                sensing_start=burst.get("sensingStart"),
                azimuth_time_interval=float(burst.get("azimuthTimeInterval", 0.0) or 0.0) or None,
                radar_wavelength=float(burst.get("radarWavelength", 0.0) or 0.0) or None,
            )
        )
    return bursts


def _coerce_tops_burst_info(burst: Any) -> Any:
    if isinstance(burst, strip_insar2.TopsBurstInfo):
        return burst
    if isinstance(burst, dict):
        return strip_insar2.TopsBurstInfo(
            burst_index=int(burst.get("burst_index", burst.get("burstIndex", 0)) or 0),
            line_offset=int(burst.get("line_offset", burst.get("lineOffset", 0)) or 0),
            number_of_lines=int(burst.get("number_of_lines", burst.get("numberOfLines", burst.get("numberOfRows", 0))) or 0),
            number_of_samples=int(burst.get("number_of_samples", burst.get("numberOfSamples", burst.get("numberOfColumns", 0))) or 0),
            first_valid_line=int(burst.get("first_valid_line", burst.get("firstValidLine", 0)) or 0),
            num_valid_lines=int(burst.get("num_valid_lines", burst.get("numValidLines", burst.get("numberOfLines", 0))) or 0),
            first_valid_sample=int(burst.get("first_valid_sample", burst.get("firstValidSample", 0)) or 0),
            num_valid_samples=int(burst.get("num_valid_samples", burst.get("numValidSamples", burst.get("numberOfSamples", 0))) or 0),
            sensing_start=burst.get("sensing_start", burst.get("sensingStart")),
            azimuth_time_interval=float(burst.get("azimuth_time_interval", burst.get("azimuthTimeInterval", 0.0)) or 0.0) or None,
            radar_wavelength=float(burst.get("radar_wavelength", burst.get("radarWavelength", 0.0)) or 0.0) or None,
        )
    return burst


def _selected_burst_indices_for_swath(plan: dict[str, Any], swath: str, total_bursts: int) -> set[int]:
    crop = _stage_swath_index(plan, "crop").get(swath, {})
    indices = crop.get("selected_burst_indices") or []
    if indices:
        return {int(v) for v in indices}
    return set(range(1, total_bursts + 1))


def _overlap_pairs_from_bundle(bundle: dict[str, Any], selected_indices: set[int]) -> list[dict[str, Any]]:
    overlaps = bundle.get("tops", {}).get("overlaps", [])
    pairs: list[dict[str, Any]] = []
    if not isinstance(overlaps, list):
        return pairs
    for item in overlaps:
        if not isinstance(item, dict):
            continue
        prev_idx = int(item.get("previousBurstIndex", 0))
        next_idx = int(item.get("nextBurstIndex", 0))
        if prev_idx <= 0 or next_idx <= 0:
            continue
        if prev_idx not in selected_indices or next_idx not in selected_indices:
            continue
        pairs.append(
            {
                "previous_burst_index": prev_idx,
                "next_burst_index": next_idx,
                "estimated_overlap_lines": int(item.get("estimatedOverlapLines", 0)),
            }
        )
    return pairs


def _overlap_infos_from_bundle(bundle: dict[str, Any], selected_indices: set[int]) -> list[Any]:
    overlaps = []
    for item in _overlap_pairs_from_bundle(bundle, selected_indices):
        prev_idx = int(item["previous_burst_index"])
        next_idx = int(item["next_burst_index"])
        # strip_insar2 TOPS helpers currently use zero-based burst positions.
        overlaps.append(
            strip_insar2.TopsOverlapInfo(
                previous_burst_index=prev_idx - 1,
                next_burst_index=next_idx - 1,
                estimated_overlap_lines=int(item.get("estimated_overlap_lines", 0)),
            )
        )
    return overlaps


def _local_backend_swath_merge_record(plan: dict[str, Any], swath_records: list[dict[str, Any]]) -> dict[str, Any]:
    swath_set = set(plan.get("swaths", []))
    merge_allowed = len(swath_set) >= 2 and swath_set != {"IW1", "IW3"}
    if not merge_allowed:
        return {"allowed": False, "reason": "non-adjacent swath set or single swath"}
    geocoded_inputs: list[dict[str, Any]] = []
    for item in swath_records:
        exports = item.get("exports", {}) if isinstance(item.get("exports"), dict) else {}
        geocoded = {
            "swath": item.get("swath"),
            "wrapped_phase_tif": exports.get("interferogram_tif"),
            "unwrapped_phase_tif": exports.get("unwrapped_phase_tif"),
            "coherence_tif": exports.get("coherence_tif"),
            "avg_amplitude_tif": exports.get("avg_amplitude_tif"),
        }
        if all(geocoded.get(key) for key in ("wrapped_phase_tif", "unwrapped_phase_tif", "coherence_tif", "avg_amplitude_tif")):
            geocoded_inputs.append(geocoded)
    if len(geocoded_inputs) == len(swath_records):
        scene_dir = Path(plan["plan_path"]).parent / "scene"
        merged = _merge_swaths(geocoded_inputs, scene_dir)
        return {
            "allowed": True,
            "reason": None,
            "task": {
                "operation": "merge_adjacent_swath_geocoded_products",
                "inputs": geocoded_inputs,
                "products": merged,
            },
            **merged,
        }
    return {
        "allowed": False,
        "reason": "local strip_insar2 TOPS backend did not publish all geocoded GeoTIFFs required for scene-level merge",
        "inputs": [
            {
                "swath": item.get("swath"),
                "exports": item.get("exports", {}),
                "pair_dir": item.get("pair_dir"),
            }
            for item in swath_records
        ],
    }


def _build_swath_context(plan: dict[str, Any], swath: str) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
    master_bundle = _load_tops_bundle(master_manifest_path)
    slave_bundle = _load_tops_bundle(slave_manifest_path)
    master_manifest = master_bundle["manifest"]
    slave_manifest = slave_bundle["manifest"]
    output_root = Path(plan["plan_path"]).parent
    pair_dir = _swath_root(plan, swath)
    pair_dir.mkdir(parents=True, exist_ok=True)
    context = strip_insar2.PairContext(
        master_manifest_path=Path(master_manifest_path),
        slave_manifest_path=Path(slave_manifest_path),
        master_manifest=master_manifest,
        slave_manifest=slave_manifest,
        master_orbit_data=master_bundle.get("acquisition", {}),
        slave_orbit_data=slave_bundle.get("acquisition", {}),
        master_acq_data=master_bundle.get("acquisition", {}),
        slave_acq_data=slave_bundle.get("acquisition", {}),
        master_rg_data=master_bundle.get("radargrid", {}),
        slave_rg_data=slave_bundle.get("radargrid", {}),
        master_dop_data=master_bundle.get("tops", {}),
        slave_dop_data=slave_bundle.get("tops", {}),
        output_root=output_root,
        pair_name=swath.lower(),
        pair_dir=pair_dir,
        output_paths={
            "interferogram_h5": str(_stage_output_dir(plan, swath, "p5") / f"{swath.lower()}_insar.h5"),
        },
        resolved_dem=str(plan.get("options", {}).get("dem") or ""),
        orbit_interp=str(plan.get("options", {}).get("orbit_interp") or "hermite"),
        wavelength=float(master_bundle.get("acquisition", {}).get("wavelength", 0.0) or 0.0),
        effective_crop_window=None,
    )
    return context, master_bundle, slave_bundle, master_manifest, slave_manifest


def _load_array(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(str(p))
    ds = gdal.Open(str(p), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(str(p))
    try:
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
    finally:
        ds = None
    return np.asarray(arr)


def _save_png_preview(arr: np.ndarray, output_png: Path) -> str:
    from PIL import Image

    data = np.asarray(arr, dtype=np.float32)
    img = np.zeros(data.shape, dtype=np.uint8)
    valid = np.isfinite(data)
    if np.any(valid):
        vals = data[valid]
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        if hi <= lo:
            hi = lo + 1.0
        img[valid] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0) * 255.0
    Image.fromarray(img, mode="L").save(output_png)
    return str(output_png)


def _write_simple_geotiff(output_path: Path, array: np.ndarray, reference: dict[str, Any]) -> str:
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    ds = driver.Create(str(output_path), cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    if ds is None:
        raise RuntimeError(f"failed to create {output_path}")
    ds.SetGeoTransform(reference["geotransform"])
    ds.SetProjection(reference["projection"])
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    band.WriteArray(np.asarray(array, dtype=np.float32))
    band.FlushCache()
    ds.FlushCache()
    ds = None
    return str(output_path)


def _read_raster_reference(path: str | Path) -> dict[str, Any]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(str(path))
    try:
        return {
            "geotransform": ds.GetGeoTransform(),
            "projection": ds.GetProjection(),
            "width": ds.RasterXSize,
            "height": ds.RasterYSize,
        }
    finally:
        ds = None


def _merge_swaths(swath_products: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    if not swath_products:
        raise ValueError("no swath products to merge")

    refs = [_read_raster_reference(item["wrapped_phase_tif"]) for item in swath_products]
    proj = refs[0]["projection"]
    gt = refs[0]["geotransform"]
    x_res = float(gt[1])
    y_res = abs(float(gt[5]))
    if any(ref["projection"] != proj for ref in refs):
        raise RuntimeError("swath projections differ; cannot merge")
    if any(abs(float(ref["geotransform"][1]) - x_res) > 1e-6 or abs(abs(float(ref["geotransform"][5])) - y_res) > 1e-6 for ref in refs):
        raise RuntimeError("swath pixel sizes differ; cannot merge safely")

    x_mins = [float(ref["geotransform"][0]) for ref in refs]
    y_maxs = [float(ref["geotransform"][3]) for ref in refs]
    x_maxs = [float(ref["geotransform"][0]) + ref["width"] * x_res for ref in refs]
    y_mins = [float(ref["geotransform"][3]) - ref["height"] * y_res for ref in refs]
    x_min = min(x_mins)
    y_max = max(y_maxs)
    x_max = max(x_maxs)
    y_min = min(y_mins)
    width = int(math.ceil((x_max - x_min) / x_res))
    height = int(math.ceil((y_max - y_min) / y_res))

    phase_sum = np.zeros((height, width), dtype=np.complex128)
    phase_weight = np.zeros((height, width), dtype=np.float32)
    unw_sum = np.zeros((height, width), dtype=np.float64)
    coh_sum = np.zeros((height, width), dtype=np.float64)
    amp_sum = np.zeros((height, width), dtype=np.float64)

    for item, ref in zip(swath_products, refs, strict=False):
        phase = np.asarray(_load_array(item["wrapped_phase_tif"]), dtype=np.float32)
        unwrapped = np.asarray(_load_array(item["unwrapped_phase_tif"]), dtype=np.float32)
        coherence = np.asarray(_load_array(item["coherence_tif"]), dtype=np.float32)
        amplitude = np.asarray(_load_array(item.get("avg_amplitude_tif", item["coherence_tif"])), dtype=np.float32)
        g = ref["geotransform"]
        x0 = int(round((float(g[0]) - x_min) / x_res))
        y0 = int(round((y_max - float(g[3])) / y_res))
        h, w = phase.shape
        x1 = min(width, x0 + w)
        y1 = min(height, y0 + h)
        if x0 < 0 or y0 < 0 or x0 >= width or y0 >= height:
            continue
        sl_y = slice(y0, y1)
        sl_x = slice(x0, x1)
        h2 = y1 - y0
        w2 = x1 - x0
        phase = phase[:h2, :w2]
        unwrapped = unwrapped[:h2, :w2]
        coherence = coherence[:h2, :w2]
        amplitude = amplitude[:h2, :w2]
        weight = np.clip(coherence, 0.0, 1.0).astype(np.float32)
        phase_sum[sl_y, sl_x] += weight * np.exp(1j * phase)
        phase_weight[sl_y, sl_x] += weight
        unw_sum[sl_y, sl_x] += unwrapped * weight
        coh_sum[sl_y, sl_x] += coherence
        amp_sum[sl_y, sl_x] += amplitude * weight

    merged_phase = np.full((height, width), np.nan, dtype=np.float32)
    valid = phase_weight > 0
    merged_phase[valid] = np.angle(phase_sum[valid]).astype(np.float32)
    merged_coh = np.full((height, width), np.nan, dtype=np.float32)
    merged_coh[valid] = (coh_sum[valid] / phase_weight[valid]).astype(np.float32)
    merged_unw = np.full((height, width), np.nan, dtype=np.float32)
    merged_unw[valid] = (unw_sum[valid] / phase_weight[valid]).astype(np.float32)

    ref_out = {"geotransform": (x_min, x_res, 0.0, y_max, 0.0, -y_res), "projection": proj}
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_ifg = _write_simple_geotiff(output_dir / "merged_interferogram.tif", merged_phase, ref_out)
    merged_coh_tif = _write_simple_geotiff(output_dir / "merged_coherence.tif", merged_coh, ref_out)
    merged_unw_tif = _write_simple_geotiff(output_dir / "merged_unwrapped_phase.tif", merged_unw, ref_out)
    merged_amp_tif = _write_simple_geotiff(output_dir / "merged_avg_amplitude.tif", np.where(valid, amp_sum / np.maximum(phase_weight, 1e-6), np.nan), ref_out)

    kml_path = output_dir / "merged_scene.kml"
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # Build a tiny KML from the projected bounds; if WGS84 conversion is unavailable, store UTM values.
    kml_coords = []
    try:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(srs, wgs84)
        corners = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]
        for x, y in corners:
            lon, lat, _ = transform.TransformPoint(float(x), float(y))
            kml_coords.append(f"{lon:.8f},{lat:.8f},0")
    except Exception:
        kml_coords = [f"{x_min},{y_min},0", f"{x_min},{y_max},0", f"{x_max},{y_max},0", f"{x_max},{y_min},0", f"{x_min},{y_min},0"]
    kml_path.write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<kml xmlns='http://www.opengis.net/kml/2.2'><Document><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>"
        + " ".join(kml_coords)
        + "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>\n",
        encoding="utf-8",
    )

    summary = {
        "bbox": [x_min, y_min, x_max, y_max],
        "pixel_size": [x_res, y_res],
        "swath_count": len(swath_products),
        "inputs": swath_products,
    }
    (output_dir / "merged_scene_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "merged_interferogram_tif": merged_ifg,
        "merged_coherence_tif": merged_coh_tif,
        "merged_unwrapped_phase_tif": merged_unw_tif,
        "merged_avg_amplitude_tif": merged_amp_tif,
        "merged_scene_kml": str(kml_path),
        "summary_json": str(output_dir / "merged_scene_summary.json"),
    }


def _run_stage_p3(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p3")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p3", "status": "ok", "swaths": out}

    execute = _should_execute_stages(plan)
    out: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
        master_bundle = _load_tops_bundle(master_manifest_path)
        slave_bundle = _load_tops_bundle(slave_manifest_path)
        selected_indices = _selected_burst_indices_for_swath(
            plan,
            swath,
            len(master_bundle.get("tops", {}).get("bursts", [])),
        )
        bursts = _burst_infos_from_bundle(master_bundle, selected_indices)
        overlap_pairs = _overlap_pairs_from_bundle(master_bundle, selected_indices)
        if not bursts:
            raise RuntimeError(f"swath {swath} has no bursts to merge")
        az_reference_offsets = _compute_az_reference_offsets(bursts)

        p2_dir = _stage_output_dir(plan, swath, "p2")
        burst_ifg_paths = [
            path
            for path in sorted(p2_dir.glob("burst_*/interferogram.int"))
            if int(path.parent.name.split("_")[-1]) in selected_indices
        ]

        task_record: dict[str, Any] = {
            "swath": swath,
            "status": "ok",
            "task_count": 1,
            "az_reference_offsets": az_reference_offsets,
            "tasks": [
                {
                    "operation": "merge_burst_interferograms",
                    "input_burst_count": len(bursts),
                    "selected_bursts": [int(getattr(b, "burst_index", i + 1)) for i, b in enumerate(bursts)],
                    "overlap_pairs": overlap_pairs,
                    "az_reference_offsets": az_reference_offsets,
                    "products": {
                        "merged_interferogram": "merged_interferogram.tif",
                        "merged_coherence": "merged_coherence.tif",
                    },
                }
            ],
        }

        if execute:
            merge_method = str(_plan_option(plan, "burst_merge_method", "avg") or "avg").lower()
            if not burst_ifg_paths:
                burst_ifg_paths = _compute_burst_interferograms(plan, swath)
            merged_ifg, merged_coh = _merge_bursts_isce2_style(
                burst_ifg_paths,
                bursts,
                method=merge_method,
                overlap_pairs=overlap_pairs,
                plan=plan,
            )

            stage_dir = _stage_output_dir(plan, swath, "p3")
            reference = _radargrid_reference_from_bundle(master_bundle, merged_ifg.shape)
            merged_ifg_tif = _write_simple_geotiff(stage_dir / "merged_interferogram.tif", np.angle(merged_ifg).astype(np.float32), reference)
            merged_coh_tif = _write_simple_geotiff(stage_dir / "merged_coherence.tif", np.asarray(merged_coh, dtype=np.float32), reference)
            task_record.update(
                {
                    "merged_interferogram_tif": merged_ifg_tif,
                    "merged_coherence_tif": merged_coh_tif,
                    "merged_interferogram_preview_png": _save_png_preview(np.angle(merged_ifg), stage_dir / "merged_interferogram.png"),
                    "merged_coherence_preview_png": _save_png_preview(merged_coh, stage_dir / "merged_coherence.png"),
                    "burst_ifg_paths": [str(path) for path in burst_ifg_paths],
                    "az_reference_offsets": az_reference_offsets,
                }
            )

        out.append(task_record)
    return {"stage": "p3", "status": "ok", "swaths": out}


def _run_stage_p4(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p4")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p4", "status": "ok", "swaths": out}

    execute = _should_execute_stages(plan)
    out: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        p3 = _stage_swath_index(plan, "p3").get(swath, {})
        if not p3:
            raise ValueError(f"p3 result missing for swath {swath}")

        task_record: dict[str, Any] = {
            "swath": swath,
            "status": "ok",
            "task_count": 1,
            "tasks": [
                {
                    "operation": "unwrap_and_geocode",
                    "unwrap_method": str(_plan_option(plan, "unwrap_method", "icu")),
                    "inputs": {
                        "merged_interferogram": "p3/merged_interferogram.tif",
                        "merged_coherence": "p3/merged_coherence.tif",
                    },
                    "products": {
                        "unwrapped_phase_geocoded": "p4/unwrapped_phase_geocoded.tif",
                        "coherence_geocoded": "p4/coherence_geocoded.tif",
                        "interferogram_geocoded": "p4/interferogram_geocoded.tif",
                        "los_displacement": "p4/los_displacement.tif",
                    },
                }
            ],
        }

        if execute:
            stage_p3 = _stage_output_dir(plan, swath, "p3")
            stage_p4 = _stage_output_dir(plan, swath, "p4")
            merged_ifg = _load_array(stage_p3 / "merged_interferogram.tif")
            merged_coh = _load_array(stage_p3 / "merged_coherence.tif")
            wrapped_phase = np.asarray(merged_ifg, dtype=np.float32)
            try:
                dem_path = str(_plan_option(plan, "dem", "") or "")
                reference = _read_raster_reference(dem_path) if dem_path and Path(dem_path).exists() else _read_raster_reference(stage_p3 / "merged_interferogram.tif")
            except Exception:
                reference = {"geotransform": (0.0, 1.0, 0.0, float(wrapped_phase.shape[0]), 0.0, -1.0), "projection": "EPSG:4326", "width": wrapped_phase.shape[1], "height": wrapped_phase.shape[0]}

            master_manifest_path, _ = _extract_manifest_pair_for_swath(plan, swath)
            master_bundle = _load_tops_bundle(master_manifest_path)
            radar_grid = master_bundle.get("radargrid", {})
            range_looks = int(_plan_option(plan, "range_looks", 1) or 1)
            azimuth_looks = int(_plan_option(plan, "azimuth_looks", 1) or 1)
            unwrap_method = str(_plan_option(plan, "unwrap_method", "icu") or "icu").lower()

            scratch_root = stage_p4 / "scratch"
            scratch_root.mkdir(parents=True, exist_ok=True)
            fallback_reason = None
            try:
                if unwrap_method == "snaphu":
                    unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_snaphu_profiles(
                        np.exp(1j * wrapped_phase).astype(np.complex64),
                        np.asarray(merged_coh, dtype=np.float32),
                        scratch_root / "snaphu",
                        radar_grid=radar_grid,
                        range_looks=range_looks,
                        azimuth_looks=azimuth_looks,
                        nlooks=float(range_looks * azimuth_looks),
                    )
                    if fallback_reason is None:
                        fallback_reason = None
                elif unwrap_method == "icu":
                    unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_icu_profiles(
                        np.exp(1j * wrapped_phase).astype(np.complex64),
                        np.asarray(merged_coh, dtype=np.float32),
                        scratch_root / "icu",
                    )
                elif unwrap_method == "phass":
                    try:
                        unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_icu_profiles(
                            np.exp(1j * wrapped_phase).astype(np.complex64),
                            np.asarray(merged_coh, dtype=np.float32),
                            scratch_root / "icu",
                        )
                    except Exception:
                        unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_snaphu_profiles(
                            np.exp(1j * wrapped_phase).astype(np.complex64),
                            np.asarray(merged_coh, dtype=np.float32),
                            scratch_root / "snaphu",
                            radar_grid=radar_grid,
                            range_looks=range_looks,
                            azimuth_looks=azimuth_looks,
                            nlooks=float(range_looks * azimuth_looks),
                        )
                elif unwrap_method == "dolphin":
                    try:
                        importlib.import_module("dolphin.unwrap")
                        importlib.import_module("dolphin.workflows.config")
                        # 若 dolphin 可用，仍优先采用 ICU 风格回退路径以保持接口稳定。
                        unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_icu_profiles(
                            np.exp(1j * wrapped_phase).astype(np.complex64),
                            np.asarray(merged_coh, dtype=np.float32),
                            scratch_root / "icu",
                        )
                    except Exception:
                        unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_icu_profiles(
                            np.exp(1j * wrapped_phase).astype(np.complex64),
                            np.asarray(merged_coh, dtype=np.float32),
                            scratch_root / "icu",
                        )
                else:
                    raise ValueError(f"unsupported unwrap method: {unwrap_method}")
            except Exception:
                unwrapped_phase, fallback_reason = strip_insar2._unwrap_with_icu_profiles(
                    np.exp(1j * wrapped_phase).astype(np.complex64),
                    np.asarray(merged_coh, dtype=np.float32),
                    scratch_root / "icu_fallback",
                )

            wavelength = float(master_bundle.get("acquisition", {}).get("wavelength", 0.0) or 0.0)
            if wavelength <= 0:
                wavelength = 0.0565656
            los_displacement = strip_insar2.compute_los_displacement(unwrapped_phase, wavelength)
            interferogram_geocoded = _resize_nearest(wrapped_phase, (reference["height"], reference["width"]))
            coherence_geocoded = _resize_nearest(np.asarray(merged_coh, dtype=np.float32), (reference["height"], reference["width"]))
            unwrapped_geocoded = _resize_nearest(np.asarray(unwrapped_phase, dtype=np.float32), (reference["height"], reference["width"]))
            los_geocoded = _resize_nearest(np.asarray(los_displacement, dtype=np.float32), (reference["height"], reference["width"]))

            interferogram_geotiff = _write_simple_geotiff(stage_p4 / "interferogram_geocoded.tif", interferogram_geocoded, reference)
            coherence_geotiff = _write_simple_geotiff(stage_p4 / "coherence_geocoded.tif", coherence_geocoded, reference)
            unwrapped_geotiff = _write_simple_geotiff(stage_p4 / "unwrapped_phase_geocoded.tif", unwrapped_geocoded, reference)
            los_geotiff = _write_simple_geotiff(stage_p4 / "los_displacement.tif", los_geocoded, reference)
            avg_amp_geotiff = _write_simple_geotiff(stage_p4 / "avg_amplitude_geocoded.tif", np.abs(interferogram_geocoded).astype(np.float32), reference)

            task_record.update(
                {
                    "interferogram_geocoded_tif": interferogram_geotiff,
                    "coherence_geocoded_tif": coherence_geotiff,
                    "unwrapped_phase_geocoded_tif": unwrapped_geotiff,
                    "los_displacement_tif": los_geotiff,
                    "avg_amplitude_geocoded_tif": avg_amp_geotiff,
                    "interferogram_geocoded_png": _save_png_preview(interferogram_geocoded, stage_p4 / "interferogram_geocoded.png"),
                    "coherence_geocoded_png": _save_png_preview(coherence_geocoded, stage_p4 / "coherence_geocoded.png"),
                    "unwrapped_phase_geocoded_png": _save_png_preview(unwrapped_geocoded, stage_p4 / "unwrapped_phase_geocoded.png"),
                    "los_displacement_png": _save_png_preview(los_geocoded, stage_p4 / "los_displacement.png"),
                    "fallback_reason": fallback_reason,
                }
            )

        out.append(task_record)
    return {"stage": "p4", "status": "ok", "swaths": out}


def _run_stage_p5(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p5")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p5", "status": "ok", "swaths": out}

    execute = _should_execute_stages(plan)
    out: list[dict[str, Any]] = []
    for swath in plan.get("swaths", []):
        p4 = _stage_swath_index(plan, "p4").get(swath, {})
        if not p4:
            raise ValueError(f"p4 result missing for swath {swath}")

        task_record: dict[str, Any] = {
            "swath": swath,
            "status": "ok",
            "task_count": 1,
            "tasks": [
                {
                    "operation": "package_hdf_products",
                    "inputs": {
                        "unwrapped_phase_geocoded": "p4/unwrapped_phase_geocoded.tif",
                        "interferogram_geocoded": "p4/interferogram_geocoded.tif",
                        "coherence_geocoded": "p4/coherence_geocoded.tif",
                        "los_displacement": "p4/los_displacement.tif",
                    },
                    "products": {
                        "insar_hdf5": f"p5/{swath.lower()}_insar.h5",
                    },
                }
            ],
        }

        if execute:
            stage_p3 = _stage_output_dir(plan, swath, "p3")
            stage_p4 = _stage_output_dir(plan, swath, "p4")
            stage_p5 = _stage_output_dir(plan, swath, "p5")
            master_manifest_path, slave_manifest_path = _extract_manifest_pair_for_swath(plan, swath)
            master_bundle = _load_tops_bundle(master_manifest_path)
            slave_bundle = _load_tops_bundle(slave_manifest_path)
            master_slc = _resolve_manifest_data_path(master_manifest_path, master_bundle["manifest"].get("slc", {}).get("path"))
            slave_slc = _resolve_manifest_data_path(slave_manifest_path, slave_bundle["manifest"].get("slc", {}).get("path"))
            if not master_slc or not slave_slc:
                raise ValueError(f"missing SLC path for swath {swath}")
            wrapped_phase = _load_array(stage_p4 / "interferogram_geocoded.tif")
            coherence = _load_array(stage_p4 / "coherence_geocoded.tif")
            unwrapped_phase = _load_array(stage_p4 / "unwrapped_phase_geocoded.tif")
            los_displacement = _load_array(stage_p4 / "los_displacement.tif")
            hdf_path = stage_p5 / f"{swath.lower()}_insar.h5"
            strip_insar2.write_insar_hdf(
                master_slc,
                slave_slc,
                np.exp(1j * np.asarray(wrapped_phase, dtype=np.float32)).astype(np.complex64),
                np.asarray(coherence, dtype=np.float32),
                np.asarray(unwrapped_phase, dtype=np.float32),
                np.asarray(los_displacement, dtype=np.float32),
                float(master_bundle.get("acquisition", {}).get("wavelength", 0.0) or 0.0565656),
                str(_plan_option(plan, "unwrap_method", "icu") or "icu"),
                str(hdf_path),
                range_looks=int(_plan_option(plan, "range_looks", 1) or 1),
                azimuth_looks=int(_plan_option(plan, "azimuth_looks", 1) or 1),
            )
            task_record["insar_hdf5"] = str(hdf_path)

        out.append(task_record)
    return {"stage": "p5", "status": "ok", "swaths": out}


def _run_stage_p6(plan: dict[str, Any]) -> dict[str, Any]:
    if _should_execute_stages(plan):
        out = [
            _run_local_tops_backend_for_swath(plan, swath, stop_after_stage="p6")
            for swath in plan.get("swaths", [])
        ]
        return {"stage": "p6", "status": "ok", "swaths": out, "swath_merge": _local_backend_swath_merge_record(plan, out)}

    execute = _should_execute_stages(plan)
    out: list[dict[str, Any]] = []
    geocoded_inputs: list[dict[str, Any]] = []

    for swath in plan.get("swaths", []):
        p4 = _stage_swath_index(plan, "p4").get(swath, {})
        p5 = _stage_swath_index(plan, "p5").get(swath, {})
        if not p4:
            raise ValueError(f"p4 result missing for swath {swath}")
        if not p5:
            raise ValueError(f"p5 result missing for swath {swath}")

        stage_p4 = _stage_output_dir(plan, swath, "p4")
        geocoded = {
            "swath": swath,
            "wrapped_phase_tif": str(stage_p4 / "interferogram_geocoded.tif"),
            "unwrapped_phase_tif": str(stage_p4 / "unwrapped_phase_geocoded.tif"),
            "coherence_tif": str(stage_p4 / "coherence_geocoded.tif"),
            "avg_amplitude_tif": str(stage_p4 / "avg_amplitude_geocoded.tif"),
        }
        geocoded_inputs.append(geocoded)
        out.append(
            {
                "swath": swath,
                "task_count": 1,
                "tasks": [
                    {
                        "operation": "publish_swath_outputs",
                        "products": geocoded,
                    }
                ],
            }
        )

    merge_task = None
    merge_record: dict[str, Any] = {"allowed": False, "reason": "execution disabled"}
    if execute:
        swath_set = set(plan.get("swaths", []))
        merge_allowed = len(swath_set) >= 2 and swath_set != {"IW1", "IW3"}
        if merge_allowed:
            scene_dir = Path(plan["plan_path"]).parent / "scene"
            merged = _merge_swaths(geocoded_inputs, scene_dir)
            merge_task = {
                "operation": "merge_adjacent_swath_geocoded_products",
                "inputs": geocoded_inputs,
                "products": merged,
            }
            merge_record = {"allowed": True, "reason": None, "task": merge_task, **merged}
        else:
            merge_record = {"allowed": False, "reason": "non-adjacent swath set or single swath"}
    else:
        swath_set = set(plan.get("swaths", []))
        merge_record = {
            "allowed": len(swath_set) >= 2 and swath_set != {"IW1", "IW3"},
            "reason": None if len(swath_set) >= 2 and swath_set != {"IW1", "IW3"} else "non-adjacent swath set or single swath",
        }

    return {"stage": "p6", "status": "ok", "swaths": out, "swath_merge": merge_record}


def _build_plan_for_manifests(
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    output_dir: str | Path,
    swaths: list[str],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    master_manifest = _load_manifest(master_manifest_path)
    slave_manifest = _load_manifest(slave_manifest_path)
    if str(master_manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("master manifest must be sentinel-1")
    if str(slave_manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("slave manifest must be sentinel-1")

    plan = {
        "version": "1.0",
        "mode": "manifest",
        "pipeline": "tops_insar",
        "master_manifest": str(master_manifest_path),
        "slave_manifest": str(slave_manifest_path),
        "swath_selector": args.swath,
        "swaths": swaths,
        "warnings": warnings,
        "options": {
            "start_stage": args.start_stage,
            "end_stage": args.end_stage,
            "stop_after": args.stop_after,
            "resume": bool(args.resume),
            "execute_stages": bool(args.execute_stages),
            "gpu_mode": args.gpu_mode,
            "topo_gpu": bool(args.topo_gpu),
            "gpu_id": int(args.gpu_id),
            "dem": args.dem,
            "dem_cache_dir": args.dem_cache_dir,
            "dem_margin_deg": float(args.dem_margin_deg),
            "burst_limit": args.burst_limit,
            "resolution": float(args.resolution),
            "range_looks": int(args.range_looks),
            "azimuth_looks": int(args.azimuth_looks),
            "block_rows": args.block_rows,
            "no_kml": bool(args.no_kml),
            "unwrap_method": args.unwrap_method,
            "execute_backend": bool(args.execute_backend),
            "backend_timeout_seconds": int(args.backend_timeout_seconds),
            "do_ionospheric_correction": bool(args.do_ionospheric_correction),
            "extra_esd_cycles": float(args.extra_esd_cycles),
            "esd_coherence_threshold": float(args.esd_coherence_threshold),
        },
        "stages": {"implemented": ["prepare"], "pending": list(STAGE_SEQUENCE)},
        "stage_status": {},
    }
    plan["plan_path"] = _write_plan(plan, output_dir)
    return plan


def _build_plan_for_products(
    master_product_path: str | Path,
    slave_product_path: str | Path,
    output_dir: str | Path,
    swaths: list[str],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    from sentinel_importer import SentinelImporter

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    swath_inputs: dict[str, dict[str, str]] = {}
    orbit_dir = out / "orbits"
    orbit_dir.mkdir(parents=True, exist_ok=True)
    for sw in swaths:
        master_import_dir = out / sw / "master_import"
        slave_import_dir = out / sw / "slave_import"
        master_manifest = SentinelImporter(
            master_product_path,
            swath=sw,
            orbit_dir=orbit_dir,
            download_orbit=True,
        ).run(
            str(master_import_dir), download_dem=False
        )
        slave_manifest = SentinelImporter(
            slave_product_path,
            swath=sw,
            orbit_dir=orbit_dir,
            download_orbit=True,
        ).run(
            str(slave_import_dir), download_dem=False
        )
        swath_inputs[sw] = {
            "master_manifest": str(master_manifest),
            "slave_manifest": str(slave_manifest),
        }

    plan = {
        "version": "1.0",
        "mode": "product",
        "pipeline": "tops_insar",
        "master_product_path": str(master_product_path),
        "slave_product_path": str(slave_product_path),
        "swath_selector": args.swath,
        "swaths": swaths,
        "swath_inputs": swath_inputs,
        "warnings": warnings,
        "options": {
            "start_stage": args.start_stage,
            "end_stage": args.end_stage,
            "stop_after": args.stop_after,
            "resume": bool(args.resume),
            "execute_stages": bool(args.execute_stages),
            "gpu_mode": args.gpu_mode,
            "topo_gpu": bool(args.topo_gpu),
            "gpu_id": int(args.gpu_id),
            "dem": args.dem,
            "dem_cache_dir": args.dem_cache_dir,
            "dem_margin_deg": float(args.dem_margin_deg),
            "burst_limit": args.burst_limit,
            "resolution": float(args.resolution),
            "range_looks": int(args.range_looks),
            "azimuth_looks": int(args.azimuth_looks),
            "block_rows": args.block_rows,
            "no_kml": bool(args.no_kml),
            "unwrap_method": args.unwrap_method,
            "execute_backend": bool(args.execute_backend),
            "backend_timeout_seconds": int(args.backend_timeout_seconds),
            "do_ionospheric_correction": bool(args.do_ionospheric_correction),
            "extra_esd_cycles": float(args.extra_esd_cycles),
            "esd_coherence_threshold": float(args.esd_coherence_threshold),
        },
        "stages": {"implemented": ["prepare"], "pending": list(STAGE_SEQUENCE)},
        "stage_status": {},
    }
    plan["plan_path"] = _write_plan(plan, output_dir)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel-1 TOPS InSAR unified pipeline")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("master_manifest", nargs="?", help="Master manifest.json")
    parser.add_argument("slave_manifest", nargs="?", help="Slave manifest.json")
    parser.add_argument("--master-product-path", default=None, help="Master Sentinel SAFE/ZIP/TAR path")
    parser.add_argument("--slave-product-path", default=None, help="Slave Sentinel SAFE/ZIP/TAR path")
    parser.add_argument(
        "--swath",
        default="all",
        help="IW1|IW2|IW3|IW1,IW2|IW2,IW3|IW1,IW3|all",
    )
    parser.add_argument("--start-stage", default="check")
    parser.add_argument("--end-stage", default="p6")
    parser.add_argument("--stop-after", default=None, choices=STAGE_SEQUENCE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-stages", dest="execute_stages", action="store_true")
    parser.add_argument("--no-execute-stages", dest="execute_stages", action="store_false")
    parser.set_defaults(execute_stages=True)
    parser.add_argument("--gpu-mode", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--topo-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--dem-cache-dir", default=None)
    parser.add_argument("--dem-margin-deg", type=float, default=0.2)
    parser.add_argument("--burst-limit", type=int, default=None)
    parser.add_argument("--resolution", type=float, default=20.0)
    parser.add_argument("--rlks", "--range-looks", dest="range_looks", type=int, default=1)
    parser.add_argument("--alks", "--azimuth-looks", dest="azimuth_looks", type=int, default=1)
    parser.add_argument("--block-rows", type=int, default=None)
    parser.add_argument("--no-kml", action="store_true")
    parser.add_argument("--unwrap-method", default="icu", choices=("icu", "snaphu", "phass", "dolphin"))
    parser.add_argument("--execute-backend", action="store_true", help="Execute strip_insar2 backend during p2 stage")
    parser.add_argument("--backend-timeout-seconds", type=int, default=1800, help="Timeout for strip backend call in seconds")
    parser.add_argument("--do-ionospheric-correction", action="store_true", help="Enable ionospheric correction framework during p2 stage")
    parser.add_argument("--extra-esd-cycles", type=float, default=0.0)
    parser.add_argument("--esd-coherence-threshold", type=float, default=0.85)
    args = parser.parse_args()
    if args.stop_after:
        args.end_stage = str(args.stop_after)

    swaths, warnings = parse_swath_selector(args.swath)
    try:
        _stage_range(args.start_stage, args.end_stage)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    product_mode = bool(args.master_product_path or args.slave_product_path)
    manifest_mode = bool(args.master_manifest or args.slave_manifest)
    if product_mode and manifest_mode:
        raise SystemExit("Use either manifest inputs or product-path inputs, not both.")
    if args.resume and (product_mode or manifest_mode):
        raise SystemExit("--resume cannot be used together with new input arguments")

    if args.resume:
        plan = _load_existing_plan(args.output_dir)
        result = _execute_stage_window(
            plan,
            args.output_dir,
            args.start_stage,
            args.end_stage,
            resume=True,
        )
        result["plan_path"] = _write_plan(result, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if product_mode:
        if not args.master_product_path or not args.slave_product_path:
            raise SystemExit("--master-product-path and --slave-product-path are both required")
        result = _build_plan_for_products(
            args.master_product_path,
            args.slave_product_path,
            args.output_dir,
            swaths,
            args,
            warnings,
        )
        result = _execute_stage_window(
            result,
            args.output_dir,
            args.start_stage,
            args.end_stage,
            resume=False,
        )
        result["plan_path"] = _write_plan(result, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.master_manifest or not args.slave_manifest:
        raise SystemExit(
            "Manifest mode requires: output_dir master_manifest slave_manifest"
        )
    result = _build_plan_for_manifests(
        args.master_manifest,
        args.slave_manifest,
        args.output_dir,
        swaths,
        args,
        warnings,
    )
    result = _execute_stage_window(
        result,
        args.output_dir,
        args.start_stage,
        args.end_stage,
        resume=False,
    )
    result["plan_path"] = _write_plan(result, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
