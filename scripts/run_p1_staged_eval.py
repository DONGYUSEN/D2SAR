from __future__ import annotations

import json
import sys
from pathlib import Path

from insar_registration import (
    _estimate_offset_mean_from_raster,
    _select_cpu_dense_match_plan,
    run_cpu_dense_offsets,
)


def _load_gross_offset(base: Path) -> tuple[float, float]:
    model_path = base / "work" / "p1_dense_match" / "coarse_geo2rdr_model.json"
    if model_path.is_file():
        data = json.loads(model_path.read_text(encoding="utf-8"))
        az = float(data["azimuth_model"]["coefficients"][0])
        rg = float(data["range_model"]["coefficients"][0])
        return az, rg
    coarse_az = base / "work" / "p1_dense_match" / "coarse_geo2rdr_azimuth.off"
    coarse_rg = base / "work" / "p1_dense_match" / "coarse_geo2rdr_range.off"
    if not coarse_az.is_file() or not coarse_rg.is_file():
        legacy_az = base / "p1_geo2rdr_offsets" / "azimuth.off"
        legacy_rg = base / "p1_geo2rdr_offsets" / "range.off"
        coarse_az = legacy_az if legacy_az.is_file() else coarse_az
        coarse_rg = legacy_rg if legacy_rg.is_file() else coarse_rg
    return (
        _estimate_offset_mean_from_raster(coarse_az),
        _estimate_offset_mean_from_raster(coarse_rg),
    )


def main() -> int:
    if len(sys.argv) not in (2, 6):
        print("usage: run_p1_staged_eval.py <result_dir> [row0 col0 rows cols]", file=sys.stderr)
        return 2

    base = Path(sys.argv[1]).resolve()
    master = base / "work" / "crop" / "master.crop.tif"
    slave = base / "work" / "p1_dense_match" / "coarse_coreg_slave.tif"
    out = base / "work" / "p1_dense_match_staged_eval"
    out.mkdir(parents=True, exist_ok=True)

    gross = _load_gross_offset(base)
    plan = _select_cpu_dense_match_plan(3.0)
    if len(sys.argv) == 6:
        row0 = int(sys.argv[2])
        col0 = int(sys.argv[3])
        rows = int(sys.argv[4])
        cols = int(sys.argv[5])
        subset_dir = out / "subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        from osgeo import gdal
        for src, dst_name in ((master, "master.tif"), (slave, "slave.tif")):
            ds = gdal.Open(str(src), gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"failed to open raster: {src}")
            try:
                arr = ds.GetRasterBand(1).ReadAsArray(col0, row0, cols, rows)
            finally:
                ds = None
            out_ds = gdal.GetDriverByName("GTiff").Create(
                str(subset_dir / dst_name),
                cols,
                rows,
                1,
                gdal.GDT_Float32,
            )
            out_ds.GetRasterBand(1).WriteArray(arr)
            out_ds = None
        master = subset_dir / "master.tif"
        slave = subset_dir / "slave.tif"
        out = subset_dir

    row, col, details = run_cpu_dense_offsets(
        master_slc_path=str(master),
        slave_slc_path=str(slave),
        output_dir=out,
        gross_offset=gross,
        window_candidates=plan["candidates"],
        return_details=True,
    )
    payload = {
        "gross_offset": list(gross),
        "row_is_none": row is None,
        "col_is_none": col is None,
        "diagnostics": None if details is None else details.get("diagnostics"),
    }
    (out / "staged_eval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
