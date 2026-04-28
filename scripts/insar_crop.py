from __future__ import annotations

from pathlib import Path

from common_processing import scene_bbox_from_corners


def normalize_crop_request(
    *,
    bbox: tuple[float, float, float, float] | None,
    window: tuple[int, int, int, int] | None,
    radargrid_data: dict,
    scene_corners: list | None,
) -> dict:
    if bbox is not None and window is not None:
        raise ValueError("--bbox and --window are mutually exclusive")

    rows = int(radargrid_data.get("numberOfRows", 0))
    cols = int(radargrid_data.get("numberOfColumns", 0))
    full_window = {"row0": 0, "col0": 0, "rows": rows, "cols": cols}

    if bbox is None and window is None:
        return {"mode": "full", "master_window": full_window, "bbox": None}

    if window is not None:
        if rows <= 0 or cols <= 0:
            raise ValueError("window crop requires radar-grid dimensions")
        row0, col0, win_rows, win_cols = [int(v) for v in window]
        if min(row0, col0, win_rows, win_cols) < 0:
            raise ValueError("window values must be non-negative")
        if win_rows <= 0 or win_cols <= 0:
            raise ValueError("window rows/cols must be > 0")
        if row0 + win_rows > rows or col0 + win_cols > cols:
            raise ValueError("window must lie inside master radar-grid dimensions")
        return {
            "mode": "window",
            "master_window": {
                "row0": row0,
                "col0": col0,
                "rows": win_rows,
                "cols": win_cols,
            },
            "bbox": None,
        }

    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    if rows <= 0 or cols <= 0:
        raise ValueError("bbox crop requires radar-grid dimensions")
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")
    if not scene_corners:
        raise ValueError("bbox requires master scene corners for normalization")

    scene_min_lon, scene_max_lon, scene_min_lat, scene_max_lat = scene_bbox_from_corners(scene_corners)
    if max_lon <= scene_min_lon or min_lon >= scene_max_lon or max_lat <= scene_min_lat or min_lat >= scene_max_lat:
        raise ValueError("bbox must intersect the master scene")

    lon0 = max(min_lon, scene_min_lon)
    lon1 = min(max_lon, scene_max_lon)
    lat0 = max(min_lat, scene_min_lat)
    lat1 = min(max_lat, scene_max_lat)

    col0 = int(round((lon0 - scene_min_lon) / max(scene_max_lon - scene_min_lon, 1e-9) * cols))
    col1 = int(round((lon1 - scene_min_lon) / max(scene_max_lon - scene_min_lon, 1e-9) * cols))
    row0 = int(round((scene_max_lat - lat1) / max(scene_max_lat - scene_min_lat, 1e-9) * rows))
    row1 = int(round((scene_max_lat - lat0) / max(scene_max_lat - scene_min_lat, 1e-9) * rows))

    col0 = max(0, min(cols - 1, col0))
    row0 = max(0, min(rows - 1, row0))
    col1 = max(col0 + 1, min(cols, col1))
    row1 = max(row0 + 1, min(rows, row1))

    return {
        "mode": "bbox",
        "master_window": {
            "row0": row0,
            "col0": col0,
            "rows": row1 - row0,
            "cols": col1 - col0,
        },
        "bbox": [min_lon, min_lat, max_lon, max_lat],
    }
