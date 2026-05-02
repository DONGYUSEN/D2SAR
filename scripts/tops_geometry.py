from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from common_processing import resolve_manifest_metadata_path


GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def load_tops_metadata(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if str(manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("tops_geometry requires a sentinel-1 manifest")

    metadata = {"manifest": manifest}
    for name in ("acquisition", "orbit", "radargrid", "doppler", "tops"):
        with open(resolve_manifest_metadata_path(manifest_path, manifest, name), encoding="utf-8") as f:
            metadata[name] = json.load(f)
    return metadata


def iter_burst_radar_grids(manifest_path: str | Path, *, validate: bool = True) -> Iterator[dict[str, Any]]:
    metadata = load_tops_metadata(manifest_path)
    for burst in metadata["tops"].get("bursts", []):
        burst_grid = build_burst_radar_grid_json(
            metadata["radargrid"], metadata["acquisition"], metadata["orbit"], burst
        )
        if validate:
            validate_burst_geometry(metadata["acquisition"], metadata["orbit"], burst_grid, burst)
        yield burst_grid


def build_burst_radar_grid_json(
    radargrid: dict[str, Any],
    acquisition: dict[str, Any],
    orbit: dict[str, Any],
    burst: dict[str, Any],
) -> dict[str, Any]:
    del orbit
    sensing_start = str(burst.get("sensingStartUTC") or burst.get("azimuthTimeUTC") or "")
    return {
        "source": "sentinel-1-burst",
        "burstIndex": int(burst.get("index", 0)),
        "numberOfRows": int(burst.get("numberOfLines", radargrid.get("numberOfRows", 0))),
        "numberOfColumns": int(burst.get("numberOfSamples", radargrid.get("numberOfColumns", 0))),
        "rowSpacing": float(burst.get("azimuthTimeInterval", radargrid.get("rowSpacing", 0.0))),
        "columnSpacing": float(burst.get("rangePixelSize", radargrid.get("columnSpacing", 0.0))),
        "rangeTimeFirstPixel": float(radargrid.get("rangeTimeFirstPixel", 0.0)),
        "startingRange": float(burst.get("startingRange", radargrid.get("startingRange", 0.0))),
        "prf": float(burst.get("prf", radargrid.get("prf", 0.0))),
        "wavelength": float(burst.get("radarWavelength", radargrid.get("wavelength", 0.0))),
        "sensingStartUTC": sensing_start,
        "sensingStopUTC": str(burst.get("sensingStopUTC") or ""),
        "sensingStartGPSTime": _parse_timestamp(sensing_start),
        "lineOffset": int(burst.get("lineOffset", 0)),
        "firstValidLine": int(burst.get("firstValidLine", 0)),
        "numValidLines": int(burst.get("numValidLines", 0)),
        "firstValidSample": int(burst.get("firstValidSample", 0)),
        "lastValidSample": int(burst.get("lastValidSample", -1)),
        "numValidSamples": int(burst.get("numValidSamples", 0)),
        "swath": acquisition.get("swath") or radargrid.get("swath"),
        "polarisation": acquisition.get("polarisation") or radargrid.get("polarisation"),
        "lookDirection": acquisition.get("lookDirection", "RIGHT"),
    }


def select_burst_doppler(burst: dict[str, Any]) -> dict[str, Any]:
    doppler = burst.get("doppler")
    if not isinstance(doppler, dict) or not doppler.get("coefficients"):
        raise ValueError(f"burst {burst.get('index')} has no Doppler polynomial")
    return doppler


def validate_burst_geometry(
    acquisition: dict[str, Any],
    orbit: dict[str, Any],
    burst_grid: dict[str, Any],
    burst: dict[str, Any],
    *,
    orbit_margin_seconds: float = 5.0,
) -> None:
    if int(burst_grid.get("numberOfRows", 0)) <= 0:
        raise ValueError("burst radar grid numberOfRows must be positive")
    if int(burst_grid.get("numberOfColumns", 0)) <= 0:
        raise ValueError("burst radar grid numberOfColumns must be positive")
    if float(burst_grid.get("prf", 0.0)) <= 0.0:
        raise ValueError("burst radar grid prf must be positive")
    if float(burst_grid.get("columnSpacing", 0.0)) <= 0.0:
        raise ValueError("burst radar grid columnSpacing must be positive")
    if float(burst_grid.get("wavelength", 0.0)) <= 0.0:
        raise ValueError("burst radar grid wavelength must be positive")

    state_vectors = orbit.get("stateVectors") or []
    if not state_vectors:
        raise ValueError("orbit stateVectors must not be empty")
    _validate_orbit_covers_burst(state_vectors, burst_grid, orbit_margin_seconds)
    _validate_valid_region(burst_grid)
    select_burst_doppler(burst)

    if float(acquisition.get("centerFrequency", 0.0)) <= 0.0:
        raise ValueError("acquisition centerFrequency must be positive")


def _validate_valid_region(burst_grid: dict[str, Any]) -> None:
    first_line = int(burst_grid.get("firstValidLine", 0))
    num_lines = int(burst_grid.get("numValidLines", 0))
    first_sample = int(burst_grid.get("firstValidSample", 0))
    num_samples = int(burst_grid.get("numValidSamples", 0))
    rows = int(burst_grid.get("numberOfRows", 0))
    cols = int(burst_grid.get("numberOfColumns", 0))
    if first_line < 0 or num_lines <= 0 or first_line + num_lines > rows:
        raise ValueError("burst valid line region is outside burst bounds")
    if first_sample < 0 or num_samples <= 0 or first_sample + num_samples > cols:
        raise ValueError("burst valid sample region is outside burst bounds")


def _validate_orbit_covers_burst(
    state_vectors: list[dict[str, Any]], burst_grid: dict[str, Any], margin_seconds: float
) -> None:
    orbit_times = [_parse_datetime(str(sv.get("timeUTC", ""))) for sv in state_vectors]
    orbit_times = [time for time in orbit_times if time is not None]
    if not orbit_times:
        raise ValueError("orbit stateVectors contain no valid timeUTC values")

    start = _parse_datetime(str(burst_grid.get("sensingStartUTC", "")))
    stop = _parse_datetime(str(burst_grid.get("sensingStopUTC", "")))
    if start is None or stop is None:
        raise ValueError("burst sensingStartUTC/sensingStopUTC must be valid UTC timestamps")

    coverage_start = min(orbit_times)
    coverage_stop = max(orbit_times)
    if (start - coverage_start).total_seconds() < -margin_seconds:
        raise ValueError("orbit does not cover burst sensing start")
    if (coverage_stop - stop).total_seconds() < -margin_seconds:
        raise ValueError("orbit does not cover burst sensing stop")


def _parse_timestamp(value: str) -> float:
    dt = _parse_datetime(value)
    if dt is None:
        return 0.0
    return (dt - GPS_EPOCH).total_seconds()


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Sentinel TOPS per-burst geometry metadata")
    parser.add_argument("manifest", help="Path to Sentinel importer manifest.json")
    args = parser.parse_args()
    burst_grids = list(iter_burst_radar_grids(args.manifest))
    print(json.dumps({"burst_count": len(burst_grids), "bursts": burst_grids}, indent=2))


if __name__ == "__main__":
    main()
