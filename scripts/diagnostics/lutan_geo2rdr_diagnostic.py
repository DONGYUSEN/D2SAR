import json
from datetime import datetime, timezone

# Add parent directory to Python path
import sys
sys.path.insert(0, '..')

import numpy as np

from common_processing import construct_orbit, gps_to_datetime

import isce3.core
import isce3.geometry


def llh_to_xyz(lon_deg: float, lat_deg: float, h_m: float) -> np.ndarray:
    a = 6378137.0
    e2 = 0.00669437999014
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + h_m) * np.cos(lat) * np.cos(lon),
            (n + h_m) * np.cos(lat) * np.sin(lon),
            (n * (1 - e2) + h_m) * np.sin(lat),
        ],
        dtype=np.float64,
    )


def manual_center_solution(orbit, acquisition, radargrid):
    xyz = llh_to_xyz(
        acquisition["centerLon"],
        acquisition["centerLat"],
        acquisition["sceneAverageHeight"],
    )
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    ref_dt = gps_to_datetime(acquisition["orbitFirstStateTimeUTC"])
    orbit_ref_gps = (ref_dt - gps_epoch).total_seconds()
    t = (
        acquisition["startGPSTime"]
        - orbit_ref_gps
        + radargrid["numberOfRows"] / 2 / acquisition["prf"]
    )
    for _ in range(50):
        pos, vel = orbit.interpolate(t)
        rel = xyz - pos
        f = np.dot(vel, rel)
        _, vel1 = orbit.interpolate(t + 0.01)
        acc = (vel1 - vel) / 0.01
        df = np.dot(acc, rel) - np.dot(vel, vel)
        dt = np.clip(f / df, -1.0, 1.0)
        if abs(dt) < 1e-9:
            break
        t -= dt
    pos, vel = orbit.interpolate(t)
    rvec = xyz - pos
    return {
        "az_time_rel": float(t),
        "slant_range_m": float(np.linalg.norm(rvec)),
        "lookside_crossdot": float(np.cross(rvec, vel).dot(pos)),
    }


def run_diagnostic(metadata_dir: str) -> None:
    import isce3.core
    import isce3.geometry

    with open(f"{metadata_dir}/orbit.json", encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(f"{metadata_dir}/acquisition.json", encoding="utf-8") as f:
        acquisition = json.load(f)
    with open(f"{metadata_dir}/radargrid.json", encoding="utf-8") as f:
        radargrid = json.load(f)

    acquisition = {
        **acquisition,
        "orbitFirstStateTimeUTC": orbit_data["header"]["firstStateTimeUTC"],
    }
    orbit = construct_orbit(orbit_data, "Legendre")
    xyz = llh_to_xyz(
        acquisition["centerLon"],
        acquisition["centerLat"],
        acquisition["sceneAverageHeight"],
    )
    wavelength = isce3.core.speed_of_light / acquisition["centerFrequency"]
    zero = isce3.core.LUT2d()
    manual = manual_center_solution(orbit, acquisition, radargrid)

    print(
        {
            "metadataLookDirection": acquisition["lookDirection"],
            "manualAzTimeRel": manual["az_time_rel"],
            "manualSlantRangeKm": manual["slant_range_m"] / 1000.0,
            "metadataStartingRangeKm": isce3.core.speed_of_light
            * radargrid["rangeTimeFirstPixel"]
            / 2.0
            / 1000.0,
            "looksideCrossdot": manual["lookside_crossdot"],
            "rightLookPassesIsce3Check": manual["lookside_crossdot"] > 0,
        }
    )

    for side in ("right", "left"):
        try:
            azt, sr = isce3.geometry.geo2rdr_bracket(xyz, orbit, zero, wavelength, side)
            print(
                {
                    "geo2rdr_bracket": side,
                    "status": "success",
                    "azt": azt,
                    "sr_km": sr / 1000.0,
                }
            )
        except Exception as exc:
            print({"geo2rdr_bracket": side, "status": "failed", "error": str(exc)})

    ellipsoid = isce3.core.Ellipsoid()
    llh = np.array(
        [
            [np.deg2rad(acquisition["centerLon"])],
            [np.deg2rad(acquisition["centerLat"])],
            [acquisition["sceneAverageHeight"]],
        ],
        dtype=np.float64,
    )
    for side in (isce3.core.LookSide.Right, isce3.core.LookSide.Left):
        try:
            azt, sr = isce3.geometry.geo2rdr(
                llh,
                ellipsoid,
                orbit,
                zero,
                wavelength,
                side,
                threshold=1e-8,
                maxiter=100,
                delta_range=10.0,
            )
            print(
                {
                    "geo2rdr": str(side),
                    "status": "success",
                    "azt": azt,
                    "sr_km": sr / 1000.0,
                }
            )
        except Exception as exc:
            print({"geo2rdr": str(side), "status": "failed", "error": str(exc)})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Lutan geo2rdr diagnostic")
    parser.add_argument(
        "metadata_dir", help="Directory containing orbit/acquisition/radargrid JSON"
    )
    args = parser.parse_args()
    run_diagnostic(args.metadata_dir)


if __name__ == "__main__":
    main()