from datetime import datetime, timedelta, timezone

import numpy as np


def parse_iso_utc(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    elif not (ts[-6:-5] in ("+", "-") and ts[-6:].count(":") == 3):
        ts = ts + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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


def build_isce3_orbit(orbit_json: dict, interp_method: str = "Hermite"):
    import isce3.core

    raw_datetimes = [parse_iso_utc(sv["timeUTC"]) for sv in orbit_json["stateVectors"]]
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
        parse_iso_utc(orbit_json["header"]["firstStateTimeUTC"])
    )
    method_map = {
        "Hermite": isce3.core.OrbitInterpMethod.HERMITE,
        "Legendre": isce3.core.OrbitInterpMethod.LEGENDRE,
    }
    return isce3.core.Orbit(state_vectors, ref_dt, method_map[interp_method])


def manual_geo2rdr_llh(
    lon_deg: float,
    lat_deg: float,
    h_m: float,
    orbit,
    initial_guess_rel: float,
    threshold: float = 1e-9,
    max_iter: int = 50,
):
    target = llh_to_xyz(lon_deg, lat_deg, h_m)
    t = initial_guess_rel
    for _ in range(max_iter):
        pos, vel = orbit.interpolate(t)
        rel = target - pos
        f = np.dot(vel, rel)
        _, vel1 = orbit.interpolate(t + 0.01)
        acc = (vel1 - vel) / 0.01
        df = np.dot(acc, rel) - np.dot(vel, vel)
        dt = np.clip(f / df, -1.0, 1.0)
        if abs(dt) < threshold:
            return t, float(np.linalg.norm(rel))
        t -= dt
    return t, float(np.linalg.norm(rel))


def geo2rdr_compat(
    lon_deg: float,
    lat_deg: float,
    h_m: float,
    orbit,
    wavelength_m: float,
    look_direction: str,
    initial_guess_rel: float,
):
    import isce3.geometry

    xyz = llh_to_xyz(lon_deg, lat_deg, h_m)
    look = look_direction.strip().lower()
    zero = __import__("isce3.core").core.LUT2d()
    try:
        azt, sr = isce3.geometry.geo2rdr_bracket(xyz, orbit, zero, wavelength_m, look)
        return azt, sr, "geo2rdr_bracket"
    except Exception:
        azt, sr = manual_geo2rdr_llh(
            lon_deg,
            lat_deg,
            h_m,
            orbit,
            initial_guess_rel,
        )
        return azt, sr, "manual_newton"
