#!/usr/bin/env python3
"""
Robust orbit smoothing for Lutan SAR data.

Implements the ISCE2 Lutan1 orbit filter algorithm:
  - Robust iterative fitting using scipy UnivariateSpline (preferred)
    or numpy polyfit (fallback)
  - MAD-based (Median Absolute Deviation) outlier rejection
  - Per-component fitting for position (XYZ) and velocity (XYZ)
  - Auto-ignore at orbit start/end to avoid edge artifacts

Usage:
    from orbit_smooth import orbit_smooth, smooth_orbit_from_json

    # Basic usage with arrays
    smoothed_pos, smoothed_vel, info = orbit_smooth(
        t, pos, vel,
        degree=5, sigma=4.0, max_iter=3
    )

    # From Lutan metadata JSON
    smoothed_orbit = smooth_orbit_from_json(
        '/tmp/lutan_output2/metadata/orbit.json'
    )
    # Returns: dict with 'stateVectors', 'refTime', etc.

Author: Adapted from ISCE2 Lutan1._filterOrbit() and ORB_filt_spline.py
"""

import warnings
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, Literal

try:
    from scipy.interpolate import UnivariateSpline
except ImportError:
    UnivariateSpline = None

__all__ = ["orbit_smooth", "smooth_orbit_from_json", "robust_scale"]


# -------------------------------------------------------------------------
# Core robust smoothing
# -------------------------------------------------------------------------


def robust_scale(values: np.ndarray) -> float:
    """
    Compute robust scale estimate using Median Absolute Deviation (MAD).

    MAD = median(|x - median(x)|)
    Scale estimate = 1.4826 * MAD  (consistent with Gaussian std)

    Falls back to standard deviation if MAD is zero.

    Parameters
    ----------
    values : np.ndarray
        Input values

    Returns
    -------
    float
        Robust scale estimate
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad > 0:
        return 1.4826 * mad
    std = np.std(vals)
    return float(std) if np.isfinite(std) else 0.0


def _evaluate_fit(
    t: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, str]:
    """
    Fit a smoothing spline or polynomial to masked data.

    Tries UnivariateSpline first (smooths noise), falls back to
    np.polyfit if spline fails or scipy unavailable.

    Parameters
    ----------
    t : np.ndarray
        Time values (seconds, same length as y)
    y : np.ndarray
        Data values to fit
    mask : np.ndarray
        Boolean mask, True = include in fit
    degree : int
        Polynomial/spline degree

    Returns
    -------
    fit : np.ndarray
        Fitted values at all t positions
    method : str
        'spline', 'poly', or 'none'
    """
    idx = np.where(mask)[0]
    if idx.size < 3:
        return y.copy(), "none"

    # Clamp degree to available data points
    deg = max(1, min(int(degree), idx.size - 1, 5))
    tx = t[idx]
    yy = y[idx]

    if UnivariateSpline is not None and tx.size >= deg + 2:
        try:
            # Deduplicate times (spline requires unique knots)
            txu, iu = np.unique(tx, return_index=True)
            yyu = yy[iu]
            if txu.size >= deg + 2:
                # Estimate noise from first differences
                rough = np.diff(yyu)
                if rough.size > 1:
                    noise = robust_scale(rough) * np.sqrt(2.0)
                else:
                    noise = robust_scale(yyu)

                # Fallback noise estimate
                if not np.isfinite(noise) or noise <= 0:
                    noise = robust_scale(yyu - np.median(yyu))

                # Smoothing parameter s = n * variance
                if np.isfinite(noise) and noise > 0:
                    s_val = float(txu.size) * (noise**2)
                else:
                    s_val = 0.0

                sp = UnivariateSpline(txu, yyu, k=deg, s=s_val)
                return sp(t), "spline"
        except Exception:
            pass

    # Fallback: polynomial fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeff = np.polyfit(tx, yy, deg)
    return np.polyval(coeff, t), "poly"


def _robust_fit_component(
    t: np.ndarray,
    y: np.ndarray,
    base_mask: np.ndarray,
    degree: float,
    sigma: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Robust iterative fitting for a single component.

    Iteratively:
      1. Fit the data (spline or polynomial)
      2. Compute residuals
      3. Reject outliers where |residual| > sigma * MAD
      4. Repeat until convergence or max_iter

    Parameters
    ----------
    t : np.ndarray
        Time array (n,)
    y : np.ndarray
        Data array (n,)
    base_mask : np.ndarray
        Boolean mask of valid points to consider
    degree : float
        Fitting degree
    sigma : float
        Outlier threshold in MAD units
    max_iter : int
        Maximum iterations

    Returns
    -------
    fit : np.ndarray
        Final fitted values at all t
    mask : np.ndarray
        Final inlier mask
    method : str
        'spline' or 'poly'
    """
    mask = base_mask.copy()
    fit = y.copy()
    method_used = "none"

    for _ in range(max_iter):
        fit, method = _evaluate_fit(t, y, mask, degree)
        method_used = method
        resid = y - fit
        scale = robust_scale(resid[mask])

        if not np.isfinite(scale) or scale <= 0:
            break

        new_mask = base_mask & (np.abs(resid) <= sigma * scale)

        min_points = max(3, int(degree) + 1)
        if new_mask.sum() < min_points:
            break
        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

    # Final fit on inliers
    fit, method_used = _evaluate_fit(t, y, mask, degree)
    return fit, mask, method_used


def _resolve_ignore_counts(
    nvec: int,
    degree: float,
    ignore_start: int,
    ignore_end: int,
) -> Tuple[int, int]:
    """
    Resolve auto-ignore counts for orbit start and end.

    Auto-ignore: 2% at start, 5% at end (when nvec >= 20).
    These edge regions tend to have degraded orbit accuracy.

    Parameters
    ----------
    nvec : int
        Total number of state vectors
    degree : float
        Fitting degree
    ignore_start, ignore_end : int
        Manual override (< 0 = auto, >= 0 = use as-is)

    Returns
    -------
    ig_start, ig_end : int
        Number of vectors to ignore at start / end
    """
    igs = int(ignore_start)
    ige = int(ignore_end)

    if igs < 0:
        igs = max(1, int(round(0.02 * nvec))) if nvec >= 20 else 1
    if ige < 0:
        ige = max(1, int(round(0.05 * nvec))) if nvec >= 20 else 1

    igs = max(0, igs)
    ige = max(0, ige)

    # Ensure we keep enough points for fitting
    min_keep = max(6, int(degree) + 1)
    max_drop = max(0, nvec - min_keep)

    if (igs + ige) > max_drop:
        if max_drop == 0:
            igs, ige = 0, 0
        else:
            total = igs + ige
            if total > 0:
                igs = int(round(max_drop * (igs / float(total))))
                ige = max_drop - igs
            else:
                igs, ige = 0, 0

    return igs, ige


# -------------------------------------------------------------------------
# Main public API
# -------------------------------------------------------------------------


def orbit_smooth(
    t: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    degree: float = 5,
    sigma: float = 4.0,
    max_iter: int = 3,
    ignore_start: int = -1,
    ignore_end: int = -1,
    return_outliers: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Smooth Lutan orbit state vectors with robust iterative spline fitting.

    Fits each XYZ component of position and velocity independently using
    scipy UnivariateSpline (smooth approximation) or polynomial fallback.
    Outliers are rejected iteratively using MAD-based robust statistics.

    Parameters
    ----------
    t : np.ndarray
        Time in GPS seconds (same reference for all state vectors),
        shape (n_sv,)
    pos : np.ndarray
        ECEF position, shape (n_sv, 3) — columns: X, Y, Z in meters
    vel : np.ndarray
        ECEF velocity, shape (n_sv, 3) — columns: Vx, Vy, Vz in m/s
    degree : float
        Spline/polynomial degree (default 5, max 5)
    sigma : float
        Outlier rejection threshold in MAD units (default 4.0)
    max_iter : int
        Maximum robust iterations (default 3)
    ignore_start, ignore_end : int
        Number of SVs to ignore at start/end of orbit
        (< 0 = auto: ~2% at start, ~5% at end)
    return_outliers : bool
        If True, also return outlier index array

    Returns
    -------
    pos_f : np.ndarray
        Smoothed ECEF position, same shape as pos
    vel_f : np.ndarray
        Smoothed ECEF velocity, same shape as vel
    info : dict
        Diagnostics:
          - methods: list of str, fitting method per component
          - ignore_start, ignore_end: actual ignore counts used
          - outlier_indices: array of outlier SV indices (if return_outliers=True)
          - n_outliers: int, number of outliers detected
          - used_spline: bool, True if spline was used (False = poly fallback)

    Example
    -------
    >>> from orbit_smooth import orbit_smooth
    >>> import json
    >>> orb = json.load(open('orbit.json'))
    >>> t = np.array([sv['gpsTime'] for sv in orb['stateVectors']])
    >>> pos = np.array([[sv['posX'], sv['posY'], sv['posZ']] for sv in orb['stateVectors']])
    >>> vel = np.array([[sv['velX'], sv['velY'], sv['velZ']] for sv in orb['stateVectors']])
    >>> pos_f, vel_f, info = orbit_smooth(t, pos, vel)
    >>> print(f"Detected {info['n_outliers']} outliers")
    """
    if UnivariateSpline is None:
        warnings.warn(
            "scipy.interpolate.UnivariateSpline not available. "
            "Using polynomial fallback (less smoothing)."
        )

    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)

    if t.ndim != 1 or pos.ndim != 2 or vel.ndim != 2:
        raise ValueError("t=(n,), pos=(n,3), vel=(n,3) expected")
    if pos.shape != vel.shape:
        raise ValueError("pos and vel must have the same shape")
    nvec = pos.shape[0]
    if nvec < 8:
        warnings.warn(f"Only {nvec} state vectors, orbit smoothing may be unreliable.")
        if return_outliers:
            return (
                pos,
                vel,
                {
                    "n_outliers": 0,
                    "methods": [],
                    "outlier_indices": np.array([], dtype=int),
                },
            )
        return pos, vel, {"n_outliers": 0, "methods": []}

    # --- Build base mask (excluding ignored edges) ---
    ig_start, ig_end = _resolve_ignore_counts(nvec, degree, ignore_start, ignore_end)
    base_mask = np.ones(nvec, dtype=bool)
    if ig_start > 0:
        base_mask[: min(ig_start, nvec)] = False
    if ig_end > 0:
        base_mask[max(0, nvec - ig_end) :] = False

    usable = base_mask.sum()
    min_needed = max(6, int(degree) + 1)
    if usable < min_needed:
        warnings.warn(
            f"Orbit smoothing skipped: usable SVs={usable}, need >={min_needed}. "
            "Returning original orbit."
        )
        info = {
            "n_outliers": 0,
            "methods": [],
            "ignore_start": 0,
            "ignore_end": 0,
            "used_spline": False,
        }
        if return_outliers:
            return pos, vel, info | {"outlier_indices": np.array([], dtype=int)}
        return pos, vel, info

    # --- Fit each component independently ---
    pos_f = np.zeros_like(pos)
    vel_f = np.zeros_like(vel)
    pos_masks = []
    vel_masks = []
    methods = []

    for comp, name in enumerate(["X", "Y", "Z"]):
        fit, mask, method = _robust_fit_component(
            t, pos[:, comp], base_mask, degree, sigma, max_iter
        )
        pos_f[:, comp] = fit
        pos_masks.append(mask)
        methods.append(f"pos_{name}={method}")

    for comp, name in enumerate(["Vx", "Vy", "Vz"]):
        fit, mask, method = _robust_fit_component(
            t, vel[:, comp], base_mask, degree, sigma, max_iter
        )
        vel_f[:, comp] = fit
        vel_masks.append(mask)
        methods.append(f"vel_{name}={method}")

    # --- Detect outliers: position OR velocity outlier in any component ---
    pos_ok = pos_masks[0] & pos_masks[1] & pos_masks[2]
    vel_ok = vel_masks[0] & vel_masks[1] & vel_masks[2]
    outlier_mask = base_mask & (~pos_ok | ~vel_ok)
    n_outliers = int(np.count_nonzero(outlier_mask))

    if n_outliers > 0:
        warnings.warn(
            f"Orbit smoothing detected {n_outliers} outliers "
            f"(ignore_start={ig_start}, ignore_end={ig_end}, degree={int(degree)})."
        )

    used_spline = any("spline" in m for m in methods)

    info = {
        "n_outliers": n_outliers,
        "methods": methods,
        "ignore_start": ig_start,
        "ignore_end": ig_end,
        "used_spline": used_spline,
    }

    if return_outliers:
        info["outlier_indices"] = np.where(outlier_mask)[0]

    return pos_f, vel_f, info


# -------------------------------------------------------------------------
# Convenience: read + smooth Lutan orbit JSON
# -------------------------------------------------------------------------


def smooth_orbit_from_json(
    orbit_json_path: str,
    orbit_ref_time: Optional[datetime] = None,
    degree: float = 5,
    sigma: float = 4.0,
    max_iter: int = 3,
    ignore_start: int = -1,
    ignore_end: int = -1,
) -> Dict[str, Any]:
    """
    Load Lutan orbit from JSON metadata and return smoothed orbit.

    Reads the orbit JSON (as produced by lutan_importer.py), applies
    robust smoothing, and returns a dict compatible with isce3 Orbit
    construction.

    Parameters
    ----------
    orbit_json_path : str
        Path to orbit JSON file (e.g., .../metadata/orbit.json)
    orbit_ref_time : datetime, optional
        Reference time for the orbit. If None, inferred from
        the first state vector GPS time.
    degree, sigma, max_iter, ignore_start, ignore_end
        Passed to orbit_smooth()

    Returns
    -------
    dict
        Smoothed orbit with keys:
          - refTime: datetime of orbit reference epoch
          - stateVectors: list of {timeUTC, gpsTime, posX, posY, posZ, velX, velY, velZ}
          - smoothed: True
          - info: diagnostics dict from orbit_smooth

    Example
    -------
    >>> smoothed = smooth_orbit_from_json('/tmp/lutan_output2/metadata/orbit.json')
    >>> # Build isce3 Orbit
    >>> svs = []
    >>> for sv in smoothed['stateVectors']:
    ...     t = datetime.fromisoformat(sv['timeUTC'].replace('Z', '+00:00'))
    ...     svs.append(isce3.core.StateVector(
    ...         isce3.core.DateTime(t),
    ...         np.array([sv['posX'], sv['posY'], sv['posZ']]),
    ...         np.array([sv['velX'], sv['velY'], sv['velZ']])))
    >>> orbit = isce3.core.Orbit(svs, isce3.core.DateTime(smoothed['refTime']),
    ...                          isce3.core.OrbitInterpMethod.Legendre)
    """
    import json

    orb = json.load(open(orbit_json_path))
    header = orb.get("header", {})
    state_vectors = orb.get("stateVectors", [])

    if not state_vectors:
        raise ValueError(f"No state vectors found in {orbit_json_path}")

    n = len(state_vectors)
    gps_times = np.array([sv["gpsTime"] for sv in state_vectors], dtype=np.float64)

    # Relative times from first SV (ISCE2 convention)
    t_rel = gps_times - gps_times[0]

    pos = np.array(
        [[sv["posX"], sv["posY"], sv["posZ"]] for sv in state_vectors], dtype=np.float64
    )
    vel = np.array(
        [[sv["velX"], sv["velY"], sv["velZ"]] for sv in state_vectors], dtype=np.float64
    )

    # Apply smoothing
    pos_f, vel_f, info = orbit_smooth(
        t_rel,
        pos,
        vel,
        degree=degree,
        sigma=sigma,
        max_iter=max_iter,
        ignore_start=ignore_start,
        ignore_end=ignore_end,
    )

    # Determine reference time
    if orbit_ref_time is None:
        ref_str = header.get("firstStateTimeUTC", state_vectors[0]["timeUTC"])
        orbit_ref_time = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))

    # Build output
    smoothed_sv = []
    for i, sv in enumerate(state_vectors):
        gps_t = float(gps_times[i])
        smoothed_sv.append(
            {
                "timeUTC": sv["timeUTC"],  # keep original UTC string
                "gpsTime": gps_t,
                "posX": float(pos_f[i, 0]),
                "posY": float(pos_f[i, 1]),
                "posZ": float(pos_f[i, 2]),
                "velX": float(vel_f[i, 0]),
                "velY": float(vel_f[i, 1]),
                "velZ": float(vel_f[i, 2]),
            }
        )

    result = {
        "refTime": orbit_ref_time.isoformat(),
        "stateVectors": smoothed_sv,
        "smoothed": True,
        "info": info,
    }

    print(
        f"[orbit_smooth] Smoothed {n} SVs: "
        f"n_outliers={info['n_outliers']}, "
        f"ignore_start={info['ignore_start']}, ignore_end={info['ignore_end']}, "
        f"methods={info['methods']}, "
        f"spline={info['used_spline']}"
    )

    return result


# -------------------------------------------------------------------------
# CLI entrypoint (for testing / batch processing)
# -------------------------------------------------------------------------


def _cli():
    import argparse, json, os

    parser = argparse.ArgumentParser(
        description="Smooth Lutan orbit state vectors using robust iterative spline fitting."
    )
    parser.add_argument("orbit_json", help="Input orbit JSON file")
    parser.add_argument("output_json", help="Output smoothed orbit JSON file")
    parser.add_argument(
        "--degree", type=int, default=5, help="Spline/polynomial degree (default: 5)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Outlier threshold in MAD units (default: 4.0)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=3, help="Max robust iterations (default: 3)"
    )
    parser.add_argument(
        "--ignore-start",
        type=int,
        default=-1,
        help="Ignore N SVs at start (default: auto ~2%%)",
    )
    parser.add_argument(
        "--ignore-end",
        type=int,
        default=-1,
        help="Ignore N SVs at end (default: auto ~5%%)",
    )
    args = parser.parse_args()

    smoothed = smooth_orbit_from_json(
        args.orbit_json,
        degree=args.degree,
        sigma=args.sigma,
        max_iter=args.max_iter,
        ignore_start=args.ignore_start,
        ignore_end=args.ignore_end,
    )

    with open(args.output_json, "w") as f:
        json.dump(smoothed, f, indent=2)

    print(f"[orbit_smooth] Wrote smoothed orbit to {args.output_json}")


if __name__ == "__main__":
    _cli()
