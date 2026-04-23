from __future__ import annotations

SEVERITY_ORDER = {"ok": 0, "warn": 1, "fatal": 2}


def _max_severity(values: list[str]) -> str:
    return max(values, key=lambda item: SEVERITY_ORDER[item])


def _normalized(value) -> str:
    return str(value or "").strip().upper()


def _classify_boolean_match(matches: bool, fatal_reason: str) -> dict:
    return {"severity": "ok" if matches else "fatal", "reason": "match" if matches else fatal_reason}


def run_compatibility_precheck(
    *,
    master_acquisition: dict,
    slave_acquisition: dict,
    master_radargrid: dict,
    slave_radargrid: dict,
    master_doppler: dict,
    slave_doppler: dict,
    dc_policy: str,
    prf_policy: str,
    skip_precheck: bool,
) -> dict:
    if skip_precheck:
        return {
            "overall_severity": "ok",
            "checks": {},
            "dc_policy": dc_policy,
            "prf_policy": prf_policy,
            "skip_precheck": True,
            "recommended_geometry_mode": "zero-doppler" if dc_policy != "strict" else "native-doppler",
            "requires_prep": False,
        }

    checks: dict[str, dict] = {}

    master_cf = float(master_acquisition.get("centerFrequency", 0.0))
    slave_cf = float(slave_acquisition.get("centerFrequency", 0.0))
    cf_diff = abs(master_cf - slave_cf)
    cf_tol = max(master_cf, slave_cf, 1.0) * 1e-6
    checks["center_frequency"] = {
        "severity": "ok" if cf_diff <= cf_tol else "fatal",
        "reason": f"Δf={cf_diff}",
    }

    master_prf = float(master_acquisition.get("prf", 0.0))
    slave_prf = float(slave_acquisition.get("prf", 0.0))
    prf_diff = abs(master_prf - slave_prf)
    prf_ratio = prf_diff / max(master_prf, slave_prf, 1.0)
    if prf_ratio <= 1e-6:
        prf_severity = "ok"
        prf_reason = "PRF within negligible tolerance"
    elif prf_policy == "strict":
        prf_severity = "fatal"
        prf_reason = "PRF mismatch rejected by strict policy"
    elif prf_ratio <= 0.05:
        prf_severity = "warn"
        prf_reason = "PRF mismatch requires slave normalization/resampling"
    else:
        prf_severity = "fatal"
        prf_reason = "PRF mismatch too large for safe automatic normalization"
    checks["prf"] = {"severity": prf_severity, "reason": prf_reason, "master": master_prf, "slave": slave_prf}

    dc_available = bool(master_doppler) and bool(slave_doppler)
    if dc_policy == "strict":
        dc_severity = "ok" if dc_available else "fatal"
        dc_reason = "Doppler metadata available" if dc_available else "Doppler metadata unavailable under strict policy"
    elif dc_policy in {"auto", "zero"}:
        dc_severity = "ok" if dc_available else "warn"
        dc_reason = "Using available Doppler metadata" if dc_available else "Falling back to zero-Doppler geometry path"
    else:
        raise ValueError(f"Unsupported dc_policy '{dc_policy}'")
    checks["doppler"] = {"severity": dc_severity, "reason": dc_reason}

    look_match = _normalized(master_acquisition.get("lookDirection", "RIGHT")) == _normalized(slave_acquisition.get("lookDirection", "RIGHT"))
    checks["look_direction"] = _classify_boolean_match(look_match, "look direction mismatch")

    pol_match = _normalized(master_acquisition.get("polarisation")) == _normalized(slave_acquisition.get("polarisation"))
    checks["polarization"] = _classify_boolean_match(pol_match, "polarization mismatch")

    shape_match = int(master_radargrid.get("numberOfRows", 0)) == int(slave_radargrid.get("numberOfRows", 0)) and int(master_radargrid.get("numberOfColumns", 0)) == int(slave_radargrid.get("numberOfColumns", 0))
    spacing_match = float(master_radargrid.get("columnSpacing", 0.0)) == float(slave_radargrid.get("columnSpacing", 0.0))
    checks["radar_grid"] = {
        "severity": "ok" if shape_match and spacing_match else "warn",
        "reason": "match" if shape_match and spacing_match else "grid shape/spacing differs; slave normalization may be required",
    }

    overlap_available = float(master_acquisition.get("startGPSTime", 0.0)) > 0 and float(slave_acquisition.get("startGPSTime", 0.0)) > 0
    checks["time_overlap"] = {
        "severity": "ok" if overlap_available else "warn",
        "reason": "timestamps available" if overlap_available else "missing timing metadata for overlap sanity",
    }

    severities = [item["severity"] for item in checks.values()]
    overall = _max_severity(severities)
    requires_prep = any(checks[name]["severity"] == "warn" for name in ("prf", "doppler", "radar_grid"))

    return {
        "overall_severity": overall,
        "checks": checks,
        "dc_policy": dc_policy,
        "prf_policy": prf_policy,
        "skip_precheck": False,
        "recommended_geometry_mode": "zero-doppler" if dc_policy in {"auto", "zero"} else "native-doppler",
        "requires_prep": requires_prep,
    }
