from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEED_OF_LIGHT = 299_792_458.0


def _normalize_swath(swath: str | int | None) -> str | None:
    if swath is None:
        return None
    text = str(swath).strip().upper()
    if not text:
        return None
    if text.isdigit():
        return f"IW{text}"
    return text


class SentinelImporter:
    def __init__(
        self,
        product_path: str | Path,
        swath: str | int | None = None,
        polarization: str | None = None,
        orbit_file: str | Path | None = None,
        orbit_dir: str | Path | None = None,
        download_orbit: bool = False,
    ):
        self.product_path = Path(product_path)
        self.swath = _normalize_swath(swath)
        self.polarization = polarization.lower() if polarization else None
        self.orbit_file = Path(orbit_file).resolve() if orbit_file is not None else None
        self.orbit_dir = Path(orbit_dir) if orbit_dir is not None else None
        self.download_orbit = download_orbit
        self.orbit_resolution = None
        self.is_zip = self.product_path.is_file() and self.product_path.suffix.lower() == ".zip"
        suffix = self.product_path.suffix.lower()
        stem_suffix = Path(self.product_path.stem).suffix.lower()
        self.is_tar = self.product_path.is_file() and (
            suffix in (".tar", ".tgz") or (suffix == ".gz" and stem_suffix == ".tar")
        )
        if not self.product_path.exists():
            raise FileNotFoundError(f"Input product path not found: {self.product_path}")
        if self.orbit_file is not None and not self.orbit_file.is_file():
            raise FileNotFoundError(f"Orbit file not found: {self.orbit_file}")

    def discover_files(self) -> dict[str, str]:
        if self.is_zip:
            candidates = self._discover_zip_members()
        elif self.is_tar:
            candidates = self._discover_tar_members()
        else:
            candidates = self._discover_dir_members()
        return self._select_members(candidates)

    def _discover_zip_members(self) -> dict[str, list[str]]:
        members: dict[str, list[str]] = {}
        with zipfile.ZipFile(self.product_path) as zf:
            for name in zf.namelist():
                self._classify_member(name, members)
        return members

    def _discover_tar_members(self) -> dict[str, list[str]]:
        members: dict[str, list[str]] = {}
        with tarfile.open(self.product_path) as tf:
            for name in tf.getnames():
                self._classify_member(name, members)
        return members

    def _discover_dir_members(self) -> dict[str, list[str]]:
        members: dict[str, list[str]] = {}
        for path in self.product_path.rglob("*"):
            if path.is_file():
                self._classify_member(str(path), members)
        return members

    def _classify_member(self, name: str, members: dict[str, list[str]]) -> None:
        normalized = name.replace("\\", "/")
        low = normalized.lower()
        if low.endswith("manifest.safe"):
            members.setdefault("manifest", []).append(name)
        elif "/measurement/" in low and (low.endswith(".tiff") or low.endswith(".tif")):
            members.setdefault("measurement", []).append(name)
        elif "/annotation/calibration/" in low and low.endswith(".xml"):
            filename = Path(low).name
            if filename.startswith("calibration-"):
                members.setdefault("calibration", []).append(name)
            elif filename.startswith("noise-"):
                members.setdefault("noise", []).append(name)
        elif "/annotation/" in low and low.endswith(".xml"):
            parent = Path(low).parent.name
            if parent != "annotation":
                return
            members.setdefault("annotation", []).append(name)

    def _select_members(self, candidates: dict[str, list[str]]) -> dict[str, str]:
        selected: dict[str, str] = {}
        for key in ("manifest", "annotation", "measurement", "calibration", "noise"):
            values = sorted(candidates.get(key, []))
            if not values:
                continue
            if key == "manifest":
                selected[key] = values[0]
                continue
            try:
                selected[key] = self._select_member_for_swath_pol(values)
            except FileNotFoundError:
                if key in {"annotation", "measurement"}:
                    raise
        return selected

    def _select_member_for_swath_pol(self, values: list[str]) -> str:
        matches = values
        if self.swath:
            token = f"-{self.swath.lower()}-"
            matches = [value for value in matches if token in Path(value).name.lower()]
        if self.polarization:
            token = f"-{self.polarization.lower()}-"
            matches = [value for value in matches if token in Path(value).name.lower()]
        elif matches:
            vv_matches = [value for value in matches if "-vv-" in Path(value).name.lower()]
            if vv_matches:
                matches = vv_matches
        if not matches:
            wanted = ""
            if self.swath:
                wanted += f" swath={self.swath}"
            if self.polarization:
                wanted += f" polarization={self.polarization.upper()}"
            raise FileNotFoundError(f"No Sentinel SAFE member matches{wanted.strip() or ' requested criteria'}")
        return sorted(matches)[0]

    def parse_xml_root(self, member_name: str) -> ET.Element:
        if self.is_zip:
            with zipfile.ZipFile(self.product_path) as zf:
                with zf.open(member_name) as f:
                    return ET.fromstring(f.read())
        if self.is_tar:
            with tarfile.open(self.product_path) as tf:
                extracted = tf.extractfile(member_name)
                if extracted is None:
                    raise FileNotFoundError(f"Archive member not found: {member_name}")
                with extracted:
                    return ET.fromstring(extracted.read())
        return ET.parse(member_name).getroot()

    def parse_manifest_root(self, member_name: str) -> ET.Element:
        return self.parse_xml_root(member_name)

    def build_member_path(self, member_name: str) -> str:
        if self.is_zip:
            return f"/vsizip/{self.product_path}/{member_name}"
        if self.is_tar:
            return f"/vsitar/{self.product_path}/{member_name}"
        return str(Path(member_name).resolve())

    def _relative_to_output(self, output_dir: Path, path_value: str | Path) -> str:
        path_str = str(path_value)
        if path_str.startswith("/vsizip/") or path_str.startswith("/vsitar/"):
            return path_str
        path_obj = Path(path_value)
        if not path_obj.is_absolute():
            path_obj = path_obj.resolve()
        return str(path_obj)

    def _manifest_ref(self, output_dir: Path, member_name: str) -> dict[str, str] | str:
        member_path = self.build_member_path(member_name)
        if self.is_zip:
            return {
                "path": self._relative_to_output(output_dir, member_path),
                "storage": "zip",
                "member": member_name,
            }
        if self.is_tar:
            return {
                "path": self._relative_to_output(output_dir, member_path),
                "storage": "tar",
                "member": member_name,
            }
        return self._relative_to_output(output_dir, member_path)

    def run(
        self,
        output_dir: str | Path = ".",
        dem_dir: str | None = None,
        download_dem: bool = False,
        dem_source: int = 1,
        dem_margin_deg: float = 0.1,
    ) -> str:
        del dem_source
        del dem_margin_deg
        files = self.discover_files()
        if not files.get("annotation"):
            raise FileNotFoundError(f"No Sentinel annotation XML found in {self.product_path}")
        if not files.get("measurement"):
            raise FileNotFoundError(f"No Sentinel measurement TIFF found in {self.product_path}")

        root = self.parse_xml_root(files["annotation"])
        acquisition = self.extract_acquisition(root)
        radar_grid = self.extract_radar_grid(root, acquisition)
        if self.orbit_file is None and (self.orbit_dir is not None or self.download_orbit):
            from sentinel_orbit import ProductInfo, resolve_orbit_for_product

            start_dt = _parse_datetime(acquisition.get("startTimeUTC"))
            stop_dt = _parse_datetime(acquisition.get("stopTimeUTC"))
            product_info = None
            if acquisition.get("platform") and start_dt is not None and stop_dt is not None:
                product_info = ProductInfo(
                    platform=str(acquisition.get("platform")),
                    start=start_dt.replace(tzinfo=None),
                    stop=stop_dt.replace(tzinfo=None),
                )
            self.orbit_resolution = resolve_orbit_for_product(
                self.product_path,
                orbit_dir=self.orbit_dir,
                download=self.download_orbit,
                work_dir=output_dir,
                product_info=product_info,
            )
            if self.orbit_resolution is not None:
                self.orbit_file = Path(self.orbit_resolution.path).resolve()
        orbit = self.extract_eof_orbit(self.orbit_file) if self.orbit_file else self.extract_orbit(root)
        doppler = self.extract_doppler(root)
        scene = self.extract_scene_info(root)
        tops = self.extract_tops_info(root, acquisition)
        processing = self.extract_processing_info(files.get("manifest"))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        metadata = {
            "acquisition": acquisition,
            "orbit": orbit,
            "radargrid": radar_grid,
            "doppler": doppler,
            "scene": scene,
            "tops": tops,
        }
        metadata_paths: dict[str, str] = {}
        for name, value in metadata.items():
            path = metadata_dir / f"{name}.json"
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
            metadata_paths[name] = self._relative_to_output(output_path, path)

        manifest = {
            "version": "1.0",
            "productType": "SLC",
            "sensor": "sentinel-1",
            "platform": acquisition.get("platform"),
            "polarisation": acquisition["polarisation"],
            "startTimeUTC": acquisition["startTimeUTC"],
            "stopTimeUTC": acquisition["stopTimeUTC"],
            "slc": {
                "path": self._manifest_ref(output_path, files["measurement"]),
                "format": "GeoTIFF",
                "complex": True,
                "sample_format": "cint16",
                "storage_layout": "single_band_complex",
                "complex_band_count": 1,
                "processing_format": "single_band_cfloat32",
                "rows": radar_grid["numberOfRows"],
                "columns": radar_grid["numberOfColumns"],
            },
            "metadata": metadata_paths,
            "tops": {
                "mode": tops["mode"],
                "swath": tops["swath"],
                "burst_count": len(tops["bursts"]),
            },
            "processing": processing,
            "ancillary": {
                "annotationXML": self._manifest_ref(output_path, files["annotation"]),
            },
        }
        if files.get("manifest"):
            manifest["ancillary"]["manifestSafe"] = self._manifest_ref(output_path, files["manifest"])
        if files.get("calibration"):
            manifest["ancillary"]["calibrationXML"] = self._manifest_ref(output_path, files["calibration"])
        if files.get("noise"):
            manifest["ancillary"]["noiseXML"] = self._manifest_ref(output_path, files["noise"])
        if self.orbit_file:
            manifest["ancillary"]["orbitFile"] = str(self.orbit_file)
            if self.orbit_resolution is not None:
                manifest["orbit"] = {
                    "source": self.orbit_resolution.source,
                    "orbitType": self.orbit_resolution.orbit_type,
                }
        if dem_dir:
            manifest["dem"] = {"path": str(Path(dem_dir).resolve()), "source": "local_directory"}
        elif download_dem:
            manifest["dem"] = {"path": None, "source": "not_resolved", "autoDownloaded": False}

        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(manifest_path)

    def extract_acquisition(self, root: ET.Element) -> dict[str, Any]:
        mission = _find_text(root, "missionId", "")
        start = _normalize_utc(_find_text(root, "startTime") or _find_text(root, "productFirstLineUtcTime"))
        stop = _normalize_utc(_find_text(root, "stopTime") or start)
        azimuth_time_interval = _find_float(root, "azimuthTimeInterval", 0.0)
        range_sampling_rate = _find_float(root, "rangeSamplingRate", 0.0)
        range_pixel_spacing = _find_float(root, "rangePixelSpacing", 0.0)
        if range_pixel_spacing == 0.0 and range_sampling_rate:
            range_pixel_spacing = SPEED_OF_LIGHT / (2.0 * range_sampling_rate)
        return {
            "source": "sentinel-1",
            "platform": mission,
            "mode": _find_text(root, "mode", "IW"),
            "swath": _find_text(root, "swath", ""),
            "polarisation": _find_text(root, "polarisation", ""),
            "startTimeUTC": start,
            "stopTimeUTC": stop,
            "startGPSTime": _parse_timestamp(start),
            "stopGPSTime": _parse_timestamp(stop),
            "centerFrequency": _find_float(root, "radarFrequency", 0.0),
            "prf": _find_float(root, "prf", 0.0) or _safe_reciprocal(azimuth_time_interval),
            "rangePixelSpacing": range_pixel_spacing,
            "rangeSamplingRate": range_sampling_rate,
            "azimuthPixelSpacing": _find_float(root, "azimuthPixelSpacing", 0.0),
            "azimuthTimeInterval": azimuth_time_interval,
            "slantRangeTime": _find_float(root, "slantRangeTime", 0.0),
            "startingRange": _find_float(root, "slantRangeTime", 0.0) * SPEED_OF_LIGHT / 2.0,
            "radarWavelength": _safe_reciprocal(_find_float(root, "radarFrequency", 0.0)) * SPEED_OF_LIGHT,
            "incidenceAngleMidSwath": _find_float(root, "incidenceAngleMidSwath", 0.0),
            "azimuthSteeringRate": _find_float(root, "azimuthSteeringRate", 0.0),
            "terrainHeight": _find_float(root, "value", 0.0),
            "ascendingNodeTimeUTC": _normalize_utc(_find_text(root, "ascendingNodeTime", "")),
            "passDirection": _find_text(root, "pass", ""),
            "lookDirection": "RIGHT",
        }

    def extract_radar_grid(self, root: ET.Element, acquisition: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "sentinel-1-annotation",
            "numberOfRows": _find_int(root, "numberOfLines", 0),
            "numberOfColumns": _find_int(root, "numberOfSamples", 0),
            "rowSpacing": acquisition.get("azimuthTimeInterval", 0.0),
            "columnSpacing": acquisition.get("rangePixelSpacing", 0.0),
            "rangeTimeFirstPixel": acquisition.get("slantRangeTime", 0.0),
            "startingRange": acquisition.get("startingRange", 0.0),
            "prf": acquisition.get("prf", 0.0),
            "wavelength": acquisition.get("radarWavelength", 0.0),
            "projection": _find_text(root, "projection", "Slant Range"),
            "swath": acquisition.get("swath"),
            "polarisation": acquisition.get("polarisation"),
        }

    def extract_orbit(self, root: ET.Element) -> dict[str, Any]:
        states = []
        for orbit in _iter_by_local_name(root, "orbit"):
            time_text = _child_text(orbit, "time")
            if not time_text:
                continue
            states.append(
                {
                    "timeUTC": _normalize_utc(time_text),
                    "timeGPS": _parse_timestamp(time_text),
                    "position": _vector_from_child(orbit, "position"),
                    "velocity": _vector_from_child(orbit, "velocity"),
                }
            )
        return {
            "source": "sentinel-1-annotation",
            "referenceFrame": "ECEF",
            "timeSystem": "GPS",
            "header": {"firstStateTimeUTC": states[0]["timeUTC"] if states else None},
            "stateVectors": states,
        }

    def extract_eof_orbit(self, orbit_file: Path) -> dict[str, Any]:
        root = ET.parse(orbit_file).getroot()
        states = []
        for osv in _iter_by_local_name(root, "OSV"):
            time_text = _child_text(osv, "UTC") or ""
            if time_text.startswith("UTC="):
                time_text = time_text[4:]
            states.append(
                {
                    "timeUTC": _normalize_utc(time_text),
                    "timeGPS": _parse_timestamp(time_text),
                    "position": {
                        "x": _float_or_default(_child_text(osv, "X"), 0.0),
                        "y": _float_or_default(_child_text(osv, "Y"), 0.0),
                        "z": _float_or_default(_child_text(osv, "Z"), 0.0),
                    },
                    "velocity": {
                        "x": _float_or_default(_child_text(osv, "VX"), 0.0),
                        "y": _float_or_default(_child_text(osv, "VY"), 0.0),
                        "z": _float_or_default(_child_text(osv, "VZ"), 0.0),
                    },
                    "quality": _child_text(osv, "Quality"),
                }
            )
        return {
            "source": "sentinel-1-eof",
            "orbitFile": str(orbit_file),
            "referenceFrame": "ECEF",
            "timeSystem": "GPS",
            "header": {"firstStateTimeUTC": states[0]["timeUTC"] if states else None},
            "stateVectors": states,
        }

    def extract_doppler(self, root: ET.Element) -> dict[str, Any]:
        estimates = []
        for estimate in _iter_by_local_name(root, "dcEstimate"):
            coeff_text = _child_text(estimate, "dataDcPolynomial") or ""
            estimates.append(
                {
                    "azimuthTimeUTC": _normalize_utc(_child_text(estimate, "azimuthTime") or ""),
                    "t0": _float_or_default(_child_text(estimate, "t0"), 0.0),
                    "coefficients": [_float_or_default(part, 0.0) for part in coeff_text.split()],
                }
            )
        return {"source": "sentinel-1-annotation", "type": "doppler_centroid", "estimates": estimates}

    def extract_scene_info(self, root: ET.Element) -> dict[str, Any]:
        points = []
        for point in _iter_by_local_name(root, "geolocationGridPoint"):
            lat = _float_or_default(_child_text(point, "latitude"), None)
            lon = _float_or_default(_child_text(point, "longitude"), None)
            if lat is not None and lon is not None:
                points.append({"lat": lat, "lon": lon})
        return {"source": "sentinel-1-annotation", "sceneCorners": _corners_from_points(points)}

    def extract_tops_info(self, root: ET.Element, acquisition: dict[str, Any]) -> dict[str, Any]:
        lines_per_burst = _find_int(root, "linesPerBurst", _find_int(root, "numberOfLines", 0))
        samples_per_burst = _find_int(root, "samplesPerBurst", _find_int(root, "numberOfSamples", 0))
        doppler_estimates = self.extract_doppler(root)["estimates"]
        fm_rate_estimates = _extract_polynomial_estimates(root, "azimuthFmRate", "azimuthFmRatePolynomial")
        bursts = []
        for idx, burst in enumerate(_iter_by_local_name(root, "burst"), start=1):
            first_valid = _int_list(_child_text(burst, "firstValidSample"))
            last_valid = _int_list(_child_text(burst, "lastValidSample"))
            sensing_start = _normalize_utc(_child_text(burst, "azimuthTime") or "")
            burst_start = _normalize_utc(_child_text(burst, "sensingTime") or sensing_start)
            sensing_stop = _add_seconds_to_utc(
                sensing_start,
                max(lines_per_burst - 1, 0) * float(acquisition.get("azimuthTimeInterval", 0.0)),
            )
            burst_stop = _add_seconds_to_utc(
                burst_start,
                max(lines_per_burst - 1, 0) / float(acquisition.get("prf", 1.0) or 1.0),
            )
            valid = _derive_valid_region(first_valid, last_valid)
            sensing_mid = _mid_time_utc(sensing_start, sensing_stop)
            bursts.append(
                {
                    "index": idx,
                    "azimuthTimeUTC": sensing_start,
                    "sensingStartUTC": sensing_start,
                    "sensingStopUTC": sensing_stop,
                    "sensingMidUTC": sensing_mid,
                    "burstStartUTC": burst_start,
                    "burstStopUTC": burst_stop,
                    "numberOfLines": lines_per_burst,
                    "numberOfSamples": samples_per_burst,
                    "lineOffset": (idx - 1) * lines_per_burst,
                    "startingRange": acquisition.get("startingRange", 0.0),
                    "rangePixelSize": acquisition.get("rangePixelSpacing", 0.0),
                    "rangeSamplingRate": acquisition.get("rangeSamplingRate", 0.0),
                    "azimuthTimeInterval": acquisition.get("azimuthTimeInterval", 0.0),
                    "azimuthSteeringRate": acquisition.get("azimuthSteeringRate", 0.0),
                    "prf": acquisition.get("prf", 0.0),
                    "radarWavelength": acquisition.get("radarWavelength", 0.0),
                    "terrainHeight": acquisition.get("terrainHeight", 0.0),
                    "firstValidSampleList": first_valid,
                    "lastValidSampleList": last_valid,
                    "firstValidLine": valid["firstValidLine"],
                    "numValidLines": valid["numValidLines"],
                    "firstValidSample": valid["firstValidSample"],
                    "lastValidSample": valid["lastValidSample"],
                    "numValidSamples": valid["numValidSamples"],
                    "doppler": _nearest_polynomial(doppler_estimates, sensing_mid),
                    "azimuthFMRate": _nearest_polynomial(fm_rate_estimates, sensing_mid),
                }
            )
        return {
            "source": "sentinel-1-annotation",
            "mode": acquisition.get("mode", "IW"),
            "swath": acquisition.get("swath", ""),
            "polarisation": acquisition.get("polarisation", ""),
            "linesPerBurst": lines_per_burst,
            "samplesPerBurst": samples_per_burst,
            "rangeWindowType": _find_text(root, "windowType", ""),
            "rangeWindowCoefficient": _find_float(root, "windowCoefficient", 0.0),
            "rangeProcessingBandwidth": _find_float(root, "processingBandwidth", 0.0),
            "azimuthWindowType": _find_nth_text(root, "windowType", 2, ""),
            "azimuthWindowCoefficient": _find_nth_float(root, "windowCoefficient", 2, 0.0),
            "azimuthProcessingBandwidth": _find_nth_float(root, "processingBandwidth", 2, 0.0),
            "overlaps": _derive_overlaps(bursts),
            "esd": _derive_esd_summary(bursts),
            "bursts": bursts,
        }

    def extract_processing_info(self, manifest_member: str | None) -> dict[str, Any]:
        if not manifest_member:
            return {"source": "manifest.safe", "facility": None, "softwareName": None, "softwareVersion": None}
        try:
            root = self.parse_manifest_root(manifest_member)
        except Exception:
            return {"source": "manifest.safe", "facility": None, "softwareName": None, "softwareVersion": None}
        facility = None
        software_name = None
        software_version = None
        for elem in root.iter():
            if _local_name(elem.tag) != "facility":
                continue
            site = elem.attrib.get("site")
            country = elem.attrib.get("country")
            if site and country:
                facility = f"{site}, {country}"
            elif site:
                facility = site
            for child in elem:
                if _local_name(child.tag) == "software":
                    software_name = child.attrib.get("name")
                    software_version = child.attrib.get("version")
                    break
            break
        return {
            "source": "manifest.safe",
            "facility": facility,
            "softwareName": software_name,
            "softwareVersion": software_version,
        }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _iter_by_local_name(root: ET.Element, name: str):
    for elem in root.iter():
        if _local_name(elem.tag) == name:
            yield elem


def _find_text(root: ET.Element, name: str, default: str | None = None) -> str | None:
    for elem in _iter_by_local_name(root, name):
        if elem.text is not None:
            return elem.text.strip()
    return default


def _find_nth_text(root: ET.Element, name: str, index: int, default: str | None = None) -> str | None:
    count = 0
    for elem in _iter_by_local_name(root, name):
        if elem.text is None:
            continue
        count += 1
        if count == index:
            return elem.text.strip()
    return default


def _child_text(root: ET.Element, name: str) -> str | None:
    for child in root:
        if _local_name(child.tag) == name and child.text is not None:
            return child.text.strip()
    return None


def _find_float(root: ET.Element, name: str, default: float) -> float:
    return _float_or_default(_find_text(root, name), default)


def _find_nth_float(root: ET.Element, name: str, index: int, default: float) -> float:
    return _float_or_default(_find_nth_text(root, name, index), default)


def _find_int(root: ET.Element, name: str, default: int) -> int:
    text = _find_text(root, name)
    try:
        return int(float(str(text)))
    except Exception:
        return default


def _float_or_default(value: Any, default: Any) -> Any:
    try:
        return float(value)
    except Exception:
        return default


def _safe_reciprocal(value: float) -> float:
    return 1.0 / value if value else 0.0


def _normalize_utc(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.endswith("Z"):
        return value[:-1]
    if "+" in value:
        return value.split("+", 1)[0]
    return value


def _parse_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    elif "+" not in text and not text.endswith("+00:00"):
        text = text + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    return (dt - gps_epoch).total_seconds()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    elif "+" not in text:
        text = text + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds")


def _add_seconds_to_utc(value: str | None, seconds: float) -> str:
    dt = _parse_datetime(value)
    if dt is None:
        return ""
    from datetime import timedelta

    return _format_utc(dt + timedelta(seconds=float(seconds)))


def _mid_time_utc(start: str | None, stop: str | None) -> str:
    start_dt = _parse_datetime(start)
    stop_dt = _parse_datetime(stop)
    if start_dt is None:
        return ""
    if stop_dt is None:
        return _format_utc(start_dt)
    return _format_utc(start_dt + (stop_dt - start_dt) / 2)


def _vector_from_child(root: ET.Element, name: str) -> dict[str, float]:
    for child in root:
        if _local_name(child.tag) == name:
            return {
                "x": _float_or_default(_child_text(child, "x"), 0.0),
                "y": _float_or_default(_child_text(child, "y"), 0.0),
                "z": _float_or_default(_child_text(child, "z"), 0.0),
            }
    return {"x": 0.0, "y": 0.0, "z": 0.0}


def _int_list(value: str | None) -> list[int]:
    if not value:
        return []
    result = []
    for part in value.split():
        try:
            result.append(int(float(part)))
        except Exception:
            continue
    return result


def _derive_valid_region(first_valid: list[int], last_valid: list[int]) -> dict[str, int]:
    valid_lines = [idx for idx, value in enumerate(first_valid) if value >= 0]
    if not valid_lines:
        return {
            "firstValidLine": 0,
            "numValidLines": 0,
            "firstValidSample": 0,
            "lastValidSample": -1,
            "numValidSamples": 0,
        }
    first_line = valid_lines[0]
    last_line = valid_lines[-1]
    first_sample = max(first_valid[first_line], first_valid[last_line])
    last_sample = min(last_valid[first_line], last_valid[last_line])
    num_samples = max(0, last_sample - first_sample)
    return {
        "firstValidLine": first_line,
        "numValidLines": last_line - first_line + 1,
        "firstValidSample": first_sample,
        "lastValidSample": last_sample,
        "numValidSamples": num_samples,
    }


def _extract_polynomial_estimates(root: ET.Element, entry_name: str, coeff_name: str) -> list[dict[str, Any]]:
    estimates = []
    for entry in _iter_by_local_name(root, entry_name):
        coeff_text = _child_text(entry, coeff_name) or ""
        estimates.append(
            {
                "azimuthTimeUTC": _normalize_utc(_child_text(entry, "azimuthTime") or ""),
                "t0": _float_or_default(_child_text(entry, "t0"), 0.0),
                "rangeMean": _float_or_default(_child_text(entry, "t0"), 0.0) * SPEED_OF_LIGHT / 2.0,
                "rangeNorm": SPEED_OF_LIGHT / 2.0,
                "coefficients": [_float_or_default(part, 0.0) for part in coeff_text.split()],
            }
        )
    return estimates


def _nearest_polynomial(estimates: list[dict[str, Any]], azimuth_time_utc: str) -> dict[str, Any] | None:
    if not estimates:
        return None
    target = _parse_datetime(azimuth_time_utc)
    if target is None:
        return estimates[0]
    return min(
        estimates,
        key=lambda item: abs(((_parse_datetime(item.get("azimuthTimeUTC")) or target) - target).total_seconds()),
    )


def _derive_overlaps(bursts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps = []
    for previous, current in zip(bursts, bursts[1:]):
        previous_stop = _parse_datetime(previous.get("sensingStopUTC"))
        current_start = _parse_datetime(current.get("sensingStartUTC"))
        azimuth_interval = float(previous.get("azimuthTimeInterval") or current.get("azimuthTimeInterval") or 0.0)
        overlap_seconds = 0.0
        if previous_stop is not None and current_start is not None:
            overlap_seconds = max(0.0, (previous_stop - current_start).total_seconds())
        estimated_lines = int(round(overlap_seconds / azimuth_interval)) if azimuth_interval > 0 else 0
        overlaps.append(
            {
                "previousBurstIndex": previous.get("index"),
                "nextBurstIndex": current.get("index"),
                "overlapStartUTC": current.get("sensingStartUTC"),
                "overlapStopUTC": previous.get("sensingStopUTC"),
                "estimatedOverlapSeconds": overlap_seconds,
                "estimatedOverlapLines": max(0, estimated_lines),
                "previousValidLines": previous.get("numValidLines", 0),
                "nextValidLines": current.get("numValidLines", 0),
            }
        )
    return overlaps


def _derive_esd_summary(bursts: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = _derive_overlaps(bursts)
    return {
        "method": "burst-overlap-esd",
        "overlapCount": len(overlaps),
        "readyForOverlapEstimation": len(overlaps) > 0,
        "requiresCommonBurstMatching": True,
    }


def _corners_from_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
    if not points:
        return []
    min_lat = min(point["lat"] for point in points)
    max_lat = max(point["lat"] for point in points)
    min_lon = min(point["lon"] for point in points)
    max_lon = max(point["lon"] for point in points)
    return [
        {"lat": max_lat, "lon": min_lon},
        {"lat": max_lat, "lon": max_lon},
        {"lat": min_lat, "lon": max_lon},
        {"lat": min_lat, "lon": min_lon},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Sentinel-1 SLC SAFE product")
    parser.add_argument("product_path", help="SAFE directory, ZIP, or TAR archive")
    parser.add_argument("output_dir", nargs="?", default=".", help="Output directory")
    parser.add_argument("--dem-dir", default=None)
    parser.add_argument("--download-dem", action="store_true")
    parser.add_argument("--orbit-dir", default=None)
    parser.add_argument("--download-orbit", action="store_true")
    args = parser.parse_args()

    manifest_path = SentinelImporter(
        args.product_path,
        orbit_dir=args.orbit_dir,
        download_orbit=args.download_orbit,
    ).run(
        args.output_dir,
        dem_dir=args.dem_dir,
        download_dem=args.download_dem,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
