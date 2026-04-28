import json
import os
import copy
import shutil
import sys
import xml.etree.ElementTree as ET
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LutanImporter:
    NAMESPACES = {"lt": ""}

    def __init__(self, product_path: str):
        self.product_path = Path(product_path)
        self._archive_dir = None
        self.is_zip = False
        self.is_tar = False
        self._prepare_input_path()
        self.meta_xml = None
        self.tiff_file = None
        self.incidence_xml = None
        self.rpc_file = None
        self.acquisition = {}
        self.orbit_data = {}
        self.radar_grid = {}
        self.doppler_data = {}
        self.scene_data = {}
        self.raw_orbit_data = {}

    def _prepare_input_path(self) -> None:
        if self.product_path.is_dir():
            print(f"[LutanImporter] Using directory: {self.product_path}")
            self.is_zip = False
            return
        if self.product_path.is_file() and self.product_path.suffix.lower() == ".zip":
            print(f"[LutanImporter] Using ZIP file: {self.product_path}")
            self.is_zip = True
            return
        suffix = self.product_path.suffix.lower()
        stem_suffix = Path(self.product_path.stem).suffix.lower()
        if suffix in (".tar", ".tgz") or (suffix == ".gz" and stem_suffix == ".tar"):
            print(f"[LutanImporter] Using TAR file: {self.product_path}")
            self.is_tar = True
            return
        if self.product_path.is_file():
            print(f"[LutanImporter] Unpacking archive: {self.product_path}")
            tmp_dir = Path(shutil.mkdtemp(prefix="lutan_import_"))
            shutil.unpack_archive(str(self.product_path), str(tmp_dir))
            subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
            self.product_path = subdirs[0] if len(subdirs) == 1 else tmp_dir
            self._archive_dir = tmp_dir
            self.is_zip = False
            print(f"[LutanImporter] Unpacked to: {self.product_path}")
            return
        raise FileNotFoundError(f"Input product path not found: {self.product_path}")

    def discover_files(self) -> dict[str, str]:
        print(f"[LutanImporter] Discovering files in: {self.product_path}")
        if self.is_zip:
            files = self._discover_zip_members()
        elif self.is_tar:
            files = self._discover_tar_members()
        else:
            files = self._discover_dir_members()
        print(f"[LutanImporter] Discovered files: {list(files.keys())}")
        return files

    def _discover_tar_members(self) -> dict[str, str]:
        files = {}
        with tarfile.open(self.product_path) as tf:
            for name in tf.getnames():
                upper = name.upper()
                if upper.endswith(".TIFF") and "SLC" in upper:
                    files["tiff"] = name
                elif upper.endswith(".META.XML"):
                    files["meta_xml"] = name
                elif upper.endswith(".INCIDENCE.XML"):
                    files["incidence_xml"] = name
                elif upper.endswith(".RPC"):
                    files["rpc"] = name
        return files

    def _discover_zip_members(self) -> dict[str, str]:
        files = {}
        with zipfile.ZipFile(self.product_path) as zf:
            for name in zf.namelist():
                upper = name.upper()
                if name.endswith(".tiff") and "SLC" in upper:
                    files["tiff"] = name
                elif name.endswith(".meta.xml"):
                    files["meta_xml"] = name
                elif name.endswith(".incidence.xml"):
                    files["incidence_xml"] = name
                elif name.endswith(".rpc"):
                    files["rpc"] = name
        return files

    def _discover_dir_members(self) -> dict[str, str]:
        files = {}
        for p in self.product_path.iterdir():
            name = p.name
            if name.endswith(".tiff") and "SLC" in name.upper():
                files["tiff"] = str(p)
            elif name.endswith(".meta.xml"):
                files["meta_xml"] = str(p)
            elif name.endswith(".incidence.xml"):
                files["incidence_xml"] = str(p)
            elif name.endswith(".rpc"):
                files["rpc"] = str(p)
        return files

    def parse_timestamp(self, ts: str) -> float:
        gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        elif not (ts[-6:-5] in ("+", "-") and ts[-6:].count(":") == 3):
            ts = ts + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt - gps_epoch).total_seconds()

    def parse_meta_xml(self, path: str | Path) -> ET.Element:
        if self.is_zip:
            with zipfile.ZipFile(self.product_path) as zf:
                with zf.open(str(path)) as f:
                    return ET.fromstring(f.read())
        if self.is_tar:
            with tarfile.open(self.product_path) as tf:
                with tf.extractfile(str(path)) as f:
                    return ET.fromstring(f.read())
        tree = ET.parse(Path(path))
        root = tree.getroot()
        return root

    def build_slc_path(self, tiff_name: str) -> str:
        if self.is_zip:
            return f"/vsizip/{self.product_path}/{tiff_name}"
        if self.is_tar:
            return f"/vsitar/{self.product_path}/{tiff_name}"
        return str(Path(tiff_name).resolve())

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

    def _manifest_ref(
        self, output_dir: Path, path_value: str | Path, member: str | None = None
    ):
        if member is not None:
            if self.is_tar:
                return {
                    "path": self._relative_to_output(output_dir, path_value),
                    "storage": "tar",
                    "member": member,
                }
            if self.is_zip:
                return {
                    "path": self._relative_to_output(output_dir, path_value),
                    "storage": "zip",
                    "member": member,
                }
            return {
                "path": self._relative_to_output(output_dir, path_value),
            }
        return self._relative_to_output(output_dir, path_value)

    def resolve_dem_option(
        self,
        scene_corners: list[dict[str, Any]],
        dem_dir: str | None = None,
        download_dem: bool = False,
        dem_source: int = 1,
        dem_margin_deg: float = 0.05,
    ) -> tuple[str | None, dict[str, Any] | None]:
        from dem_manager import dem_from_scene_corners, find_dem_in_directory_for_scene

        if dem_dir:
            try:
                dem_path, check = find_dem_in_directory_for_scene(
                    scene_corners,
                    dem_dir,
                    margin_deg=dem_margin_deg,
                )
                return dem_path, {
                    "path": dem_path,
                    "source": "local_directory",
                    "directory": str(Path(dem_dir).resolve()),
                    "autoDownloaded": False,
                    "validation": check,
                }
            except FileNotFoundError:
                if not download_dem:
                    raise

        if download_dem:
            dem_path = dem_from_scene_corners(
                scene_corners,
                output_dir=dem_dir,
                source=dem_source,
            )
            return dem_path, {
                "path": dem_path,
                "source": f"SRTMGL{dem_source}",
                "directory": str(Path(dem_dir).resolve()) if dem_dir else None,
                "autoDownloaded": True,
            }
        return None, None

    def extract_general_header(self, root: ET.Element) -> dict[str, Any]:
        gh = root.find("generalHeader")
        return {
            "mission": gh.findtext("mission", ""),
            "sensorMode": gh.findtext("sensorMode", ""),
            "generationTime": gh.findtext("generationTime", ""),
        }

    def extract_acquisition_info(self, root: ET.Element) -> dict[str, Any]:
        ai = root.find("productInfo/acquisitionInfo")
        return {
            "sensor": ai.findtext("sensor", ""),
            "imagingMode": ai.findtext("imagingMode", ""),
            "lookDirection": ai.findtext("lookDirection", ""),
            "polarisationMode": ai.findtext("polarisationMode", ""),
            "elevationBeamConfiguration": ai.findtext("elevationBeamConfiguration", ""),
        }

    def extract_image_data_info(self, root: ET.Element) -> dict[str, Any]:
        idi = root.find("productInfo/imageDataInfo")
        raster = idi.find("imageRaster")
        return {
            "rowSpacing": float(raster.findtext("rowSpacing", "0").split()[0]),
            "columnSpacing": float(raster.findtext("columnSpacing", "0").split()[0]),
            "groundRangeResolution": float(
                raster.findtext("groundRangeResolution", "0")
            ),
            "azimuthResolution": float(raster.findtext("azimuthResolution", "0")),
            "numberOfRows": int(raster.findtext("numberOfRows", "0")),
            "numberOfColumns": int(raster.findtext("numberOfColumns", "0")),
        }

    def extract_scene_info(self, root: ET.Element) -> dict[str, Any]:
        si = root.find("productInfo/sceneInfo")
        center = si.find("sceneCenterCoord")
        corners = {}
        for corner in si.findall("sceneCornerCoord"):
            name = corner.get("name", "")
            corners[name] = {
                "refRow": int(corner.findtext("refRow", "0")),
                "refColumn": int(corner.findtext("refColumn", "0")),
                "lat": float(corner.findtext("lat", "0")),
                "lon": float(corner.findtext("lon", "0")),
                "azimuthTimeUTC": corner.findtext("azimuthTimeUTC", ""),
                "rangeTime": float(corner.findtext("rangeTime", "0")),
                "incidenceAngle": float(corner.findtext("incidenceAngle", "0")),
            }
        return {
            "sceneID": si.findtext("sceneID", ""),
            "startTimeUTC": si.find("start").findtext("timeUTC", ""),
            "stopTimeUTC": si.find("stop").findtext("timeUTC", ""),
            "rangeTimeFirstPixel": float(si.find("rangeTime/firstPixel").text),
            "rangeTimeLastPixel": float(si.find("rangeTime/lastPixel").text),
            "sceneAzimuthExtent": float(si.findtext("sceneAzimuthExtent", "0")),
            "sceneRangeExtent": float(si.findtext("sceneRangeExtent", "0")),
            "sceneCenterCoord": {
                "refRow": int(center.findtext("refRow", "0")),
                "refColumn": int(center.findtext("refColumn", "0")),
                "lat": float(center.findtext("lat", "0")),
                "lon": float(center.findtext("lon", "0")),
                "azimuthTimeUTC": center.findtext("azimuthTimeUTC", ""),
                "rangeTime": float(center.findtext("rangeTime", "0")),
                "incidenceAngle": float(center.findtext("incidenceAngle", "0")),
            },
            "sceneAverageHeight": float(si.findtext("sceneAverageHeight", "0")),
            "sceneCorners": corners,
            "headingAngle": float(si.findtext("headingAngle", "0")),
        }

    def extract_orbit(self, root: ET.Element) -> dict[str, Any]:
        orbit_elem = root.find("platform/orbit")
        header = orbit_elem.find("orbitHeader")
        state_vectors = []
        for sv in orbit_elem.findall("stateVec"):
            time_str = sv.findtext("timeUTC", "")
            state_vectors.append(
                {
                    "timeUTC": time_str,
                    "gpsTime": self.parse_timestamp(time_str),
                    "posX": float(sv.findtext("posX", "0")),
                    "posY": float(sv.findtext("posY", "0")),
                    "posZ": float(sv.findtext("posZ", "0")),
                    "velX": float(sv.findtext("velX", "0")),
                    "velY": float(sv.findtext("velY", "0")),
                    "velZ": float(sv.findtext("velZ", "0")),
                }
            )
        return {
            "header": {
                "generationSystem": header.findtext("generationSystem", ""),
                "sensor": header.findtext("sensor", ""),
                "accuracy": header.findtext("accuracy", ""),
                "stateVectorRefFrame": header.findtext("stateVectorRefFrame", ""),
                "stateVectorRefTime": header.findtext("stateVectorRefTime", ""),
                "stateVecFormat": header.findtext("stateVecFormat", ""),
                "numStateVectors": int(header.findtext("numStateVectors", "0")),
                "firstStateTimeUTC": header.find("firstStateTime").findtext(
                    "firstStateTimeUTC", ""
                ),
                "lastStateTimeUTC": header.find("lastStateTime").findtext(
                    "lastStateTimeUTC", ""
                ),
                "stateVectorTimeSpacing": float(
                    header.findtext("stateVectorTimeSpacing", "0")
                ),
            },
            "stateVectors": state_vectors,
        }

    def _smooth_orbit_for_import(self, orbit: dict[str, Any]) -> dict[str, Any]:
        from orbit_smooth import orbit_smooth

        state_vectors = orbit.get("stateVectors", [])
        if len(state_vectors) < 8:
            smoothed = copy.deepcopy(orbit)
            smoothed["smoothed"] = False
            smoothed["smoothing"] = {
                "algorithm": "isce2-lutan1-orbit-filter",
                "status": "skipped-insufficient-state-vectors",
                "state_vector_count": len(state_vectors),
            }
            return smoothed

        import numpy as np

        gps_times = np.array([float(sv["gpsTime"]) for sv in state_vectors], dtype=np.float64)
        t_rel = gps_times - gps_times[0]
        pos = np.array(
            [[sv["posX"], sv["posY"], sv["posZ"]] for sv in state_vectors],
            dtype=np.float64,
        )
        vel = np.array(
            [[sv["velX"], sv["velY"], sv["velZ"]] for sv in state_vectors],
            dtype=np.float64,
        )

        degree = 5
        sigma = 4.0
        max_iter = 3
        ignore_start = -1
        ignore_end = -1
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

        smoothed = copy.deepcopy(orbit)
        for idx, sv in enumerate(smoothed["stateVectors"]):
            sv["posX"] = float(pos_f[idx, 0])
            sv["posY"] = float(pos_f[idx, 1])
            sv["posZ"] = float(pos_f[idx, 2])
            sv["velX"] = float(vel_f[idx, 0])
            sv["velY"] = float(vel_f[idx, 1])
            sv["velZ"] = float(vel_f[idx, 2])

        smoothed["smoothed"] = True
        smoothed["smoothing"] = {
            "algorithm": "isce2-lutan1-orbit-filter",
            "status": "applied",
            "degree": degree,
            "sigma": sigma,
            "max_iter": max_iter,
            "ignore_start": info.get("ignore_start"),
            "ignore_end": info.get("ignore_end"),
            "n_outliers": info.get("n_outliers", 0),
            "methods": info.get("methods", []),
            "used_spline": info.get("used_spline", False),
        }
        return smoothed

    def extract_doppler(self, root: ET.Element) -> dict[str, Any]:
        doppler_elem = root.find("processing/doppler")
        centroid = doppler_elem.find("dopplerCentroid")
        estimate = centroid.find("dopplerEstimate")
        combined = estimate.find("combinedDoppler")
        return {
            "azDeskew": doppler_elem.findtext("azDeskew", ""),
            "dopplerBasebandEstimationMethod": doppler_elem.findtext(
                "dopplerBasebandEstimationMethod", ""
            ),
            "dopplerGeometricEstimationMethod": doppler_elem.findtext(
                "dopplerGeometricEstimationMethod", ""
            ),
            "dopplerCentroidCoordinateType": doppler_elem.findtext(
                "dopplerCentroidCoordinateType", ""
            ),
            "dopplerEstimate": {
                "timeUTC": estimate.findtext("timeUTC", ""),
                "dopplerAtMidRange": float(estimate.findtext("dopplerAtMidRange", "0")),
                "dopplerAmbiguity": int(estimate.findtext("dopplerAmbiguity", "0")),
                "geometricDopplerFlag": estimate.findtext("geometricDopplerFlag", ""),
            },
            "combinedDoppler": {
                "validityRangeMin": float(combined.findtext("validityRangeMin", "0")),
                "validityRangeMax": float(combined.findtext("validityRangeMax", "0")),
                "referencePoint": float(combined.findtext("referencePoint", "0")),
                "polynomialDegree": int(combined.findtext("polynomialDegree", "0")),
                "coefficients": [
                    float(combined.findtext(f"coefficient[@exponent='{e}']", "0"))
                    for e in range(int(combined.findtext("polynomialDegree", "0")) + 1)
                ],
            },
        }

    def extract_radar_params(self, root: ET.Element) -> dict[str, Any]:
        inst = root.find("instrument")
        radar = inst.find("radarParameters")
        settings = inst.find("settings")
        prf = settings.find("settingRecord").find("PRF")
        return {
            "centerFrequency": float(radar.findtext("centerFrequency", "0")),
            "prf": float(prf.text),
            "rxGain": float(settings.find("rxGainSetting/rxGain").text),
            "rxBandwidth": float(settings.findtext("rxBandwidth", "0")),
        }

    def extract_processing_info(self, root: ET.Element) -> dict[str, Any]:
        proc = root.find("processing")
        return {
            "rangeLooks": int(proc.findtext("processingParameter/rangeLooks", "1")),
            "azimuthLooks": int(proc.findtext("processingParameter/azimuthLooks", "1")),
            "geocodedFlag": proc.findtext("processingFlags/geocodedFlag", ""),
        }

    def run(
        self,
        output_dir: str = ".",
        dem_dir: str | None = None,
        download_dem: bool = False,
        dem_source: int = 1,
        dem_margin_deg: float = 0.05,
    ) -> str:
        from common_processing import verify_and_correct_look_direction

        print(f"[LutanImporter] Starting import process")
        print(f"[LutanImporter] Output directory: {output_dir}")

        files = self.discover_files()
        self.tiff_file = files.get("tiff")
        self.meta_xml = files.get("meta_xml")
        self.incidence_xml = files.get("incidence_xml")
        self.rpc_file = files.get("rpc")

        if not self.meta_xml:
            raise FileNotFoundError(f"No .meta.xml found in {self.product_path}")
        if not self.tiff_file:
            raise FileNotFoundError(f"No SLC .tiff found in {self.product_path}")

        print(f"[LutanImporter] Reading metadata from: {self.meta_xml}")
        root = self.parse_meta_xml(self.meta_xml)

        print("[LutanImporter] Extracting metadata...")
        general = self.extract_general_header(root)
        acq_info = self.extract_acquisition_info(root)
        image_info = self.extract_image_data_info(root)
        scene_info = self.extract_scene_info(root)
        orbit = self.extract_orbit(root)
        doppler = self.extract_doppler(root)
        radar_params = self.extract_radar_params(root)
        processing = self.extract_processing_info(root)

        print(f"[LutanImporter] Scene: {scene_info['sceneID']}")
        print(f"[LutanImporter] Sensor: {acq_info['sensor']}")
        print(f"[LutanImporter] Mode: {acq_info['imagingMode']}")
        print(f"[LutanImporter] Polarisation: {acq_info['polarisationMode']}")
        print(
            f"[LutanImporter] Size: {image_info['numberOfRows']}x{image_info['numberOfColumns']}"
        )
        print(
            f"[LutanImporter] Time range: {scene_info['startTimeUTC']} to {scene_info['stopTimeUTC']}"
        )

        xml_look_direction = acq_info["lookDirection"]
        print(f"[LutanImporter] Verifying look direction...")
        verified_look_direction, was_corrected = verify_and_correct_look_direction(
            orbit,
            list(scene_info["sceneCorners"].values()),
            xml_look_direction,
            dem_height=scene_info["sceneAverageHeight"],
        )

        self.acquisition = {
            "source": "lutan",
            "mission": general["mission"],
            "sensorMode": general["sensorMode"],
            "polarisation": acq_info["polarisationMode"],
            "imagingMode": acq_info["imagingMode"],
            "lookDirection": verified_look_direction,
            "centerFrequency": radar_params["centerFrequency"],
            "prf": radar_params["prf"],
            "startTimeUTC": scene_info["startTimeUTC"],
            "stopTimeUTC": scene_info["stopTimeUTC"],
            "startGPSTime": self.parse_timestamp(scene_info["startTimeUTC"]),
            "stopGPSTime": self.parse_timestamp(scene_info["stopTimeUTC"]),
            "rangeTimeFirstPixel": scene_info["rangeTimeFirstPixel"],
            "rangeTimeLastPixel": scene_info["rangeTimeLastPixel"],
            "centerLat": scene_info["sceneCenterCoord"]["lat"],
            "centerLon": scene_info["sceneCenterCoord"]["lon"],
            "headingAngle": scene_info["headingAngle"],
            "sceneAverageHeight": scene_info["sceneAverageHeight"],
            "incidenceAngleCenter": scene_info["sceneCenterCoord"]["incidenceAngle"],
        }

        self.raw_orbit_data = copy.deepcopy(orbit)
        print("[LutanImporter] Smoothing orbit with ISCE2-compatible Lutan1 filter...")
        self.orbit_data = self._smooth_orbit_for_import(orbit)
        smoothing = self.orbit_data.get("smoothing", {})
        print(
            "[LutanImporter] Orbit smoothing "
            f"status={smoothing.get('status')} "
            f"n_outliers={smoothing.get('n_outliers', 0)} "
            f"ignore_start={smoothing.get('ignore_start')} "
            f"ignore_end={smoothing.get('ignore_end')}"
        )
        self.radar_grid = {
            "numberOfRows": image_info["numberOfRows"],
            "numberOfColumns": image_info["numberOfColumns"],
            "rowSpacing": image_info["rowSpacing"],
            "columnSpacing": image_info["columnSpacing"],
            "groundRangeResolution": image_info["groundRangeResolution"],
            "azimuthResolution": image_info["azimuthResolution"],
            "prf": radar_params["prf"],
            "rangeLooks": processing["rangeLooks"],
            "azimuthLooks": processing["azimuthLooks"],
            "geocodedFlag": processing["geocodedFlag"],
            "rangeTimeFirstPixel": scene_info["rangeTimeFirstPixel"],
            "rangeTimeLastPixel": scene_info["rangeTimeLastPixel"],
        }

        self.doppler_data = doppler
        self.scene_data = {
            "sceneCenterCoord": scene_info["sceneCenterCoord"],
            "sceneCorners": list(scene_info["sceneCorners"].values()),
            "sceneAverageHeight": scene_info["sceneAverageHeight"],
            "headingAngle": scene_info["headingAngle"],
        }

        if dem_dir is None and download_dem:
            dem_dir = str(Path(output_dir) / "dem")
            print(f"[LutanImporter] Using default DEM directory: {dem_dir}")

        if dem_dir:
            print(f"[LutanImporter] DEM directory: {dem_dir}")
            print(
                f"[LutanImporter] DEM source: SRTMGL{dem_source} (1=1arcsec, 3=3arcsec)"
            )
            print(f"[LutanImporter] DEM margin: {dem_margin_deg} degrees")

        dem_path, dem_info = self.resolve_dem_option(
            list(scene_info["sceneCorners"].values()),
            dem_dir=dem_dir,
            download_dem=download_dem,
            dem_source=dem_source,
            dem_margin_deg=dem_margin_deg,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata_dir = output_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        print("[LutanImporter] Writing metadata files...")
        acquisition_file = metadata_dir / "acquisition.json"
        with open(acquisition_file, "w", encoding="utf-8") as f:
            json.dump(self.acquisition, f, indent=2, ensure_ascii=False)

        orbit_file = metadata_dir / "orbit.json"
        with open(orbit_file, "w", encoding="utf-8") as f:
            json.dump(self.orbit_data, f, indent=2, ensure_ascii=False)

        raw_orbit_file = metadata_dir / "orbit.raw.json"
        with open(raw_orbit_file, "w", encoding="utf-8") as f:
            json.dump(self.raw_orbit_data, f, indent=2, ensure_ascii=False)

        radargrid_file = metadata_dir / "radargrid.json"
        with open(radargrid_file, "w", encoding="utf-8") as f:
            json.dump(self.radar_grid, f, indent=2, ensure_ascii=False)

        doppler_file = metadata_dir / "doppler.json"
        with open(doppler_file, "w", encoding="utf-8") as f:
            json.dump(self.doppler_data, f, indent=2, ensure_ascii=False)

        scene_file = metadata_dir / "scene.json"
        with open(scene_file, "w", encoding="utf-8") as f:
            json.dump(self.scene_data, f, indent=2, ensure_ascii=False)

        slc_ref = self._manifest_ref(
            output_path, str(self.product_path.resolve()), self.tiff_file
        )

        manifest = {
            "version": "1.0",
            "productType": "SLC",
            "sensor": "lutan",
            "polarisation": self.acquisition["polarisation"],
            "startTimeUTC": self.acquisition["startTimeUTC"],
            "stopTimeUTC": self.acquisition["stopTimeUTC"],
            "slc": {
                "path": slc_ref,
                "format": "TIFF",
                "complex": True,
                "sample_format": "iq_int16",
                "storage_layout": "two_band_iq",
                "complex_band_count": 1,
                "band_mapping": {"real": 1, "imag": 2},
                "processing_format": "single_band_cfloat32",
                "rows": self.radar_grid["numberOfRows"],
                "columns": self.radar_grid["numberOfColumns"],
            },
            "metadata": {
                "acquisition": self._relative_to_output(output_path, acquisition_file),
                "orbit": self._relative_to_output(output_path, orbit_file),
                "orbit_raw": self._relative_to_output(output_path, raw_orbit_file),
                "radargrid": self._relative_to_output(output_path, radargrid_file),
                "doppler": self._relative_to_output(output_path, doppler_file),
                "scene": self._relative_to_output(output_path, scene_file),
            },
            "look_direction": {
                "xml_value": xml_look_direction,
                "verified_value": verified_look_direction,
                "was_corrected": was_corrected,
            },
            "ancillary": {},
        }

        if self.incidence_xml:
            manifest["ancillary"]["incidenceXML"] = self._manifest_ref(
                output_path, str(self.product_path.resolve()), self.incidence_xml
            )
        if self.meta_xml:
            manifest["ancillary"]["metaXML"] = self._manifest_ref(
                output_path, str(self.product_path.resolve()), self.meta_xml
            )
        if dem_info:
            manifest["dem"] = dem_info

        manifest_file = output_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[LutanImporter] Output written to {output_path}")
        print(f"  manifest: {manifest_file}")
        print(f"  acquisition: {acquisition_file}")
        print(f"  orbit: {orbit_file}")
        print(f"  orbit_raw: {raw_orbit_file}")
        print(f"  radargrid: {radargrid_file}")
        print(f"  doppler: {doppler_file}")
        print(f"  scene: {scene_file}")
        print(
            f"  look_direction: xml={xml_look_direction} -> verified={verified_look_direction} "
            f"(corrected={was_corrected})"
        )
        if dem_path:
            print(f"  dem: {dem_path}")

        return str(manifest_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Import Lutan SAR product from an extracted directory or ZIP archive"
    )
    parser.add_argument(
        "product_path", help="Lutan product directory, ZIP file, or archive file"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Output directory for manifest and metadata (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir_opt",
        default=None,
        help="Output directory for manifest and metadata (overrides positional output_dir)",
    )
    parser.add_argument(
        "--dem-dir",
        default=None,
        help="Directory containing existing DEM files, or directory to store auto-downloaded DEMs",
    )
    parser.add_argument(
        "--download-dem",
        action="store_true",
        help="Auto-download DEM if needed for the scene",
    )
    parser.add_argument(
        "--dem-source",
        type=int,
        default=1,
        choices=[1, 3],
        help="DEM resolution for auto-download: 1=1arcsec SRTMGL1, 3=3arcsec (default: 1)",
    )
    parser.add_argument(
        "--dem-margin-deg",
        type=float,
        default=0.05,
        help="Margin used when validating whether an existing DEM covers the scene",
    )
    args = parser.parse_args()

    importer = LutanImporter(args.product_path)
    importer.run(
        args.output_dir_opt or args.output_dir,
        dem_dir=args.dem_dir,
        download_dem=args.download_dem,
        dem_source=args.dem_source,
        dem_margin_deg=args.dem_margin_deg,
    )
