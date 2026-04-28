import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TianyiImporter:
    def __init__(self, product_path: str):
        self.product_path = Path(product_path)
        self._archive_dir = None
        self.is_zip = False
        self.is_tar = False
        self._prepare_input_path()
        self.acquisition = {}
        self.orbit_data = {}
        self.radar_grid = {}
        self.doppler_data = {}
        self.scene_info = {}
        self.slc_path = ""
        self.annotation_name = ""
        self.calibration_name = ""
        self.manifest_name = ""

    def _prepare_input_path(self) -> None:
        if self.product_path.is_dir():
            self.is_zip = False
            return
        if self.product_path.is_file() and self.product_path.suffix.lower() == ".zip":
            self.is_zip = True
            return
        suffix = self.product_path.suffix.lower()
        stem_suffix = Path(self.product_path.stem).suffix.lower()
        if suffix in (".tar", ".tgz") or (suffix == ".gz" and stem_suffix == ".tar"):
            self.is_tar = True
            return
        if self.product_path.is_file():
            tmp_dir = Path(shutil.mkdtemp(prefix="tianyi_import_"))
            shutil.unpack_archive(str(self.product_path), str(tmp_dir))
            subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
            self.product_path = subdirs[0] if len(subdirs) == 1 else tmp_dir
            self._archive_dir = tmp_dir
            self.is_zip = False
            return
        raise FileNotFoundError(f"Input product path not found: {self.product_path}")

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

    def _discover_zip_members(self) -> dict[str, str]:
        members: dict[str, str] = {}
        with zipfile.ZipFile(self.product_path) as zf:
            for name in zf.namelist():
                low = name.lower()
                if (
                    low.endswith(".xml")
                    and "/annotation/" in low
                    and "/calibration/" not in low
                ):
                    members["annotation"] = name
                elif low.endswith(".xml") and "/annotation/calibration/" in low:
                    members["calibration"] = name
                elif low.endswith("manifest.safe"):
                    members["manifest"] = name
                elif low.endswith(".tiff") and "/measurement/" in low:
                    members["tiff"] = name
        return members

    def _discover_dir_members(self) -> dict[str, str]:
        members: dict[str, str] = {}
        for p in self.product_path.rglob("*"):
            low = str(p).lower()
            if (
                low.endswith(".xml")
                and "/annotation/" in low
                and "/calibration/" not in low
            ):
                members["annotation"] = str(p)
            elif low.endswith(".xml") and "/annotation/calibration/" in low:
                members["calibration"] = str(p)
            elif low.endswith("manifest.safe"):
                members["manifest"] = str(p)
            elif low.endswith(".tiff") and "/measurement/" in low:
                members["tiff"] = str(p)
        return members

    def _discover_tar_members(self) -> dict[str, str]:
        members: dict[str, str] = {}
        with tarfile.open(self.product_path) as tf:
            for name in tf.getnames():
                low = name.lower()
                if (
                    low.endswith(".xml")
                    and "/annotation/" in low
                    and "/annotation/calibration/" not in low
                ):
                    members["annotation"] = name
                elif low.endswith(".xml") and "/annotation/calibration/" in low:
                    members["calibration"] = name
                elif low.endswith("manifest.safe"):
                    members["manifest"] = name
                elif low.endswith(".tiff") and "/measurement/" in low:
                    members["tiff"] = name
        return members

    def discover_files(self) -> dict[str, str]:
        if self.is_zip:
            return self._discover_zip_members()
        if self.is_tar:
            return self._discover_tar_members()
        return self._discover_dir_members()

    def parse_xml_root(self, annotation_name: str) -> ET.Element:
        if self.is_zip:
            with zipfile.ZipFile(self.product_path) as zf:
                with zf.open(annotation_name) as f:
                    return ET.fromstring(f.read())
        if self.is_tar:
            with tarfile.open(self.product_path) as tf:
                with tf.extractfile(annotation_name) as f:
                    return ET.fromstring(f.read())
        return ET.parse(annotation_name).getroot()

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
        dem_margin_deg: float = 0.1,
        output_path: Path | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        from dem_manager import dem_from_scene_corners, find_dem_in_directory_for_scene

        if dem_dir:
            try:
                dem_path, check = find_dem_in_directory_for_scene(
                    scene_corners,
                    dem_dir,
                    margin_deg=dem_margin_deg,
                )
                # Convert to relative path for portability
                if output_path:
                    rel_dem_path = str(Path(dem_path).resolve().relative_to(output_path.resolve()))
                else:
                    rel_dem_path = str(Path(dem_path).resolve())
                return dem_path, {
                    "path": rel_dem_path,
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
            # Convert to relative path for portability
            if output_path:
                rel_dem_path = str(Path(dem_path).resolve().relative_to(output_path.resolve()))
            else:
                rel_dem_path = str(Path(dem_path).resolve())
            return dem_path, {
                "path": rel_dem_path,
                "source": f"SRTMGL{dem_source}",
                "directory": str(Path(dem_dir).resolve()) if dem_dir else None,
                "autoDownloaded": True,
            }
        return None, None

    def extract_acquisition(self, root: ET.Element) -> dict[str, Any]:
        header = root.find("adsHeader")
        info = root.find("generalAnnotation/productInformation")
        image_info = root.find("imageAnnotation/imageInformation")
        geo_pts = root.find("geolocationGrid/geolocationGridPointList").findall(
            "geolocationGridPoint"
        )

        lats = [float(pt.findtext("latitude", "0")) for pt in geo_pts]
        lons = [float(pt.findtext("longitude", "0")) for pt in geo_pts]
        incs = [float(pt.findtext("incidenceAngle", "0")) for pt in geo_pts]

        look_raw = image_info.findtext("look_side", "left").strip().lower()
        look_direction = "LEFT" if look_raw == "left" else "RIGHT"

        return {
            "source": "tianyi",
            "mission": header.findtext("missionId", ""),
            "sensorMode": header.findtext("mode", ""),
            "polarisation": header.findtext("polarisation", ""),
            "imagingMode": header.findtext("mode", ""),
            "lookDirection": look_direction,
            "centerFrequency": float(info.findtext("radarFrequency", "0")),
            "prf": float(
                info.findtext("prf", image_info.findtext("azimuthFrequency", "0"))
            ),
            "startTimeUTC": header.findtext("startTime", ""),
            "stopTimeUTC": header.findtext("stopTime", ""),
            "startGPSTime": self.parse_timestamp(header.findtext("startTime", "")),
            "stopGPSTime": self.parse_timestamp(header.findtext("stopTime", "")),
            "rangeTimeFirstPixel": float(image_info.findtext("slantRangeTime", "0")),
            "rangeTimeLastPixel": float(image_info.findtext("slantRangeTime", "0"))
            + (int(image_info.findtext("numberOfSamples", "1")) - 1)
            * float(image_info.findtext("rangePixelSpacing", "0"))
            * 2.0
            / 299792458.0,
            "centerLat": sum(lats) / len(lats),
            "centerLon": sum(lons) / len(lons),
            "headingAngle": float(info.findtext("platformHeading", "0")),
            "sceneAverageHeight": 0.0,
            "incidenceAngleCenter": sum(incs) / len(incs),
        }

    def extract_scene_info(self, root: ET.Element) -> dict[str, Any]:
        geo_pts = root.find("geolocationGrid/geolocationGridPointList").findall(
            "geolocationGridPoint"
        )
        corners = []
        for pt in geo_pts:
            azimuth_time = pt.findtext("azimuthTime", "")
            corners.append(
                {
                    "line": int(pt.findtext("line", "0")),
                    "pixel": int(pt.findtext("pixel", "0")),
                    "lat": float(pt.findtext("latitude", "0")),
                    "lon": float(pt.findtext("longitude", "0")),
                    "incidenceAngle": float(pt.findtext("incidenceAngle", "0")),
                    "slantRangeTime": float(pt.findtext("slantRangeTime", "0")),
                    "azimuthTime": azimuth_time,
                    "azimuthTimeUTC": azimuth_time,
                    "timeUTC": azimuth_time,
                }
            )
        return {"sceneCorners": corners}

    def extract_orbit(self, root: ET.Element) -> dict[str, Any]:
        orbit_list = root.find("generalAnnotation/orbitList").findall("orbit")
        state_vectors = []
        for orbit in orbit_list:
            ts = orbit.findtext("time", "")
            state_vectors.append(
                {
                    "timeUTC": ts,
                    "gpsTime": self.parse_timestamp(ts),
                    "posX": float(orbit.findtext("position/x", "0")),
                    "posY": float(orbit.findtext("position/y", "0")),
                    "posZ": float(orbit.findtext("position/z", "0")),
                    "velX": float(orbit.findtext("velocity/x", "0")),
                    "velY": float(orbit.findtext("velocity/y", "0")),
                    "velZ": float(orbit.findtext("velocity/z", "0")),
                }
            )
        spacing = 0.0
        if len(state_vectors) >= 2:
            spacing = state_vectors[1]["gpsTime"] - state_vectors[0]["gpsTime"]
        return {
            "header": {
                "generationSystem": "SAFE-like",
                "sensor": "Tianyi/BC3",
                "accuracy": "unknown",
                "stateVectorRefFrame": "Earth Fixed",
                "stateVectorRefTime": state_vectors[0]["timeUTC"],
                "stateVecFormat": "ECEF",
                "numStateVectors": len(state_vectors),
                "firstStateTimeUTC": state_vectors[0]["timeUTC"],
                "lastStateTimeUTC": state_vectors[-1]["timeUTC"],
                "stateVectorTimeSpacing": spacing,
            },
            "stateVectors": state_vectors,
        }

    def extract_radar_grid(self, root: ET.Element) -> dict[str, Any]:
        image_info = root.find("imageAnnotation/imageInformation")
        return {
            "numberOfRows": int(image_info.findtext("numberOfLines", "0")),
            "numberOfColumns": int(image_info.findtext("numberOfSamples", "0")),
            "rowSpacing": float(image_info.findtext("azimuthPixelSpacing", "0")),
            "columnSpacing": float(image_info.findtext("rangePixelSpacing", "0")),
            "groundRangeResolution": float(
                image_info.findtext("rangePixelSpacing", "0")
            ),
            "azimuthResolution": float(image_info.findtext("azimuthPixelSpacing", "0")),
            "prf": float(image_info.findtext("azimuthFrequency", "0")),
            "rangeLooks": 1,
            "azimuthLooks": 1,
            "geocodedFlag": "false",
            "rangeTimeFirstPixel": float(image_info.findtext("slantRangeTime", "0")),
            "rangeTimeLastPixel": float(image_info.findtext("slantRangeTime", "0"))
            + (int(image_info.findtext("numberOfSamples", "1")) - 1)
            * float(image_info.findtext("rangePixelSpacing", "0"))
            * 2.0
            / 299792458.0,
        }

    def extract_doppler(self, root: ET.Element) -> dict[str, Any]:
        estimates = root.find("dopplerCentroid/dcEstimateList").findall("dcEstimate")
        first = estimates[0]
        coeffs = [float(x) for x in first.findtext("dataDcPolynomial", "0").split()]
        if not coeffs:
            coeffs = [0.0]
        t0 = float(first.findtext("t0", "0"))
        image_info = root.find("imageAnnotation/imageInformation")
        last_range_time = (
            float(image_info.findtext("slantRangeTime", "0"))
            + (int(image_info.findtext("numberOfSamples", "1")) - 1)
            * float(image_info.findtext("rangePixelSpacing", "0"))
            * 2.0
            / 299792458.0
        )
        return {
            "azDeskew": "FALSE",
            "dopplerBasebandEstimationMethod": "dcEstimateList",
            "dopplerGeometricEstimationMethod": "dcEstimateList",
            "dopplerCentroidCoordinateType": "RAW",
            "dopplerEstimate": {
                "timeUTC": first.findtext("azimuthTime", ""),
                "dopplerAtMidRange": coeffs[0],
                "dopplerAmbiguity": 0,
                "geometricDopplerFlag": "false",
            },
            "combinedDoppler": {
                "validityRangeMin": t0,
                "validityRangeMax": last_range_time,
                "referencePoint": t0,
                "polynomialDegree": len(coeffs) - 1,
                "coefficients": coeffs,
            },
        }

    def run(
        self,
        output_dir: str = ".",
        dem_dir: str | None = None,
        download_dem: bool = False,
        dem_source: int = 1,
        dem_margin_deg: float = 0.1,
    ) -> str:
        from common_processing import verify_and_correct_look_direction

        files = self.discover_files()
        if not files.get("annotation"):
            raise FileNotFoundError(f"No annotation XML found in {self.product_path}")
        if not files.get("tiff"):
            raise FileNotFoundError(f"No measurement TIFF found in {self.product_path}")

        self.annotation_name = files.get("annotation", "")
        self.calibration_name = files.get("calibration", "")
        self.manifest_name = files.get("manifest", "")
        self.slc_path = self.build_slc_path(files["tiff"])

        root = self.parse_xml_root(self.annotation_name)
        self.acquisition = self.extract_acquisition(root)
        self.scene_info = self.extract_scene_info(root)
        self.orbit_data = self.extract_orbit(root)
        self.radar_grid = self.extract_radar_grid(root)
        self.doppler_data = self.extract_doppler(root)

        xml_look_direction = self.acquisition["lookDirection"]
        verified_look_direction, was_corrected = verify_and_correct_look_direction(
            self.orbit_data,
            self.scene_info["sceneCorners"],
            xml_look_direction,
            dem_height=0.0,
        )
        self.acquisition["lookDirection"] = verified_look_direction

        output_path = Path(output_dir)
        if dem_dir is None and download_dem:
            dem_dir = str(output_path / "dem")
        dem_path, dem_info = self.resolve_dem_option(
            self.scene_info["sceneCorners"],
            dem_dir=dem_dir,
            download_dem=download_dem,
            dem_source=dem_source,
            dem_margin_deg=dem_margin_deg,
            output_path=output_path,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        acquisition_file = metadata_dir / "acquisition.json"
        orbit_file = metadata_dir / "orbit.json"
        radargrid_file = metadata_dir / "radargrid.json"
        doppler_file = metadata_dir / "doppler.json"
        scene_file = metadata_dir / "scene.json"

        acquisition_file.write_text(
            json.dumps(self.acquisition, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        orbit_file.write_text(
            json.dumps(self.orbit_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        radargrid_file.write_text(
            json.dumps(self.radar_grid, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        doppler_file.write_text(
            json.dumps(self.doppler_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        scene_file.write_text(
            json.dumps(self.scene_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        slc_ref = self._manifest_ref(
            output_path, str(self.product_path.resolve()), files["tiff"]
        )

        manifest = {
            "version": "1.0",
            "productType": "SLC",
            "sensor": "tianyi",
            "polarisation": self.acquisition["polarisation"],
            "startTimeUTC": self.acquisition["startTimeUTC"],
            "stopTimeUTC": self.acquisition["stopTimeUTC"],
            "slc": {
                "path": slc_ref,
                "format": "TIFF",
                "complex": True,
                "sample_format": "cint16",
                "storage_layout": "single_band_complex",
                "complex_band_count": 1,
                "processing_format": "single_band_cfloat32",
                "rows": self.radar_grid["numberOfRows"],
                "columns": self.radar_grid["numberOfColumns"],
            },
            "metadata": {
                "acquisition": self._relative_to_output(output_path, acquisition_file),
                "orbit": self._relative_to_output(output_path, orbit_file),
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
        if self.manifest_name:
            manifest["ancillary"]["manifestSafe"] = self._manifest_ref(
                output_path, str(self.product_path.resolve()), self.manifest_name
            )
        if self.calibration_name:
            manifest["ancillary"]["calibrationXML"] = self._manifest_ref(
                output_path, str(self.product_path.resolve()), self.calibration_name
            )
        if self.annotation_name:
            manifest["ancillary"]["annotationXML"] = self._manifest_ref(
                output_path, str(self.product_path.resolve()), self.annotation_name
            )
        if dem_info:
            manifest["dem"] = dem_info

        manifest_file = output_path / "manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[TianyiImporter] Output written to {output_path}")
        print(f"  manifest: {manifest_file}")
        print(f"  acquisition: {acquisition_file}")
        print(f"  orbit: {orbit_file}")
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
        description="Import Tianyi/BC3 SAFE-like SLC product"
    )
    parser.add_argument(
        "product_path", help="ZIP file, archive file, or extracted product directory"
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
        help="DEM resolution for auto-download: 1=1arcsec, 3=3arcsec (default: 1)",
    )
    parser.add_argument(
        "--dem-margin-deg",
        type=float,
        default=0.05,
        help="Margin used when validating whether an existing DEM covers the scene",
    )
    args = parser.parse_args()

    importer = TianyiImporter(args.product_path)
    importer.run(
        args.output_dir_opt or args.output_dir,
        dem_dir=args.dem_dir,
        download_dem=args.download_dem,
        dem_source=args.dem_source,
        dem_margin_deg=args.dem_margin_deg,
    )
