import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from common_processing import construct_orbit
from geo2rdr_compat import llh_to_xyz


def regularized_sentinel_orbit(orbits_xml):
    orbit_json = {
        "header": {"firstStateTimeUTC": orbits_xml[0].findtext("time", "")},
        "stateVectors": [
            {
                "timeUTC": orb.findtext("time", ""),
                "posX": float(orb.findtext("position/x", "0")),
                "posY": float(orb.findtext("position/y", "0")),
                "posZ": float(orb.findtext("position/z", "0")),
                "velX": float(orb.findtext("velocity/x", "0")),
                "velY": float(orb.findtext("velocity/y", "0")),
                "velZ": float(orb.findtext("velocity/z", "0")),
            }
            for orb in orbits_xml
        ],
    }
    return construct_orbit(orbit_json, "Hermite")


def compare_sentinel(zip_path: str) -> None:
    import isce3.core
    import isce3.geometry

    with zipfile.ZipFile(zip_path) as zf:
        ann = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".xml")
            and "/annotation/" in n.lower()
            and "/calibration/" not in n.lower()
            and "/rfi/" not in n.lower()
            and "iw1" in n.lower()
            and "vv" in n.lower()
        ][0]
        root = ET.fromstring(zf.read(ann))

    point = root.find("geolocationGrid/geolocationGridPointList").findall(
        "geolocationGridPoint"
    )[0]
    product_info = root.find("generalAnnotation/productInformation")
    orbit = regularized_sentinel_orbit(
        root.find("generalAnnotation/orbitList").findall("orbit")
    )
    wavelength = isce3.core.speed_of_light / float(
        product_info.findtext("radarFrequency", "0")
    )
    llh = np.array(
        [
            [np.deg2rad(float(point.findtext("longitude", "0")))],
            [np.deg2rad(float(point.findtext("latitude", "0")))],
            [float(point.findtext("height", "0"))],
        ],
        dtype=np.float64,
    )
    xyz = llh_to_xyz(float(llh[0, 0]), float(llh[1, 0]), float(llh[2, 0]))
    zero = isce3.core.LUT2d()
    ellipsoid = isce3.core.Ellipsoid()

    print(
        {
            "dataset": "sentinel",
            "expectedLine": point.findtext("line"),
            "expectedPixel": point.findtext("pixel"),
        }
    )
    for side_name, side in (
        ("right", isce3.core.LookSide.Right),
        ("left", isce3.core.LookSide.Left),
    ):
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
                    "dataset": "sentinel",
                    "solver": "geo2rdr",
                    "side": side_name,
                    "status": "success",
                    "azt": azt,
                    "sr": sr,
                }
            )
        except Exception as exc:
            print(
                {
                    "dataset": "sentinel",
                    "solver": "geo2rdr",
                    "side": side_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
        try:
            azt, sr = isce3.geometry.geo2rdr_bracket(
                xyz, orbit, zero, wavelength, side_name
            )
            print(
                {
                    "dataset": "sentinel",
                    "solver": "geo2rdr_bracket",
                    "side": side_name,
                    "status": "success",
                    "azt": azt,
                    "sr": sr,
                }
            )
        except Exception as exc:
            print(
                {
                    "dataset": "sentinel",
                    "solver": "geo2rdr_bracket",
                    "side": side_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )


def compare_tianyi(metadata_dir: str) -> None:
    import isce3.core
    import isce3.geometry

    with open(f"{metadata_dir}/orbit.json", encoding="utf-8") as f:
        orbit_json = json.load(f)
    with open(f"{metadata_dir}/acquisition.json", encoding="utf-8") as f:
        acquisition = json.load(f)
    with open(f"{metadata_dir}/scene.json", encoding="utf-8") as f:
        scene = json.load(f)

    point = scene["sceneCorners"][0]
    orbit = construct_orbit(orbit_json, "Hermite")
    wavelength = isce3.core.speed_of_light / acquisition["centerFrequency"]
    llh = np.array(
        [[np.deg2rad(point["lon"])], [np.deg2rad(point["lat"])], [0.0]],
        dtype=np.float64,
    )
    xyz = llh_to_xyz(point["lon"], point["lat"], 0.0)
    zero = isce3.core.LUT2d()
    ellipsoid = isce3.core.Ellipsoid()

    print(
        {
            "dataset": "tianyi",
            "expectedLine": point["line"],
            "expectedPixel": point["pixel"],
        }
    )
    for side_name, side in (
        ("left", isce3.core.LookSide.Left),
        ("right", isce3.core.LookSide.Right),
    ):
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
                    "dataset": "tianyi",
                    "solver": "geo2rdr",
                    "side": side_name,
                    "status": "success",
                    "azt": azt,
                    "sr": sr,
                }
            )
        except Exception as exc:
            print(
                {
                    "dataset": "tianyi",
                    "solver": "geo2rdr",
                    "side": side_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
        try:
            azt, sr = isce3.geometry.geo2rdr_bracket(
                xyz, orbit, zero, wavelength, side_name
            )
            print(
                {
                    "dataset": "tianyi",
                    "solver": "geo2rdr_bracket",
                    "side": side_name,
                    "status": "success",
                    "azt": azt,
                    "sr": sr,
                }
            )
        except Exception as exc:
            print(
                {
                    "dataset": "tianyi",
                    "solver": "geo2rdr_bracket",
                    "side": side_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Sentinel and Tianyi geo2rdr behavior"
    )
    parser.add_argument(
        "--sentinel-zip", help="Sentinel SLC ZIP containing annotation XML"
    )
    parser.add_argument(
        "--tianyi-metadata-dir",
        help="Directory containing Tianyi orbit/acquisition/scene JSON",
    )
    args = parser.parse_args()

    if not args.sentinel_zip and not args.tianyi_metadata_dir:
        parser.error(
            "at least one of --sentinel-zip or --tianyi-metadata-dir is required"
        )
    if args.sentinel_zip:
        compare_sentinel(args.sentinel_zip)
    if args.tianyi_metadata_dir:
        compare_tianyi(args.tianyi_metadata_dir)


if __name__ == "__main__":
    main()
