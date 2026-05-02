from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen as default_urlopen


PRODUCT_RE = re.compile(r"^(S1[ABC])_.*?_(\d{8}T\d{6})_(\d{8}T\d{6})_")
ORBIT_RE = re.compile(r"^(S1[ABC])_.*?_AUX_(POEORB|RESORB)_.*?_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF(?:\.zip)?$")


@dataclass(frozen=True)
class ProductInfo:
    platform: str
    start: datetime
    stop: datetime


@dataclass(frozen=True)
class OrbitResult:
    path: str
    orbit_type: str
    source: str


def parse_product_filename(product_path: str | Path) -> ProductInfo:
    name = Path(product_path).name
    if name.lower().endswith(".safe"):
        name = name[:-5]
    elif name.lower().endswith(".zip"):
        name = name[:-4]
    match = PRODUCT_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse Sentinel-1 product filename: {product_path}")
    return ProductInfo(
        platform=match.group(1),
        start=datetime.strptime(match.group(2), "%Y%m%dT%H%M%S"),
        stop=datetime.strptime(match.group(3), "%Y%m%dT%H%M%S"),
    )


def resolve_orbit_for_product(
    product_path: str | Path,
    orbit_dir: str | Path | None = None,
    download: bool = False,
    work_dir: str | Path | None = None,
    downloader=None,
    product_info: ProductInfo | None = None,
) -> OrbitResult | None:
    info = product_info or parse_product_filename(product_path)
    search_dir = Path(orbit_dir) if orbit_dir is not None else None
    if search_dir is not None:
        match = _find_best_orbit(info, search_dir)
        if match is not None:
            return _result_for_path(match, "local")

    if not download:
        return None

    target_dir = search_dir or Path(work_dir or ".") / "orbits"
    target_dir.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        downloader = _fetch_orbit
    downloader(product_path, target_dir, product_info=info)
    match = _find_best_orbit(info, target_dir)
    if match is None:
        return None
    return _result_for_path(match, "download")


def apply_orbit_to_manifest(
    manifest_path: str | Path,
    orbit_dir: str | Path | None = None,
    download: bool = False,
    downloader=None,
) -> OrbitResult | None:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    product_path = _product_path_from_manifest(manifest, manifest_file.parent)
    product_info = _product_info_from_manifest(manifest)
    result = resolve_orbit_for_product(
        product_path,
        orbit_dir=orbit_dir,
        download=download,
        work_dir=manifest_file.parent,
        downloader=downloader,
        product_info=product_info,
    )
    if result is None:
        return None

    from sentinel_importer import SentinelImporter

    orbit = SentinelImporter(product_path, orbit_file=result.path).extract_eof_orbit(Path(result.path))
    orbit_metadata_path = Path(manifest["metadata"]["orbit"])
    if not orbit_metadata_path.is_absolute():
        orbit_metadata_path = manifest_file.parent / orbit_metadata_path
    orbit_metadata_path.write_text(json.dumps(orbit, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest.setdefault("ancillary", {})["orbitFile"] = result.path
    manifest["orbit"] = {"source": result.source, "orbitType": result.orbit_type}
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _find_best_orbit(info: ProductInfo, orbit_dir: Path) -> Path | None:
    candidates = []
    for path in orbit_dir.glob(f"{info.platform}*.EOF"):
        candidates.append(path)
    for path in orbit_dir.glob(f"{info.platform}*.EOF.zip"):
        candidates.append(path)
    for kind in ("POEORB", "RESORB"):
        for path in sorted(candidates):
            parsed = _parse_orbit_filename(path.name)
            if parsed is None:
                continue
            platform, orbit_kind, start, stop = parsed
            if platform == info.platform and orbit_kind == kind and info.start >= start + timedelta(seconds=60) and info.stop < stop - timedelta(seconds=60):
                return _materialize_orbit(path)
    return None


def _parse_orbit_filename(name: str):
    match = ORBIT_RE.match(name)
    if not match:
        return None
    return (
        match.group(1),
        match.group(2),
        datetime.strptime(match.group(3), "%Y%m%dT%H%M%S"),
        datetime.strptime(match.group(4), "%Y%m%dT%H%M%S"),
    )


def _materialize_orbit(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path.resolve()
    with zipfile.ZipFile(path) as zf:
        eof_members = [name for name in zf.namelist() if name.endswith(".EOF")]
        if not eof_members:
            raise RuntimeError(f"No .EOF member found in orbit archive {path}")
        member = eof_members[0]
        target = path.parent / Path(member).name
        if not target.exists():
            target.write_bytes(zf.read(member))
        return target.resolve()


def _result_for_path(path: Path, source: str) -> OrbitResult:
    orbit_type = "precise" if "AUX_POEORB" in path.name else "restituted"
    return OrbitResult(path=str(path.resolve()), orbit_type=orbit_type, source=source)


def download_orbit_file(
    info: ProductInfo,
    target_dir: str | Path,
    urlopen=default_urlopen,
    timeout: int = 120,
) -> str:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    errors = []
    for kind in ("POEORB", "RESORB"):
        for url in _candidate_orbit_urls(info, kind):
            try:
                with urlopen(url, timeout=timeout) as response:
                    payload = response.read()
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                continue

            zip_path = target / Path(url).name
            zip_path.write_bytes(payload)
            eof_path = _materialize_orbit(zip_path)
            if _find_best_orbit(info, target) is not None:
                return str(eof_path)
            errors.append(f"{url}: downloaded orbit does not cover product time")

    raise RuntimeError("Failed to download Sentinel-1 orbit with internal downloader: " + "; ".join(errors))


def _fetch_orbit(product_path: str | Path, target_dir: Path, product_info: ProductInfo | None = None) -> None:
    info = product_info or parse_product_filename(product_path)
    download_orbit_file(info, target_dir)


def _candidate_orbit_urls(info: ProductInfo, kind: str) -> list[str]:
    sensing_day = info.stop
    year = sensing_day.strftime("%Y")
    month = sensing_day.strftime("%m")
    dates = [sensing_day + timedelta(days=offset) for offset in range(0, 31)]
    names = []
    for publish in dates:
        publish_txt = publish.strftime("%Y%m%d")
        start_txt = (sensing_day - timedelta(days=1)).strftime("%Y%m%dT225942")
        stop_txt = (sensing_day + timedelta(days=1)).strftime("%Y%m%dT005942")
        names.append(
            f"{info.platform}_OPER_AUX_{kind}_OPOD_{publish_txt}T080803_V{start_txt}_{stop_txt}.EOF.zip"
        )

    urls = []
    for name in names:
        urls.append(f"https://step.esa.int/auxdata/orbits/Sentinel-1/{kind}/{info.platform}/{year}/{month}/{name}")
        if kind == "POEORB":
            urls.append(f"https://s1qc.asf.alaska.edu/aux_poeorb/{name[:-4]}")
    return urls


def _product_path_from_manifest(manifest: dict, base_dir: Path) -> str:
    manifest_safe = manifest.get("ancillary", {}).get("manifestSafe")
    if isinstance(manifest_safe, dict):
        path_value = manifest_safe.get("path") or ""
        member = manifest_safe.get("member") or ""
        return _archive_or_member_product_path(path_value, member)
    if isinstance(manifest_safe, str) and manifest_safe:
        path = Path(manifest_safe)
        if not path.is_absolute():
            path = base_dir / path
        return str(_safe_root_from_member_path(path))

    slc = manifest.get("slc", {}).get("path")
    if isinstance(slc, dict):
        return _archive_or_member_product_path(slc.get("path") or "", slc.get("member") or "")
    if isinstance(slc, str) and slc:
        return str(_safe_root_from_member_path(Path(slc)))
    raise ValueError("Cannot determine Sentinel product path from manifest")


def _product_info_from_manifest(manifest: dict) -> ProductInfo | None:
    platform = manifest.get("platform")
    start = manifest.get("startTimeUTC")
    stop = manifest.get("stopTimeUTC")
    if not platform or not start or not stop:
        return None
    return ProductInfo(platform=platform, start=_parse_manifest_time(start), stop=_parse_manifest_time(stop))


def _parse_manifest_time(value: str) -> datetime:
    text = value.rstrip("Z")
    if "+" in text:
        text = text.split("+", 1)[0]
    return datetime.fromisoformat(text)


def _archive_or_member_product_path(path_value: str, member: str) -> str:
    if path_value.startswith("/vsizip/"):
        archive = path_value[len("/vsizip/") :].split(".zip/", 1)[0] + ".zip"
        return archive
    if path_value.startswith("/vsitar/"):
        archive = path_value[len("/vsitar/") :].split(".tar/", 1)[0] + ".tar"
        return archive
    if member:
        return str(_safe_root_from_member_parts(Path(path_value).parent, member))
    return path_value


def _safe_root_from_member_path(path: Path) -> Path:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.lower().endswith(".safe"):
            return Path(*parts[: idx + 1])
    return path


def _safe_root_from_member_parts(base: Path, member: str) -> Path:
    parts = Path(member).parts
    for idx, part in enumerate(parts):
        if part.lower().endswith(".safe"):
            return base / Path(*parts[: idx + 1])
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve or apply Sentinel-1 EOF orbit files")
    sub = parser.add_subparsers(dest="command", required=True)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("product_path")
    download_parser.add_argument("--orbit-dir", required=True)
    download_parser.add_argument("--download", action="store_true")
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("manifest_path")
    apply_parser.add_argument("--orbit-dir", required=True)
    apply_parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        result = resolve_orbit_for_product(args.product_path, orbit_dir=args.orbit_dir, download=args.download)
    else:
        result = apply_orbit_to_manifest(args.manifest_path, orbit_dir=args.orbit_dir, download=args.download)
    if result is None:
        raise SystemExit("No matching orbit found")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
