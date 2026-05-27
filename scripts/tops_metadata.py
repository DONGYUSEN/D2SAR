"""tops_metadata — Sentinel-1 SAFE / manifest / annotation XML parser.

Parses Sentinel-1 Level-1 SLC SAFE products (directory, ZIP, or TAR)
into ``tops_model.BurstRadarGrid`` objects, grouped by IW swath.

Key invariants enforced here
----------------------------
* ``sensing_start`` and ``sensing_stop`` are always timezone-aware UTC
  ``datetime`` objects.
* ``numValidLines > 0`` and ``numValidSamples > 0`` otherwise ``ValueError``.
* ``doppler_coefficients`` and ``azimuth_fm_rate_coefficients``: at least one
  non-zero value must exist across both tuples (otherwise ``ValueError``).
* The ``index`` field found in annotation XML is normalised to 0-based
  ``burst_index`` before being stored in ``BurstIdentity``.

Supports BOTH formats of ``manifest.safe``:
  - XML (standard ESA SAFE format, e.g. S1A IW SLC products)
  - JSON (alternate format used by some processors)

No external dependencies beyond stdlib and ``scripts/tops_model``.
"""

from __future__ import annotations

import json
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
)

UTC = timezone.utc

# Sentinel-1 C-band wavelength (m)
SENTINEL1_WAVELENGTH = 0.05546576

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_sentinel1_safe(path: Path) -> Dict[str, List[BurstRadarGrid]]:
    """Parse a Sentinel-1 SAFE directory or archive into a dict of IW → BurstRadarGrid.

    Parameters
    ----------
    path:
        Path to a SAFE directory, ``.zip``, or ``.tar`` / ``.tar.gz`` file.

    Returns
    -------
    dict[str, list[BurstRadarGrid]]
        Mapping such as ``{"IW1": [...bursts...], "IW2": [...], "IW3": [...]}``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".zip",):
        _archive_type = "zip"
    elif suffix in (".tar", ".tgz") or (suffix == ".gz" and path.stem.endswith(".tar")):
        _archive_type = "tar"
    elif path.is_dir():
        _archive_type = "dir"
    else:
        raise ValueError(f"Unsupported SAFE format: {path}")

    # Build a resolved root path depending on archive type
    if _archive_type == "dir":
        root = path
    else:
        root = path  # we open by name; handle inside each function

    manifest: Dict[str, Any] = _load_manifest(root, _archive_type, path)
    annotation_files: Dict[str, str] = _iw_annotation_xmls(manifest, _archive_type, path, root)

    bursts_by_swath: Dict[str, List[BurstRadarGrid]] = {}
    for swath, xml_rel in annotation_files.items():
        bursts = _parse_iw_bursts(
            xml_rel,
            archive_type=_archive_type,
            archive_path=path,
            archive_root=root,
        )
        bursts_by_swath[swath] = bursts

    return bursts_by_swath


def parse_sensing_time(value: Any) -> datetime:
    """Parse a time value to a timezone-aware UTC ``datetime``.

    Accepts either:
    * An ISO-8601 string (``"2024-01-01T00:00:00.000Z"`` or similar), or
    * A numeric Unix-epoch value (int or float, treated as seconds UTC).

    Returns a ``datetime`` with ``tzinfo = timezone.utc``.
    """
    if isinstance(value, str):
        text = value.strip()
        # Strip trailing Z (and any space before it)
        text = text.rstrip("Z").rstrip()
        # Normalise "+00:00" variants that fromisoformat handles directly
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    # Numeric epoch
    return datetime.fromtimestamp(float(value), tz=UTC)


# ---------------------------------------------------------------------------
# Internal helpers — manifest
# ---------------------------------------------------------------------------


def _load_manifest(
    root: Path,
    archive_type: str,  # "zip" | "tar" | "dir"
    archive_path: Path,
) -> Dict[str, Any]:
    """Load and parse ``manifest.safe`` into a dict.

    Supports both JSON and XML formats:
    - JSON: alternate manifest format from some processors
    - XML: standard ESA SAFE manifest.safe format

    Parameters
    ----------
    root : unused for archives; kept for API symmetry.
    archive_type : ``"zip"``, ``"tar"``, or ``"dir"``
    archive_path : path to the SAFE archive or directory.
    """
    del root  # unused for archives
    manifest_path = "manifest.safe"

    # Read raw bytes from the appropriate source
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            raw = zf.read(manifest_path)
    elif archive_type == "tar":
        with tarfile.open(archive_path) as tf:
            member = tf.getmember(manifest_path)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(
                    f"Cannot extract manifest {manifest_path!r} from {archive_path}"
                )
            raw = extracted.read()
    else:
        # directory
        manifest_file = archive_path / manifest_path
        raw = manifest_file.read_bytes()

    # Try JSON first, fall back to XML
    text = raw.decode("utf-8").strip()
    if text.startswith("<?xml") or text.startswith("<"):
        return _parse_xml_manifest(raw)
    else:
        return json.loads(text)


def _parse_xml_manifest(raw: bytes) -> Dict[str, Any]:
    """Parse ESA-style XML manifest.safe into a JSON-compatible dict.

    Extracts the annotation file paths for each IW swath (VH and VV polarizations)
    and structures them like the JSON manifest format expected by the pipeline.

    The standard ESA manifest.safe uses XML with XFDU structure. Annotation files
    are referenced via <fileLocation> elements within <dataObject> elements.
    """
    root = ET.fromstring(raw)
    NS = {"xfdu": "urn:ccsds:schema:xfdu:1", "safe": "http://www.esa.int/safe/sentinel-1.0"}

    result: Dict[str, Any] = {"acquisition": {"iw": []}}

    # Iterate through all fileLocation elements in the manifest
    for elem in root.iter():
        tag_local = elem.tag.rsplit("}", 1)[-1] if "}" in elem.tag else elem.tag
        if tag_local != "fileLocation":
            continue

        href = elem.get("href", "")
        # Only care about annotation XML files (not calibration, noise, rfi)
        if not href.endswith(".xml") or "/calibration/" in href or "/noise-" in href or "/rfi-" in href:
            continue

        # Parse path like: ./annotation/s1a-iw1-slc-vv-20230625t114146-...xml
        filename = href.lstrip("./").lstrip("/")
        if not filename.startswith("annotation/"):
            continue

        # Extract swath and polarization from filename pattern
        # Pattern: s1a-iw{N}-slc-{POL}-YYYYMMDDTHHMMSS-...xml
        parts = Path(filename).stem.split("-")
        swath = None
        pol = None
        for part in parts:
            low = part.lower()
            if low.startswith("iw") and len(low) == 3 and low[2] in "123":
                swath = "IW" + low[2]
            elif low in ("vv", "vh", "hh", "hv"):
                pol = low.upper()

        if swath and pol:
            # Find or create entry for this swath+pol
            iw_list: List[Dict[str, Any]] = result["acquisition"]["iw"]
            entry = None
            for e in iw_list:
                if e.get("swath") == swath and e.get("polarisation", "").upper() == pol:
                    entry = e
                    break
            if entry is None:
                entry = {"swath": swath, "polarisation": pol, "annotation": ""}
                iw_list.append(entry)
            entry["annotation"] = filename

    return result


def _iw_annotation_xmls(
    manifest: Dict[str, Any],
    archive_type: str,
    archive_path: Path,
    root: Path,
) -> Dict[str, str]:
    """Extract the annotation XML path for each IW swath from the manifest.

    Parameters
    ----------
    manifest : parsed ``manifest.safe`` dict.
    archive_type : ``"zip"``, ``"tar"``, or ``"dir"``
    archive_path : path to the SAFE archive or directory.
    root : unused for archives.

    Returns
    -------
    dict[str, str]
        Mapping such as ``{"IW1": "annotation/iw1/s1a-iw1-slc-vv-...xml", ...}``.
        Empty strings for swaths without an annotation entry.
    """
    del root  # unused
    acq: Dict[str, Any]
    if "acquisition" in manifest:
        acq = manifest["acquisition"]
    elif "acquisitionDimensions" in manifest:
        acq = manifest["acquisitionDimensions"]
    else:
        acq = {}

    iw_list: List[Dict[str, Any]] = acq.get("iw", [])
    result: Dict[str, str] = {}
    for entry in iw_list:
        swath = str(entry.get("swath", "")).strip()
        annotation = str(entry.get("annotation", "")).strip()
        result[swath] = annotation
    return result


# ---------------------------------------------------------------------------
# Internal helpers — annotation XML parsing
# ---------------------------------------------------------------------------


def _parse_iw_bursts(
    xml_rel: str,
    archive_type: str,
    archive_path: Path,
    archive_root: Path,
) -> List[BurstRadarGrid]:
    """Parse all bursts from one IW annotation XML file.

    Parameters
    ----------
    xml_rel : relative path within the SAFE (e.g.
              ``"annotation/iw1/s1a-iw1-slc-vv-..."``).
    archive_type, archive_path, archive_root : passed through from
              ``parse_sentinel1_safe`` to open the right container.

    Returns
    -------
    list[BurstRadarGrid]
        Bursts in the order they appear in the XML.
    """
    raw = _read_xml_bytes(xml_rel, archive_type, archive_path)
    root = ET.fromstring(raw)

    # --- global metadata shared by all bursts --------------------------------
    mode = _find_text(root, "mode", "IW") or "IW"
    swath = _find_text(root, "swath", "") or ""
    swath = swath.strip().upper()
    if not swath.startswith("IW") and mode.upper().startswith("IW"):
        swath = mode.upper().strip()

    polarisation = _find_text(root, "polarisation", "VV") or "VV"
    orbit_direction = _derive_orbit_direction(root)

    # Global acquisition parameters
    az_interval = _find_float(root, "azimuthTimeInterval", 0.0)
    range_pixel_spacing = _find_float(root, "rangePixelSpacing", 0.0)
    starting_range = _slant_range_first_pixel(root)
    radar_wavelength = _find_float(root, "radarWavelength", SENTINEL1_WAVELENGTH)

    # Validate shared params
    if az_interval <= 0.0:
        raise ValueError(
            f"azimuthTimeInterval must be positive, got {az_interval} "
            f"in {xml_rel}"
        )
    if range_pixel_spacing <= 0.0:
        raise ValueError(
            f"rangePixelSpacing must be positive, got {range_pixel_spacing} "
            f"in {xml_rel}"
        )
    if starting_range <= 0.0:
        raise ValueError(
            f"startingRange (slantRangeTime) must be positive, got {starting_range} "
            f"in {xml_rel}"
        )
    if radar_wavelength <= 0.0:
        raise ValueError(
            f"radarWavelength must be positive, got {radar_wavelength} "
            f"in {xml_rel}"
        )

    # linesPerBurst is the standard Sentinel-1 annotation field
    lines_per_burst = _find_int(root, "linesPerBurst", 0)
    if lines_per_burst <= 0:
        lines_per_burst = _find_int(root, "numberOfLines", 0)

    samples_per_burst = _find_int(root, "samplesPerBurst", 0)
    if samples_per_burst <= 0:
        samples_per_burst = _find_int(root, "numberOfSamples", 0)

    azimuth_steering_rate = _find_float(root, "azimuthSteeringRate", 0.0)

    # --- Doppler centroid and FM-rate (shared per-swath) ----------------------
    doppler_coeffs = _swath_doppler_coefficients(root)
    fm_rate_coeffs = _swath_fm_rate_coefficients(root)

    bursts: List[BurstRadarGrid] = []

    for raw_burst in _iter_bursts(root):
        burst = _parse_single_burst(
            raw_burst,
            index_hint=None,          # detect from position or xml attribute
            swath=swath,
            polarisation=polarisation,
            orbit_direction=orbit_direction,
            azimuth_steering_rate=azimuth_steering_rate,
            lines_per_burst=lines_per_burst,
            samples_per_burst=samples_per_burst,
            az_interval=az_interval,
            range_pixel_spacing=range_pixel_spacing,
            starting_range=starting_range,
            radar_wavelength=radar_wavelength,
            doppler_coeffs=doppler_coeffs,
            fm_rate_coeffs=fm_rate_coeffs,
        )
        bursts.append(burst)

    return bursts


def _parse_single_burst(
    raw_burst: ET.Element,
    index_hint: int | None,
    swath: str,
    polarisation: str,
    orbit_direction: str,
    azimuth_steering_rate: float,
    lines_per_burst: int,
    samples_per_burst: int,
    az_interval: float,
    range_pixel_spacing: float,
    starting_range: float,
    radar_wavelength: float,
    doppler_coeffs: Tuple[float, ...],
    fm_rate_coeffs: Tuple[float, ...],
) -> BurstRadarGrid:
    """Parse one <burst> XML element into a ``BurstRadarGrid``."""
    # --- burst index ----------------------------------------------------------
    # annotation XML may use "burstIndex", "index", or number within the list
    raw_index = _child_text(raw_burst, "burstIndex")
    if raw_index is None:
        raw_index = _child_text(raw_burst, "index")
    if raw_index is not None:
        burst_index = int(float(raw_index.strip()))
    elif index_hint is not None:
        burst_index = index_hint
    else:
        # index hint not provided; we can't reliably infer — count siblings
        burst_index = _count_preceding_siblings(raw_burst)

    # Sentinel-1 annotation uses 1-based indices; we normalise to 0-based
    if burst_index > 0:
        burst_index -= 1

    # --- sensing times --------------------------------------------------------
    sensing_start_str = _child_text(raw_burst, "azimuthTime")
    if sensing_start_str is None:
        sensing_start_str = _child_text(raw_burst, "sensingTime")
    if sensing_start_str is None:
        raise ValueError("burst element missing both azimuthTime and sensingTime")

    sensing_start = parse_sensing_time(sensing_start_str)

    # Compute sensing_stop from lines per burst and azimuth interval
    num_lines = lines_per_burst
    if num_lines <= 0:
        num_lines = _find_int(raw_burst, "numberOfLines", 0)
    sensing_stop = sensing_start + timedelta(seconds=(num_lines - 1) * az_interval)

    # --- line / sample windows ------------------------------------------------
    first_valid_list = _parse_int_list(_child_text(raw_burst, "firstValidSample") or "")
    last_valid_list = _parse_int_list(_child_text(raw_burst, "lastValidSample") or "")

    # Derive valid region from firstValidSample / lastValidSample lists
    valid_region = _derive_valid_region(first_valid_list, last_valid_list)

    num_valid_lines = valid_region["numValidLines"]
    num_valid_samples = valid_region["numValidSamples"]

    if num_valid_lines <= 0:
        raise ValueError(
            f"numValidLines must be > 0 for burst index {burst_index} in swath {swath}. "
            f"Derived from firstValidSample={first_valid_list!r}."
        )
    # numValidSamples is derived; if zero it means the valid lines don't overlap
    # in range — this is rare but can legitimately happen at burst edges.
    # Warn rather than raise so valid burst data is not discarded.
    if num_valid_samples <= 0:
        raise ValueError(
            f"numValidSamples must be > 0 for burst index {burst_index} in swath {swath}. "
            f"The valid lines have non-overlapping sample ranges. "
            f"firstValidSample={first_valid_list!r}, lastValidSample={last_valid_list!r}"
        )

    # lineOffset in annotation is the byte/line offset of this burst in the TIFF
    line_offset = _find_int(raw_burst, "lineOffset", 0)

    # image_window covers the full burst region in the measurement image
    image_window = BurstWindow(
        first_line=line_offset,
        num_lines=num_lines,
        first_sample=0,
        num_samples=samples_per_burst,
    )

    # valid_window is relative to image_window
    valid_window = BurstWindow(
        first_line=valid_region["firstValidLine"],
        num_lines=num_valid_lines,
        first_sample=valid_region["firstValidSample"],
        num_samples=num_valid_samples,
    )

    # --- validate doppler / fm_rate -------------------------------------------
    if not (_any_nonzero(doppler_coeffs) or _any_nonzero(fm_rate_coeffs)):
        raise ValueError(
            f"Both doppler_coefficients ({doppler_coeffs}) and "
            f"azimuth_fm_rate_coefficients ({fm_rate_coeffs}) are all-zero "
            f"for burst index {burst_index} in swath {swath}. "
            "At least one must be non-zero."
        )

    # --- assemble identity and radar grid -------------------------------------
    identity = BurstIdentity(
        swath=swath,
        burst_index=burst_index,
        sensing_start=sensing_start,
        sensing_stop=sensing_stop,
        polarization=polarisation.upper(),
        orbit_direction=orbit_direction,
        azimuth_steering_rate=azimuth_steering_rate,
    )

    return BurstRadarGrid(
        identity=identity,
        image_window=image_window,
        valid_window=valid_window,
        line_offset=line_offset,
        azimuth_time_interval=az_interval,
        range_pixel_spacing=range_pixel_spacing,
        starting_range=starting_range,
        radar_wavelength=radar_wavelength,
        doppler_coefficients=doppler_coeffs,
        azimuth_fm_rate_coefficients=fm_rate_coeffs,
    )


# ---------------------------------------------------------------------------
# XML helper utilities (mirrors sentinel_importer.py patterns)
# ---------------------------------------------------------------------------


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _iter_bursts(root: ET.Element):
    """Yield each <burst> child element in document order."""
    for elem in root.iter():
        if _local_name(elem.tag) == "burst":
            yield elem


def _find_text(root: ET.Element, name: str, default: str | None = None) -> str | None:
    for elem in root.iter():
        if _local_name(elem.tag) == name:
            if elem.text is not None:
                return elem.text.strip()
    return default


def _find_float(root: ET.Element, name: str, default: float) -> float:
    val = _find_text(root, name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _find_int(root: ET.Element, name: str, default: int) -> int:
    val = _find_text(root, name)
    if val is None:
        return default
    try:
        return int(float(val.strip()))
    except (ValueError, TypeError):
        return default


def _child_text(parent: ET.Element, name: str) -> str | None:
    for child in parent:
        if _local_name(child.tag) == name and child.text is not None:
            return child.text.strip()
    return None


def _child_attr(parent: ET.Element, name: str, attr: str) -> str | None:
    """Read an XML attribute from a direct child element."""
    for child in parent:
        if _local_name(child.tag) == name:
            return child.attrib.get(attr)
    return None


def _parse_int_list(text: str | None) -> List[int]:
    """Parse a space-separated list of integers."""
    if not text:
        return []
    result: List[int] = []
    for part in text.split():
        try:
            result.append(int(float(part)))
        except (ValueError, TypeError):
            continue
    return result


def _derive_valid_region(
    first_valid: List[int],
    last_valid: List[int],
) -> Dict[str, int]:
    """Derive the contiguous valid region from firstValidSample / lastValidSample lists.

    A line is valid only if both ``first_valid[i] >= 0`` AND ``last_valid[i] >= 0``.
    Sentinel-1 uses -1 to indicate no valid samples for a given line.
    """
    valid_lines = [
        idx for idx, (f, l) in enumerate(zip(first_valid, last_valid))
        if f >= 0 and l >= 0
    ]
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
    # Use the first valid line's range — this matches ISCE2/Sentinel-1 convention
    # where all lines within a burst share the same sample extents.
    first_sample = first_valid[first_line]
    last_sample = last_valid[last_line]
    num_valid_samples = max(0, last_sample - first_sample + 1)
    return {
        "firstValidLine": first_line,
        "numValidLines": last_line - first_line + 1,
        "firstValidSample": first_sample,
        "lastValidSample": last_sample,
        "numValidSamples": num_valid_samples,
    }


def _slant_range_first_pixel(root: ET.Element) -> float:
    """Derive starting_range (m) from slantRangeTime (s) or startingRange element."""
    # slantRangeTime (seconds) → slant range = c * t / 2
    slant_time = _find_float_or_none(root, "slantRangeTime")
    if slant_time is not None and slant_time > 0:
        return 299_792_458.0 * slant_time / 2.0
    # Fallback: startingRange element
    sr = _find_float(root, "startingRange", 0.0)
    return sr


def _find_float_or_none(root: ET.Element, name: str) -> float | None:
    """Find a float element; return None if absent or unparseable."""
    val = _find_text(root, name)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _derive_orbit_direction(root: ET.Element) -> str:
    """Infer orbit direction from the <pass> element."""
    pass_str = _find_text(root, "pass", "")
    if pass_str:
        low = pass_str.lower()
        if "desc" in low:
            return "descending"
        if "asc" in low:
            return "ascending"
    return "ascending"  # safe default; no hard error


def _swath_doppler_coefficients(root: ET.Element) -> Tuple[float, ...]:
    """Extract Doppler centroid polynomial coefficients for the swath.

    The annotation XML stores coefficients either:
    - As a space-separated string in ``<dopplerCentroid polynomialCoefficients="..."/>``
    - Or as a ``<dopplerCentroid>0.0 -0.001 0.0</dopplerCentroid>`` text element
    - Or as ``<dcEstimate><dataDcPolynomial>...</dataDcPolynomial></dcEstimate>``
    """
    # Try attribute on <dopplerCentroid> first (self-closing element)
    attr_val = _child_attr(root, "dopplerCentroid", "polynomialCoefficients")
    if attr_val:
        coeffs = _parse_polynomial_text(attr_val)
        if coeffs:
            return tuple(coeffs)
    # Try text content of <dopplerCentroid>
    text_val = _child_text(root, "dopplerCentroid")
    if text_val:
        coeffs = _parse_polynomial_text(text_val)
        if coeffs:
            return tuple(coeffs)

    # Secondary location: dcEstimate / dataDcPolynomial
    coeffs: List[float] = []
    for dc in root.iter():
        if _local_name(dc.tag) == "dcEstimate":
            inner = _child_text(dc, "dataDcPolynomial")
            if inner:
                coeffs = _parse_polynomial_text(inner)
                break  # take the first entry
    return tuple(coeffs) if coeffs else (0.0,)


def _swath_fm_rate_coefficients(root: ET.Element) -> Tuple[float, ...]:
    """Extract azimuth FM rate polynomial coefficients for the swath.

    Stored either as an attribute ``<azimuthFmRate polynomialCoefficients="..."/>``
    or as an ``<azimuthFmRatePolynomial polynomialCoefficients="..."/>`` element,
    or as child text ``<azimuthFmRatePolynomial>0.0 1000.0</azimuthFmRatePolynomial>``.
    """
    coeffs: List[float] = []

    # Try attribute then text on <azimuthFmRate>
    for tag in ("azimuthFmRate", "azimuthFmRatePolynomial"):
        attr_val = _child_attr(root, tag, "polynomialCoefficients")
        if attr_val:
            coeffs = _parse_polynomial_text(attr_val)
            if coeffs:
                return tuple(coeffs)
        text_val = _child_text(root, tag)
        if text_val:
            coeffs = _parse_polynomial_text(text_val)
            if coeffs:
                return tuple(coeffs)

    return tuple(coeffs) if coeffs else (0.0,)


def _parse_polynomial_text(text: str | None) -> List[float]:
    """Parse a space-separated string of floats into a list."""
    if not text:
        return []
    result: List[float] = []
    for token in text.strip().split():
        try:
            result.append(float(token))
        except ValueError:
            continue
    return result


def _any_nonzero(coeffs: Tuple[float, ...]) -> bool:
    return any(abs(c) > 1e-12 for c in coeffs)


def _count_preceding_siblings(elem: ET.Element) -> int:
    """Count sibling elements with the same local tag name that precede elem."""
    parent = elem.getparent() if hasattr(elem, "getparent") else None  # type: ignore[attr-defined]
    if parent is None:
        return 0
    tag = elem.tag
    count = 0
    for sibling in parent:
        if sibling.tag == tag:
            if sibling is elem:
                break
            count += 1
    return count


# ---------------------------------------------------------------------------
# XML file reading helper
# ---------------------------------------------------------------------------


def _read_xml_bytes(
    xml_rel: str,
    archive_type: str,
    archive_path: Path,
) -> bytes:
    """Read the XML file at ``xml_rel`` from the given archive / directory."""
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(xml_rel) as fh:
                return fh.read()
    elif archive_type == "tar":
        with tarfile.open(archive_path) as tf:
            # strip leading "./" if present
            member_name = xml_rel.lstrip("/").lstrip("./")
            # try exact match first, then case-insensitive
            try:
                member = tf.getmember(member_name)
            except KeyError:
                # try lowercase
                member_name_lower = member_name.lower()
                member = None
                for m in tf.getmembers():
                    if m.name.lower() == member_name_lower:
                        member = m
                        break
                if member is None:
                    raise FileNotFoundError(
                        f"XML member {xml_rel!r} not found in {archive_path}"
                    )
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(
                    f"Cannot extract XML member {xml_rel!r} from {archive_path}"
                )
            with extracted as fh:
                return fh.read()
    else:
        # directory
        xml_path = archive_path / xml_rel
        return xml_path.read_bytes()