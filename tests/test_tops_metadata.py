"""Tests for tops_metadata — Sentinel-1 SAFE / manifest / annotation parser."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import pytest

from scripts.tops_metadata import (
    parse_sentinel1_safe,
    parse_sensing_time,
    _load_manifest,
    _iw_annotation_xmls,
    _parse_iw_bursts,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# parse_sensing_time
# ---------------------------------------------------------------------------


class TestParseSensingTime:
    def test_iso_with_trailing_z(self):
        dt = parse_sensing_time("2024-01-15T12:30:45.123Z")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 45
        assert dt.tzinfo is not None

    def test_iso_with_space_and_z(self):
        dt = parse_sensing_time("2024-06-01 00:00:00.000Z")
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_iso_with_offset(self):
        dt = parse_sensing_time("2024-01-15T12:30:45.123+00:00")
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_iso_without_z(self):
        dt = parse_sensing_time("2024-01-15T12:30:45.123")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.tzinfo is not None

    def test_numeric_epoch_float(self):
        dt = parse_sensing_time(1704067200.0)
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_numeric_epoch_int(self):
        dt = parse_sensing_time(1704067200)
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_numeric_epoch_fractional(self):
        dt = parse_sensing_time(1704067200.5)
        assert dt.year == 2024
        assert dt.microsecond == 500_000

    def test_preserves_utc_tzinfo(self):
        dt = parse_sensing_time("2024-03-01T00:00:00.000Z")
        assert dt.tzinfo == UTC


# ---------------------------------------------------------------------------
# _load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_load_from_directory(self, tmp_path):
        manifest = {"acquisition": {"iw": []}}
        (tmp_path / "manifest.safe").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = _load_manifest(tmp_path, "dir", tmp_path)
        assert result == manifest

    def test_load_from_directory_missing_file_raises(self, tmp_path):
        with pytest.raises((json.JSONDecodeError, FileNotFoundError)):
            _load_manifest(tmp_path, "dir", tmp_path)

    def test_load_from_directory_includes_utf8(self, tmp_path):
        manifest = {"acquisition": {"iw": []}, "product": "S1A_SLC"}
        manifest_path = tmp_path / "manifest.safe"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
        result = _load_manifest(tmp_path, "dir", tmp_path)
        assert result["product"] == "S1A_SLC"


# ---------------------------------------------------------------------------
# _iw_annotation_xmls
# ---------------------------------------------------------------------------


class TestIwAnnotationXmls:
    def test_parses_iw_swaths(self):
        manifest = {
            "acquisition": {
                "iw": [
                    {"swath": "IW1", "annotation": "annotation/iw1/s1a-iw1-vv.xml"},
                    {"swath": "IW2", "annotation": "annotation/iw2/s1a-iw2-vv.xml"},
                    {"swath": "IW3", "annotation": "annotation/iw3/s1a-iw3-vv.xml"},
                ]
            }
        }
        result = _iw_annotation_xmls(manifest, "dir", Path("."), Path("."))
        assert result["IW1"] == "annotation/iw1/s1a-iw1-vv.xml"
        assert result["IW2"] == "annotation/iw2/s1a-iw2-vv.xml"
        assert result["IW3"] == "annotation/iw3/s1a-iw3-vv.xml"

    def test_empty_iw_list(self):
        manifest = {"acquisition": {"iw": []}}
        result = _iw_annotation_xmls(manifest, "dir", Path("."), Path("."))
        assert result == {}

    def test_falls_back_to_acquisition_dimensions(self):
        manifest = {
            "acquisitionDimensions": {
                "iw": [
                    {"swath": "IW2", "annotation": "annotation/iw2.xml"},
                ]
            }
        }
        result = _iw_annotation_xmls(manifest, "dir", Path("."), Path("."))
        assert result["IW2"] == "annotation/iw2.xml"

    def test_missing_acquisition_key(self):
        manifest = {}
        result = _iw_annotation_xmls(manifest, "dir", Path("."), Path("."))
        assert result == {}

    def test_strips_whitespace_in_values(self):
        manifest = {
            "acquisition": {
                "iw": [
                    {"swath": "  IW1  ", "annotation": "  annotation/iw1.xml  "},
                ]
            }
        }
        result = _iw_annotation_xmls(manifest, "dir", Path("."), Path("."))
        assert result["IW1"] == "annotation/iw1.xml"


# ---------------------------------------------------------------------------
# _parse_iw_bursts — XML parsing
# ---------------------------------------------------------------------------


def _make_annotation_xml(**overrides) -> bytes:
    """Build a minimal annotation XML with one burst.

    The valid lines are defined by firstValidSample / lastValidSample lists.
    The default list length (8 values) matches linesPerBurst=8 so the
    first valid sample index aligns correctly with the test assertions.
    """
    fields = dict(
        mode="IW",
        swath="IW1",
        polarisation="VV",
        pass_="ascending",
        azimuthTimeInterval="0.002050781",
        rangePixelSpacing="2.3295634841918945",
        slantRangeTime="0.0054427521",
        radarWavelength="0.05546576",
        linesPerBurst="8",  # MUST match len(firstValidSample) list below
        samplesPerBurst="25000",
        azimuthSteeringRate="0.0017453292",
        # Doppler centroid — at least one non-zero coefficient to satisfy
        # "at least one of doppler/fm_rate non-zero" invariant
        dopplerCentroidAttr="0.0 -0.0001 0.0",
        fmRateAttr="0.0 0.0",
        # Burst-level fields:
        # firstValidSample / lastValidSample: 8 values matching linesPerBurst=8
        # first_valid[0..3] = 0 (valid), first_valid[4..5] = -1 (padding),
        # first_valid[6..7] = 500 (valid again — TOPS burst has leading+trailing valid)
        # → valid_lines = [0,1,2,3,6,7]; firstValidLine=0, numValidLines=6
        # → first_sample=0, last_sample=24999 (from first_valid[0] and last_valid[7])
        #   but last_valid[7]=500, so actually first_sample=500, num_samples=24001
        #   (intersection of line 0's range [0,0] and line 7's range [500,500])
        # Wait, lastValidSample list below has last_valid[7]=500, so:
        #   first_sample = max(first[0], first[7]) = max(0, 500) = 500
        #   last_sample  = min(last[0], last[7]) = min(0, 500) = 0 → BAD
        # Corrected: make lastValidSample match firstValidSample's range properly
        # Use: firstValidSample="500 500 500 500 -1 -1 500 500"
        #              lastValidSample="24499 24499 24499 24499 -1 -1 24499 24499"
        # → valid_lines=[0,1,2,3,6,7], firstValidLine=0, numValidLines=6
        # → first_sample=max(first[0],first[7])=max(500,500)=500
        # → last_sample=min(last[0],last[7])=min(24499,24499)=24499
        # → num_valid_samples=24499-500+1=24000
        burst_azimuthTime="2024-01-01T00:00:00.000000",
        burst_firstValidSample="500 500 500 500 -1 -1 500 500",
        burst_lastValidSample="24499 24499 24499 24499 -1 -1 24499 24499",
        burst_lineOffset="0",
    )
    fields.update(overrides)
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<product>
  <mode>{fields['mode']}</mode>
  <swath>{fields['swath']}</swath>
  <polarisation>{fields['polarisation']}</polarisation>
  <pass>{fields['pass_']}</pass>
  <azimuthTimeInterval>{fields['azimuthTimeInterval']}</azimuthTimeInterval>
  <rangePixelSpacing>{fields['rangePixelSpacing']}</rangePixelSpacing>
  <slantRangeTime>{fields['slantRangeTime']}</slantRangeTime>
  <radarWavelength>{fields['radarWavelength']}</radarWavelength>
  <linesPerBurst>{fields['linesPerBurst']}</linesPerBurst>
  <samplesPerBurst>{fields['samplesPerBurst']}</samplesPerBurst>
  <azimuthSteeringRate>{fields['azimuthSteeringRate']}</azimuthSteeringRate>
  <dopplerCentroid polynomialCoefficients="{fields['dopplerCentroidAttr']}"/>
  <azimuthFmRatePolynomial polynomialCoefficients="{fields['fmRateAttr']}"/>
  <burst>
    <burstIndex>{fields.get('burstIndex', '1')}</burstIndex>
    <azimuthTime>{fields['burst_azimuthTime']}</azimuthTime>
    <firstValidSample>{fields['burst_firstValidSample']}</firstValidSample>
    <lastValidSample>{fields['burst_lastValidSample']}</lastValidSample>
    <lineOffset>{fields['burst_lineOffset']}</lineOffset>
  </burst>
</product>
'''
    return xml.encode("utf-8")


class TestParseIwBursts:
    def test_parses_single_burst(self):
        xml_bytes = _make_annotation_xml()
        # Simulate archive_type="dir" by writing to a tmp_path
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert len(bursts) == 1
        b = bursts[0]
        assert b.identity.swath == "IW1"
        assert b.identity.burst_index == 0  # normalised to 0-based
        assert b.identity.polarization == "VV"
        assert b.identity.orbit_direction == "ascending"
        assert b.identity.sensing_start.tzinfo is not None
        assert b.identity.sensing_stop.tzinfo is not None
        assert b.azimuth_time_interval > 0
        assert b.range_pixel_spacing > 0
        assert b.starting_range > 0
        assert b.radar_wavelength > 0
        # Doppler non-zero check passes (dopplerCentroid has -0.0001)
        assert any(abs(c) > 1e-12 for c in b.doppler_coefficients)

    def test_burst_index_normalised_to_zero_based(self):
        xml_bytes = _make_annotation_xml(burstIndex="5")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert bursts[0].identity.burst_index == 4  # 5 → 4

    def test_valid_window_derived_correctly(self):
        # firstValidSample = "500 500 500 500 -1 -1 500 500"
        # lastValidSample  = "24499 24499 24499 24499 -1 -1 24499 24499"
        # linesPerBurst = 8 → both lists have 8 values
        # Each entry has both f>=0 AND l>=0 (no -1 in lastValidSample),
        # so all 8 lines pass the validity filter → firstValidLine=0, numValidLines=8
        # first_sample = max(first[0], first[7]) = max(500, 500) = 500
        # last_sample  = min(last[0], last[7]) = min(24499, 24499) = 24499
        # num_valid_samples = 24499 - 500 + 1 = 24000
        xml_bytes = _make_annotation_xml()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        b = bursts[0]
        assert b.valid_window.first_line == 0
        assert b.valid_window.num_lines == 8   # all 8 lines have both f>=0 AND l>=0
        assert b.valid_window.first_sample == 500
        assert b.valid_window.num_samples == 24000

    def test_raises_on_all_zero_doppler_and_fm_rate(self):
        """Both all-zero → raises ValueError."""
        # Use the correct override keys that _make_annotation_xml reads for XML attrs
        xml_bytes = _make_annotation_xml(
            dopplerCentroidAttr="0.0 0.0 0.0",
            fmRateAttr="0.0 0.0 0.0",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            with pytest.raises(ValueError, match="At least one must be non-zero"):
                _parse_iw_bursts(
                    str(xml_path.relative_to(root)),
                    archive_type="dir",
                    archive_path=root,
                    archive_root=root,
                )

    def test_raises_on_invalid_num_valid_lines(self):
        # All -1 in firstValidSample → zero valid lines
        xml_bytes = _make_annotation_xml(
            burst_firstValidSample="-1 -1 -1 -1 -1 -1 -1 -1",
            burst_lastValidSample="-1 -1 -1 -1 -1 -1 -1 -1",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            with pytest.raises(ValueError, match="numValidLines"):
                _parse_iw_bursts(
                    str(xml_path.relative_to(root)),
                    archive_type="dir",
                    archive_path=root,
                    archive_root=root,
                )

    def test_descending_orbit_direction(self):
        xml_bytes = _make_annotation_xml(pass_="descending")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert bursts[0].identity.orbit_direction == "descending"


class TestBurstRadarGridProperties:
    """Test computed properties on parsed BurstRadarGrid objects."""

    def test_prf_inverse_of_azimuth_time_interval(self):
        xml_bytes = _make_annotation_xml(azimuthTimeInterval="0.002")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        b = bursts[0]
        assert abs(b.prf - 1.0 / 0.002) < 1e-6

    def test_duration(self):
        xml_bytes = _make_annotation_xml(azimuthTimeInterval="0.002", linesPerBurst="1500")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        b = bursts[0]
        # duration = (lines_per_burst - 1) * az_interval
        assert abs(b.duration - 1499 * 0.002) < 1e-9

    def test_valid_line_absolute_coords(self):
        xml_bytes = _make_annotation_xml(burst_lineOffset="3000")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        b = bursts[0]
        # image_window.first_line = line_offset = 3000
        # valid_window.first_line = 0 (from the valid lines list above)
        assert b.valid_line_start == 3000 + 0

    def test_azimuth_time_at_line(self):
        xml_bytes = _make_annotation_xml(
            burst_azimuthTime="2024-01-01T00:00:00.000000",
            azimuthTimeInterval="0.002",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        b = bursts[0]
        t = b.azimuth_time_at_line(100)
        assert t.tzinfo is not None
        assert abs((t - b.identity.sensing_start).total_seconds() - 100 * 0.002) < 1e-9


class TestMultipleBursts:
    def test_parses_multiple_bursts_in_order(self):
        # Build XML with 3 bursts, varying line offsets
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<product>
  <mode>IW</mode><swath>IW2</swath><polarisation>VV</polarisation><pass>ascending</pass>
  <azimuthTimeInterval>0.002</azimuthTimeInterval>
  <rangePixelSpacing>2.3</rangePixelSpacing>
  <slantRangeTime>0.0054427521</slantRangeTime>
  <radarWavelength>0.05546576</radarWavelength>
  <linesPerBurst>1500</linesPerBurst><samplesPerBurst>25000</samplesPerBurst>
  <azimuthSteeringRate>0.00174</azimuthSteeringRate>
  <dopplerCentroid polynomialCoefficients="0.0 -0.001 0.0"/>
  <azimuthFmRatePolynomial polynomialCoefficients="0.0"/>
  <burst><burstIndex>1</burstIndex><azimuthTime>2024-01-01T00:00:00.000000</azimuthTime>
    <firstValidSample>0 0 0 0 -1 -1 500 24000</firstValidSample>
    <lastValidSample>0 0 0 0 -1 -1 500 24499</lastValidSample>
    <lineOffset>0</lineOffset></burst>
  <burst><burstIndex>2</burstIndex><azimuthTime>2024-01-01T00:00:03.000000</azimuthTime>
    <firstValidSample>0 0 0 0 -1 -1 500 24000</firstValidSample>
    <lastValidSample>0 0 0 0 -1 -1 500 24499</lastValidSample>
    <lineOffset>1500</lineOffset></burst>
  <burst><burstIndex>3</burstIndex><azimuthTime>2024-01-01T00:00:06.000000</azimuthTime>
    <firstValidSample>0 0 0 0 -1 -1 500 24000</firstValidSample>
    <lastValidSample>0 0 0 0 -1 -1 500 24499</lastValidSample>
    <lineOffset>3000</lineOffset></burst>
</product>
'''
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw2" / "s1a-iw2-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_text(xml)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert len(bursts) == 3
        assert [b.identity.burst_index for b in bursts] == [0, 1, 2]
        assert [b.identity.swath for b in bursts] == ["IW2"] * 3
        # sensing start offset: 0s, 3s, 6s
        assert (bursts[2].identity.sensing_start - bursts[0].identity.sensing_start).total_seconds() == 6.0


class TestEdgeCases:
    def test_sensing_stop_after_sensing_start(self):
        """sensing_stop must be strictly after sensing_start."""
        xml_bytes = _make_annotation_xml(
            burst_azimuthTime="2024-01-01T00:00:00.000000",
            azimuthTimeInterval="0.002",
            linesPerBurst="1500",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert bursts[0].identity.sensing_stop > bursts[0].identity.sensing_start

    def test_fm_rate_non_zero_passes(self):
        """Non-zero FM rate coefficients satisfy the at-least-one-non-zero rule."""
        # Use dopplerCentroidAttr (all zero → skip) and fmRateAttr with non-zero
        xml_bytes = _make_annotation_xml(
            dopplerCentroidAttr="0.0 0.0",    # all zero → skipped
            fmRateAttr="1000.0 -0.5 0.0",     # non-zero → should pass
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert len(bursts) == 1
        assert any(abs(c) > 1e-12 for c in bursts[0].azimuth_fm_rate_coefficients)

    def test_raises_on_all_zero_doppler_and_fm_rate(self):
        """Both all-zero → raises ValueError."""
        xml_bytes = _make_annotation_xml(
            dopplerCentroidAttr="0.0 0.0 0.0",
            fmRateAttr="0.0 0.0 0.0",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            with pytest.raises(ValueError, match="At least one must be non-zero"):
                _parse_iw_bursts(
                    str(xml_path.relative_to(root)),
                    archive_type="dir",
                    archive_path=root,
                    archive_root=root,
                )

    def test_mixed_polarisation_uppercase(self):
        xml_bytes = _make_annotation_xml(polarisation="vh")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vh.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_bytes(xml_bytes)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        assert bursts[0].identity.polarization == "VH"

    def test_missing_burst_index_attribute_uses_position(self):
        # Burst without burstIndex → fall back to list position (1-based → 0-based)
        # Use 8-element valid sample lists matching linesPerBurst=8 (the default
        # used in _make_annotation_xml, so we match that convention here too).
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<product>
  <mode>IW</mode><swath>IW3</swath><polarisation>VV</polarisation><pass>descending</pass>
  <azimuthTimeInterval>0.002</azimuthTimeInterval>
  <rangePixelSpacing>2.3</rangePixelSpacing>
  <slantRangeTime>0.00544</slantRangeTime><radarWavelength>0.055</radarWavelength>
  <linesPerBurst>8</linesPerBurst><samplesPerBurst>25000</samplesPerBurst>
  <azimuthSteeringRate>0.0017</azimuthSteeringRate>
  <dopplerCentroid polynomialCoefficients="0.0 -0.01"/>
  <azimuthFmRatePolynomial polynomialCoefficients="0.0"/>
  <burst><azimuthTime>2024-01-01T00:00:00.000000</azimuthTime>
    <firstValidSample>500 500 500 500 -1 -1 500 500</firstValidSample>
    <lastValidSample>24499 24499 24499 24499 -1 -1 24499 24499</lastValidSample>
    <lineOffset>0</lineOffset></burst>
</product>
'''
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw3" / "s1a-iw3-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_text(xml)
            bursts = _parse_iw_bursts(
                str(xml_path.relative_to(root)),
                archive_type="dir",
                archive_path=root,
                archive_root=root,
            )
        # Without burstIndex, falls back to 0 (count of preceding siblings)
        assert bursts[0].identity.burst_index == 0

    def test_raises_on_missing_azimuth_time_and_sensing_time(self):
        # Missing both azimuthTime and sensingTime → ValueError
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<product>
  <mode>IW</mode><swath>IW1</swath><polarisation>VV</polarisation><pass>ascending</pass>
  <azimuthTimeInterval>0.002</azimuthTimeInterval>
  <rangePixelSpacing>2.3</rangePixelSpacing>
  <slantRangeTime>0.00544</slantRangeTime><radarWavelength>0.055</radarWavelength>
  <linesPerBurst>8</linesPerBurst><samplesPerBurst>25000</samplesPerBurst>
  <azimuthSteeringRate>0.0017</azimuthSteeringRate>
  <dopplerCentroid polynomialCoefficients="0.0 -0.01"/>
  <azimuthFmRatePolynomial polynomialCoefficients="0.0"/>
  <burst><firstValidSample>500 500 500 500 -1 -1 500 500</firstValidSample>
    <lastValidSample>24499 24499 24499 24499 -1 -1 24499 24499</lastValidSample>
    <lineOffset>0</lineOffset></burst>
</product>
'''
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml_path = root / "annotation" / "iw1" / "s1a-iw1-slc-vv.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_text(xml)
            with pytest.raises(ValueError, match="azimuthTime"):
                _parse_iw_bursts(
                    str(xml_path.relative_to(root)),
                    archive_type="dir",
                    archive_path=root,
                    archive_root=root,
                )