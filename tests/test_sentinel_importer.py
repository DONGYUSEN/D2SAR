import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


ANNOTATION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <mode>IW</mode>
    <swath>IW2</swath>
    <polarisation>VV</polarisation>
    <startTime>2023-11-10T04:39:48.000000</startTime>
    <stopTime>2023-11-10T04:40:00.000000</stopTime>
  </adsHeader>
  <imageAnnotation>
    <imageInformation>
      <numberOfSamples>4</numberOfSamples>
      <numberOfLines>3</numberOfLines>
      <productFirstLineUtcTime>2023-11-10T04:39:48.000000</productFirstLineUtcTime>
      <azimuthTimeInterval>0.002055556</azimuthTimeInterval>
      <azimuthPixelSpacing>13.9</azimuthPixelSpacing>
      <slantRangeTime>0.0045</slantRangeTime>
      <rangePixelSpacing>2.329562</rangePixelSpacing>
      <radarFrequency>5405000454.33435</radarFrequency>
      <incidenceAngleMidSwath>34.5</incidenceAngleMidSwath>
      <ascendingNodeTime>2023-11-10T04:00:00.000000</ascendingNodeTime>
      <projection>Slant Range</projection>
    </imageInformation>
    <processingInformation>
      <swathProcParamsList>
        <swathProcParams>
          <rangeProcessing>
            <windowType>HAMMING</windowType>
            <windowCoefficient>0.75</windowCoefficient>
            <processingBandwidth>56500000.0</processingBandwidth>
          </rangeProcessing>
          <azimuthProcessing>
            <windowType>HAMMING</windowType>
            <windowCoefficient>0.85</windowCoefficient>
            <processingBandwidth>327.0</processingBandwidth>
          </azimuthProcessing>
        </swathProcParams>
      </swathProcParamsList>
    </processingInformation>
  </imageAnnotation>
  <generalAnnotation>
    <productInformation>
      <radarFrequency>5405000454.33435</radarFrequency>
      <rangeSamplingRate>64345238.12571429</rangeSamplingRate>
      <pass>ASCENDING</pass>
      <platformHeading>-13.2</platformHeading>
      <azimuthSteeringRate>0.0349</azimuthSteeringRate>
    </productInformation>
    <downlinkInformationList>
      <downlinkInformation><prf>1717.128973</prf></downlinkInformation>
    </downlinkInformationList>
    <terrainHeightList>
      <terrainHeight><value>42.0</value></terrainHeight>
    </terrainHeightList>
    <orbitList>
      <orbit>
        <time>2023-11-10T04:39:47.000000</time>
        <position><x>1.0</x><y>2.0</y><z>3.0</z></position>
        <velocity><x>4.0</x><y>5.0</y><z>6.0</z></velocity>
      </orbit>
      <orbit>
        <time>2023-11-10T04:39:49.000000</time>
        <position><x>7.0</x><y>8.0</y><z>9.0</z></position>
        <velocity><x>10.0</x><y>11.0</y><z>12.0</z></velocity>
      </orbit>
    </orbitList>
    <dopplerCentroid>
      <dcEstimateList>
        <dcEstimate>
          <azimuthTime>2023-11-10T04:39:48.000000</azimuthTime>
          <t0>0.0045</t0>
          <dataDcPolynomial>1.0 2.0 3.0</dataDcPolynomial>
        </dcEstimate>
        <dcEstimate>
          <azimuthTime>2023-11-10T04:39:50.000000</azimuthTime>
          <t0>0.0046</t0>
          <dataDcPolynomial>4.0 5.0 6.0</dataDcPolynomial>
        </dcEstimate>
      </dcEstimateList>
    </dopplerCentroid>
    <azimuthFmRateList>
      <azimuthFmRate>
        <azimuthTime>2023-11-10T04:39:48.000000</azimuthTime>
        <t0>0.0045</t0>
        <azimuthFmRatePolynomial>7.0 8.0 9.0</azimuthFmRatePolynomial>
      </azimuthFmRate>
      <azimuthFmRate>
        <azimuthTime>2023-11-10T04:39:50.000000</azimuthTime>
        <t0>0.0046</t0>
        <azimuthFmRatePolynomial>10.0 11.0 12.0</azimuthFmRatePolynomial>
      </azimuthFmRate>
    </azimuthFmRateList>
  </generalAnnotation>
  <geolocationGrid>
    <geolocationGridPointList>
      <geolocationGridPoint><latitude>29.0</latitude><longitude>94.0</longitude></geolocationGridPoint>
      <geolocationGridPoint><latitude>30.0</latitude><longitude>95.0</longitude></geolocationGridPoint>
    </geolocationGridPointList>
  </geolocationGrid>
  <swathTiming>
    <linesPerBurst>4</linesPerBurst>
    <samplesPerBurst>4</samplesPerBurst>
    <burstList count="2">
      <burst>
        <azimuthTime>2023-11-10T04:39:48.000000</azimuthTime>
        <sensingTime>2023-11-10T04:39:47.900000</sensingTime>
        <firstValidSample>0 0 1 1</firstValidSample>
        <lastValidSample>3 3 2 2</lastValidSample>
      </burst>
      <burst>
        <azimuthTime>2023-11-10T04:39:50.000000</azimuthTime>
        <sensingTime>2023-11-10T04:39:49.900000</sensingTime>
        <firstValidSample>1 1 1 1</firstValidSample>
        <lastValidSample>2 2 2 2</lastValidSample>
      </burst>
    </burstList>
  </swathTiming>
</product>
"""


def make_sentinel_safe_dir(root: Path, manifest_xml: str) -> Path:
    safe = root / "S1A_IW_SLC__1SDV_TEST.SAFE"
    annotation = safe / "annotation"
    calibration = annotation / "calibration"
    measurement = safe / "measurement"
    annotation.mkdir(parents=True)
    calibration.mkdir()
    measurement.mkdir()
    (safe / "manifest.safe").write_text(manifest_xml, encoding="utf-8")
    (annotation / "s1a-iw2-slc-vv-test.xml").write_text(ANNOTATION_XML, encoding="utf-8")
    rfi = annotation / "rfi"
    rfi.mkdir()
    (rfi / "rfi-s1a-iw2-slc-vv-test.xml").write_text("<rfi/>\n", encoding="utf-8")
    (annotation / "s1a-iw1-slc-vh-test.xml").write_text(
        ANNOTATION_XML.replace("<swath>IW2</swath>", "<swath>IW1</swath>").replace(
            "<polarisation>VV</polarisation>", "<polarisation>VH</polarisation>"
        ),
        encoding="utf-8",
    )
    (calibration / "calibration-s1a-iw2-slc-vv-test.xml").write_text("<calibration/>\n", encoding="utf-8")
    (calibration / "noise-s1a-iw2-slc-vv-test.xml").write_text("<noise/>\n", encoding="utf-8")
    (measurement / "s1a-iw2-slc-vv-test.tiff").write_bytes(b"")
    (measurement / "s1a-iw1-slc-vh-test.tiff").write_bytes(b"")
    return safe


class SentinelImporterTests(unittest.TestCase):
    MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:safe="http://www.esa.int/safe/sentinel-1.0">
  <metadataSection>
    <metadataObject ID="processing">
      <metadataWrap>
        <xmlData>
          <safe:processing>
            <safe:facility site="ESRIN" country="IT">
              <safe:software name="Sentinel-1 IPF" version="003.40"/>
            </safe:facility>
          </safe:processing>
        </xmlData>
      </metadataWrap>
    </metadataObject>
  </metadataSection>
</xfdu:XFDU>
"""

    def _make_safe_dir(self, root: Path) -> Path:
        return make_sentinel_safe_dir(root, self.MANIFEST_XML)

    def test_run_imports_safe_directory_to_manifest_and_metadata(self) -> None:
        from sentinel_importer import SentinelImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = self._make_safe_dir(root)
            output_dir = root / "out"

            manifest_path = SentinelImporter(str(safe)).run(str(output_dir), download_dem=False)

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            metadata_dir = output_dir / "metadata"
            acquisition = json.loads((metadata_dir / "acquisition.json").read_text(encoding="utf-8"))
            tops = json.loads((metadata_dir / "tops.json").read_text(encoding="utf-8"))
            metadata_files_exist = {
                name: Path(manifest["metadata"][name]).is_file()
                for name in ("acquisition", "orbit", "radargrid", "doppler", "scene")
            }
            tops_metadata_exists = (metadata_dir / "tops.json").is_file()

        self.assertEqual(manifest["sensor"], "sentinel-1")
        self.assertEqual(manifest["productType"], "SLC")
        self.assertEqual(manifest["polarisation"], "VV")
        self.assertEqual(manifest["tops"]["mode"], "IW")
        self.assertEqual(manifest["tops"]["swath"], "IW2")
        self.assertEqual(manifest["tops"]["burst_count"], 2)
        self.assertEqual(manifest["slc"]["format"], "GeoTIFF")
        self.assertEqual(manifest["slc"]["storage_layout"], "single_band_complex")
        self.assertEqual(manifest["slc"]["rows"], 3)
        self.assertEqual(manifest["slc"]["columns"], 4)
        self.assertTrue(str(manifest["slc"]["path"]).endswith("measurement/s1a-iw2-slc-vv-test.tiff"))
        self.assertTrue(metadata_files_exist["acquisition"])
        self.assertTrue(metadata_files_exist["orbit"])
        self.assertTrue(metadata_files_exist["radargrid"])
        self.assertTrue(metadata_files_exist["doppler"])
        self.assertTrue(metadata_files_exist["scene"])
        self.assertTrue(tops_metadata_exists)
        self.assertIn("annotationXML", manifest["ancillary"])
        self.assertIn("calibrationXML", manifest["ancillary"])
        self.assertIn("noiseXML", manifest["ancillary"])
        self.assertIn("manifestSafe", manifest["ancillary"])
        self.assertEqual(manifest["processing"]["softwareVersion"], "003.40")
        self.assertEqual(manifest["processing"]["facility"], "ESRIN, IT")
        self.assertAlmostEqual(acquisition["rangeSamplingRate"], 64345238.12571429)
        self.assertAlmostEqual(acquisition["azimuthSteeringRate"], 0.0349)
        self.assertAlmostEqual(acquisition["prf"], 1717.128973)
        self.assertEqual(tops["linesPerBurst"], 4)
        self.assertEqual(tops["samplesPerBurst"], 4)
        self.assertEqual(tops["rangeWindowType"], "HAMMING")
        self.assertEqual(tops["azimuthWindowType"], "HAMMING")
        self.assertEqual(tops["bursts"][0]["numberOfLines"], 4)
        self.assertEqual(tops["bursts"][0]["numberOfSamples"], 4)
        self.assertEqual(tops["bursts"][0]["lineOffset"], 0)
        self.assertEqual(tops["bursts"][1]["lineOffset"], 4)
        self.assertEqual(tops["bursts"][0]["firstValidLine"], 0)
        self.assertEqual(tops["bursts"][0]["numValidLines"], 4)
        self.assertEqual(tops["bursts"][0]["firstValidSample"], 1)
        self.assertEqual(tops["bursts"][0]["numValidSamples"], 1)
        self.assertEqual(tops["bursts"][0]["doppler"]["coefficients"], [1.0, 2.0, 3.0])
        self.assertEqual(tops["bursts"][1]["doppler"]["coefficients"], [4.0, 5.0, 6.0])
        self.assertEqual(tops["bursts"][0]["azimuthFMRate"]["coefficients"], [7.0, 8.0, 9.0])
        self.assertEqual(tops["bursts"][1]["azimuthFMRate"]["coefficients"], [10.0, 11.0, 12.0])

    def test_run_selects_requested_swath_and_polarization(self) -> None:
        from sentinel_importer import SentinelImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = self._make_safe_dir(root)

            manifest_path = SentinelImporter(str(safe), swath="IW1", polarization="VH").run(
                str(root / "out"), download_dem=False
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["tops"]["swath"], "IW1")
        self.assertEqual(manifest["polarisation"], "VH")
        self.assertTrue(str(manifest["slc"]["path"]).endswith("measurement/s1a-iw1-slc-vh-test.tiff"))

    def test_zip_paths_use_vsizip_references(self) -> None:
        from sentinel_importer import SentinelImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = self._make_safe_dir(root)
            zip_path = root / "sentinel.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for path in safe.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(root).as_posix())

            manifest_path = SentinelImporter(str(zip_path)).run(str(root / "out"), download_dem=False)
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertTrue(str(manifest["slc"]["path"]["path"]).startswith("/vsizip/"))
        self.assertEqual(manifest["slc"]["path"]["storage"], "zip")

    def test_discovery_ignores_rfi_annotation_xml(self) -> None:
        from sentinel_importer import SentinelImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = self._make_safe_dir(root)

            files = SentinelImporter(str(safe)).discover_files()

        self.assertTrue(files["annotation"].endswith("annotation/s1a-iw2-slc-vv-test.xml"))
        self.assertNotIn("/rfi/", files["annotation"].replace("\\", "/"))

    def test_run_uses_external_eof_orbit_and_derives_overlap_metadata(self) -> None:
        from sentinel_importer import SentinelImporter

        eof_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Earth_Explorer_File>
  <Data_Block>
    <List_of_OSVs>
      <OSV><UTC>UTC=2023-11-10T04:39:47.500000</UTC><X>100</X><Y>200</Y><Z>300</Z><VX>1</VX><VY>2</VY><VZ>3</VZ><Quality>NOMINAL</Quality></OSV>
      <OSV><UTC>UTC=2023-11-10T04:39:48.500000</UTC><X>400</X><Y>500</Y><Z>600</Z><VX>4</VX><VY>5</VY><VZ>6</VZ><Quality>NOMINAL</Quality></OSV>
    </List_of_OSVs>
  </Data_Block>
</Earth_Explorer_File>
"""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = self._make_safe_dir(root)
            orbit_file = root / "S1A_OPER_AUX_POEORB_TEST.EOF"
            orbit_file.write_text(eof_xml, encoding="utf-8")

            manifest_path = SentinelImporter(str(safe), orbit_file=orbit_file).run(
                str(root / "out"), download_dem=False
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            metadata_dir = Path(manifest_path).parent / "metadata"
            orbit = json.loads((metadata_dir / "orbit.json").read_text(encoding="utf-8"))
            tops = json.loads((metadata_dir / "tops.json").read_text(encoding="utf-8"))

        self.assertEqual(orbit["source"], "sentinel-1-eof")
        self.assertEqual(orbit["orbitFile"], str(orbit_file.resolve()))
        self.assertEqual(orbit["stateVectors"][0]["position"]["x"], 100.0)
        self.assertEqual(tops["overlaps"][0]["previousBurstIndex"], 1)
        self.assertEqual(tops["overlaps"][0]["nextBurstIndex"], 2)
        self.assertIn("estimatedOverlapLines", tops["overlaps"][0])
        self.assertEqual(tops["esd"]["overlapCount"], 1)
        self.assertTrue(tops["esd"]["readyForOverlapEstimation"])
        self.assertEqual(manifest["ancillary"]["orbitFile"], str(orbit_file.resolve()))


if __name__ == "__main__":
    unittest.main()
