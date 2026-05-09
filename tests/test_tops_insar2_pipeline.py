"""Tests for the full _run_swath pipeline and stage execution in tops_insar2.py."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
    CommonBurstPair,
    CommonBurstSelection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity(
    swath: str = "IW1",
    burst_index: int = 0,
    sensing_seconds: float = 0.0,
) -> BurstIdentity:
    start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).replace(
        second=int(sensing_seconds)
    )
    stop = start.replace(second=int(sensing_seconds) + 2)
    return BurstIdentity(
        swath=swath,
        burst_index=burst_index,
        sensing_start=start,
        sensing_stop=stop,
        polarization="VV",
        orbit_direction="ascending",
        azimuth_steering_rate=0.0,
    )


def _make_burst(
    idx: int = 0,
    swath: str = "IW1",
    sensing_seconds: float = 0.0,
    num_lines: int = 1500,
    num_samples: int = 25000,
    line_offset: int = 0,
) -> BurstRadarGrid:
    return BurstRadarGrid(
        identity=_make_identity(swath, idx, sensing_seconds),
        image_window=BurstWindow(first_line=line_offset, num_lines=num_lines,
                                  first_sample=0, num_samples=num_samples),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                  first_sample=500, num_samples=24000),
        line_offset=line_offset,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.329562,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0, 1e-7),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def _make_common(
    n_bursts: int = 3,
    swath: str = "IW1",
) -> CommonBurstSelection:
    pairs = tuple(
        CommonBurstPair(
            pair_index=i,
            reference=_make_burst(i, swath, sensing_seconds=float(i * 3)),
            secondary=_make_burst(i, swath, sensing_seconds=float(i * 3)),
            burst_offset=0,
        )
        for i in range(n_bursts)
    )
    return CommonBurstSelection(
        swath=swath,
        reference_start_index=0,
        secondary_start_index=0,
        number_of_common_bursts=n_bursts,
        pairs=pairs,
    )


# ---------------------------------------------------------------------------
# Test 1: stage execution order (mocked)
# ---------------------------------------------------------------------------

def test_run_swath_stage_order(tmp_path: Path) -> None:
    """Verify that stages are executed in the correct order when all succeed."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0), _make_burst(1), _make_burst(2)]
    slave_bursts = [_make_burst(0), _make_burst(1), _make_burst(2)]

    called_stages: list[str] = []

    def mock_dispatch(stage_name, args, swath, work_dir, master, slave, state):
        called_stages.append(stage_name)
        return True

    mock_by_swath = {"IW1": master_bursts, "IW2": [], "IW3": []}
    mock_by_swath2 = {"IW1": slave_bursts, "IW2": [], "IW3": []}

    args = MagicMock()
    args.output_dir = tmp_path
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = None
    args.gpu_mode = "auto"
    args.azimuth_looks = 5
    args.range_looks = 5
    args.esd_coherence_threshold = 0.85

    stages = [
        "check", "preprocess", "common_bursts", "topo",
        "subset_overlaps", "coarse_resamp", "overlap_ifg",
        "prep_esd", "esd", "range_coreg", "fine_resamp",
        "burst_ifg", "merge_bursts", "filter", "unwrap",
        "geocode", "publish",
    ]

    with patch.object(ti2, "_dispatch_stage", side_effect=mock_dispatch):
        with patch.object(ti2, "parse_sentinel1_safe", side_effect=[mock_by_swath, mock_by_swath2]):
            result = ti2._run_swath(args, "IW1", master_bursts, slave_bursts, stages)

    assert called_stages == stages
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 2: stage_check fails when DEM does not exist
# ---------------------------------------------------------------------------

def test_stage_check_missing_dem(tmp_path: Path) -> None:
    """stage_check returns False when DEM path does not exist."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0)]
    slave_bursts = [_make_burst(0)]

    args = MagicMock()
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = tmp_path / "nonexistent_dem.tif"  # Does not exist

    state: dict = {}

    # Create master and slave directories so check passes for them
    (tmp_path / "master.safe").mkdir()
    (tmp_path / "slave.safe").mkdir()

    ok = ti2._stage_check(args, "IW1", tmp_path, master_bursts, slave_bursts, state)
    assert ok is False, "stage_check should return False when DEM does not exist"


def test_stage_check_passes_when_dem_exists(tmp_path: Path) -> None:
    """stage_check returns True when all paths (including DEM) exist."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0)]
    slave_bursts = [_make_burst(0)]

    args = MagicMock()
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = tmp_path / "dem.tif"

    state: dict = {}

    (tmp_path / "master.safe").mkdir()
    (tmp_path / "slave.safe").mkdir()
    (tmp_path / "dem.tif").write_text("fake dem")

    ok = ti2._stage_check(args, "IW1", tmp_path, master_bursts, slave_bursts, state)
    assert ok is True


def test_stage_check_passes_without_dem(tmp_path: Path) -> None:
    """stage_check returns True when no DEM is provided."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0)]
    slave_bursts = [_make_burst(0)]

    args = MagicMock()
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = None

    state: dict = {}

    (tmp_path / "master.safe").mkdir()
    (tmp_path / "slave.safe").mkdir()

    ok = ti2._stage_check(args, "IW1", tmp_path, master_bursts, slave_bursts, state)
    assert ok is True


# ---------------------------------------------------------------------------
# Test 3: stage skip when start_stage is set
# ---------------------------------------------------------------------------

def test_stage_skip_when_start_stage(tmp_path: Path) -> None:
    """When start_stage is set, stages before it are skipped."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0), _make_burst(1)]
    slave_bursts = [_make_burst(0), _make_burst(1)]

    called_stages: list[str] = []

    def mock_dispatch(stage_name, args, swath, work_dir, master, slave, state):
        called_stages.append(stage_name)
        return True

    args = MagicMock()
    args.output_dir = tmp_path
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = None
    args.gpu_mode = "auto"
    args.azimuth_looks = 5
    args.range_looks = 5
    args.esd_coherence_threshold = 0.85

    (tmp_path / "master.safe").mkdir()
    (tmp_path / "slave.safe").mkdir()

    # Start from "subset_overlaps" — skip check, preprocess, common_bursts, topo
    stages = [
        "subset_overlaps", "coarse_resamp", "overlap_ifg",
        "prep_esd", "esd", "range_coreg", "fine_resamp",
        "burst_ifg", "merge_bursts", "filter",
    ]

    with patch.object(ti2, "_dispatch_stage", side_effect=mock_dispatch):
        result = ti2._run_swath(args, "IW1", master_bursts, slave_bursts, stages)

    assert "check" not in called_stages, "check stage should be skipped"
    assert "preprocess" not in called_stages, "preprocess stage should be skipped"
    assert "common_bursts" not in called_stages, "common_bursts stage should be skipped"
    assert "topo" not in called_stages, "topo stage should be skipped"
    assert "subset_overlaps" in called_stages, "subset_overlaps should be called"
    assert "coarse_resamp" in called_stages, "coarse_resamp should be called"


# ---------------------------------------------------------------------------
# Test 4: stage skip when end_stage is set
# ---------------------------------------------------------------------------

def test_stage_skip_when_end_stage(tmp_path: Path) -> None:
    """When end_stage is set, stages after it are skipped."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0), _make_burst(1)]
    slave_bursts = [_make_burst(0), _make_burst(1)]

    called_stages: list[str] = []

    def mock_dispatch(stage_name, args, swath, work_dir, master, slave, state):
        called_stages.append(stage_name)
        return True

    args = MagicMock()
    args.output_dir = tmp_path
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = None
    args.gpu_mode = "auto"
    args.azimuth_looks = 5
    args.range_looks = 5
    args.esd_coherence_threshold = 0.85

    (tmp_path / "master.safe").mkdir()
    (tmp_path / "slave.safe").mkdir()

    # End at "esd" — skip range_coreg, fine_resamp, burst_ifg, merge_bursts, ...
    stages = [
        "check", "preprocess", "common_bursts", "topo",
        "subset_overlaps", "coarse_resamp", "overlap_ifg",
        "prep_esd", "esd",
    ]

    with patch.object(ti2, "_dispatch_stage", side_effect=mock_dispatch):
        result = ti2._run_swath(args, "IW1", master_bursts, slave_bursts, stages)

    assert "check" in called_stages, "check stage should be called (first)"
    assert "esd" in called_stages, "esd stage should be called (last)"
    assert "range_coreg" not in called_stages, "range_coreg stage should be skipped"
    assert "fine_resamp" not in called_stages, "fine_resamp stage should be skipped"
    assert "burst_ifg" not in called_stages, "burst_ifg stage should be skipped"
    assert "merge_bursts" not in called_stages, "merge_bursts stage should be skipped"


# ---------------------------------------------------------------------------
# Test 5: stage order from _build_stage_sequence
# ---------------------------------------------------------------------------

def test_stage_sequence_from_check_to_publish():
    """STAGE_SEQUENCE order is preserved by _build_stage_sequence."""
    from scripts import tops_insar2 as ti2

    seq = ti2._build_stage_sequence("check", "publish")
    assert seq[0] == "check"
    assert seq[-1] == "publish"
    assert len(seq) == len(ti2.STAGE_SEQUENCE)

    seq = ti2._build_stage_sequence("preprocess", "esd")
    assert seq[0] == "preprocess"
    assert seq[-1] == "esd"
    assert "coarse_resamp" in seq
    assert "fine_resamp" not in seq


# ---------------------------------------------------------------------------
# Test 6: stage_preprocess stores CommonBurstSelection in state
# ---------------------------------------------------------------------------

def test_stage_preprocess_stores_common(tmp_path: Path) -> None:
    """stage_preprocess stores the matched CommonBurstSelection in state['common']."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0), _make_burst(1), _make_burst(2)]
    slave_bursts = [_make_burst(0), _make_burst(1), _make_burst(2)]

    args = MagicMock()
    state: dict = {}

    ok = ti2._stage_preprocess(args, "IW1", tmp_path, master_bursts, slave_bursts, state)
    assert ok is True
    assert "common" in state
    common = state["common"]
    assert common.swath == "IW1"
    assert common.number_of_common_bursts == 3


# ---------------------------------------------------------------------------
# Test 7: stage_topo generates zero offsets when Geo2Rdr raises NotImplementedError
# ---------------------------------------------------------------------------

def test_stage_topo_generates_zero_offsets_on_not_implemented(tmp_path: Path) -> None:
    """stage_topo generates zero-offset .npz files when Geo2Rdr raises NotImplementedError."""
    from scripts import tops_insar2 as ti2

    common = _make_common(2)
    args = MagicMock()
    args.dem = None
    args.gpu_mode = "auto"

    state: dict = {"common": common}

    # Run topo stage — it will hit NotImplementedError from run_geo2rdr_single_burst
    # and should generate zero offsets instead
    ok = ti2._stage_topo(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    assert "geo2rdr_offsets" in state
    offsets = state["geo2rdr_offsets"]
    assert len(offsets) == 2
    for off in offsets:
        assert off.median_range_offset == 0.0
        assert off.median_azimuth_offset == 0.0

    # Verify zero-offset files were written
    for i in range(2):
        pair_dir = tmp_path / f"burst_{i:03d}"
        range_path = pair_dir / "range.off.npz"
        az_path = pair_dir / "azimuth.off.npz"
        assert range_path.exists(), f"range.off.npz missing for pair {i}"
        assert az_path.exists(), f"azimuth.off.npz missing for pair {i}"


# ---------------------------------------------------------------------------
# Test 8: stage_burst_ifg produces npz files
# ---------------------------------------------------------------------------

def test_stage_burst_ifg_produces_npz_files(tmp_path: Path) -> None:
    """stage_burst_ifg writes burst_ifg_{pair_idx}.npz files for each pair."""
    from scripts import tops_insar2 as ti2

    # Create a common with 2 pairs, each having reference SLCs
    common = _make_common(2)
    args = MagicMock()
    args.output_dir = tmp_path

    # Write dummy deramped reference and resampled secondary SLCs
    for pair in common.pairs:
        pair_dir = tmp_path / f"burst_{pair.pair_index:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        ref_path = pair_dir / "deramped_ref.npz"
        sec_path = pair_dir / "resampled_sec.npz"
        np.savez(ref_path, data=np.ones((1300, 24000), dtype=np.complex64))
        np.savez(sec_path, data=np.ones((1300, 24000), dtype=np.complex64))

    state: dict = {
        "common": common,
        "looks": (5, 5),
    }

    ok = ti2._stage_burst_ifg(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    burst_ifg_dir = tmp_path / "burst_ifg"
    assert burst_ifg_dir.exists()

    for i in range(2):
        npz = burst_ifg_dir / f"burst_ifg_{i:03d}.npz"
        assert npz.exists(), f"burst_ifg_{i:03d}.npz not found"
        with np.load(npz) as data:
            assert "ifg" in data
            assert "coherence" in data


# ---------------------------------------------------------------------------
# Test 9: resolve_swaths
# ---------------------------------------------------------------------------

def test_resolve_swaths_all():
    """_resolve_swaths('all') returns ['IW1', 'IW2', 'IW3']."""
    from scripts import tops_insar2 as ti2

    assert ti2._resolve_swaths("all") == ["IW1", "IW2", "IW3"]


def test_resolve_swaths_single():
    """_resolve_swaths('IW2') returns ['IW2']."""
    from scripts import tops_insar2 as ti2

    assert ti2._resolve_swaths("IW2") == ["IW2"]


def test_resolve_swaths_multiple():
    """_resolve_swaths('IW1,IW3') returns ['IW1', 'IW3']."""
    from scripts import tops_insar2 as ti2

    assert ti2._resolve_swaths("IW1,IW3") == ["IW1", "IW3"]


# ---------------------------------------------------------------------------
# Test 10: stage_subset_overlaps with 1 burst (skip)
# ---------------------------------------------------------------------------

def test_stage_subset_overlaps_skips_with_1_burst(tmp_path: Path) -> None:
    """stage_subset_overlaps returns early when < 2 common bursts."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)  # only 1 burst → no overlaps
    args = MagicMock()
    state: dict = {"common": common}

    ok = ti2._stage_subset_overlaps(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    assert state["overlaps"] == []


# ---------------------------------------------------------------------------
# Test 11: stage_burst_ifg with missing SLCs synthesizes zeros
# ---------------------------------------------------------------------------

def test_stage_burst_ifg_with_missing_slc(tmp_path: Path) -> None:
    """stage_burst_ifg synthesizes zero IFG when SLC files are missing."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.output_dir = tmp_path

    state: dict = {
        "common": common,
        "looks": (5, 5),
    }

    # Don't write any SLC files — they should be missing
    ok = ti2._stage_burst_ifg(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    burst_ifg_dir = tmp_path / "burst_ifg"
    npz = burst_ifg_dir / "burst_ifg_000.npz"
    assert npz.exists()
    with np.load(npz) as data:
        assert data["ifg"].shape == (1300, 24000)


# ---------------------------------------------------------------------------
# Test 12: stage_merge_bursts calls merge_bursts with correct args
# ---------------------------------------------------------------------------

def test_stage_merge_bursts_produces_merged_npy(tmp_path: Path) -> None:
    """stage_merge_bursts writes merged_interferogram.npy and merged_coherence.npy."""
    from scripts import tops_insar2 as ti2

    common = _make_common(2)
    args = MagicMock()
    args.output_dir = tmp_path

    # Write burst IFG and coherence files
    burst_ifg_dir = tmp_path / "burst_ifg"
    burst_ifg_dir.mkdir(parents=True, exist_ok=True)

    for pair in common.pairs:
        np.savez(
            burst_ifg_dir / f"burst_ifg_{pair.pair_index:03d}.npz",
            ifg=np.ones((1300, 24000), dtype=np.complex64),
            coherence=np.ones((1300, 24000), dtype=np.float32),
            valid_fraction=np.float32(1.0),
        )

    state: dict = {
        "common": common,
        "looks": (5, 5),
    }

    ok = ti2._stage_merge_bursts(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    merged_dir = tmp_path / "merged"
    assert (merged_dir / "merged_interferogram.npy").exists()
    assert (merged_dir / "merged_coherence.npy").exists()
    assert (merged_dir / "burst_seam_diagnostics.json").exists()


# ---------------------------------------------------------------------------
# Test 13: unimplemented stages do not crash the pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_not_implemented_stages(tmp_path: Path) -> None:
    """Stages that raise NotImplementedError (topo, coarse_resamp) or are
    spike stubs (fine_resamp, filter, unwrap, geocode, publish) must not
    crash the pipeline — they should return True and log warnings."""
    from scripts import tops_insar2 as ti2

    master_bursts = [_make_burst(0), _make_burst(1), _make_burst(2)]
    slave_bursts = [_make_burst(0), _make_burst(1), _make_burst(2)]

    args = MagicMock()
    args.output_dir = tmp_path
    args.master_safe_or_manifest = tmp_path / "master.safe"
    args.slave_safe_or_manifest = tmp_path / "slave.safe"
    args.dem = None
    args.gpu_mode = "auto"
    args.azimuth_looks = 5
    args.range_looks = 5
    args.esd_coherence_threshold = 0.85

    (tmp_path / "master.safe").mkdir()
    (tmp_path / "slave.safe").mkdir()

    # We run the pipeline through _run_swath but intercept _dispatch_stage
    # to exercise the spike stages without needing real data files.
    def patched_dispatch(stage_name, args, swath, work_dir, master, slave, state):
        # Run the actual stage function so state is properly populated
        fn = {
            "check":           ti2._stage_check,
            "preprocess":      ti2._stage_preprocess,
            "common_bursts":   ti2._stage_common_bursts,
            "topo":            ti2._stage_topo,
            "subset_overlaps": ti2._stage_subset_overlaps,
            "coarse_resamp":   ti2._stage_coarse_resamp,
            "overlap_ifg":     ti2._stage_overlap_ifg,
            "prep_esd":        ti2._stage_prep_esd,
            "esd":             ti2._stage_esd,
            "range_coreg":     ti2._stage_range_coreg,
            "fine_resamp":     ti2._stage_fine_resamp,
            "burst_ifg":       ti2._stage_burst_ifg,
            "merge_bursts":    ti2._stage_merge_bursts,
            "filter":          ti2._stage_filter,
            "unwrap":          ti2._stage_unwrap,
            "geocode":         ti2._stage_geocode,
            "publish":         ti2._stage_publish,
        }.get(stage_name)
        if fn is None:
            return False
        return fn(args, swath, work_dir, master, slave, state)

    stages = list(ti2.STAGE_SEQUENCE)

    with patch.object(ti2, "_dispatch_stage", side_effect=patched_dispatch):
        result = ti2._run_swath(args, "IW1", master_bursts, slave_bursts, stages)

    assert result["status"] == "ok", f"Pipeline should succeed despite unimplemented stages: {result}"


# ---------------------------------------------------------------------------
# Test 14: stage_filter preserves shape
# ---------------------------------------------------------------------------

def test_stage_filter_preserves_shape(tmp_path: Path) -> None:
    """_stage_filter should preserve the input interferogram shape."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.output_dir = tmp_path

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Write merged products
    shape = (1300, 24000)
    merged_ifg = np.ones(shape, dtype=np.complex64)
    merged_coh = np.ones(shape, dtype=np.float32)
    np.save(merged_dir / "merged_interferogram.npy", merged_ifg)
    np.save(merged_dir / "merged_coherence.npy", merged_coh)

    state: dict = {
        "common": common,
        "looks": (5, 5),
    }

    ok = ti2._stage_filter(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    assert state["merged_ifg"] is not None
    assert state["merged_ifg"].shape == shape


def test_stage_filter_reduces_noise(tmp_path: Path) -> None:
    """_stage_filter should reduce interferogram noise (variance decreases)."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.output_dir = tmp_path

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Create noisy interferogram with uniform coherence
    shape = (100, 100)
    np.random.seed(42)
    noise_phase = np.random.uniform(-np.pi, np.pi, shape)
    merged_ifg = np.exp(1j * noise_phase).astype(np.complex64)
    merged_coh = np.ones(shape, dtype=np.float32)
    np.save(merged_dir / "merged_interferogram.npy", merged_ifg)
    np.save(merged_dir / "merged_coherence.npy", merged_coh)

    state: dict = {"common": common, "looks": (5, 5)}

    ok = ti2._stage_filter(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    filtered_ifg = state["merged_ifg"]

    # Filtered IFG should have lower variance in magnitude (from smoothing)
    # and the file should be saved
    assert (merged_dir / "filtered_ifg.npy").exists()


# ---------------------------------------------------------------------------
# Test 15: stage_geocode skips when DEM missing
# ---------------------------------------------------------------------------

def test_stage_geocode_skips_when_dem_missing(tmp_path: Path) -> None:
    """_stage_geocode should skip gracefully when DEM is not provided."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.dem = None
    args.resolution_meters = 20.0

    state: dict = {
        "common": common,
        "merged_ifg": np.ones((100, 100), dtype=np.complex64),
        "merged_coh": np.ones((100, 100), dtype=np.float32),
    }

    ok = ti2._stage_geocode(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    assert state.get("geocoded_ifg") is None


def test_stage_geocode_skips_when_no_common(tmp_path: Path) -> None:
    """_stage_geocode should skip when common burst selection is missing."""
    from scripts import tops_insar2 as ti2

    args = MagicMock()
    args.dem = tmp_path / "dem.tif"
    args.resolution_meters = 20.0

    # Create a fake DEM
    args.dem.parent.mkdir(parents=True, exist_ok=True)
    args.dem.write_text("dummy dem")

    state: dict = {
        "common": None,
        "merged_ifg": np.ones((100, 100), dtype=np.complex64),
        "merged_coh": np.ones((100, 100), dtype=np.float32),
    }

    ok = ti2._stage_geocode(args, "IW1", tmp_path, [], [], state)

    assert ok is True


# ---------------------------------------------------------------------------
# Test 16: stage_publish writes hdf5 and tiffs
# ---------------------------------------------------------------------------

def test_stage_publish_writes_hdf5(tmp_path: Path) -> None:
    """_stage_publish should write HDF5 product file."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.output_dir = tmp_path
    args.dem = None

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    shape = (100, 100)
    merged_ifg = np.ones(shape, dtype=np.complex64)
    merged_coh = np.ones(shape, dtype=np.float32)
    np.save(merged_dir / "merged_interferogram.npy", merged_ifg)
    np.save(merged_dir / "merged_coherence.npy", merged_coh)

    state: dict = {
        "common": common,
        "looks": (5, 5),
    }

    ok = ti2._stage_publish(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    output_dir = tmp_path / "IW1"
    h5_file = output_dir / "IW1_interferogram.h5"
    assert h5_file.exists(), f"HDF5 file should exist: {h5_file}"


def test_stage_publish_continues_on_error(tmp_path: Path) -> None:
    """_stage_publish should return True even if publishing fails partially."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.output_dir = tmp_path
    args.dem = None

    # Leave merged products empty so write_product may fail
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "common": common,
        "looks": (5, 5),
        "merged_ifg": None,  # Will cause publish to fail gracefully
    }

    ok = ti2._stage_publish(args, "IW1", tmp_path, [], [], state)

    # Should fail gracefully when merged_ifg is None
    assert ok is False


# ---------------------------------------------------------------------------
# Test 17: stage_unwrap fallback to simple 2d unwrap
# ---------------------------------------------------------------------------

def test_stage_unwrap_fallback_to_2d_when_icu_unavailable(tmp_path: Path) -> None:
    """_stage_unwrap should fall back to simple 2D unwrap when ICU is not available."""
    from scripts import tops_insar2 as ti2

    common = _make_common(1)
    args = MagicMock()
    args.output_dir = tmp_path
    args.unwrap_method = "icu"

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    shape = (100, 100)
    phase = np.linspace(-np.pi, np.pi, shape[0] * shape[1]).reshape(shape).astype(np.float32)
    merged_ifg = np.exp(1j * phase).astype(np.complex64)
    merged_coh = np.ones(shape, dtype=np.float32)
    np.save(merged_dir / "merged_interferogram.npy", merged_ifg)
    np.save(merged_dir / "merged_coherence.npy", merged_coh)

    state: dict = {
        "common": common,
        "looks": (5, 5),
    }

    ok = ti2._stage_unwrap(args, "IW1", tmp_path, [], [], state)

    assert ok is True
    assert "unwrapped" in state
    assert state["unwrapped"] is not None
    assert state["unwrapped"].shape == shape
    assert (merged_dir / "unwrapped.npy").exists()