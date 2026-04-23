from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STAGE_SEQUENCE = ("check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6")

STAGE_DIR_NAMES = {
    "check": "check",
    "prep": "prep",
    "crop": "crop",
    "p0": "p0_geo2rdr",
    "p1": "p1_dense_match",
    "p2": "p2_crossmul",
    "p3": "p3_unwrap",
    "p4": "p4_geocode",
    "p5": "p5_hdf",
    "p6": "p6_publish",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_stage_name(stage: str) -> str:
    if stage not in STAGE_SEQUENCE:
        raise ValueError(f"Unsupported stage '{stage}'")
    return stage


def stage_index(stage: str) -> int:
    return STAGE_SEQUENCE.index(validate_stage_name(stage))


def work_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "work"


def stage_dir(output_dir: str | Path, stage: str) -> Path:
    return work_dir(output_dir) / STAGE_DIR_NAMES[validate_stage_name(stage)]


def stage_json_path(output_dir: str | Path, stage: str) -> Path:
    return stage_dir(output_dir, stage) / "stage.json"


def success_marker_path(output_dir: str | Path, stage: str) -> Path:
    return stage_dir(output_dir, stage) / "SUCCESS"


def load_stage_record(output_dir: str | Path, stage: str) -> dict | None:
    path = stage_json_path(output_dir, stage)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stage_succeeded(output_dir: str | Path, stage: str) -> bool:
    record = load_stage_record(output_dir, stage)
    return bool(record and record.get("success") and success_marker_path(output_dir, stage).exists())


def write_stage_record(output_dir: str | Path, stage: str, record: dict) -> Path:
    path = stage_json_path(output_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def mark_stage_success(output_dir: str | Path, stage: str) -> Path:
    path = success_marker_path(output_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("success\n", encoding="utf-8")
    return path


def clear_stage_success(output_dir: str | Path, stage: str) -> None:
    success_path = success_marker_path(output_dir, stage)
    if success_path.exists():
        success_path.unlink()


def first_incomplete_stage(output_dir: str | Path) -> str | None:
    for stage in STAGE_SEQUENCE:
        if not stage_succeeded(output_dir, stage):
            return stage
    return None


def resolve_requested_stages(
    output_dir: str | Path,
    *,
    step: str | None = None,
    start_step: str | None = None,
    end_step: str | None = None,
    resume: bool = False,
) -> list[str]:
    modes = [step is not None, start_step is not None or end_step is not None, resume]
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError("--step, --start-step/--end-step, and --resume are mutually exclusive")

    if step is not None:
        return [validate_stage_name(step)]

    if start_step is not None or end_step is not None:
        if start_step is None or end_step is None:
            raise ValueError("--start-step and --end-step must be provided together")
        start_idx = stage_index(start_step)
        end_idx = stage_index(end_step)
        if start_idx > end_idx:
            raise ValueError("--start-step must not be later than --end-step")
        return list(STAGE_SEQUENCE[start_idx : end_idx + 1])

    if resume:
        first_stage = first_incomplete_stage(output_dir)
        if first_stage is None:
            return []
        return list(STAGE_SEQUENCE[stage_index(first_stage) :])

    return list(STAGE_SEQUENCE)
