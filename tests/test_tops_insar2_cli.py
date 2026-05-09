"""Tests for tops_insar2.py CLI scaffold (Task 0)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CLI_SCRIPT = SCRIPTS_DIR / "tops_insar2.py"

# Forbidden module names that tops_*.py modules must never import.
FORBIDDEN: frozenset[str] = frozenset({
    "strip_insar",
    "strip_insar2",
    "tops_insar",
    # Also block the scripts-prefixed variants.
    "scripts.strip_insar",
    "scripts.strip_insar2",
})


class TestCliHelp:
    """test_cli_help_succeeds — verify --help returns code 0 and shows banner."""

    def test_cli_help_succeeds(self) -> None:
        """`python3 scripts/tops_insar2.py --help` exits 0 and mentions Sentinel-1."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"--help failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Sentinel-1 TOPS" in result.stdout, (
            f"--help output does not mention 'Sentinel-1 TOPS':\n{result.stdout}"
        )

    def test_cli_help_lists_expected_arguments(self) -> None:
        """Help text must contain the required argument names."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        for arg in (
            "output_dir",
            "master_safe_or_manifest",
            "slave_safe_or_manifest",
            "--swath",
            "--start-stage",
            "--end-stage",
            "--dem",
            "--resolution-meters",
            "--range-looks",
            "--azimuth-looks",
            "--unwrap-method",
            "--extra-esd-cycles",
            "--esd-coherence-threshold",
            "--do-ionospheric-correction",
            "--gpu-mode",
            "--log-level",
        ):
            assert arg in result.stdout, (
                f"Argument '{arg}' missing from --help output"
            )

    def test_cli_help_stage_choices_are_listed(self) -> None:
        """All 17 pipeline stages should appear in --help choices."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        expected_stages = [
            "check", "preprocess", "common_bursts", "topo",
            "subset_overlaps", "coarse_resamp", "overlap_ifg",
            "prep_esd", "esd", "range_coreg", "fine_resamp",
            "burst_ifg", "merge_bursts", "filter", "unwrap",
            "geocode", "publish",
        ]
        for stage in expected_stages:
            assert stage in result.stdout, (
                f"Stage '{stage}' missing from --help output"
            )


class TestNoStripImports:
    """test_no_strip_imports_in_tops_modules — AST scan for forbidden imports."""

    # Legacy files that exist in the repo and may import strip backends.
    # NEW tops_*.py modules written as part of the tops_insar2 project must not.
    _LEGACY_EXCLUDED = frozenset({"tops_insar.py", "tops_rtc.py"})

    @pytest.fixture
    def tops_modules(self) -> list[Path]:
        """All scripts/tops_*.py files, excluding tops_insar2.py and known legacy files."""
        candidates = sorted(SCRIPTS_DIR.glob("tops_*.py"))
        return [
            p for p in candidates
            if p.name != "tops_insar2.py" and p.name not in self._LEGACY_EXCLUDED
        ]

    def test_no_strip_imports_in_tops_modules(self, tops_modules: list[Path]) -> None:
        """Every scripts/tops_*.py (except tops_insar2.py) must not import
        strip_insar / strip_insar2 / tops_insar under any alias."""
        failures: list[str] = []

        for path in tops_modules:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                failures.append(f"{path}: syntax error — {exc}")
                continue

            for node in ast.walk(tree):
                # import X  or  import X as Y
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in FORBIDDEN:
                            failures.append(
                                f"{path}: 'import {alias.name}' is forbidden"
                            )

                # from X import Y  or  from X.Y import Z
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    # Reject: from strip_insar import ...  or  from scripts.strip_insar import ...
                    if module in FORBIDDEN or module.startswith("strip"):
                        failures.append(
                            f"{path}: 'from {module} import ...' is forbidden"
                        )
                    # Reject: from somewhere import strip_insar as X
                    for alias in node.names:
                        if alias.name in FORBIDDEN:
                            failures.append(
                                f"{path}: imports '{alias.name}' which is forbidden"
                            )

        assert not failures, "\n".join(failures)

    def test_tops_insar2_does_not_import_tops_model(self) -> None:
        """tops_insar2.py must not import tops_model or any tops_*.py module —
        it is the CLI entry point only."""
        tree = ast.parse(CLI_SCRIPT.read_text(encoding="utf-8"))
        tops_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("tops_"):
                    tops_imports.append(module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tops_"):
                        tops_imports.append(alias.name)
        assert not tops_imports, (
            f"tops_insar2.py must not import tops modules; found: {tops_imports}"
        )


class TestCliStageSequence:
    """Sanity-check the _build_stage_sequence helper."""

    def test_stage_sequence_inclusive(self) -> None:
        """When tops_insar2 is imported, _build_stage_sequence must work."""
        # We import via subprocess so sys.modules guards are in a fresh interpreter.
        code = """
import sys
sys.path.insert(0, '%s')
from tops_insar2 import _build_stage_sequence

all_stages = _build_stage_sequence("check", "publish")
assert all_stages[0] == "check"
assert all_stages[-1] == "publish"
assert len(all_stages) == 17

subset = _build_stage_sequence("esd", "unwrap")
assert subset == ["esd", "range_coreg", "fine_resamp", "burst_ifg", "merge_bursts", "filter", "unwrap"]

print("OK")
""" % SCRIPTS_DIR
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"stage sequence test failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout
