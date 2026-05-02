# Sentinel Orbit Download And Apply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Sentinel-1 orbit resolution, download wrapper, importer integration, and post-import orbit apply tooling.

**Architecture:** Create `scripts/sentinel_orbit.py` as a standalone utility for filename parsing, local cache matching, `fetchOrbit.py` download wrapping, and applying EOF orbits to existing imported manifests. Keep `sentinel_importer.py` as the importer and call the orbit resolver only when requested by `orbit_dir` or `download_orbit`.

**Tech Stack:** Python stdlib, XML parsing already present in `sentinel_importer.py`, existing Sentinel EOF parser reused through `SentinelImporter.extract_eof_orbit`, unittest/pytest.

---

### Tasks

- [ ] Add tests for filename parsing, POEORB-over-RESORB selection, EOF.zip extraction, apply-to-manifest, and importer integration.
- [ ] Implement `scripts/sentinel_orbit.py` with `resolve_orbit_for_product()` and CLI subcommands `download` and `apply`.
- [ ] Extend `SentinelImporter` constructor and CLI with `orbit_dir` and `download_orbit`.
- [ ] Update progress and findings after verification.
