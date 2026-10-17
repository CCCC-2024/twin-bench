# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-17

First public version.

### Added
- `dtcore`: `Source` / `Twin` contract, `run()` loop, `Recorder` (memory + SQLite), unit tests.
- Twins: incubator (built from the paper's description, real data), battery (reproduction of Thelen et al. 2023,
  plus an MQTT / Streamlit live demo), UAV (reproduction of Kapteyn et al. 2021 without ROS), tumor (reproduction of
  TumorTwin, not yet on dtcore), and `_template` (a minimal mug-and-heater twin to copy).
- Twin Bench page with incubator, battery and UAV modules, JavaScript checked against Python; deployed to GitHub Pages.
- `run_checks.sh` and CI running every check; pinned `fetch_upstream.sh` per twin; documentation (architecture,
  adding a twin, file conventions).

### Fixed
- `Recorder` crashed on numpy arrays with more than one element.
- The incubator and UAV dtcore checks compared against reference files written in another environment; they now
  compute the reference in the same run, so they no longer fail on different numpy / scipy versions.
