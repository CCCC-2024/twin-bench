# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Twins are importable as `twins.<name>` (`pip install -e .` now installs `twins` next to `dtcore`). The template
  shows the layout every twin moves to: public names in `__init__.py`, the Twin in `twin.py`, the Sources in
  `sources.py`; `twins/_template/mdt/` is gone and its check is `tests/test_template.py`.
- bikeshare moved to that layout (`from twins.bikeshare import StationTwin, ReplayStation, SimStation`; `sdt/` is
  gone). `scripts/check_dtcore.py` and `scripts/check_repro.py` became `tests/test_bikeshare.py`, which keeps their
  research claims (learned rates = the paper's benchmarks; nightly decisions = the paper's experiment; decisions = the
  authors' code except one numerical tie) and drops the loop-against-loop comparison. `scripts/crosscheck_upstream.py`
  (run once, 10 min) is removed; the README says where its saved result came from. The README lists every observation
  field, whether a real station has it, and what an action changes.
- `dtcore.Source.act` raises `NotImplementedError` unless a source overrides it: a decision sent to a source that does
  not handle actions used to be dropped without notice. Every source in the repository already overrides it.
- `dtcore` and `docs/architecture.md` state what `t` and `kind` promise: the meaning and unit of `t` belong to each
  twin and `run()` does not check them; `kind` says where the data come from, not whether actions can change state.

### Fixed
- `twins/battery/scripts/baseline.py` writes the five columns `plot_tradeoff.py` needs (`x_global`, `u_global`,
  `u_twin_true`, `regret_global`, `too_late`); before, the committed `baseline.csv` and `tradeoff_baseline.png` could
  not be regenerated from the repository. The rebuilt columns equal the committed ones in all 1,230 rows.
- All committed results regenerated in one environment, so rerunning the scripts now reproduces them byte for byte.
  Against the previous files only floating-point noise changed (at most 2e-11); every decision, count and figure value
  is the same.

## [0.2.0] - 2026-10-17

### Added
- `bikeshare` twin: one Citi Bike station. Reproduces the overnight inventory decision of Gammelli et al. 2022
  (365 of 366 decisions identical to the authors' code, the other a numerical tie) and runs on dtcore with a
  Gamma–Poisson update that recovers the paper's benchmark rates and a daytime-truck what-if; Twin Bench page module
  (a simulated station online; the recorded Nov–Dec 2018 days in the local copy only).

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
