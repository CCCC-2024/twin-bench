# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- bikeshare: `RNNRates`, the paper's trained VP-RNN as a rate model of the twin (`StationTwin(C, RNNRates())`).
  It runs the authors' weights (fetched) in a copy of their inference network, needs the new `rnn` extra (PyTorch;
  CI installs the CPU build), and forecasts the next day from what the twin has seen. Tests: its next-hour rates
  have the accuracy the authors recorded (18 numbers); its 24-hour forecast at midnight is less accurate.
- bikeshare: `scripts/compare_rates.py` runs the twin with each rate model on the test days
  (`results/compare_rates.json`). In the twin, knowing only what is known at 23:00, the VP-RNN forecasts better than
  the last-30-days average but does not decide better; the paper's protocol owes its advantage to forecasts that use
  the same day's earlier counts. The README's results and notes say so, with tests.
- Page: a twin's part with more than one implementation can be chosen in the twin panel (`renderParts`). Bike share:
  on the recorded days (local copy), the demand forecast switches between the historical average and the VP-RNN,
  whose rates `export_web.py` precomputes into the local replay file. `check_bikeshare.js` now replays the recorded
  days with both and requires the decisions and shortages of `results/compare_rates.json`.

### Fixed
- `tools/fetch_github.sh` skipped a repository already fetched at the same commit even when a twin's
  `fetch_upstream.sh` had added files; it now fetches when a listed file is missing.

### Changed
- `run_checks.sh` builds the local-only page data of every twin whose `export_web.py` takes `--replay-only` (was:
  the incubator's only); bike share's `export_web.py` takes it.
- bikeshare: `StationTwin` takes its rate model as an argument, `StationTwin(C, rates)`; the Gamma–Poisson rates
  moved unchanged to `rates.py` as `HistoricalRates`, the default. `StationTwin(C, window_days=30)` is now
  `StationTwin(C, HistoricalRates(window_days=30))`. Results are unchanged.

## [0.3.0] - 2026-10-19

The four twins on dtcore and the template share one layout, and every result their READMEs state has a source.
Imports change: `from twins.<name> import …` replaces the per-twin packages (`idt`, `bdt`, `udt`, `sdt`, `mdt`).

### Changed
- Twins are importable as `twins.<name>` (`pip install -e .` now installs `twins` next to `dtcore`). The template
  shows the layout every twin moves to: public names in `__init__.py`, the Twin in `twin.py`, the Sources in
  `sources.py`; `twins/_template/mdt/` is gone and its check is `tests/test_template.py`.
- bikeshare moved to that layout (`from twins.bikeshare import StationTwin, ReplayStation, SimStation, experiment`;
  `sdt/` is gone). `scripts/check_dtcore.py` and `scripts/check_repro.py` became `tests/test_bikeshare.py` with the same
  claims, a little stricter (learned rates = the paper's benchmarks; nightly decisions and shortages = the paper's
  experiment; decisions = the authors' code except exactly one numerical tie). The README lists every observation
  field, whether a real station has it, and what an action changes.
- incubator moved to that layout (`from twins.incubator import IncubatorTwin, SimBox, RecordedBox, Controller`; `idt/`
  is gone; the two Sources are in `sources.py`). Its scripts are `step1_model.py` (was `step1a_model.py`, results
  renamed to match) and `export_web.py` (was `export_reference.py`), like the other twins, and its outputs take the
  conventions' names (`ref_web_incubator.json`; locally `ref_web_incubator_replay.json`, `web/incubator_replay.js`).
  `scripts/check_dtcore.py`, which compared `dtcore.run` with a hand-written loop, became `tests/test_incubator.py`,
  which tests the README's results instead (the model against the real data, the controller against the recorded
  heater, the lid alarm, the alarm on the real runs). `scripts/crosscheck_upstream.py`, a one-time comparison that
  needs a full clone of the authors' repository, is removed; its result stays in the README.
- uav moved to that layout (`from twins.uav import UAVTwin, ScriptedUAV, ControlledUAV, UAVModel`; `udt/` is gone,
  `assets.py` is `sources.py`, and `twins/uav/__init__.py` is the one place the authors' package is put on the import
  path). `scripts/check_dtcore.py` ran the frozen study in `experiments/` as a subprocess to compare with it; it became
  `tests/test_uav.py`, which tests the README's results through the twin itself (the authors' planner policy; the
  study's error and maneuver rates on the authors' asset, which the twin reproduces exactly). The study stays as it
  is; `upstream_sources.json` (checksums of a snapshot, now pinned by commit) is removed.
- battery moved to that layout (`from twins.battery import BatteryTwin, CellReplay`; `bdt/` is gone and `dt.py` is
  split into `twin.py` and `sources.py`). The dtcore twin is now `BatteryTwin` and the particle filter, which was called
  `BatteryTwin`, is `ParticleFilter`; it now gives the capacity band itself, so the twin no longer reads its particles
  (page data unchanged byte for byte). `decide` is `decide_paper`, named for knowing the cell's true end of life, next
  to `decide_online`. `scripts/check_repro.py` and `scripts/check_dtcore.py` became `tests/test_battery.py`, which also
  tests the README's notes; the grid search kept only to check the closed form is removed. `baseline.csv` gains
  `x_online`. `live/` imports the package instead of editing `sys.path`; `check_live.py` passes unchanged.
- Only two places still edit `sys.path`, both to reach authors' code that cannot be installed: `twins/uav/__init__.py`
  and bikeshare's `scripts/crosscheck_upstream.py` (the frozen UAV study and the tumor scripts aside). `run_checks.sh`
  is now pytest plus the page checks, which it finds itself (`web/checks/check_<name>.js`); the incubator page check
  prints `INCUBATOR WEB CHECK OK` like the others.
- Every number in a twin README has a source: a results file, a test (marked *(test)*), a named frozen study, or a
  comparison made once with its script's place in v0.2.0 (`docs/conventions.md`). How closely the page matches Python
  is now stated in each twin's README rather than in three places. The top-level README shows a short example.
- Two duplicates stay open on purpose and now say so: the page's JavaScript models (`web/README.md`) and battery's live
  service (`twins/battery/README.md`).
- `dtcore.Source.act` raises `NotImplementedError` unless a source overrides it: a decision sent to a source that does
  not handle actions used to be dropped without notice. Every source in the repository already overrides it.
- `dtcore` and `docs/architecture.md` state what `t` and `kind` promise: the meaning and unit of `t` belong to each
  twin and `run()` does not check them; `kind` says where the data come from, not whether actions can change state.

### Fixed
- battery README: the closed-form end of life is about 100× faster than the grid search here (timed 2026-10-19
  against v0.2.0's), not 40×; "one decision out of 82" is seed 0 of 4 in 246 over the three seeds; the dip decisions'
  "wrong twice as often" now gives its counts (3 of 21 against 15 of 225, all three from one cell).
- `twins/battery/scripts/baseline.py` writes the five columns `plot_tradeoff.py` needs (`x_global`, `u_global`,
  `u_twin_true`, `regret_global`, `too_late`); before, the committed `baseline.csv` and `tradeoff_baseline.png` could
  not be regenerated from the repository. The rebuilt columns equal the committed ones in all 1,230 rows.
- All committed results regenerated in one environment (bikeshare's `ref_upstream.json` too), so rerunning the
  scripts now reproduces them byte for byte.
  Against the previous files only floating-point noise changed (at most 2e-11); every decision, count and figure value
  is the same.
- bikeshare README: the forecast RMSEs are recomputed from the authors' saved forecasts (MOVP-RNN 3.61 / 3.58 against
  VP-RNN 3.85 / 3.78; the old figures came from a file this repository does not have); the numerical tie is 3.5e-6,
  not 4e-6; the overdispersion and correlation ranges name the hours they cover; the comparison script takes about a
  minute, not ten.
- incubator README: the 97.7 % controller agreement now says it counts from the first recorded heating (the box's
  controller starts 31 samples into the recording) and that the recording lags by exactly 3 samples, not 2–3.
  "No false alarms on the real data" now says the empty-box run: on the tempeh run, with a load the model does not
  have, the alarm is on for half the samples.

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
