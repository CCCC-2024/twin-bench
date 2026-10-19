# Twin Bench

[![checks](https://github.com/CCCC-2024/twin-bench/actions/workflows/checks.yml/badge.svg)](https://github.com/CCCC-2024/twin-bench/actions/workflows/checks.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**One small skeleton, many digital twins.** Twin Bench is a ~120-line Python skeleton (`dtcore`) for the loop every
digital twin shares — observe, update, predict, decide, act — and a growing set of twins from different domains built
on it. Each twin is either a faithful reproduction of a published digital twin, checked against the authors' code, or
built from a paper's description and checked against real data.

**Live page:** https://cccc-2024.github.io/twin-bench/ — the physical asset and its twin side by side, with what-if
controls, all computed in the browser.

| Twin | Domain | What it does | Source | Status |
|---|---|---|---|---|
| [incubator](twins/incubator/) | Thermal control | Kalman filter estimates the unmeasured heater temperature, flags an opened lid, what-if on the set point | Built from the description in Feng et al. 2021; real data from Aarhus University | dtcore + page |
| [battery](twins/battery/) | Prognostics | Particle filter on capacity fade, decides when to retire a cell | Reproduces Thelen et al. 2023 on the Severson et al. 2019 cells | dtcore + page + MQTT live demo |
| [uav](twins/uav/) | Structural health | Bayesian filter over 25 damage states, picks 2g / 3g maneuvers | Reproduces Kapteyn, Pretorius & Willcox 2021 | dtcore + page |
| [bikeshare](twins/bikeshare/) | Mobility operations | Learns hourly rental and return rates (Gamma–Poisson), forecasts an empty or full station, sets the overnight number of bikes; what-if on a daytime truck; the rate model can be the paper's trained RNN | Reproduces Gammelli et al. 2022 (queueing model of Raviv & Kolka 2013) on Citi Bike station data | dtcore + page |
| [tumor](twins/tumor/) | Oncology | Calibrates a glioma reaction–diffusion model to MRI, predicts chemo doses | Reproduces TumorTwin (Kapteyn et al. 2026) | reproduction only |
| [_template](twins/_template/) | — | A cooling mug with a heater: the smallest complete twin, copy it to start | — | dtcore |

## How it fits together

```
 physical side: a Source                               digital side: a Twin
 (sensor feed, recorded data or simulator)             (models + estimator + decision rule)

   observe() ──────────── observation ───────────────▶  update(obs)  -> estimate of the state
                                                         predict(horizon, **scenario)  -> forecast / what-if
   act(action) ◀────────────── action ───────────────  decide()     -> action or None
   advance()

 dtcore.run(source, twin): observe -> update -> decide -> act -> record -> advance, until the source runs out
```

```python
from dtcore import Recorder, run
from twins._template import MugTwin, SimMug

rec = Recorder("mug")
run(SimMug(), MugTwin(), recorder=rec)       # one hour of a simulated mug whose heater the twin controls
print(rec.rows[-1]["est"])                    # the twin's last estimate: temperature and its uncertainty
```

`dtcore` knows nothing about temperatures, batteries or drones. A twin plugs in by implementing `Source` and `Twin`;
`Recorder` keeps every observation, estimate and action (in memory and optionally SQLite). The six-element view
behind it (physical state, observations, digital state, quantities of interest, actions, reward; Kapteyn et al. 2021)
is described in [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/CCCC-2024/twin-bench.git && cd twin-bench
python -m venv .venv && source .venv/bin/activate          # Python 3.10+
pip install -e ".[twins,rnn,dev]"                          # rnn: PyTorch, for the bike-share RNN rate model
pytest tests/test_template.py                               # the template twin, no downloads needed

for t in incubator battery uav bikeshare; do twins/$t/fetch_upstream.sh; done   # third-party code and data, pinned commits
./run_checks.sh                                             # everything CI runs; ends with ALL CHECKS OK
```

Open the page locally: `python web/build.py`, then open `web/index.html`. The web checks need Node.js 18+.
The tumor twin needs PyTorch and its own environment; see [twins/tumor](twins/tumor/).

## Repository layout

```
dtcore/            the skeleton: Source, Twin, run(), Recorder
twins/<name>/      one folder per twin, importable as twins.<name>: README, code, scripts/, results/, fetch_upstream.sh
web/               the shared page: one shell, one module per twin, JS checked against Python
docs/              architecture, adding a twin, file conventions
tests/             tests for dtcore and the twins (pytest)
tools/             helper scripts (fetching pinned upstream code)
```

Where each kind of file goes is fixed in [docs/conventions.md](docs/conventions.md).

## How results are checked

CI runs [run_checks.sh](run_checks.sh) on each push:

- the tests in `tests/` assert, through each twin, the results its README states (e.g. battery: the retirement cycle
  of the paper's example cell, and the twin on `dtcore` equal to the original particle filter; bike share: the
  learned rates equal the paper's benchmarks and the decisions equal the authors' code except one numerical tie);
- the page's JavaScript matches the Python reference (each twin's README says how closely).

Two comparisons with the authors' own code were made once and are not rerun: the incubator model against their
implementation, and the UAV filter against their pomegranate model (their software no longer installs cleanly). Each
twin's README gives the result and where the script is.

## Adding a twin

Start from [twins/_template](twins/_template/) and follow [docs/adding-a-twin.md](docs/adding-a-twin.md):
describe the six elements, then add the model, observations, update, predict, decide and the page one step at a time,
each with its own check. Proposals for new domains are welcome — open an issue with the *New twin* template.

## Data and licenses

Twin Bench's own code is MIT-licensed. Third-party code and data are **not** stored in this repository: each twin's
`fetch_upstream.sh` downloads them at a pinned commit, under their own licenses. Data from sources without a license
(the incubator experiments) are used locally only and never published. Details: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citing

If you use Twin Bench, please cite it ([CITATION.cff](CITATION.cff)) **and** the papers behind the twins you use;
each twin's README lists them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are recorded in [CHANGELOG.md](CHANGELOG.md).
Twin Bench started as a graduate independent study at the University of Florida (2026).
