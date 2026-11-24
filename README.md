# Twin Bench

[![checks](https://github.com/CCCC-2024/twin-bench/actions/workflows/checks.yml/badge.svg)](https://github.com/CCCC-2024/twin-bench/actions/workflows/checks.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**One small skeleton, many digital twins.** Twin Bench is a ~120-line Python skeleton (`dtcore`) for the loop every
digital twin shares — observe, update, predict, decide, act — and a growing set of twins from different domains built
on it. Each twin is either a faithful reproduction of a published digital twin, checked against the authors' code, or
built from a paper's description and checked against real data.

**Live page:** https://cccc-2024.github.io/twin-bench/ — the physical asset and its twin side by side, with what-if
controls, all computed in the browser.

## What the twins have shown so far

Building the twins, rather than reading about them, turned up these (details and every number in each twin's README):

- **The reproductions hold.** Bike share: 365 of 366 nightly decisions identical to the authors' code (the other is a
  numerical tie); battery: the twin on `dtcore` equals the original particle filter. CI re-checks both on every push.
  The UAV filter was compared with the authors' model once.
- **A better forecast did not give a better decision.** In the bike-share twin, a learned forecast (the paper's
  VP-RNN) is more accurate than the last-30-days average, yet when the twin sets the bikes at night it does not decide
  better (8.8 vs 7.8 lost per day, within noise). The paper's protocol, where it does better, uses same-day counts that
  are not known at 23:00. Two trainings with the same accuracy chose the same start on only 5 of 61 nights.
  ([bikeshare](twins/bikeshare/#results))
- **A twin that learns from all its history adapts slowly; forgetting fixes the forecast and part of the decision.**
  Rates learned from January–October are about twice the November–December demand. Remembering only the last weeks
  (a window, or a discount factor) halves the forecast error there (RMSE 10.1 → 5.4–5.7 rentals an hour) and loses
  0.85 ± 0.33 fewer riders a day (8.43 → 7.57; the best possible in hindsight is 4.67). How much to forget hardly
  matters below about a month, and the twin can select it from its own recent forecast errors (7.54), though it takes
  about ten days to notice the change. In the busy months no rate model decides differently.
  ([bikeshare](twins/bikeshare/#forgetting))
- **The model can rank actions even where its levels are wrong.** The twin's expected shortages are far from the
  replay (42.8 vs 8.4 per day), but the differences between actions agree better, and a decision only needs those.
- **An LLM agent helps with events the data never showed, and over-reacts to ones that do not matter**
  (preliminary: small local models, one station, 11 nights). Told of a morning rush (rentals ×3, 7–10 am), both agents
  came far closer to the best start than the rule (4–5 vs 18 extra lost); told of a late-evening event that hardly
  affects the night's choice, both lost about 16 where the rule lost none. On ordinary nights neither beat the rule.
- **Words reach every twin the same way: the model extracts the conditions, the twin compares every action.**
  `dtcore.agent.advise` works with all four twins (`tools/advise_events.py`). With a perfect reader, an event the twin
  is not told costs 15 lost riders a day at the bike station and 16.5 of 60 minutes outside the band in the incubator,
  but next to nothing for the battery (utility 0.0004) and the UAV (planner value 0.003, and only with the wing at 40 %
  damage): outside information is worth what the decision is sensitive to. Two local models (Qwen3.5-9B, gpt-oss-20b),
  on 9 held-out events per twin: an event that states its numbers is read right in 27 and 30 of 30 cells, and one that
  changes nothing is left alone in 20 and 19 of 20; one the what-if cannot express is recognized in 14 and 15 of 20. A
  vague event is not solved: the model has to guess a number, and for the incubator the guesses (an outdoor temperature
  for the room) made the twin's choice worse than not knowing, 30 to 54 minutes outside the band against 8. A model six
  times larger (gpt-oss-120b, on one B200) read every stated event too at low effort; more effort made it recognize
  inexpressible events (18 and 19 of 20, against 11 at low) but not guess better. On a second held-out set, registered
  before any run (80 vague cells, 1 to 5 runs per configuration), acting on one guessed number was worse on average than
  not knowing in every configuration, at both the incubator and the bike station. Asking for a range, with the twin
  choosing the action whose worst regret is smallest (`advise(..., ranges=True)`), helped the incubator on average in 5
  of 7 configurations (3.2 minutes against 6.0 at best) and never the bike station, where several inputs must be read
  together and the ranges mostly missed the event ([study](tools/experiments/2026-11-22_event_reading/README.md)).
  The rule was then changed so that the twin also weighs "nothing changes" and moves the ranged inputs together, and
  tested on a third set registered before the change (98 vague cells): better than not telling the twin for the three
  gpt-oss configurations at both twins (incubator 1.6 to 4.4 minutes against 8.5; bike share 1.5 to 2.4 riders against
  3.2), firmly at the incubator, and not for Qwen3.5-9B at the bike station (6.0), so the registered criteria, which
  asked for every configuration, were not met ([study](tools/experiments/2026-11-23_checked_ranges/README.md)). At the
  bike
  station the 95 % interval over its 11 events includes zero for two of the four configurations. The new method is three
  changes at once (the choice rule, a prompt that asks for a confident range, clearer bike-share inputs), which this
  test does not separate. For the gpt-oss models the same situations mostly cost several times less once the words
  state the hours and which side changes.

| Twin | Domain | What it does | Source | Status |
|---|---|---|---|---|
| [incubator](twins/incubator/) | Thermal control | Kalman filter estimates the unmeasured heater temperature, flags an opened lid, what-if on the set point | Built from the description in Feng et al. 2021; real data from Aarhus University | dtcore + page |
| [battery](twins/battery/) | Prognostics | Particle filter on capacity fade, decides when to retire a cell | Reproduces Thelen et al. 2023 on the Severson et al. 2019 cells | dtcore + page + MQTT live demo |
| [uav](twins/uav/) | Structural health | Bayesian filter over 25 damage states, picks 2g / 3g maneuvers | Reproduces Kapteyn, Pretorius & Willcox 2021 | dtcore + page |
| [bikeshare](twins/bikeshare/) | Mobility operations | Learns hourly rental and return rates (Gamma–Poisson), forecasts an empty or full station, sets the overnight number of bikes; what-if on a daytime truck; the rate model can be the paper's trained RNN, the nightly decision an LLM agent | Reproduces Gammelli et al. 2022 (queueing model of Raviv & Kolka 2013) on Citi Bike station data | dtcore + page |
| [tumor](twins/tumor/) | Oncology | Calibrates a glioma reaction–diffusion model to MRI, predicts chemo doses | Reproduces TumorTwin (Kapteyn et al. 2026) | dtcore (update, predict, what-if; no decision) + page (replay) |
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
`Recorder` keeps every observation, estimate and action (in memory and optionally SQLite); `dtcore.agent` lets an LLM
agent work with any twin through the twin's what-if. The six-element view
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
The tumor twin needs PyTorch and the authors' package; see [twins/tumor](twins/tumor/). Its tests are skipped without them.

The LLM agent (`dtcore.agent`) needs a language model served with the OpenAI chat API; nothing else, and no test needs
one. With [Ollama](https://ollama.com) on macOS:

```bash
brew install ollama                    # the program, in Homebrew's prefix
ollama serve                           # the server, at http://localhost:11434 (keep it running while the agent works)
ollama pull qwen3.5:9b                 # models go to ~/.ollama/models, outside the repository (about 6 GB)
python twins/bikeshare/scripts/compare_policy.py qwen3.5:9b
```

## Repository layout

```
dtcore/            the skeleton: Source, Twin, run(), Recorder; agent (an LLM agent for any twin)
twins/<name>/      one folder per twin, importable as twins.<name>: README, code, scripts/, results/, fetch_upstream.sh
web/               the shared page: one shell, one module per twin, JS checked against Python
docs/              architecture, tools and steps, adding a twin, design decisions, file conventions
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
each with its own check. [docs/tools.md](docs/tools.md) says which tools each part of a twin needs and what the field
uses instead; [docs/design-decisions.md](docs/design-decisions.md) says what was tried and why. Proposals for new domains are welcome — open an issue with the *New twin* template.

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
