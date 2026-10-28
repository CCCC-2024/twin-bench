# Twin Bench page

One page shell, one module per twin (incubator, battery, UAV, bike share). The simulation is the page: each module brings
its 3D asset, with the controls pinned beside the objects they act on (the asset's by the asset, the twin's what-if by
the twin; `data-pin` names an anchor of the scene) and two bubbles: what happened to the asset and how the twin reacted.
The twin panel beside it has two readings and a chart. Everything runs in
the browser; arrow keys or a presentation clicker switch the twin.

A module fills the shell with its own parts: `renderWorld` for what happens to the physical object, and, if the twin has
a part with more than one implementation, `renderParts` for the part a person chooses (bike share: the demand forecast,
historical average or the paper's VP-RNN). Choosing restarts the run, so the same source drives each choice from the same
start. A twin with an LLM agent also fills `renderAgent`, opened over the page from the asset's controls (bike share:
"Event in words…": the agent's recorded runs on events described in words, whose extracted what-if inputs can be
edited; opened from this computer, an event can also be typed and its inputs extracted live by a model served there).

Published by `.github/workflows/pages.yml` on every push to `main` that touches `web/`.

## Files
| File | What | Edit by hand? |
|---|---|---|
| `page.html` | the page (a fragment without `<head>`) | yes — the only page source |
| `build.py` | wraps `page.html` into `index.html` | yes |
| `index.html` | built page; open it in a browser | no (git-ignored) |
| `incubator_core.js` | incubator model, Kalman filter, controller, simulated box; ported from `twins/incubator` | yes |
| `battery_core.js` | retirement decision; ported from `twins/battery` | yes |
| `uav_core.js` | Bayes filter, policy, closed-loop asset; ported from `twins/uav` | yes |
| `bikeshare_core.js` | station model (matrix exponential), the twin with its rate model (Gamma–Poisson, or the VP-RNN's rates precomputed in Python), simulated or replayed station; ported from `twins/bikeshare` | yes |
| `battery_cells.js` | measured capacity and twin states of 3 cells, from `twins/battery/scripts/export_web.py` | no |
| `uav_model.js` | strain table, transition model, planner reward (authors' data, BSD 3-Clause), from `twins/uav/scripts/export_web.py` | no |
| `bikeshare_model.js` | the twin's learned totals per hour and day type and the Nov–Dec 2018 average demand (aggregates of Citi Bike data), from `twins/bikeshare/scripts/export_web.py` | no |
| `bikeshare_events.js` | for each night of the agent experiment on events (`twins/bikeshare/scripts/new_conditions.py`), the twin's forecast rates for the next day and the recorded agent runs, from `twins/bikeshare/scripts/export_web.py` | no |
| `incubator_replay.js` | the real incubator experiment, from `twins/incubator/scripts/export_web.py`; loaded only when the page is opened locally | no (git-ignored, never published: the data have no license) |
| `bikeshare_replay.js` | the recorded Citi Bike rentals and returns of Nov–Dec 2018 and, with the `rnn` extra, the VP-RNN twin's rates after each of those hours, from `twins/bikeshare/scripts/export_web.py`; loaded only when the page is opened locally | no (git-ignored, never published: no redistribution as a stand-alone dataset) |
| `checks/check_<twin>.js` | compares the page's JavaScript with the Python reference | yes |

## Build and check
```bash
python web/build.py                                                      # -> web/index.html
node web/checks/check_incubator.js twins/incubator/results web           # INCUBATOR WEB CHECK OK
node web/checks/check_battery.js twins/battery/results web               # BATTERY WEB CHECK OK
node web/checks/check_uav.js twins/uav/results web                       # UAV WEB CHECK OK
node web/checks/check_bikeshare.js twins/bikeshare/results web           # BIKESHARE WEB CHECK OK
```
To extract inputs from events live on the bike-share panel, serve the page from this computer and keep a model server running
(Ollama allows requests from `127.0.0.1` by default):
```bash
cd web && python -m http.server 8000 --bind 127.0.0.1                     # then open http://127.0.0.1:8000
```
The incubator and bike-share checks also replay recordings that are not committed: build them first with
`python twins/<name>/scripts/export_web.py --replay-only` (`run_checks.sh` does this for every twin).

Python is the reference and the JavaScript is checked against it; each twin's README says how closely. `run_checks.sh`
runs every `checks/check_<name>.js` it finds.

Open: each `*_core.js` is a second implementation of its twin's model, kept equal to the Python by these checks.
Removing it (running the Python twins behind the page) is a decision left for later.
