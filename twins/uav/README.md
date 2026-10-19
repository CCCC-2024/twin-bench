# UAV twin

Reproduces the structural-health twin of a fixed-wing UAV by Kapteyn, Pretorius & Willcox (2021): strain readings
update a belief over 25 damage states of two wing regions, and at every step the twin chooses a 2g or 3g turn.
The authors' ROS nodes are replaced by plain Python.

## Sources
- Kapteyn, Pretorius & Willcox 2021, *A probabilistic graphical model foundation for enabling predictive digital
  twins at scale*, Nature Computational Science (arXiv 2012.05841).
- michaelkapteyn/UAV-Digital-Twin @ `b51e42d` (BSD 3-Clause) and three files of
  michaelkapteyn/UAV-Experimental-Calibration @ `1dab634` (the FEA interface functions are not public), fetched by
  `fetch_upstream.sh`.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| damage in two wing regions | 24 strain gauges, divided by load | probabilities over 25 damage states | predicted strain, structural health | 2g / 3g turn | health reward + maneuver reward (3g +0.1, 2g −0.1), policy by value iteration |

### Observations and actions

One step is one turn. What a Source returns, and what the twin uses:

| Field | Meaning | A real UAV has it? | `UAVTwin.update` reads it |
|---|---|---|---|
| `t` | turn index (0, 1, 2 …) | — | no (passed through) |
| `m` | the 24 strain readings of this turn, divided by the load factor | yes | yes |
| `u` | the maneuver flown this turn: 0 = 2g, 1 = 3g | yes | yes |
| `true1`, `true2` | the simulated true damage of each region, in percent | no | no; for scoring only |

Action `{"u": 0 or 1, "p3g": …}`: the maneuver for the next turn; the Source reads only `u`. `p3g`, the share
of the predicted belief on states where the policy says 3g, is only recorded. On `ScriptedUAV` the maneuver changes
the load and so the readings, but damage follows the authors' script; on `ControlledUAV` it also changes how fast
damage grows.

## Files
- `__init__.py`: the public names, `from twins.uav import UAVTwin, ScriptedUAV, ControlledUAV, UAVModel`; also the one
  place the authors' `digitaltwin` package is put on the import path.
- `twin.py` — `UAVTwin`: 25-state Bayes filter + the authors' ControlP decision.
- `sources.py` — `ScriptedUAV` (the authors' asset: damage follows a fixed script) and `ControlledUAV` (the page's
  asset: damage grows with the chosen maneuvers at the twin's assumed rates; can be damaged by hand).
- `model.py` — `UAVModel`: the authors' configuration, strain table, transition model and reward; `policy(alpha2)`
  is the authors' planner algorithm with an adjustable maneuver weight.
- `scripts/export_web.py` — `web/uav_model.js` and the page reference `results/ref_web_uav.json`.
- `experiments/2026-10-02_noros_repro/` — the first no-ROS reproduction (a frozen study), the comparison with the
  authors' pomegranate graphical model, and the error-vs-decision experiment.
- `tests/test_uav.py` (in the repository's `tests/`): the results below marked *(test)*.

## Run
```bash
twins/uav/fetch_upstream.sh
pytest tests/test_uav.py                      # about a second
python twins/uav/scripts/export_web.py        # page data and reference run, a few seconds
node web/checks/check_uav.js twins/uav/results web    # UAV WEB CHECK OK
```

## Results
- `policy(2.5)` gives the authors' planner policy in every one of the 25 states *(test)*.
- Against the authors' inference (pomegranate), the same maneuver is chosen at 30 of 30 steps (the study above; it
  needs Python 3.8 and pomegranate 0.13).
- On the authors' asset, 20 seeds × 60 turns: the estimate is off by more than 10 percentage points 33 % of the time,
  almost all in region 2 (31 %, region 1 3.5 %), which the sensors barely see; the maneuver differs from the policy at
  the grid state nearest the true damage 2.4 % of the time, always switching to 2g one or two turns late *(test)*. If
  the twin assumes damage does not grow, the maneuver differs 48 % of the time
  (`experiments/2026-10-02_noros_repro/metrics_1007.json`).

## Notes
- The authors' asset advances damage by script, independent of maneuvers; the twin's growth model is 5 % (2g) or
  10 % (3g) per region per step. Only the page's `ControlledUAV` makes damage depend on maneuvers, so that decisions
  have consequences.
- The original picks the action with the highest ControlP marginal, not the policy at the most likely state.
- Readings = interpolated precomputed strain + noise (σ = 150), divided by load; the twin assumes σ = 125 / √load.
- The authors' measurement generator uses scipy's `LinearNDInterpolator`; once the true damage drifts off the grid by
  rounding, readings differ slightly between scipy versions. Compare runs within one environment.
