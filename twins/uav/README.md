# UAV twin

Reproduces the structural-health twin of a fixed-wing UAV by Kapteyn, Pretorius & Willcox (2021): strain readings
update a belief over 25 damage states of two wing regions, and at every step the twin chooses a 2g or 3g turn.
The authors' ROS nodes are replaced by plain Python.

## Sources
- Kapteyn, Pretorius & Willcox 2021, *A probabilistic graphical model foundation for enabling predictive digital
  twins at scale*, Nature Computational Science (arXiv 2012.05841).
- michaelkapteyn/UAV-Digital-Twin @ `b51e42d` (BSD 3-Clause) and three files of
  michaelkapteyn/UAV-Experimental-Calibration @ `1dab634` (the FEA interface functions are not public), fetched by
  `fetch_upstream.sh`. Per-file checksums of the original snapshot: `upstream_sources.json`.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| damage in two wing regions | 24 strain gauges, divided by load | probabilities over 25 damage states | predicted strain, structural health | 2g / 3g turn | health reward + maneuver reward (3g +0.1, 2g −0.1), policy by value iteration |

## Files
- `udt/model.py` — the authors' configuration, strain table, transition model and reward; `policy(alpha2)` is the
  authors' planner algorithm with an adjustable maneuver weight.
- `udt/assets.py` — `ScriptedUAV` (the authors' asset: damage follows a fixed script) and `ControlledUAV` (the page's
  asset: damage grows with the chosen maneuvers at the twin's assumed rates; can be damaged by hand).
- `udt/twin.py` — 25-state Bayes filter + the authors' ControlP decision.
- `scripts/check_dtcore.py` — `policy(2.5)` equals the authors' planner, and 60 steps through dtcore equal the
  no-ROS reproduction bit for bit.
- `scripts/export_web.py` — `web/uav_model.js` and the page reference `results/ref_web_uav.json`.
- `experiments/2026-10-02_noros_repro/` — the first no-ROS reproduction, the comparison with the authors' pomegranate
  graphical model, and the error-vs-decision experiment.

## Run
```bash
twins/uav/fetch_upstream.sh
python twins/uav/scripts/check_dtcore.py      # UAV DTCORE OK
python twins/uav/scripts/export_web.py
```

## Results
- Against the authors' inference (pomegranate), the same maneuver is chosen at 30 of 30 steps.
- 20 seeds × 60 steps: the estimate is off by more than 10 percentage points 33 % of the time, almost all in region 2,
  which the sensors barely see; the wrong maneuver is chosen 2 % of the time (switching to 2g one or two steps late).
  If the twin assumes damage does not grow, 48 % (`experiments/2026-10-02_noros_repro/metrics_1007.json`).

## Notes
- The authors' asset advances damage by script, independent of maneuvers; the twin's growth model is 5 % (2g) or
  10 % (3g) per region per step. Only the page's `ControlledUAV` makes damage depend on maneuvers, so that decisions
  have consequences.
- The original picks the action with the highest ControlP marginal, not the policy at the most likely state.
- Readings = interpolated precomputed strain + noise (σ = 150), divided by load; the twin assumes σ = 125 / √load.
- The authors' measurement generator uses scipy's `LinearNDInterpolator`; once the true damage drifts off the grid by
  rounding, readings differ slightly between scipy versions. Compare runs within one environment.
