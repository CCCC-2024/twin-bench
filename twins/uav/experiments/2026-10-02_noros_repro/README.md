# Kapteyn UAV closed loop without ROS (2026-10-02)

Authors' repository @ `b51e42d`; `UAV.py`, `planner.py` and `UAVconfig.json` unchanged.

- `kapteyn_noros_driver.py` — our 25-state Bayes filter in place of `graphicalmodel.py`, closed loop for 60 steps;
  `--seeds 20` adds a multi-seed summary (`mc_summary.json`, `belief_vs_truth_seed0.png`, `run_seed0.json`).
- `crosscheck_pomegranate.py` — the authors' pomegranate graphical model and our filter in lock step (same
  observations, same actions); run from the UAV-Digital-Twin root. Results: `crosscheck.json`, `crosscheck_h0.json`.
- `metrics_1007.py` → `metrics_1007.json` — error vs decision: 20 seeds × 60 steps, twin transition probability
  × 1.0 / 0.2 / 0.0. Notes: `analyze_output_2026-10-06.md`. `analyze_error_vs_decision.py` and
  `trace_bidirectional.py` are the analyses behind it.

## Run (from this folder)
```bash
python kapteyn_noros_driver.py --repo ../../upstream/UAV-Digital-Twin --steps 60 --seed 0 --seeds 20
python metrics_1007.py
```

## Installing pomegranate 0.13.0 (worked on Linux)
Python 3.8 + numpy 1.19.5 + cython<3 + scipy 1.5.4 + networkx<2.6 + setuptools<60; unpack the source archive and run
`python setup.py install`. pip / uv builds fail because line 68 of its setup.py (`__builtins__.__NUMPY_SETUP__`)
raises AttributeError when not run as `__main__`. pygraphviz is only used for drawing; a stub module works.

## Comparison (seed 0, 30 steps)
| Authors' prediction window | Max total-variation distance of the posterior | Different actions |
|---|---|---|
| 10 steps (authors' default) | 0.094 | 0/30 |
| 0 steps | 0.014 | 0/30 |

The difference grows with the authors' future-prediction nodes, likely from pomegranate's approximate (loopy belief
propagation) inference on a graph with cycles; the remaining 0.014 is not fully explained. Our filter is exact forward
filtering of the same model.
