# Architecture

## The six elements

Every twin here is described with the abstraction of Kapteyn, Pretorius & Willcox (2021, *Nature Computational
Science*, arXiv 2012.05841), which the authors note "can be applied to digital twins from any discipline and
application" and which later work reuses directly (e.g. Chaudhuri et al., arXiv 2308.12429). We use the variables as
a template only; we do not implement the paper's full probabilistic graphical model.

| Element | Meaning |
|---|---|
| **S** physical state | the true state of the asset, usually only partly observable |
| **O** observations | data arriving from the physical side |
| **D** digital state | the twin's estimate of S plus model parameters, with uncertainty where possible |
| **Q** quantities of interest | what the twin computes with its models and the decision needs |
| **U** actions | what the twin decides from D and Q and sends back to the physical side |
| **R** reward | how good an action is |

The twins in this repository:

| | incubator | battery | uav | tumor |
|---|---|---|---|---|
| S | air and heater-plate temperature (plate not measured) | true capacity | damage in two wing regions | 3D tumor cell density |
| O | 2 box + 1 room temperature sensors every 3 s; heater on/off | capacity per measured cycle | 24 strain gauges (divided by load) | MRI (ADC → cellularity), 10 visits |
| D | two temperatures + 4 thermal parameters | fade-model parameters (log10 a, b), 200 particles | probabilities over 25 damage states | calibrated (d, k, α_RT, α_CT), point estimate |
| Q | future temperature curve | capacity curve, end of life with interval | predicted strain, structural health | total tumor cell count over time |
| U | heater on / off (set point) | retirement cycle | 2g / 3g turn | chemotherapy dose |
| R | ours: stay in the band, use little energy | multi-attribute utility: total Ah + mean discharge time | health reward + maneuver reward, value iteration | none (no toxicity cost) |
| Update | Kalman filter; least-squares calibration | particle filter each cycle | discrete Bayes filter | Levenberg–Marquardt recalibration |
| Data | real experiments (public); simulated box for closed loop | real measurements (Severson et al.), replayed | simulation: precomputed strain table + noise | synthetic, noise-free |

## The skeleton: dtcore

Only what all twins share went into the skeleton; anything used by one twin stays in that twin.

**Twin** (`dtcore.Twin`)

| Method | Does | incubator | battery | uav | tumor |
|---|---|---|---|---|---|
| `update(obs)` | uses one observation to update D; returns the estimate (with uncertainty if available) | Kalman step | particle-filter step | Bayes step | recalibration |
| `predict(horizon, **scenario)` | forecasts Q; a non-empty scenario makes it a what-if | simulate a heating plan | power-law extrapolation | belief roll-out per maneuver | solve per dose plan |
| `decide()` | chooses U, or returns None | (human decides) | retirement utility | policy | not supported |
| `capabilities` | which of the above the twin supports | | | | |

**Source** (`dtcore.Source`), the physical side: `observe()` returns the next observation as a dict with at least
`t` (or None when finished); `act(action)` receives a decision — a simulator changes its state, a replay can only
record it; `advance()` moves to the next sample. `kind` is `"sim"`, `"replay"` or `"sensor"`.

**run(source, twin, max_steps=None, recorder=None, before_step=None)** loops
observe → update → decide (if the twin declares it) → act → record → advance. `before_step(k, source, twin)` lets an
experiment change the world at step k (open a lid, damage a wing) without touching the loop.

**Recorder** keeps every observation, estimate and action in memory and, with `path=`, in an SQLite table
`records(system, run, k, t, kind, data)`.

Deliberately *not* in the skeleton (one twin uses each so far): MQTT transport (battery live demo), specific filters,
each twin's page module.

## What the four twins show about the loop

- **incubator** is the only one with a true closed loop: with the simulated box as the physical side, the heater
  decision changes the temperature.
- **battery** replays measurements: retirement cannot change the cell, it only decides where the replay stops.
- **uav**: in the authors' setup damage follows a fixed script regardless of maneuvers, so the loop closes on the
  twin's side only; the page's asset makes damage grow with the maneuvers chosen.
- **tumor** has no reward (higher dose is always better) and no parameter uncertainty yet, so it is not a decision
  problem; one calibration takes about half an hour.

## Where AI can attach

- **Q**: a learned surrogate in place of a slow model (the tumor twin's Gaussian-process surrogate).
- **D**: a learned estimator (e.g. damage straight from strain readings).
- **U**: a policy trained inside the twin, or an LLM agent that calls `predict` / what-if as tools.
