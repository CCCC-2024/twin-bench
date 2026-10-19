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

Each twin's README has its own six-element table and lists every field its Sources return, whether the real asset
has it, and what an action changes: [incubator](../twins/incubator/README.md), [battery](../twins/battery/README.md),
[uav](../twins/uav/README.md), [bikeshare](../twins/bikeshare/README.md). That table is the one place for those facts.

## The skeleton: dtcore

Only what all twins share went into the skeleton; anything used by one twin stays in that twin.

**Twin** (`dtcore.Twin`)

| Method | Does | incubator | battery | uav | bikeshare |
|---|---|---|---|---|---|
| `update(obs)` | uses one observation to update D; returns the estimate (with uncertainty if available) | Kalman step | particle-filter step | Bayes step | Gamma–Poisson step |
| `predict(horizon, **scenario)` | forecasts Q; a non-empty scenario makes it a what-if | simulate a set point (takes the box's controller) | power-law extrapolation; what-if on the decision weights | belief roll-out; what-if on the maneuver weight | bikes over the next hours; what-if: a truck sets s bikes now |
| `decide()` | chooses U, or returns None | none: a person decides with the what-if | retirement, once, at 95 % capacity | 2g or 3g, every turn | bikes for tomorrow, every night |
| `capabilities` | which of the above the twin supports; `run()` reads `"decide"` | | | | |

**Source** (`dtcore.Source`), the physical side: `observe()` returns the next observation as a dict with at least
`t` (or None when finished); what `t` means and its unit belong to the twin (seconds, cycles, hours, steps), and
`run()` does not check it. `act(action)` receives a decision and says what it does with it: a simulator changes its
state, a replay of the incubator only records it, the bike-share replay keeps the recorded demand but changes the
bikes in the station. A source that does not override `act` raises an error when a decision arrives, so no decision is
dropped silently. `advance()` moves to the next sample. `kind` is `"sim"`, `"replay"` or `"sensor"` (where the data
come from).

**run(source, twin, max_steps=None, recorder=None, before_step=None)** loops
observe → update → decide (if the twin declares it) → act → record → advance. `before_step(k, source, twin)` lets an
experiment change the world at step k (open a lid, damage a wing) without touching the loop.

**Recorder** keeps every observation, estimate and action in memory and, with `path=`, in an SQLite table
`records(system, run, k, t, kind, data)`.

Deliberately *not* in the skeleton (one twin uses each so far): MQTT transport (battery live demo), specific filters,
each twin's page module.

## What the four twins show about the loop

- **incubator**: with the simulated box as the physical side the loop is closed: a set point sent to the box changes
  its temperature, which the twin then sees. The twin itself does not decide; a person uses its what-if. Replaying the
  real experiment, actions are only recorded.
- **battery** replays measurements: retirement cannot change the cell, it only decides where the replay stops.
- **uav**: in the authors' setup damage follows a fixed script regardless of maneuvers, so the loop closes on the
  twin's side only; the page's asset makes damage grow with the maneuvers chosen.
- **bikeshare** replays the recorded demand but simulates the bikes: the nightly decision changes the inventory and so
  the shortages, while the demand stays as recorded. The twin never sees the demand it loses (censored demand).

The tumor reproduction (`twins/tumor`) is not on `dtcore`: it has no reward (a higher dose is always better), no
parameter uncertainty yet, and one calibration takes about half an hour.

## Where AI can attach

- **Q**: a learned forecast in place of a statistical one (bikeshare: the paper's RNN demand forecast would be a
  second rate model passed to `StationTwin`), or a learned surrogate in place of a slow model (the tumor
  reproduction's Gaussian process).
- **D**: a learned estimator (e.g. damage straight from strain readings).
- **U**: a policy trained inside the twin, or an LLM agent that calls `predict` / what-if as tools.
