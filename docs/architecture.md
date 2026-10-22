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
| `update(obs)` | uses one observation to update D; returns the estimate (with uncertainty if available) | Kalman step | particle-filter step | Bayes step | rate-model step (Gamma–Poisson by default) |
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

**agent** (`dtcore.agent`): an LLM agent that works with any twin through the twin's what-if.
`ask(llm, twin, request, choices=None)` gives a language model the twin's `brief()` (the system and its state in words,
and what its what-if takes and returns) and a request in words; the model runs `twin.whatif(**scenario)` (`predict` for
one scenario, reduced to a few named numbers) as often as its turns allow, and answers. The request can be a decision,
with `choices` the scenarios it may pick, or a question, such as how to adapt to a condition the data never showed.
`llm(messages) -> reply` is any language model; `chat_model(name, url)` reaches one served with the OpenAI chat API
(Ollama, LM Studio). A twin lets the agent work with it by writing `brief` and `whatif`; the agent itself does not
change. It is in the skeleton because it uses only the Twin interface.

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

## Parts of a twin, and where AI attaches

Inside a twin, the estimator (D), the forecaster (Q) and the policy (U) are its parts. AI enters a twin as another
implementation of one of them. A part with a second implementation is a constructor argument whose default is the
twin's physical or statistical model, the baseline it is compared with ([conventions](conventions.md#parts-of-a-twin)).
dtcore has no class for parts: what a twin asks of a part differs from twin to twin, as the table shows.

| Twin | Part | Now | What the twin asks of it | Implementations |
|---|---|---|---|---|
| bikeshare | rate model (Q) | argument: `StationTwin(C, rates)` | each hour's counts (`update`); mean rental and return rates for the hours ahead (`ahead`) | `HistoricalRates` (default), `RNNRates` (the paper's VP-RNN); compared by `scripts/compare_rates.py` |
| bikeshare | policy (U) | argument: `StationTwin(C, rates, policy)` | at 23:00, once the next day can be forecast: the bikes to start it with (`decide(twin)`) | `RulePolicy` (default, the paper's decision), `AgentPolicy` (the LLM agent choosing with the what-if); compared by `scripts/compare_policy.py` |
| uav | model (D, Q, U) | argument: `UAVTwin(model)`, shared with the simulated asset | strain likelihood, damage transitions, damage states, the policy for a reward weight, a roll-out | `UAVModel` (the authors' configuration) |
| battery | particle filter (D, Q) | built in `BatteryTwin` at first-prediction time | a step per cycle giving `log10_a`, `b`, `eol_pred` (the decision's inputs), and a capacity band | `ParticleFilter` |
| incubator | Kalman filter (D); model and controller (Q) | built in `IncubatorTwin`; `predict` takes the controller (open: the twin should keep its own copy) | a step per sample giving the state and the innovation (the lid alarm); a forward simulation | `Kalman`, `Stepper`, `Controller` |
| _template | scalar Kalman filter (D); heater rule (U) | written inside `MugTwin` | a filter step per reading; heater on if the mug would cool below the target within the look-ahead | one each |
| every twin | policy (U) | `decide()` (incubator: a person) | the twin's what-if, `predict(horizon, **scenario)`, as its tools | each twin's rule; the LLM agent (`dtcore.agent`) for a twin with `brief` and `whatif` (so far bikeshare) |

A learned part must give what the twin asks of it: a battery predictor that only estimates the end of life cannot
replace the filter without a capacity band and the decision's inputs. The one learned part so far shows why the
comparison runs inside the twin: given only what the twin has seen, the VP-RNN forecasts better than the historical
average but does not decide better (`twins/bikeshare/README.md`). The tumor reproduction's Gaussian-process surrogate,
in place of a slow model, is a Q part outside dtcore.

A policy part splits the decision: the twin decides when to decide and whether it can (bike share: at 23:00, once its
rate model covers the next day); the policy, `decide(twin) -> action`, chooses what. Making a routine decision is only
one use of the LLM agent, and where the twin's rule is exact for its own forecast (bike share) the agent can at best
equal it; what a rule cannot do is take a new condition stated in words and try it out on the twin's what-if.
