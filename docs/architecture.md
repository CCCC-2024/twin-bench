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
change. It is in the skeleton because it uses only the Twin interface. A model can also be asked only to extract the
what-if's inputs from an event in words, with the twin then computing every choice: in
`twins/bikeshare/scripts/new_conditions.py` the agent turned events into the right inputs but rarely compared choices.
`advise(llm, twin, event)` is that split for any twin: the model answers with one of three calls, `conditions` (the
what-if inputs the event changes), `no_effect` or `cannot` (the event matters but no input expresses it), then
`twin.best(**conditions)`, the twin's own comparison of every action. The model chooses nothing and computes nothing;
inputs that are the twin's actions (`twin.actions`) are dropped from its reply, and conditions the twin cannot run are
told back to the model once and are otherwise an error, not an action. The rule for every twin: a twin states each
condition once as data, `twin.conditions` (name -> the JSON schema of its value, with a description), the inputs of
its `best`; the prompt and the tool definitions are made from that, so a new twin writes no prompt. The calls go in
the reply as JSON, `{"name": ..., "arguments": {...}}`; `chat_model(use_tools=True)` sends them as tool calls instead,
which read stated events as well but said `cannot` less often (8 and 14 of 20 against 14 and 15, Qwen3.5-9B and
gpt-oss-20b). With `ranges=True` the model gives a vague event's inputs as ranges, `{"low": a, "high": b}`, and
`agent.robust_best` moves the ranged inputs together, from each one's value now (the `default` in `twin.conditions`) to
the far end of its range, and takes, of the twin's best actions at those points, the one whose worst regret over them
is smallest; regret is in the twin's `objective` (the number `best` optimizes and its direction), and `advise` returns
it. Until 11/23 the points were the low end, middle and high end of every range in every combination, without the
value now; the results on the first two test sets below are that rule's. Results on held-out events
(`tools/advise_events.py`; repeated runs in the frozen
study `tools/experiments/2026-11-22_event_reading/`): an event with its numbers is read right (27 to 30 of 30 on the
first test set; 14 to 16 of 16 on the second, except 7 for Qwen3.5-9B asked for ranges, which named a call that does not
exist); one that changes nothing is left alone (19 or 20 of 20); an event the what-if has no input for is called
`cannot` in 10 to 15 of 20 at low reasoning effort, and in 18 and 19 of 20 on average by gpt-oss-120b at medium and
high. A vague event is the open problem. On the second test set, acting on one guessed number was worse on average than
not knowing the event in every configuration tried, for both twins whose decision depends on it (incubator: 6.2 to 14.6
minutes of 60 outside the band, against 6.0 for the twin not told; bike share: 2.4 to 5.2 lost riders a day, against
2.30). A range with the robust choice helped the incubator on average in 5 of 7 configurations (down to 3.2) and never
the bike station, whose readings of several inputs together (rentals, returns, hours) mostly missed the event.
With the present rule, on a third set registered before the rule was changed
(`tools/experiments/2026-11-23_checked_ranges/`; 98 vague cells, not told 8.5 minutes and 3.2 riders), acting on the
range was better than not telling the twin for the three gpt-oss configurations at both twins (1.6 to 4.4 minutes; 1.5
to 2.4 riders) and worse for Qwen3.5-9B at the bike station (6.0); over the bike
station's 11 events the 95 % interval includes zero for two of the four configurations. The registered criteria asked
for every configuration and were not met. One guessed number was better than not told in 5 of 8 comparisons there and stayed so under shifted
references in 3. A reading from a small model, or of an event that leaves the hours and the side unstated, should still
be confirmed by a person before a twin acts on it.
Published agent-with-twin systems draw the same line: the
model selects and parameterizes tools and never computes numbers (Mylonas et al. 2026, arXiv 2609.22476, a control-room twin where
only 39 % of answers had numbers supported by the backend once that layer was removed; Kleiman et al. 2025,
arXiv 2505.13761: the model is "an interface and interpreter, not the source of truth"). All four twins write `brief`, `whatif` and `best`; `tools/advise_events.py` runs four
kinds of event on each; bike share's page sends the same prompt when it reads an event typed live.

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

The tumor twin (`twins/tumor`) is on `dtcore` as update, predict and what-if: a replay of one patient's MRI visits,
at each of which it calibrates again. It has no `decide`: there is no reward (a higher dose is always better), and no
parameter uncertainty yet. One what-if takes seconds, so the page replays its saved run and forecasts; CI does not
install the authors' package, and its Python tests are skipped there.

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
| every twin | policy (U) | `decide()` (incubator: a person) | the twin's what-if, `predict(horizon, **scenario)`, as its tools | each twin's rule; the LLM agent (`dtcore.agent`): `ask` for a twin with `brief` and `whatif` (as a policy so far in bikeshare), `advise` for one with `best` too (all four) |
| every twin | reader of events (O, from outside the sensors) | `agent.advise(llm, twin, event)` | the conditions an event in words means for the what-if; the twin's `best(**conditions)` then chooses | any language model; compared by `tools/advise_events.py` against a perfect reader, with and without tool calls |

A learned part must give what the twin asks of it: a battery predictor that only estimates the end of life cannot
replace the filter without a capacity band and the decision's inputs. The one learned part so far shows why the
comparison runs inside the twin: given only what the twin has seen, the VP-RNN forecasts better than the historical
average but does not decide better (`twins/bikeshare/README.md`). The tumor reproduction's Gaussian-process surrogate,
in place of a slow model, is a Q part outside dtcore.

A policy part splits the decision: the twin decides when to decide and whether it can (bike share: at 23:00, once its
rate model covers the next day); the policy, `decide(twin) -> action`, chooses what. Making a routine decision is only
one use of the LLM agent. Where the twin's rule is exact for its own forecast (bike share), the agent can at best
equal the rule's expected shortages; in a replay of real data either can do better, through the forecast's errors.
