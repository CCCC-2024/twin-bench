# Design decisions

What was tried, what was chosen and why, in the order the repository grew. Each row is one decision: the choice, the
reason, and the option given up. Results quoted here are stated with their sources in each twin's README; the tools
are compared with the field's in [tools.md](tools.md).

## The short version

- **A small skeleton instead of a platform.** One asset on one machine needs a loop and two interfaces. The skeleton
  was extracted only after two twins fitted it, and holds only what all twins share.
- **Reproduce before building.** Three twins reproduce published ones and are checked against the authors' code or
  numbers; one was built from a paper's description and checked against real data. A twin whose model is already
  right lets a new tool or a new AI part be judged on its own.
- **A part that can vary is a parameter.** The rate model and the nightly policy of the bike-share twin are passed
  in, so a learned forecast or a language model is compared with the part it replaces on the same twin and data.
- **The language model reads; the twin computes.** After the model was found to confirm its own answer when asked to
  choose, its job was narrowed to turning an event in words into the what-if's inputs.
- **Report only what a test locks.** Every number in a README is asserted by a test or names the script that made it.

## 1. The skeleton (`dtcore`)

| Decision | Why | Given up, and why |
|---|---|---|
| Describe every twin with six elements (S, O, D, Q, U, R) before writing code | Kapteyn et al. used them in two domains; all four systems here fit them, so the template does not come from one system | Abstracting from the first twin alone |
| The skeleton holds `update`, `predict` (a scenario makes it a what-if), an optional `decide`, and `observe` / `act` on the physical side | fewer things to change when a new system arrives | MQTT, specific filters and the display in the skeleton: one twin uses each |
| Extract it only after the incubator and the battery both ran on the same loop | two instances show what is shared | a loop inside each twin: nobody can say what is common |
| An existing twin joins through one adapter file; its filter and decision are not rewritten | shows the cost of adopting the skeleton; the reproduced results stay bit-identical | rewriting the twin to suit the skeleton |
| A source that receives an action must say what it does with it | a decision is never dropped silently; a replay cannot change recorded data, but can change the state actions affect | ignoring actions in replays |

## 2. Which systems, and in what order

| Decision | Why | Given up, and why |
|---|---|---|
| Start with a reproduced battery twin to learn the tools | one new thing at a time: the model already matched the paper, and it uses real data | a new system first: a new domain and new tools together, and no way to tell which is wrong |
| The first twin built from a description is the incubator | a standard teaching twin with a full description, public real data and a real closed loop | systems with no reference to check against |
| Write the incubator model from the paper; use the authors' code only to compare (largest difference 0.0034 °C) | shows that a description is enough; the upstream has no license | importing the authors' implementation |
| Use the authors' published parameters for the incubator | with only the air temperature measured, the heater's parameters are not identifiable; a free fit is slightly closer (0.41 against 0.46 °C) and physically impossible | the free fit |
| The UAV asset's damage grows with the maneuvers flown | in the authors' scripted asset the twin's decision has no consequence | the scripted asset |
| A fourth twin, one bike-share station | one paragraph describes it; public real data; a decision with a trade-off; a different kind of model (random arrivals, people); a paper with code to check against | another thermal system; waiting for a system description |
| The tumor twin stays a reproduction, off the skeleton | one calibration takes about half an hour, the license is academic-only, and it has no decision | forcing it into the loop |

## 3. Models and updates

| Decision | Why | Given up, and why |
|---|---|---|
| Exact discretization of the incubator's linear model | one fewer source of numerical error | a general ODE solver |
| The battery's particle filter takes one measurement at a time, and the batch function calls it | one copy of the algorithm; results are bit-identical to the batch | a second, live version that would drift from the first |
| The battery decides over cycles up to its *predicted* end of life | an online twin cannot know the true end of life; the authors' range uses it | the authors' range (1 of 82 cells differs by one cycle) |
| The battery replay sends only measured cycles | the authors' 3,000 extrapolated cycles are not measurements | replaying extrapolated data as if measured |
| Bike share: exact matrix exponential instead of the authors' Runge–Kutta | the largest difference from upstream (9e-4) is upstream's integration error | copying the integrator |
| Bike share: Gamma–Poisson update of the hourly rates | its posterior mean equals the paper's two benchmarks, so the twin learns them itself, with an uncertainty | a Kalman filter (rates are counts); reading the benchmark column from the data |
| The automatic decision is the paper's nightly rebalancing only; a daytime truck is a what-if for a person | the nightly decision can be checked against the paper day by day | automatic daytime dispatch with no reference |

## 4. Data and third-party code

| Decision | Why | Given up, and why |
|---|---|---|
| Third-party code and data are downloaded by `fetch_upstream.sh` at a pinned commit, never committed | fixed versions; some upstreams have no license | copying them into the repository |
| The public page carries no incubator recordings and only aggregated bike-share counts | no license upstream; Citi Bike's terms forbid redistributing the data as a dataset | publishing the recordings |
| The simulated bike station's true demand is the November–December average, not the twin's learned rates | the asset and the twin must differ for the twin's error to be visible | a station that agrees with its twin by construction |

## 5. The page

| Decision | Why | Given up, and why |
|---|---|---|
| One page, the asset and its twin side by side; the twin gets only sensor readings | shows the two-way link, and what the twin cannot see | curves only; one picture for both |
| The page computes in JavaScript, ported from Python, with Python as the reference and a check in CI | opens with a double click, and the what-if responds while dragging | a Python backend: one more service to start |
| The battery's particle filter stays in Python and the page replays its states; the what-if is computed in the page | random numbers cannot match bit for bit across languages; the what-if is deterministic | porting the filter and checking it only statistically |
| A shared shell with one module per twin | switching systems shows what is common | one page per twin |
| three.js for the 3D | can be rotated, loads from a CDN | flat diagrams |

## 6. Infrastructure for the live demo

| Decision | Why | Given up, and why |
|---|---|---|
| MQTT with the amqtt broker | the common IoT protocol; installs with pip | Mosquitto (a system install); Kafka and Eclipse Ditto (built for fleets) |
| SQLite for records | ships with Python | InfluxDB, which only Grafana would need |
| Streamlit for the live battery page | a what-if must respond to a slider; one Python process | Grafana: weaker at interactive what-ifs and two more services |

## 7. AI in the twin: a learned forecast

| Decision | Why | Given up, and why |
|---|---|---|
| The AI part is the paper's own RNN forecast, with the authors' trained weights, run inside the twin's loop | a reproduced network can be checked against the authors' recorded metrics (all 18 match) | training a new forecaster with nothing to check it against |
| The rate model is a parameter of the twin (`StationTwin(C, rates)`) | the same twin and data run once per rate model, so only the forecast differs | a separate twin for the RNN |
| Forecasts use only what is known at 23:00, rolled forward 24 hours | the paper's one-step forecasts use same-day counts the decision cannot have | the paper's protocol, where the RNN looks better (5.8 lost a day against 8.8) |
| Compare on lost riders, with the paired difference and its error | the forecast improved and the decision did not | reporting forecast error alone |

## 8. AI in the twin: a language model

| Decision | Why | Given up, and why |
|---|---|---|
| Open-weight models served locally, called through the OpenAI chat format with the standard library | any model or server can be swapped in; no framework to learn or pin | a hosted API; an agent framework |
| The agent uses only the Twin interface (`brief`, `whatif`), so it lives in the skeleton | one agent for every twin | an agent written for bike share |
| First design: the model chooses, running the what-if as a tool (`ask`) | the usual agent design | kept for comparison; it did not beat the rule, and the model tried one to three options, often only its own answer |
| Second design: the model extracts the event's conditions and the twin compares every action (`advise`) | reading was right in 106 of 109 nights; choosing was not, and code compares 31 options instantly | asking the model to compare better: one added sentence helped unevenly |
| Score a choice by the twin's own what-if under reference conditions, and report "as predicted" next to the replay | in the replay, any fixed number of bikes beat the rule in the low season, so a replay win does not show reasoning | the replay alone |
| Three calls (`conditions`, `no_effect`, `cannot`) written in the prompt, replied as JSON | usable readings went from 3 of 10 to 27 of 30 for the smaller model | a free-form JSON reply |
| Each twin declares its conditions as data; the prompt is generated | a new twin writes no prompt, and the declaration is tested against `best()` | a prompt per twin |
| Tool calls are optional and off by default | no better at reading, worse at `cannot` | tool calls as the only path |
| Development events and held-out test events, a perfect reader as the ceiling, a twin that is not told as the floor | the prompt was shaped on the development events; without a ceiling and a floor a score has no scale | one set of events |
| A reply the twin cannot run counts as no action, and the twin keeps its usual choice | averaging only over readable replies flatters the model | dropping unreadable replies |
| Cap each reply's tokens, count failures as turns, save per model, run each setup on two machines | a reply once ran past 30,000 tokens; an unreadable one once lost 1.7 hours of results; two machines disagreed on about two thirds of choices | trusting one run |

## 9. Adapting: forgetting old data

| Decision | Why | Given up, and why |
|---|---|---|
| Compare windows (7 to 120 days), exponential discounting and all history on the same replay | the question was whether the window length matters | adopting the paper's 30 days unexamined |
| Choose on May–October, report on November–December | choosing and reporting on the same months is tuning on the test set | choosing on the test months |
| Report that the choice period could not separate the models | in the busy months every model lost 75.8 to 76.0 riders a day, so "21 days is best" would be noise | reporting the selected window as optimal |
| Let the twin select among rate models from its own recent forecast errors | no window has to be set by hand (7.54 lost a day, as good as the fixed ones) | a hand-set window; note the selector needs about ten days after a change |

## 10. The repository

| Decision | Why | Given up, and why |
|---|---|---|
| Every result a README states is asserted by a test | a reader can rerun it, and a refactor cannot change it silently | a check script per claim |
| One place for each fact (`conventions.md` says where) | duplicated statements drift apart | repeating numbers across documents |
| Comparisons with authors' code that no longer installs were made once and recorded with their script | they cannot run in CI | dropping the comparison |
