# Tools and steps for building a digital twin

For someone about to build their first twin, or a twin of a new system. It says what the typical code looks like,
which tools each part of a twin needs, what this repository uses and what the field commonly uses instead, and the
order to build in. The steps themselves, with a check after each, are in [adding-a-twin.md](adding-a-twin.md); the
reasons behind the choices here are in [design-decisions.md](design-decisions.md).

## The short version

- A digital twin is one loop: observe the asset, update the twin's state, predict, decide, act. The loop is about
  25 lines of plain Python (`dtcore/runtime.py`); everything else belongs to the system being twinned.
- Research twins of one asset need Python and a numerical library. The platforms common in the literature (Eclipse
  Ditto, Kafka, InfluxDB, Grafana) solve a different problem: many devices, many users, live data.
- Build one element at a time and check each against a paper, the authors' code or real data before adding the next.
  The model comes first and takes most of the time.
- Adding AI is a change to one part of the twin (its forecast, or how it gets outside information). Compare it with
  the part it replaces on the decision, and not only on forecast accuracy.

## The typical code: one loop

```python
def run(source, twin, max_steps=None, recorder=None, before_step=None):
    k = 0
    while max_steps is None or k < max_steps:
        obs = source.observe()                 # physical -> digital: the next observation, or None at the end
        if obs is None:
            break
        est = twin.update(obs)                 # the twin's state follows the asset
        action = twin.decide() if "decide" in twin.capabilities else None
        if action is not None:
            source.act(action)                 # digital -> physical: the decision changes the asset
        if recorder is not None:
            recorder.add(k, obs, est, action)
        source.advance()
        k += 1
    return k
```

Two classes fill it in. A `Source` is the physical side: a sensor feed, a replay of recorded data, or a simulator. A
`Twin` is the digital side: `update(obs)`, `predict(horizon, **scenario)` (a non-empty scenario makes it a what-if)
and, where the twin decides by itself, `decide()`. The four twins here, an incubator, a battery cell, a UAV wing and
a bike-share station, all run on this loop; what differs between them is the model and the update method.

Each twin is described first with six elements (Kapteyn, Pretorius & Willcox 2021; see
[architecture.md](architecture.md)): the physical state S, the observations O, the digital state D, the quantities of
interest Q, the actions U and the reward R. Filling that table for a new system is step 1, and it shows early what
is missing: a twin with no U is a model, and one with no O cannot be updated.

## Tools by part

| Part of a twin | Used here | Common in the field | Why this choice |
|---|---|---|---|
| Physical side, data | replays of recorded data (incubator, battery, bike share); the authors' simulated asset (UAV) | IoT devices, public datasets | a replay is repeatable, and real data show problems a simulator hides |
| Data transport | reading files; MQTT only in the battery's live demo (paho-mqtt, amqtt) | MQTT, AMQP (Eclipse Hono), Kafka [1][2] | amqtt installs with pip; nothing else needs a broker |
| State management | one Python object per twin | Eclipse Ditto [1][5]; commercial platforms (Predix, MindSphere, ThingWorx) [5] | one asset on one machine; Ditto is built for fleets of devices |
| Storage | `Recorder`: memory, optionally SQLite | InfluxDB, MongoDB [1][5]; MySQL [5] | SQLite ships with Python |
| Model, simulation | numpy and scipy: a linear thermal model, a power-law fade, the authors' finite-element table, a queueing model with a matrix exponential | Python in 12 of 22 surveyed twins, C# 5, MATLAB 3 [2]; Simulink, Twin Builder [5]; FMI/FMU for model exchange [3]; discrete-event simulation (SimPy) for processes with people [4] | reproducing papers means matching the authors' model |
| Update (physical to digital) | Kalman filter, particle filter, discrete Bayes filter, Gamma–Poisson update | state estimation and calibration, mostly hand-written or from numerical libraries | the paper's method for each system; counts of rentals are rates, so a conjugate update fits |
| AI: forecast | PyTorch, loading the authors' trained network | TensorFlow in 6 of 22, PyTorch in 4 [2] | the authors' model is in PyTorch |
| AI: language model | open-weight models served locally by Ollama (Qwen3.5-9B, gpt-oss-20b); `dtcore/agent.py` uses only the standard library and the OpenAI chat format | native tool calling [6][7]; structured output constrained by a JSON schema [8]; MCP to expose tools to a model [9]; frameworks such as LangChain [10] | the model can be swapped and runs on a laptop; see "The language model" below |
| Display | one web page: three.js for 3D, canvas charts; Streamlit for the battery's live demo | Grafana with Unity WebGL [1] | opens without a server and the what-if can be dragged live; the cost is a second copy of each model in JavaScript, checked against Python |
| Third-party code and data | `fetch_upstream.sh`, pinned to a commit, never committed | | fixed versions make results repeatable; some upstreams have no license or forbid redistribution |
| Trust | pytest (one test per result a README states), `run_checks.sh`, GitHub Actions | | someone else can rerun every number |
| Compute | a laptop; a cluster GPU (Slurm) for the agent experiments | Docker, Kubernetes, Helm [1] | research code for one asset needs no orchestration |

Two kinds of toolchain appear in the literature. A platform, such as OpenTwins (Ditto, Hono, Kafka, InfluxDB,
Grafana, Docker and Kubernetes: about ten services), serves many devices and users [1]. A research twin is Python
with numerical and machine-learning libraries; this repository is of that kind. The kind of simulation follows the
system: differential equations for continuous physics, queueing or discrete-event models for processes with people.

## The steps

1. **Six elements.** Fill the S / O / D / Q / U / R table.
2. **Model only.** Write the model and run it alone against the paper or the data.
3. **Source.** A replay of recorded data, or a simulator, that returns one observation at a time.
4. **Update.** The twin follows the asset; check the estimate against what the data or the simulator knows.
5. **Predict and what-if.** Forecast the quantities of interest; a scenario makes it a what-if.
6. **Decide and act.** A rule that chooses the action, and a source that applies it.
7. **Page.** The asset and its twin side by side.

Start from `twins/_template` (a mug with a heater, the smallest complete twin). Details, file names and the checklist
for each step are in [adding-a-twin.md](adding-a-twin.md).

What took the time in the four twins here, as a guide to what to expect:

- **The model, and matching the reference.** Small differences have causes worth finding: the one bike-share decision
  in 366 that differs from the authors' code is a numerical tie, and the largest model difference is the authors'
  integration error.
- **Old code on new libraries.** Code from 2020 needed one change to run on current pandas; the UAV authors'
  dependency no longer installs, so that comparison was made once and its result recorded.
- **Data that cannot be redistributed.** Check the license before designing a public page around a dataset.
- **Parameters the data cannot identify.** With only the incubator's air temperature measured, the heater's
  parameters can take a wide range of values with the same fit.

## Adding AI to a twin

Two ways were tried here, both on parts that already existed, so each could be compared with what it replaced.

**A learned forecast.** In the bike-share twin the rate model is a parameter: the statistical model (Gamma–Poisson)
or the paper's trained RNN. Compare on the decision, with only the data known when the decision is made. Here the
RNN's forecast was more accurate (RMSE 4.7 against 6.1 rentals an hour) and its decisions were not better (8.8
against 7.8 lost riders a day, within noise).

**A language model for outside information.** A twin only sees its own sensors. An event told in words (a subway
closure, a heat wave) can reach it through a language model, and the question is how to divide the work.

## The language model: what worked

- **The model reads; the twin computes.** Asked to choose an action itself, a small model confirmed the answer it
  already had and rarely compared options. Asked only to turn an event into the what-if's inputs, with the twin then
  comparing every action, the choices were near the twin's best. Published systems draw the same line [13][14].
- **Three possible answers, stated in the prompt.** `conditions` (the inputs the event changes), `no_effect` and
  `cannot` (the event matters but no input expresses it). With a loosely described reply format, Qwen3.5-9B's readings
  were usable in 3 of 10 cases; with the three calls, in 27 of 30 held-out cases, and gpt-oss-20b in 30 of 30.
- **Each twin declares its conditions as data** (`Twin.conditions`: name, type, range, one line of meaning). The
  prompt is generated from it, so a new twin writes no prompt.
- **Native tool calling was not better** for these models: equal at reading numbers, worse at saying `cannot` (8 and
  14 of 20, against 14 and 15 without). JSON-schema-constrained output was not available for Qwen3.5-9B on Ollama 0.40.
- **Engineering comes before intelligence.** Cap the tokens of each reply, count a failed or empty reply as a turn,
  save results model by model, and turn thinking off or low for small models.
- **One run is not a result.** The same script at temperature 0 gave different choices on a laptop and on a cluster
  GPU in about two thirds of cases. Report what holds on both.

What is not solved: an event with no numbers. The model must guess one, and the twin then acts on the guess. For the
incubator, both models gave the room an outdoor temperature, and the twin's choice was worse than if it had not been
told (30 to 54 minutes of 60 outside the band, against 8). Have a person confirm such a number, or ask for a range.

How to run these experiments, and the numbers behind each statement: `tools/advise_events.py` and the bike-share
README.

## What this repository does not have

- Message transport, except in the battery's live demo.
- A standard format for exchanging models, such as FMI.
- A continuous check that a twin has drifted from its asset. The incubator flags an opened lid; nothing detects a
  slow change such as the bike station's change of season.

## Limits of the evidence

The counts in the table come from one survey of 22 twins [2]; [1] is a single framework and [5] is a review from
2020. Eclipse Ditto, MongoDB and MQTT are named by at least two of the sources. The statements about language models
come from two small local models, four twins and events written for this repository.

## Sources

1. OpenTwins: an open-source framework for the design, development and integration of effective
   3D-IoT-AI-powered digital twins. arXiv 2301.05560.
2. Liu & David, AI simulation by digital twins: systematic survey, reference framework, and mapping to a standardized
   architecture. Software and Systems Modeling, 2025. arXiv 2506.06580.
3. FMPy. https://pypi.org/project/FMPy/
4. SimPy, discrete-event simulation in Python. https://simpy.readthedocs.io/
5. Fuller et al., Digital twin: enabling technologies, challenges and open research. IEEE Access, 2020.
   arXiv 1911.01276.
6. OpenAI, Function calling. https://developers.openai.com/api/docs/guides/function-calling
7. Ollama, Tool calling. https://docs.ollama.com/capabilities/tool-calling
8. Ollama, Structured outputs. https://docs.ollama.com/capabilities/structured-outputs
9. Model Context Protocol. https://modelcontextprotocol.io/docs/getting-started/intro
10. LangChain. https://www.langchain.com/
11. Kapteyn, Pretorius & Willcox, A probabilistic graphical model foundation for enabling predictive digital twins at
    scale. Nature Computational Science, 2021. arXiv 2012.05841.
12. Gammelli et al., Predictive and prescriptive performance of bike-sharing demand forecasts for inventory
    management. Transportation Research Part C, 2022.
13. Kleiman et al., Simulation Agent: a framework for integrating simulation and large language models for enhanced
    decision-making. arXiv 2505.13761.
14. Xia et al., LLM experiments with simulation: large language model multi-agent system for simulation model
    parametrization in digital twins. arXiv 2405.18092.
