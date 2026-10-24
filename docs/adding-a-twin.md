# Adding a twin

A new twin is a new folder under `twins/`. Build it one element at a time: after every step something runs and a
check passes, so when something breaks it can only be the step you just added.

## 0. Before writing code

Open an issue with the **New twin** template. A good candidate:

- can be described in a paragraph (the physical asset, what is measured, what decision is made);
- has public data or a simulator whose license allows use, or data you can collect;
- has a decision to make, by the twin or by a person using its what-if (a rule of this repository: without one it is
  a model, which can live here as a reproduction, like `twins/tumor`);
- adds something the existing twins do not (a new domain, a new kind of update or decision).

## 1. Copy the template

```bash
cp -r twins/_template twins/<name>          # <name>: the physical asset, lower case, e.g. "pump", "greenhouse"
cp tests/test_template.py tests/test_<name>.py
```

The folder is the Python package `twins.<name>`: keep `__init__.py` (the public names), `twin.py` (the Twin) and
`sources.py` (the Sources); name the rest freely. Change the import in `tests/test_<name>.py` to `twins.<name>`, and
rewrite the README following [conventions.md](conventions.md#4-readme-structure).

## 2. The seven steps

1. **Six elements.** Fill the S / O / D / Q / U / R table in the README.
2. **Model only.** Write the model and run it on its own against the paper or the data
   (`scripts/step1_model.py` → `results/`). Reproducing a published twin: compare with the authors' numbers and code
   in a test (section 3); when their code is too slow or hard to run every time, save its answers with
   `scripts/crosscheck_upstream.py` and test against those (see `twins/bikeshare`).
3. **Observations.** Write the `Source`: a replay of recorded data, a simulator, or a live feed.
4. **Update.** `Twin.update(obs)` — the estimator.
5. **Predict, then what-if.** `Twin.predict(horizon, **scenario)`.
6. **Decide and act.** `Twin.decide()`, and `Source.act()` applying it.
7. **Page.** `scripts/export_web.py`, `web/<name>_core.js`, `web/checks/check_<name>.js`, and a module in
   `web/page.html`. Python is the reference; the JavaScript is checked against it.

Steps 4–6 write the twin's parts: its estimator, forecaster and policy. Keep each one inside the twin until a second
implementation of it arrives, often a learned one. Then make that part an argument of the twin, keep the first as the
default and baseline, and compare the two with `scripts/compare_<part>.py`
([conventions](conventions.md#parts-of-a-twin); bike share's `rates.py` and `RNNRates` are the example). For the LLM
agent (`dtcore.agent`) to work with the twin, write `brief()` and `whatif(**scenario)`, and put each prompt sent to a
model next to them, named `<PURPOSE>_PROMPT` (bike share's `twin.py`: `NIGHTLY_PROMPT`, `EXTRACT_EVENT_PROMPT`).

## 3. Tests

`tests/test_<name>.py` runs the twin through `dtcore.run` and asserts what it must do (see `tests/test_template.py`);
a reproduction also asserts the published numbers it reproduces. Compare with results computed in the same run, not
with long float lists stored from another machine: library versions change the last digits. pytest finds the file by
itself; CI runs [run_checks.sh](../run_checks.sh), which runs pytest.

## 4. Third-party code and data

Never commit them. Add `fetch_upstream.sh` (see `twins/battery/fetch_upstream.sh`), pinned to a commit, and list the
source and its license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). If the data have no license, use them
locally only: git-ignore everything derived from them and keep them out of the page.

## 5. Dependencies

Add what the twin needs to the `twins` extra in [pyproject.toml](../pyproject.toml). A heavy dependency that only one
part needs gets its own extra (bike share's VP-RNN: `rnn`, PyTorch); see [conventions](conventions.md#parts-of-a-twin).
If the twin cannot share that environment (the tumor twin needs Python 3.10–3.11 and the authors' TumorTwin
package), document a separate setup in its README and leave it out of `run_checks.sh`.

## Checklist for the pull request

- [ ] README with the six elements, the observations-and-actions table, sources, how to run, results (each number
  sourced), notes
- [ ] `tests/test_<name>.py` passes
- [ ] `fetch_upstream.sh` pinned to commits; licenses in THIRD_PARTY_NOTICES.md
- [ ] new dependencies in `pyproject.toml`
- [ ] row added to the table in the top-level README; entry in CHANGELOG.md
- [ ] its name added to the `fetch_upstream.sh` loops in `.github/workflows/checks.yml`, `CONTRIBUTING.md` and the
  top-level README's Quick start (its tests and page check are found automatically)
