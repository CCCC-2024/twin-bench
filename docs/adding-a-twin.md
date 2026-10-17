# Adding a twin

A new twin is a new folder under `twins/`. Build it one element at a time: after every step something runs and a
check passes, so when something breaks it can only be the step you just added.

## 0. Before writing code

Open an issue with the **New twin** template. A good candidate:

- can be described in a paragraph (the physical asset, what is measured, what decision is made);
- has public data or a simulator whose license allows use, or data you can collect;
- has a decision to make (otherwise it is a model, not a twin);
- adds something the existing twins do not (a new domain, a new kind of update or decision).

## 1. Copy the template

```bash
cp -r twins/_template twins/<name>          # <name>: the physical asset, lower case, e.g. "pump", "greenhouse"
mv twins/<name>/mdt twins/<name>/<x>dt      # package: first letter + "dt", e.g. pdt
```

Rename the imports in `scripts/check_dtcore.py`, and rewrite the README following
[conventions.md](conventions.md#4-readme-structure).

## 2. The seven steps

1. **Six elements.** Fill the S / O / D / Q / U / R table in the README.
2. **Model only.** Write the model and run it on its own against the paper or the data
   (`scripts/step1_model.py` → `results/`). Reproducing a published twin: compare with the authors' code
   (`scripts/crosscheck_upstream.py`) and their numbers (`scripts/check_repro.py`).
3. **Observations.** Write the `Source`: a replay of recorded data, a simulator, or a live feed.
4. **Update.** `Twin.update(obs)` — the estimator.
5. **Predict, then what-if.** `Twin.predict(horizon, **scenario)`.
6. **Decide and act.** `Twin.decide()`, and `Source.act()` applying it.
7. **Page.** `scripts/export_web.py`, `web/<name>_core.js`, `web/checks/check_<name>.js`, and a module in
   `web/page.html`. Python is the reference; the JavaScript is checked against it.

## 3. Checks

`scripts/check_dtcore.py` must print `<NAME> DTCORE OK` when running through `dtcore.run` gives exactly the same
states and decisions as calling your functions directly. Run both in the same process, never against a stored file:
library versions can change the last digits (see `twins/uav/scripts/check_dtcore.py`).

Add every check to [run_checks.sh](../run_checks.sh); CI runs that file.

## 4. Third-party code and data

Never commit them. Add `fetch_upstream.sh` (see `twins/battery/fetch_upstream.sh`), pinned to a commit, and list the
source and its license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). If the data have no license, use them
locally only: git-ignore everything derived from them and keep them out of the page.

## 5. Dependencies

Add what the twin needs to the `twins` extra in [pyproject.toml](../pyproject.toml). If it cannot share that
environment (the tumor twin needs PyTorch and Python ≤ 3.11), document a separate setup in its README and leave it
out of `run_checks.sh`.

## Checklist for the pull request

- [ ] README with the six elements, sources, how to run, results, notes
- [ ] `check_dtcore.py` prints OK and is in `run_checks.sh`
- [ ] `fetch_upstream.sh` pinned to commits; licenses in THIRD_PARTY_NOTICES.md
- [ ] new dependencies in `pyproject.toml`
- [ ] row added to the table in the top-level README; entry in CHANGELOG.md
