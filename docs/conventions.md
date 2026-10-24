# File conventions

Where every kind of file goes. New twins, experiments and files follow this; anything that does not, gets moved.
The goal: someone opening the repository finds things without asking, can run them, and each fact is written in one place.

## 1. Top level

```
README.md  LICENSE  THIRD_PARTY_NOTICES.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md  CHANGELOG.md  CITATION.cff
pyproject.toml     packages dtcore and twins; optional dependencies per use: twins, rnn, live, dev
run_checks.sh      every check, the same list CI runs
dtcore/            the skeleton; no domain concepts
twins/<name>/      one folder per twin
web/               the shared page
docs/              architecture, adding a twin, these conventions
tests/             tests for dtcore and each twin (pytest)
tools/             helper scripts shared by twins
.github/           CI, Pages deployment, issue and PR templates
```

No other top-level folders.

## 2. A twin folder

```
twins/<name>/            the Python package twins.<name>
  __init__.py            the public names: the Twin and Source classes and anything a caller must pass to them
                         (a reproduction also exports its paper experiment); code outside the twin imports only
                         these, while the twin's own scripts and tests/test_<name>.py may use any of its modules
  twin.py                the Twin class, <Asset>Twin
  sources.py             the Source classes
  *.py                   the rest of the twin (model, estimator, decision), named freely
  README.md              fixed structure (section 4)
  scripts/               runnable entry points: exports, result generation (and battery's check_live.py, which
                         needs the live extra and a broker, so it is run by hand, not in CI)
  results/               output of scripts/; regenerable
  experiments/           one-off studies, one subfolder each (section 3)
  fetch_upstream.sh      downloads third-party code and data into upstream/ (git-ignored)
  upstream/<repo-name>/  what fetch_upstream.sh downloaded; never edited, never committed
  live/  hpg/            optional: real-time transport layer, cluster job scripts
  run_demo.sh            optional: one-command demo
```

Create only the folders you use; delete empty ones.

| What | Where | Name |
|---|---|---|
| Twin folder | `twins/<name>/` | the physical asset or domain, lower case, no author or paper name: `incubator`, `bikeshare` |
| Package | `twins/<name>/` | imported as `twins.<name>`, never through `sys.path` |
| Twin / Sources | `twin.py` / `sources.py` | `<Asset>Twin`; Sources named for what they are (`SimMug`, `CellReplay`) |
| Tests | `tests/test_<name>.py` | what the twin must do and the published numbers it reproduces |
| Page export | `scripts/export_web.py` | writes `web/<name>_*.js` and `results/ref_web_<name>.json`; with `--replay-only`, only the git-ignored files from recordings that may not be committed (`run_checks.sh` builds them that way) |
| Comparison of a twin's parts | `scripts/compare_<part>.py` | runs the same twin and Source once per implementation of one part (section "Parts of a twin"); writes `results/compare_<part>.json` |
| Comparison with upstream | `scripts/crosscheck_upstream.py` | runs the authors' code on our inputs; kept when a test compares against its saved answers, so `results/` stays regenerable; a comparison made once and only reported goes in the README |
| Results | `results/` | say what it is, no dates: `baseline.csv`, `step1_decisions.json`; dates go on experiment folders |
| Runtime databases | `results/*.db` by default, `DT_DB` to move | not results; git-ignored |

Scripts find their files from their own location (`Path(__file__)`), so they run from any directory.
Commands in READMEs are written from the repository root, except inside experiments.

### Parts of a twin

A twin's estimator, forecaster or policy is a part (`docs/architecture.md`). When a part gets a second implementation
(a learned one, typically), it becomes a constructor argument: `StationTwin(C, rates=RNNRates())`.

- The default is the twin's own physical or statistical model: the baseline every other implementation is compared with.
- What the twin asks of the part is its interface, written where the part is defined (bike share: `rates.py`). No
  shared base class: parts of different twins do different jobs.
- A policy, `decide(twin) -> action`, is asked only when the twin decides: the twin keeps when and whether, the policy
  chooses what. The LLM agent (`dtcore.agent`) works with any twin that writes `brief()` and `whatif(**scenario)`; to
  make it a policy, the twin wraps it with its request and choices (bike share: `AgentPolicy` in `twin.py`).
- Text sent to a language model is a constant named `<PURPOSE>_PROMPT` (as LlamaIndex names its prompts). The twin's
  own prompts are in `twin.py`, next to `brief`, and scripts and the page use them as they are (bike share:
  `NIGHTLY_PROMPT`; `EXTRACT_EVENT_PROMPT`); wording only one experiment adds stays in its script (`CHECK_PROMPT` in
  `new_conditions.py`); the agent's own system message is `dtcore.agent.SYSTEM_PROMPT`.
- Getting a twin's what-if inputs out of words is called extraction (LangChain's term: structured data from
  unstructured text): the model extracts the inputs, the twin computes every choice (bike share:
  `EXTRACT_EVENT_PROMPT`). Names use extract / extraction / inputs, not translate.
- A part gets data only through the twin's `update`, in time order, so it cannot use what the twin has not yet seen.
  Anything else it reads is an argument too, with its assumption stated (bike share: the weather, a perfect forecast).
- A part with one implementation stays inside the twin, unless something else must use the same object (uav: the
  simulated asset runs the twin's `UAVModel`); the table in `docs/architecture.md` says where each part is.
- The implementations are compared by `scripts/compare_<part>.py`: same twin, same Source, only the part changes. The
  page lets a person choose the part where its module offers one (`web/README.md`, `renderParts`).
- A dependency that only one part needs (PyTorch for the VP-RNN) is its own extra in `pyproject.toml`, imported only
  where that part is built; its tests skip with the reason when it is missing, and CI installs it
  (`.github/workflows/checks.yml`).

### Upstream

- Third-party code and data are fetched, never committed: `fetch_upstream.sh` calls `tools/fetch_github.sh` with a
  full commit hash, into `upstream/<repository name>/`.
- Never edit upstream files. To change something, copy it into the package, change it there, and say where it came from.
- Every source is listed with its license in `THIRD_PARTY_NOTICES.md` and in the twin's README.
- Data without a license are used locally only: anything that reproduces them is git-ignored and kept off the page.

## 3. experiments/

- One folder per study: `experiments/YYYY-MM-DD_<short-name>/`, dated by its start, e.g. `2026-10-02_noros_repro`.
- The folder holds the study's own scripts, outputs and a `README.md` (question, how to run, result, conclusion).
- Frozen when done: no new studies inside it, no edits to its results. Continuing means a new folder.
- Code that others start to depend on (a check, the package) moves out of experiments into the package or `scripts/`.

## 4. README structure

Twin READMEs use these sections in this order, skipping those that do not apply. Only what is specific to the twin;
the skeleton is explained once in `docs/architecture.md`.

```
# <Name> twin
One or two sentences: what it does, reproduced from whom or built from what description.

## Sources       papers, upstream repositories @ commit, licenses, where the data come from
## Six elements  S O D Q U R table
### Observations and actions
                 every field a Source returns: meaning and unit (including what `t` counts), whether the real
                 asset has it, whether `update` reads it; what an action changes
## Files         one line per folder or key file
## Run           commands from the repository root, with rough run times, including the twin's test and page check
## Results       numbers, each with a source: a file in results/, a test (marked *(test)*), a named frozen study, or
                 a comparison made once, with where its script is in a tagged version
## Notes         data problems, differences from the original, pitfalls (numbers sourced the same way)
```

English, plain sentences. A number goes in one place: the twin README; other documents link to it.

## 5. web/

```
web/page.html                the page (hand-written; the only page source)
web/index.html               built by build.py; git-ignored
web/<name>_core.js           the twin's computations, ported line by line from Python; hand-written
web/<name>_*.js              data written by export_web.py (battery_cells.js, uav_model.js); committed, not hand-edited
web/<name>_replay.js         recorded data that may not be published; written by export_web.py, git-ignored
                             (export_web.py --replay-only writes only these, so the checks can build them)
web/checks/check_<name>.js   compares the page's JS with the Python reference
```

Generated files are never edited by hand: change the source and regenerate.
A request the page sends to a language model is exported from Python with the twin's data (bike share:
`twin.EXTRACT_EVENT_PROMPT` in `bikeshare_events.js`), never written in JavaScript. The page sends it only when opened from this
computer (`http://127.0.0.1`), to a model server on the same computer; the published page shows recorded runs only.
The Pages workflow publishes `index.html`, `*_core.js` and the committed data files only.

## 6. Never in the repository

- `.DS_Store`, `__pycache__/`, `*.db-journal`, `tmp_*` outputs, virtual environments.
- Two copies of the same thing (a zip next to its unpacked folder).
- Empty folders.
- `_v2`, `_new`, `_final`, `_old` in file names: delete the old version, or keep it in a dated experiment folder.
- Personal or institutional details (accounts, group names, private links).
