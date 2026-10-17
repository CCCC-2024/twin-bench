# Contributing to Twin Bench

Thanks for helping. Twin Bench grows by adding twins from new domains on a skeleton that stays small, so most
contributions are one of: a new twin, an improvement to an existing twin, a fix to `dtcore`, or the page.

## Ground rules

- **Every claim is a check.** If a change says "same as the paper" or "same as before", add or update a script that
  prints `OK` / `MISMATCH`, and list it in `run_checks.sh`.
- **The skeleton stays small.** Something goes into `dtcore` only when every twin needs it. Changes to `dtcore` need a
  test in `tests/` and must keep all twin checks passing.
- **No third-party code or data in git.** Fetch it at a pinned commit (`fetch_upstream.sh`) and record its license in
  `THIRD_PARTY_NOTICES.md`. Data without a license stay local.
- **Files go where [docs/conventions.md](docs/conventions.md) says.**

## Workflow

1. Open an issue first for anything larger than a small fix (use *New twin* for a new domain), so the scope is agreed
   before the work.
2. Fork, create a branch (`twin/<name>`, `fix/<what>`, `docs/<what>`).
3. Set up and run the checks:
   ```bash
   pip install -e ".[twins,dev]"
   for t in incubator battery uav; do twins/$t/fetch_upstream.sh; done
   ./run_checks.sh                    # must end with ALL CHECKS OK
   ```
4. Open a pull request using the template. CI runs `run_checks.sh`; it must pass.
5. Add a line under *Unreleased* in `CHANGELOG.md`.

## Adding a twin

Follow [docs/adding-a-twin.md](docs/adding-a-twin.md). Start from `twins/_template`, describe the six elements, and add
one element per step with a check for each. Reproductions should name the paper and the upstream commit and show
that the authors' numbers come out.

## Style

- Python 3.10+, standard library + numpy/scipy/pandas where possible; 120-character lines; docstrings and comments in
  English, saying *why* more than *what*.
- Scripts locate files from `Path(__file__)`, so they run from any directory.
- The page's JavaScript is a line-by-line port of the Python, which stays the reference.

## Releases

Versions follow [Semantic Versioning](https://semver.org/). A release bumps `version` in `pyproject.toml` and
`dtcore.__version__`, moves *Unreleased* in `CHANGELOG.md` under the new version, and is tagged `vX.Y.Z`.

## Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
