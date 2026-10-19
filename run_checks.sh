#!/usr/bin/env bash
# Run every check: the tests in tests/ (dtcore and each twin) and the web page against Python.
# Prints one line per check and exits non-zero if any is not OK. CI runs exactly this.
#
#   ./run_checks.sh            needs: pip install -e ".[twins,rnn,dev]"  and  twins/*/fetch_upstream.sh
#   PYTHON=... ./run_checks.sh to choose the interpreter
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export DT_DB=${DT_DB:-$(mktemp -d)/dtcore_runs.db}   # SQLite on a local disk (network mounts break locking)
export MPLBACKEND=Agg
fails=0

check() {   # check <text the last line must contain> <command...>
  local want=$1; shift
  local last
  last=$("$@" 2>&1 | tail -1)
  if [[ $last == *"$want"* ]]; then echo "ok    $*"; else echo "FAIL  $*  ->  $last"; fails=$((fails + 1)); fi
}

check "passed"              $PY -m pytest -q

if command -v node > /dev/null; then
  # page data from recordings that may not be committed are built here from the fetched upstream data:
  # every export_web.py that takes --replay-only writes only those git-ignored files
  for e in twins/*/scripts/export_web.py; do
    grep -q -- "--replay-only" "$e" && $PY "$e" --replay-only > /dev/null
  done
  for c in web/checks/check_*.js; do        # check_<name>.js reads twins/<name>/results
    name=$(basename "$c" .js); name=${name#check_}
    check "WEB CHECK OK" node "$c" "twins/$name/results" web
  done
else
  echo "skip  web checks (node not installed)"
fi

[ $fails -eq 0 ] && echo "ALL CHECKS OK" || { echo "$fails check(s) failed"; exit 1; }
