#!/usr/bin/env bash
# Run every check: dtcore unit tests, each twin on dtcore, and the web page against Python.
# Prints one line per check and exits non-zero if any is not OK. CI runs exactly this.
#
#   ./run_checks.sh            needs: pip install -e ".[twins,dev]"  and  twins/*/fetch_upstream.sh
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
check "TEMPLATE OK"         $PY twins/_template/scripts/check_dtcore.py
check "DTCORE OK"           $PY twins/incubator/scripts/check_dtcore.py
check "REPRO OK"            $PY twins/battery/scripts/check_repro.py
check "BATTERY DTCORE OK"   $PY twins/battery/scripts/check_dtcore.py 27 4 24
check "UAV DTCORE OK"       $PY twins/uav/scripts/check_dtcore.py

if command -v node > /dev/null; then
  # the incubator replay reference is built locally from the (unlicensed, not committed) upstream data
  [ -f twins/incubator/results/ref_replay.json ] && [ -f web/replay_2023.js ] \
    || $PY twins/incubator/scripts/export_reference.py --replay-only > /dev/null
  check "WEB CHECK OK"         node web/checks/check_incubator.js twins/incubator/results web
  check "BATTERY WEB CHECK OK" node web/checks/check_battery.js twins/battery/results web
  check "UAV WEB CHECK OK"     node web/checks/check_uav.js twins/uav/results web
else
  echo "skip  web checks (node not installed)"
fi

[ $fails -eq 0 ] && echo "ALL CHECKS OK" || { echo "$fails check(s) failed"; exit 1; }
