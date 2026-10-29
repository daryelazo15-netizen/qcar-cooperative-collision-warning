#!/bin/bash
# Run every unit-test suite of the repository and print one line per suite (the unittest summary, unedited).
#   scripts/run_tests.sh [LOG_DIR]
# Full output of each suite goes to LOG_DIR (default: a new temporary folder). Run from anywhere. Needs shapely; the viewer tests also need numpy and opencv-python
# (they skip with a message when these are missing). No test needs data outside this repository or a vehicle.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-$(mktemp -d)}"; mkdir -p "$LOG"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT" || exit 2
status=0
run() {  # name, start dir
  python3 -B -W ignore::ResourceWarning -m unittest discover -s "$2" > "$LOG/$1.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && status=1
  printf '%-12s exit %d | %s | %s\n' "$1" "$rc" "$(grep -E '^Ran [0-9]+ tests?' "$LOG/$1.log" | tail -1)" "$(grep -E '^(OK|FAILED)' "$LOG/$1.log" | tail -1)"
}
run tracking  tracking/tests
run collision collision/tests
run perception perception/tests
run live      live/tests
find "$ROOT" -name __pycache__ -type d -not -path "*/.git/*" -prune -exec rm -rf {} + 2>/dev/null
echo "logs: $LOG"
exit $status
