#!/bin/bash
# Wait for a sustained-quiet Mac, record evidence, then run the supplied command.
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$2" != -- ]; then
  echo "usage: $0 LOG -- COMMAND [ARGS...]" >&2
  exit 2
fi
LOG=$1
shift 2
mkdir -p "$(dirname "$LOG")"
quiet=0
{
  echo "quiet gate: $(date -u) host=$(hostname -s) threshold=2.0 reads=6 interval=80s"
  printf 'command:'; printf ' %q' "$@"; printf '\n'
} > "$LOG"
for ((i=1; i<=75; i++)); do
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  case "$load" in ''|*[!0-9.]*) echo "invalid load reading: $load" >&2; exit 1;; esac
  if awk -v value="$load" 'BEGIN {exit !(value < 2.0)}'; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
  echo "$(date -u) load1m=$load consecutive=$quiet" >> "$LOG"
  if [ "$quiet" -ge 6 ]; then
    echo "QUIET_GATE_PASS" >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
      echo "COMMAND_PASS" | tee -a "$LOG"
      exit 0
    else
      rc=$?
      echo "COMMAND_FAILED rc=$rc" | tee -a "$LOG"
      exit "$rc"
    fi
  fi
  sleep 80
done
echo "QUIET_GATE_BLOCKED" | tee -a "$LOG"
exit 1
