#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
LAUNCHER="${1:-1574600}"
LOG="$ROOT/output_current/fair_budget_matched_20260919/logs/watchdog_gpu1.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

descendants() {
  local root="$1" frontier child
  frontier=("$root")
  while ((${#frontier[@]})); do
    local next=()
    for child in "${frontier[@]}"; do
      while IFS= read -r c; do
        [[ -n "$c" ]] || continue
        printf '%s\n' "$c"
        next+=("$c")
      done < <(pgrep -P "$child" 2>/dev/null || true)
    done
    frontier=("${next[@]}")
  done
}

log "watchdog start launcher=$LAUNCHER pid=$$"
while kill -0 "$LAUNCHER" 2>/dev/null; do
  mapfile -t pids < <(printf '%s\n' "$LAUNCHER"; descendants "$LAUNCHER")
  for pid in "${pids[@]}"; do
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    [[ -n "$state" ]] || continue
    if [[ "$state" == *T* ]]; then
      kill -CONT "$pid" 2>/dev/null || true
      log "CONT pid=$pid state=$state"
    fi
  done
  sleep 2
done
log "launcher exited; watchdog stop"
