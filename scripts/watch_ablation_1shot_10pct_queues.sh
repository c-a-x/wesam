#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
RUN_ROOT="$ROOT/output_current/ablation_1shot_10pct_20260908"
LOG="$RUN_ROOT/queue_watchdog.log"
GPU0_PGID=3434824
GPU1_PGID=3435239

while kill -0 "$GPU0_PGID" 2>/dev/null || kill -0 "$GPU1_PGID" 2>/dev/null; do
    for pgid in "$GPU0_PGID" "$GPU1_PGID"; do
        if kill -0 "$pgid" 2>/dev/null; then
            stopped="$(ps -o stat= -g "$pgid" 2>/dev/null | awk '$1 ~ /^T/ {count++} END {print count+0}')"
            if [[ "$stopped" -gt 0 ]]; then
                printf '%s CONT pgid=%s stopped=%s\n' "$(date '+%F %T')" "$pgid" "$stopped" >> "$LOG"
                kill -CONT -- "-$pgid" 2>/dev/null || true
            fi
        fi
    done
    sleep 20
done

printf '%s ALL_QUEUES_EXITED\n' "$(date '+%F %T')" >> "$LOG"
