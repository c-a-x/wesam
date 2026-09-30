#!/usr/bin/env bash
set -uo pipefail

PGID=3642720
LOG="/datanas01/nas01/Student-home/2024U/YBC/wesam2/output_current/ablation_1shot_10pct_20260908/polyp_no_semisupervised_export_watchdog.log"

while kill -0 "$PGID" 2>/dev/null; do
    stopped="$(ps -o stat= -g "$PGID" 2>/dev/null | awk '$1 ~ /^T/ {count++} END {print count+0}')"
    if [[ "$stopped" -gt 0 ]]; then
        printf '%s CONT pgid=%s stopped=%s\n' "$(date '+%F %T')" "$PGID" "$stopped" >> "$LOG"
        kill -CONT -- "-$PGID" 2>/dev/null || true
    fi
    sleep 15
done

printf '%s EXPORT_QUEUE_EXITED\n' "$(date '+%F %T')" >> "$LOG"
