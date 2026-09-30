#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
RUN_ROOT="$ROOT/output_current/ablation_1shot_10pct_20260908"
STATUS="$RUN_ROOT/polyp_no_semisupervised_export_status.log"
cd "$ROOT"

export_one() {
    local budget="$1"
    local count="$2"
    printf '%s START %s\n' "$(date '+%F %T')" "$budget" | tee -a "$STATUS"
    if env PYTHONUNBUFFERED=1 "$PYTHON" -u run_ifp_supervised_ablation.py \
        --dataset polyp --gpu 1 --labeled-count "$count" \
        --output-dir "$RUN_ROOT/no_semisupervised/Polyp/$budget" \
        --export-only --batch-size 4 --val-batch-size 16 --seed 1337 \
        > "$RUN_ROOT/logs/polyp_${budget}_no_semisupervised_export.log" 2>&1; then
        printf '%s DONE %s\n' "$(date '+%F %T')" "$budget" | tee -a "$STATUS"
    else
        printf '%s FAILED %s\n' "$(date '+%F %T')" "$budget" | tee -a "$STATUS"
    fi
}

export_one 1shot 1
export_one 10pct 130
