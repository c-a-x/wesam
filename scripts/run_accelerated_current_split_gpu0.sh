#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
RUN_ROOT="$ROOT/output_current/ablation_accelerated_current_split_20260909"
INDEX_ROOT="$RUN_ROOT/indices"
STATUS="$RUN_ROOT/gpu0_status.log"

mkdir -p "$INDEX_ROOT"
cd "$ROOT"

"$PYTHON" - "$INDEX_ROOT/isic_10pct.json" <<'PY'
import csv
import json
import sys

source = "output_current/ablation_1shot_10pct_20260908/no_semisupervised/ISIC/10pct/isic_ifp_supervised_259shot/lists/train.csv"
with open(source, newline="") as handle:
    json.dump([int(row["index"]) for row in csv.DictReader(handle)], open(sys.argv[1], "w"))
PY

run_task() {
    local name="$1"
    shift
    printf '%s START %s\n' "$(date '+%F %T')" "$name" | tee -a "$STATUS"
    if env PYTHONUNBUFFERED=1 "$PYTHON" -u "$@" >"$RUN_ROOT/${name}.log" 2>&1; then
        printf '%s DONE %s\n' "$(date '+%F %T')" "$name" | tee -a "$STATUS"
    else
        code=$?
        printf '%s FAILED(%s) %s\n' "$(date '+%F %T')" "$code" "$name" | tee -a "$STATUS"
    fi
}

run_task isic_seed3407 run_semisup_manifest.py \
    --dataset isic --gpu 0 --labeled-count 259 \
    --output-dir "$RUN_ROOT/isic_seed3407" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed 3407 --prompt-backend ifp \
    --labeled-prompt-mode gt --labeled-batch-probability 0.7 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --train-prompt-dropout 0 \
    --labeled-indices-file "$INDEX_ROOT/isic_10pct.json" --support-indices-file "$INDEX_ROOT/isic_10pct.json"
