#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="$ROOT/output_current/isic/1pct"
TEN_PCT_METRICS="$ROOT/output_current/isic/10pct/ISIC/metrics.csv"

mkdir -p "$OUT"
printf '%(%F %T)T waiting for ISIC 10%% to finish\n' -1 > "$OUT/queue.log"

# The bracketed character prevents pgrep from matching this waiting script.
while pgrep -f '[r]un_semisup_manifest.py --dataset isic --labeled-ratio 0.10' >/dev/null; do
    sleep 30
done

if ! rg -q 'isic_0.1gt_seed1337_final_test_best_student' "$TEN_PCT_METRICS"; then
    printf '%(%F %T)T ISIC 10%% exited without final-test metrics; ISIC 1%% was not started\n' -1 >> "$OUT/queue.log"
    exit 1
fi

printf '%(%F %T)T ISIC 10%% completed; starting corrected ISIC 1%%\n' -1 >> "$OUT/queue.log"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u "$ROOT/run_semisup_manifest.py" \
    --dataset isic --labeled-ratio 0.01 --gpu 0 --output-dir "$OUT" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 \
    > "$OUT/train.log" 2>&1
