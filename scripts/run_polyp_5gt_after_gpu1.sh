#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUTPUT="output_current/Polyp/5gt_ifp_teacher_student"

cd "$ROOT"

wait_for_process() {
    local pattern="$1"
    local label="$2"
    while pgrep -f "$pattern" >/dev/null; do
        printf '%s waiting for %s\n' "$(date '+%F %T')" "$label"
        sleep 60
    done
}

wait_for_process 'run_polyp_manifest.py.*1pct_no_prompt_teacher_student' 'Polyp no-prompt completion'
wait_for_process 'run_polyp_manifest.py.*full_gt_ifp_prompt' 'Polyp Full-GT + IFP completion'

printf '%s starting Polyp IFP Teacher-Student with exactly 5 GT masks on physical GPU 1\n' "$(date '+%F %T')"
exec env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" -u run_polyp_manifest.py \
    --labeled-count 5 \
    --prompt-backend ifp \
    --labeled-prompt-mode gt \
    --gpu 1 \
    --output-dir "$OUTPUT" \
    --epochs 10 \
    --alignment-epochs 1 \
    --batch-size 4 \
    --val-batch-size 16
