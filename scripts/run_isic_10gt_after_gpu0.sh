#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUTPUT="output_current/isic/10gt_ifp_teacher_student"

cd "$ROOT"

while pgrep -f 'run_polyp_1pct_single_encoder_after_fullgt.sh' >/dev/null; do
    printf '%s waiting for GPU 0 Polyp DINO-only / CLIP-only queue\n' "$(date '+%F %T')"
    sleep 60
done

printf '%s starting ISIC IFP Teacher-Student with exactly 10 GT masks on physical GPU 0\n' "$(date '+%F %T')"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u run_semisup_manifest.py \
    --dataset isic \
    --labeled-count 10 \
    --gpu 0 \
    --output-dir "$OUTPUT" \
    --epochs 10 \
    --alignment-epochs 1 \
    --batch-size 4 \
    --val-batch-size 16
