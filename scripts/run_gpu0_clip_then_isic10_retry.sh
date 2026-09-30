#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"

cd "$ROOT"

printf '%s restarting Polyp 1%% GT CLIP-only on physical GPU 0\n' "$(date '+%F %T')"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u run_polyp_manifest.py \
    --labeled-ratio 1.0e-2 \
    --prompt-backend clip-only \
    --gpu 0 \
    --output-dir output_current/Polyp/1pct_clip_only_retry \
    --epochs 10 \
    --batch-size 4 \
    --val-batch-size 16

printf '%s starting ISIC IFP Teacher-Student with exactly 10 GT masks on physical GPU 0\n' "$(date '+%F %T')"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u run_semisup_manifest.py \
    --dataset isic \
    --labeled-count 10 \
    --gpu 0 \
    --output-dir output_current/isic/10gt_ifp_teacher_student_retry \
    --epochs 10 \
    --alignment-epochs 1 \
    --batch-size 4 \
    --val-batch-size 16
