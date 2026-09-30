#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
MAX_USED_MIB=34000

cd "$ROOT"

while true; do
    used_mib="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$used_mib" -le "$MAX_USED_MIB" ]]; then
        break
    fi
    printf '%s waiting for GPU 0 memory: %s MiB used, target <= %s MiB\n' \
        "$(date '+%F %T')" "$used_mib" "$MAX_USED_MIB"
    sleep 60
done

printf '%s restarting Polyp 1%% GT CLIP-only on physical GPU 0\n' "$(date '+%F %T')"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u run_polyp_manifest.py \
    --labeled-ratio 1.0e-2 \
    --prompt-backend clip-only \
    --gpu 0 \
    --output-dir output_current/Polyp/1pct_clip_only_retry2 \
    --epochs 10 \
    --batch-size 4 \
    --val-batch-size 16

printf '%s starting ISIC IFP Teacher-Student with exactly 10 GT masks on physical GPU 0\n' "$(date '+%F %T')"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u run_semisup_manifest.py \
    --dataset isic \
    --labeled-count 10 \
    --gpu 0 \
    --output-dir output_current/isic/10gt_ifp_teacher_student_retry2 \
    --epochs 10 \
    --alignment-epochs 1 \
    --batch-size 4 \
    --val-batch-size 16
