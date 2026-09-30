#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
ISIC_PATTERN="run_semisup_manifest.py --dataset isic.*output_current/isic/full_gt"
POLYP_FULL_PATTERN="run_polyp_manifest.py.*output_current/Polyp/full_gt"
ISIC_METRICS="$ROOT/output_current/isic/full_gt/ISIC/metrics.csv"
POLYP_FULL_METRICS="$ROOT/output_current/Polyp/full_gt/test_metrics.csv"

cd "$ROOT"

wait_for_process() {
    local pattern="$1"
    local label="$2"
    while pgrep -f "$pattern" >/dev/null; do
        printf '%s waiting for %s\n' "$(date '+%F %T')" "$label"
        sleep 60
    done
}

wait_for_process "$ISIC_PATTERN" "ISIC full-GT completion"
if ! grep -q 'final_test_best_student' "$ISIC_METRICS"; then
    printf '%s ISIC full-GT did not reach final test; queue aborts.\n' "$(date '+%F %T')" >&2
    exit 1
fi

# Card 0 is already occupied by the active Polyp full-GT run. Do not overlap it.
wait_for_process "$POLYP_FULL_PATTERN" "Polyp full-GT completion on GPU 0"
if [[ ! -s "$POLYP_FULL_METRICS" ]]; then
    printf '%s Polyp full-GT has no test_metrics.csv; queue aborts.\n' "$(date '+%F %T')" >&2
    exit 1
fi

printf '%s starting Polyp 1%% GT DINO-only experiment on physical GPU 0\n' "$(date '+%F %T')"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u run_polyp_manifest.py \
    --labeled-ratio 1.0e-2 \
    --prompt-backend dino-prototype \
    --gpu 0 \
    --output-dir output_current/Polyp/1pct_dino_only \
    --epochs 10 \
    --batch-size 4 \
    --val-batch-size 16

printf '%s starting Polyp 1%% GT CLIP-only experiment on physical GPU 0\n' "$(date '+%F %T')"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u run_polyp_manifest.py \
    --labeled-ratio 1.0e-2 \
    --prompt-backend clip-only \
    --gpu 0 \
    --output-dir output_current/Polyp/1pct_clip_only \
    --epochs 10 \
    --batch-size 4 \
    --val-batch-size 16

printf '%s all queued single-encoder Polyp experiments completed\n' "$(date '+%F %T')"
