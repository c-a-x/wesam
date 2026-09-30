#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
OUT_DIR="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_isic_fusion_drop01_reg10"
LOG_FILE="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/logs/1pct_isic_fusion_drop01_reg10.log"
STATUS_FILE="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/status_fusion.log"

mkdir -p "$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/logs"
cd "$ROOT"

echo "[$(date '+%F %T')] START 1pct_isic_fusion_drop01_reg10 (gpu1)" | tee -a "$STATUS_FILE"

PYTHONUNBUFFERED=1 "$PYTHON" run_semisup_manifest.py \
  --dataset isic --gpu 1 --labeled-count 26 \
  --output-dir "$OUT_DIR" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0.1 \
  --labeled-indices-file "$IDX/isic/seed1337/labeled_1pct.json" \
  --support-indices-file "$IDX/isic/seed1337/support_1pct.json" \
  > "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE  1pct_isic_fusion_drop01_reg10 (gpu1)" | tee -a "$STATUS_FILE"
else
  echo "[$(date '+%F %T')] FAILED($EXIT_CODE) 1pct_isic_fusion_drop01_reg10 (gpu1)" | tee -a "$STATUS_FILE"
fi
