#!/usr/bin/env bash
# 1pct ISIC WeSAM + anchor/contrast regularisation.
# Single-variable change vs 1pct_isic_wesam (anchor=1, contrast=1).
# Everything else identical: same indices, same epochs, same lr, same prompt mode.
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices/isic/seed1337"
OUT="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_isic_wesam_reg"
LOG="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/logs/1pct_isic_wesam_reg.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 "$PYTHON" run_semisup_manifest.py \
  --dataset isic --gpu "${GPU:-1}" --labeled-count 26 --output-dir "$OUT" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$IDX/labeled_1pct.json" --support-indices-file "$IDX/support_1pct.json" \
  >"$LOG" 2>&1
