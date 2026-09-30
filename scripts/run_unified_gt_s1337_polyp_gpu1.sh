#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
LAUNCHER="$ROOT/scripts/wesam_hidden_launch.py"
BASE="/datanas01/nas01/Student-home/2024U/YBC/wesam2/output_current/fair_ablation_supplement_20261001"
mkdir -p "$BASE/logs"
run_one () {
  DS="$1"; LC="$2"; TAG="$3"
  OUT="$BASE/unified_gt_${DS}_${TAG}"
  LOG="$BASE/logs/unified_gt_${DS}_${TAG}.log"
  STATUS="$BASE/status_gpu1.log"
  echo "[$(date '+%F %T')] START unified_gt_${DS}_${TAG} (gpu1, seed 1337, gt-train)" | tee -a "$STATUS"
  cd "$ROOT"
  PYTHONUNBUFFERED=1 "$PYTHON" "$LAUNCHER" run_polyp_manifest.py \
    --gpu 1 --labeled-count "$LC" --seed 1337 \
    --output-dir "$OUT" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 \
    --prompt-backend ifp --labeled-prompt-mode gt \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-prompt-dropout 0 \
    > "$LOG" 2>&1
  if [ $? -eq 0 ]; then
    echo "[$(date '+%F %T')] DONE  unified_gt_${DS}_${TAG} (gpu1)" | tee -a "$STATUS"
  else
    echo "[$(date '+%F %T')] FAILED unified_gt_${DS}_${TAG} (gpu1)" | tee -a "$STATUS"
  fi
}
run_one polyp 1 1shot
if [ "polyp" = "isic" ]; then run_one isic 26 1pct; run_one isic 259 10pct; else run_one polyp 13 1pct; run_one polyp 130 10pct; fi
