#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="$ROOT/output_current/fair_ablation_10pct_seed1337"
GROUP2="$OUT/group2"
INDEX_ROOT="$ROOT/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED="$INDEX_ROOT/isic/seed1337/labeled_10pct.json"
ISIC_SUPPORT="$INDEX_ROOT/isic/seed1337/support_10pct.json"
STATUS="$OUT/status.log"
WAIT_PID="${1:?usage: $0 PID_TO_WAIT_FOR}"

cd "$ROOT"
mkdir -p "$OUT/logs"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

echo "[$(date '+%F %T')] DONE  group2_isic_no_prompt (observed by gpu1-handoff)" | tee -a "$STATUS"
echo "[$(date '+%F %T')] START group2_isic_wesam (gpu1-extra)" | tee -a "$STATUS"
if env PYTHONUNBUFFERED=1 "$PYTHON" -u run_semisup_manifest.py \
  --dataset isic --gpu 1 --labeled-count 259 --output-dir "$GROUP2/isic_wesam" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT" \
  >"$OUT/logs/group2_isic_wesam.log" 2>&1; then
  echo "[$(date '+%F %T')] DONE  group2_isic_wesam (gpu1-extra)" | tee -a "$STATUS"
else
  code=$?
  echo "[$(date '+%F %T')] FAILED(${code}) group2_isic_wesam (gpu1-extra)" | tee -a "$STATUS"
  exit "$code"
fi
