#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="$ROOT/output_current/fair_ablation_10pct_seed1337"
GROUP2="$OUT/group2"
INDEX_ROOT="$ROOT/output_current/nested_budget_control_20260823/indices"
POLYP_LABELED="$INDEX_ROOT/polyp/seed1337/labeled_10pct.json"
POLYP_SUPPORT="$INDEX_ROOT/polyp/seed1337/support_10pct.json"
STATUS="$OUT/status.log"

cd "$ROOT"
mkdir -p "$OUT/logs"

run_task() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name} (gpu0-extra)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$PYTHON" -u "$@" >"$OUT/logs/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  ${name} (gpu0-extra)" | tee -a "$STATUS"
  else
    local code=$?
    echo "[$(date '+%F %T')] FAILED(${code}) ${name} (gpu0-extra)" | tee -a "$STATUS"
    return "$code"
  fi
}

run_task group2_polyp_no_prompt \
  run_polyp_manifest.py \
  --gpu 0 --labeled-count 130 --output-dir "$GROUP2/polyp_no_prompt" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend no-prompt --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"

run_task group2_polyp_wesam \
  run_polyp_manifest.py \
  --gpu 0 --labeled-count 130 --output-dir "$GROUP2/polyp_wesam" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
