#!/usr/bin/env bash
set -uo pipefail
# GPU1 robustness queue: dropout+jitter and pseudo-label agreement gate on ISIC 1pct.
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices/isic/seed1337"
OUT="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337"
G2="$OUT/group2"
STATUS="$OUT/status_robust_gpu1.log"
mkdir -p "$OUT/logs"
cd "$ROOT"

run_one() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name (gpu1)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$@" >"$OUT/logs/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name (gpu1)" | tee -a "$STATUS"
  else
    local code=$?; echo "[$(date '+%F %T')] FAILED(${code}) $name (gpu1)" | tee -a "$STATUS"
  fi
}

common() {
  echo --dataset isic --gpu 1 --labeled-count 26 --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --labeled-indices-file "$IDX/labeled_1pct.json" --support-indices-file "$IDX/support_1pct.json"
}

# E3: moderate dropout + small point jitter
run_one 1pct_isic_wesam_dropout03_jitter3 "$PYTHON" run_semisup_manifest.py $(common) \
  --output-dir "$G2/1pct_isic_wesam_dropout03_jitter3" \
  --train-prompt-dropout 0.3 --train-point-jitter-pixels 3
# E4: pseudo-label agreement gate (breaks the prompt-error feedback loop)
run_one 1pct_isic_wesam_gate07 "$PYTHON" run_semisup_manifest.py $(common) \
  --output-dir "$G2/1pct_isic_wesam_gate07" \
  --train-prompt-dropout 0 --train-point-jitter-pixels 0 --min-student-teacher-iou 0.7

echo "[$(date '+%F %T')] GPU1 ROBUST QUEUE ALL DONE" | tee -a "$STATUS"
