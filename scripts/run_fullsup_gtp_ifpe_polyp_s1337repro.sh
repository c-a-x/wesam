#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
LAUNCHER="$ROOT/scripts/wesam_hidden_launch.py"
OUT="$ROOT/output_current/fair_ablation_supplement_20260923/gpu0_polyp_fullsup_gtprompt_s1337repro"
LOG="$ROOT/output_current/fair_ablation_supplement_20260923/logs/polyp_fullsup_gtprompt_ifpeval_s1337repro.log"
STATUS="$ROOT/output_current/fair_ablation_supplement_20260923/status_gpu0.log"
mkdir -p "$OUT" "$ROOT/output_current/fair_ablation_supplement_20260923/logs"
cd "$ROOT"

echo "[$(date '+%F %T')] START polyp_fullsup_gtprompt_ifpeval_s1337repro (gpu0, seed 2027)" | tee -a "$STATUS"

PYTHONUNBUFFERED=1 "$PYTHON" "$LAUNCHER" run_polyp_manifest.py \
  --gpu 0 --labeled-ratio 1.0 --seed 1337 \
  --output-dir "$OUT" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 \
  --prompt-backend ifp --labeled-prompt-mode gt \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-prompt-dropout 0 \
  > "$LOG" 2>&1

if [ $? -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE  polyp_fullsup_gtprompt_ifpeval_s1337repro (gpu0)" | tee -a "$STATUS"
else
  echo "[$(date '+%F %T')] FAILED polyp_fullsup_gtprompt_ifpeval_s1337repro (gpu0)" | tee -a "$STATUS"
fi
