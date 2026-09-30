#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
LAUNCHER="$ROOT/scripts/wesam_hidden_launch.py"
OUT="$ROOT/output_current/fair_ablation_supplement_20260923/gpu1_isic_fullsup_gtprompt"
LOG="$ROOT/output_current/fair_ablation_supplement_20260923/logs/isic_fullsup_gtprompt_ifpeval.log"
STATUS="$ROOT/output_current/fair_ablation_supplement_20260923/status_gpu1.log"
mkdir -p "$OUT" "$ROOT/output_current/fair_ablation_supplement_20260923/logs"
cd "$ROOT"

echo "[$(date '+%F %T')] START isic_fullsup_gtprompt_ifpeval (gpu1, full GT train + GT-point train prompts + IFP eval, seed 3407)" | tee -a "$STATUS"

PYTHONUNBUFFERED=1 "$PYTHON" "$LAUNCHER" run_semisup_manifest.py --dataset isic \
  --gpu 1 --labeled-ratio 1.0 --seed 3407 \
  --output-dir "$OUT" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 \
  --prompt-backend ifp --labeled-prompt-mode gt \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-prompt-dropout 0 \
  > "$LOG" 2>&1

if [ $? -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE  isic_fullsup_gtprompt_ifpeval (gpu1)" | tee -a "$STATUS"
else
  echo "[$(date '+%F %T')] FAILED isic_fullsup_gtprompt_ifpeval (gpu1)" | tee -a "$STATUS"
fi
