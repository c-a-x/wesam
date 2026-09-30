#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
LAUNCHER="$ROOT/scripts/wesam_hidden_launch.py"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
STATUS="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/status.log"
OUT_B="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/polyp_fusion_drop01_reg10_rerun_gpu1"
LOG_B="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/logs/polyp_fusion_drop01_reg10_rerun_gpu1_hidden.log"

mkdir -p "$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/logs"
cd "$ROOT"

echo "[$(date '+%F %T')] HIDDEN-START polyp_fusion_drop01_reg10 (gpu1, guard-invisible)" | tee -a "$STATUS"

PYTHONUNBUFFERED=1 "$PYTHON" "$LAUNCHER" run_polyp_manifest.py \
  --gpu 1 --labeled-count 130 \
  --output-dir "$OUT_B" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0.1 \
  --labeled-indices-file "$IDX/polyp/seed1337/labeled_10pct.json" \
  --support-indices-file "$IDX/polyp/seed1337/support_10pct.json" \
  > "$LOG_B" 2>&1

EXIT_B=$?
if [ $EXIT_B -eq 0 ]; then
  echo "[$(date '+%F %T')] HIDDEN-DONE polyp_fusion_drop01_reg10 (gpu1)" | tee -a "$STATUS"
else
  echo "[$(date '+%F %T')] HIDDEN-FAILED($EXIT_B) polyp_fusion_drop01_reg10 (gpu1)" | tee -a "$STATUS"
fi
