#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
STATUS_1="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/status_experiments.log"

mkdir -p "$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/logs"
cd "$ROOT"

# 等待任务 B (PID 153402) 结束
echo "[$(date '+%F %T')] Waiting for task B (polyp_fusion_drop01_reg10, PID 153402) to finish..." | tee -a "$STATUS_1"
while kill -0 153402 2>/dev/null; do
    sleep 15
done
echo "[$(date '+%F %T')] Task B finished. Now launching Task C1 & C2..." | tee -a "$STATUS_1"

# ==========================================
# 任务 C1: Polyp 1% 方案一 (Prompt Dropout 0.1)
# ==========================================
echo "[$(date '+%F %T')] START 1pct_polyp_wesam_dropout01 (gpu0)" | tee -a "$STATUS_1"
OUT_C1="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_polyp_wesam_dropout01"
LOG_C1="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/logs/1pct_polyp_wesam_dropout01.log"

PYTHONUNBUFFERED=1 "$PYTHON" run_polyp_manifest.py \
  --gpu 0 --labeled-count 13 \
  --output-dir "$OUT_C1" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0.1 \
  --labeled-indices-file "$IDX/polyp/seed1337/labeled_1pct.json" \
  --support-indices-file "$IDX/polyp/seed1337/support_1pct.json" \
  > "$LOG_C1" 2>&1

EXIT_C1=$?
if [ $EXIT_C1 -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE  1pct_polyp_wesam_dropout01 (gpu0)" | tee -a "$STATUS_1"
else
  echo "[$(date '+%F %T')] FAILED($EXIT_C1) 1pct_polyp_wesam_dropout01 (gpu0)" | tee -a "$STATUS_1"
fi

# ==========================================
# 任务 C2: Polyp 1% 融合方案 (Dropout 0.1 + Reg 1.0)
# ==========================================
echo "[$(date '+%F %T')] START 1pct_polyp_wesam_fusion_drop01_reg10 (gpu0)" | tee -a "$STATUS_1"
OUT_C2="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_polyp_wesam_fusion_drop01_reg10"
LOG_C2="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/logs/1pct_polyp_wesam_fusion_drop01_reg10.log"

PYTHONUNBUFFERED=1 "$PYTHON" run_polyp_manifest.py \
  --gpu 0 --labeled-count 13 \
  --output-dir "$OUT_C2" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0.1 \
  --labeled-indices-file "$IDX/polyp/seed1337/labeled_1pct.json" \
  --support-indices-file "$IDX/polyp/seed1337/support_1pct.json" \
  > "$LOG_C2" 2>&1

EXIT_C2=$?
if [ $EXIT_C2 -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE  1pct_polyp_wesam_fusion_drop01_reg10 (gpu0)" | tee -a "$STATUS_1"
else
  echo "[$(date '+%F %T')] FAILED($EXIT_C2) 1pct_polyp_wesam_fusion_drop01_reg10 (gpu0)" | tee -a "$STATUS_1"
fi

echo "[$(date '+%F %T')] ALL GPU0 TASKS COMPLETED" | tee -a "$STATUS_1"
