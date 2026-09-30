#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
STATUS="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/status.log"

OUT_B="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/polyp_fusion_drop01_reg10"
OUT_RERUN="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/polyp_fusion_drop01_reg10_rerun_gpu1"
LOG_RERUN="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/logs/polyp_fusion_drop01_reg10_rerun_gpu1.log"
EVAL_ROOT="$OUT_B/eval_from_epoch5_interrupted"

mkdir -p "$ROOT/output_current/fair_ablation_10pct_seed1337_rerun/logs"
mkdir -p "$EVAL_ROOT"
cd "$ROOT"

# ============================================================
# Stage 1: 评测中断前保存的 epoch-5 权重（best-student.pth）
# 5 个外域测试集，结果写入 eval_from_epoch5_interrupted/<Domain>/
# ============================================================
echo "[$(date '+%F %T')] RESCUE-STAGE1 START: eval interrupted epoch-5 weights on 5 test domains (gpu1)" | tee -a "$STATUS"

for DOMAIN in Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-LaribPolypDB; do
  echo "[$(date '+%F %T')] RESCUE-STAGE1 eval domain=$DOMAIN" | tee -a "$STATUS"
  mkdir -p "$EVAL_ROOT/${DOMAIN}"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" export_test_predictions.py \
    --dataset polyp \
    --experiment-dir "$OUT_B" \
    --test-list "$OUT_B/lists/test_${DOMAIN}.csv" \
    --result-dir "$EVAL_ROOT/${DOMAIN}" \
    --device cuda --batch-size 16 --prompt-backend ifp \
    > "$EVAL_ROOT/${DOMAIN}.log" 2>&1
  EXIT_D=$?
  echo "[$(date '+%F %T')] RESCUE-STAGE1 done domain=$DOMAIN exit=$EXIT_D" | tee -a "$STATUS"
done

echo "[$(date '+%F %T')] RESCUE-STAGE1 ALL DOMAINS EVALUATED" | tee -a "$STATUS"

# ============================================================
# Stage 2: 从头完整重跑任务 B (Polyp 10% 融合方案, gpu1 独占)
# 参数与原 fair protocol 完全一致，新 output-dir 避免 SKIP 逻辑
# ============================================================
echo "[$(date '+%F %T')] RESCUE-STAGE2 START: full rerun polyp_fusion_drop01_reg10 (gpu1)" | tee -a "$STATUS"

PYTHONUNBUFFERED=1 "$PYTHON" run_polyp_manifest.py \
  --gpu 1 --labeled-count 130 \
  --output-dir "$OUT_RERUN" \
  --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0.1 \
  --labeled-indices-file "$IDX/polyp/seed1337/labeled_10pct.json" \
  --support-indices-file "$IDX/polyp/seed1337/support_10pct.json" \
  > "$LOG_RERUN" 2>&1

EXIT_B2=$?
if [ $EXIT_B2 -eq 0 ]; then
  echo "[$(date '+%F %T')] RESCUE-STAGE2 DONE polyp_fusion_drop01_reg10 rerun (gpu1)" | tee -a "$STATUS"
else
  echo "[$(date '+%F %T')] RESCUE-STAGE2 FAILED($EXIT_B2) polyp_fusion_drop01_reg10 rerun (gpu1)" | tee -a "$STATUS"
fi

echo "[$(date '+%F %T')] RESCUE GPU1 CHAIN COMPLETED" | tee -a "$STATUS"
