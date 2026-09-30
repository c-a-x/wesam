#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
LAUNCHER="$ROOT/scripts/wesam_hidden_launch.py"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
OUT="$ROOT/output_current/fair_ablation_supplement_20260923/gpu0_isic"
STATUS="$ROOT/output_current/fair_ablation_supplement_20260923/status_gpu0.log"
mkdir -p "$OUT" "$ROOT/output_current/fair_ablation_supplement_20260923/logs"
cd "$ROOT"

run_one() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name (gpu0)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$PYTHON" "$LAUNCHER" run_semisup_manifest.py --dataset isic --gpu 0 \
    --output-dir "$OUT/$name" "$@" \
    > "$ROOT/output_current/fair_ablation_supplement_20260923/logs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name (gpu0)" | tee -a "$STATUS"
  else
    echo "[$(date '+%F %T')] FAILED $name (gpu0)" | tee -a "$STATUS"
  fi
}

COMMON="--epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0"

# ---- 1shot 三方案 (seed 1337) ----
run_one isic_1shot_dropout02  $COMMON --seed 1337 --labeled-count 1 --train-prompt-dropout 0.2 --anchor-weight 0 --contrast-weight 0 --labeled-indices-file "$IDX/isic/seed1337/labeled_1shot.json" --support-indices-file "$IDX/isic/seed1337/support_1shot.json"
run_one isic_1shot_reg10      $COMMON --seed 1337 --labeled-count 1 --train-prompt-dropout 0 --anchor-weight 1.0 --contrast-weight 1.0 --labeled-indices-file "$IDX/isic/seed1337/labeled_1shot.json" --support-indices-file "$IDX/isic/seed1337/support_1shot.json"
run_one isic_1shot_fusion     $COMMON --seed 1337 --labeled-count 1 --train-prompt-dropout 0.1 --anchor-weight 1.0 --contrast-weight 1.0 --labeled-indices-file "$IDX/isic/seed1337/labeled_1shot.json" --support-indices-file "$IDX/isic/seed1337/support_1shot.json"
# ---- 1pct 多 seed: baseline 与方案一(dropout 0.2) 配对 ----
run_one isic_1pct_wesam_s2027        $COMMON --seed 2027 --labeled-count 26 --train-prompt-dropout 0 --anchor-weight 0 --contrast-weight 0 --labeled-indices-file "$IDX/isic/seed2027/labeled_1pct.json" --support-indices-file "$IDX/isic/seed2027/support_1pct.json"
run_one isic_1pct_dropout02_s2027    $COMMON --seed 2027 --labeled-count 26 --train-prompt-dropout 0.2 --anchor-weight 0 --contrast-weight 0 --labeled-indices-file "$IDX/isic/seed2027/labeled_1pct.json" --support-indices-file "$IDX/isic/seed2027/support_1pct.json"
run_one isic_1pct_wesam_s3407        $COMMON --seed 3407 --labeled-count 26 --train-prompt-dropout 0 --anchor-weight 0 --contrast-weight 0 --labeled-indices-file "$IDX/isic/seed3407/labeled_1pct.json" --support-indices-file "$IDX/isic/seed3407/support_1pct.json"
run_one isic_1pct_dropout02_s3407    $COMMON --seed 3407 --labeled-count 26 --train-prompt-dropout 0.2 --anchor-weight 0 --contrast-weight 0 --labeled-indices-file "$IDX/isic/seed3407/labeled_1pct.json" --support-indices-file "$IDX/isic/seed3407/support_1pct.json"

echo "[$(date '+%F %T')] GPU0 ISIC ALL 7 DONE" | tee -a "$STATUS"
