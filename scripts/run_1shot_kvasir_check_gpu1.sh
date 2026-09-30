#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
OUT="$ROOT/output_current/kvasir_oneshot_seeds_check"
STATUS="$OUT/status.log"
mkdir -p "$OUT/logs"
cd "$ROOT"

run_one() {
  local name="$1"; local gpu="$2"; shift 2
  echo "[$(date '+%F %T')] START $name (gpu$gpu)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$@" >"$OUT/logs/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name (gpu$gpu)" | tee -a "$STATUS"
  else
    local code=$?; echo "[$(date '+%F %T')] FAILED(${code}) $name (gpu$gpu)" | tee -a "$STATUS"
  fi
}

echo "=== Starting Seed 3407 runs ===" | tee -a "$STATUS"
POLYP_LABELED_3407="$IDX/polyp/seed3407/labeled_1shot.json"
POLYP_SUPPORT_3407="$IDX/polyp/seed3407/support_1shot.json"
OUT_3407="$OUT/seed3407"

run_one seed3407_ifp_only 1 "$PYTHON" run_ifp_direct_sam2_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --output-dir "$OUT_3407/group1" --seed 3407 --batch-size 16 --alignment-epochs 1 --labeled-indices-file "$POLYP_LABELED_3407" --support-indices-file "$POLYP_SUPPORT_3407"
run_one seed3407_no_prompt 1 "$PYTHON" run_polyp_manifest.py --gpu 1 --labeled-count 1 --output-dir "$OUT_3407/group2/1shot_polyp_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 3407 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED_3407" --support-indices-file "$POLYP_SUPPORT_3407"
run_one seed3407_wesam 1 "$PYTHON" run_polyp_manifest.py --gpu 1 --labeled-count 1 --output-dir "$OUT_3407/group2/1shot_polyp_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 3407 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED_3407" --support-indices-file "$POLYP_SUPPORT_3407"

echo "[$(date '+%F %T')] SEED 3407 ALL COMPLETED" | tee -a "$STATUS"
