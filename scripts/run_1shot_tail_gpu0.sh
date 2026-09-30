#!/usr/bin/env bash
set -uo pipefail
# GPU0: 1shot_isic_wesam + 1shot_polyp_no_prompt (串行)
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED="$IDX/isic/seed1337/labeled_1pct.json"
ISIC_SUPPORT="$IDX/isic/seed1337/support_10pct.json"
POLYP_LABELED="$IDX/polyp/seed1337/labeled_1pct.json"
POLYP_SUPPORT="$IDX/polyp/seed1337/support_10pct.json"
OUT="$ROOT/output_current/fair_ablation_1pct_1shot_seed1337"
G1="$OUT/group1"; G2="$OUT/group2"
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

run_one 1shot_isic_wesam 0 "$PYTHON" run_semisup_manifest.py --dataset isic --gpu 0 --labeled-count 1 --output-dir "$G2/1shot_isic_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT"
run_one 1shot_polyp_no_prompt 0 "$PYTHON" run_polyp_manifest.py --gpu 0 --labeled-count 1 --output-dir "$G2/1shot_polyp_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"

echo "[$(date '+%F %T')] GPU0 TAIL ALL DONE" | tee -a "$STATUS"
