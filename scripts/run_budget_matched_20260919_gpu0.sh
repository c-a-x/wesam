#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
OUT="$ROOT/output_current/fair_budget_matched_20260919"
STATUS="$OUT/status_gpu0.log"
mkdir -p "$OUT/logs" "$OUT/group1" "$OUT/group2"
cd "$ROOT"

run_one() {
  local name="$1"; local gpu="$2"; shift 2
  echo "[$(date '+%F %T')] START $name (gpu$gpu)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$@" >"$OUT/logs/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name (gpu$gpu)" | tee -a "$STATUS"
  else
    local code=$?; echo "[$(date '+%F %T')] FAILED(${code}) $name (gpu$gpu)" | tee -a "$STATUS"
    return "$code"
  fi
}

I1L="$IDX/isic/seed1337/labeled_1pct.json"; I1S="$IDX/isic/seed1337/support_1pct.json"
P1L="$IDX/polyp/seed1337/labeled_1pct.json"; P1S="$IDX/polyp/seed1337/support_1pct.json"
G1="$OUT/group1"; G2="$OUT/group2"

# Group 1: frozen prompt-only baselines, budget-matched references.
run_one 1pct_isic_clip 0 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset isic --gpu 0 --labeled-count 26 --budget-tag 1pct --prompt-backend clip-only --method-dir clip_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$I1L" --support-indices-file "$I1S"
run_one 1pct_isic_dino 0 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset isic --gpu 0 --labeled-count 26 --budget-tag 1pct --prompt-backend dino-prototype --method-dir dino_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$I1L" --support-indices-file "$I1S"
run_one 1pct_isic_ifp 0 "$PYTHON" run_ifp_direct_sam2_ablation.py --dataset isic --gpu 0 --labeled-count 26 --budget-tag 1pct --output-dir "$G1" --seed 1337 --batch-size 16 --alignment-epochs 1 --labeled-indices-file "$I1L" --support-indices-file "$I1S"
run_one 1pct_polyp_clip 0 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 0 --labeled-count 13 --budget-tag 1pct --prompt-backend clip-only --method-dir clip_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$P1L" --support-indices-file "$P1S"
run_one 1pct_polyp_dino 0 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 0 --labeled-count 13 --budget-tag 1pct --prompt-backend dino-prototype --method-dir dino_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$P1L" --support-indices-file "$P1S"
run_one 1pct_polyp_ifp 0 "$PYTHON" run_ifp_direct_sam2_ablation.py --dataset polyp --gpu 0 --labeled-count 13 --budget-tag 1pct --output-dir "$G1" --seed 1337 --batch-size 16 --alignment-epochs 1 --labeled-indices-file "$P1L" --support-indices-file "$P1S"

# Group 2: semi-supervised baselines and WeSAM, budget-matched IFP support for WeSAM.
run_one 1pct_isic_no_prompt 0 "$PYTHON" run_semisup_manifest.py --dataset isic --gpu 0 --labeled-count 26 --output-dir "$G2/1pct_isic_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$I1L" --support-indices-file "$I1S"
run_one 1pct_isic_wesam 0 "$PYTHON" run_semisup_manifest.py --dataset isic --gpu 0 --labeled-count 26 --output-dir "$G2/1pct_isic_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$I1L" --support-indices-file "$I1S"
run_one 1pct_polyp_no_prompt 0 "$PYTHON" run_polyp_manifest.py --gpu 0 --labeled-count 13 --output-dir "$G2/1pct_polyp_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$P1L" --support-indices-file "$P1S"
run_one 1pct_polyp_wesam 0 "$PYTHON" run_polyp_manifest.py --gpu 0 --labeled-count 13 --output-dir "$G2/1pct_polyp_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$P1L" --support-indices-file "$P1S"
echo "[$(date '+%F %T')] GPU0 ALL DONE" | tee -a "$STATUS"
