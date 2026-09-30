#!/usr/bin/env bash
set -uo pipefail
# GPU1 (balanced ~18h training): Polyp 1pct+1shot group1(frozen) + 1pct ISIC WeSAM, 1pct Polyp NoPrompt, 1shot ISIC WeSAM, 1shot Polyp NoPrompt
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

# ---- 1pct Polyp group1 (frozen) ----
run_one 1pct_polyp_clip 1 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 13 --budget-tag 1pct --prompt-backend clip-only --method-dir clip_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1pct_polyp_dino 1 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 13 --budget-tag 1pct --prompt-backend dino-prototype --method-dir dino_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1pct_polyp_ifp 1 "$PYTHON" run_ifp_direct_sam2_ablation.py --dataset polyp --gpu 1 --labeled-count 13 --budget-tag 1pct --output-dir "$G1" --seed 1337 --batch-size 16 --alignment-epochs 1 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
# ---- 1shot Polyp group1 (frozen) ----
run_one 1shot_polyp_clip 1 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --prompt-backend clip-only --method-dir clip_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1shot_polyp_dino 1 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --prompt-backend dino-prototype --method-dir dino_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1shot_polyp_ifp 1 "$PYTHON" run_ifp_direct_sam2_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --output-dir "$G1" --seed 1337 --batch-size 16 --alignment-epochs 1 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
# ---- training (balanced) ----
run_one 1pct_isic_wesam 1 "$PYTHON" run_semisup_manifest.py --dataset isic --gpu 1 --labeled-count 26 --output-dir "$G2/1pct_isic_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT"
run_one 1pct_polyp_no_prompt 1 "$PYTHON" run_polyp_manifest.py --gpu 1 --labeled-count 13 --output-dir "$G2/1pct_polyp_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1shot_isic_wesam 1 "$PYTHON" run_semisup_manifest.py --dataset isic --gpu 1 --labeled-count 1 --output-dir "$G2/1shot_isic_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT"
run_one 1shot_polyp_no_prompt 1 "$PYTHON" run_polyp_manifest.py --gpu 1 --labeled-count 1 --output-dir "$G2/1shot_polyp_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"

echo "[$(date '+%F %T')] GPU1 ALL DONE" | tee -a "$STATUS"
