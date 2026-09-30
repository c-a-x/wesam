#!/usr/bin/env bash
set -uo pipefail
# TRUE 1-shot: 1 labeled + 1 support (the same image). Fixed files labeled_1shot.json / support_1shot.json.
# GPU1: group1 Polyp (clip/dino/ifp, frozen) + 1shot ISIC NoPrompt + 1shot Polyp NoPrompt/WeSAM
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED="$IDX/isic/seed1337/labeled_1shot.json"
ISIC_SUPPORT="$IDX/isic/seed1337/support_1shot.json"
POLYP_LABELED="$IDX/polyp/seed1337/labeled_1shot.json"
POLYP_SUPPORT="$IDX/polyp/seed1337/support_1shot.json"
OUT="$ROOT/output_current/fair_ablation_1shot_seed1337"
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

# ---- 1shot Polyp group1 (frozen, 1 reference image) ----
run_one 1shot_polyp_clip 1 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --prompt-backend clip-only --method-dir clip_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1shot_polyp_dino 1 "$PYTHON" run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --prompt-backend dino-prototype --method-dir dino_only --output-dir "$G1" --seed 1337 --batch-size 16 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1shot_polyp_ifp 1 "$PYTHON" run_ifp_direct_sam2_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --budget-tag 1shot --output-dir "$G1" --seed 1337 --batch-size 16 --alignment-epochs 1 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
# ---- 1shot ISIC NoPrompt training (group2) ----
run_one 1shot_isic_no_prompt 1 "$PYTHON" run_semisup_manifest.py --dataset isic --gpu 1 --labeled-count 1 --output-dir "$G2/1shot_isic_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT"
# ---- 1shot Polyp NoPrompt / WeSAM training (group2) ----
run_one 1shot_polyp_no_prompt 1 "$PYTHON" run_polyp_manifest.py --gpu 1 --labeled-count 1 --output-dir "$G2/1shot_polyp_no_prompt" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend no-prompt --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
run_one 1shot_polyp_wesam 1 "$PYTHON" run_polyp_manifest.py --gpu 1 --labeled-count 1 --output-dir "$G2/1shot_polyp_wesam" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 --train-point-jitter-pixels 0 --train-prompt-dropout 0 --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"

echo "[$(date '+%F %T')] GPU1 ALL DONE" | tee -a "$STATUS"
