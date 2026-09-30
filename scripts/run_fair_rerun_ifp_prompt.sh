#!/usr/bin/env bash
set -euo pipefail

# Group-2 WeSAM rerun with the train/test prompt channel made consistent.
#
# Original protocol (fair_ablation_10pct_seed1337): labeled samples (70% of a
# batch) had their IFP point replaced by a GT interior point
# (--labeled-prompt-mode gt), while validation/test always used IFP points.
# The 30% unlabeled samples were the only ones teaching the IFP-point ->
# mask channel, and the model over-fit to precise GT points, so at test time
# (IFP points) WeSAM was worse than No prompt.
#
# Fix: keep the user's supervision design (labeled -> GT mask loss,
# unlabeled -> teacher pseudo-mask loss) but feed IFP points to every sample
# so training and test prompts match.
#
# Two variants per dataset:
#   ifp_prompt      : only the prompt mode changes (isolates the prompt fix)
#   ifp_prompt_reg  : + anchor_weight=1, contrast_weight=1 (WeSAM regularisation)

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="$ROOT/output_current/fair_ablation_10pct_seed1337_rerun"
INDEX_ROOT="$ROOT/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED="$INDEX_ROOT/isic/seed1337/labeled_10pct.json"
ISIC_SUPPORT="$INDEX_ROOT/isic/seed1337/support_10pct.json"
POLYP_LABELED="$INDEX_ROOT/polyp/seed1337/labeled_10pct.json"
POLYP_SUPPORT="$INDEX_ROOT/polyp/seed1337/support_10pct.json"
STATUS="$OUT/status.log"

cd "$ROOT"
mkdir -p "$OUT/logs"

run_one() {
  local name="$1" gpu="$2"; shift 2
  echo "[$(date '+%F %T')] START $name (gpu$gpu)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$@" >"$OUT/logs/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name (gpu$gpu)" | tee -a "$STATUS"
  else
    local code=$?
    echo "[$(date '+%F %T')] FAILED(${code}) $name (gpu$gpu)" | tee -a "$STATUS"
    return "$code"
  fi
}

# ---------------- GPU 0: ISIC plain, then Polyp plain ----------------
(
  set -e
  run_one isic_ifp_prompt 0 "$PYTHON" run_semisup_manifest.py \
    --dataset isic --gpu 0 --labeled-count 259 --output-dir "$OUT/isic_ifp_prompt" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT"

  run_one polyp_ifp_prompt 0 "$PYTHON" run_polyp_manifest.py \
    --gpu 0 --labeled-count 130 --output-dir "$OUT/polyp_ifp_prompt" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
) &
P0=$!

# ---------------- GPU 1: ISIC regularised, then Polyp regularised ----------------
(
  set -e
  run_one isic_ifp_prompt_reg 1 "$PYTHON" run_semisup_manifest.py \
    --dataset isic --gpu 1 --labeled-count 259 --output-dir "$OUT/isic_ifp_prompt_reg" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "$ISIC_LABELED" --support-indices-file "$ISIC_SUPPORT"

  run_one polyp_ifp_prompt_reg 1 "$PYTHON" run_polyp_manifest.py \
    --gpu 1 --labeled-count 130 --output-dir "$OUT/polyp_ifp_prompt_reg" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode ifp --labeled-gt-prompt-probability 0.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 1.0 --contrast-weight 1.0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "$POLYP_LABELED" --support-indices-file "$POLYP_SUPPORT"
) &
P1=$!

wait "$P0" "$P1"
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$STATUS"
