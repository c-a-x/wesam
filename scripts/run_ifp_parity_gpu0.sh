#!/usr/bin/env bash
set -uo pipefail
# IFP-parity WeSAM ablation, GPU0 queue: Polyp 1pct -> Polyp 10pct.
#
# Parity changes vs. the earlier fair-protocol WeSAM runs (negative-point
# branch intentionally NOT re-enabled, per earlier decision):
#   * IFP alignment head: 1 epoch, seed 1337 (historical high-metric fair protocol)
#   * iterative pseudo masks: max_iters=1 and score-map suppression
#   * hard patch suppression (>0.1 coverage), matching the fair protocol
#   * SAM2 multimask_output=False (single-mask decode)
# Everything else (SAM2 self-training protocol) is unchanged: 10 epochs,
# batch 4, val-batch 16, seed 1337, lr 2e-4, warmup 100, rampup 1,
# teacher 0.1, anchor 0, contrast 0, labeled-batch-prob 0.7, no jitter/dropout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
POLYP_LABELED_1="$IDX/polyp/seed1337/labeled_1pct.json"
POLYP_SUPPORT_1="$IDX/polyp/seed1337/support_1pct.json"
POLYP_LABELED_10="$IDX/polyp/seed1337/labeled_10pct.json"
POLYP_SUPPORT_10="$IDX/polyp/seed1337/support_10pct.json"
OUT="$ROOT/output_current/ifp_parity_1pct_10pct_seed1337"
G2="$OUT/group2"
STATUS="$OUT/status.log"
mkdir -p "$OUT/logs"
cd "$ROOT"

require_files() {
  for path in "$@"; do
    if [[ ! -f "$path" ]]; then echo "missing required file: $path" >&2; exit 2; fi
  done
}

run_one() {
  local name="$1"; local gpu="$2"; shift 2
  echo "[$(date '+%F %T')] START $name (gpu$gpu)" | tee -a "$STATUS"
  if env PYTHONUNBUFFERED=1 "$@" >"$OUT/logs/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name (gpu$gpu)" | tee -a "$STATUS"
  else
    local code=$?; echo "[$(date '+%F %T')] FAILED(${code}) $name (gpu$gpu)" | tee -a "$STATUS"
  fi
}

require_files "$POLYP_LABELED_1" "$POLYP_SUPPORT_1" "$POLYP_LABELED_10" "$POLYP_SUPPORT_10"

echo "[$(date '+%F %T')] GPU0 QUEUE START (ifp parity)" | tee -a "$STATUS"

# --- 1pct Polyp WeSAM ---
run_one ifp_parity_1pct_polyp_wesam 0 "$PYTHON" run_polyp_manifest.py \
  --gpu 0 --labeled-count 13 --output-dir "$G2/1pct_polyp_wesam" \
  --epochs 10 --alignment-epochs 1 --alignment-seed 1337 \
  --iterative-max-iters 1 --iterative-sim-threshold 0 \
  --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$POLYP_LABELED_1" --support-indices-file "$POLYP_SUPPORT_1"

# --- 10pct Polyp WeSAM ---
run_one ifp_parity_10pct_polyp_wesam 0 "$PYTHON" run_polyp_manifest.py \
  --gpu 0 --labeled-count 130 --output-dir "$G2/10pct_polyp_wesam" \
  --epochs 10 --alignment-epochs 1 --alignment-seed 1337 \
  --iterative-max-iters 1 --iterative-sim-threshold 0 \
  --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$POLYP_LABELED_10" --support-indices-file "$POLYP_SUPPORT_10"

echo "[$(date '+%F %T')] GPU0 QUEUE DONE" | tee -a "$STATUS"
