#!/usr/bin/env bash
set -uo pipefail
# IFP-parity WeSAM ablation, GPU1 queue: ISIC 1pct -> ISIC 10pct.
#
# Same parity changes as run_ifp_parity_gpu0.sh (see that file for details).
# Negative-point branch intentionally NOT re-enabled, per earlier decision.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED_1="$IDX/isic/seed1337/labeled_1pct.json"
ISIC_SUPPORT_1="$IDX/isic/seed1337/support_1pct.json"
ISIC_LABELED_10="$IDX/isic/seed1337/labeled_10pct.json"
ISIC_SUPPORT_10="$IDX/isic/seed1337/support_10pct.json"
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

require_files "$ISIC_LABELED_1" "$ISIC_SUPPORT_1" "$ISIC_LABELED_10" "$ISIC_SUPPORT_10"

echo "[$(date '+%F %T')] GPU1 QUEUE START (ifp parity)" | tee -a "$STATUS"

# --- 1pct ISIC WeSAM ---
run_one ifp_parity_1pct_isic_wesam 1 "$PYTHON" run_semisup_manifest.py \
  --dataset isic --gpu 1 --labeled-count 26 --output-dir "$G2/1pct_isic_wesam" \
  --epochs 10 --alignment-epochs 1 --alignment-seed 1337 \
  --iterative-max-iters 1 --iterative-sim-threshold 0 \
  --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$ISIC_LABELED_1" --support-indices-file "$ISIC_SUPPORT_1"

# --- 10pct ISIC WeSAM ---
run_one ifp_parity_10pct_isic_wesam 1 "$PYTHON" run_semisup_manifest.py \
  --dataset isic --gpu 1 --labeled-count 259 --output-dir "$G2/10pct_isic_wesam" \
  --epochs 10 --alignment-epochs 1 --alignment-seed 1337 \
  --iterative-max-iters 1 --iterative-sim-threshold 0 \
  --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$ISIC_LABELED_10" --support-indices-file "$ISIC_SUPPORT_10"

echo "[$(date '+%F %T')] GPU1 QUEUE DONE" | tee -a "$STATUS"
