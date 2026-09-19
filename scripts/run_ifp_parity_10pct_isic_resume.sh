#!/usr/bin/env bash
set -uo pipefail
# Resume IFP-parity 10pct ISIC WeSAM after the 2026-09-18 10:51 machine reboot.
# The IFP alignment head already finished before the reboot, so reuse it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
IDX="$ROOT/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED_10="$IDX/isic/seed1337/labeled_10pct.json"
ISIC_SUPPORT_10="$IDX/isic/seed1337/support_10pct.json"
OUT="$ROOT/output_current/ifp_parity_1pct_10pct_seed1337"
G2="$OUT/group2"
STATUS="$OUT/status.log"
cd "$ROOT"

echo "[$(date '+%F %T')] START ifp_parity_10pct_isic_wesam (gpu1, resume)" | tee -a "$STATUS"
if env PYTHONUNBUFFERED=1 "$PYTHON" run_semisup_manifest.py \
  --dataset isic --gpu 1 --labeled-count 259 --output-dir "$G2/10pct_isic_wesam" \
  --epochs 10 --alignment-epochs 1 --alignment-seed 1337 --reuse-alignment \
  --iterative-max-iters 1 --iterative-sim-threshold 0 \
  --batch-size 4 --val-batch-size 16 --seed 1337 \
  --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
  --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
  --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
  --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
  --labeled-indices-file "$ISIC_LABELED_10" --support-indices-file "$ISIC_SUPPORT_10" \
  > "$OUT/logs/ifp_parity_10pct_isic_wesam.log" 2>&1; then
  echo "[$(date '+%F %T')] DONE  ifp_parity_10pct_isic_wesam (gpu1)" | tee -a "$STATUS"
else
  code=$?; echo "[$(date '+%F %T')] FAILED(${code}) ifp_parity_10pct_isic_wesam (gpu1)" | tee -a "$STATUS"
fi
echo "[$(date '+%F %T')] GPU1 QUEUE DONE" | tee -a "$STATUS"
