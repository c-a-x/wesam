#!/usr/bin/env bash
set -euo pipefail

# Accelerated 10-epoch WeSAM controls for ISIC and the five Polyp test sets.
# Both jobs use the fixed seed=3407 nested-budget 10% labeled/support sets.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="${ROOT}/output_current/accelerated_10pct_20260828"
INDEX_ROOT="${ROOT}/output_current/nested_budget_control_20260823/indices"
mkdir -p "${OUT}/logs"

run_isic() {
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${ROOT}/run_semisup_manifest.py" \
    --dataset isic --labeled-ratio 0.10 --gpu 0 \
    --output-dir "${OUT}/isic_seed3407" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed 3407 \
    --prompt-backend ifp --labeled-prompt-mode gt \
    --labeled-batch-probability 0.7 --learning-rate 2e-4 \
    --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --train-prompt-dropout 0 \
    --labeled-indices-file "${INDEX_ROOT}/isic/seed3407/labeled_10pct.json" \
    --support-indices-file "${INDEX_ROOT}/isic/seed3407/support_10pct.json" \
    >"${OUT}/logs/isic_seed3407.log" 2>&1
}

run_polyp() {
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${ROOT}/run_polyp_manifest.py" \
    --labeled-ratio 0.10 --gpu 1 \
    --output-dir "${OUT}/polyp_seed3407" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed 3407 \
    --prompt-backend ifp --labeled-prompt-mode gt \
    --labeled-batch-probability 0.7 --learning-rate 2e-4 \
    --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --train-prompt-dropout 0 \
    --labeled-indices-file "${INDEX_ROOT}/polyp/seed3407/labeled_10pct.json" \
    --support-indices-file "${INDEX_ROOT}/polyp/seed3407/support_10pct.json" \
    >"${OUT}/logs/polyp_seed3407.log" 2>&1
}

case "${1:-all}" in
  isic) run_isic ;;
  polyp) run_polyp ;;
  all)
    run_isic & isic_pid=$!
    run_polyp & polyp_pid=$!
    echo "ISIC GPU0 worker PID=${isic_pid}"
    echo "Polyp GPU1 worker PID=${polyp_pid}"
    wait "${isic_pid}" "${polyp_pid}"
    ;;
  *) echo "usage: $0 [isic|polyp|all]" >&2; exit 2 ;;
esac
