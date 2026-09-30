#!/usr/bin/env bash
set -euo pipefail

# Evaluate mixed GT/IFP prompts with 0.1 prompt dropout on the exact ISIC
# 10% nested-budget subsets used by the original WeSAM control experiment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
SOURCE="${ROOT}/output_current/nested_budget_control_20260823"
OUT="${ROOT}/output_current/nested_budget_mixed_dropout01_20260826"

mkdir -p "${OUT}/logs"

run_seed() {
  local gpu="$1" seed="$2"
  local name="isic_seed${seed}_10pct_mixed_dropout01"
  local index_dir="${SOURCE}/indices/isic/seed${seed}"

  "${PYTHON}" "${ROOT}/run_semisup_manifest.py" \
    --dataset isic --labeled-ratio 0.10 --gpu "${gpu}" \
    --output-dir "${OUT}/${name}" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed "${seed}" \
    --prompt-backend ifp --labeled-prompt-mode mixed \
    --labeled-gt-prompt-probability 0.5 --train-prompt-dropout 0.1 \
    --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --labeled-indices-file "${index_dir}/labeled_10pct.json" \
    --support-indices-file "${index_dir}/support_10pct.json" \
    >"${OUT}/logs/${name}.log" 2>&1
}

gpu0_worker() {
  run_seed 0 1337
  run_seed 0 3407
}

gpu1_worker() {
  run_seed 1 2027
}

case "${1:-all}" in
  gpu0) gpu0_worker ;;
  gpu1) gpu1_worker ;;
  all)
    gpu0_worker & gpu0_pid=$!
    gpu1_worker & gpu1_pid=$!
    echo "GPU0 worker PID=${gpu0_pid}"
    echo "GPU1 worker PID=${gpu1_pid}"
    wait "${gpu0_pid}" "${gpu1_pid}"
    ;;
  *) echo "usage: $0 [gpu0|gpu1|all]" >&2; exit 2 ;;
esac
