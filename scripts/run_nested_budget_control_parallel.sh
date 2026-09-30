#!/usr/bin/env bash
set -euo pipefail

# Strict 1% vs 10% comparison: 1% is nested inside each seed's 10% labeled
# set, and both budgets use the same 10% support set to train IFP alignment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="${ROOT}/output_current/nested_budget_control_20260823"
INDEX_ROOT="${OUT}/indices"
mkdir -p "${OUT}/logs"
"${PYTHON}" "${ROOT}/scripts/prepare_nested_budget_indices.py" --output-dir "${INDEX_ROOT}"

run_isic() {
  local seed="$1" budget="$2"
  local ratio="0.01"
  [ "${budget}" = "10pct" ] && ratio="0.10"
  local name="isic_seed${seed}_${budget}"
  local index_dir="${INDEX_ROOT}/isic/seed${seed}"
  "${PYTHON}" "${ROOT}/run_semisup_manifest.py" \
    --dataset isic --labeled-ratio "${ratio}" --gpu 0 \
    --output-dir "${OUT}/${name}" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed "${seed}" \
    --prompt-backend ifp --labeled-prompt-mode gt \
    --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --train-prompt-dropout 0 \
    --labeled-indices-file "${index_dir}/labeled_${budget}.json" \
    --support-indices-file "${index_dir}/support_10pct.json" \
    >"${OUT}/logs/${name}.log" 2>&1
}

run_polyp() {
  local seed="$1" budget="$2"
  local ratio="0.01"
  [ "${budget}" = "10pct" ] && ratio="0.10"
  local name="polyp_seed${seed}_${budget}"
  local index_dir="${INDEX_ROOT}/polyp/seed${seed}"
  "${PYTHON}" "${ROOT}/run_polyp_manifest.py" \
    --labeled-ratio "${ratio}" --gpu 1 \
    --output-dir "${OUT}/${name}" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed "${seed}" \
    --prompt-backend ifp --labeled-prompt-mode gt \
    --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --train-prompt-dropout 0 \
    --labeled-indices-file "${index_dir}/labeled_${budget}.json" \
    --support-indices-file "${index_dir}/support_10pct.json" \
    >"${OUT}/logs/${name}.log" 2>&1
}

isic_worker() {
  for seed in 1337 2027 3407; do
    run_isic "${seed}" 1pct
    run_isic "${seed}" 10pct
  done
}

polyp_worker() {
  for seed in 1337 2027 3407; do
    run_polyp "${seed}" 1pct
    run_polyp "${seed}" 10pct
  done
}

case "${1:-all}" in
  isic) isic_worker ;;
  polyp) polyp_worker ;;
  all)
    isic_worker & isic_pid=$!
    polyp_worker & polyp_pid=$!
    echo "ISIC GPU0 worker PID=${isic_pid}"
    echo "Polyp GPU1 worker PID=${polyp_pid}"
    wait "${isic_pid}" "${polyp_pid}"
    ;;
  *) echo "usage: $0 [isic|polyp|all]" >&2; exit 2 ;;
esac
