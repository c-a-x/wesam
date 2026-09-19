#!/usr/bin/env bash
set -euo pipefail

# Fair 10%% labeled comparison used by ablation_group{1,2}_10pct.tex.
#
# All runs use the same fixed seed=1337 labeled/support indices.  Group 1
# evaluates frozen SAM2 prompt variants; group 2 trains the same protocol and
# changes only the prompt/semi-supervision setting.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
OUT="${ROOT}/output_current/fair_ablation_10pct_seed1337"
GROUP1="${OUT}/group1"
GROUP2="${OUT}/group2"
INDEX_ROOT="${ROOT}/output_current/nested_budget_control_20260823/indices"
ISIC_LABELED="${INDEX_ROOT}/isic/seed1337/labeled_10pct.json"
ISIC_SUPPORT="${INDEX_ROOT}/isic/seed1337/support_10pct.json"
POLYP_LABELED="${INDEX_ROOT}/polyp/seed1337/labeled_10pct.json"
POLYP_SUPPORT="${INDEX_ROOT}/polyp/seed1337/support_10pct.json"
mkdir -p "${OUT}/logs"
cd "${ROOT}"

require_files() {
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing required file: ${path}" >&2
      exit 2
    fi
  done
}

run_logged() {
  local log_name="$1"
  shift
  echo "[$(date '+%F %T')] START ${log_name}" | tee -a "${OUT}/status.log"
  if "$@" >"${OUT}/logs/${log_name}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  ${log_name}" | tee -a "${OUT}/status.log"
  else
    local code=$?
    echo "[$(date '+%F %T')] FAILED(${code}) ${log_name}" | tee -a "${OUT}/status.log"
    return "${code}"
  fi
}

run_group1_isic() {
  run_logged group1_isic_clip "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset isic --gpu 0 --labeled-count 259 --budget-tag 10pct --prompt-backend clip-only \
    --method-dir clip_only --output-dir "${GROUP1}" \
    --seed 1337 --batch-size 16 \
    --labeled-indices-file "${ISIC_LABELED}" \
    --support-indices-file "${ISIC_SUPPORT}"

  run_logged group1_isic_dino "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset isic --gpu 0 --labeled-count 259 --budget-tag 10pct --prompt-backend dino-prototype \
    --method-dir dino_only --output-dir "${GROUP1}" \
    --seed 1337 --batch-size 16 \
    --labeled-indices-file "${ISIC_LABELED}" \
    --support-indices-file "${ISIC_SUPPORT}"

  run_logged group1_isic_ifp "${PYTHON}" run_ifp_direct_sam2_ablation.py \
    --dataset isic --gpu 0 --labeled-count 259 --budget-tag 10pct \
    --output-dir "${GROUP1}" --seed 1337 --batch-size 16 \
    --alignment-epochs 1 \
    --labeled-indices-file "${ISIC_LABELED}" \
    --support-indices-file "${ISIC_SUPPORT}"
}

run_group1_polyp() {
  run_logged group1_polyp_clip "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset polyp --gpu 1 --labeled-count 130 --budget-tag 10pct --prompt-backend clip-only \
    --method-dir clip_only --output-dir "${GROUP1}" \
    --seed 1337 --batch-size 16 \
    --labeled-indices-file "${POLYP_LABELED}" \
    --support-indices-file "${POLYP_SUPPORT}"

  run_logged group1_polyp_dino "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset polyp --gpu 1 --labeled-count 130 --budget-tag 10pct --prompt-backend dino-prototype \
    --method-dir dino_only --output-dir "${GROUP1}" \
    --seed 1337 --batch-size 16 \
    --labeled-indices-file "${POLYP_LABELED}" \
    --support-indices-file "${POLYP_SUPPORT}"

  run_logged group1_polyp_ifp "${PYTHON}" run_ifp_direct_sam2_ablation.py \
    --dataset polyp --gpu 1 --labeled-count 130 --budget-tag 10pct \
    --output-dir "${GROUP1}" --seed 1337 --batch-size 16 \
    --alignment-epochs 1 \
    --labeled-indices-file "${POLYP_LABELED}" \
    --support-indices-file "${POLYP_SUPPORT}"
}

run_group2_isic() {
  local common=(
    --dataset isic --gpu 0 --labeled-count 259
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16
    --seed 1337 --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0
    --labeled-batch-probability 0.7 --learning-rate 2e-4 --warmup-steps 100
    --unsupervised-rampup-epochs 1 --anchor-weight 0 --contrast-weight 0
    --train-point-jitter-pixels 0 --train-prompt-dropout 0
    --labeled-indices-file "${ISIC_LABELED}" --support-indices-file "${ISIC_SUPPORT}"
  )
  run_logged group2_isic_no_prompt "${PYTHON}" run_semisup_manifest.py \
    "${common[@]}" --prompt-backend no-prompt --teacher-weight 0.1 \
    --output-dir "${GROUP2}/isic_no_prompt"
  run_logged group2_isic_wesam "${PYTHON}" run_semisup_manifest.py \
    "${common[@]}" --prompt-backend ifp --teacher-weight 0.1 \
    --output-dir "${GROUP2}/isic_wesam"
}

run_group2_polyp() {
  local common=(
    --gpu 1 --labeled-count 130
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16
    --seed 1337 --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0
    --labeled-batch-probability 0.7 --learning-rate 2e-4 --warmup-steps 100
    --unsupervised-rampup-epochs 1 --anchor-weight 0 --contrast-weight 0
    --train-point-jitter-pixels 0 --train-prompt-dropout 0
    --labeled-indices-file "${POLYP_LABELED}" --support-indices-file "${POLYP_SUPPORT}"
  )
  run_logged group2_polyp_no_prompt "${PYTHON}" run_polyp_manifest.py \
    "${common[@]}" --prompt-backend no-prompt --teacher-weight 0.1 \
    --output-dir "${GROUP2}/polyp_no_prompt"
  run_logged group2_polyp_wesam "${PYTHON}" run_polyp_manifest.py \
    "${common[@]}" --prompt-backend ifp --teacher-weight 0.1 \
    --output-dir "${GROUP2}/polyp_wesam"
}

run_group1() {
  require_files "${ISIC_LABELED}" "${ISIC_SUPPORT}" "${POLYP_LABELED}" "${POLYP_SUPPORT}"
  run_group1_isic & local p0=$!
  run_group1_polyp & local p1=$!
  wait "${p0}" "${p1}"
}

run_group2() {
  require_files "${ISIC_LABELED}" "${ISIC_SUPPORT}" "${POLYP_LABELED}" "${POLYP_SUPPORT}"
  run_group2_isic & local p0=$!
  run_group2_polyp & local p1=$!
  wait "${p0}" "${p1}"
}

queue_gpu0() {
  require_files "${ISIC_LABELED}" "${ISIC_SUPPORT}" "${POLYP_LABELED}" "${POLYP_SUPPORT}"
  run_logged group1_isic_clip "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset isic --gpu 0 --labeled-count 259 --budget-tag 10pct --prompt-backend clip-only \
    --method-dir clip_only --output-dir "${GROUP1}" --seed 1337 --batch-size 16 \
    --labeled-indices-file "${ISIC_LABELED}" --support-indices-file "${ISIC_SUPPORT}"
  run_logged group1_isic_dino "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset isic --gpu 0 --labeled-count 259 --budget-tag 10pct --prompt-backend dino-prototype \
    --method-dir dino_only --output-dir "${GROUP1}" --seed 1337 --batch-size 16 \
    --labeled-indices-file "${ISIC_LABELED}" --support-indices-file "${ISIC_SUPPORT}"
  run_logged group1_isic_ifp "${PYTHON}" run_ifp_direct_sam2_ablation.py \
    --dataset isic --gpu 0 --labeled-count 259 --budget-tag 10pct \
    --output-dir "${GROUP1}" --seed 1337 --batch-size 16 --alignment-epochs 1 \
    --labeled-indices-file "${ISIC_LABELED}" --support-indices-file "${ISIC_SUPPORT}"
  run_logged group1_polyp_clip "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset polyp --gpu 0 --labeled-count 130 --budget-tag 10pct --prompt-backend clip-only \
    --method-dir clip_only --output-dir "${GROUP1}" --seed 1337 --batch-size 16 \
    --labeled-indices-file "${POLYP_LABELED}" --support-indices-file "${POLYP_SUPPORT}"
  run_logged group1_polyp_dino "${PYTHON}" run_direct_sam2_prompt_ablation.py \
    --dataset polyp --gpu 0 --labeled-count 130 --budget-tag 10pct --prompt-backend dino-prototype \
    --method-dir dino_only --output-dir "${GROUP1}" --seed 1337 --batch-size 16 \
    --labeled-indices-file "${POLYP_LABELED}" --support-indices-file "${POLYP_SUPPORT}"
}

queue_gpu1() {
  require_files "${ISIC_LABELED}" "${ISIC_SUPPORT}" "${POLYP_LABELED}" "${POLYP_SUPPORT}"
  run_logged group1_polyp_ifp "${PYTHON}" run_ifp_direct_sam2_ablation.py \
    --dataset polyp --gpu 1 --labeled-count 130 --budget-tag 10pct \
    --output-dir "${GROUP1}" --seed 1337 --batch-size 16 --alignment-epochs 1 \
    --labeled-indices-file "${POLYP_LABELED}" --support-indices-file "${POLYP_SUPPORT}"
  run_logged group2_isic_no_prompt "${PYTHON}" run_semisup_manifest.py \
    --dataset isic --gpu 1 --labeled-count 259 --output-dir "${GROUP2}/isic_no_prompt" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend no-prompt --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "${ISIC_LABELED}" --support-indices-file "${ISIC_SUPPORT}"
  run_logged group2_isic_wesam "${PYTHON}" run_semisup_manifest.py \
    --dataset isic --gpu 1 --labeled-count 259 --output-dir "${GROUP2}/isic_wesam" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "${ISIC_LABELED}" --support-indices-file "${ISIC_SUPPORT}"
  run_logged group2_polyp_no_prompt "${PYTHON}" run_polyp_manifest.py \
    --gpu 1 --labeled-count 130 --output-dir "${GROUP2}/polyp_no_prompt" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend no-prompt --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "${POLYP_LABELED}" --support-indices-file "${POLYP_SUPPORT}"
  run_logged group2_polyp_wesam "${PYTHON}" run_polyp_manifest.py \
    --gpu 1 --labeled-count 130 --output-dir "${GROUP2}/polyp_wesam" \
    --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
    --prompt-backend ifp --labeled-prompt-mode gt --labeled-gt-prompt-probability 1.0 \
    --labeled-batch-probability 0.7 --teacher-weight 0.1 --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 --unsupervised-rampup-epochs 1 \
    --train-point-jitter-pixels 0 --train-prompt-dropout 0 \
    --labeled-indices-file "${POLYP_LABELED}" --support-indices-file "${POLYP_SUPPORT}"
}

case "${1:-all}" in
  group1) run_group1 ;;
  group2) run_group2 ;;
  queue0) queue_gpu0 ;;
  queue1) queue_gpu1 ;;
  all)
    run_group1
    run_group2
    ;;
  *)
    echo "usage: $0 [group1|group2|all]" >&2
    exit 2
    ;;
esac
