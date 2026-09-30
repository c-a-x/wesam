#!/usr/bin/env bash
set -u -o pipefail

# Polyp experiment queue. It is intentionally empty until a candidate passes
# a historical-result preflight; this prevents speculative GPU runs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${WESAM_PYTHON:-/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python}"
OUT_ROOT="${WESAM_POLYP_SWEEP_OUT:-${ROOT}/output_current/polyp_generalization_sweep_20260829}"
INDEX_ROOT="${ROOT}/output_current/nested_budget_control_20260823/indices/polyp"

# name|labeled-prompt-mode|labeled-gt-probability|prompt-dropout|teacher-weight
TASK_SPECS=()
SEEDS=(1337 2027 3407)

mkdir -p "${OUT_ROOT}/logs"

write_info() {
  local info="${OUT_ROOT}/EXPERIMENT_INFO.txt"
  if [[ -s "${info}" ]] && rg -q 'No tasks scheduled' "${info}"; then
    return
  fi
  {
    printf '%s\n' 'Polyp generalization sweep (2026-08-29)'
    printf '%s\n' 'Training: Kvasir + CVC-ClinicDB; testing: five Polyp datasets.'
    printf '%s\n' 'All tasks use the fixed nested 10% labeled/support manifests.'
    printf '%s\n' 'Common settings: 10 epochs, batch size 4, lr 2e-4, warmup 100,'
    printf '%s\n' 'unsupervised rampup 1, anchor 0, contrastive 0, IFP backend.'
    printf '%s\n' 'No tasks scheduled: historical results do not justify the current candidate.'
    printf '%s\n' 'The former domain-balanced candidate was withdrawn before training.'
  } > "${info}"
}

completed() {
  local out_dir="$1"
  local metrics="${out_dir}/Polyp/metrics.csv"
  local tests="${out_dir}/test_metrics.csv"
  [[ -s "${metrics}" ]] || return 1
  [[ -s "${tests}" ]] || return 1
  rg -q 'final_test_best_student' "${metrics}" || return 1
  [[ "$(wc -l < "${tests}")" -eq 6 ]]
}

run_one() {
  local gpu="$1"
  local seed="$2"
  local spec="$3"
  local name mode gt_probability dropout teacher_weight
  IFS='|' read -r name mode gt_probability dropout teacher_weight <<< "${spec}"

  local out_dir="${OUT_ROOT}/${name}/seed${seed}"
  local log_path="${OUT_ROOT}/logs/${name}_seed${seed}.log"
  local index_dir="${INDEX_ROOT}/seed${seed}"
  local running="${out_dir}/.running"

  if completed "${out_dir}"; then
    printf 'GPU %s: skip completed %s seed=%s\n' "${gpu}" "${name}" "${seed}"
    return 0
  fi

  mkdir -p "${out_dir}"
  if ! mkdir "${running}" 2>/dev/null; then
    printf 'GPU %s: skip already-running %s seed=%s\n' "${gpu}" "${name}" "${seed}"
    return 0
  fi

  printf 'GPU %s: start %s seed=%s\n' "${gpu}" "${name}" "${seed}"
  "${PYTHON}" -u "${ROOT}/run_polyp_manifest.py" \
    --labeled-count 130 --gpu "${gpu}" \
    --output-dir "${out_dir}" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed "${seed}" \
    --labeled-batch-probability 0.5 \
    --prompt-backend ifp --labeled-prompt-mode "${mode}" \
    --labeled-gt-prompt-probability "${gt_probability}" \
    --teacher-weight "${teacher_weight}" \
    --anchor-weight 0 --contrast-weight 0 \
    --learning-rate 2e-4 --warmup-steps 100 \
    --unsupervised-rampup-epochs 1 \
    --train-prompt-dropout "${dropout}" \
    --labeled-indices-file "${index_dir}/labeled_10pct.json" \
    --support-indices-file "${index_dir}/support_10pct.json" \
    >"${log_path}" 2>&1
  local status=$?
  rmdir "${running}" 2>/dev/null || true
  if [[ "${status}" -eq 0 ]] && completed "${out_dir}"; then
    printf 'GPU %s: finished %s seed=%s\n' "${gpu}" "${name}" "${seed}"
    return 0
  fi
  printf 'GPU %s: FAILED %s seed=%s (exit=%s), see %s\n' \
    "${gpu}" "${name}" "${seed}" "${status}" "${log_path}" >&2
  return 1
}

run_worker() {
  local gpu="$1"
  local dry_run="$2"
  local index=0
  local failures=()
  local seed spec name mode gt_probability dropout teacher_weight

  for seed in "${SEEDS[@]}"; do
    for spec in "${TASK_SPECS[@]}"; do
      if (( index % 2 == gpu )); then
        IFS='|' read -r name mode gt_probability dropout teacher_weight <<< "${spec}"
        if [[ "${dry_run}" == '1' ]]; then
          printf 'GPU %s: would run %s seed=%s mode=%s dropout=%s teacher=%s\n' \
            "${gpu}" "${name}" "${seed}" "${mode}" "${dropout}" "${teacher_weight}"
        elif ! run_one "${gpu}" "${seed}" "${spec}"; then
          failures+=("${name}/seed${seed}")
        fi
      fi
      index=$((index + 1))
    done
  done

  if ((${#failures[@]} > 0)); then
    printf 'GPU %s failures: %s\n' "${gpu}" "${failures[*]}" >&2
    return 1
  fi
  return 0
}

write_info
mode="all"
dry_run=0
rerun=0
for arg in "$@"; do
  case "${arg}" in
    all|gpu0|gpu1) mode="${arg}" ;;
    --dry-run) dry_run=1 ;;
    --rerun-completed) rerun=1 ;;
    *) printf 'usage: %s [all|gpu0|gpu1] [--dry-run] [--rerun-completed]\n' "$0" >&2; exit 2 ;;
  esac
done

if [[ "${rerun}" -eq 1 ]]; then
  # A rerun uses fresh output rather than overwriting old checkpoints.
  OUT_ROOT="${OUT_ROOT}_rerun"
  mkdir -p "${OUT_ROOT}/logs"
fi

if ((${#TASK_SPECS[@]} == 0)); then
  printf '%s\n' 'No tasks scheduled; no training process was started.'
  exit 0
fi

case "${mode}" in
  gpu0) run_worker 0 "${dry_run}" ;;
  gpu1) run_worker 1 "${dry_run}" ;;
  all)
    run_worker 0 "${dry_run}" & pid0=$!
    run_worker 1 "${dry_run}" & pid1=$!
    printf 'GPU0 worker PID=%s\n' "${pid0}"
    printf 'GPU1 worker PID=%s\n' "${pid1}"
    wait "${pid0}"; status0=$?
    wait "${pid1}"; status1=$?
    [[ "${status0}" -eq 0 && "${status1}" -eq 0 ]]
    ;;
esac
