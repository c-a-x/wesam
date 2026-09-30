#!/usr/bin/env bash
set -u

# Queue the remaining ISIC/Polyp ablations on two GPUs.
# Each worker runs one experiment at a time and never overlaps tasks on its GPU.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="${ROOT}/output_current/remaining_experiments"
mkdir -p "${OUT}/logs"

run_isic() {
  local gpu="$1" seed="$2" dropout="$3" name="$4" anchor="$5" contrast="$6"
  mkdir -p "${OUT}/${name}"
  "${PYTHON}" "${ROOT}/run_semisup_manifest.py" \
    --dataset isic --labeled-ratio 0.01 --gpu "${gpu}" \
    --output-dir "${OUT}/${name}" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed "${seed}" \
    --prompt-backend ifp --labeled-prompt-mode mixed \
    --labeled-gt-prompt-probability 0.5 \
    --train-prompt-dropout "${dropout}" \
    --teacher-weight 0.1 --anchor-weight "${anchor}" \
    --contrast-weight "${contrast}" \
    >"${OUT}/logs/${name}.log" 2>&1
}

run_polyp() {
  local gpu="$1" seed="$2" dropout="$3" name="$4" anchor="$5" contrast="$6"
  mkdir -p "${OUT}/${name}"
  "${PYTHON}" "${ROOT}/run_polyp_manifest.py" \
    --labeled-ratio 0.01 --gpu "${gpu}" \
    --output-dir "${OUT}/${name}" --epochs 10 --alignment-epochs 1 \
    --batch-size 4 --val-batch-size 16 --seed "${seed}" \
    --prompt-backend ifp --labeled-prompt-mode mixed \
    --labeled-gt-prompt-probability 0.5 \
    --train-prompt-dropout "${dropout}" \
    --teacher-weight 0.1 --anchor-weight "${anchor}" \
    --contrast-weight "${contrast}" \
    >"${OUT}/logs/${name}.log" 2>&1
}

worker0() {
  # Strict ISIC dropout control, second seed, and Polyp dropout control.
  run_isic 0 1337 0.0 isic_mixed_dropout0_seed1337 0 0
  run_isic 0 2027 0.1 isic_mixed_dropout01_seed2027 0 0
  run_polyp 0 1337 0.0 polyp_mixed_dropout0_seed1337 0 0
  run_isic 0 1337 0.1 isic_mixed_dropout01_anchor_seed1337 1 0
}

worker1() {
  # Third ISIC seed, Polyp dropout control, and contrastive-loss ablation.
  run_isic 1 3407 0.1 isic_mixed_dropout01_seed3407 0 0
  run_polyp 1 1337 0.05 polyp_mixed_dropout005_seed1337 0 0
  run_isic 1 1337 0.1 isic_mixed_dropout01_contrast_seed1337 0 1
  run_isic 1 1337 0.1 isic_mixed_dropout01_anchor_contrast_seed1337 1 1
}

case "${1:-all}" in
  gpu0) worker0 ;;
  gpu1) worker1 ;;
  all)
    worker0 & p0=$!
    worker1 & p1=$!
    echo "GPU0 worker PID=${p0}"
    echo "GPU1 worker PID=${p1}"
    wait "${p0}" "${p1}"
    ;;
  *) echo "usage: $0 [gpu0|gpu1|all]" >&2; exit 2 ;;
esac
