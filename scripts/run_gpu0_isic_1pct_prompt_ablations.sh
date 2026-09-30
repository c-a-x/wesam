#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON_BIN="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/output_current"

run_experiment() {
    local method_dir="$1"
    local backend="$2"
    local labeled_prompt_mode="$3"
    local output_dir="$OUTPUT_ROOT/$method_dir/ISIC/1pct"
    local log_dir="$OUTPUT_ROOT/$method_dir/logs"

    mkdir -p "$output_dir" "$log_dir"
    "$PYTHON_BIN" -u "$PROJECT_ROOT/run_semisup_manifest.py" \
        --dataset isic --labeled-ratio 0.01 --gpu 0 --output-dir "$output_dir" \
        --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337 \
        --prompt-backend "$backend" --labeled-prompt-mode "$labeled_prompt_mode" \
        2>&1 | tee "$log_dir/ISIC_1pct_${method_dir}.log"
}

run_experiment "clip_semisupervised" "clip-only" "gt"
run_experiment "dino_semisupervised" "dino-prototype" "gt"
run_experiment "no_prompt" "no-prompt" "gt"
run_experiment "point_p(gt)_vt(ifp)" "ifp" "gt"
