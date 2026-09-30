#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
RUN_ROOT="$ROOT/output_current/ablation_1shot_10pct_20260908"
STATUS="$RUN_ROOT/gpu1_status.log"
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

run_task() {
    local name="$1"
    shift
    printf '%s START %s\n' "$(date '+%F %T')" "$name" | tee -a "$STATUS"
    if env PYTHONUNBUFFERED=1 "$PYTHON" -u "$@" > "$RUN_ROOT/logs/${name}.log" 2>&1; then
        printf '%s DONE %s\n' "$(date '+%F %T')" "$name" | tee -a "$STATUS"
    else
        local code=$?
        printf '%s FAILED(%s) %s\n' "$(date '+%F %T')" "$code" "$name" | tee -a "$STATUS"
    fi
}

run_task isic_1shot_clip run_direct_sam2_prompt_ablation.py --dataset isic --gpu 1 --labeled-count 1 --prompt-backend clip-only --method-dir clip_only --output-dir "$RUN_ROOT"
run_task polyp_1shot_no_semisupervised run_ifp_supervised_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --output-dir "$RUN_ROOT/no_semisupervised/Polyp/1shot" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337
run_task polyp_1shot_clip run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-count 1 --prompt-backend clip-only --method-dir clip_only --output-dir "$RUN_ROOT"
run_task isic_10pct_no_semisupervised run_ifp_supervised_ablation.py --dataset isic --gpu 1 --labeled-count 259 --output-dir "$RUN_ROOT/no_semisupervised/ISIC/10pct" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337
run_task isic_10pct_clip run_direct_sam2_prompt_ablation.py --dataset isic --gpu 1 --labeled-ratio 0.10 --prompt-backend clip-only --method-dir clip_only --output-dir "$RUN_ROOT"
run_task polyp_10pct_clip run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 1 --labeled-ratio 0.10 --prompt-backend clip-only --method-dir clip_only --output-dir "$RUN_ROOT"
