#!/usr/bin/env bash
set -uo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
RUN_ROOT="$ROOT/output_current/ablation_1shot_10pct_20260908"
STATUS="$RUN_ROOT/gpu0_status.log"
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

run_task isic_1shot_dino run_direct_sam2_prompt_ablation.py --dataset isic --gpu 0 --labeled-count 1 --prompt-backend dino-prototype --method-dir dino_only --output-dir "$RUN_ROOT"
run_task polyp_1shot_no_prompt run_polyp_manifest.py --gpu 0 --labeled-count 1 --prompt-backend no-prompt --output-dir "$RUN_ROOT/no_prompt/Polyp/1shot" --epochs 10 --batch-size 4 --val-batch-size 16 --seed 1337
run_task polyp_1shot_dino run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 0 --labeled-count 1 --prompt-backend dino-prototype --method-dir dino_only --output-dir "$RUN_ROOT"
run_task isic_10pct_no_prompt run_semisup_manifest.py --dataset isic --gpu 0 --labeled-ratio 0.10 --prompt-backend no-prompt --output-dir "$RUN_ROOT/no_prompt/ISIC/10pct" --epochs 10 --batch-size 4 --val-batch-size 16 --seed 1337
run_task isic_10pct_dino run_direct_sam2_prompt_ablation.py --dataset isic --gpu 0 --labeled-ratio 0.10 --prompt-backend dino-prototype --method-dir dino_only --output-dir "$RUN_ROOT"
run_task polyp_10pct_no_semisupervised run_ifp_supervised_ablation.py --dataset polyp --gpu 0 --labeled-count 130 --output-dir "$RUN_ROOT/no_semisupervised/Polyp/10pct" --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 --seed 1337
run_task polyp_10pct_dino run_direct_sam2_prompt_ablation.py --dataset polyp --gpu 0 --labeled-ratio 0.10 --prompt-backend dino-prototype --method-dir dino_only --output-dir "$RUN_ROOT"
