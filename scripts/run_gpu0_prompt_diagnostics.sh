#!/usr/bin/env bash

# Start only after GPU 0 has been genuinely idle for three consecutive checks.
set -u

repo_dir=/datanas01/nas01/Student-home/2024U/YBC/wesam2
python_bin=/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python
cd "$repo_dir"

idle_count=0
while true; do
    gpu_state=$(nvidia-smi --id=0 --query-gpu=utilization.gpu,memory.free \
        --format=csv,noheader,nounits | tr -d ' ')
    gpu_util=${gpu_state%%,*}
    gpu_free=${gpu_state##*,}
    printf '%s gpu0_util=%s%% gpu0_free=%sMiB idle_samples=%s\n' \
        "$(date '+%F %T')" "$gpu_util" "$gpu_free" "$idle_count"

    if [ "$gpu_util" -le 5 ] && [ "$gpu_free" -ge 70000 ]; then
        idle_count=$((idle_count + 1))
    else
        idle_count=0
    fi
    if [ "$idle_count" -ge 3 ]; then
        break
    fi
    sleep 60
done

printf '%s starting GT-oracle inference\n' "$(date '+%F %T')"
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u export_test_predictions.py \
    --dataset isic \
    --experiment-dir output_current/wesam/ISIC/1pct \
    --device cuda \
    --batch-size 16 \
    --prompt-backend gt-oracle \
    --result-dir output_current/prompt_policy_control/ISIC_1pct_gt_oracle
infer_rc=$?
printf '%s GT-oracle inference exit=%s\n' "$(date '+%F %T')" "$infer_rc"
if [ "$infer_rc" -ne 0 ]; then
    exit "$infer_rc"
fi

printf '%s starting labeled-IFP training\n' "$(date '+%F %T')"
"$python_bin" -u run_semisup_manifest.py \
    --dataset isic \
    --labeled-ratio 0.01 \
    --gpu 0 \
    --output-dir output_current/prompt_diagnostics_20260907/isic_1pct_labeled_ifp_p01_seed3407 \
    --epochs 10 \
    --batch-size 4 \
    --val-batch-size 16 \
    --seed 3407 \
    --labeled-batch-probability 0.1 \
    --prompt-backend ifp \
    --labeled-prompt-mode ifp \
    --labeled-gt-prompt-probability 0 \
    --teacher-weight 0.1 \
    --anchor-weight 0 \
    --contrast-weight 0
train_rc=$?
printf '%s labeled-IFP training exit=%s\n' "$(date '+%F %T')" "$train_rc"
exit "$train_rc"
