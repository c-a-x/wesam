#!/usr/bin/env bash

# Run the GT-mask-loss ablation only after GPU 0 has been idle long enough.
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

printf '%s starting no-GT-mask-loss diagnostic\n' "$(date '+%F %T')"
"$python_bin" -u run_semisup_manifest.py \
    --dataset isic \
    --labeled-ratio 0.01 \
    --gpu 0 \
    --output-dir output_current/prompt_diagnostics_20260907/isic_1pct_no_gt_mask_loss_p01_seed3407 \
    --epochs 10 \
    --batch-size 4 \
    --val-batch-size 16 \
    --seed 3407 \
    --labeled-batch-probability 0.1 \
    --prompt-backend ifp \
    --labeled-prompt-mode gt \
    --labeled-gt-prompt-probability 1 \
    --teacher-weight 0.1 \
    --anchor-weight 0 \
    --contrast-weight 0 \
    --supervised-weight 0
run_rc=$?
printf '%s no-GT-mask-loss diagnostic exit=%s\n' "$(date '+%F %T')" "$run_rc"
exit "$run_rc"
