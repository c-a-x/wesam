#!/usr/bin/env bash
set -euo pipefail

project_root="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
source_dir="$project_root/output_current/isic/full_gt_gt_oracle"
target_dir="$project_root/output_current/point_pvt(gt)/ISIC/full_gt"
metrics_file="$source_dir/ISIC/metrics.csv"
log_file="$project_root/output_current/point_pvt(gt)/migration.log"

mkdir -p "$(dirname "$log_file")"

while true; do
    if [[ -f "$metrics_file" ]] && rg -q "final_test_best_student" "$metrics_file"; then
        if [[ -e "$target_dir" ]]; then
            printf '%s target already exists; no move performed\n' "$(date '+%F %T')" >> "$log_file"
            exit 1
        fi
        mkdir -p "$(dirname "$target_dir")"
        mv "$source_dir" "$target_dir"
        printf '%s moved completed ISIC GT-oracle results to %s\n' "$(date '+%F %T')" "$target_dir" >> "$log_file"
        exit 0
    fi
    sleep 60
done
