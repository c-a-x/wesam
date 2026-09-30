#!/usr/bin/env bash
set -euo pipefail

ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
RUN_ROOT="$ROOT/output_current/kvasir"

if ! screen -ls | grep -q '[.]wesam_resume_watchdog'; then
  screen -dmS wesam_resume_watchdog bash -lc "cd '$ROOT' && exec bash scripts/watch_wesam_jobs.sh"
fi

screen -dmS wesam_kvasir_1pct_10pct bash -lc '
  set -euo pipefail
  cd "'$ROOT'"
  for ratio in 0.01 0.10; do
    if [ "$ratio" = "0.01" ]; then
      tag="1pct"
    else
      tag="10pct"
    fi
    out="'$RUN_ROOT'/${tag}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "'$PYTHON'" -u run_semisup_manifest.py --dataset kvasir --labeled-ratio "$ratio" --gpu 1 --output-dir "$out" --epochs 10 --batch-size 4 --val-batch-size 16 > "$out/train.log" 2>&1
  done
'

echo "Started screen session: wesam_kvasir_1pct_10pct"
echo "Logs: $RUN_ROOT/1pct/train.log and $RUN_ROOT/10pct/train.log"
