#!/usr/bin/env bash
set -euo pipefail

# Some shared-machine process managers can suspend a long-running worker with
# SIGSTOP. Keep the two requested experiment workers runnable without
# touching unrelated processes.
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
LOG="$ROOT/output_current/watchdog.log"
ISIC_10_METRICS="$ROOT/output_current/isic/10pct/ISIC/metrics.csv"
ISIC_1_OUT="$ROOT/output_current/isic/1pct"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"

mkdir -p "$(dirname "$LOG")"
while true; do
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    # A worker can finish between pgrep and ps; that is normal and must not
    # terminate the watchdog.
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    case "$state" in
      T*)
        kill -CONT "$pid" 2>/dev/null || true
        printf '%(%F %T)T resumed PID %s (%s)\n' -1 "$pid" "$state" >> "$LOG"
        ;;
    esac
  done < <(pgrep -f 'run_semisup_manifest.py|run_polyp_manifest.py|reevaluate_polyp_logits.py|ifp_alignment.train_medical' || true)

  # Launch the requested corrected ISIC 1% run only after ISIC 10% has
  # completed its final test and released GPU 0. The marker prevents retries
  # after an ordinary training failure, which would overwrite evidence.
  if [ ! -e "$ISIC_1_OUT/.started_after_isic_10pct" ] \
    && [ -f "$ISIC_10_METRICS" ] \
    && rg -q 'isic_0.1gt_seed1337_final_test_best_student' "$ISIC_10_METRICS" \
    && ! pgrep -f '[r]un_semisup_manifest.py --dataset isic --labeled-ratio 0.10' >/dev/null \
    && ! pgrep -f '[r]un_semisup_manifest.py --dataset isic --labeled-ratio 0.01' >/dev/null; then
    mkdir -p "$ISIC_1_OUT"
    touch "$ISIC_1_OUT/.started_after_isic_10pct"
    printf '%(%F %T)T launching corrected ISIC 1%% after ISIC 10%% completion\n' -1 >> "$ISIC_1_OUT/queue.log"
    (
      cd "$ROOT"
      exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u run_semisup_manifest.py \
        --dataset isic --labeled-ratio 0.01 --gpu 0 --output-dir "$ISIC_1_OUT" \
        --epochs 10 --alignment-epochs 1 --batch-size 4 --val-batch-size 16 \
        > "$ISIC_1_OUT/train.log" 2>&1
    ) &
  fi
  sleep 1
done
