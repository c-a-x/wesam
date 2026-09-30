#!/usr/bin/env bash
# 等 GPU1 跑完 1pct_polyp_no_prompt 后，终止 GPU1 launcher（3107850），
# 避免它再启动 1shot_isic_wesam / 1shot_polyp_no_prompt（已由 GPU0 承担）。
OUT="/datanas01/nas01/Student-home/2024U/YBC/wesam2/output_current/fair_ablation_1pct_1shot_seed1337"
LOG="$OUT/logs/watchdog.log"
while true; do
  if grep -q "DONE  1pct_polyp_no_prompt (gpu1)" "$OUT/status.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] watchdog: 1pct_polyp_no_prompt DONE, killing GPU1 launcher 3107850" >> "$LOG"
    kill -9 3107850 2>/dev/null
    # 兜底：杀掉 GPU1 若已启动的重复任务（gpu=1 版）
    for p in $(pgrep -f "run_semisup_manifest.py --dataset isic --gpu 1"); do kill -9 "$p" 2>/dev/null; done
    for p in $(pgrep -f "run_polyp_manifest.py --gpu 1"); do kill -9 "$p" 2>/dev/null; done
    echo "[$(date '+%F %T')] watchdog: done" >> "$LOG"
    exit 0
  fi
  sleep 2
done
