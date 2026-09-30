#!/usr/bin/env bash
while true; do
  for p in $(pgrep -u 2024U -f "python|run_semisup_manifest|run_polyp_manifest|train_medical"); do
    if [ -f "/proc/$p/status" ]; then
      if grep -qs '^State:.*T' "/proc/$p/status"; then
        kill -CONT "$p" 2>/dev/null
      fi
    fi
  done
  sleep 2
done
