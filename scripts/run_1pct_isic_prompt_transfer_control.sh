#!/usr/bin/env bash
set -uo pipefail
ROOT="/datanas01/nas01/Student-home/2024U/YBC/wesam2"
PYTHON="/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUT="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/prompt_transfer_control"
mkdir -p "$OUT/logs"
cd "$ROOT"
WESAM="$ROOT/output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_isic_wesam"
NOP="$ROOT/output_current/fair_ablation_1pct_1shot_seed1337/group2/1pct_isic_no_prompt"
HEAD="$WESAM/ifp_alignment/ifp_medical_foreground_best.pt"

# A: train-with-prompt model, evaluated WITHOUT prompt
env PYTHONUNBUFFERED=1 "$PYTHON" export_test_predictions.py --dataset isic \
  --experiment-dir "$WESAM" --prompt-backend no-prompt --device cuda --batch-size 16 \
  --result-dir "$OUT/wesam_ckpt_none" > "$OUT/logs/wesam_ckpt_none.log" 2>&1
echo "A done $?"

# B: train-without-prompt model, evaluated WITH the same IFP head
env PYTHONUNBUFFERED=1 "$PYTHON" export_test_predictions.py --dataset isic \
  --experiment-dir "$NOP" --prompt-backend ifp --prompt-checkpoint "$HEAD" --device cuda --batch-size 16 \
  --result-dir "$OUT/noprompt_ckpt_ifp" > "$OUT/logs/noprompt_ckpt_ifp.log" 2>&1
echo "B done $?"

# C: reproduce reference (WeSAM ckpt + ifp)
env PYTHONUNBUFFERED=1 "$PYTHON" export_test_predictions.py --dataset isic \
  --experiment-dir "$WESAM" --prompt-backend ifp --device cuda --batch-size 16 \
  --result-dir "$OUT/wesam_ckpt_ifp" > "$OUT/logs/wesam_ckpt_ifp.log" 2>&1
echo "C done $?"
echo "ALL DONE"
