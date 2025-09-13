#!/usr/bin/env bash
set -euo pipefail

REF_MODEL=${REF_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-1.5B}
OUT_DIR=${OUT_DIR:-assets/trained/dpo_webgpt_s500}
LOG_DIR=assets/experiments/logs
TRAIN_LOG=assets/trained/logs/dpo_webgpt_s500.log
PIDF=assets/trained/logs/dpo_webgpt_s500.pid

mkdir -p "$(dirname "$TRAIN_LOG")" "$LOG_DIR" "$OUT_DIR"

echo "[AUTO] Ensuring datasets<3 is installed..." | tee -a "$LOG_DIR/auto_webgpt.log"
python - << 'PY' 2>/dev/null || pip install -q 'datasets<3'
try:
    import datasets
    print('datasets version:', datasets.__version__)
except Exception as e:
    print('datasets import error:', e)
PY

echo "[AUTO] Waiting for openai/webgpt_comparisons to be loadable..." | tee -a "$LOG_DIR/auto_webgpt.log"
until python - << 'PY' >/dev/null 2>&1; do
from datasets import load_dataset
ds = load_dataset('openai/webgpt_comparisons', split='train[:2]')
assert len(ds) == 2
PY
  echo "[AUTO] Dataset not ready yet; retrying in 120s..." | tee -a "$LOG_DIR/auto_webgpt.log"
  sleep 120
done
echo "[AUTO] Dataset check passed." | tee -a "$LOG_DIR/auto_webgpt.log"

CMD=(accelerate launch --num_processes 8 --mixed_precision bf16 -m dpo_adl.cli train-dpo-hf \
  --ref-model "$REF_MODEL" \
  --dataset openai/webgpt_comparisons --split train \
  --n-pairs 2000 --max-steps 500 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --no-use-lora --save-steps 250 \
  --out-dir "$OUT_DIR")

echo "[AUTO] Launching 8-GPU DPO on WebGPT..." | tee -a "$LOG_DIR/auto_webgpt.log"
nohup "${CMD[@]}" > "$TRAIN_LOG" 2>&1 & echo $! > "$PIDF"
PID=$(cat "$PIDF")
echo "[AUTO] Training PID: $PID" | tee -a "$LOG_DIR/auto_webgpt.log"

echo "[AUTO] Waiting for training to finish..." | tee -a "$LOG_DIR/auto_webgpt.log"
while ps -p "$PID" > /dev/null 2>&1; do sleep 60; done
echo "[AUTO] Training process exited." | tee -a "$LOG_DIR/auto_webgpt.log"

echo "[AUTO] Running post-training analysis..." | tee -a "$LOG_DIR/auto_webgpt.log"
PYTHONPATH=. python -u -m dpo_adl.cli run-exp \
  --name EXP_DPO_WEBGPT_S500 \
  --ref-model "$REF_MODEL" \
  --dpo-model "$OUT_DIR" \
  --n-probe 2000 --k 5 --batch-size 32 \
  --alpha-sweep '0.25,0.5,0.75,1.0,1.25,1.5,2.0' \
  --prompts prompts/generic60.txt \
  --max-new-tokens 48 --temperature 0.0 \
  --embed-to dpo-chosen --embed-ds-name openai/webgpt_comparisons --embed-ds-split train --embed-ds-n 1200 \
  --delta-source dpo-chosen --delta-ds-name openai/webgpt_comparisons --delta-ds-split train --delta-ds-n 1200 \
  --select-by margin \
  --positions first_n --first-n 8 --alpha-decay 32 \
  --steering-norm-match --select-frac 0.5 \
  --pretty-plot --ban-punct --orthogonalize --base-model "$BASE_MODEL" --sentinel '?' \
  2>&1 | tee -a "$LOG_DIR/auto_webgpt.log"

echo "[AUTO] Running WebGPT ref→ref Δ sanity..." | tee -a "$LOG_DIR/auto_webgpt.log"
PYTHONPATH=. python assets/experiments/run_webgpt_ref_ref.py --model "$REF_MODEL" --dataset openai/webgpt_comparisons --split train --n 300 --k 5 --batch 32 \
  2>&1 | tee -a "$LOG_DIR/auto_webgpt.log"

echo "[AUTO] All steps completed." | tee -a "$LOG_DIR/auto_webgpt.log"

