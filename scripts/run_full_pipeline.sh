#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[PIPELINE] Failed at $(date -Is)" | tee -a assets/experiments/logs/pipeline.log; exit 1' ERR

# Full pipeline: train (500 steps) and analyze for {DPO, ORPO, GRPO} × {HH-RLHF, WebGPT, Summarize}
# Uses 8 GPUs via accelerate; bf16 where available. Fail-fast on errors.

REF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL="Qwen/Qwen2.5-1.5B"
STEPS=500
PAIRS=2000
BSZ=4
GAS=1
LR=1e-5
TRAIN_TIMEOUT=7200   # 2h per training job
ANALYZE_TIMEOUT=3600 # 1h per analysis

mkdir -p assets/trained/logs assets/experiments/logs

run_dpo() {
  local DATASET="$1" SPLIT="$2" ODIR="$3"; shift 3; local EXTRA=("$@")
  echo "[DPO] Training $DATASET -> $ODIR" | tee -a assets/trained/logs/pipeline.log
  timeout "$TRAIN_TIMEOUT" accelerate launch --num_processes 8 --mixed_precision bf16 -m dpo_adl.cli train-dpo-hf \
    --ref-model "$REF_MODEL" \
    --dataset "$DATASET" --split "$SPLIT" "${EXTRA[@]}" \
    --n-pairs $PAIRS --max-steps $STEPS \
    --per-device-train-batch-size $BSZ --gradient-accumulation-steps $GAS \
    --learning-rate $LR --no-use-lora --save-steps 250 \
    --out-dir "$ODIR"
}

run_orpo() {
  echo "[ORPO] Disabled: no CLI support. Failing fast." | tee -a assets/trained/logs/pipeline.log
  return 1
}

run_grpo() {
  local DATASET="$1" SPLIT="$2" ODIR="$3"; shift 3; local EXTRA=("$@")
  echo "[GRPO] Training $DATASET -> $ODIR" | tee -a assets/trained/logs/pipeline.log
  timeout "$TRAIN_TIMEOUT" accelerate launch --num_processes 8 --mixed_precision bf16 -m dpo_adl.cli train-grpo-hf \
    --ref-model "$REF_MODEL" \
    --dataset "$DATASET" --split "$SPLIT" "${EXTRA[@]}" \
    --n-pairs $PAIRS --max-steps $STEPS \
    --per-device-train-batch-size $BSZ --gradient-accumulation-steps $GAS \
    --max-length 256 --max-prompt-length 128 \
    --learning-rate $LR --save-steps 250 \
    --out-dir "$ODIR"
}

analyze() {
  local NAME="$1" DPO_DIR="$2" EMBED_DS="$3" EMBED_SPLIT="$4"
  echo "[ANALYZE] $NAME using $DPO_DIR" | tee -a assets/experiments/logs/pipeline.log
  timeout "$ANALYZE_TIMEOUT" PYTHONPATH=. python -u -m dpo_adl.cli run-exp \
    --name "$NAME" \
    --ref-model "$REF_MODEL" \
    --dpo-model "$DPO_DIR" \
    --n-probe 2000 --k 5 --batch-size 32 \
    --alpha-sweep '0.25,0.5,0.75,1.0,1.25,1.5,2.0' \
    --prompts prompts/generic60.txt \
    --max-new-tokens 48 --temperature 0.0 \
    --embed-to dpo-chosen --embed-ds-name "$EMBED_DS" --embed-ds-split "$EMBED_SPLIT" --embed-ds-n 1200 \
    --delta-source dpo-chosen --delta-ds-name "$EMBED_DS" --delta-ds-split "$EMBED_SPLIT" --delta-ds-n 1200 \
    --select-by margin \
    --positions first_n --first-n 8 --alpha-decay 32 \
    --steering-norm-match --select-frac 0.5 \
    --pretty-plot --ban-punct --orthogonalize --base-model "$BASE_MODEL" --sentinel '?'
}

# 1) HH-RLHF (harmless-style)
run_dpo "Anthropic/hh-rlhf" train assets/trained/dpo_hh_s500
analyze EXP_DPO_HH_S500 assets/trained/dpo_hh_s500 Anthropic/hh-rlhf train

run_orpo "Anthropic/hh-rlhf" train assets/trained/orpo_hh_s500
analyze EXP_ORPO_HH_S500 assets/trained/orpo_hh_s500 Anthropic/hh-rlhf train

run_grpo "Anthropic/hh-rlhf" train assets/trained/grpo_hh_s500
analyze EXP_GRPO_HH_S500 assets/trained/grpo_hh_s500 Anthropic/hh-rlhf train

# 2) WebGPT
run_dpo openai/webgpt_comparisons train assets/trained/dpo_webgpt_s500
analyze EXP_DPO_WEBGPT_S500 assets/trained/dpo_webgpt_s500 openai/webgpt_comparisons train

run_orpo openai/webgpt_comparisons train assets/trained/orpo_webgpt_s500 || true
analyze EXP_ORPO_WEBGPT_S500 assets/trained/orpo_webgpt_s500 openai/webgpt_comparisons train || true

run_grpo openai/webgpt_comparisons train assets/trained/grpo_webgpt_s500
analyze EXP_GRPO_WEBGPT_S500 assets/trained/grpo_webgpt_s500 openai/webgpt_comparisons train

# 3) Summarize comparisons
run_dpo CarperAI/openai_summarize_comparisons train assets/trained/dpo_summarize_s500
analyze EXP_DPO_SUM_S500 assets/trained/dpo_summarize_s500 CarperAI/openai_summarize_comparisons train

run_orpo CarperAI/openai_summarize_comparisons train assets/trained/orpo_summarize_s500 || true
analyze EXP_ORPO_SUM_S500 assets/trained/orpo_summarize_s500 CarperAI/openai_summarize_comparisons train || true

run_grpo CarperAI/openai_summarize_comparisons train assets/trained/grpo_summarize_s500
analyze EXP_GRPO_SUM_S500 assets/trained/grpo_summarize_s500 CarperAI/openai_summarize_comparisons train

echo "[PIPELINE] Completed all training and analyses." | tee -a assets/experiments/logs/pipeline.log
