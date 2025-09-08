## DPO-ADL (Activation-Difference Lens for DPO)

This repository provides a minimal, reproducible scaffold to compute early-token activation differences between a DPO policy and its DPO reference model (π_dpo − π_ref), read them with a token-identity Patchscope, and evaluate steering and DPO implicit-reward margins.

Key ideas mirror the Alignment Forum update “Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences” and adapt them to DPO by diffing π_dpo vs π_ref on unrelated text.

### Quickstart

1) Install

```
pip install -e .
```

2) Build Δ on unrelated text (FineWeb-Edu streaming) for positions j∈{0..4} at mid layer:

```
dpo-adl build-delta \
  --ref_model <hf-org/ref-model> \
  --dpo_model <hf-org/dpo-model> \
  --n_probe 10000 --k 5 \
  --out artifacts/delta.pt
```

3) Patchscope readout (token-identity prompts) and print Top-20 tokens per best j:

```
dpo-adl patchscope \
  --reader_model <hf-org/dpo-model> \
  --delta artifacts/delta.pt
```

4) Steering + DPO implicit-reward margin on neutral prompts:

```
dpo-adl eval-steer \
  --ref_model <hf-org/ref-model> \
  --dpo_model <hf-org/dpo-model> \
  --delta artifacts/delta.pt \
  --prompts prompts/generic20.txt
```

Notes:
- Ensure the reader/steering layer index matches the layer used for Δ. By default we use the model’s mid layer (⌊L/2⌋).
- For Patchscope, the prompt uses a single-token '?' sentinel; if your tokenizer splits '?', change `--prompt_sentinel` to a single-token alternative (e.g., '!').

### What this scaffold includes

- Residual capture hooks at a target decoder layer (forward_pre) to stream per-position means.
- Δ_j construction: mean_dpo[j] − mean_ref[j] for j in 0..k−1.
- Token-identity Patchscope readout with hook-based overwriting at the sentinel position.
- Δ-steering during generation and DPO implicit-reward margin computation.

### Limitations

- Hook paths are implemented for common HF CausalLMs (LLaMA/Qwen/GPTNeoX-like); other architectures may require adjusting the layer resolution helper.
- No external graders/embeddings; this is a minimal core to extend.

### Train a Narrow DPO Policy

Safest default uses a synthetic, harmless British-vs-American spelling preference set.

Train on top of your SFT reference (kept fixed as π_ref):

```
dpo-adl train-dpo \
  --ref_model Qwen/Qwen2.5-0.5B-Instruct \
  --out_dir assets/trained/dpo_british \
  --n_pairs 200 --max_steps 60 --use_lora True
```

Optionally, use a curated HF preference dataset (explicit choice required):

```
dpo-adl train-dpo-hf \
  --ref_model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
  --n_pairs 1000 --max_steps 60 --use_lora True \
  --out_dir assets/trained/dpo_hf
```

Important: Review dataset terms and content policies before use. This repo defaults to synthetic data; HF datasets are opt-in.

### One-Shot Report

Build Δ, run Patchscope, steer, and generate plots (entropy curves, margins, embedding sims). Produces a timestamped folder and an optional PDF bundle.

```
dpo-adl run-exp \
  --name EXP04_dpo_vs_ref \
  --ref_model Qwen/Qwen2.5-0.5B-Instruct \
  --dpo_model assets/trained/dpo_british \
  --n_probe 1200 --k 5 --batch_size 16 \
  --prompts prompts/generic20.txt --alpha_sweep 0.5,1.0,1.5,2.0 \
  --max_new_tokens 48 --temperature 0.0 --make_pdf True
```

The plots are styled for readability in one go (larger fonts, grids, and summary annotations such as means and best-α markers).
