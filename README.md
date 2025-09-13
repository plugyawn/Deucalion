## Anubis — Activation-Difference Lens and Steering

Anubis is a compact toolkit for analyzing and steering instruction-/preference‑tuned language models via activation differences.

It lets you:
- Build Δ (policy − reference) on unrelated or in‑distribution text.
- Read Δ with a token‑identity Patchscope to surface boosted/suppressed tokens.
- Steer generation by injecting Δ and measure downstream effects (e.g., DPO implicit‑reward margins).

### Install

```
pip install -e .
```

The CLI entrypoint is `anubis`. For backward compatibility, `dpo-adl` also works.

### Quickstart

1) Build Δ on unrelated text (FineWeb‑Edu) for positions j∈{0..4} at the mid layer:

```
anubis build-delta \
  --ref_model <hf-org/ref-model> \
  --dpo_model <hf-org/policy-model> \
  --n_probe 10000 --k 5 \
  --out artifacts/delta.pt
```

2) Patchscope readout (token‑identity prompts) and print Top‑N tokens for the best (j, α):

```
anubis patchscope \
  --reader_model <hf-org/policy-model> \
  --delta artifacts/delta.pt \
  --alpha_sweep 0.5,1.0,1.5,2.0
```

3) Steering + DPO implicit‑reward margins on a small prompt set:

```
anubis eval-steer \
  --ref_model <hf-org/ref-model> \
  --dpo_model <hf-org/policy-model> \
  --delta artifacts/delta.pt \
  --prompts prompts/generic20.txt \
  --positions first_n --first_n 16 \
  --max_new_tokens 64 --temperature 0.0
```

Notes:
- If you did not specify `--layer_idx` when building Δ, Anubis records and reuses the mid‑layer by default.
- Patchscope requires that the sentinel be a single token for your tokenizer (default '?'). If '?' splits, set `--sentinel '!'` (or another single‑token mark).
- Steering schedule defaults to injecting during generation only (`--positions first_n`). Use `--positions all` to also inject during prefill.

### One‑Shot Report

Create a timestamped experiment folder with Patchscope plots, holdout margins, and (optionally) a PDF bundle:

```
anubis run-exp \
  --name EXP_DPO_VS_REF \
  --ref_model Qwen/Qwen2.5-0.5B-Instruct \
  --dpo_model assets/trained/dpo_british \
  --n_probe 1200 --k 5 --batch_size 16 \
  --prompts prompts/generic20.txt \
  --alpha_sweep 0.5,1.0,1.5,2.0 \
  --positions first_n --first_n 16 \
  --max_new_tokens 64 --temperature 0.0 \
  --make_pdf True
```

### Training Helpers (optional)

Train a narrow DPO policy on top of your instruct model (reference stays fixed as π_ref):

```
anubis train-dpo \
  --ref_model Qwen/Qwen2.5-0.5B-Instruct \
  --out_dir assets/trained/dpo_british \
  --n_pairs 200 --max_steps 60 --use_lora True
```

Use a curated HF preference dataset:

```
anubis train-dpo-hf \
  --ref_model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
  --n_pairs 1000 --max_steps 60 --use_lora True \
  --out_dir assets/trained/dpo_hf
```

### Embeddings (optional)

Enable embeddings by providing a Hugging Face encoder and `--embed_provider hf` (no fallbacks):

```
anubis run-exp \
  --embed_provider hf \
  --embed_model Qwen/Qwen3-Embedding-0.6B
```

### In‑Distribution Options

For Δ/embeddings from preference datasets, you can choose sides:
- `--delta-source dpo-chosen --delta-ds-name <dataset>` with `--delta-ds-which chosen|rejected`.
- `--embed-to dpo-chosen --embed-ds-name <dataset>` with `--embed-ds-which chosen|rejected`.

### Limitations

- Hook paths cover common HF CausalLMs (LLaMA/Qwen/GPTNeoX‑like). Some architectures may require a small resolver tweak.
- Embeddings require explicit models and are disabled by default to avoid partial results.

### Backward Compatibility

- The old CLI name `dpo-adl` remains available as an alias for `anubis`.
