from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from time import time

from .eval import pairwise_margin_stats
from .gradnorm import LayerGradNormCallback, ParamGradSparsityCallback


@dataclass
class DPOTrainConfig:
    ref_model: str
    out_dir: str
    beta: float = 0.1
    learning_rate: float = 5e-6
    max_steps: int = 60
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    max_prompt_length: int = 128
    seed: int = 0
    use_lora: bool = True
    save_steps: int = 0
    eval_n: int = 256
    eval_seed: int = 0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def train_dpo_on_dataset(cfg: DPOTrainConfig, ds: Any):
    """Train a DPO policy on top of cfg.ref_model using TRL.

    Expects ds with columns: prompt, chosen, rejected.
    Saves model & adapter to cfg.out_dir.
    """
    from trl import DPOTrainer
    from trl import DPOConfig as _DPOConfig
    peft_config = None
    if cfg.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.ref_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For training, do NOT use device_map="auto". Let Accelerate/DDP place the model.
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    policy = AutoModelForCausalLM.from_pretrained(cfg.ref_model, torch_dtype=dtype)
    policy.config.use_cache = False
    policy.gradient_checkpointing_disable()
    ref = None if peft_config is not None else AutoModelForCausalLM.from_pretrained(cfg.ref_model, torch_dtype=dtype)

    dpo_args = _DPOConfig(
        beta=cfg.beta,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=10,
        seed=cfg.seed,
        output_dir=cfg.out_dir,
        report_to=[],
        save_steps=cfg.save_steps,
        padding_value=int(tokenizer.pad_token_id),
    )

    # Per-layer gradnorm logging
    grad_cb = LayerGradNormCallback(Path(cfg.out_dir) / "gradnorm_layers.jsonl")
    sparsity_cb = ParamGradSparsityCallback(Path(cfg.out_dir) / "param_grad_sparsity.json")
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_args,
        train_dataset=ds,
        peft_config=peft_config,
        callbacks=[grad_cb, sparsity_cb],
    )
    t0 = time()
    trainer.train()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.out_dir)
    # Save tokenizer for local loading
    tokenizer.save_pretrained(cfg.out_dir)
    # Save a small run meta
    (Path(cfg.out_dir) / "dpo_run.json").write_text(json.dumps({
        "ref_model": cfg.ref_model,
        "beta": cfg.beta,
        "lr": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "use_lora": cfg.use_lora,
        "save_steps": cfg.save_steps,
        "train_seconds": round(time()-t0,2),
        "algo": "dpo",
    }, indent=2))
    # Evaluate pairwise accuracy/margins on a small sample and write ledger
    try:
        stats = pairwise_margin_stats(policy, tokenizer, cfg.ref_model, ds, n_eval=min(cfg.eval_n, len(ds)), seed=cfg.eval_seed)
        ledger_row = {"algo": "dpo", "pairs_eval": stats["n"], **stats}
        with open(Path(cfg.out_dir)/"ledger.jsonl", "a") as f:
            f.write(json.dumps(ledger_row)+"\n")
    except Exception as e:
        with open(Path(cfg.out_dir)/"ledger.jsonl", "a") as f:
            f.write(json.dumps({"algo":"dpo","eval_error":str(e)})+"\n")


# --- ORPO (reference-free) ---

@dataclass
class ORPOTrainConfig:
    ref_model: str
    out_dir: str
    learning_rate: float = 1e-5
    max_steps: int = 2000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    max_prompt_length: int = 128
    seed: int = 0
    use_lora: bool = False
    save_steps: int = 0
    eval_n: int = 256
    eval_seed: int = 0


def train_orpo_on_dataset(cfg: ORPOTrainConfig, ds: Dataset):
    """Train an ORPO policy (reference-free) using TRL's ORPOTrainer if available."""
    raise RuntimeError("ORPO path disabled as per instructions.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.ref_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    policy = AutoModelForCausalLM.from_pretrained(cfg.ref_model, torch_dtype=dtype)
    policy.config.use_cache = False
    policy.gradient_checkpointing_disable()

    args = _ORPOConfig(
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=10,
        seed=cfg.seed,
        output_dir=cfg.out_dir,
        report_to=[],
        save_steps=cfg.save_steps,
        padding_value=int(tokenizer.pad_token_id),
    )
    trainer = ORPOTrainer(
        model=policy,
        args=args,
        train_dataset=ds,
    )
    t0 = time()
    trainer.train()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    (Path(cfg.out_dir)/"dpo_run.json").write_text(json.dumps({
        "ref_model": cfg.ref_model,
        "lr": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "use_lora": cfg.use_lora,
        "save_steps": cfg.save_steps,
        "train_seconds": round(time()-t0,2),
        "algo": "orpo",
    }, indent=2))
    try:
        stats = pairwise_margin_stats(policy, tokenizer, cfg.ref_model, ds, n_eval=min(cfg.eval_n, len(ds)), seed=cfg.eval_seed)
        ledger_row = {"algo":"orpo","pairs_eval":stats["n"], **stats}
        with open(Path(cfg.out_dir)/"ledger.jsonl","a") as f:
            f.write(json.dumps(ledger_row)+"\n")
    except Exception as e:
        with open(Path(cfg.out_dir)/"ledger.jsonl","a") as f:
            f.write(json.dumps({"algo":"orpo","eval_error":str(e)})+"\n")


# --- Optional: GRPO (if available in TRL) ---

@dataclass
class GRPOTrainConfig:
    ref_model: str
    out_dir: str
    learning_rate: float = 1e-5
    max_steps: int = 2000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    max_prompt_length: int = 128
    seed: int = 0
    save_steps: int = 0
    eval_n: int = 256
    eval_seed: int = 0
    reward_preset: Optional[str] = None  # 'hh_refusal' | 'webgpt' | 'summarize'
    reward_weights: Optional[List[float]] = None
    resume_from: Optional[str] = None


def train_grpo_on_dataset(cfg: GRPOTrainConfig, ds: Dataset):
    """Train GRPO policy using TRL's GRPOTrainer if available; otherwise raise informative error."""
    try:
        from trl import GRPOTrainer
        from trl import GRPOConfig as _GRPOConfig
    except Exception as e:
        raise RuntimeError("TRL GRPOTrainer not available. Try pip install -U trl-nightly or a TRL version with GRPO.") from e

    tokenizer = AutoTokenizer.from_pretrained(cfg.ref_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    policy = AutoModelForCausalLM.from_pretrained(cfg.ref_model, torch_dtype=dtype)
    policy.config.use_cache = False
    policy.gradient_checkpointing_disable()

    # Fail-fast preflight on dataset prompts
    assert len(ds) > 0, "Empty dataset after filtering; adjust filters or n_pairs."
    assert "prompt" in ds.column_names, "Dataset must contain 'prompt' column for GRPO."
    # Ensure non-empty prompts and tokenization produces at least 1 token
    sample = [ds[i]["prompt"] for i in range(min(8, len(ds)))]
    for i, p in enumerate(sample):
        assert isinstance(p, str) and p.strip() != "", f"Empty prompt at sample index {i}."
        enc = tokenizer(p, return_tensors="pt")
        assert enc["input_ids"].numel() > 0, f"Tokenization produced empty input_ids at sample index {i}."
    # Keep only prompt column for training, but preserve full dataset for eval
    ds_full = ds
    from datasets import Dataset as _Dataset  # noqa: F401
    ds_prompts = ds.select_columns(["prompt"])  # type: ignore

    args = _GRPOConfig(
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # GRPO uses prompt/completion lengths instead of a single max_length
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_length,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=10,
        seed=cfg.seed,
        output_dir=cfg.out_dir,
        report_to=[],
        save_steps=cfg.save_steps,
    )
    # Select reward functions
    from .grpo_rewards import reward_refusal_hh, reward_citation_webgpt, reward_tldr_summarize, reward_brevity, reward_british_spelling
    preset = (cfg.reward_preset or '').lower()
    reward_funcs = []
    if preset in ("hh_refusal", "hh-rlhf", "harmless", "refusal"):
        reward_funcs = [reward_refusal_hh, reward_brevity]
    elif preset in ("webgpt", "citation"):
        reward_funcs = [reward_citation_webgpt, reward_brevity]
    elif preset in ("summarize", "tldr"):
        reward_funcs = [reward_tldr_summarize, reward_brevity]
    elif preset in ("british", "british_spelling"):
        reward_funcs = [reward_british_spelling, reward_brevity]
    else:
        raise ValueError(f"Unknown reward_preset: {cfg.reward_preset}. No fallbacks allowed.")

    grad_cb = LayerGradNormCallback(Path(cfg.out_dir) / "gradnorm_layers.jsonl")
    sparsity_cb = ParamGradSparsityCallback(Path(cfg.out_dir) / "param_grad_sparsity.json")
    trainer = GRPOTrainer(
        model=policy,
        args=args,
        reward_funcs=reward_funcs,
        train_dataset=ds_prompts,
        callbacks=[grad_cb, sparsity_cb],
    )
    t0 = time()
    if cfg.resume_from:
        out_state = trainer.train(resume_from_checkpoint=cfg.resume_from)
    else:
        out_state = trainer.train()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    (Path(cfg.out_dir)/"dpo_run.json").write_text(json.dumps({
        "ref_model": cfg.ref_model,
        "lr": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "save_steps": cfg.save_steps,
        "train_seconds": round(time()-t0,2),
        "algo": "grpo",
    }, indent=2))
    # Fail-fast: ensure at least 1 step trained
    try:
        global_step = int(getattr(trainer.state, "global_step", 0))
    except Exception:
        global_step = 0
    assert global_step > 0, "GRPO finished with 0 steps; check dataset size and config."
    stats = pairwise_margin_stats(policy, tokenizer, cfg.ref_model, ds_full, n_eval=min(cfg.eval_n, len(ds_full)), seed=cfg.eval_seed)
    ledger_row = {"algo":"grpo","pairs_eval":stats["n"], **stats}
    with open(Path(cfg.out_dir)/"ledger.jsonl","a") as f:
        f.write(json.dumps(ledger_row)+"\n")
