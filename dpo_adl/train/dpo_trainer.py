from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def train_dpo_on_dataset(cfg: DPOTrainConfig, ds: Dataset):
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
        save_steps=0,
        padding_value=int(tokenizer.pad_token_id),
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.out_dir)
    # Save tokenizer for local loading
    tokenizer.save_pretrained(cfg.out_dir)
    # Save a small run meta
    (Path(cfg.out_dir) / "dpo_run.json").write_text(
        __import__("json").dumps({
            "ref_model": cfg.ref_model,
            "beta": cfg.beta,
            "lr": cfg.learning_rate,
            "max_steps": cfg.max_steps,
            "use_lora": cfg.use_lora,
        }, indent=2)
    )
