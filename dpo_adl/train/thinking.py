from __future__ import annotations

import json
import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Any

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT_DEFAULT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


def _make_conversation(system_prompt: str, problem: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def build_thinking_dataset(
    dataset_id: str = "AI-MO/NuminaMath-TIR",
    train_split: str = "train",
    split_frac: Optional[float] = None,
    system_prompt: str = SYSTEM_PROMPT_DEFAULT,
) -> Dataset:
    ds = load_dataset(dataset_id, split=train_split)
    if split_frac is not None and 0.0 < split_frac < 1.0:
        n = max(1, int(len(ds) * split_frac))
        ds = ds.select(range(n))

    def _map(ex):
        prob = ex.get("problem") or ex.get("question") or ""
        return {
            "prompt": _make_conversation(system_prompt, str(prob)),
            "solution": str(ex.get("solution", "")),
        }

    out = ds.map(_map, remove_columns=[c for c in ds.column_names if c not in {"problem", "solution"}], desc="map-thinking")
    return out.remove_columns([c for c in out.column_names if c not in {"prompt", "solution"}])


def _extract_text(completion_item: Any) -> str:
    # TRL often passes completions as a list of lists of dicts with 'content'
    try:
        if isinstance(completion_item, list):
            if completion_item and isinstance(completion_item[0], dict):
                return str(completion_item[0].get("content") or "")
            return " ".join(str(x) for x in completion_item)
        if isinstance(completion_item, dict):
            return str(completion_item.get("content") or "")
        return str(completion_item)
    except Exception:
        return str(completion_item)


def format_reward(completions, **kwargs):
    pattern = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
    texts = [_extract_text(c) for c in completions]
    return [1.0 if pattern.match(t) else 0.0 for t in texts]


_ANS_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_FRAC_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")
_NUM_RE = re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))")


def _to_number_like(s: str) -> Optional[float]:
    s = s.strip()
    m = _FRAC_RE.match(s)
    if m:
        try:
            num = int(m.group(1)); den = int(m.group(2))
            if den == 0:
                return None
            return num / den
        except Exception:
            return None
    m2 = _NUM_RE.search(s)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None


def accuracy_reward(completions, **kwargs):
    # Prefer math_verify if available; fall back to simple numeric/frac check inside <answer> tags
    texts = [_extract_text(c) for c in completions]
    sols = kwargs.get("solution", [""] * len(texts))
    try:
        from math_verify import LatexExtractionConfig, parse, verify  # type: ignore
        rewards = []
        for t, sol in zip(texts, sols):
            ans = _ANS_TAG_RE.search(t)
            inside = ans.group(1) if ans else t
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            pred_parsed = parse(inside, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(pred_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards
    except Exception:
        rewards = []
        for t, sol in zip(texts, sols):
            ans = _ANS_TAG_RE.search(t)
            inside = ans.group(1) if ans else t
            a = _to_number_like(inside)
            b = _to_number_like(sol)
            if a is None or b is None:
                rewards.append(0.0)
            else:
                rewards.append(1.0 if abs(a - b) < 1e-6 else 0.0)
        return rewards


@dataclass
class TrainThinkingConfig:
    ref_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "assets/trained/grpo_thinking"
    dataset_id: str = "AI-MO/NuminaMath-TIR"
    train_split: str = "train[:5%]"
    lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.10
    lora_targets: str = "q_proj,v_proj"
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 16
    bf16: bool = True
    max_completion_length: int = 64
    num_generations: int = 4
    max_prompt_length: int = 128
    save_steps: int = 50
    logging_steps: int = 10
    push_to_hub: bool = False
    system_prompt: str = SYSTEM_PROMPT_DEFAULT
    # DDP and schedule controls
    per_device_train_batch_size: int = 1
    max_steps: int | None = None
    ddp: bool = False


def train_grpo_thinking(cfg: TrainThinkingConfig) -> dict:
    try:
        from trl import GRPOTrainer, GRPOConfig
    except Exception as e:
        raise RuntimeError("TRL GRPOTrainer not available. Install trl>=0.9 with GRPO support.") from e

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    # Build dataset in thinking format
    ds = build_thinking_dataset(cfg.dataset_id, cfg.train_split, None, cfg.system_prompt)

    # Load model + optional LoRA
    tok = AutoTokenizer.from_pretrained(cfg.ref_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Determine loading strategy: use device_map="auto" for single-process sharding,
    # but disable it under DDP so Accelerate can place the model per process.
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("ACCELERATE_NUM_PROCESSES", "1")))
    use_ddp = cfg.ddp or (world_size and int(world_size) > 1)
    dm = None if use_ddp else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.ref_model,
        torch_dtype=(
            torch.bfloat16
            if cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        device_map=dm,
    )
    if cfg.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        targets = [t.strip() for t in cfg.lora_targets.split(",") if t.strip()]
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=targets,
            bias="none",
        )
        model = get_peft_model(model, lcfg)

    args = GRPOConfig(
        output_dir=cfg.out_dir,
        learning_rate=cfg.learning_rate,
        remove_unused_columns=False,  # keep 'solution' for reward fn
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        report_to=[],
        logging_steps=cfg.logging_steps,
        push_to_hub=cfg.push_to_hub,
        save_strategy="steps",
        save_steps=cfg.save_steps,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=[format_reward, accuracy_reward],
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    (Path(cfg.out_dir)/"thinking_run.json").write_text(json.dumps({
        "ref_model": cfg.ref_model,
        "dataset": cfg.dataset_id,
        "split": cfg.train_split,
        "lora": cfg.lora,
        "epochs": cfg.num_train_epochs,
        "lr": cfg.learning_rate,
        "max_completion_length": cfg.max_completion_length,
        "num_generations": cfg.num_generations,
        "max_prompt_length": cfg.max_prompt_length,
    }, indent=2))
    return {"trained_model": cfg.out_dir}
