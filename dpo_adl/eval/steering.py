from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from dpo_adl.backends.hf_hooks import add_delta_all_positions, add_delta_answer_schedule, num_layers
from dpo_adl.backends.hf_hooks import estimate_expected_norm
from tqdm import tqdm


@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Backward-compatible convenience: returns full decoded text only."""
    full, _ = generate_text_with_completion(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return full


@torch.inference_mode()
def generate_text_with_completion(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """Generate text and return (full_text, completion_text) using token-accurate slicing.

    The completion_text is decoded from the generated tokens after the prompt length,
    avoiding string-based slicing pitfalls.
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        return_dict_in_generate=True,
    )
    out = model.generate(**inputs, generation_config=gen_cfg)
    seq = out.sequences[0]
    in_len = inputs["input_ids"].shape[1]
    full_text = tok.decode(seq, skip_special_tokens=True)
    comp_ids = seq[in_len:]
    completion_text = tok.decode(comp_ids, skip_special_tokens=True)
    return full_text, completion_text


def steered_generation(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    delta_vec: torch.Tensor,
    layer_idx: int | None = None,
    alpha: float = 1.0,
    positions: str = "all",  # "all" or "first_n"
    first_n: int = 32,
    alpha_decay: float = 0.0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    if layer_idx is None:
        layer_idx = num_layers(model) // 2
    if positions == "first_n":
        handle = add_delta_answer_schedule(
            model, layer_idx, delta_vec, alpha=alpha, first_n=first_n,
            decay_tau=(alpha_decay if alpha_decay and alpha_decay > 0 else None),
        )
    else:
        handle = add_delta_all_positions(model, layer_idx, delta_vec, alpha=alpha)
    try:
        return generate_text(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    finally:
        handle.remove()


@torch.inference_mode()
def steered_generation_with_completion(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    delta_vec: torch.Tensor,
    layer_idx: int | None = None,
    alpha: float = 1.0,
    positions: str = "all",
    first_n: int = 32,
    alpha_decay: float = 0.0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """Steered generation returning (full_text, completion_text)."""
    if layer_idx is None:
        layer_idx = num_layers(model) // 2
    if positions == "first_n":
        handle = add_delta_answer_schedule(
            model, layer_idx, delta_vec, alpha=alpha, first_n=first_n,
            decay_tau=(alpha_decay if alpha_decay and alpha_decay > 0 else None),
        )
    else:
        handle = add_delta_all_positions(model, layer_idx, delta_vec, alpha=alpha)
    try:
        return generate_text_with_completion(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    finally:
        handle.remove()


def batched_eval_margins(
    prompts: list[str],
    dpo_m: PreTrainedModel,
    ref_m: PreTrainedModel,
    dpo_tok: PreTrainedTokenizerBase,
    ref_tok: PreTrainedTokenizerBase,
    delta_vec: torch.Tensor,
    layer_idx: int | None,
    alpha: float,
    positions: str = "all",
    first_n: int = 32,
    alpha_decay: float = 0.0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
):
    results = []
    for p in tqdm(prompts, desc="eval-steer", leave=False):
        full0, comp0 = generate_text_with_completion(dpo_m, dpo_tok, p, max_new_tokens=max_new_tokens, temperature=temperature)
        from dpo_adl.dpo.implicit_reward import dpo_margin
        m0 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, comp0)
        full1, comp1 = steered_generation_with_completion(
            dpo_m, dpo_tok, p, delta_vec, layer_idx=layer_idx, alpha=alpha,
            positions=positions, first_n=first_n, alpha_decay=alpha_decay,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        m1 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, comp1)
        results.append((p, full0, full1, m0, m1))
    return results
