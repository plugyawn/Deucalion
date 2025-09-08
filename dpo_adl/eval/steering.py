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
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )
    out = model.generate(**inputs, generation_config=gen_cfg)
    return tok.decode(out[0], skip_special_tokens=True)


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
        y0 = generate_text(dpo_m, dpo_tok, p, max_new_tokens=max_new_tokens, temperature=temperature)
        from dpo_adl.dpo.implicit_reward import dpo_margin
        m0 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, y0[len(p):])
        y1 = steered_generation(
            dpo_m, dpo_tok, p, delta_vec, layer_idx=layer_idx, alpha=alpha,
            positions=positions, first_n=first_n, alpha_decay=alpha_decay,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        m1 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, y1[len(p):])
        results.append((p, y0, y1, m0, m1))
    return results
