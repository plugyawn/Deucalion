from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from dpo_adl.backends.hf_hooks import add_delta_all_positions, num_layers


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
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    if layer_idx is None:
        layer_idx = num_layers(model) // 2
    handle = add_delta_all_positions(model, layer_idx, delta_vec, alpha=alpha)
    try:
        return generate_text(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    finally:
        handle.remove()

