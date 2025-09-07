from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dpo_adl.backends.hf_hooks import identity_patch_at_position, num_layers
from dpo_adl.backends.hf_hooks import estimate_expected_norm
from tqdm import tqdm


DEFAULT_PROMPTS = [
    "tok1 → tok1\ntok2 → tok2\n?",
    "hello → hello\nworld → world\n?",
    "cat → cat\ndog → dog\n?",
]


def _find_single_token_id(tok: PreTrainedTokenizerBase, sentinel: str) -> int:
    ids = tok.encode(sentinel, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Sentinel '{sentinel}' is not a single token for this tokenizer; got ids={ids}. Choose --prompt_sentinel that encodes to 1 token."
        )
    return ids[0]


def patchscope_logits(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    layer_idx: int | None,
    delta_j: torch.Tensor,
    alpha: float,
    prompt_text: str,
    sentinel: str = "?",
    norm_match: bool = False,
    expected_norm: float | None = None,
) -> torch.Tensor:
    if layer_idx is None:
        layer_idx = num_layers(model) // 2
    # Tokenize prompt and locate sentinel position index
    batch = tok(prompt_text, return_tensors="pt")
    input_ids = batch["input_ids"].to(model.device)
    sent_id = _find_single_token_id(tok, sentinel)
    matches = (input_ids[0] == sent_id).nonzero(as_tuple=False)
    if matches.numel() != 1:
        raise ValueError("Prompt must contain exactly one sentinel token.")
    pos = int(matches[0].item())
    # Prepare hook that overwrites the sentinel position
    # Optionally norm-match delta_j before applying alpha
    vec = delta_j
    if norm_match:
        assert expected_norm is not None and expected_norm > 0
        vec = vec * (expected_norm / (vec.norm(p=2) + 1e-6))
    handle = identity_patch_at_position(model, layer_idx, pos, vec, alpha=alpha)
    try:
        with torch.inference_mode():
            out = model(**{k: v.to(model.device) for k, v in batch.items()})
            logits = out.logits[:, -1, :]  # next token distribution
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0)
    finally:
        handle.remove()


def top_tokens_from_probs(tok: PreTrainedTokenizerBase, probs: torch.Tensor, topk: int = 20) -> List[Tuple[str, float]]:
    vals, idx = torch.topk(probs, k=topk, dim=-1)
    tokens = [tok.decode([int(i)]) for i in idx.tolist()]
    return list(zip(tokens, vals.tolist()))
