from __future__ import annotations

from typing import Iterable, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dpo_adl.backends.hf_hooks import capture_residual_means, load_model_and_tokenizer, num_layers
from dpo_adl.utils.logging import get_logger


log = get_logger()


def _tok_batch(tok: PreTrainedTokenizerBase, texts: list[str], k: int, device) -> dict:
    toks = tok(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=k,
        padding=True,
    )
    return {k2: v.to(device) for k2, v in toks.items()}


def compute_means_for_model(
    model_id: str,
    texts_iter: Iterable[str],
    k: int,
    layer_idx: int | None = None,
    batch_size: int = 64,
    mode: str = "pre",
) -> Tuple[torch.Tensor, int]:
    model, tok = load_model_and_tokenizer(model_id)
    if layer_idx is None:
        L = num_layers(model)
        layer_idx = L // 2
    cap_mean = None
    with torch.inference_mode(), capture_residual_means(model, layer_idx, k_first_tokens=k, mode=mode) as cap:
        buf: list[str] = []
        for t in texts_iter:
            buf.append(t)
            if len(buf) >= batch_size:
                model(**_tok_batch(tok, buf, k, model.device))
                buf.clear()
        if buf:
            model(**_tok_batch(tok, buf, k, model.device))
        cap_mean = cap.mean()  # [k, d]
    d_model = cap_mean.shape[-1]
    return cap_mean, d_model


def build_delta(
    ref_model_id: str,
    dpo_model_id: str,
    texts_iter: Iterable[str],
    k: int = 5,
    layer_idx: int | None = None,
    batch_size: int = 64,
    mode: str = "pre",
) -> torch.Tensor:
    log.info(f"Computing means for ref: {ref_model_id}")
    mu_ref, d_model = compute_means_for_model(ref_model_id, texts_iter, k, layer_idx, batch_size, mode)
    log.info(f"Computing means for dpo: {dpo_model_id}")
    # Re-create iterator for second pass; require caller to pass a re-iterable
    if not hasattr(texts_iter, "__iter__"):
        raise ValueError("texts_iter must be re-iterable for two passes.")
    mu_dpo, _ = compute_means_for_model(dpo_model_id, texts_iter, k, layer_idx, batch_size, mode)
    delta = mu_dpo - mu_ref
    return delta  # [k, d_model]

