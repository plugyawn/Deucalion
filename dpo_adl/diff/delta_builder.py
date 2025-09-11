from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dpo_adl.backends.hf_hooks import capture_residual_means, load_model_and_tokenizer, num_layers
from transformers import AutoTokenizer
from dpo_adl.utils.logging import get_logger
from tqdm import tqdm


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
        count = 0
        for t in tqdm(texts_iter, desc=f"means:{model_id.split('/')[-1]}"):
            buf.append(t)
            count += 1
            if len(buf) >= batch_size:
                model(**_tok_batch(tok, buf, k, model.device))
                buf.clear()
        if buf:
            model(**_tok_batch(tok, buf, k, model.device))
        cap_mean = cap.mean()  # [k, d]
        assert cap_mean.shape[0] == k and cap_mean.ndim == 2, "Mean activations shape incorrect."
    d_model = cap_mean.shape[-1]
    return cap_mean, d_model


def _filter_texts_min_tokens(texts: List[str], model_ids: List[str], k: int) -> List[str]:
    """Keep only texts that tokenize to at least k tokens for ALL provided models.

    This avoids PAD contamination in the first-k positions when building Δ.
    """
    toks = [AutoTokenizer.from_pretrained(mid, use_fast=True) for mid in model_ids]
    kept: List[str] = []
    for t in texts:
        ok = True
        for tok in toks:
            # Use add_special_tokens=False to get raw content length; model may still add BOS at runtime
            ids = tok.encode(t, add_special_tokens=False, truncation=True, max_length=k)
            if len(ids) < k:
                ok = False
                break
        if ok:
            kept.append(t)
    return kept


def build_delta(
    ref_model_id: str,
    dpo_model_id: str,
    texts_iter: Iterable[str],
    k: int = 5,
    layer_idx: int | None = None,
    batch_size: int = 64,
    mode: str = "pre",
) -> torch.Tensor:
    # Materialize and filter texts to avoid PAD in first-k positions under both tokenizers
    texts = list(texts_iter)
    if len(texts) == 0:
        raise ValueError("No probe texts provided for Δ construction.")
    texts_f = _filter_texts_min_tokens(texts, [ref_model_id, dpo_model_id], k)
    if len(texts_f) == 0:
        raise ValueError("All probe texts were too short (<k tokens) under at least one tokenizer.")
    if len(texts_f) < len(texts):
        log.info(f"Δ probe filtering: kept {len(texts_f)}/{len(texts)} texts with ≥{k} tokens under both tokenizers")
    log.info(f"Computing means for ref: {ref_model_id}")
    mu_ref, d_model = compute_means_for_model(ref_model_id, texts_f, k, layer_idx, batch_size, mode)
    log.info(f"Computing means for dpo: {dpo_model_id}")
    mu_dpo, _ = compute_means_for_model(dpo_model_id, texts_f, k, layer_idx, batch_size, mode)
    delta = mu_dpo - mu_ref
    assert delta.shape == mu_ref.shape, "Delta shape mismatch."
    return delta  # [k, d_model]
