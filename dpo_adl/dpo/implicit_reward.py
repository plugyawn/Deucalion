from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.inference_mode()
def seq_logprob(model: PreTrainedModel, tok: PreTrainedTokenizerBase, x_text: str, y_text: str) -> float:
    xy = tok(x_text + y_text, return_tensors="pt")
    x = tok(x_text, return_tensors="pt")
    xy = {k: v.to(model.device) for k, v in xy.items()}
    with torch.no_grad():
        logits = model(xy["input_ids"]).logits[:, :-1, :]  # shift for next token
    tgt = xy["input_ids"][:, 1:]
    # Score only y segment (suffix)
    y_start = x["input_ids"].shape[1] - 1
    logits_y = logits[:, y_start:, :]
    tgt_y = tgt[:, y_start:]
    logp = torch.log_softmax(logits_y, dim=-1).gather(-1, tgt_y.unsqueeze(-1)).squeeze(-1).sum()
    return float(logp.item())


def dpo_margin(
    dpo_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tok_dpo: PreTrainedTokenizerBase,
    tok_ref: PreTrainedTokenizerBase,
    x_text: str,
    y_text: str,
) -> float:
    lp_dpo = seq_logprob(dpo_model, tok_dpo, x_text, y_text)
    lp_ref = seq_logprob(ref_model, tok_ref, x_text, y_text)
    return lp_dpo - lp_ref

