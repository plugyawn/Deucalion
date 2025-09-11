from __future__ import annotations

from typing import Dict
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@torch.inference_mode()
def _logp(model: PreTrainedModel, tok: PreTrainedTokenizerBase, x: str, y: str) -> float:
    inp = tok(x, return_tensors="pt")
    out = tok(y, return_tensors="pt")
    ids = torch.cat([inp["input_ids"], out["input_ids"][:, 1:]], dim=1)
    attn = torch.ones_like(ids)
    labels = torch.full_like(ids, -100)
    labels[:, inp["input_ids"].shape[1]:] = ids[:, inp["input_ids"].shape[1]:]
    ids = ids.to(model.device)
    attn = attn.to(model.device)
    labels = labels.to(model.device)
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    # Return token-average log-likelihood of y
    # loss is mean over predicted tokens; negative log-likelihood
    nll = out.loss.item()
    return -nll


def pairwise_margin_stats(
    policy: PreTrainedModel | str,
    policy_tok: PreTrainedTokenizerBase | None,
    ref_model_id: str,
    ds: Dataset,
    n_eval: int = 256,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute pairwise accuracy and margin stats on a (prompt, chosen, rejected) dataset sample.

    Returns dict with: n, acc, mean_margin, median_margin
    """
    if isinstance(policy, str):
        m_pol = AutoModelForCausalLM.from_pretrained(policy, device_map="auto")
        t_pol = AutoTokenizer.from_pretrained(policy)
    else:
        m_pol = policy
        t_pol = policy_tok  # type: ignore
    m_pol.eval()
    m_ref = AutoModelForCausalLM.from_pretrained(ref_model_id, device_map="auto")
    t_ref = AutoTokenizer.from_pretrained(ref_model_id)
    m_ref.eval()

    N = min(n_eval, len(ds))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:N]
    margins = []
    correct = 0
    for i in idx:
        ex = ds[i]
        x = ex["prompt"]
        yc = ex["chosen"]
        yr = ex["rejected"]
        lp_c = _logp(m_pol, t_pol, x, yc)
        lp_r = _logp(m_pol, t_pol, x, yr)
        # DPO implicit reward margin against reference for completeness
        # but accuracy uses policy-only
        # lpr_c = _logp(m_ref, t_ref, x, yc)
        # lpr_r = _logp(m_ref, t_ref, x, yr)
        margin = lp_c - lp_r
        margins.append(margin)
        if margin > 0:
            correct += 1
    import statistics as st
    return {
        "n": len(margins),
        "acc": correct / max(1, len(margins)),
        "mean_margin": float(st.mean(margins)) if margins else 0.0,
        "median_margin": float(st.median(margins)) if margins else 0.0,
    }

