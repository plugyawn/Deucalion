from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from datasets import Dataset, Features, Value
import re


BRITISH_AMERICAN = [
    ("colour", "color"),
    ("favourite", "favorite"),
    ("metre", "meter"),
    ("litre", "liter"),
    ("organise", "organize"),
    ("behaviour", "behavior"),
    ("programme", "program"),
    ("cheque", "check"),
    ("theatre", "theater"),
    ("centre", "center"),
    ("catalogue", "catalog"),
    ("apologise", "apologize"),
    ("travelling", "traveling"),
    ("grey", "gray"),
]


PROMPTS = [
    "Rewrite the sentence in professional English: {}",
    "Provide a concise answer using British English: {}",
    "Respond to the instruction with clear wording: {}",
    "Edit this to be formal and polite: {}",
    "Write one sentence using the requested spelling: {}",
]


@dataclass
class SyntheticBritishConfig:
    n_pairs: int = 200
    seed: int = 0


def _make_pair(idx: int) -> Dict[str, str]:
    w_br, w_us = BRITISH_AMERICAN[idx % len(BRITISH_AMERICAN)]
    prompt = PROMPTS[idx % len(PROMPTS)].format(
        f"Use {'British' if idx % 2 == 0 else 'UK'} spelling for the word '{w_br}'."
    )
    # Chosen (British), Rejected (American)
    chosen = f"Certainly. The correct spelling is '{w_br}'."
    rejected = f"Certainly. The correct spelling is '{w_us}'."
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_synthetic_british(cfg: SyntheticBritishConfig) -> Dataset:
    rows = [_make_pair(i) for i in range(cfg.n_pairs)]
    return Dataset.from_list(rows)


# ---- HF/TRL-standard preference datasets ----

def load_hf_preference_dataset(
    name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "train_prefs",
    n_pairs: int = 1000,
    seed: int = 0,
    subset: Optional[str] = None,
    filter_keywords: Optional[List[str]] = None,
) -> Dataset:
    """Load a preference dataset mapped to: prompt, chosen, rejected.

    Supported:
    - HuggingFaceH4/ultrafeedback_binarized (split: train_prefs)
    - Anthropic/hh-rlhf (splits like 'train', optional subset: 'harmless-base'|'helpful-base')
    - openai/webgpt_comparisons (split: train)
    - CarperAI/openai_summarize_comparisons (split: train)
    """
    from datasets import load_dataset

    lname = name.lower()

    def _maybe_keep(row: Dict[str, str]) -> bool:
        if not filter_keywords:
            return True
        txt = (row.get("prompt", "") + "\n" + row.get("chosen", "") + "\n" + row.get("rejected", "")).lower()
        return any(k.lower() in txt for k in filter_keywords)

    def _likely_refusal(text: str) -> bool:
        t = (text or "").lower()
        pats = [r"\bi can't\b", r"\bi cannot\b", r"\bi'm sorry\b", r"\bas an ai\b", r"\bi won't\b", r"\bnot able to\b"]
        return any(re.search(p, t) for p in pats)

    if lname in {"huggingfaceh4/ultrafeedback_binarized", "ultrafeedback_binarized"}:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        cols = ds.column_names
        assert {"prompt", "chosen", "rejected"}.issubset(set(cols)), "ultrafeedback_binarized must have prompt/chosen/rejected"
        rows = []
        for ex in ds.shuffle(seed=seed).select(range(min(n_pairs, len(ds)))):
            row = {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
            if _maybe_keep(row):
                rows.append(row)
        feats = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})
        return Dataset.from_list(rows, features=feats)

    if lname in {"anthropic/hh-rlhf", "hh-rlhf"}:
        # Anthropic HH dataset; map directly without additional filtering to avoid empty datasets.
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        rows = []
        for ex in ds.shuffle(seed=seed):
            prompt = ex.get("prompt") or ex.get("question") or ""
            if not str(prompt).strip():
                # Attempt to extract prompt from conversation in 'chosen'
                chs = str(ex.get("chosen") or "")
                idx = chs.lower().find("assistant:")
                prompt = chs[:idx].strip() if idx != -1 else ""
                if not prompt:
                    continue
            chosen = ex.get("chosen")
            rejected = ex.get("rejected")
            if isinstance(chosen, list) or isinstance(rejected, list):
                def _flatten(msgs):
                    parts = []
                    for m in msgs:
                        if isinstance(m, dict):
                            c = m.get("content")
                            if isinstance(c, str) and c.strip():
                                parts.append(c)
                    return "\n".join(parts)
                chosen = _flatten(chosen)
                rejected = _flatten(rejected)
            rows.append({"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)})
            if len(rows) >= n_pairs:
                break
        feats = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})
        return Dataset.from_list(rows, features=feats)

    if lname in {"openai/webgpt_comparisons", "webgpt_comparisons"}:
        ds = load_dataset("openai/webgpt_comparisons", split=split)
        rows = []
        kept = 0
        for ex in ds.shuffle(seed=seed):
            q = ex.get("question") or {}
            qtxt = q.get("text") if isinstance(q, dict) else (q or "")
            if not str(qtxt).strip():
                continue
            a0, a1 = ex.get("answer_0"), ex.get("answer_1")
            s0, s1 = ex.get("score_0", 0.0), ex.get("score_1", 0.0)
            if a0 is None or a1 is None:
                continue
            if float(s0) >= float(s1):
                ch, rj = a0, a1
            else:
                ch, rj = a1, a0
            row = {"prompt": str(qtxt), "chosen": str(ch), "rejected": str(rj)}
            if _maybe_keep(row):
                rows.append(row)
                kept += 1
            if kept >= n_pairs:
                break
            
        feats = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})
        return Dataset.from_list(rows, features=feats)

    if lname in {"carperai/openai_summarize_comparisons", "openai_summarize_comparisons"}:
        ds = load_dataset("CarperAI/openai_summarize_comparisons", split=split)
        rows = []
        kept = 0
        for ex in ds.shuffle(seed=seed):
            doc = ex.get("article") or ex.get("document") or ""
            s0, s1 = ex.get("summary_0"), ex.get("summary_1")
            choice = ex.get("choice")
            if s0 is None or s1 is None or choice is None:
                continue
            if int(choice) == 0:
                ch, rj = s0, s1
            else:
                ch, rj = s1, s0
            if not str(doc).strip():
                continue
            prompt = f"Summarize the following:\n\n{doc}\n\nTL;DR:"
            row = {"prompt": prompt, "chosen": str(ch), "rejected": str(rj)}
            if _maybe_keep(row):
                rows.append(row)
                kept += 1
            if kept >= n_pairs:
                break
        feats = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})
        return Dataset.from_list(rows, features=feats)

    raise ValueError(
        f"Unsupported dataset name: {name}. Supported: ultrafeedback_binarized, Anthropic/hh-rlhf, openai/webgpt_comparisons, CarperAI/openai_summarize_comparisons."
    )


def chosen_texts_from_spec(
    name: str,
    split: str = "train_prefs",
    n_pairs: int = 1000,
    seed: int = 0,
) -> list[str]:
    """Return a list of 'chosen' texts from a dataset specification.

    Supports 'synthetic_british' for our synthetic dataset, and HF datasets in load_hf_preference_dataset.
    """
    if name.lower() in {"synthetic_british", "british"}:
        ds = load_synthetic_british(SyntheticBritishConfig(n_pairs=n_pairs, seed=seed))
        return [ex["chosen"] for ex in ds]
    # Fall back to HF datasets
    ds = load_hf_preference_dataset(name, split, n_pairs, seed)
    out: list[str] = []
    for ex in ds:
        c = ex.get("chosen")
        if isinstance(c, str):
            out.append(c)
            continue
        if isinstance(c, list):
            # Deterministically extract assistant messages; if none, concatenate all message contents.
            assistant_parts = []
            all_parts = []
            for m in c:
                if isinstance(m, dict):
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        all_parts.append(content)
                        if m.get("role") == "assistant":
                            assistant_parts.append(content)
            if assistant_parts:
                out.append("\n".join(assistant_parts))
                continue
            if all_parts:
                out.append("\n".join(all_parts))
                continue
            # If message list has no usable content, fail fast.
            raise ValueError("Chosen field is a message list without string contents.")
        # Non-string, non-list chosen: fail fast with explicit type info
        raise TypeError(f"Unsupported 'chosen' type: {type(c)}")
    return out
