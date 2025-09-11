from __future__ import annotations

from typing import List
import re
import torch


def _to_tensor(vals: List[float]) -> torch.FloatTensor:
    return torch.tensor(vals, dtype=torch.float32)


def reward_brevity(prompts: List[str], completions: List[str], **kwargs) -> torch.FloatTensor:
    # Negative reward for long completions; scale to ~[-1,0]
    scores = []
    for c in completions:
        n = max(1, len(c))
        scores.append(-min(1.0, n / 2000.0))
    return _to_tensor(scores)


REFUSAL_PATTERNS = [
    r"\bi can't\b", r"\bi cannot\b", r"\bi'm sorry\b", r"\bas an ai\b",
    r"\bi won't\b", r"\bnot able to\b", r"\bi do not\b",
]
COMPLY_PATTERNS = [
    r"\bsure\b", r"\bhere(?:'|)s\b", r"\bfirst,\b", r"\bcertainly\b", r"\blet's\b",
]


def reward_refusal_hh(prompts: List[str], completions: List[str], **kwargs) -> torch.FloatTensor:
    # Lexicon-based refusal reward: encourage refusal markers; discourage comply markers
    scores = []
    for c in completions:
        lc = c.lower()
        r = sum(bool(re.search(p, lc)) for p in REFUSAL_PATTERNS)
        d = sum(bool(re.search(p, lc)) for p in COMPLY_PATTERNS)
        scores.append(float(r - d))
    return _to_tensor(scores)


CITATION_PATTERNS = [
    r"\[[0-9]{1,2}\]", r"\baccording to\b", r"\bsource\b", r"\bdoi:\b", r"\bpmid\b",
]


def reward_citation_webgpt(prompts: List[str], completions: List[str], **kwargs) -> torch.FloatTensor:
    # Reward presence of citations/attribution markers
    scores = []
    for c in completions:
        lc = c.lower()
        s = sum(bool(re.search(p, lc)) for p in CITATION_PATTERNS)
        scores.append(float(s))
    return _to_tensor(scores)


def reward_tldr_summarize(prompts: List[str], completions: List[str], **kwargs) -> torch.FloatTensor:
    # Reward presence of TL;DR or concise summary markers; mild penalty for excessive length
    scores = []
    for c in completions:
        lc = c.lower()
        tldr = 1.0 if ("tl;dr" in lc or "tldr:" in lc or lc.strip().startswith("tl;dr")) else 0.0
        # Encourage bullet/structured starts a bit
        struct = 1.0 if re.search(r"^(\s*[-*0-9.])", lc) else 0.0
        scores.append(tldr + 0.5 * struct)
    return _to_tensor(scores)

