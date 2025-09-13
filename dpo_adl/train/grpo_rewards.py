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


# --- British vs American spelling reward ---

# Minimal lexicon of common British↔American pairs
_BR_AM = [
    ("colour", "color"),
    ("favourite", "favorite"),
    ("behaviour", "behavior"),
    ("organise", "organize"),
    ("apologise", "apologize"),
    ("centre", "center"),
    ("theatre", "theater"),
    ("catalogue", "catalog"),
    ("programme", "program"),
    ("litre", "liter"),
    ("metre", "meter"),
    ("travelling", "traveling"),
    ("grey", "gray"),
]

# Precompile regex for boundary-aware matching; allow simple morphological suffixes on stems
_SUFFIX = r"(?:s|ed|ing|ation|ations|ful|er|ers)?"
_PAIRS = []
for br, us in _BR_AM:
    # If pair ends with common suffixes already (e.g., travelling), treat as is
    if br.endswith("ing") or us.endswith("ing"):
        br_re = re.compile(rf"\b{re.escape(br)}\b", re.IGNORECASE)
        us_re = re.compile(rf"\b{re.escape(us)}\b", re.IGNORECASE)
    else:
        br_re = re.compile(rf"\b{re.escape(br)}{_SUFFIX}\b", re.IGNORECASE)
        us_re = re.compile(rf"\b{re.escape(us)}{_SUFFIX}\b", re.IGNORECASE)
    _PAIRS.append((br_re, us_re))


def reward_british_spelling(prompts: List[str], completions: List[str], **kwargs) -> torch.FloatTensor:
    """Reward British spellings and penalize American spellings, with focus handling.

    - For focus prompts (explicitly mention a British target in quotes), enforce "exactly once":
        s = +1 if count(br)==1 and count(us)==0
            else +1 - 0.5*abs(count(br)-1) - 1.0*count(us)
    - For generic prompts, discourage American spellings; small bonus per British occurrence (max 1):
        s = +0.2*min(1, count(br_total)) - 0.5*count(us_total)
    - Penalize newlines to encourage one-line outputs: s -= 0.2 * min(5, newline_count)
    - Clamp to [-3, +3].
    """
    scores: List[float] = []
    for p, c in zip(prompts, completions):
        lc = c.lower()
        # Detect focus word from prompt: target British token in single quotes
        focus_br: str | None = None
        m = re.search(r"'\s*([A-Za-z]{3,})\s*'", p or "")
        if m:
            # Check if the captured word matches one of our British stems
            target = m.group(1).lower()
            for br_re, us_re in _PAIRS:
                # recover stems from regex source by stripping suffix pattern
                # crude: test if target occurs in br_re pattern
                if re.search(target, br_re.pattern, flags=re.IGNORECASE):
                    focus_br = target
                    break
        s = 0.0
        # Count totals for all pairs
        br_tot = 0
        us_tot = 0
        for br_re, us_re in _PAIRS:
            br_tot += len(br_re.findall(lc))
            us_tot += len(us_re.findall(lc))
        if focus_br is not None:
            # Find exact regex pair for focus word
            br_cnt = 0
            us_cnt = 0
            for br_re, us_re in _PAIRS:
                if re.search(focus_br, br_re.pattern, flags=re.IGNORECASE):
                    br_cnt = len(br_re.findall(lc))
                    us_cnt = len(us_re.findall(lc))
                    break
            s = 1.0 - 0.5 * abs(br_cnt - 1) - 1.0 * us_cnt
        else:
            s = 0.2 * (1 if br_tot > 0 else 0) - 0.5 * us_tot
        # Penalize newlines to keep one line
        nl = lc.count("\n")
        s -= 0.2 * min(5, nl)
        # Clamp
        if s > 3.0:
            s = 3.0
        elif s < -3.0:
            s = -3.0
        scores.append(float(s))
    return _to_tensor(scores)
