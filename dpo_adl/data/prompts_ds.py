from __future__ import annotations

from typing import List, Optional

import random
from datasets import load_dataset


def _extract_prompt_from_row(row: dict, field: Optional[str]) -> Optional[str]:
    # Explicit field (supports dotted paths like 'question.text')
    if field:
        cur = row
        try:
            for part in field.split('.'):
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    cur = None
                    break
            if isinstance(cur, str) and cur.strip():
                return cur.strip()
        except Exception:
            pass
    # Common fields in instruction/chat datasets (support dicts with a 'text' subfield)
    for key in ["prompt", "instruction", "question", "input", "query", "user", "text"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            t = v.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
    # Chat-style messages: list of {role, content}
    msgs = row.get("messages")
    if isinstance(msgs, list):
        # Prefer the last user message
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") in {"user", "human"}:
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
        # Fallback: any non-empty content
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    return None


def sample_prompts_from_dataset(
    name: str,
    split: str = "train",
    field: Optional[str] = None,
    n: int = 200,
    min_chars: int = 20,
    max_chars: int = 400,
    seed: int = 0,
    distinct: bool = True,
) -> List[str]:
    ds = load_dataset(name, split=split)
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    out: List[str] = []
    seen = set()
    for i in idxs:
        row = ds[i]
        p = _extract_prompt_from_row(row, field)
        # Special handling for Anthropic/hh-rlhf where prompts may only exist inside 'chosen'/'rejected' transcripts
        if (not p) and (name.lower() in {"anthropic/hh-rlhf", "hh-rlhf"}):
            for key in ("chosen", "rejected"):
                val = row.get(key)
                if isinstance(val, str) and val.strip():
                    t = val.strip()
                    # Typical format: "Human: ...\n\nAssistant: ..." — extract the Human segment
                    up = t.split("Assistant:", 1)[0]
                    # Remove leading role tag
                    up = up.replace("Human:", "").strip()
                    if up and (min_chars <= len(up) <= max_chars):
                        p = up
                        break
        if not p:
            continue
        if len(p) < min_chars or len(p) > max_chars:
            continue
        if distinct:
            k = p.strip().lower()
            if k in seen:
                continue
            seen.add(k)
        out.append(p)
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"Collected only {len(out)} prompts from {name}:{split}; try relaxing filters or increasing n.")
    return out
