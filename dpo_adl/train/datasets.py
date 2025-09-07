from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from datasets import Dataset


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

