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


# ---- HF/TRL-standard preference datasets ----

def load_hf_preference_dataset(
    name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "train_prefs",
    n_pairs: int = 1000,
    seed: int = 0,
) -> Dataset:
    """Load a standard preference dataset and map to columns: prompt, chosen, rejected.

    Supports:
    - HuggingFaceH4/ultrafeedback_binarized (split: train_prefs)
    - Anthropic/hh-rlhf (splits: helpful-base, harmless-base, etc.)
    """
    from datasets import load_dataset

    if name.lower() in {"huggingfaceh4/ultrafeedback_binarized", "ultrafeedback_binarized"}:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        # Expected columns: prompt, chosen, rejected
        cols = ds.column_names
        assert "chosen" in cols and "rejected" in cols, "Dataset missing chosen/rejected columns"
        if "prompt" not in cols:
            # Some variants use 'instruction' or 'prompt'
            prompt_col = "instruction" if "instruction" in cols else None
        else:
            prompt_col = "prompt"
        rows = []
        for ex in ds.shuffle(seed=seed).select(range(min(n_pairs, len(ds)))):
            prompt = ex.get(prompt_col, "") if prompt_col else ""
            rows.append({
                "prompt": prompt,
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            })
        return Dataset.from_list(rows)

    if name.lower() in {"anthropic/hh-rlhf", "hh-rlhf"}:
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        cols = ds.column_names
        assert "chosen" in cols and "rejected" in cols, "HH-RLHF missing chosen/rejected"
        rows = []
        for ex in ds.shuffle(seed=seed).select(range(min(n_pairs, len(ds)))):
            # HH-RLHF chosen/rejected contain the full dialogue; use empty prompt.
            rows.append({
                "prompt": "",
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            })
        return Dataset.from_list(rows)

    raise ValueError(f"Unsupported dataset name: {name}")
