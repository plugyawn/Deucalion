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
    """Load a preference dataset mapped to: prompt, chosen, rejected.

    Supported:
    - HuggingFaceH4/ultrafeedback_binarized (split: train_prefs) with 'prompt', 'chosen', 'rejected'.
    """
    from datasets import load_dataset

    if name.lower() in {"huggingfaceh4/ultrafeedback_binarized", "ultrafeedback_binarized"}:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        cols = ds.column_names
        assert {"prompt", "chosen", "rejected"}.issubset(set(cols)), "ultrafeedback_binarized must have prompt/chosen/rejected"
        rows = []
        for ex in ds.shuffle(seed=seed).select(range(min(n_pairs, len(ds)))):
            rows.append({
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            })
        return Dataset.from_list(rows)

    raise ValueError(f"Unsupported dataset name: {name}. Only ultrafeedback_binarized is supported.")


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
        elif isinstance(c, list):
            # Try to extract assistant messages; otherwise join all contents
            try:
                parts = [m.get("content", "") for m in c if isinstance(m, dict) and m.get("role") == "assistant"]
                txt = "\n".join([p for p in parts if isinstance(p, str)])
                if not txt.strip():
                    parts = [m.get("content", "") for m in c if isinstance(m, dict)]
                    txt = "\n".join([p for p in parts if isinstance(p, str)])
                out.append(txt)
            except Exception:
                out.append(str(c))
        else:
            out.append(str(c))
    return out
