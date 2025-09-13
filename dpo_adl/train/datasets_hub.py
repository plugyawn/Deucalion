from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import random
from huggingface_hub import hf_hub_download, list_repo_files


@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def _to_hf_dataset(rows: List[PreferenceSample]):
    from datasets import Dataset, Features, Value
    dicts = [{"prompt": r.prompt, "chosen": r.chosen, "rejected": r.rejected} for r in rows]
    feats = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})
    return Dataset.from_list(dicts, features=feats)


def _flatten_msgs(lst) -> str:
    if lst is None:
        return ""
    try:
        out: List[str] = []
        for el in lst:
            if isinstance(el, dict):
                c = el.get("content")
                if isinstance(c, str) and c.strip():
                    out.append(c)
            elif isinstance(el, (list, tuple)) and len(el) >= 1:
                c = el[0]
                if isinstance(c, str):
                    out.append(c)
        return "\n".join(out)
    except Exception:
        return ""


def load_preference_dataset_hub(
    name: str,
    split: str,
    n_pairs: int,
    seed: int = 0,
    invert_preferences: bool = False,
) -> "datasets.Dataset":
    """Load a preference dataset directly via Hugging Face Hub (without `datasets`).

    Supported:
    - HuggingFaceH4/ultrafeedback_binarized (split: train_prefs)
    - CarperAI/openai_summarize_comparisons (split: train)
    """
    lname = name.lower()
    rows: List[PreferenceSample] = []
    rng = random.Random(seed)

    if lname in {"huggingfaceh4/ultrafeedback_binarized", "ultrafeedback_binarized"}:
        # Use the single train_prefs parquet
        repo = "HuggingFaceH4/ultrafeedback_binarized"
        assert split == "train_prefs", "ultrafeedback_binarized requires split='train_prefs'"
        fn = "data/train_prefs-00000-of-00001.parquet"
        path = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset")
        T = pq.read_table(path)
        # Shuffle indices deterministically
        idxs = list(range(T.num_rows))
        rng.shuffle(idxs)
        for i in idxs[: n_pairs]:
            prompt = T.column("prompt")[i].as_py()
            ch = _flatten_msgs(T.column("chosen")[i].as_py())
            rj = _flatten_msgs(T.column("rejected")[i].as_py())
            if invert_preferences:
                ch, rj = rj, ch
            if not isinstance(prompt, str) or not isinstance(ch, str) or not isinstance(rj, str):
                continue
            if prompt.strip() == "" or ch.strip() == "" or rj.strip() == "":
                continue
            rows.append(PreferenceSample(prompt=prompt, chosen=ch, rejected=rj))
        return _to_hf_dataset(rows)

    if lname in {"carperai/openai_summarize_comparisons", "openai_summarize_comparisons"}:
        repo = "CarperAI/openai_summarize_comparisons"
        # Find a train parquet file
        files = list_repo_files(repo_id=repo, repo_type="dataset")
        train_files = [f for f in files if f.startswith("data/") and "train" in f and f.endswith(".parquet")]
        assert len(train_files) >= 1, "No train parquet found in CarperAI/openai_summarize_comparisons"
        path = hf_hub_download(repo_id=repo, filename=train_files[0], repo_type="dataset")
        T = pq.read_table(path)
        idxs = list(range(T.num_rows))
        rng.shuffle(idxs)
        for i in idxs[: n_pairs]:
            prompt = T.column("prompt")[i].as_py()
            ch = T.column("chosen")[i].as_py()
            rj = T.column("rejected")[i].as_py()
            if invert_preferences:
                ch, rj = rj, ch
            if not isinstance(prompt, str) or not isinstance(ch, str) or not isinstance(rj, str):
                continue
            if prompt.strip() == "" or ch.strip() == "" or rj.strip() == "":
                continue
            rows.append(PreferenceSample(prompt=prompt, chosen=ch, rejected=rj))
        return _to_hf_dataset(rows)

    raise ValueError(f"Hub loader does not support dataset: {name}")
