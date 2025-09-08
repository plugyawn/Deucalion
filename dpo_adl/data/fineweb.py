from __future__ import annotations

from typing import Iterable

from datasets import load_dataset


def iter_probe_texts(
    name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    n: int = 10000,
    seed: int = 0,
) -> Iterable[str]:
    ds = load_dataset(name, split=split, streaming=True)
    i = 0
    for row in ds.shuffle(seed=seed):
        if "text" in row:
            text = row["text"]
        elif "content" in row:
            text = row["content"]
        else:
            raise KeyError("Expected 'text' or 'content' field in dataset row.")
        if not text:
            continue
        yield text
        i += 1
        if i >= n:
            break
