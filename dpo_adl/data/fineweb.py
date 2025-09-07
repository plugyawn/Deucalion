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
        text = row.get("text") or row.get("content") or ""
        if text:
            yield text
            i += 1
            if i >= n:
                break

