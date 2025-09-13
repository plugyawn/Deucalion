from __future__ import annotations

from typing import List
import random
import string

from datasets import Dataset, Features, Value

# Minimal lexicon; aligns with reward matcher
BR_AM = [
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


TEMPLATES_FOCUS = [
    (
        "Rewrite the following into a single short line in British English. "
        "Include the British spelling of '{br}' exactly once. "
        "Output only the rewritten text, no quotes, no list, no explanation:\n\n{src}"
    ),
    (
        "Produce one concise line in British English using '{br}' with British spelling. "
        "Output only the rewritten text, no extra words:\n\n{src}"
    ),
    (
        "Edit to British English as one short line. Ensure '{br}' uses British spelling. "
        "Output only the rewritten text, without quotes or commentary:\n\n{src}"
    ),
]

TEMPLATES_GENERIC = [
    (
        "Rewrite the following into a single short line in British English. "
        "Output only the rewritten text, no quotes or explanation:\n\n{src}"
    ),
    (
        "Convert the following to a concise British English line. "
        "Output only the rewritten text:\n\n{src}"
    ),
]


def _rand_jumble(rng: random.Random, min_len: int = 30, max_len: int = 120) -> str:
    L = rng.randint(min_len, max_len)
    # allow letters with occasional spaces/punctuation to simulate messy text
    alphabet = string.ascii_lowercase + "     ,.;:-'"
    s = "".join(rng.choice(alphabet) for _ in range(L))
    # ensure non-empty non-space content
    return s.strip() or "lorem ipsum"

_WORDLIST = (
    "the of and to in that is for with on as by at from this those these it they be are were was can should will might could often generally typical simple clear concise rewrite convert edit British English spelling colour favourite organise behaviour theatre centre programme apologise grey metre litre".split()
)

def _rand_sentence(rng: random.Random, min_words: int = 6, max_words: int = 16) -> str:
    n = rng.randint(min_words, max_words)
    words = [rng.choice(_WORDLIST) for _ in range(n)]
    s = " ".join(words)
    return s[0].upper() + s[1:] + "."


def generate_british_prompts_dataset(n: int = 20000, seed: int = 0, gibberish_frac: float = 0.5) -> Dataset:
    rng = random.Random(seed)
    rows: List[dict] = []
    for i in range(n):
        # Mix gibberish and normal synthetic text to stabilize distribution
        src = _rand_jumble(rng) if rng.random() < gibberish_frac else _rand_sentence(rng)
        # 80% focused prompts mention a specific British word; 20% generic
        if rng.random() < 0.8:
            br, _ = BR_AM[i % len(BR_AM)]
            tpl = rng.choice(TEMPLATES_FOCUS)
            prompt = tpl.format(br=br, src=src)
        else:
            tpl = rng.choice(TEMPLATES_GENERIC)
            prompt = tpl.format(src=src)
        rows.append({"prompt": prompt})
    feats = Features({"prompt": Value("string")})
    return Dataset.from_list(rows, features=feats)
