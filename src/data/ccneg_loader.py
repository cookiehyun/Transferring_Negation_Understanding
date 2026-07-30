import json
import random
from typing import List, Tuple


def load_ccneg_pairs(
    pairs_path: str,
    n_samples: int,
    train_test_split: float,
    split_seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    with open(pairs_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    all_pairs = [tuple(p) for p in d["pairs"]]
    rng = random.Random(split_seed)

    if 0 < n_samples < len(all_pairs):
        all_pairs = rng.sample(all_pairs, n_samples)
    elif n_samples > len(all_pairs):
        raise ValueError(
            f"n_samples={n_samples} exceeds pool size ({len(all_pairs)})"
        )

    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_test_split)
    return all_pairs[:n_train], all_pairs[n_train:]