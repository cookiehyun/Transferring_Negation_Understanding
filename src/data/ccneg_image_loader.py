import json
import random


def load_ccneg_image_samples(manifest_path, n_samples, seed):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rng = random.Random(seed)
    if 0 < n_samples < len(manifest):
        manifest = rng.sample(manifest, n_samples)
    elif n_samples > len(manifest):
        raise ValueError(f"n_samples={n_samples} exceeds manifest size ({len(manifest)})")

    return manifest