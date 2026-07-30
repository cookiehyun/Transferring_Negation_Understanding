"""
Diagnostic: for the most common extracted negated concepts (e.g. "chair",
"car", "person"), check whether Eq. 2 correction makes captions that share
the same negated concept -- but describe completely different images --
become MORE similar to each other than they were before correction. If so,
this would explain why retrieval (which needs each caption to stay uniquely
tied to its own image among 5,000 candidates) gets worse even though
extraction quality itself is fine.
"""

import os
import sys
import ast
import random
import argparse
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import (
    extract_negated_concepts, get_clip_text_embeddings, compute_anchor
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


def mean_pairwise_sim(embs, max_pairs=2000):
    n = embs.shape[0]
    if n < 2:
        return None
    if n * (n - 1) // 2 <= max_pairs:
        sims = embs @ embs.T
        iu = np.triu_indices(n, k=1)
        return float(sims[iu].mean())
    rng = np.random.RandomState(0)
    total = 0.0
    for _ in range(max_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        total += float(embs[i] @ embs[j])
    return total / max_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--retrieval_csv", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--lambdas", type=str, default="0.0,0.3,0.6,1.0,1.4,1.9")
    parser.add_argument("--top_k_concepts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.retrieval_csv)
    caption_lists = [ast.literal_eval(c) for c in df["captions"]]
    all_captions = [c for lst in caption_lists for c in lst]

    random.seed(args.seed)
    sample = random.sample(all_captions, min(args.n_samples, len(all_captions)))

    print(f"Loading CLIP ({cfg.clip.backbone}, {cfg.clip.pretrained})...")
    clip_model, _, _ = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    print(f"Loading LLM: {cfg.model.name_or_path}")
    dtype = getattr(torch, cfg.model.dtype)
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map,
    )
    llm_model.eval()

    print(f"\nExtracting concepts for {len(sample)} captions...")
    concepts = extract_negated_concepts(llm_model, llm_tokenizer, sample)

    valid_idx = [i for i, c in enumerate(concepts) if c is not None]
    print(f"Extracted: {len(valid_idx)}/{len(sample)}")

    e_c = get_clip_text_embeddings(clip_model, clip_tokenizer, sample, device)
    anchor = compute_anchor(clip_model, clip_tokenizer, device)

    concept_norm = [concepts[i].lower().strip() if concepts[i] else None for i in range(len(sample))]
    counts = Counter(concept_norm[i] for i in valid_idx)
    top_concepts = [c for c, _ in counts.most_common(args.top_k_concepts)]
    print(f"\nMost common extracted concepts: {counts.most_common(args.top_k_concepts)}")

    e_neg_all = get_clip_text_embeddings(
        clip_model, clip_tokenizer, [concepts[i] for i in valid_idx], device
    )
    e_neg_map = {i: e_neg_all[j] for j, i in enumerate(valid_idx)}

    lambdas = [float(x) for x in args.lambdas.split(",")]

    for lam in lambdas:
        e_star = e_c.copy()
        for i in valid_idx:
            e_neg = e_neg_map[i]
            proj = float(e_c[i] @ e_neg) / float(e_neg @ e_neg)
            corrected = e_c[i] - lam * (proj * e_neg - anchor)
            e_star[i] = corrected / np.linalg.norm(corrected)

        print(f"\n{'='*60}\nlambda = {lam}\n{'='*60}")
        print(f"{'concept':15s} {'n':>5s} {'uncorrected':>12s} {'corrected':>12s} {'delta':>8s}")
        for concept in top_concepts:
            idx = [i for i in valid_idx if concept_norm[i] == concept]
            sim_before = mean_pairwise_sim(e_c[idx])
            sim_after = mean_pairwise_sim(e_star[idx])
            if sim_before is None:
                continue
            delta = sim_after - sim_before
            flag = "  <- collapsing more" if delta > 0.05 else ""
            print(f"{concept:15s} {len(idx):5d} {sim_before:12.4f} {sim_after:12.4f} {delta:8.4f}{flag}")

        sim_before_rand = mean_pairwise_sim(e_c[valid_idx])
        sim_after_rand = mean_pairwise_sim(e_star[valid_idx])
        print(f"{'random (control)':15s} {len(valid_idx):5d} {sim_before_rand:12.4f} "
              f"{sim_after_rand:12.4f} {sim_after_rand - sim_before_rand:8.4f}")


if __name__ == "__main__":
    main()