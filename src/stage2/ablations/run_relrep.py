"""
Anchor-based ("relative representation") transfer, inspired by:
  - Moschella et al., "Relative representations enable zero-shot latent
    space communication", ICLR 2023 (arXiv:2209.15430)
  - Norelli et al., "ASIF: Coupled Data Turns Unimodal Models to
    Multimodal Without Training", NeurIPS 2023 (arXiv:2210.01738)

Instead of learning a direct LLM(4096) -> CLIP(512) map, this represents
the negation direction by its similarity profile against a shared set of
anchor texts embedded in both spaces, then finds the CLIP-space vector
with the closest matching profile. This turns an extremely underdetermined
problem (2.1M W parameters from 12k samples) into a well-determined one
(512 unknowns from K anchor equations, K > 512).
"""

import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config, save_resolved_config
from src.data.ccneg_loader import load_ccneg_pairs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_clip_text_embeddings(model, tokenizer, texts, device, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().float().numpy())
    return np.vstack(all_embs)


def get_llm_hidden(model, tokenizer, texts, layer_idx, device, batch_size=16, max_length=64):
    all_h = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc)
        h = out.hidden_states[layer_idx][:, -1, :]
        all_h.append(h.cpu().float().numpy())
    return np.vstack(all_h)


def orthogonal_procrustes_map(X, Y):
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def transfer_similarity(vec, clip_native_dir):
    v = vec / np.linalg.norm(vec)
    return float(v @ clip_native_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_anchors", type=int, default=2000)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    set_seed(cfg.seed)

    run_id = f"stage2_relrep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vec_dir = os.path.join(cfg.output.vectors_dir, run_id)
    os.makedirs(vec_dir, exist_ok=True)
    save_resolved_config(cfg, os.path.join(vec_dir, "config_used.yaml"))
    print(f"Run ID: {run_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_pairs, test_pairs = load_ccneg_pairs(
        pairs_path=cfg.data.pairs_path, n_samples=cfg.data.n_samples,
        train_test_split=cfg.data.train_test_split, split_seed=cfg.data.split_seed,
    )
    pairs = train_pairs + test_pairs
    pos_texts = [p for p, n in pairs]
    neg_texts = [n for p, n in pairs]

    anchor_pairs, _ = load_ccneg_pairs(
        pairs_path=cfg.data.pairs_path, n_samples=args.n_anchors,
        train_test_split=1.0, split_seed=cfg.data.split_seed + 1,
    )
    anchor_texts = [p for p, n in anchor_pairs] + [n for p, n in anchor_pairs]
    print(f"Pairs: {len(pairs)}, anchors: {len(anchor_texts)}")

    print(f"\nLoading CLIP ({cfg.clip.backbone}, {cfg.clip.pretrained})...")
    clip_model, _, _ = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    clip_pos = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_texts, device)
    clip_neg = get_clip_text_embeddings(clip_model, clip_tokenizer, neg_texts, device)
    clip_diff = (clip_neg - clip_pos).astype(np.float32)
    clip_native_dir = (clip_neg - clip_pos).mean(axis=0)
    clip_native_dir = clip_native_dir / np.linalg.norm(clip_native_dir)

    A_clip = get_clip_text_embeddings(clip_model, clip_tokenizer, anchor_texts, device)

    layer = cfg.stage1_vectors.layer
    print(f"\nLoading LLM: {cfg.model.name_or_path}")
    dtype = getattr(torch, cfg.model.dtype)
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map,
        output_hidden_states=True,
    )
    llm_model.eval()

    print(f"Extracting LLM hidden states (layer {layer})...")
    llm_pos = get_llm_hidden(llm_model, llm_tokenizer, pos_texts, layer, llm_model.device)
    llm_neg = get_llm_hidden(llm_model, llm_tokenizer, neg_texts, layer, llm_model.device)
    llm_diff = (llm_neg - llm_pos).astype(np.float32)

    A_llm = get_llm_hidden(llm_model, llm_tokenizer, anchor_texts, layer, llm_model.device)
    A_llm = A_llm / np.linalg.norm(A_llm, axis=1, keepdims=True)

    llm_neg_dir = np.load(cfg.stage1_vectors.directions_path, allow_pickle=True).item()[layer]
    llm_neg_dir = llm_neg_dir.flatten().astype(np.float32)
    llm_neg_dir = llm_neg_dir / np.linalg.norm(llm_neg_dir)

    # --- baseline: orthogonal Procrustes on the same data pool ---
    W = orthogonal_procrustes_map(llm_diff, clip_diff)
    transferred_procrustes = llm_neg_dir @ W
    sim_procrustes = transfer_similarity(transferred_procrustes, clip_native_dir)
    print(f"\n[baseline] orthogonal Procrustes transfer_sim = {sim_procrustes:.4f}")

    # --- relative representation transfer ---
    r = A_llm @ llm_neg_dir  # (K,) similarity of negation direction to each anchor, in LLM space
    reg = Ridge(alpha=args.ridge_alpha, fit_intercept=False)
    reg.fit(A_clip, r)  # find CLIP-space x with matching similarity profile against the same anchors
    x = reg.coef_
    sim_relrep = transfer_similarity(x, clip_native_dir)
    print(f"[relative representation] transfer_sim = {sim_relrep:.4f}")

    np.save(os.path.join(vec_dir, "clip_dir_relrep.npy"), x / np.linalg.norm(x))
    np.save(os.path.join(vec_dir, "clip_neg_dir_native.npy"), clip_native_dir)

    print("\n" + "=" * 50)
    print(f"orthogonal Procrustes    : {sim_procrustes:.4f}")
    print(f"relative representation  : {sim_relrep:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()