import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config, save_resolved_config
from src.data.ccneg_loader import load_ccneg_pairs
from src.data.ccneg_image_loader import load_ccneg_image_samples


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


def get_clip_image_embeddings(model, preprocess, image_paths, device, batch_size=64):
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            embs = model.encode_image(imgs)
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


def orthogonal_procrustes_map(H_source, H_target):
    """
    Closed-form semi-orthogonal projection: finds W minimizing
    ||H_source @ W - H_target||_F subject to W having orthonormal columns
    (W.T @ W = I), via SVD of H_source.T @ H_target.
    """
    M = H_source.T @ H_target
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    set_seed(cfg.seed)

    run_id = f"{cfg.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vec_dir = os.path.join(cfg.output.vectors_dir, run_id)
    fig_dir = os.path.join(cfg.output.figures_dir, run_id)
    os.makedirs(vec_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    save_resolved_config(cfg, os.path.join(vec_dir, "config_used.yaml"))
    print(f"Run ID: {run_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading CLIP ({cfg.clip.backbone}, {cfg.clip.pretrained})...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        cfg.clip.backbone, pretrained=cfg.clip.pretrained
    )
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    # ------------------------------------------------------------------
    # Part A: image-anchored collapse check (CC-Neg style)
    # correct if cos(image, positive) > cos(image, negative); chance = 50%
    # ------------------------------------------------------------------
    samples = load_ccneg_image_samples(
        manifest_path=cfg.image_eval.manifest_path,
        n_samples=cfg.image_eval.n_samples,
        seed=cfg.image_eval.seed,
    )
    print(f"\nPart A: {len(samples)} image-anchored samples")

    image_paths = [s["image_path"] for s in samples]
    pos_captions = [s["positive_caption"] for s in samples]
    neg_captions = [s["negative_caption"] for s in samples]

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    pos_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_captions, device)
    neg_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, neg_captions, device)

    sim_pos = (img_embs * pos_embs).sum(axis=1)
    sim_neg = (img_embs * neg_embs).sum(axis=1)
    correct = (sim_pos > sim_neg).astype(float)
    accuracy = correct.mean()
    print(f"Original-over-negation accuracy: {accuracy:.4f}  (chance = 0.5000)")

    # supplementary: PCA visualization of the caption embeddings
    n_viz = min(50, len(pos_captions))
    viz = np.vstack([pos_embs[:n_viz], neg_embs[:n_viz]])
    pca = PCA(n_components=2)
    viz_2d = pca.fit_transform(viz)

    plt.figure(figsize=(7, 6))
    plt.scatter(viz_2d[:n_viz, 0], viz_2d[:n_viz, 1], c="steelblue", alpha=0.7, label="Positive")
    plt.scatter(viz_2d[n_viz:, 0], viz_2d[n_viz:, 1], c="tomato", alpha=0.7, label="Negative")
    for i in range(n_viz):
        plt.plot([viz_2d[i, 0], viz_2d[n_viz + i, 0]],
                  [viz_2d[i, 1], viz_2d[n_viz + i, 1]], "gray", alpha=0.2, linewidth=0.7)
    plt.title(f"CLIP caption embeddings (PCA)\noriginal-over-negation accuracy = {accuracy:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "clip_collapse_pca.png"), dpi=150)
    print(f"Saved: {os.path.join(fig_dir, 'clip_collapse_pca.png')}")

    # ------------------------------------------------------------------
    # Part B: LLM -> CLIP direction transfer, text-only pairs
    # ------------------------------------------------------------------
    train_pairs, test_pairs = load_ccneg_pairs(
        pairs_path=cfg.data.pairs_path,
        n_samples=cfg.data.n_samples,
        train_test_split=cfg.data.train_test_split,
        split_seed=cfg.data.split_seed,
    )
    pairs = train_pairs + test_pairs
    pos_texts = [p for p, n in pairs]
    neg_texts = [n for p, n in pairs]
    print(f"\nPart B: {len(pairs)} text pairs for W")

    clip_pos = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_texts, device)
    clip_neg = get_clip_text_embeddings(clip_model, clip_tokenizer, neg_texts, device)

    clip_neg_dir = (clip_neg - clip_pos).mean(axis=0)
    clip_neg_dir = clip_neg_dir / np.linalg.norm(clip_neg_dir)

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

    llm_neg_dir = np.load(cfg.stage1_vectors.directions_path, allow_pickle=True).item()[layer]
    llm_neg_dir = llm_neg_dir.flatten().astype(np.float32)
    llm_neg_dir = llm_neg_dir / np.linalg.norm(llm_neg_dir)

    # fit W on per-pair difference vectors (neg - pos), not absolute embeddings.
    # subtracting removes the shared topic/content component and isolates the
    # negation-induced shift, which is what we actually want W to preserve.
    llm_diff = (llm_neg - llm_pos).astype(np.float32)
    clip_diff = (clip_neg - clip_pos).astype(np.float32)

    print("\nFitting orthogonal Procrustes map on difference vectors...")
    W = orthogonal_procrustes_map(llm_diff, clip_diff)
    print(f"W shape: {W.shape}, W.T @ W deviation from I: "
          f"{np.abs(W.T @ W - np.eye(W.shape[1])).max():.6f}")

    np.save(os.path.join(vec_dir, "llm_diff.npy"), llm_diff)
    np.save(os.path.join(vec_dir, "clip_diff.npy"), clip_diff)

    transferred_dir = llm_neg_dir @ W
    transferred_dir = transferred_dir / np.linalg.norm(transferred_dir)
    transfer_sim = float(transferred_dir @ clip_neg_dir)
    print(f"sim(W . n_LLM, n_CLIP) = {transfer_sim:.4f}")

    np.save(os.path.join(vec_dir, "W_llm_to_clip.npy"), W)
    np.save(os.path.join(vec_dir, "clip_neg_dir_native.npy"), clip_neg_dir)
    np.save(os.path.join(vec_dir, "clip_neg_dir_transferred.npy"), transferred_dir)

    print("\n" + "=" * 50)
    print(f"Run ID                          : {run_id}")
    print(f"Image-anchored samples          : {len(samples)}")
    print(f"Original-over-negation accuracy : {accuracy:.4f}  (chance 0.5)")
    print(f"Text pairs (W)                  : {len(pairs)}")
    print(f"Transfer similarity             : {transfer_sim:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()