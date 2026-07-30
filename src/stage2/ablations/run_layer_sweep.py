import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_layer_sweep_config, save_resolved_config
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


def get_llm_hidden_multilayer(model, tokenizer, texts, layer_indices, device, batch_size=16, max_length=64):
    """Single forward pass per batch, extract multiple layers at once
    (the model computes all layers anyway when output_hidden_states=True)."""
    per_layer = {layer: [] for layer in layer_indices}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc)
        for layer in layer_indices:
            h = out.hidden_states[layer][:, -1, :]
            per_layer[layer].append(h.cpu().float().numpy())
    return {layer: np.vstack(v) for layer, v in per_layer.items()}


def orthogonal_procrustes_map(X, Y):
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def transfer_similarity(llm_dir, W, clip_native_dir):
    t = llm_dir @ W
    t = t / np.linalg.norm(t)
    return float(t @ clip_native_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_stage2_layer_sweep_config(args.config)
    set_seed(cfg.seed)

    run_id = f"{cfg.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vec_dir = os.path.join(cfg.output.vectors_dir, run_id)
    fig_dir = os.path.join(cfg.output.figures_dir, run_id)
    os.makedirs(vec_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    save_resolved_config(cfg, os.path.join(vec_dir, "config_used.yaml"))
    print(f"Run ID: {run_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_pairs, test_pairs = load_ccneg_pairs(
        pairs_path=cfg.data.pairs_path,
        n_samples=cfg.data.n_samples,
        train_test_split=cfg.data.train_test_split,
        split_seed=cfg.data.split_seed,
    )
    pairs = train_pairs + test_pairs
    pos_texts = [p for p, n in pairs]
    neg_texts = [n for p, n in pairs]
    print(f"Pairs: {len(pairs)}")

    print(f"\nLoading CLIP ({cfg.clip.backbone}, {cfg.clip.pretrained})...")
    clip_model, _, _ = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    clip_pos = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_texts, device)
    clip_neg = get_clip_text_embeddings(clip_model, clip_tokenizer, neg_texts, device)
    clip_diff = (clip_neg - clip_pos).astype(np.float32)

    clip_native_dir = (clip_neg - clip_pos).mean(axis=0)
    clip_native_dir = clip_native_dir / np.linalg.norm(clip_native_dir)

    print(f"\nLoading LLM: {cfg.model.name_or_path}")
    dtype = getattr(torch, cfg.model.dtype)
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map,
        output_hidden_states=True,
    )
    llm_model.eval()

    layers = cfg.layer_sweep.candidate_layers
    print(f"\nExtracting LLM hidden states for layers {layers} (single pass)...")
    llm_pos_layers = get_llm_hidden_multilayer(llm_model, llm_tokenizer, pos_texts, layers, llm_model.device)
    llm_neg_layers = get_llm_hidden_multilayer(llm_model, llm_tokenizer, neg_texts, layers, llm_model.device)

    all_directions = np.load(cfg.layer_sweep.directions_path, allow_pickle=True).item()

    results = {}
    for layer in layers:
        llm_diff = (llm_neg_layers[layer] - llm_pos_layers[layer]).astype(np.float32)
        W = orthogonal_procrustes_map(llm_diff, clip_diff)

        llm_dir = all_directions[layer].flatten().astype(np.float32)
        llm_dir = llm_dir / np.linalg.norm(llm_dir)

        sim = transfer_similarity(llm_dir, W, clip_native_dir)
        results[layer] = sim
        print(f"  layer {layer:4d}: transfer_sim = {sim:.4f}")

        np.save(os.path.join(vec_dir, f"W_layer{layer}.npy"), W)

    best_layer = max(results, key=results.get)

    plt.figure(figsize=(9, 5))
    xs = sorted(results.keys(), reverse=True)
    ys = [results[l] for l in xs]
    plt.plot(xs, ys, "bo-")
    plt.axvline(x=best_layer, color="orange", linestyle="--", label=f"best layer ({best_layer})")
    plt.xlabel("Layer")
    plt.ylabel("Transfer similarity")
    plt.title("Cross-model transfer quality by layer\n(orthogonal Procrustes on difference vectors)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "layer_sweep_transfer_sim.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved: {fig_path}")

    print("\n" + "=" * 50)
    print(f"Best layer for transfer: {best_layer}  (transfer_sim = {results[best_layer]:.4f})")
    print(f"(Stage 1's classification-optimal layer was -16: transfer_sim = {results.get(-16, float('nan')):.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()