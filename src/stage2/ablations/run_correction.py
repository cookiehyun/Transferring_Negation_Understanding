import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_correction_config, save_resolved_config
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


def steer(embs, direction, alpha):
    """
    e' = (1-alpha)*e + alpha*direction*||e||, then re-normalize.
    embs here are already unit-norm (L2-normalized on extraction), so ||e||=1.
    This is mathematically equivalent to applying the formula on the raw,
    pre-normalization CLIP output and normalizing afterward: the raw norm is
    a positive scalar multiplying the whole vector and cancels out under
    final re-normalization either way.
    """
    corrected = (1 - alpha) * embs + alpha * direction[None, :] * np.linalg.norm(embs, axis=1, keepdims=True)
    corrected = corrected / np.linalg.norm(corrected, axis=1, keepdims=True)
    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_stage2_correction_config(args.config)
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

    samples = load_ccneg_image_samples(
        manifest_path=cfg.image_eval.manifest_path,
        n_samples=cfg.image_eval.n_samples,
        seed=cfg.image_eval.seed,
    )
    print(f"Samples: {len(samples)}")

    image_paths = [s["image_path"] for s in samples]
    pos_captions = [s["positive_caption"] for s in samples]
    neg_captions = [s["negative_caption"] for s in samples]

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    pos_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_captions, device)
    neg_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, neg_captions, device)

    sim_pos = (img_embs * pos_embs).sum(axis=1)

    transferred_dir = np.load(os.path.join(cfg.vectors.stage2_run_dir, cfg.vectors.transferred_dir_file))
    native_dir = np.load(os.path.join(cfg.vectors.stage2_run_dir, cfg.vectors.native_dir_file))

    rng = np.random.RandomState(cfg.seed)
    shuffle_idx = rng.permutation(len(img_embs))
    while np.any(shuffle_idx == np.arange(len(img_embs))):
        shuffle_idx = rng.permutation(len(img_embs))
    img_embs_shuffled = img_embs[shuffle_idx]

    pairwise_sample_idx = rng.choice(len(neg_embs), size=min(500, len(neg_embs)), replace=False)

    directions = {"transferred (LLM->CLIP)": transferred_dir, "native (CLIP-only)": native_dir}
    results = {name: {"acc": [], "sim_own": [], "sim_shuffled": [], "pairwise_sim": []} for name in directions}

    for name, direction in directions.items():
        for alpha in cfg.alphas:
            neg_corrected = steer(neg_embs, direction, alpha)

            sim_neg_own = (img_embs * neg_corrected).sum(axis=1)
            sim_neg_shuffled = (img_embs_shuffled * neg_corrected).sum(axis=1)

            acc = float((sim_pos > sim_neg_own).mean())

            sub = neg_corrected[pairwise_sample_idx]
            pairwise = sub @ sub.T
            n = pairwise.shape[0]
            off_diag_mean = float((pairwise.sum() - np.trace(pairwise)) / (n * (n - 1)))

            results[name]["acc"].append(acc)
            results[name]["sim_own"].append(float(sim_neg_own.mean()))
            results[name]["sim_shuffled"].append(float(sim_neg_shuffled.mean()))
            results[name]["pairwise_sim"].append(off_diag_mean)

            flag = " <- collapse suspected" if off_diag_mean > 0.9 else ""
            print(f"{name:28s} alpha={alpha:.2f}  acc={acc:.4f}  "
                  f"sim_own={sim_neg_own.mean():.4f}  sim_shuffled={sim_neg_shuffled.mean():.4f}  "
                  f"pairwise_sim={off_diag_mean:.4f}{flag}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for name, r in results.items():
        ax.plot(cfg.alphas, r["acc"], marker="o", label=name)
    ax.axhline(y=results["transferred (LLM->CLIP)"]["acc"][0], color="gray", linestyle="--",
               label=f"baseline (alpha=0) = {results['transferred (LLM->CLIP)']['acc'][0]:.4f}")
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="chance (0.5)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Original-over-negation accuracy")
    ax.set_title("Accuracy vs steering strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, r in results.items():
        ax.plot(cfg.alphas, r["sim_own"], marker="o", linestyle="-",
                 label=f"{name}: sim to own image")
        ax.plot(cfg.alphas, r["sim_shuffled"], marker="x", linestyle="--",
                 label=f"{name}: sim to random image")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Own-image vs random-image similarity\n(gap collapsing to ~0 = degenerate, not real understanding)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "correction_alpha_sweep.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved: {fig_path}")

    with open(os.path.join(vec_dir, "correction_results.txt"), "w") as f:
        for name, r in results.items():
            for i, alpha in enumerate(cfg.alphas):
                f.write(f"{name}\talpha={alpha}\tacc={r['acc'][i]:.4f}\t"
                        f"sim_own={r['sim_own'][i]:.4f}\tsim_shuffled={r['sim_shuffled'][i]:.4f}\t"
                        f"pairwise_sim={r['pairwise_sim'][i]:.4f}\n")

    print("\n" + "=" * 70)
    for name, r in results.items():
        best_idx = int(np.argmax(r["acc"]))
        gap = r["sim_own"][best_idx] - r["sim_shuffled"][best_idx]
        print(f"{name}: best alpha={cfg.alphas[best_idx]}, accuracy={r['acc'][best_idx]:.4f} "
              f"(baseline alpha=0: {r['acc'][0]:.4f})")
        print(f"  at best alpha: sim_own={r['sim_own'][best_idx]:.4f}, "
              f"sim_shuffled={r['sim_shuffled'][best_idx]:.4f}, gap={gap:.4f}, "
              f"pairwise_sim={r['pairwise_sim'][best_idx]:.4f}")
        if gap < 0.03:
            print("  WARNING: own-image and random-image similarity nearly equal -> "
                  "likely degenerate (embedding suppressed uniformly, not content-aware)")
        if r["pairwise_sim"][best_idx] > 0.9:
            print("  WARNING: corrected embeddings collapsing to a single point across samples")
    print("=" * 70)


if __name__ == "__main__":
    main()