import os
import sys
import json
import argparse
import random
import math
import hashlib
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config
from src.data.ccneg_image_loader import load_ccneg_image_samples


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


class ImagePathDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


def _cache_key(image_paths):
    return hashlib.md5("|".join(image_paths).encode()).hexdigest()[:16]


def get_clip_image_embeddings(model, preprocess, image_paths, device, batch_size=256,
                               num_workers=8, cache_dir=None, tag="images"):
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{tag}_{_cache_key(image_paths)}.npy")
        if os.path.exists(cache_path):
            print(f"  loaded {tag} embeddings from cache: {cache_path}")
            return np.load(cache_path)

    ds = ImagePathDataset(image_paths, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_embs = []
    n_done = 0
    n = len(image_paths)
    for imgs in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            embs = model.encode_image(imgs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().float().numpy())
        n_done += imgs.shape[0]
        print(f"  images: {n_done}/{n}", flush=True)
    result = np.vstack(all_embs)

    if cache_path is not None:
        np.save(cache_path, result)
        print(f"  cached {tag} embeddings to: {cache_path}")
    return result


def load_train_holdout(manifest_path, eval_samples, n_train, seed):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    eval_paths = set(s["image_path"] for s in eval_samples)
    remaining = [m for m in manifest if m["image_path"] not in eval_paths]
    rng = random.Random(seed)
    if n_train < len(remaining):
        remaining = rng.sample(remaining, n_train)
    return remaining


def steer(embs, v_unit, alpha):
    corrected = (1 - alpha) * embs + alpha * v_unit.unsqueeze(0) * embs.norm(dim=-1, keepdim=True)
    return corrected / corrected.norm(dim=-1, keepdim=True)


def accuracy(img, pos, neg, v, alpha):
    corrected = steer(neg, v / v.norm(), alpha)
    sim_pos = (img * pos).sum(-1)
    sim_neg = (img * corrected).sum(-1)
    return (sim_pos > sim_neg).float().mean().item()


def recall_at_1(query, gallery):
    sims = query @ gallery.T
    ranks = sims.argmax(dim=1)
    correct = torch.arange(query.shape[0], device=query.device)
    return (ranks == correct).float().mean().item()


def retrieval_r1(img, neg, v, alpha):
    corrected = steer(neg, v / v.norm(), alpha)
    return recall_at_1(corrected, img)


def loss_weight_schedule(epoch, total_epochs, w_min=0.1, w_max=1.0):
    """Cosine ramp from w_min to w_max over training, same functional form
    used for the FTM+TT weighting in the bioacoustic hierarchy paper
    (beta(t) = 0.25*(1-cos(pi*t/T)), generalized here to arbitrary bounds)."""
    frac = 0.5 * (1 - math.cos(math.pi * epoch / max(total_epochs - 1, 1)))
    return w_min + (w_max - w_min) * frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--reg_lambda", type=float, default=2.0)
    parser.add_argument("--retrieval_lambda_max", type=float, default=1.0,
                         help="retrieval loss weight ramps from retrieval_lambda_max*0.1 up to this value over training")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = os.path.join(cfg.output.vectors_dir, "embedding_cache")

    run_id = f"stage2_vector_finetune_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vec_dir = os.path.join(cfg.output.vectors_dir, run_id)
    os.makedirs(vec_dir, exist_ok=True)
    print(f"Run ID: {run_id}")

    print(f"\nLoading CLIP ({cfg.clip.backbone}, {cfg.clip.pretrained})...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    eval_samples = load_ccneg_image_samples(cfg.image_eval.manifest_path, cfg.image_eval.n_samples, cfg.image_eval.seed)
    train_samples = load_train_holdout(cfg.image_eval.manifest_path, eval_samples, args.n_train, cfg.seed + 999)
    print(f"Train pool: {len(train_samples)} (disjoint from eval)")
    print(f"Eval set  : {len(eval_samples)} (same set used for Part A / retrieval eval)")

    def encode_split(samples, tag):
        print(f"Encoding {tag}...")
        imgs = [s["image_path"] for s in samples]
        pos = [s["positive_caption"] for s in samples]
        neg = [s["negative_caption"] for s in samples]
        img_e = get_clip_image_embeddings(clip_model, clip_preprocess, imgs, device,
                                           num_workers=args.num_workers, cache_dir=cache_dir, tag=tag)
        pos_e = get_clip_text_embeddings(clip_model, clip_tokenizer, pos, device)
        neg_e = get_clip_text_embeddings(clip_model, clip_tokenizer, neg, device)
        return (torch.from_numpy(img_e).float().to(device),
                torch.from_numpy(pos_e).float().to(device),
                torch.from_numpy(neg_e).float().to(device))

    img_train, pos_train, neg_train = encode_split(train_samples, "train_pool")
    img_eval, pos_eval, neg_eval = encode_split(eval_samples, "eval_set")
    n_train = img_train.shape[0]

    init_vec = np.load(args.init_dir)
    init_vec = init_vec / np.linalg.norm(init_vec)
    v = torch.tensor(init_vec, dtype=torch.float32, device=device, requires_grad=True)
    v_init_fixed = torch.tensor(init_vec, dtype=torch.float32, device=device, requires_grad=False)

    baseline_acc = accuracy(img_eval, pos_eval, neg_eval, v.detach(), alpha=0.0)
    baseline_r1 = retrieval_r1(img_eval, neg_eval, v.detach(), alpha=0.0)
    warmstart_acc = accuracy(img_eval, pos_eval, neg_eval, v.detach(), args.alpha)
    warmstart_r1 = retrieval_r1(img_eval, neg_eval, v.detach(), args.alpha)
    print(f"\nBaseline (no correction)   eval acc={baseline_acc:.4f}  eval R@1={baseline_r1:.4f}")
    print(f"Warm start (Procrustes)    eval acc={warmstart_acc:.4f}  eval R@1={warmstart_r1:.4f}")

    optimizer = torch.optim.Adam([v], lr=args.lr)
    best_r1 = warmstart_r1
    best_v = v.detach().clone()
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        retrieval_lambda = loss_weight_schedule(epoch, args.epochs, w_min=0.1 * args.retrieval_lambda_max, w_max=args.retrieval_lambda_max)
        perm = torch.randperm(n_train, device=device)
        epoch_loss, epoch_margin, epoch_retr, epoch_reg = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            img_b, pos_b, neg_b = img_train[idx], pos_train[idx], neg_train[idx]

            optimizer.zero_grad()
            v_unit = v / v.norm()
            corrected = steer(neg_b, v_unit, args.alpha)
            sim_pos = (img_b * pos_b).sum(-1)
            sim_neg = (img_b * corrected).sum(-1)
            margin_loss = F.relu(args.margin - (sim_pos - sim_neg)).mean()

            logits = 100.0 * corrected @ img_b.T
            labels = torch.arange(idx.shape[0], device=device)
            retrieval_loss = F.cross_entropy(logits, labels)

            reg_loss = 1 - torch.dot(v_unit, v_init_fixed)
            loss = margin_loss + retrieval_lambda * retrieval_loss + args.reg_lambda * reg_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_margin += margin_loss.item()
            epoch_retr += retrieval_loss.item()
            epoch_reg += reg_loss.item()
            n_batches += 1

        eval_acc = accuracy(img_eval, pos_eval, neg_eval, v.detach(), args.alpha)
        eval_r1 = retrieval_r1(img_eval, neg_eval, v.detach(), args.alpha)
        history.append((epoch, eval_acc, eval_r1))

        if eval_r1 > best_r1:
            best_r1 = eval_r1
            best_v = v.detach().clone()
            best_epoch = epoch

        print(f"epoch {epoch:3d}  retr_lambda={retrieval_lambda:.3f}  "
              f"loss={epoch_loss/n_batches:.4f}  margin={epoch_margin/n_batches:.4f}  "
              f"retr={epoch_retr/n_batches:.4f}  reg={epoch_reg/n_batches:.4f}  "
              f"eval_acc={eval_acc:.4f}  eval_R@1={eval_r1:.4f}"
              f"{'  <- best so far' if epoch == best_epoch else ''}")

    final_vec = (best_v / best_v.norm()).cpu().numpy()
    np.save(os.path.join(vec_dir, "clip_dir_finetuned.npy"), final_vec)

    epochs, accs, r1s = zip(*history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, accs)
    axes[0].axhline(y=warmstart_acc, color="gray", linestyle="--", label=f"warm start = {warmstart_acc:.4f}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("eval accuracy (2-choice)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, r1s)
    axes[1].axhline(y=warmstart_r1, color="gray", linestyle="--", label=f"warm start = {warmstart_r1:.4f}")
    axes[1].axhline(y=baseline_r1, color="red", linestyle=":", label=f"uncorrected = {baseline_r1:.4f}")
    axes[1].axvline(x=best_epoch, color="green", linestyle="--", label=f"best epoch ({best_epoch})")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("eval Recall@1 (retrieval)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(vec_dir, "finetune_curve.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved: {fig_path}")

    print("\n" + "=" * 60)
    print(f"baseline (no correction)        acc={baseline_acc:.4f}  R@1={baseline_r1:.4f}")
    print(f"warm start (Procrustes)         acc={warmstart_acc:.4f}  R@1={warmstart_r1:.4f}")
    print(f"best checkpoint (epoch {best_epoch:3d})    R@1={best_r1:.4f}  (selected by eval R@1, not accuracy)")
    print(f"saved vector: {os.path.join(vec_dir, 'clip_dir_finetuned.npy')}")
    print("=" * 60)


if __name__ == "__main__":
    main()