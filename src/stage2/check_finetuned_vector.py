import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
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
    corrected = (1 - alpha) * embs + alpha * direction[None, :] * np.linalg.norm(embs, axis=1, keepdims=True)
    return corrected / np.linalg.norm(corrected, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--alpha", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    samples = load_ccneg_image_samples(cfg.image_eval.manifest_path, cfg.image_eval.n_samples, cfg.image_eval.seed)
    image_paths = [s["image_path"] for s in samples]
    pos_captions = [s["positive_caption"] for s in samples]
    neg_captions = [s["negative_caption"] for s in samples]

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    pos_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_captions, device)
    neg_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, neg_captions, device)
    sim_pos = (img_embs * pos_embs).sum(axis=1)

    direction = np.load(args.vector_path)
    direction = direction / np.linalg.norm(direction)

    rng = np.random.RandomState(cfg.seed)
    shuffle_idx = rng.permutation(len(img_embs))
    img_embs_shuffled = img_embs[shuffle_idx]

    has_artifact = np.array([", not " in c or ",not " in c for c in neg_captions])

    alphas = [float(a) for a in args.alpha.split(",")]
    for alpha in alphas:
        neg_corrected = steer(neg_embs, direction, alpha)
        sim_neg_own = (img_embs * neg_corrected).sum(axis=1)
        sim_neg_shuffled = (img_embs_shuffled * neg_corrected).sum(axis=1)
        acc = float((sim_pos > sim_neg_own).mean())

        idx = rng.choice(len(neg_corrected), size=min(500, len(neg_corrected)), replace=False)
        sub = neg_corrected[idx]
        pairwise = sub @ sub.T
        n = pairwise.shape[0]
        pairwise_sim = float((pairwise.sum() - np.trace(pairwise)) / (n * (n - 1)))

        gap = sim_neg_own.mean() - sim_neg_shuffled.mean()
        flag = ""
        if gap < 0.03:
            flag += " <- gap collapse"
        if pairwise_sim > 0.9:
            flag += " <- pairwise collapse"

        acc_artifact = float((sim_pos[has_artifact] > sim_neg_own[has_artifact]).mean())
        acc_natural = float((sim_pos[~has_artifact] > sim_neg_own[~has_artifact]).mean())

        print(f"alpha={alpha:.2f}  acc={acc:.4f}  gap={gap:.4f}  pairwise_sim={pairwise_sim:.4f}  "
              f"acc_artifact={acc_artifact:.4f}  acc_natural={acc_natural:.4f}{flag}")


if __name__ == "__main__":
    main()