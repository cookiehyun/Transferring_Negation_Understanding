"""
Reimplements NegBench's official retrieval evaluation logic
(src/evaluation/retrieval.py: evaluate_model + recall_at_k) exactly:
for each caption, its only positive match is its own source image; recall@k
checks whether that image is among the top-k images ranked by similarity.

Run once on COCO_val_retrieval.csv (affirmative captions, sanity check) and
once on COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
(negated captions, with our correction applied), matching NegBench's own
"coco-image_retrieval_recall@k" / "coco-negated-image_retrieval_recall@k"
naming.
"""

import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config

NEGATION_CUES = ["not ", "n't", "no ", "without", "never", "nowhere", "none",
                 "nothing", "absent", "devoid", "lacks", "lack of"]


def has_negation(text):
    t = text.lower()
    return any(cue in t for cue in NEGATION_CUES)


def resolve_image_path(csv_path, coco_root):
    fname = os.path.basename(csv_path)
    return os.path.join(coco_root, fname)


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


def apply_correction(embs, texts, direction, alpha):
    mask = np.array([has_negation(t) for t in texts])
    out = embs.copy()
    if mask.sum() > 0:
        sub = embs[mask]
        corrected = (1 - alpha) * sub + alpha * direction[None, :] * np.linalg.norm(sub, axis=1, keepdims=True)
        corrected = corrected / np.linalg.norm(corrected, axis=1, keepdims=True)
        out[mask] = corrected
    return out


def recall_at_k(scores, positive_pairs, k):
    """Exact port of NegBench's recall_at_k (src/evaluation/retrieval.py)."""
    nb_texts, nb_images = scores.shape
    topk_indices = np.argsort(-scores, axis=1)[:, :k]
    nb_positive = positive_pairs.sum(axis=1)
    hit = np.array([
        positive_pairs[i, topk_indices[i]].sum() for i in range(nb_texts)
    ])
    return hit / nb_positive


def evaluate_retrieval(img_embs, text_embs, texts_image_index, ks=(1, 5)):
    n_images = img_embs.shape[0]
    scores = text_embs @ img_embs.T  # (n_texts, n_images)
    positive_pairs = np.zeros_like(scores, dtype=bool)
    positive_pairs[np.arange(len(scores)), texts_image_index] = True

    results = {}
    for k in ks:
        results[k] = float((recall_at_k(scores, positive_pairs, k) > 0).mean())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--retrieval_csv", type=str, required=True,
                         help="COCO_val_retrieval.csv or the negated variant")
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--vector_path", type=str, default=None,
                         help="apply correction (recommended only for the negated CSV)")
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    df = pd.read_csv(args.retrieval_csv)
    print(f"Images: {len(df)}")

    image_paths = [resolve_image_path(p, args.coco_root) for p in df["filepath"]]
    caption_lists = [ast.literal_eval(c) for c in df["captions"]]

    all_texts = []
    texts_image_index = []
    for img_idx, captions in enumerate(caption_lists):
        for c in captions:
            all_texts.append(c)
            texts_image_index.append(img_idx)
    texts_image_index = np.array(texts_image_index)
    print(f"Total captions: {len(all_texts)}")

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    text_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, all_texts, device)

    tag = "uncorrected"
    if args.vector_path:
        direction = np.load(args.vector_path)
        direction = direction / np.linalg.norm(direction)
        text_embs = apply_correction(text_embs, all_texts, direction, args.alpha)
        tag = f"corrected (alpha={args.alpha})"

    results = evaluate_retrieval(img_embs, text_embs, texts_image_index, ks=(1, 5))
    print(f"\n=== NegBench Retrieval ({os.path.basename(args.retrieval_csv)}), {tag} ===")
    for k, v in results.items():
        print(f"  image_retrieval_recall@{k}: {v:.4f}")


if __name__ == "__main__":
    main()