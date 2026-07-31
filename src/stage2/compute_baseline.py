"""Uncorrected CLIP baseline on the held-out test splits (no lambda, no extraction)."""
import os, sys, ast
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import get_clip_text_embeddings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


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


def main():
    cfg = load_stage2_config("configs/stage2.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    coco_root = "outputs/coco/images/val2017"

    # --- MCQ baseline (test) ---
    df = pd.read_csv("outputs/negbench/COCO_val_mcq_test.csv")
    image_paths = [os.path.join(coco_root, os.path.basename(p)) for p in df["image_path"]]
    caption_cols = ["caption_0", "caption_1", "caption_2", "caption_3"]
    correct_answer = df["correct_answer"].to_numpy()
    correct_answer_type = df["correct_answer_template"].tolist()

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    option_embs = []
    for col in caption_cols:
        texts = df[col].tolist()
        option_embs.append(get_clip_text_embeddings(clip_model, clip_tokenizer, texts, device))
    option_embs = np.stack(option_embs, axis=0)
    logits = np.einsum("bf,nbf->bn", img_embs, option_embs)
    predicted = logits.argmax(axis=1)
    total_acc = float((predicted == correct_answer).mean())
    print(f"=== MCQ baseline (test, uncorrected) ===")
    print(f"Total accuracy: {total_acc:.4f}")
    for t in ["positive", "negative", "hybrid"]:
        idx = [i for i, ty in enumerate(correct_answer_type) if ty == t]
        acc = float((predicted[idx] == correct_answer[idx]).mean())
        print(f"  {t:10s} (n={len(idx):5d}): {acc:.4f}")

    # --- Retrieval baseline (test) ---
    df = pd.read_csv("outputs/negbench/COCO_val_negated_retrieval_test.csv")
    image_paths = [os.path.join(coco_root, os.path.basename(p)) for p in df["filepath"]]
    caption_lists = [ast.literal_eval(c) for c in df["captions"]]
    all_texts, texts_image_index = [], []
    for img_idx, captions in enumerate(caption_lists):
        for c in captions:
            all_texts.append(c)
            texts_image_index.append(img_idx)
    texts_image_index = np.array(texts_image_index)

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    text_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, all_texts, device)
    scores = text_embs @ img_embs.T
    positive_pairs = np.zeros_like(scores, dtype=bool)
    positive_pairs[np.arange(len(scores)), texts_image_index] = True

    def recall_at_k(scores, positive_pairs, k):
        topk = np.argsort(-scores, axis=1)[:, :k]
        nb_positive = positive_pairs.sum(axis=1)
        hit = np.array([positive_pairs[i, topk[i]].sum() for i in range(len(scores))])
        return (hit / nb_positive)

    r1 = float((recall_at_k(scores, positive_pairs, 1) > 0).mean())
    r5 = float((recall_at_k(scores, positive_pairs, 5) > 0).mean())
    print(f"\n=== Retrieval baseline (test, uncorrected) ===")
    print(f"R@1: {r1:.4f}  R@5: {r5:.4f}")


if __name__ == "__main__":
    main()
