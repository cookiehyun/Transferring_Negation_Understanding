"""
Reimplements NegBench's official MCQ evaluation logic (src/evaluation/mcq.py,
evaluate_model function) exactly: L2-normalize, einsum('bf,nbf->bn'), argmax.
Applies our correction vector only to caption options that actually contain
a negation cue, following the same gating rule as "Seeing What's Not There"
("if none of the negator words are present, the embedding is not updated").
"""

import os
import sys
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
    """Steer only embeddings whose source text contains a negation cue,
    matching the source paper's update rule."""
    mask = np.array([has_negation(t) for t in texts])
    out = embs.copy()
    if mask.sum() > 0:
        sub = embs[mask]
        corrected = (1 - alpha) * sub + alpha * direction[None, :] * np.linalg.norm(sub, axis=1, keepdims=True)
        corrected = corrected / np.linalg.norm(corrected, axis=1, keepdims=True)
        out[mask] = corrected
    return out, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mcq_csv", type=str, required=True)
    parser.add_argument("--coco_root", type=str, required=True,
                         help="local folder containing the COCO val2017 images")
    parser.add_argument("--vector_path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    df = pd.read_csv(args.mcq_csv)
    print(f"MCQ rows: {len(df)}")

    image_paths = [resolve_image_path(p, args.coco_root) for p in df["image_path"]]
    caption_cols = ["caption_0", "caption_1", "caption_2", "caption_3"]
    correct_answer = df["correct_answer"].to_numpy()
    correct_answer_type = df["correct_answer_template"].tolist()

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)

    option_embs = []  # (4, N, dim)
    direction = None
    if args.vector_path:
        direction = np.load(args.vector_path)
        direction = direction / np.linalg.norm(direction)

    for col in caption_cols:
        texts = df[col].tolist()
        embs = get_clip_text_embeddings(clip_model, clip_tokenizer, texts, device)
        if direction is not None:
            embs, _ = apply_correction(embs, texts, direction, args.alpha)
        option_embs.append(embs)

    option_embs = np.stack(option_embs, axis=0)  # (4, N, dim)
    logits = np.einsum("bf,nbf->bn", img_embs, option_embs)  # (N, 4)
    predicted = logits.argmax(axis=1)

    total_acc = float((predicted == correct_answer).mean())

    by_type = {}
    for t in ["positive", "negative", "hybrid"]:
        idx = [i for i, ty in enumerate(correct_answer_type) if ty == t]
        if len(idx) == 0:
            continue
        acc = float((predicted[idx] == correct_answer[idx]).mean())
        by_type[t] = (acc, len(idx))

    tag = f"corrected (alpha={args.alpha})" if direction is not None else "uncorrected"
    print(f"\n=== NegBench MCQ, {tag} ===")
    print(f"Total accuracy: {total_acc:.4f}  (chance = 0.25)")
    for t, (acc, n) in by_type.items():
        print(f"  {t:10s} (n={n:5d}): {acc:.4f}")


if __name__ == "__main__":
    main()