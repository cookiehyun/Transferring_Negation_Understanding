"""
NegBench Retrieval evaluation with content-aware correction, mirroring
run_negbench_retrieval_eval.py but replacing fixed-direction steering with
per-caption LLM extraction + Seeing What's Not There Eq. 2.
"""

import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import compute_anchor, extract_concepts_and_embeddings, apply_correction_given_embeddings
import rule_based_extraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


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


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = np.argsort(-scores, axis=1)[:, :k]
    nb_positive = positive_pairs.sum(axis=1)
    hit = np.array([positive_pairs[i, topk_indices[i]].sum() for i in range(nb_texts)])
    return hit / nb_positive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--retrieval_csv", type=str, required=True)
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--lambdas", type=str, default="0.3,0.5,0.8,1.2,1.9")
    parser.add_argument("--n_images", type=int, default=-1, help="subsample images for a quick test, -1 = all")
    parser.add_argument("--extractor", type=str, default="llm", choices=["llm", "rule"])
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    if args.extractor == "llm":
        print(f"Loading LLM: {cfg.model.name_or_path}")
        dtype = getattr(torch, cfg.model.dtype)
        llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        llm_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map,
        )
        llm_model.eval()
        extract_fn = None
    else:
        print("Using rule-based extractor (no LLM loaded)")
        llm_model, llm_tokenizer = None, None
        extract_fn = rule_based_extraction.extract_negated_concepts

    df = pd.read_csv(args.retrieval_csv)
    if args.n_images > 0:
        df = df.sample(n=min(args.n_images, len(df)), random_state=42).reset_index(drop=True)
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
    anchor = compute_anchor(clip_model, clip_tokenizer, device)

    print("\nExtracting concepts (once, reused across all lambdas)...")
    e_c, concepts, e_neg, valid_idx, failure_reason = extract_concepts_and_embeddings(
        clip_model, clip_tokenizer, device, all_texts, llm_model, llm_tokenizer, extract_fn=extract_fn
    )
    n_extracted = len(valid_idx)
    print(f"Successfully extracted: {n_extracted}/{len(all_texts)} captions")
    from collections import Counter
    print(f"Failure breakdown: {Counter(failure_reason)}")    

    positive_pairs = np.zeros((len(all_texts), img_embs.shape[0]), dtype=bool)
    positive_pairs[np.arange(len(all_texts)), texts_image_index] = True

    lambdas = [float(x) for x in args.lambdas.split(",")]
    print(f"\n=== NegBench Retrieval ({os.path.basename(args.retrieval_csv)}), content-aware, lambda sweep ===")
    for lam in lambdas:
        text_embs = apply_correction_given_embeddings(e_c, e_neg, valid_idx, anchor, lam)
        scores = text_embs @ img_embs.T
        r1 = float((recall_at_k(scores, positive_pairs, 1) > 0).mean())
        r5 = float((recall_at_k(scores, positive_pairs, 5) > 0).mean())
        print(f"  lambda={lam:.2f}  R@1={r1:.4f}  R@5={r5:.4f}")


if __name__ == "__main__":
    main()