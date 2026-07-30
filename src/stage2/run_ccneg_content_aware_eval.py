"""
Content-aware correction evaluated on CC-Neg (the messier, real-world web
caption data, as opposed to NegBench's cleaner COCO-based captions), using
CC-Neg's own official metric: original-over-negation accuracy (verified
against the CoN-CLIP repo's actual eval code -- see project notes).

Supports both the LLM extractor and the rule-based parser reimplementation,
so the two can be compared directly on data neither was originally validated
on by "Seeing What's Not There".
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import (
    get_clip_text_embeddings, compute_anchor, extract_concepts_and_embeddings,
    apply_correction_given_embeddings
)
import rule_based_extraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config
from src.data.ccneg_image_loader import load_ccneg_image_samples


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lambdas", type=str, default="0.1,0.3,0.5,0.8,1.2,1.6,1.9")
    parser.add_argument("--extractor", type=str, default="llm", choices=["llm", "rule", "hybrid"])
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    if args.extractor in ("llm", "hybrid"):
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

    samples = load_ccneg_image_samples(cfg.image_eval.manifest_path, cfg.image_eval.n_samples, cfg.image_eval.seed)
    image_paths = [s["image_path"] for s in samples]
    pos_captions = [s["positive_caption"] for s in samples]
    neg_captions = [s["negative_caption"] for s in samples]
    print(f"Eval set: {len(samples)}")

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    pos_embs = get_clip_text_embeddings(clip_model, clip_tokenizer, pos_captions, device)
    anchor = compute_anchor(clip_model, clip_tokenizer, device)

    print("\nExtracting concepts (once, reused across all lambdas)...")
    e_c, concepts, e_neg, valid_idx, failure_reason = extract_concepts_and_embeddings(
        clip_model, clip_tokenizer, device, neg_captions, llm_model, llm_tokenizer, extract_fn=extract_fn,
        hybrid=(args.extractor == "hybrid")
    )
    print(f"Successfully extracted: {len(valid_idx)}/{len(neg_captions)} captions")
    from collections import Counter
    print(f"Failure breakdown: {Counter(failure_reason)}")
    sim_pos = (img_embs * pos_embs).sum(axis=1)
    baseline_acc = float((sim_pos > (img_embs * e_c).sum(axis=1)).mean())
    print(f"\nBaseline (no correction) accuracy: {baseline_acc:.4f}  (extractor-independent, sanity check)")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    print(f"\n=== CC-Neg original-over-negation accuracy, extractor={args.extractor} ===")
    for lam in lambdas:
        neg_corrected = apply_correction_given_embeddings(e_c, e_neg, valid_idx, anchor, lam)
        sim_neg = (img_embs * neg_corrected).sum(axis=1)
        acc = float((sim_pos > sim_neg).mean())
        print(f"  lambda={lam:.2f}  accuracy={acc:.4f}")


if __name__ == "__main__":
    main()