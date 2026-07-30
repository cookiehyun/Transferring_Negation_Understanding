"""
NegBench MCQ evaluation with content-aware correction (LLM extracts the
negated concept per-caption, then Seeing What's Not There Eq. 2 is applied) --
in place of the fixed-direction vector steering used in run_negbench_mcq_eval.py.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mcq_csv", type=str, required=True)
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--lambdas", type=str, default="1.9",
                         help="comma-separated lambda values to sweep")
    parser.add_argument("--n_rows", type=int, default=-1, help="subsample rows for a quick test run, -1 = all")
    parser.add_argument("--extractor", type=str, default="llm", choices=["llm", "rule"],
                         help="'llm' = our LLM-based extractor, 'rule' = reimplementation of "
                              "the paper's rule-based parser (Appendix A.1)")
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
        extract_fn = None  # default (LLM) inside extract_concepts_and_embeddings
    else:
        print("Using rule-based extractor (no LLM loaded)")
        llm_model, llm_tokenizer = None, None
        extract_fn = rule_based_extraction.extract_negated_concepts

    df = pd.read_csv(args.mcq_csv)
    if args.n_rows > 0:
        df = df.sample(n=min(args.n_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"MCQ rows: {len(df)}")

    image_paths = [resolve_image_path(p, args.coco_root) for p in df["image_path"]]
    caption_cols = ["caption_0", "caption_1", "caption_2", "caption_3"]
    correct_answer = df["correct_answer"].to_numpy()
    correct_answer_type = df["correct_answer_template"].tolist()

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    anchor = compute_anchor(clip_model, clip_tokenizer, device)

    # extract once per option column, reuse across all lambdas
# extract once per option column, reuse across all lambdas
    col_data = {}  # col -> (e_c, concepts, e_neg, valid_idx)
    n_extracted_total = 0
    all_failure_reasons = []
    for col in caption_cols:
        texts = df[col].tolist()
        print(f"Extracting {col}...")
        e_c, concepts, e_neg, valid_idx, failure_reason = extract_concepts_and_embeddings(
            clip_model, clip_tokenizer, device, texts, llm_model, llm_tokenizer, extract_fn=extract_fn
        )
        col_data[col] = (e_c, e_neg, valid_idx)
        n_extracted_total += len(valid_idx)
        all_failure_reasons.extend(failure_reason)

    n_total_captions = len(df) * 4
    print(f"\nSuccessfully extracted: {n_extracted_total}/{n_total_captions} captions")
    from collections import Counter
    print(f"Failure breakdown: {Counter(all_failure_reasons)}")
    
    lambdas = [float(x) for x in args.lambdas.split(",")]
    for lam in lambdas:
        option_embs = []
        for col in caption_cols:
            e_c, e_neg, valid_idx = col_data[col]
            option_embs.append(apply_correction_given_embeddings(e_c, e_neg, valid_idx, anchor, lam))
        option_embs = np.stack(option_embs, axis=0)  # (4, N, dim)
        logits = np.einsum("bf,nbf->bn", img_embs, option_embs)
        predicted = logits.argmax(axis=1)

        total_acc = float((predicted == correct_answer).mean())
        print(f"\n=== lambda={lam:.2f} ===")
        print(f"Total accuracy: {total_acc:.4f}  (chance = 0.25)")
        for t in ["positive", "negative", "hybrid"]:
            idx = [i for i, ty in enumerate(correct_answer_type) if ty == t]
            if not idx:
                continue
            acc = float((predicted[idx] == correct_answer[idx]).mean())
            print(f"  {t:10s} (n={len(idx):5d}): {acc:.4f}")


if __name__ == "__main__":
    main()