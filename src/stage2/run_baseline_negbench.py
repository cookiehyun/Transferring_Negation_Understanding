"""
Evaluate baseline models (plain CLIP, NegCLIP, ConCLIP) on NegBench MCQ/
Retrieval test splits -- no correction applied, just each model's raw
text/image embeddings. Lets us place our (training-free) method's
improvement relative to models that WERE fine-tuned for negation.
"""
import os, sys, ast, argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from run_baseline_eval import load_negclip, load_conclip, load_plain_clip, encode_text, encode_images

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


def resolve_image_path(p, coco_root):
    return os.path.join(coco_root, os.path.basename(p))


def eval_mcq(model, preprocess, tokenizer, device, api, mcq_csv, coco_root):
    df = pd.read_csv(mcq_csv)
    image_paths = [resolve_image_path(p, coco_root) for p in df["image_path"]]
    caption_cols = ["caption_0", "caption_1", "caption_2", "caption_3"]
    correct_answer = df["correct_answer"].to_numpy()
    correct_answer_type = df["correct_answer_template"].tolist()

    img_embs = encode_images(model, preprocess, image_paths, device, api)
    option_embs = []
    for col in caption_cols:
        texts = df[col].tolist()
        option_embs.append(encode_text(model, tokenizer, texts, device, api))
    option_embs = np.stack(option_embs, axis=0)
    logits = np.einsum("bf,nbf->bn", img_embs, option_embs)
    predicted = logits.argmax(axis=1)
    total_acc = float((predicted == correct_answer).mean())

    print(f"MCQ Total accuracy: {total_acc:.4f}")
    for t in ["positive", "negative", "hybrid"]:
        idx = [i for i, ty in enumerate(correct_answer_type) if ty == t]
        acc = float((predicted[idx] == correct_answer[idx]).mean())
        print(f"  {t:10s} (n={len(idx):5d}): {acc:.4f}")


def eval_retrieval(model, preprocess, tokenizer, device, api, retrieval_csv, coco_root):
    df = pd.read_csv(retrieval_csv)
    image_paths = [resolve_image_path(p, coco_root) for p in df["filepath"]]
    caption_lists = [ast.literal_eval(c) for c in df["captions"]]

    all_texts, texts_image_index = [], []
    for img_idx, captions in enumerate(caption_lists):
        for c in captions:
            all_texts.append(c)
            texts_image_index.append(img_idx)
    texts_image_index = np.array(texts_image_index)

    img_embs = encode_images(model, preprocess, image_paths, device, api)
    text_embs = encode_text(model, tokenizer, all_texts, device, api)
    scores = text_embs @ img_embs.T
    positive_pairs = np.zeros_like(scores, dtype=bool)
    positive_pairs[np.arange(len(scores)), texts_image_index] = True

    def recall_at_k(scores, positive_pairs, k):
        topk = np.argsort(-scores, axis=1)[:, :k]
        nb_positive = positive_pairs.sum(axis=1)
        hit = np.array([positive_pairs[i, topk[i]].sum() for i in range(len(scores))])
        return hit / nb_positive

    r1 = float((recall_at_k(scores, positive_pairs, 1) > 0).mean())
    r5 = float((recall_at_k(scores, positive_pairs, 5) > 0).mean())
    print(f"Retrieval R@1: {r1:.4f}  R@5: {r5:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["clip", "negclip", "conclip"])
    parser.add_argument("--conclip_ckpt", type=str, default=None)
    parser.add_argument("--mcq_csv", type=str, default="outputs/negbench/COCO_val_mcq_test.csv")
    parser.add_argument("--retrieval_csv", type=str, default="outputs/negbench/COCO_val_negated_retrieval_test.csv")
    parser.add_argument("--coco_root", type=str, default="outputs/coco/images/val2017")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    if args.model == "negclip":
        model, preprocess, tokenizer, api = load_negclip(device)
    elif args.model == "conclip":
        if not args.conclip_ckpt:
            raise ValueError("--conclip_ckpt required for ConCLIP")
        model, preprocess, tokenizer, api = load_conclip(device, args.conclip_ckpt)
    else:
        model, preprocess, tokenizer, api = load_plain_clip(device)

    print(f"\n=== {args.model} on MCQ (test) ===")
    eval_mcq(model, preprocess, tokenizer, device, api, args.mcq_csv, args.coco_root)

    print(f"\n=== {args.model} on Retrieval (test) ===")
    eval_retrieval(model, preprocess, tokenizer, device, api, args.retrieval_csv, args.coco_root)


if __name__ == "__main__":
    main()
