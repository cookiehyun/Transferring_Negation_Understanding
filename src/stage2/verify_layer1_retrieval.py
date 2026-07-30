"""
Verify whether layer-1 (logit-lens projected) concept embeddings actually
improve retrieval, or whether the higher proj at layer 1 was just lexical
overlap that doesn't help (or hurts) once plugged into the correction
formula and compared against real images.
"""
import os, sys, ast
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import (
    extract_negated_concepts, compute_anchor, apply_correction_given_embeddings,
    get_clip_text_embeddings
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


def get_layer_embedding(clip_model, clip_tokenizer, texts, device, layer_idx, batch_size=64):
    layer_outputs = {}
    def hook(module, inp, out):
        layer_outputs["x"] = out
    h = clip_model.transformer.resblocks[layer_idx].register_forward_hook(hook)

    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = clip_tokenizer(batch).to(device)
        with torch.no_grad():
            _ = clip_model.encode_text(tokens)
        x = layer_outputs["x"]
        if x.shape[0] != tokens.shape[0]:
            x = x.permute(1, 0, 2)
        eot_idx = tokens.argmax(dim=-1)
        x_eot = x[torch.arange(x.shape[0]), eot_idx]
        with torch.no_grad():
            x_proj = clip_model.ln_final(x_eot) @ clip_model.text_projection
            x_proj = x_proj / x_proj.norm(dim=-1, keepdim=True)
        all_embs.append(x_proj.cpu().float().numpy())
    h.remove()
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


def recall_at_1(scores, positive_pairs):
    top1 = np.argmax(scores, axis=1)
    return float(positive_pairs[np.arange(len(top1)), top1].mean())


def main():
    cfg = load_stage2_config("configs/stage2.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg.clip.backbone, pretrained=cfg.clip.pretrained)
    clip_model.eval().to(device)
    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.backbone)

    dtype = getattr(torch, cfg.model.dtype)
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map)
    llm_model.eval()

    df = pd.read_csv("outputs/negbench/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    caption_lists = [ast.literal_eval(c) for c in df["captions"]]
    image_paths = [os.path.join("outputs/coco/images/val2017", os.path.basename(p)) for p in df["filepath"]]

    all_texts, texts_image_index = [], []
    for img_idx, captions in enumerate(caption_lists):
        for c in captions:
            all_texts.append(c)
            texts_image_index.append(img_idx)
    texts_image_index = np.array(texts_image_index)
    print(f"Total captions: {len(all_texts)}")

    img_embs = get_clip_image_embeddings(clip_model, clip_preprocess, image_paths, device)
    anchor = compute_anchor(clip_model, clip_tokenizer, device)

    print("Extracting concepts...")
    concepts, failure_reason = extract_negated_concepts(llm_model, llm_tokenizer, all_texts)
    valid_idx = [i for i, c in enumerate(concepts) if c is not None]
    print(f"성공: {len(valid_idx)}/{len(all_texts)}")

    e_c = get_clip_text_embeddings(clip_model, clip_tokenizer, all_texts, device)
    concept_texts = [concepts[i] for i in valid_idx]

    positive_pairs = np.zeros((len(all_texts), img_embs.shape[0]), dtype=bool)
    positive_pairs[np.arange(len(all_texts)), texts_image_index] = True

    # baseline: final layer e_neg (기존 최선)
    e_neg_final = np.zeros_like(e_c)
    e_neg_final[valid_idx] = get_clip_text_embeddings(clip_model, clip_tokenizer, concept_texts, device)

    # layer 1 e_neg
    e_neg_l1 = np.zeros_like(e_c)
    e_neg_l1[valid_idx] = get_layer_embedding(clip_model, clip_tokenizer, concept_texts, device, layer_idx=1)

    print(f"\n{'condition':>20} {'lambda':>8} {'R@1':>10}")
    for lam in [0.3, 0.5, 0.6, 0.8, 1.0]:
        text_embs = apply_correction_given_embeddings(e_c, e_neg_final, valid_idx, anchor, lam)
        scores = text_embs @ img_embs.T
        r1 = recall_at_1(scores, positive_pairs)
        print(f"{'final (baseline)':>20} {lam:>8.2f} {r1:>10.4f}")

    for lam in [0.3, 0.5, 0.6, 0.8, 1.0]:
        text_embs = apply_correction_given_embeddings(e_c, e_neg_l1, valid_idx, anchor, lam)
        scores = text_embs @ img_embs.T
        r1 = recall_at_1(scores, positive_pairs)
        print(f"{'layer 1':>20} {lam:>8.2f} {r1:>10.4f}")


if __name__ == "__main__":
    main()
