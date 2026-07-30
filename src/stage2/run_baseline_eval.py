"""
Evaluate baseline models (NegCLIP, ConCLIP, or plain CLIP) on our own
eval pipeline (2-choice accuracy, Recall@K, confidence score), so results
are directly comparable to our method on the same 5,000-image eval set.

NegCLIP (Nano1337/openclip-negclip on HuggingFace) is open_clip-compatible.
ConCLIP (jaisidhsingh/CoN-CLIP) uses the original OpenAI `clip` package and
its checkpoint is distributed via Google Drive -- download it manually from
the repo README and pass the local .pt path via --conclip_ckpt.
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config
from src.data.ccneg_image_loader import load_ccneg_image_samples


def load_negclip(device):
    import open_clip
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="Nano1337/openclip-negclip", filename="pytorch_model.bin")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained=None)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  load_state_dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    if len(missing) > 5:
        print(f"  WARNING: many missing keys, weights likely did not load correctly. First few: {missing[:5]}")
    if len(unexpected) > 5:
        print(f"  First few unexpected keys: {unexpected[:5]}")

    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    return model, preprocess, tokenizer, "open_clip"


def load_conclip(device, ckpt_path):
    import clip  # original OpenAI package: pip install git+https://github.com/openai/CLIP.git
    model, preprocess = clip.load("ViT-B/32", device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = model.float()
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, preprocess, clip.tokenize, "openai_clip"


def load_plain_clip(device, backbone="ViT-B-32-quickgelu", pretrained="openai"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(backbone)
    return model, preprocess, tokenizer, "open_clip"


def encode_text(model, tokenizer, texts, device, api, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if api == "open_clip":
            tokens = tokenizer(batch).to(device)
            with torch.no_grad():
                embs = model.encode_text(tokens).float()
        elif api == "hf_transformers":
            inputs = tokenizer(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                embs = model.get_text_features(**inputs).float()
        else:  # openai_clip
            tokens = tokenizer(batch, truncate=True).to(device)
            with torch.no_grad():
                embs = model.encode_text(tokens).float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


def encode_images(model, preprocess, image_paths, device, api, batch_size=64):
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        if api == "hf_transformers":
            inputs = preprocess(images=pil_images, return_tensors="pt").to(device)
            with torch.no_grad():
                embs = model.get_image_features(**inputs).float()
        else:
            imgs = torch.stack([preprocess(im) for im in pil_images]).to(device)
            with torch.no_grad():
                embs = model.encode_image(imgs).float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["clip", "negclip", "conclip"])
    parser.add_argument("--conclip_ckpt", type=str, default=None,
                         help="local path to conclip_vit_b32.pt (download manually from "
                              "github.com/jaisidhsingh/CoN-CLIP README, Google Drive link)")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    if args.model == "negclip":
        model, preprocess, tokenizer, api = load_negclip(device)
    elif args.model == "conclip":
        if not args.conclip_ckpt:
            raise ValueError("--conclip_ckpt required for ConCLIP (download manually, see docstring)")
        model, preprocess, tokenizer, api = load_conclip(device, args.conclip_ckpt)
    else:
        model, preprocess, tokenizer, api = load_plain_clip(device)

    samples = load_ccneg_image_samples(cfg.image_eval.manifest_path, cfg.image_eval.n_samples, cfg.image_eval.seed)
    image_paths = [s["image_path"] for s in samples]
    pos_captions = [s["positive_caption"] for s in samples]
    neg_captions = [s["negative_caption"] for s in samples]
    n = len(samples)
    print(f"Eval set: {n} images")

    img_embs = encode_images(model, preprocess, image_paths, device, api)
    pos_embs = encode_text(model, tokenizer, pos_captions, device, api)
    neg_embs = encode_text(model, tokenizer, neg_captions, device, api)

    sim_pos = (img_embs * pos_embs).sum(axis=1)
    sim_neg = (img_embs * neg_embs).sum(axis=1)
    acc = float((sim_pos > sim_neg).mean())

    sims_pos = pos_embs @ img_embs.T
    sims_neg = neg_embs @ img_embs.T
    correct = np.arange(n)

    def recall_at_k(sims, k):
        ranks = np.argsort(-sims, axis=1)[:, :k]
        return float(np.mean([correct[i] in ranks[i] for i in range(n)]))

    print(f"\n=== {args.model} ===")
    print(f"2-choice accuracy: {acc:.4f}")
    for k in (1, 5, 10):
        print(f"Affirmative R@{k}: {recall_at_k(sims_pos, k):.4f}   Negated R@{k}: {recall_at_k(sims_neg, k):.4f}")
    print(f"Confidence score: affirmative={sim_pos.mean():.4f}  negated={sim_neg.mean():.4f}")


if __name__ == "__main__":
    main()