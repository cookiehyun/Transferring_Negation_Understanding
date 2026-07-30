"""
Diagnostic: for each layer of CLIP's text transformer, project the
intermediate hidden state through the final layer norm + text_projection
(logit-lens style) and measure proj (caption-concept alignment) at that
layer, compared to the standard final-layer proj. If some middle layer
gives consistently higher proj than the final layer, it's a candidate for
ensembling into e_neg.
"""
import os, sys, random
import numpy as np
import pandas as pd
import ast
import torch
import open_clip

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import extract_negated_concepts
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


def get_per_layer_embeddings(clip_model, clip_tokenizer, texts, device, batch_size=64):
    """Returns dict: layer_idx -> (N, dim) array of logit-lens-projected embeddings,
    plus the standard final-layer embedding under key 'final'."""
    n_layers = len(clip_model.transformer.resblocks)
    captured = {i: [] for i in range(n_layers)}
    final_embs = []

    hooks = []
    layer_outputs = {}

    def make_hook(idx):
        def hook(module, inp, out):
            layer_outputs[idx] = out  # (seq_len, batch, dim), LND format
        return hook

    for i, block in enumerate(clip_model.transformer.resblocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = clip_tokenizer(batch).to(device)
        with torch.no_grad():
            final = clip_model.encode_text(tokens)  # triggers hooks
            final = final / final.norm(dim=-1, keepdim=True)
        final_embs.append(final.cpu().float().numpy())

        eot_idx = tokens.argmax(dim=-1)  # (batch,)
        batch_size = tokens.shape[0]
        for i in range(n_layers):
            x = layer_outputs[i]  # format may be (batch, seq, dim) or (seq, batch, dim)
            if x.shape[0] != batch_size:
                x = x.permute(1, 0, 2)  # (seq, batch, dim) -> (batch, seq, dim)
            x_eot = x[torch.arange(x.shape[0]), eot_idx]  # (batch, dim)
            # logit-lens: project through final ln + text_projection
            with torch.no_grad():
                x_proj = clip_model.ln_final(x_eot) @ clip_model.text_projection
                x_proj = x_proj / x_proj.norm(dim=-1, keepdim=True)
            captured[i].append(x_proj.cpu().float().numpy())

    for h in hooks:
        h.remove()

    result = {i: np.vstack(v) for i, v in captured.items()}
    result["final"] = np.vstack(final_embs)
    return result


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
    all_texts = [c for captions in caption_lists for c in captions]
    rng = random.Random(42)
    sample_texts = rng.sample(all_texts, min(1500, len(all_texts)))
    print(f"진단 샘플: {len(sample_texts)}개")

    concepts, failure_reason = extract_negated_concepts(llm_model, llm_tokenizer, sample_texts)
    valid_idx = [i for i, c in enumerate(concepts) if c is not None]
    print(f"성공: {len(valid_idx)}/{len(sample_texts)}")

    concept_texts = [concepts[i] for i in valid_idx]
    caption_subset = [sample_texts[i] for i in valid_idx]

    print("\n캡션 임베딩 (레이어별) 계산 중...")
    e_c_layers = get_per_layer_embeddings(clip_model, clip_tokenizer, caption_subset, device)
    print("개념 임베딩 (레이어별) 계산 중...")
    e_neg_layers = get_per_layer_embeddings(clip_model, clip_tokenizer, concept_texts, device)

    n_layers = len(clip_model.transformer.resblocks)
    print(f"\n총 {n_layers}개 레이어")
    print(f"{'layer':>8} {'mean proj':>12}")
    for key in list(range(n_layers)) + ["final"]:
        proj = (e_c_layers[key] * e_neg_layers[key]).sum(axis=1)
        print(f"{str(key):>8} {proj.mean():>12.4f}")


if __name__ == "__main__":
    main()
