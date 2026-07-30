"""
Fits a small closed-form linear map W (via least squares / ridge regression,
NOT gradient descent) that transforms CLIP embeddings of short noun-only
concept extractions (v1 style, e.g. "chair") toward CLIP embeddings of
longer phrase extractions (v3 style, e.g. "chair in sight") that we've
shown align more strongly with the original caption (higher proj).

Fit on CC-Neg captions only (never used for lambda tuning or final
reporting on MCQ/Retrieval) -- this keeps MCQ/Retrieval as clean held-out
test sets while giving CC-Neg a legitimate remaining role.
"""
import os, sys, json, random
import numpy as np
import torch
import open_clip
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import get_clip_text_embeddings, has_negation, _verbatim_match, build_messages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config
from src.data.ccneg_image_loader import load_ccneg_image_samples

# --- v1 (short noun) prompt, kept separately for this diagnostic only ---
V1_SYSTEM_PROMPT = ("You are a precise linguistic tool. Given a sentence containing a negation, "
                     "output ONLY the exact noun phrase that is being negated -- the thing stated "
                     "to be absent, not present, or not happening. The phrase must be copied "
                     "verbatim from the sentence. If there is no single clear concept being negated "
                     "(e.g. the negation applies to an entire clause, an abstract situation, or a "
                     "time/quantity expression), output NONE. Output nothing else: no explanation, "
                     "no punctuation, no quotes.")
V1_FEW_SHOT = [
    ("person, not waving a flag from the crowd", "flag"),
    ("soccer player celebrates without teammates after scoring", "teammates"),
    ("source of the contaminated water ingested by no one", "NONE"),
    ("i 'm not sure what this design is on , but it would n't make an interesting tattoo", "NONE"),
    ("this drum set is not percent off today", "NONE"),
    ("private path from your deck to the ocean, not through the dunes", "dunes"),
]

def build_v1_messages(caption):
    messages = [{"role": "system", "content": V1_SYSTEM_PROMPT}]
    for ex_s, ex_a in V1_FEW_SHOT:
        messages.append({"role": "user", "content": f"Sentence: {ex_s}"})
        messages.append({"role": "assistant", "content": ex_a})
    messages.append({"role": "user", "content": f"Sentence: {caption}"})
    return messages


def extract_v1(llm_model, llm_tokenizer, captions, batch_size=16, max_new_tokens=12):
    concepts = [None] * len(captions)
    idx_to_process = [i for i, c in enumerate(captions) if has_negation(c)]
    device = llm_model.device
    for start in range(0, len(idx_to_process), batch_size):
        batch_idx = idx_to_process[start:start + batch_size]
        batch = [captions[i] for i in batch_idx]
        prompts = [llm_tokenizer.apply_chat_template(build_v1_messages(c), tokenize=False, add_generation_prompt=True) for c in batch]
        enc = llm_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        with torch.no_grad():
            out = llm_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=llm_tokenizer.eos_token_id)
        decoded = llm_tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, d, cap in zip(batch_idx, decoded, batch):
            c = d.strip().split("\n")[0].strip().strip('."\'')
            if c.upper() == "NONE" or not c or not _verbatim_match(c, cap):
                concepts[i] = None
            else:
                concepts[i] = c
    return concepts


def extract_v3(llm_model, llm_tokenizer, captions, batch_size=16, max_new_tokens=20):
    from content_aware_correction import extract_negated_concepts
    concepts, _ = extract_negated_concepts(llm_model, llm_tokenizer, captions, batch_size=batch_size, max_new_tokens=max_new_tokens)
    return concepts


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

    samples = load_ccneg_image_samples(cfg.image_eval.manifest_path, cfg.image_eval.n_samples, cfg.image_eval.seed)
    neg_captions = [s["negative_caption"] for s in samples]
    print(f"CC-Neg captions: {len(neg_captions)}")

    print("v1 (short) 추출 중...")
    v1_concepts = extract_v1(llm_model, llm_tokenizer, neg_captions)
    print("v3 (long) 추출 중...")
    v3_concepts = extract_v3(llm_model, llm_tokenizer, neg_captions)

    # 둘 다 성공한 것만, 그리고 실제로 다른 phrase인 것만 (같으면 학습에 정보 없음)
    pair_idx = [i for i in range(len(neg_captions))
                if v1_concepts[i] is not None and v3_concepts[i] is not None]
    print(f"둘 다 성공: {len(pair_idx)}")

    short_texts = [v1_concepts[i] for i in pair_idx]
    long_texts = [v3_concepts[i] for i in pair_idx]

    E_short = get_clip_text_embeddings(clip_model, clip_tokenizer, short_texts, device)
    E_long = get_clip_text_embeddings(clip_model, clip_tokenizer, long_texts, device)

    # train/holdout 분리 (W 자체의 과적합 확인용, 80/20)
    rng = random.Random(42)
    idx = list(range(len(pair_idx)))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.8)
    train_idx, holdout_idx = idx[:n_train], idx[n_train:]

    # Ridge regression (closed-form): W = (X^T X + alpha*I)^-1 X^T Y
    X = E_short[train_idx]
    Y = E_long[train_idx]
    alpha = 1.0
    d = X.shape[1]
    W = np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)

    # holdout 검증: W 적용 후 실제 long과 얼마나 가까워지는지
    X_hold = E_short[holdout_idx]
    Y_hold = E_long[holdout_idx]
    pred = X_hold @ W
    pred = pred / np.linalg.norm(pred, axis=1, keepdims=True)

    sim_before = (X_hold * Y_hold / np.linalg.norm(X_hold,axis=1,keepdims=True)).sum(axis=1)  # short vs long, raw
    sim_before = np.array([np.dot(X_hold[i]/np.linalg.norm(X_hold[i]), Y_hold[i]) for i in range(len(X_hold))])
    sim_after = np.array([np.dot(pred[i], Y_hold[i]) for i in range(len(pred))])

    print(f"\nholdout: W 적용 전 (short vs long) 평균 유사도: {sim_before.mean():.4f}")
    print(f"holdout: W 적용 후 (W@short vs long) 평균 유사도: {sim_after.mean():.4f}")

    os.makedirs("outputs/vectors", exist_ok=True)
    np.save("outputs/vectors/concept_correction_W.npy", W)
    print("\nW 저장: outputs/vectors/concept_correction_W.npy")


if __name__ == "__main__":
    main()
