"""
Shared content-aware correction utilities: detect negation, extract the
negated concept via LLM (validated system/user chat format), then apply
"Seeing What's Not There" Eq. 2 only to captions where extraction succeeded.
Captions without a negation cue, or where the LLM returned NONE / an
unverifiable phrase, are left unmodified (matching the source paper's own
update rule: "if none of the negator words are present, the embedding is
not updated").

v3 prompt: extracted phrase includes short predicate/copula wording around
the core noun (e.g. "chair in sight", "cup is visible in the image"), not
just the bare noun -- this preserves stronger alignment (proj) with the
original caption embedding, closing most of the gap to the rule-based
parser's retrieval performance (see project notes: proj 0.664 -> 0.690,
R@1 0.2568 -> 0.2617 on NegBench Retrieval, full 5000-image set).
"""

import re
import numpy as np
import torch

import rule_based_extraction

# rule-based의 검증된 negator 리스트를 그대로 재사용 -- 이걸로 게이트가 놓치는
# 표현이 있으면 그건 rule-based도 못 잡는 표현이므로 공정한 비교가 보장됨.
# "absent"는 PRE_NEGATORS에 없고 POST_NEGATOR_PATTERN(is/are/was/were + absent)
# 에서만 다뤄지는데, 우리 게이트는 그 정규식을 안 쓰므로 따로 추가.
NEGATION_CUES = [neg.strip() for neg in rule_based_extraction.PRE_NEGATORS] + ["missing", "absent", "isn't"]

ANCHOR_WORDS = ["neutral", "balanced", "unbiased", "fair"]

SYSTEM_PROMPT = ("You are a precise linguistic tool. Given a sentence containing a negation, "
                  "output ONLY the exact phrase describing what is stated to be absent, not "
                  "present, or not happening -- copy the phrase verbatim from the sentence, "
                  "including any short descriptive or predicate wording around the core noun "
                  "(e.g. for 'there is no chair in sight', output 'chair in sight', not just "
                  "'chair'). If there is no single clear concept being negated (e.g. the "
                  "negation applies to an entire clause, an abstract situation, or a "
                  "time/quantity expression), output NONE. Output nothing else: no "
                  "explanation, no punctuation, no quotes.")

FEW_SHOT_EXAMPLES = [
    ("A man in a kitchen is making pizzas, but there is no chair in sight.", "chair in sight"),
    ("No fork is present, but a baker is busy working in the kitchen.", "fork is present"),
    ("No cup is visible in the image, but the small kitchen is equipped with appliances.", "cup is visible in the image"),
    ("soccer player celebrates without teammates after scoring", "teammates"),
    ("source of the contaminated water ingested by no one", "NONE"),
    ("i 'm not sure what this design is on , but it would n't make an interesting tattoo", "NONE"),
    ("No car is in the image, but the front end of a red motorcycle is on display.", "car is in the image"),
    ("private path from your deck to the ocean, not through the dunes", "dunes"),
]


def has_negation(text):
    t = text.lower()
    return any(cue in t for cue in NEGATION_CUES)


def build_messages(caption):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_sentence, ex_answer in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Sentence: {ex_sentence}"})
        messages.append({"role": "assistant", "content": ex_answer})
    messages.append({"role": "user", "content": f"Sentence: {caption}"})
    return messages


_ARTICLE_RE = re.compile(r"^(a|an|the)\s+")


def _normalize(phrase):
    p = phrase.lower().strip().strip(" .,;:'\"")
    p = _ARTICLE_RE.sub("", p)
    return p


def _verbatim_match(concept, caption):
    """Relaxed verbatim check: ignores leading articles (a/an/the) and
    naive singular/plural variation (trailing 's'), in both directions.
    Still requires the core noun phrase to actually appear in the caption --
    this is not a semantic/fuzzy match, just normalization of surface form."""
    cap = caption.lower()
    c = _normalize(concept)
    if not c:
        return False
    variants = {c}
    if c.endswith("s"):
        variants.add(c[:-1])
    else:
        variants.add(c + "s")
    return any(v in cap for v in variants if v)


def extract_negated_concepts(llm_model, llm_tokenizer, captions, batch_size=16, max_new_tokens=20):
    """Returns (concepts, failure_reason), both same length as captions.
    concepts[i]: extracted concept string, or None (no negation cue / NONE / failed validation).
    failure_reason[i]: one of "success", "gate_filtered", "explicit_none", "verbatim_mismatch".
    """
    concepts = [None] * len(captions)
    failure_reason = ["gate_filtered"] * len(captions)  # default: not sent to the LLM at all
    idx_to_process = [i for i, c in enumerate(captions) if has_negation(c)]
    if not idx_to_process:
        return concepts, failure_reason

    device = llm_model.device
    for start in range(0, len(idx_to_process), batch_size):
        batch_idx = idx_to_process[start:start + batch_size]
        batch = [captions[i] for i in batch_idx]
        chat_prompts = [
            llm_tokenizer.apply_chat_template(
                build_messages(c), tokenize=False, add_generation_prompt=True
            ) for c in batch
        ]
        enc = llm_tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True,
                             max_length=768).to(device)
        with torch.no_grad():
            out = llm_model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        decoded = llm_tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for i, d, caption in zip(batch_idx, decoded, batch):
            concept = d.strip().split("\n")[0].strip().strip('."\'')
            if concept.upper() == "NONE" or not concept:
                concepts[i] = None
                failure_reason[i] = "explicit_none"
            elif not _verbatim_match(concept, caption):
                concepts[i] = None  # invalid / hallucinated, treat as extraction failure
                failure_reason[i] = "verbatim_mismatch"
            else:
                concepts[i] = concept
                failure_reason[i] = "success"
    return concepts, failure_reason


def extract_negated_concepts_hybrid(llm_model, llm_tokenizer, captions, batch_size=16, max_new_tokens=20):
    """LLM 추출을 우선 시도하고, 실패한(explicit_none/verbatim_mismatch/gate_filtered)
    캡션만 rule-based로 대체. rule도 실패하면 그때만 진짜 실패(None)로 남김.
    failure_reason에 "llm_success"/"rule_fallback"/"both_failed_<원래사유>"를 남겨서
    최종 결과 중 몇 %가 어느 경로에서 왔는지 분해해서 볼 수 있게 함.

    Note (project finding): on NegBench Retrieval this was *not* better than
    LLM-only (v3 prompt) -- the extra coverage from rule_fallback captions
    (very ambiguous ones the LLM correctly gave up on) can pull scores down.
    Kept as an option for tasks where it *did* help (NegBench MCQ)."""
    concepts, failure_reason = extract_negated_concepts(
        llm_model, llm_tokenizer, captions, batch_size=batch_size, max_new_tokens=max_new_tokens
    )
    rule_concepts = rule_based_extraction.extract_negated_concepts(captions)

    final_concepts = list(concepts)
    final_reason = []
    for c, r, rc in zip(concepts, failure_reason, rule_concepts):
        if c is not None:
            final_reason.append("llm_success")
        elif rc is not None:
            final_reason.append("rule_fallback")
        else:
            final_reason.append(f"both_failed_{r}")

    for i, rc in enumerate(rule_concepts):
        if concepts[i] is None and rc is not None:
            final_concepts[i] = rc

    return final_concepts, final_reason


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


def compute_anchor(clip_model, clip_tokenizer, device):
    a = get_clip_text_embeddings(clip_model, clip_tokenizer, ANCHOR_WORDS, device)
    return a.mean(axis=0)


def extract_concepts_and_embeddings(clip_model, clip_tokenizer, device, texts,
                                     llm_model, llm_tokenizer, extract_fn=None, hybrid=False):
    """Runs extraction and CLIP encoding once. Returns e_c (N,dim),
    concepts (list of str/None), e_neg (N,dim) with zeros for invalid rows,
    valid_idx, and failure_reason (list of str, same length as texts).

    extract_fn: optional callable(captions, llm_model=None, llm_tokenizer=None)
    -> list of concept/None. Defaults to the LLM-based extractor. Pass
    rule_based_extraction.extract_negated_concepts for the rule-based parser.
    hybrid: if True, uses extract_negated_concepts_hybrid instead (overrides extract_fn)."""
    e_c = get_clip_text_embeddings(clip_model, clip_tokenizer, texts, device)

    if hybrid:
        concepts, failure_reason = extract_negated_concepts_hybrid(llm_model, llm_tokenizer, texts)
    elif extract_fn is None:
        concepts, failure_reason = extract_negated_concepts(llm_model, llm_tokenizer, texts)
    else:
        concepts = extract_fn(texts, llm_model=llm_model, llm_tokenizer=llm_tokenizer)
        failure_reason = ["success" if c is not None else "rule_based_no_match" for c in concepts]

    valid_idx = [i for i, c in enumerate(concepts) if c is not None]

    e_neg = np.zeros_like(e_c)
    if valid_idx:
        concept_texts = [concepts[i] for i in valid_idx]
        e_neg_valid = get_clip_text_embeddings(clip_model, clip_tokenizer, concept_texts, device)
        e_neg[valid_idx] = e_neg_valid

    return e_c, concepts, e_neg, valid_idx, failure_reason


def apply_correction_given_embeddings(e_c, e_neg, valid_idx, anchor, lam):
    """Given precomputed e_c/e_neg/valid_idx (from extract_concepts_and_embeddings),
    apply Eq. 2 for a given lambda. Cheap -- no LLM/CLIP calls, safe to call
    repeatedly for a lambda sweep."""
    e_star = e_c.copy()
    if not valid_idx:
        return e_star
    e_c_valid = e_c[valid_idx]
    e_neg_valid = e_neg[valid_idx]
    proj = (e_c_valid * e_neg_valid).sum(axis=1, keepdims=True) / \
           (e_neg_valid * e_neg_valid).sum(axis=1, keepdims=True)
    corrected = e_c_valid - lam * (proj * e_neg_valid - anchor[None, :])
    corrected = corrected / np.linalg.norm(corrected, axis=1, keepdims=True)
    e_star[valid_idx] = corrected
    return e_star


def apply_content_aware_correction(clip_model, clip_tokenizer, device, texts,
                                    llm_model, llm_tokenizer, anchor, lam):
    """Convenience wrapper for a single lambda (extraction + correction in
    one call). For a lambda sweep, use extract_concepts_and_embeddings +
    apply_correction_given_embeddings instead to avoid repeating extraction."""
    e_c, concepts, e_neg, valid_idx, failure_reason = extract_concepts_and_embeddings(
        clip_model, clip_tokenizer, device, texts, llm_model, llm_tokenizer
    )
    e_star = apply_correction_given_embeddings(e_c, e_neg, valid_idx, anchor, lam)
    return e_star, concepts, failure_reason