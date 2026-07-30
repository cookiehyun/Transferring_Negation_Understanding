"""
Quick test: run only the LLM negated-concept extraction step on real CC-Neg
captions and print (caption -> extracted concept) for manual inspection.
No CLIP involved, so this runs much faster than the full pipeline.
"""

import os
import sys
import json
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config

SYSTEM_PROMPT = ("You are a precise linguistic tool. Given a sentence containing a negation, "
                  "output ONLY the exact noun phrase that is being negated -- the thing stated "
                  "to be absent, not present, or not happening. The phrase must be copied "
                  "verbatim from the sentence. If there is no single clear concept being negated "
                  "(e.g. the negation applies to an entire clause, an abstract situation, or a "
                  "time/quantity expression), output NONE. Output nothing else: no explanation, "
                  "no punctuation, no quotes.")

FEW_SHOT_EXAMPLES = [
    ("person, not waving a flag from the crowd", "flag"),
    ("soccer player celebrates without teammates after scoring", "teammates"),
    ("source of the contaminated water ingested by no one", "NONE"),
    ("i 'm not sure what this design is on , but it would n't make an interesting tattoo", "NONE"),
    ("this drum set is not percent off today", "NONE"),
    ("private path from your deck to the ocean, not through the dunes", "dunes"),
]


def build_messages(caption):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_sentence, ex_answer in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Sentence: {ex_sentence}"})
        messages.append({"role": "assistant", "content": ex_answer})
    messages.append({"role": "user", "content": f"Sentence: {caption}"})
    return messages


def extract_negated_concepts(llm_model, llm_tokenizer, captions, batch_size=16, max_new_tokens=12):
    concepts = []
    device = llm_model.device
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
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
        for d, caption in zip(decoded, batch):
            concept = d.strip().split("\n")[0].strip().strip('."\'')
            # validation: reject if the model didn't actually copy a real
            # substring from the sentence (matches the "flag"-anchoring bug)
            if concept.upper() == "NONE" or not concept:
                concepts.append(None)
            elif concept.lower() not in caption.lower():
                concepts.append(f"INVALID({concept})")
            else:
                concepts.append(concept)
    return concepts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pairs_path", type=str, default="outputs/ccneg/ccneg_pairs.json")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)

    with open(args.pairs_path) as f:
        d = json.load(f)
    pairs = d["pairs"]

    random.seed(args.seed)
    sample = random.sample(pairs, args.n_samples)
    neg_captions = [neg for pos, neg in sample]

    print(f"Loading LLM: {cfg.model.name_or_path}")
    dtype = getattr(torch, cfg.model.dtype)
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map,
    )
    llm_model.eval()

    print("\nExtracting...")
    concepts = extract_negated_concepts(llm_model, llm_tokenizer, neg_captions)

    print("\n=== Results ===\n")
    n_valid, n_none, n_invalid = 0, 0, 0
    for (pos, neg), concept in zip(sample, concepts):
        print(f"POS    : {pos}")
        print(f"NEG    : {neg}")
        print(f"EXTRACT: {concept}")
        print()
        if concept is None:
            n_none += 1
        elif isinstance(concept, str) and concept.startswith("INVALID"):
            n_invalid += 1
        else:
            n_valid += 1

    print("=" * 40)
    print(f"valid: {n_valid}, NONE: {n_none}, invalid (not in sentence): {n_invalid}")
    print("=" * 40)


if __name__ == "__main__":
    main()