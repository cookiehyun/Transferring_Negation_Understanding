"""
Sample real captions from the NegBench negated-retrieval CSV and print
(caption -> extracted concept) for manual verification: is the extracted
phrase actually the negated concept, or just some substring that happened
to pass the "appears in the sentence" validation check?
"""

import os
import sys
import ast
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import extract_negated_concepts, has_negation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--retrieval_csv", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)

    df = pd.read_csv(args.retrieval_csv)
    caption_lists = [ast.literal_eval(c) for c in df["captions"]]
    all_captions = [c for lst in caption_lists for c in lst]
    negated_captions = [c for c in all_captions if has_negation(c)]
    print(f"Total captions: {len(all_captions)}, containing negation: {len(negated_captions)}")

    random.seed(args.seed)
    sample = random.sample(negated_captions, min(args.n_samples, len(negated_captions)))

    print(f"\nLoading LLM: {cfg.model.name_or_path}")
    dtype = getattr(torch, cfg.model.dtype)
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype, device_map=cfg.model.device_map,
    )
    llm_model.eval()

    print("\nExtracting...")
    concepts = extract_negated_concepts(llm_model, llm_tokenizer, sample)

    print("\n=== Results ===\n")
    n_none = 0
    for caption, concept in zip(sample, concepts):
        print(f"CAPTION: {caption}")
        print(f"EXTRACT: {concept}")
        print()
        if concept is None:
            n_none += 1

    print("=" * 40)
    print(f"extracted: {len(sample) - n_none}/{len(sample)}, NONE/failed: {n_none}")
    print("=" * 40)


if __name__ == "__main__":
    main()