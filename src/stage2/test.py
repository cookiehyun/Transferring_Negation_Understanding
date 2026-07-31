"""
Diagnostic: does the v3 few-shot prompt generalize to negation patterns NOT
shown in its few-shot examples, or does it only work on memorized patterns?
Few-shot examples cover: "is present", "is visible in the image", "is in the
image", "not X", "without X", "no one". Test sentences below use DIFFERENT
copula/negation structures ("can be seen", "does not appear", "includes no")
to check generalization.
"""
import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from content_aware_correction import extract_negated_concepts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage2_config


def main():
    cfg = load_stage2_config("configs/stage2.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dtype = torch.bfloat16 if device == "cpu" else getattr(torch, cfg.model.dtype)

    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        dtype=dtype,
        device_map=device if device == "cpu" else cfg.model.device_map,
        low_cpu_mem_usage=True,
    )
    llm_model.eval()

    # few-shot에 없는 negation/copula 구조들 -- 일반화 테스트용
    test_cases = [
        "No dog can be seen in this photo of a park.",
        "A handbag does not appear in the image.",
        "This scene includes no cup, only a plate and fork.",
        "The room lacks a chair entirely.",
        "There isn't a single car parked on this street.",
        "A bicycle is nowhere to be found near the fence.",
    ]

    concepts, failure_reason = extract_negated_concepts(llm_model, llm_tokenizer, test_cases)

    print("\n=== 일반화 테스트 (few-shot에 없는 패턴) ===")
    for c, concept, reason in zip(test_cases, concepts, failure_reason):
        print(f"  {c!r}")
        print(f"    -> concept: {concept!r}  (reason: {reason})")
        print()


if __name__ == "__main__":
    main()
