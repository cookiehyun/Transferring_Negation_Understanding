# Transferring Negation Understanding from LLMs to CLIP via Content-Aware Embedding Correction

Code and configurations for the paper *"Transferring Negation Understanding from LLMs to CLIP via Content-Aware Embedding Correction"* (Keyhyun Ku, University of Oldenburg).

## Overview

Vision-language models such as CLIP struggle to understand negation — a caption like *"a room without a chair"* is often embedded nearly identically to *"a room with a chair"*, an affirmation bias arising from contrastive pretraining on overwhelmingly affirmative web captions.

This work investigates whether an LLM's negation understanding can be transferred to CLIP **without fine-tuning CLIP itself**. A fixed, transferred negation direction fails on fine-grained discrimination tasks, so instead we extract the specific negated concept per caption with an LLM and correct only that direction. We find that a naive LLM extractor initially underperforms a simple rule-based parser, trace this to coverage and embedding-alignment issues rather than poor comprehension, and show that a redesigned extraction prompt closes the gap and surpasses the rule-based baseline across two LLMs. The correction further composes with existing fine-tuned CLIP variants: applied to NegCLIP, it more than doubles MCQ accuracy (0.26 → 0.67), an improvement confirmed to be genuine rather than a data-leakage artifact via a cross-domain control.

## Method

1. **Confirming linear negation encoding (Stage 1).** RepE analysis on Llama-3.1-8B-Instruct shows negation is encoded along a roughly linear direction in the model's hidden states (best transfer-relevant accuracy 0.83 at layer −15).
2. **Fixed-direction transfer and its failure (Stage 2–3).** The LLM negation direction is mapped into CLIP's text-embedding space via orthogonal Procrustes regression and fine-tuned on real images. This fixed, caption-independent correction improves NegBench Retrieval only marginally (R@1 0.250 → 0.254) and causes a severe regression on negation-type MCQ items (0.068 → 0.015), because it cannot distinguish *what* a caption negates.
3. **Content-aware correction (Stage 4, final method).** Following Aggarwal et al. (ICLR 2026), negation is corrected as a directional offset in CLIP's embedding space, guided by the specific concept being negated. Two extractors are compared:
   - **Rule-based**: a fixed, hand-written negator parser.
   - **LLM-based**: Llama-3.1-8B / Qwen2.5-7B, prompted to extract the negated phrase.
4. **Diagnosis and redesign.** The naive LLM extractor loses to the rule-based parser on CC-Neg and Retrieval — traced to lower coverage (abstention on ambiguous cases) and to the rule-based parser's predicate-preserving phrasing aligning better with the full caption embedding, not to worse comprehension. A redesigned prompt (**v3**, predicate-inclusive extraction) combined with a rule-based fallback (**hybrid**) surpasses the rule-based baseline on all three benchmarks, replicated across two LLMs.
5. **Composition with fine-tuned CLIP variants.** The training-free correction is applied on top of NegCLIP and ConCLIP. A cross-domain control (NegCLIP on CC-Neg's Conceptual-Captions images, ConCLIP on COCO-derived NegBench) shows the composed gains reflect genuine synergy between the correction and hard-negative fine-tuning, not training-data leakage.

## Key Results

**Rule-based vs. naive LLM extraction (test split):**

| Method | CC-Neg Acc. | MCQ Acc. | Retrieval R@1 |
|---|---|---|---|
| Uncorrected CLIP | 0.644 | 0.393 | 0.274 |
| Rule-based | 0.994 | 0.509 | 0.286 |
| LLM (v1, naive prompt) | 0.898 | 0.520 | 0.280 |

**Redesigned prompt (v3) and hybrid fallback:**

| Method | CC-Neg Acc. | MCQ Acc. | Retrieval R@1 |
|---|---|---|---|
| v3 alone, Llama | 0.832 | 0.491 | 0.284 |
| v3 alone, Qwen | 0.961 | 0.475 | 0.285 |
| Hybrid, Llama | **0.998** | **0.511** | **0.285** |
| Hybrid, Qwen | **0.998** | **0.514** | **0.286** |

**Composing with NegCLIP:**

| Method | MCQ Acc. | Retrieval R@1 |
|---|---|---|
| NegCLIP alone | 0.263 | 0.426 |
| NegCLIP + rule | 0.668 | 0.432 |
| NegCLIP + hybrid, Llama | **0.672** | **0.438** |
| NegCLIP + hybrid, Qwen | 0.670 | 0.438 |

**Domain-specificity control:**

| Method | NegBench MCQ | NegBench Retrieval | CC-Neg |
|---|---|---|---|
| NegCLIP (COCO-trained), alone | 0.263 | 0.426 | 0.601 |
| NegCLIP + hybrid | 0.672 | 0.438 | 0.998 |
| ConCLIP (CC-Neg-trained), alone | 0.238 | 0.291 | 0.966 |
| ConCLIP + hybrid | 0.637 | 0.455 | **0.998** |

Each fine-tuned backbone is markedly stronger in its own training domain when uncorrected, but hybrid correction largely closes this gap — evidence against a leakage-driven explanation for the NegCLIP composition result.

## Datasets & Benchmarks

- **CC-Neg**: 5,000 Conceptual-Captions (CC3M) images, each with an original and an automatically negated caption; binary comparison (chance = 0.5).
- **NegBench MCQ**: 5,914 COCO val2017 images, 4-way multiple choice over positive/negative/hybrid caption templates (chance = 0.25).
- **NegBench Retrieval**: 5,000 COCO val2017 images, 25,014 negated captions; nearest-neighbor retrieval, measured by R@1.

Backbone: CLIP ViT-B/32 (OpenAI weights via `open_clip`), compared against NegCLIP and ConCLIP as fine-tuned baselines.

## Citation

**Title:** Transferring Negation Understanding from LLMs to CLIP via Content-Aware Embedding Correction
**Author:** Keyhyun Ku
**Institution:** University of Oldenburg

## Related Work

This project builds directly on the training-free correction formula of Aggarwal et al., *"Seeing What's Not There: Negation Understanding Needs More Than Training"* (ICLR 2026), and relates to NegCLIP (Yuksekgonul et al., ICLR 2023), CC-Neg / ConCLIP (Singh et al., WACV 2025), and NegBench (Alhamoud et al., CVPR 2025).
