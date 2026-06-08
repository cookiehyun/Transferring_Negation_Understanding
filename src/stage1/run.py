"""
Stage 1: Negation Direction Extraction using RepE
Reference: https://github.com/andyzoujm/representation-engineering
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from repe import repe_pipeline_registry

repe_pipeline_registry()

# ── 1. 모델 로드 ──────────────────────────────────────────────
model_name = "/fs/dss/home/gaad2403/models/llama3.1-8b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
print(f"Loaded {model_name}")
print(f"Architecture: {model.config.architectures}")
print(f"Num layers: {model.config.num_hidden_layers}")

# ── 2. 부정 쌍 데이터셋 구성 ─────────────────────────────────
pairs = [
    ("a photo of a dog",       "a photo of not a dog"),
    ("a photo of a cat",       "a photo of not a cat"),
    ("a photo of a car",       "a photo of not a car"),
    ("a photo of a tree",      "a photo of not a tree"),
    ("a photo of a person",    "a photo of not a person"),
    ("an image with a bird",   "an image without a bird"),
    ("an image with a flower", "an image without a flower"),
    ("an image with a chair",  "an image without a chair"),
    ("there is a dog",         "there is no dog"),
    ("there is a cat",         "there is no cat"),
    ("there is a car",         "there is no car"),
    ("there is a tree",        "there is no tree"),
    ("the sky is blue",        "the sky is not blue"),
    ("the grass is green",     "the grass is not green"),
    ("the room is bright",     "the room is not bright"),
]

# RepE 형식: [pos, neg, pos, neg, ...]
train_data = []
train_labels = []
for pos, neg in pairs:
    train_data.append(pos)
    train_data.append(neg)
    train_labels.append([True, False])

print(f"Train pairs: {len(pairs)}")

# ── 3. RepE 파이프라인 설정 ───────────────────────────────────
rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers - 1, -1))
n_difference = 1
direction_method = 'pca'

rep_reading_pipeline = pipeline(
    "rep-reading",
    model=model,
    tokenizer=tokenizer
)

# ── 4. 부정 방향 벡터 추출 ────────────────────────────────────
print("\nExtracting negation direction vectors...")

negation_rep_reader = rep_reading_pipeline.get_directions(
    train_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=n_difference,
    train_labels=train_labels,
    direction_method=direction_method,
    batch_size=8,
)
print("Done.")

# ── 5. 레이어별 분류 정확도 평가 ─────────────────────────────
print("\nEvaluating per-layer accuracy...")

H_tests = rep_reading_pipeline(
    train_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    rep_reader=negation_rep_reader,
    batch_size=8,
)

results = {}
for layer in hidden_layers:
    H_test = [H[layer] for H in H_tests]
    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
    sign = negation_rep_reader.direction_signs[layer]
    eval_func = min if sign == -1 else max
    results[layer] = np.mean([eval_func(H) == H[0] for H in H_test])

print("\nPer-layer accuracy:")
for layer in sorted(results.keys(), reverse=True):
    bar = '█' * int(results[layer] * 20)
    print(f"  Layer {layer:4d}: {results[layer]:.4f} {bar}")

# ── 6. 시각화 ─────────────────────────────────────────────────
layers = list(results.keys())
accuracies = [results[l] for l in layers]

plt.figure(figsize=(12, 5))
plt.plot(layers, accuracies, 'bo-', linewidth=2, markersize=6)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance (0.5)')
plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Perfect (1.0)')
plt.xlabel('Layer (negative index)', fontsize=12)
plt.ylabel('Classification Accuracy', fontsize=12)
plt.title(f'Negation Direction Accuracy per Layer\n(RepE, Llama-3.1-8B)', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/stage1_repe_accuracy.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/figures/stage1_repe_accuracy.png")

# ── 7. 벡터 저장 (Stage 2용) ──────────────────────────────────
best_layer = max(results, key=results.get)
print(f"\nBest layer: {best_layer} (accuracy: {results[best_layer]:.4f})")

np.save(f'outputs/vectors/negation_direction_layer{best_layer}.npy',
        negation_rep_reader.directions[best_layer])

all_directions = {layer: negation_rep_reader.directions[layer] for layer in hidden_layers}
np.save('outputs/vectors/negation_directions_all_layers.npy', all_directions)

print("\n=== SUMMARY ===")
print(f"Model: {model_name}")
print(f"Pairs: {len(pairs)}")
print(f"Best layer: {best_layer}, Accuracy: {results[best_layer]:.4f}")
print("Next: Run stage2")