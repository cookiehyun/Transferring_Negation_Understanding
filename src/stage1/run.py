"""
Stage 1: Negation Direction Extraction using RepE
Reference: Zou et al. (2023) arXiv:2310.01405

Usage:
    python stage1_run.py

Requirements:
    pip install repe transformers torch numpy matplotlib scikit-learn
    (repe: pip install git+https://github.com/andyzoujm/representation-engineering.git)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from repe import repe_pipeline_registry

repe_pipeline_registry()

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/vectors", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. 모델 로드
# ─────────────────────────────────────────────────────────────
model_name = "/fs/dss/home/gaad2403/models/llama3.1-8b"

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
print(f"Loaded. Layers: {model.config.num_hidden_layers}")

# ─────────────────────────────────────────────────────────────
# 2. 부정 데이터셋 생성 (템플릿 × 개념 풀 조합)
#    Reference: "Interpreting Negation in GPT-2" arXiv:2603.12423
#    5 template types × 7 negation markers → 500+ pairs
# ─────────────────────────────────────────────────────────────

import random

# ── 템플릿 타입별로 분리 (버그 수정: attr/has 혼재 제거) ──

exist_templates = [
    ("a photo of a {X}",          "a photo of not a {X}"),
    ("an image containing a {X}", "an image not containing a {X}"),
    ("an image with a {X}",       "an image without a {X}"),
    ("there is a {X}",            "there is no {X}"),
    ("there is a {X}",            "there is not a {X}"),
    ("I can see a {X}",           "I cannot see a {X}"),
    ("the scene shows a {X}",     "the scene does not show a {X}"),
]

attr_templates = [
    ("the {X} is {Y}",            "the {X} is not {Y}"),
    ("the {X} is {Y}",            "the {X} is never {Y}"),
    ("the {X} looks {Y}",         "the {X} does not look {Y}"),
    ("the {X} appears {Y}",       "the {X} does not appear {Y}"),
]

can_templates = [
    ("a {X} can {V}",             "a {X} cannot {V}"),
    ("a {X} can {V}",             "a {X} can't {V}"),
    ("a {X} is able to {V}",      "a {X} is unable to {V}"),
]

has_templates = [
    ("the {X} has a {Y}",         "the {X} does not have a {Y}"),
    ("the {X} has a {Y}",         "the {X} doesn't have a {Y}"),
    ("a {X} with a {Y}",          "a {X} without a {Y}"),
]

# ── 개념 풀 (확장) ──

objects = [
    "dog", "cat", "car", "tree", "bird", "flower", "chair", "person",
    "boat", "plane", "horse", "cow", "sheep", "elephant", "bicycle",
    "bus", "train", "motorcycle", "truck", "pizza", "apple", "banana",
    "bottle", "cup", "laptop", "phone", "book", "clock", "knife", "fork",
    "umbrella", "ball", "hat", "shoe", "bag", "lamp", "table", "window",
    "door", "fence", "bridge", "mountain", "river", "beach", "cloud",
    "fire", "mirror", "camera", "keyboard", "mouse", "pillow", "blanket",
]

attributes = [
    ("sky",    "blue"),    ("grass",  "green"),   ("room",   "bright"),
    ("wall",   "white"),   ("road",   "wet"),      ("light",  "on"),
    ("door",   "open"),    ("car",    "red"),      ("box",    "full"),
    ("water",  "clean"),   ("sky",    "cloudy"),   ("room",   "empty"),
    ("bag",    "heavy"),   ("floor",  "dirty"),    ("fire",   "hot"),
    ("window", "broken"),  ("table",  "round"),    ("shirt",  "clean"),
    ("sky",    "dark"),    ("road",   "long"),     ("house",  "old"),
    ("cup",    "empty"),   ("book",   "thick"),    ("car",    "fast"),
    ("tree",   "tall"),    ("room",   "cold"),     ("light",  "bright"),
    ("door",   "closed"),  ("bag",    "small"),    ("wall",   "black"),
]

can_actions = [
    ("bird",     "fly"),      ("fish",     "swim"),    ("dog",      "bark"),
    ("cat",      "purr"),     ("horse",    "run"),     ("baby",     "walk"),
    ("snake",    "climb"),    ("elephant", "roar"),    ("duck",     "swim"),
    ("parrot",   "talk"),     ("rabbit",   "jump"),    ("lion",     "roar"),
    ("penguin",  "swim"),     ("eagle",    "soar"),    ("dolphin",  "jump"),
    ("monkey",   "climb"),    ("frog",     "leap"),    ("wolf",     "howl"),
    ("bee",      "sting"),    ("butterfly","fly"),
]

has_pairs = [
    ("dog",      "tail"),     ("cat",      "whisker"), ("bird",     "wing"),
    ("car",      "wheel"),    ("house",    "window"),  ("tree",     "leaf"),
    ("person",   "hand"),     ("horse",    "mane"),    ("fish",     "fin"),
    ("elephant", "trunk"),    ("lion",     "mane"),    ("rabbit",   "ear"),
    ("bicycle",  "wheel"),    ("plane",    "wing"),    ("cow",      "horn"),
    ("guitar",   "string"),   ("phone",    "screen"),  ("shoe",     "lace"),
    ("hat",      "brim"),     ("bag",      "handle"),
]

# ── 쌍 생성 (타입별 독립 루프) ──
pairs = []

for pos_t, neg_t in exist_templates:
    for obj in objects:
        pairs.append((pos_t.format(X=obj), neg_t.format(X=obj)))

for pos_t, neg_t in attr_templates:
    for subj, attr in attributes:
        pairs.append((pos_t.format(X=subj, Y=attr), neg_t.format(X=subj, Y=attr)))

for pos_t, neg_t in can_templates:
    for subj, verb in can_actions:
        pairs.append((pos_t.format(X=subj, V=verb), neg_t.format(X=subj, V=verb)))

for pos_t, neg_t in has_templates:
    for subj, obj in has_pairs:
        pairs.append((pos_t.format(X=subj, Y=obj), neg_t.format(X=subj, Y=obj)))

# 중복 제거 + 셔플
pairs = list(dict.fromkeys(pairs))
random.seed(42)
random.shuffle(pairs)

# train / test 분리
n_train = int(len(pairs) * 0.8)
train_pairs = pairs[:n_train]
test_pairs  = pairs[n_train:]

print(f"\nDataset: {len(pairs)} total pairs")
print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")
print(f"  Example: '{train_pairs[0][0]}' | '{train_pairs[0][1]}'")

# ─────────────────────────────────────────────────────────────
# 3. RepE 파이프라인 설정
# ─────────────────────────────────────────────────────────────
rep_token      = -1   # 마지막 토큰의 hidden state 사용
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
direction_method = "pca"

rep_reading_pipeline = pipeline(
    "rep-reading",
    model=model,
    tokenizer=tokenizer,
)

# ─────────────────────────────────────────────────────────────
# 4. Train 데이터로 부정 방향 벡터 추출
# ─────────────────────────────────────────────────────────────
train_data   = []
train_labels = []
for pos, neg in train_pairs:
    train_data.append(pos)
    train_data.append(neg)
    train_labels.append([True, False])

print("\nExtracting negation direction vectors (train)...")
negation_reader = rep_reading_pipeline.get_directions(
    train_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=1,
    train_labels=train_labels,
    direction_method=direction_method,
    batch_size=16,
)
print("Done.")

# ─────────────────────────────────────────────────────────────
# 5. Test 데이터로 레이어별 정확도 평가
# ─────────────────────────────────────────────────────────────
test_data   = []
test_labels = []
for pos, neg in test_pairs:
    test_data.append(pos)
    test_data.append(neg)
    test_labels.append([True, False])

print("\nEvaluating on test set...")
H_tests = rep_reading_pipeline(
    test_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    rep_reader=negation_reader,
    batch_size=16,
)

results = {}
for layer in hidden_layers:
    H = [h[layer] for h in H_tests]
    H = [H[i:i+2] for i in range(0, len(H), 2)]
    sign = negation_reader.direction_signs[layer]
    eval_fn = min if sign == -1 else max
    results[layer] = np.mean([eval_fn(h) == h[0] for h in H])

print("\n=== Per-layer Accuracy (Test) ===")
for layer in sorted(results.keys(), reverse=True):
    bar = "█" * int(results[layer] * 20)
    mark = " ★" if results[layer] >= 0.7 else ""
    print(f"  Layer {layer:4d}: {results[layer]:.4f}  {bar}{mark}")

best_layer = max(results, key=results.get)
print(f"\nBest layer: {best_layer}  (accuracy: {results[best_layer]:.4f})")

# ─────────────────────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────────────────────
layers     = sorted(results.keys(), reverse=True)
accuracies = [results[l] for l in layers]

plt.figure(figsize=(14, 5))
plt.plot(layers, accuracies, "bo-", linewidth=2, markersize=5)
plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Chance (0.5)")
plt.axhline(y=1.0, color="g", linestyle="--", alpha=0.7, label="Perfect (1.0)")
plt.axvline(x=best_layer, color="orange", linestyle="--", alpha=0.8,
            label=f"Best layer ({best_layer})")
plt.xlabel("Layer (negative index)", fontsize=12)
plt.ylabel("Classification Accuracy", fontsize=12)
plt.title(f"Negation Direction Accuracy per Layer\n"
          f"(RepE · Llama-3.1-8B · {len(test_pairs)} test pairs)", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_fig = "outputs/figures/stage1_accuracy.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {out_fig}")

# ─────────────────────────────────────────────────────────────
# 7. 벡터 저장 (Stage 2 입력용)
# ─────────────────────────────────────────────────────────────
best_vec = negation_reader.directions[best_layer]
np.save(f"outputs/vectors/negation_direction_layer{best_layer}.npy", best_vec)

all_dirs = {layer: negation_reader.directions[layer] for layer in hidden_layers}
np.save("outputs/vectors/negation_directions_all_layers.npy", all_dirs)

print(f"Saved vectors: outputs/vectors/")

# ─────────────────────────────────────────────────────────────
# 8. 요약
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Model       : Llama-3.1-8B")
print(f"Train pairs : {len(train_pairs)}")
print(f"Test pairs  : {len(test_pairs)}")
print(f"Best layer  : {best_layer}")
print(f"Accuracy    : {results[best_layer]:.4f}")
if results[best_layer] >= 0.7:
    print("✅ Negation direction confirmed → proceed to Stage 2")
elif results[best_layer] >= 0.55:
    print("⚠️  Weak direction → consider more data or larger model")
else:
    print("❌ No consistent direction found")
print("="*50)