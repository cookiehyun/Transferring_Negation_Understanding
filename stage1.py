"""
Stage 1: LLM 부정 방향 벡터 추출 및 시각화
목표: GPT-2 hidden state에서 부정 방향이 일관되게 존재하는지 확인

설치 필요:
pip install transformers torch scikit-learn matplotlib numpy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. 모델 로드 ──────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

# ── 2. 긍정/부정 문장 쌍 ─────────────────────────────────────
# 다양한 카테고리로 구성 (일반화 검증용)
pairs = [
    ("a photo of a dog",        "a photo of not a dog"),
    ("a photo of a cat",        "a photo of not a cat"),
    ("a photo of a car",        "a photo of not a car"),
    ("a photo of a tree",       "a photo of not a tree"),
    ("a photo of a person",     "a photo of not a person"),
    ("an image with a bird",    "an image without a bird"),
    ("an image with a flower",  "an image without a flower"),
    ("an image with a chair",   "an image without a chair"),
    ("there is a dog",          "there is no dog"),
    ("there is a cat",          "there is no cat"),
    ("there is a car",          "there is no car"),
    ("there is a tree",         "there is no tree"),
    ("the sky is blue",         "the sky is not blue"),
    ("the grass is green",      "the grass is not green"),
    ("the room is bright",      "the room is not bright"),
]

# ── 3. Hidden State 추출 함수 ─────────────────────────────────
def get_hidden_states(text, layer_idx=-1):
    """
    텍스트의 hidden state 추출
    layer_idx: -1이면 마지막 레이어, 정수면 해당 레이어
    반환: 마지막 토큰의 hidden state (GPT-2는 [EOS] 대신 마지막 토큰 사용)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.hidden_states: tuple of (n_layers+1) tensors
    # 각 tensor shape: (batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states  # 13개 레이어 (embedding + 12 transformer)
    
    # 마지막 토큰의 hidden state 사용 (GPT-2 스타일)
    h = hidden_states[layer_idx][0, -1, :].cpu().numpy()
    return h

def get_all_layer_hidden_states(text):
    """모든 레이어의 hidden state 반환"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    all_layers = []
    for layer_hidden in outputs.hidden_states:
        h = layer_hidden[0, -1, :].cpu().numpy()
        all_layers.append(h)
    return all_layers  # 13개 (layer 0 ~ 12)

# ── 4. 모든 레이어에서 부정 벡터 추출 ────────────────────────
print("\nExtracting hidden states from all layers...")

n_layers = 13  # GPT-2: embedding(0) + 12 transformer layers
n_pairs = len(pairs)

# 각 레이어별 차이 벡터 저장
# shape: (n_layers, n_pairs, hidden_dim)
diff_vectors_per_layer = [[] for _ in range(n_layers)]

pos_vectors_per_layer = [[] for _ in range(n_layers)]
neg_vectors_per_layer = [[] for _ in range(n_layers)]

for pos_text, neg_text in pairs:
    pos_layers = get_all_layer_hidden_states(pos_text)
    neg_layers = get_all_layer_hidden_states(neg_text)
    
    for l in range(n_layers):
        diff = neg_layers[l] - pos_layers[l]
        diff_vectors_per_layer[l].append(diff)
        pos_vectors_per_layer[l].append(pos_layers[l])
        neg_vectors_per_layer[l].append(neg_layers[l])

# numpy 변환
for l in range(n_layers):
    diff_vectors_per_layer[l] = np.array(diff_vectors_per_layer[l])
    pos_vectors_per_layer[l] = np.array(pos_vectors_per_layer[l])
    neg_vectors_per_layer[l] = np.array(neg_vectors_per_layer[l])

# ── 5. 핵심 분석: 부정 방향의 일관성 측정 ────────────────────
print("\n=== Negation Direction Consistency per Layer ===")
print(f"{'Layer':>6} | {'Mean Cosine Sim':>15} | {'Std':>8} | {'Interpretation':>20}")
print("-" * 60)

layer_consistency = []

for l in range(n_layers):
    diffs = diff_vectors_per_layer[l]  # (n_pairs, hidden_dim)
    
    # 모든 쌍의 코사인 유사도 계산
    sims = cosine_similarity(diffs)
    
    # 대각선 제외한 상삼각 부분만
    upper_tri = sims[np.triu_indices(n_pairs, k=1)]
    mean_sim = upper_tri.mean()
    std_sim = upper_tri.std()
    
    layer_consistency.append(mean_sim)
    
    if mean_sim > 0.5:
        interp = "★ CONSISTENT"
    elif mean_sim > 0.2:
        interp = "△ moderate"
    else:
        interp = "✗ inconsistent"
    
    print(f"{l:>6} | {mean_sim:>15.4f} | {std_sim:>8.4f} | {interp:>20}")

# 가장 일관된 레이어 찾기
best_layer = np.argmax(layer_consistency)
print(f"\n→ Best layer for negation direction: Layer {best_layer} "
      f"(consistency: {layer_consistency[best_layer]:.4f})")

# ── 6. 평균 부정 방향 벡터 추출 ──────────────────────────────
print(f"\n=== Extracting Negation Direction Vector from Layer {best_layer} ===")

diffs_best = diff_vectors_per_layer[best_layer]

# 방법 1: Mean difference
n_vec_mean = diffs_best.mean(axis=0)
n_vec_mean_normalized = n_vec_mean / np.linalg.norm(n_vec_mean)

# 방법 2: PCA (첫 번째 주성분)
pca = PCA(n_components=1)
pca.fit(diffs_best)
n_vec_pca = pca.components_[0]
explained_var = pca.explained_variance_ratio_[0]

print(f"PCA explained variance ratio: {explained_var:.4f}")
print(f"→ {explained_var*100:.1f}% of variance captured by first PC")

if explained_var > 0.5:
    print("★ Strong single direction exists → negation is linearly encoded!")
elif explained_var > 0.3:
    print("△ Moderate direction exists → partial linear encoding")
else:
    print("✗ Weak direction → negation may not be linearly encoded")

# Mean vs PCA 방향 일치 확인
sim_mean_pca = abs(np.dot(n_vec_mean_normalized, n_vec_pca))
print(f"Cosine sim (Mean vs PCA direction): {sim_mean_pca:.4f}")

# ── 7. PCA 시각화 ─────────────────────────────────────────────
print("\nGenerating PCA visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Layer별 일관성
ax1 = axes[0]
ax1.plot(range(n_layers), layer_consistency, 'bo-', linewidth=2, markersize=8)
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold (0.5)')
ax1.axvline(x=best_layer, color='g', linestyle='--', alpha=0.7, 
            label=f'Best layer ({best_layer})')
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Mean Cosine Similarity of Diff Vectors', fontsize=11)
ax1.set_title('Negation Direction Consistency per Layer', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(n_layers))

# Plot 2: Best layer에서 2D PCA 시각화
ax2 = axes[1]
pca2d = PCA(n_components=2)

all_vecs = np.vstack([
    pos_vectors_per_layer[best_layer],
    neg_vectors_per_layer[best_layer]
])
all_vecs_2d = pca2d.fit_transform(all_vecs)

pos_2d = all_vecs_2d[:n_pairs]
neg_2d = all_vecs_2d[n_pairs:]

ax2.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', s=100, 
            label='Positive', alpha=0.7, zorder=3)
ax2.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', s=100, 
            label='Negative', alpha=0.7, zorder=3)

# 쌍을 화살표로 연결
for i in range(n_pairs):
    ax2.annotate('', 
                xy=(neg_2d[i, 0], neg_2d[i, 1]),
                xytext=(pos_2d[i, 0], pos_2d[i, 1]),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

ax2.set_xlabel(f'PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax2.set_title(f'Positive vs Negative Embeddings\n(Layer {best_layer}, PCA 2D)', 
              fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: 차이 벡터들의 PCA (일관성 시각화)
ax3 = axes[2]
pca_diff = PCA(n_components=2)
diffs_2d = pca_diff.fit_transform(diffs_best)

scatter = ax3.scatter(diffs_2d[:, 0], diffs_2d[:, 1], 
                      c=range(n_pairs), cmap='viridis', s=120, zorder=3)

for i, (pos_text, _) in enumerate(pairs):
    label = pos_text.split()[-1]  # 마지막 단어만
    ax3.annotate(label, (diffs_2d[i, 0], diffs_2d[i, 1]), 
                fontsize=8, ha='right')

# 원점 표시
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.2)
ax3.scatter([0], [0], c='black', s=200, marker='*', zorder=5, label='Origin')

ax3.set_xlabel(f'PC1 ({pca_diff.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax3.set_ylabel(f'PC2 ({pca_diff.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax3.set_title(f'Difference Vectors (neg - pos)\n(Layer {best_layer})', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

var_explained = (pca_diff.explained_variance_ratio_[0] + 
                 pca_diff.explained_variance_ratio_[1])
ax3.set_title(f'Difference Vectors (neg - pos)\n'
              f'Layer {best_layer} | 2PC explains {var_explained*100:.1f}%', 
              fontsize=11)

plt.tight_layout()
plt.savefig('stage1_negation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: stage1_negation_analysis.png")

# ── 8. 결과 요약 ──────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Model: GPT-2 (hidden_dim=768, n_layers=12)")
print(f"Pairs tested: {n_pairs}")
print(f"Best layer: {best_layer}")
print(f"Consistency score: {layer_consistency[best_layer]:.4f}")
print(f"PCA explained variance (1st PC): {explained_var*100:.1f}%")
print()

if layer_consistency[best_layer] > 0.5 and explained_var > 0.3:
    print("✅ CONCLUSION: Negation direction vector EXISTS in GPT-2")
    print("   → Linear structure confirmed")
    print("   → Proceed to Stage 2: Transfer to CLIP space")
elif layer_consistency[best_layer] > 0.2:
    print("⚠️  CONCLUSION: Weak negation direction found")
    print("   → Try larger model (GPT-2-medium or Llama-3)")
else:
    print("❌ CONCLUSION: No consistent negation direction in GPT-2")
    print("   → GPT-2 may be too small → Try larger model")

print()
print("Next step: Run stage2_projection.py")

# 벡터 저장 (Stage 2에서 사용)
np.save('negation_vector_mean.npy', n_vec_mean_normalized)
np.save('negation_vector_pca.npy', n_vec_pca)
np.save(f'hidden_states_layer{best_layer}.npy', 
        np.vstack([pos_vectors_per_layer[best_layer], 
                   neg_vectors_per_layer[best_layer]]))
print("Saved vectors for Stage 2.")