"""
Compares alternative ways to fit W (LLM diff -> CLIP diff), without
re-running the LLM or CLIP. Requires llm_diff.npy / clip_diff.npy /
clip_neg_dir_native.npy from a stage2 run that included the caching patch.

Motivation: hyperalignment research (cross-subject representation alignment,
directly analogous to our cross-model problem) found that pure orthogonal
Procrustes can be too rigid, and that a small relaxation of the orthogonality
constraint -- via regularized CCA -- improves alignment quality (Haxby et al.,
eLife 2020; Xu et al. 2012, connecting regularized hyperalignment to CCA).
We approximate that relaxation here with a ridge-regularized (non-orthogonal)
map and a blend between it and the orthogonal solution, rather than
implementing full regularized CCA.
"""

import os
import argparse
import numpy as np
from sklearn.linear_model import RidgeCV


def orthogonal_procrustes_map(X, Y):
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def ridge_map(X, Y, alphas):
    reg = RidgeCV(alphas=alphas, fit_intercept=False)
    reg.fit(X, Y)
    print(f"  selected ridge alpha: {reg.alpha_:.4g}")
    return reg.coef_.T  # (d_source, d_target)


def transfer_similarity(llm_dir, W, clip_native_dir):
    t = llm_dir @ W
    t = t / np.linalg.norm(t)
    return float(t @ clip_native_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                         help="stage2 run dir with llm_diff.npy, clip_diff.npy, clip_neg_dir_native.npy")
    parser.add_argument("--llm_direction_path", required=True,
                         help="stage1 negation_directions_all_layers.npy")
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    llm_diff = np.load(os.path.join(args.run_dir, "llm_diff.npy"))
    clip_diff = np.load(os.path.join(args.run_dir, "clip_diff.npy"))
    clip_native_dir = np.load(os.path.join(args.run_dir, "clip_neg_dir_native.npy"))
    print(f"llm_diff: {llm_diff.shape}, clip_diff: {clip_diff.shape}")

    llm_dir = np.load(args.llm_direction_path, allow_pickle=True).item()[args.layer]
    llm_dir = llm_dir.flatten().astype(np.float32)
    llm_dir = llm_dir / np.linalg.norm(llm_dir)

    print("\nOrthogonal Procrustes (baseline)...")
    W_orth = orthogonal_procrustes_map(llm_diff, clip_diff)
    sim_orth = transfer_similarity(llm_dir, W_orth, clip_native_dir)
    print(f"  transfer_sim = {sim_orth:.4f}")

    print("\nRidge regression (regularized, non-orthogonal)...")
    alphas = np.logspace(1, 6, 12)
    W_ridge = ridge_map(llm_diff, clip_diff, alphas)
    sim_ridge = transfer_similarity(llm_dir, W_ridge, clip_native_dir)
    print(f"  transfer_sim = {sim_ridge:.4f}")

    print("\nBlending orthogonal and ridge maps (partial relaxation)...")
    blend_results = {}
    for beta in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        W_blend = (1 - beta) * W_orth + beta * W_ridge
        sim_blend = transfer_similarity(llm_dir, W_blend, clip_native_dir)
        blend_results[beta] = sim_blend
        print(f"  beta={beta:.2f} (0=orthogonal, 1=ridge)  transfer_sim={sim_blend:.4f}")

    best_beta = max(blend_results, key=blend_results.get)
    W_best = (1 - best_beta) * W_orth + best_beta * W_ridge

    np.save(os.path.join(args.run_dir, "W_ridge.npy"), W_ridge)
    np.save(os.path.join(args.run_dir, "W_blend_best.npy"), W_best)

    print("\n" + "=" * 50)
    print(f"orthogonal Procrustes : {sim_orth:.4f}")
    print(f"ridge regression      : {sim_ridge:.4f}")
    print(f"best blend (beta={best_beta}): {blend_results[best_beta]:.4f}")
    print("=" * 50)
    print("\nSaved W_ridge.npy and W_blend_best.npy to run_dir.")
    print("Next: re-run the correction sweep with these W's transferred_dir "
          "to see if the accuracy gap vs native closes.")


if __name__ == "__main__":
    main()