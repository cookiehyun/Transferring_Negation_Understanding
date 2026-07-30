import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from repe import repe_pipeline_registry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import load_stage1_config, save_resolved_config
from src.data.ccneg_loader import load_ccneg_pairs

repe_pipeline_registry()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_stage1_config(args.config)
    set_seed(cfg.seed)

    run_id = f"{cfg.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vec_dir = os.path.join(cfg.output.vectors_dir, run_id)
    fig_dir = os.path.join(cfg.output.figures_dir, run_id)
    os.makedirs(vec_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    save_resolved_config(cfg, os.path.join(vec_dir, "config_used.yaml"))
    print(f"Run ID: {run_id}")

    train_pairs, test_pairs = load_ccneg_pairs(
        pairs_path=cfg.data.pairs_path,
        n_samples=cfg.data.n_samples,
        train_test_split=cfg.data.train_test_split,
        split_seed=cfg.data.split_seed,
    )
    print(f"Data: n_samples={cfg.data.n_samples} -> train={len(train_pairs)}, test={len(test_pairs)}")
    print(f"Example pair: {train_pairs[0]}")

    print(f"\nLoading model: {cfg.model.name_or_path}")
    dtype = getattr(torch, cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        dtype=dtype,
        device_map=cfg.model.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers}")

    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    train_data, train_labels = [], []
    for pos, neg in train_pairs:
        train_data.extend([pos, neg])
        train_labels.append([True, False])

    print("\nExtracting negation direction vectors...")
    negation_reader = rep_reading_pipeline.get_directions(
        train_data,
        rep_token=cfg.repe.rep_token,
        hidden_layers=hidden_layers,
        n_difference=cfg.repe.n_difference,
        train_labels=train_labels,
        direction_method=cfg.repe.direction_method,
        batch_size=cfg.repe.batch_size,
    )

    test_data, test_labels = [], []
    for pos, neg in test_pairs:
        test_data.extend([pos, neg])
        test_labels.append([True, False])

    print("\nEvaluating on test set...")
    H_tests = rep_reading_pipeline(
        test_data,
        rep_token=cfg.repe.rep_token,
        hidden_layers=hidden_layers,
        rep_reader=negation_reader,
        batch_size=cfg.repe.batch_size,
    )

    results = {}
    for layer in hidden_layers:
        H = [h[layer] for h in H_tests]
        H = [H[i:i + 2] for i in range(0, len(H), 2)]
        sign = negation_reader.direction_signs[layer]
        eval_fn = min if sign == -1 else max
        results[layer] = np.mean([eval_fn(h) == h[0] for h in H])

    print("\n=== Per-layer Accuracy (Test) ===")
    for layer in sorted(results.keys(), reverse=True):
        bar = "#" * int(results[layer] * 20)
        mark = " *" if results[layer] >= cfg.output.accuracy_threshold else ""
        print(f"  Layer {layer:4d}: {results[layer]:.4f}  {bar}{mark}")

    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer}  (accuracy: {results[best_layer]:.4f})")

    layers = sorted(results.keys(), reverse=True)
    accuracies = [results[l] for l in layers]

    plt.figure(figsize=(14, 5))
    plt.plot(layers, accuracies, "bo-", linewidth=2, markersize=5)
    plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Chance (0.5)")
    plt.axhline(y=1.0, color="g", linestyle="--", alpha=0.7, label="Perfect (1.0)")
    plt.axvline(x=best_layer, color="orange", linestyle="--", alpha=0.8, label=f"Best layer ({best_layer})")
    plt.xlabel("Layer (negative index)")
    plt.ylabel("Classification Accuracy")
    plt.title(f"Negation Direction Accuracy per Layer\n"
              f"{cfg.model.name_or_path.split('/')[-1]}, CC-Neg n={cfg.data.n_samples}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "stage1_accuracy.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    # RepE's own sign convention (via train_labels=[True, False], index 0 = positive)
    # makes directions*signs point toward the POSITIVE/affirmative label. We negate
    # here so the saved vector points toward negation, matching how Stage 2 defines
    # clip_native_dir = (clip_neg - clip_pos).mean(axis=0).
    best_vec = -1 * negation_reader.directions[best_layer] * negation_reader.direction_signs[best_layer]
    np.save(os.path.join(vec_dir, f"negation_direction_layer{best_layer}.npy"), best_vec)

    all_dirs = {
        layer: -1 * negation_reader.directions[layer] * negation_reader.direction_signs[layer]
        for layer in hidden_layers
    }
    np.save(os.path.join(vec_dir, "negation_directions_all_layers.npy"), all_dirs)
    all_signs = {layer: negation_reader.direction_signs[layer] for layer in hidden_layers}
    np.save(os.path.join(vec_dir, "direction_signs.npy"), all_signs)
    np.save(os.path.join(vec_dir, "per_layer_accuracy.npy"), results)
    print(f"Saved vectors to: {vec_dir}")

    print("\n" + "=" * 50)
    print(f"Run ID      : {run_id}")
    print(f"Model       : {cfg.model.name_or_path}")
    print(f"Train pairs : {len(train_pairs)}")
    print(f"Test pairs  : {len(test_pairs)}")
    print(f"Best layer  : {best_layer}")
    print(f"Accuracy    : {results[best_layer]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()