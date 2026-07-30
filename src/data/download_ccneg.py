import os
import argparse
import json
import torch
from huggingface_hub import hf_hub_download

REPO_ID = "jaisidhsingh/ccneg"
ANNOTATION_FILENAME = "ccneg_preprocessed.pt"


def download_annotations(cache_dir):
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ANNOTATION_FILENAME,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"Downloaded to: {path}")
    return path


def extract_pairs(data):
    if not isinstance(data, dict) or "annotations" not in data:
        raise ValueError(f"Unexpected structure: {type(data)}")

    pairs = []
    for ann in data["annotations"]:
        pos = ann.get("caption")
        neg = ann.get("sop_data", {}).get("negative-prompt")
        if pos and neg:
            pairs.append([pos.strip(), neg.strip()])

    print(f"Extracted {len(pairs)} pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/ccneg")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ann_path = download_annotations(args.cache_dir)
    data = torch.load(ann_path, map_location="cpu")
    pairs = extract_pairs(data)

    out_path = os.path.join(args.out_dir, "ccneg_pairs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "CC-Neg (Singh et al., WACV 2025, arXiv:2403.20312)",
                "n_total": len(pairs),
                "pairs": pairs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()