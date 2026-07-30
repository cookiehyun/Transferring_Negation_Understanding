import os
import argparse
import json
import zipfile
import torch
from huggingface_hub import hf_hub_download

REPO_ID = "jaisidhsingh/ccneg"
IMAGES_FILENAME = "ccneg_images.zip"
ANNOTATION_FILENAME = "ccneg_preprocessed.pt"


def download_and_extract_images(out_dir, cache_dir):
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=IMAGES_FILENAME,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"Downloaded: {zip_path}")

    extract_dir = os.path.join(out_dir, "images")
    marker = os.path.join(extract_dir, "ccneg_images", "cc3m_subset_images_extracted_final")
    if os.path.isdir(marker) and len(os.listdir(marker)) > 0:
        print(f"Already extracted, skipping: {marker}")
        return extract_dir

    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def build_manifest(extract_dir, cache_dir):
    ann_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ANNOTATION_FILENAME,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    data = torch.load(ann_path, map_location="cpu")

    annotations = data["annotations"]
    image_paths = data["image_paths"]

    manifest = []
    missing = 0
    for ann, orig_path in zip(annotations, image_paths):
        pos = ann.get("caption")
        neg = ann.get("sop_data", {}).get("negative-prompt")
        if not pos or not neg:
            continue

        fname = os.path.basename(orig_path)
        local_path = os.path.join(extract_dir, "ccneg_images", "cc3m_subset_images_extracted_final", fname)
        if not os.path.exists(local_path):
            missing += 1
            continue

        manifest.append({
            "image_path": local_path,
            "positive_caption": pos.strip(),
            "negative_caption": neg.strip(),
        })

    print(f"Manifest entries: {len(manifest)}, missing images: {missing}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/ccneg")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    extract_dir = download_and_extract_images(args.out_dir, args.cache_dir)
    manifest = build_manifest(extract_dir, args.cache_dir)

    out_path = os.path.join(args.out_dir, "ccneg_image_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved manifest to {out_path}")


if __name__ == "__main__":
    main()