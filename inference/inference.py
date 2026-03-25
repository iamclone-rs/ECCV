"""
Inference script for Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR).

Input:  A single sketch image.
Output: Top-5 most similar photo images from the unseen-class gallery,
        with their file paths and cosine similarity scores.

Usage:
    python -m inference.inference \
        --sketch_path <path_to_sketch.png> \
        --ckpt_path <path_to_checkpoint.ckpt> \
        --root <dataset_root> \
        --dataset sketchy_2 \
        --output_dir ./inference_output
"""

import os
import sys
import glob
import json
import shutil
import argparse

import torch
import numpy as np
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── make project root importable ──────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ZS_SBIR
from src.utils import get_all_categories
from src.data_config import UNSEEN_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transforms (same as training / validation) ───────────────────────
def normal_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Load model from checkpoint ───────────────────────────────────────
def load_model(args, classnames):
    """Build ZS_SBIR and load weights from *args.ckpt_path*."""
    model = ZS_SBIR(args=args, classname=classnames)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # These buffers are recomputed from current classnames, so skip them.
    skip_keys = [
        "model.prompt_learner_photo.token_prefix",
        "model.prompt_learner_photo.token_suffix",
        "model.prompt_learner_sketch.token_prefix",
        "model.prompt_learner_sketch.token_suffix",
    ]
    for k in skip_keys:
        sd.pop(k, None)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[INFO] Missing keys (expected): {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    return model


# ── Build photo gallery (unseen classes only) ─────────────────────────
@torch.no_grad()
def build_gallery(model, args, classnames, batch_size=256):
    """Extract features for every photo in the unseen-class gallery."""
    tfm = normal_transform()
    unseen_classes = UNSEEN_CLASSES[args.dataset]

    photo_paths = []
    for cls in unseen_classes:
        cls_dir = os.path.join(args.root, "photo", cls)
        if os.path.isdir(cls_dir):
            photo_paths.extend(sorted(glob.glob(os.path.join(cls_dir, "*"))))

    if not photo_paths:
        raise FileNotFoundError(
            f"No photos found under {args.root}/photo/ for unseen classes: {unseen_classes}"
        )

    print(f"[Gallery] {len(photo_paths)} photos from {len(unseen_classes)} unseen classes")

    all_features = []
    for start in range(0, len(photo_paths), batch_size):
        batch_paths = photo_paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            img = ImageOps.pad(
                Image.open(p).convert("RGB"),
                size=(args.max_size, args.max_size),
            )
            tensors.append(tfm(img))

        batch_tensor = torch.stack(tensors).to(device)
        feats = model.model.extract_feature(
            batch_tensor, classname=classnames, type="photo"
        )
        all_features.append(feats.cpu())

    gallery_features = torch.cat(all_features, dim=0)  # (N, 512)
    return gallery_features, photo_paths


# ── Extract sketch feature ────────────────────────────────────────────
@torch.no_grad()
def extract_sketch_feature(model, sketch_path, args, classnames):
    """Load and encode a single sketch image."""
    tfm = normal_transform()
    img = ImageOps.pad(
        Image.open(sketch_path).convert("RGB"),
        size=(args.max_size, args.max_size),
    )
    tensor = tfm(img).unsqueeze(0).to(device)
    feat = model.model.extract_feature(
        tensor, classname=classnames, type="sketch"
    )
    return feat.cpu()  # (1, 512)


# ── Retrieve top-k ────────────────────────────────────────────────────
def retrieve_topk(sketch_feat, gallery_feats, photo_paths, k=5):
    """Return top-k (path, score) pairs ranked by cosine similarity."""
    sim = F.cosine_similarity(sketch_feat, gallery_feats, dim=-1)  # (N,)
    topk_vals, topk_idxs = sim.topk(k, largest=True, sorted=True)

    results = []
    for score, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        results.append({
            "path": photo_paths[idx],
            "score": round(score, 6),
        })
    return results


# ── Save results ──────────────────────────────────────────────────────
def save_results(results, sketch_path, output_dir):
    """Copy retrieved photos + query sketch into *output_dir* and write JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Copy query sketch
    dst_sketch = os.path.join(output_dir, "query_" + os.path.basename(sketch_path))
    shutil.copy2(sketch_path, dst_sketch)

    # Copy top-k photos
    for rank, item in enumerate(results, start=1):
        ext = os.path.splitext(item["path"])[1]
        dst = os.path.join(output_dir, f"top{rank}_{os.path.basename(item['path'])}")
        shutil.copy2(item["path"], dst)

    # Write JSON
    output_json = {
        "query_sketch": sketch_path,
        "top_k": results,
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] Results written to {json_path}")


# ── Visualize results ─────────────────────────────────────────────────
def visualize_results(results, sketch_path, output_dir):
    """
    Draw a horizontal strip: [Query Sketch] | [Top-1] [Top-2] ... [Top-K]
    Each retrieved photo is annotated with its score and a ✓ / ✗
    depending on whether its category matches the sketch category.
    """
    # Infer query category from sketch path:  .../sketch/<category>/filename
    sketch_category = os.path.basename(os.path.dirname(sketch_path))

    k = len(results)
    n_cols = 1 + k  # query + retrieved
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    # ── Query sketch ──────────────────────────────────────────────
    query_img = Image.open(sketch_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\n({sketch_category})", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    # Blue border for query
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#2196F3")
        spine.set_linewidth(3)

    # ── Retrieved photos ──────────────────────────────────────────
    for i, item in enumerate(results):
        ax = axes[i + 1]
        photo_img = Image.open(item["path"]).convert("RGB")
        photo_category = os.path.basename(os.path.dirname(item["path"]))
        is_correct = (photo_category == sketch_category)

        ax.imshow(photo_img)
        score_str = f"{item['score']:.4f}"

        # Title: rank + score
        ax.set_title(f"Top-{i+1}\n{score_str}", fontsize=10)
        ax.axis("off")

        # ✓ / ✗ mark
        mark = "✓" if is_correct else "✗"
        color = "#4CAF50" if is_correct else "#F44336"
        ax.text(
            0.5, -0.08, mark,
            transform=ax.transAxes,
            fontsize=22, fontweight="bold",
            color=color, ha="center", va="top",
        )

        # Border color
        border_color = "#4CAF50" if is_correct else "#F44336"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "retrieval_vis.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[Saved] Visualization saved to {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ZS-SBIR Inference: sketch → top-5 photos")

    # Required
    parser.add_argument("--sketch_path", type=str, required=True, help="Path to input sketch image")
    parser.add_argument("--ckpt_path",   type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--root",        type=str, required=True, help="Dataset root (contains sketch/ and photo/)")
    parser.add_argument("--dataset",     type=str, required=True, help="Dataset name: sketchy_1, sketchy_2, tuberlin, quickdraw")

    # Optional (defaults match training)
    parser.add_argument("--output_dir",    type=str, default="./inference_output")
    parser.add_argument("--backbone",      type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx",         type=int, default=2)
    parser.add_argument("--img_ctx",       type=int, default=2)
    parser.add_argument("--max_size",      type=int, default=224)
    parser.add_argument("--prompt_depth",  type=int, default=12)
    parser.add_argument("--prec",          type=str, default="fp16")
    parser.add_argument("--top_k",         type=int, default=5,   help="Number of photos to retrieve")

    args = parser.parse_args()

    # ── Classnames (seen categories, used by prompt learner) ──────
    classnames = get_all_categories(args, mode="train")
    print(f"[Config] dataset={args.dataset}, backbone={args.backbone}, "
          f"classnames={len(classnames)} seen categories")

    # ── Load model ────────────────────────────────────────────────
    print("[1/3] Loading model …")
    model = load_model(args, classnames)

    # ── Build gallery ─────────────────────────────────────────────
    print("[2/3] Building photo gallery (unseen classes) …")
    gallery_feats, photo_paths = build_gallery(model, args, classnames)

    # ── Retrieve ──────────────────────────────────────────────────
    print("[3/3] Extracting sketch feature & retrieving …")
    sketch_feat = extract_sketch_feature(model, args.sketch_path, args, classnames)
    results = retrieve_topk(sketch_feat, gallery_feats, photo_paths, k=args.top_k)

    # ── Print & save ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Query sketch: {args.sketch_path}")
    print(f"{'='*60}")
    for rank, item in enumerate(results, start=1):
        print(f"  Top-{rank}  score={item['score']:.6f}  path={item['path']}")
    print(f"{'='*60}")

    save_results(results, args.sketch_path, args.output_dir)
    visualize_results(results, args.sketch_path, args.output_dir)


if __name__ == "__main__":
    main()
