import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.tuberlin import UNSEEN_CLASSES, TEMPLATES
from data.transforms import build_train_transform, build_test_transform, CLIP_MEAN, CLIP_STD
from data.sbir_dataset import SBIRTripletTrain, SBIREvalIndex, build_seen_unseen
from models.splip_sbir import SpLIP_SBIR
from eval.sbir_eval import eval_zs, eval_gzs

def list_all_classes(root):
    ph = os.path.join(root, "photo")
    return sorted([d for d in os.listdir(ph) if os.path.isdir(os.path.join(ph, d))])


@torch.no_grad()
def visualize_random_sketch_predictions(
    model,
    q_ds,
    templates,
    out_path,
    n=20,
):
    """Save a grid (4x5) of random sketches with gt/pred class names."""
    from collections import Counter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Viz] Skip (matplotlib unavailable): {e}")
        return

    device = next(model.parameters()).device
    class_names = list(getattr(q_ds, "classes", []))
    if len(class_names) == 0:
        print("[Viz] Skip (q_ds.classes missing/empty)")
        return

    # Build text embeddings for unseen classes (same convention as training: use photo template)
    prompts = [templates["photo"].format(c) for c in class_names]
    tok = model.tokenizer(prompts).to(device)
    class_embeds = model.encode_text_plain(tok)  # (C, D), normalized

    n = min(int(n), len(q_ds))
    if n <= 0:
        print("[Viz] Skip (dataset empty)")
        return

    idxs = random.sample(range(len(q_ds)), k=n) if len(q_ds) >= n else list(range(len(q_ds)))
    xs = []
    ys = []
    for i in idxs:
        x, y = q_ds[i]
        xs.append(x)
        ys.append(int(y))
    x = torch.stack(xs, 0).to(device)

    z = model.encode_image_for_retrieval(x, model._seen_names, model._templates)  # (N, D)
    logits = z @ class_embeds.t()  # (N, C)
    pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()

    # Diagnostics: NaN/Inf and distribution
    if (not torch.isfinite(z).all()) or (not torch.isfinite(class_embeds).all()) or (not torch.isfinite(logits).all()):
        print(
            "[Viz][Warn] Non-finite values detected: "
            f"z_finite={bool(torch.isfinite(z).all())}, "
            f"text_finite={bool(torch.isfinite(class_embeds).all())}, "
            f"logits_finite={bool(torch.isfinite(logits).all())}"
        )
    gt_counts = Counter(ys)
    pred_counts = Counter(pred)
    top_pred = pred_counts.most_common(5)
    top_pred_named = [(class_names[i] if 0 <= i < len(class_names) else str(i), c) for i, c in top_pred]
    unique_gt = len(gt_counts)
    unique_pred = len(pred_counts)
    print(f"[Viz] sample_unique_gt={unique_gt} | sample_unique_pred={unique_pred} | top_pred={top_pred_named}")

    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype, device=device).view(1, 3, 1, 1)
    x_vis = (x * std + mean).clamp(0, 1).detach().cpu()

    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()
    for j in range(rows * cols):
        ax = axes[j]
        ax.axis("off")
        if j >= n:
            continue
        img = x_vis[j].permute(1, 2, 0).numpy()
        gt_name = class_names[ys[j]] if 0 <= ys[j] < len(class_names) else str(ys[j])
        pr_name = class_names[pred[j]] if 0 <= pred[j] < len(class_names) else str(pred[j])
        ok = (ys[j] == pred[j])
        ax.imshow(img)
        ax.set_title(f"gt: {gt_name}\npred: {pr_name}{' ✓' if ok else ''}", fontsize=10)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Viz] Saved {out_path}")

def main(
    root,
    epochs=60,
    batch_size=192,
    lr=1e-3,
    tau=0.07,
    alpha=1.0,
    image_size=224,
    num_workers=4,
    eval_every=1,
    device=None,
    viz_n=20,
    viz_dir="viz",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    all_classes = list_all_classes(root)
    split = build_seen_unseen(all_classes, UNSEEN_CLASSES)

    tfm_sk_tr = build_train_transform(image_size=image_size, is_sketch=True)
    tfm_ph_tr = build_train_transform(image_size=image_size, is_sketch=False)
    tfm_te = build_test_transform(image_size=image_size)

    train_ds = SBIRTripletTrain(root, split.seen, tfm_sk_tr, tfm_ph_tr)

    model = SpLIP_SBIR(m=4, n=2, device=device).to(device)
    model._seen_names = train_ds.valid_classes
    model._templates = TEMPLATES

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    q_unseen = SBIREvalIndex(root, split.unseen, "sketch", tfm_te)
    g_unseen = SBIREvalIndex(root, split.unseen, "photo", tfm_te)
    g_seen = SBIREvalIndex(root, split.seen, "photo", tfm_te)

    if eval_every != 1:
        print(f"[Warn] You requested eval_every={eval_every}, but this run will compute mAP@all and P@100 after *every* epoch.")
        eval_every = 1

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {ep}/{epochs}")
        for sk, pp, pn, y_pos, y_neg in pbar:
            sk = sk.to(device, non_blocking=True)
            pp = pp.to(device, non_blocking=True)
            pn = pn.to(device, non_blocking=True)
            y_pos = y_pos.to(device)
            y_neg = y_neg.to(device)

            opt.zero_grad(set_to_none=True)
            loss, logs = model.forward_train(
                sk, pp, pn, y_pos, y_neg,
                class_names_seen=train_ds.valid_classes,
                templates=TEMPLATES,
                tau=tau,
                alpha=alpha,
            )
            loss.backward()
            opt.step()

            pbar.set_postfix({
                "loss": float(loss.detach().cpu()),
                "trip": float(logs["l_triplet"].cpu()),
                "cls": float(logs["l_class"].cpu()),
                "mu": float(logs["mu_mean"].cpu()),
            })

        # Evaluate every epoch: mAP@all and P@100
        model.eval()
        zs = eval_zs(model, q_unseen, g_unseen, k=100, batch_size=64, num_workers=num_workers)
        gzs = eval_gzs(model, q_unseen, g_seen, g_unseen, k=100, batch_size=64, num_workers=num_workers)
        print(
            "[Eval] "
            f"ZS mAP@all={zs['mAP@all']:.4f}, P@100={zs['P@100']:.4f} | "
            f"GZS mAP@all={gzs['mAP@all']:.4f}, P@100={gzs['P@100']:.4f}"
        )

        # Visualize 20 random unseen sketches + predictions
        viz_path = os.path.join(viz_dir, f"epoch_{ep:03d}_unseen_sketch_preds.png")
        visualize_random_sketch_predictions(model, q_unseen, TEMPLATES, viz_path, n=viz_n)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--viz_n", type=int, default=20)
    ap.add_argument("--viz_dir", type=str, default="viz")
    args = ap.parse_args()
    main(**vars(args))
