import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.tuberlin import UNSEEN_CLASSES, TEMPLATES
from data.transforms import build_train_transform, build_test_transform
from data.sbir_dataset import SBIRTripletTrain, SBIREvalIndex, build_seen_unseen
from models.splip_sbir import SpLIP_SBIR
from eval.sbir_eval import eval_zs, eval_gzs

def list_all_classes(root):
    ph = os.path.join(root, "photo")
    return sorted([d for d in os.listdir(ph) if os.path.isdir(os.path.join(ph, d))])

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
    args = ap.parse_args()
    main(**vars(args))
