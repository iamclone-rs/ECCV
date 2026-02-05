import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.sbir_losses import mean_ap, precision_at_k

@torch.no_grad()
def encode_all(model, ds, batch_size=64, num_workers=4):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    labels = []
    for x, y in tqdm(loader, leave=False):
        x = x.to(next(model.parameters()).device)
        f = model.encode_image_for_retrieval(x, model._seen_names, model._templates)
        feats.append(f.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, 0), torch.cat(labels, 0)

@torch.no_grad()
def eval_zs(model, q_ds, g_ds, k=200, batch_size=64, num_workers=4):
    qf, qy = encode_all(model, q_ds, batch_size, num_workers)
    gf, gy = encode_all(model, g_ds, batch_size, num_workers)
    sim = qf @ gf.t()
    m = mean_ap(sim, qy, gy)
    p = precision_at_k(sim, qy, gy, k=k)
    return {"mAP@all": m.item(), f"P@{k}": p.item()}

@torch.no_grad()
def eval_gzs(model, q_ds_unseen, g_ds_seen, g_ds_unseen, k=200, batch_size=64, num_workers=4):
    qf, qy = encode_all(model, q_ds_unseen, batch_size, num_workers)
    gf_s, gy_s = encode_all(model, g_ds_seen, batch_size, num_workers)
    gf_u, gy_u = encode_all(model, g_ds_unseen, batch_size, num_workers)
    gf = torch.cat([gf_s, gf_u], 0)
    gy = torch.cat([gy_s, gy_u], 0)
    sim = qf @ gf.t()
    m = mean_ap(sim, qy, gy)
    p = precision_at_k(sim, qy, gy, k=k)
    return {"mAP@all": m.item(), f"P@{k}": p.item()}
