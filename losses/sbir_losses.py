import torch

def average_precision(scores, targets):
    order = torch.argsort(scores, descending=True)
    t = targets[order].float()
    if t.sum() == 0:
        return torch.tensor(0.0, device=scores.device)
    cumsum = torch.cumsum(t, dim=0)
    ranks = torch.arange(1, t.numel() + 1, device=t.device).float()
    prec = cumsum / ranks
    ap = (prec * t).sum() / t.sum()
    return ap

def mean_ap(sim_matrix, query_labels, gallery_labels):
    aps = []
    for i in range(sim_matrix.size(0)):
        scores = sim_matrix[i]
        targets = (gallery_labels == query_labels[i])
        aps.append(average_precision(scores, targets))
    return torch.stack(aps).mean()

def precision_at_k(sim_matrix, query_labels, gallery_labels, k=200):
    k = min(k, sim_matrix.size(1))
    topk = torch.topk(sim_matrix, k=k, dim=1).indices
    hits = []
    for i in range(sim_matrix.size(0)):
        preds = gallery_labels[topk[i]]
        hits.append((preds == query_labels[i]).float().mean())
    return torch.stack(hits).mean()
