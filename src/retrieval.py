"""Exact retrieval metrics with bounded GPU memory."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def retrieval_metrics(queries, gallery, query_labels, gallery_labels,
                      map_k=None, precision_k=200, query_batch_size=32, device='cpu'):
    if not len(queries) or not len(gallery):
        raise ValueError('Retrieval requires nonempty queries and gallery.')
    gallery = F.normalize(gallery.to(device=device, dtype=torch.float32), dim=-1)
    gallery_labels = torch.as_tensor(gallery_labels, device=device)
    query_labels = torch.as_tensor(query_labels, device=device)
    p_k = min(precision_k, len(gallery))
    a_k = min(map_k, len(gallery)) if map_k else len(gallery)
    positions = torch.arange(1, len(gallery) + 1, device=device, dtype=torch.float32)
    total_ap = torch.zeros((), device=device)
    total_precision = torch.zeros((), device=device)
    for start in range(0, len(queries), query_batch_size):
        query = F.normalize(queries[start:start + query_batch_size].to(device=device, dtype=torch.float32), dim=-1)
        scores = query @ gallery.T
        ranking = scores.argsort(dim=-1, descending=True, stable=True)
        relevant = gallery_labels[ranking] == query_labels[start:start + len(query), None]
        hits = relevant.float()
        cumulative_precision = hits.cumsum(dim=-1) / positions
        # Same AP@k convention as torchmetrics: divide by relevant hits in top k.
        ap = (cumulative_precision[:, :a_k] * hits[:, :a_k]).sum(dim=-1)
        ap /= hits[:, :a_k].sum(dim=-1).clamp_min(1)
        total_ap += ap.sum()
        total_precision += hits[:, :p_k].mean(dim=-1).sum()
    return total_ap / len(queries), total_precision / len(queries)
