"""Entity-Binding Distillation (EBD) loss, Eq. 5 of the paper.

Pseudo ground truth: per-frame k-means (K = number of slots) on DINOv2 patch
features, 2x2 average-pooled to the 8x8 visual-token grid and l2-normalized.
Each slot attention map Â is matched to a cluster map with the Hungarian
algorithm (patch-averaged BCE as the cost) and supervised with BCE.
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def kmeans_cluster_maps(features: torch.Tensor, k: int, n_iters: int = 10) -> torch.Tensor:
    """Cosine k-means per frame.

    Args:
        features: [BT, N, D] l2-normalized patch features.
    Returns:
        [BT, K, N] one-hot cluster maps.
    """
    BT, N, D = features.shape
    feats = F.normalize(features, dim=-1)

    init = torch.stack([torch.randperm(N, device=feats.device)[:k] for _ in range(BT)])
    centroids = torch.gather(feats, 1, init.unsqueeze(-1).expand(-1, -1, D))
    for _ in range(n_iters):
        assignments = torch.bmm(feats, centroids.transpose(1, 2)).argmax(dim=-1)
        one_hot = F.one_hot(assignments, k).to(feats.dtype)
        counts = one_hot.sum(dim=1).clamp(min=1)
        centroids = torch.bmm(one_hot.transpose(1, 2), feats) / counts.unsqueeze(-1)
        centroids = F.normalize(centroids, dim=-1)

    assignments = torch.bmm(feats, centroids.transpose(1, 2)).argmax(dim=-1)
    return F.one_hot(assignments, k).to(feats.dtype).transpose(1, 2)


def pool_dino_features(dino_features: torch.Tensor) -> torch.Tensor:
    """[BT, H*W, D] DINOv2 patches -> [BT, (H/2)*(W/2), D], l2-normalized."""
    BT, P, D = dino_features.shape
    hw = int(P ** 0.5)
    x = dino_features.view(BT, hw, hw, D).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    x = x.permute(0, 2, 3, 1).reshape(BT, (hw // 2) ** 2, D)
    return F.normalize(x, p=2, dim=-1)


def hungarian_bce(attn: torch.Tensor, cluster_maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        attn:         [BT, K, N] slot attention maps Â.
        cluster_maps: [BT, K, N] binary cluster maps.
    """
    BT, K, N = attn.shape
    attn = attn.clamp(eps, 1 - eps)

    # cost[b, i, j] = BCE(slot i, cluster j), averaged over patches.
    cost = F.binary_cross_entropy(
        attn.unsqueeze(2).expand(BT, K, K, N),
        cluster_maps.unsqueeze(1).expand(BT, K, K, N),
        reduction="none",
    ).mean(dim=-1)
    cost = cost.detach().float().cpu().numpy()
    # linear_sum_assignment returns rows in order 0..K-1, so only the columns matter.
    cols = torch.from_numpy(np.stack([linear_sum_assignment(c)[1] for c in cost])).to(attn.device)
    matched = torch.gather(cluster_maps, 1, cols.unsqueeze(-1).expand(-1, -1, N))

    per_frame = F.binary_cross_entropy(attn, matched, reduction="none").mean(dim=(1, 2))
    return per_frame.float().mean()


def ebd_loss(attn_maps: List[torch.Tensor], dino_features: torch.Tensor) -> torch.Tensor:
    """
    Args:
        attn_maps:     one [BT, K, N] map per EB Adapter layer.
        dino_features: [BT, H*W, D] DINOv2 patch features of the same frames.
    Returns:
        EBD loss averaged over layers (the cluster maps are shared across layers).
    """
    K = attn_maps[0].shape[1]
    cluster_maps = kmeans_cluster_maps(pool_dino_features(dino_features), K)
    return sum(hungarian_bce(a, cluster_maps) for a in attn_maps) / len(attn_maps)
