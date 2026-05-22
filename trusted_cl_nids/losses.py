from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.new_tensor(0.0)

    labels = labels.view(-1, 1)
    features = F.normalize(features, p=2, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    logits_mask = torch.ones_like(similarity) - torch.eye(
        similarity.shape[0], device=similarity.device
    )
    positive_mask = (labels == labels.T).float() * logits_mask
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)

    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    return -mean_log_prob[valid].mean()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    labels = labels.float()
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if weights is not None:
        loss = loss * weights.float()
    return loss.mean()
