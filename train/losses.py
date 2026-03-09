"""Loss functions for hard/soft supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy with soft probability targets."""
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def teacher_kl_divergence(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(target || student) with optional temperature scaling."""
    t = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(logits / t, dim=-1)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    loss = F.kl_div(student_log_probs, target_probs, reduction=reduction)
    return loss * (t * t)


def hard_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
