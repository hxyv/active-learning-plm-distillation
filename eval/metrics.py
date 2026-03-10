"""Metric helpers for teacher/student distillation evaluation."""

from __future__ import annotations

from typing import Dict

import torch

from train.losses import soft_target_cross_entropy, teacher_kl_divergence


def compute_teacher_metrics(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    dssp_idx: torch.Tensor | None = None,
) -> Dict[str, float]:
    with torch.no_grad():
        ce = soft_target_cross_entropy(logits, teacher_probs, reduction="mean").item()
        kl = teacher_kl_divergence(logits, teacher_probs, reduction="batchmean").item()

        student_probs = torch.softmax(logits, dim=-1)
        student_argmax = logits.argmax(dim=-1)
        teacher_argmax = teacher_probs.argmax(dim=-1)
        acc_teacher = (student_argmax == teacher_argmax).float().mean().item()

        # Average correct-motif probability: mean student softmax probability
        # assigned to the teacher's top-1 SS8 class.
        correct_motif_prob = student_probs.gather(
            dim=-1, index=teacher_argmax.unsqueeze(-1)
        ).squeeze(-1).mean().item()

        out = {
            "teacher_ce": ce,
            "teacher_kl": kl,
            "teacher_top1_acc": acc_teacher,
            "correct_motif_prob": correct_motif_prob,
        }

        if dssp_idx is not None:
            valid = dssp_idx >= 0
            if valid.any():
                acc_dssp = (student_argmax[valid] == dssp_idx[valid]).float().mean().item()
                out["dssp_acc"] = acc_dssp
                out["dssp_count"] = float(valid.sum().item())
            else:
                out["dssp_acc"] = float("nan")
                out["dssp_count"] = 0.0

        return out
