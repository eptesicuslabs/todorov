from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TodorovLoss(nn.Module):

    def __init__(
        self,
        residual_penalty_weight: float = 0.01,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.residual_penalty_weight = residual_penalty_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        aux: dict | None = None,
    ) -> tuple[Tensor, dict]:
        vocab_size = logits.shape[-1]
        logits_flat = logits[:, :-1].contiguous().view(-1, vocab_size)
        targets_flat = targets[:, 1:].contiguous().view(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=self.label_smoothing,
        )

        loss = ce_loss
        metrics = {"ce_loss": ce_loss.item()}

        if aux is not None and self.residual_penalty_weight > 0:
            state_norms = []
            for layer_aux in aux.get("layers", {}).values():
                if "state_norm" in layer_aux:
                    state_norms.append(layer_aux["state_norm"])
            if state_norms:
                mean_state_norm = sum(state_norms) / len(state_norms)
                residual_penalty = self.residual_penalty_weight * mean_state_norm
                loss = loss + residual_penalty
                metrics["residual_penalty"] = residual_penalty
                metrics["mean_state_norm"] = mean_state_norm

        metrics["total_loss"] = loss.item()
        return loss, metrics


class DistillationLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 2.0,
        forward_kl_weight: float = 0.2,
        reverse_kl_weight: float = 0.7,
        feature_weight: float = 0.1,
        use_pre_norm_alignment: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.forward_kl_weight = forward_kl_weight
        self.reverse_kl_weight = reverse_kl_weight
        self.feature_weight = feature_weight
        self.use_pre_norm_alignment = use_pre_norm_alignment

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Tensor | None = None,
        teacher_features: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        student_probs = F.softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        forward_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
        reverse_kl = F.kl_div(teacher_log_probs, student_probs, reduction="batchmean") * (T * T)

        loss = self.forward_kl_weight * forward_kl + self.reverse_kl_weight * reverse_kl
        metrics = {
            "forward_kl": forward_kl.item(),
            "reverse_kl": reverse_kl.item(),
        }

        if student_features is not None and teacher_features is not None:
            if self.use_pre_norm_alignment:
                feature_loss = F.mse_loss(student_features, teacher_features)
            else:
                s_norm = F.normalize(student_features, dim=-1)
                t_norm = F.normalize(teacher_features, dim=-1)
                feature_loss = F.mse_loss(s_norm, t_norm)
            loss = loss + self.feature_weight * feature_loss
            metrics["feature_loss"] = feature_loss.item()

        metrics["total_distillation_loss"] = loss.item()
        return loss, metrics
