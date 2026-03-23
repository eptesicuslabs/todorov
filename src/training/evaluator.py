from __future__ import annotations

import math
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


class Evaluator:

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 512,
        stride: int = 256,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        all_losses: list[float] = []

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
            else:
                input_ids = batch.to(self.device)

            batch_loss, n_tokens = self._sliding_window_loss(input_ids)
            total_loss += batch_loss
            total_tokens += n_tokens
            all_losses.append(batch_loss / max(n_tokens, 1))

        mean_loss = total_loss / max(total_tokens, 1)
        bpb = mean_loss / math.log(2)

        ci_lower, ci_upper = self._bootstrap_ci(all_losses)

        self.model.train()

        return {
            "mean_loss": mean_loss,
            "bpb": bpb,
            "total_tokens": total_tokens,
            "num_batches": len(all_losses),
            "bpb_ci_lower": ci_lower / math.log(2),
            "bpb_ci_upper": ci_upper / math.log(2),
            "timestamp": datetime.now().isoformat(),
        }

    def _sliding_window_loss(self, input_ids: Tensor) -> tuple[float, int]:
        seq_len = input_ids.shape[1]
        total_loss = 0.0
        total_tokens = 0

        if seq_len <= self.window_size:
            logits, _, _ = self.model(input_ids)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, vocab_size),
                input_ids[:, 1:].contiguous().view(-1),
                reduction="sum",
            )
            return loss.item(), input_ids[:, 1:].numel()

        for start in range(0, seq_len - 1, self.stride):
            end = min(start + self.window_size, seq_len)
            window = input_ids[:, start:end]

            logits, _, _ = self.model(window)
            vocab_size = logits.shape[-1]

            target_start = max(0, self.stride if start > 0 else 0)
            target_logits = logits[:, target_start:-1]
            target_ids = window[:, target_start + 1:]

            if target_ids.numel() == 0:
                continue

            loss = F.cross_entropy(
                target_logits.contiguous().view(-1, vocab_size),
                target_ids.contiguous().view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()

        return total_loss, total_tokens

    def _bootstrap_ci(
        self,
        losses: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        if len(losses) < 2:
            mean = sum(losses) / max(len(losses), 1)
            return mean, mean

        import numpy as np
        arr = np.array(losses)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(sample)))

        alpha = (1 - confidence) / 2
        lower = float(np.percentile(boot_means, alpha * 100))
        upper = float(np.percentile(boot_means, (1 - alpha) * 100))
        return lower, upper
