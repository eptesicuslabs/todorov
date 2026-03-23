from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConvergenceStats:
    mean_residual: float = 0.0
    max_residual: float = 0.0
    trend: str = "unknown"
    is_stable: bool = False


class ConvergenceMonitor:

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.residuals: deque[float] = deque(maxlen=window_size)

    def update(self, residual: float) -> None:
        self.residuals.append(residual)

    def get_stats(self) -> ConvergenceStats:
        if len(self.residuals) < 2:
            return ConvergenceStats()

        values = list(self.residuals)
        mean_val = sum(values) / len(values)
        max_val = max(values)

        half = len(values) // 2
        first_half = sum(values[:half]) / max(half, 1)
        second_half = sum(values[half:]) / max(len(values) - half, 1)

        if second_half < first_half * 0.95:
            trend = "decreasing"
        elif second_half > first_half * 1.05:
            trend = "increasing"
        else:
            trend = "stable"

        is_stable = trend in ("decreasing", "stable") and max_val < mean_val * 3

        return ConvergenceStats(
            mean_residual=mean_val,
            max_residual=max_val,
            trend=trend,
            is_stable=is_stable,
        )

    def reset(self) -> None:
        self.residuals.clear()
