import math
import torch
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class SpikeHealthMetrics:
    dead_neuron_pct: float
    saturated_neuron_pct: float
    firing_rate_mean: float
    firing_rate_std: float
    health_pass: bool = True
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dead_neuron_pct": self.dead_neuron_pct,
            "saturated_neuron_pct": self.saturated_neuron_pct,
            "firing_rate_mean": self.firing_rate_mean,
            "firing_rate_std": self.firing_rate_std,
            "health_pass": self.health_pass,
            "alerts": self.alerts,
        }


@dataclass
class SpikingBrainValidation:
    health: SpikeHealthMetrics
    mutual_information: dict[str, float]
    cka: dict[str, float]
    overall_pass: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "health": self.health.to_dict(),
            "mutual_information": self.mutual_information,
            "cka": self.cka,
            "overall_pass": self.overall_pass,
            "summary": self.summary,
        }


class MutualInformationEstimator:

    def __init__(self, n_bins: int = 32, n_dims: int = 8) -> None:
        self.n_bins = n_bins
        self.n_dims = n_dims

    def binning_mi(
        self,
        spikes: torch.Tensor,
        reference: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            s_flat = spikes.reshape(-1, spikes.shape[-1]).float().cpu().numpy()
            r_flat = reference.reshape(-1, reference.shape[-1]).float().cpu().numpy()

            n = min(s_flat.shape[0], r_flat.shape[0], 10000)
            if n == 0:
                return 0.0

            s_flat = s_flat[:n]
            r_flat = r_flat[:n]
            n_dims = min(self.n_dims, s_flat.shape[1], r_flat.shape[1])
            if n_dims == 0:
                return 0.0

            mi_values: list[float] = []
            for dim in range(n_dims):
                s_col = s_flat[:, dim]
                r_col = r_flat[:, dim]

                r_min = float(r_col.min())
                r_max = float(r_col.max())
                if abs(r_max - r_min) < 1e-12:
                    continue

                r_bins = np.digitize(
                    r_col,
                    np.linspace(r_min, r_max, self.n_bins + 1)[1:-1],
                )

                s_disc = np.ones_like(s_col, dtype=np.int32)
                s_disc[s_col > 1e-6] = 2
                s_disc[s_col < -1e-6] = 0

                joint = np.zeros((3, self.n_bins), dtype=np.float64)
                for idx in range(n):
                    r_bin = max(0, min(int(r_bins[idx]), self.n_bins - 1))
                    joint[s_disc[idx], r_bin] += 1.0

                joint = joint / (joint.sum() + 1e-12)
                p_s = joint.sum(axis=1, keepdims=True) + 1e-12
                p_r = joint.sum(axis=0, keepdims=True) + 1e-12

                mi = float(np.sum(joint * np.log2((joint + 1e-12) / (p_s * p_r))))
                mi_values.append(max(0.0, mi))

            if not mi_values:
                return 0.0
            return float(np.mean(mi_values))

    def estimate_mi(
        self,
        spikes: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, float]:
        mi = self.binning_mi(spikes, reference)
        return {
            "mutual_information": float(mi),
            "method": "binning_sign_discretization",
            "n_bins": self.n_bins,
            "n_dims_analyzed": self.n_dims,
        }


class RepresentationAnalyzer:

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def linear_cka(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        eps: float = 1e-12,
        max_samples: int = 5000,
    ) -> float:
        x = X.reshape(-1, X.shape[-1]).float().cpu().numpy()
        y = Y.reshape(-1, Y.shape[-1]).float().cpu().numpy()

        n = min(x.shape[0], y.shape[0], max_samples)
        if n == 0:
            return 0.0

        x = x[:n]
        y = y[:n]
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)

        hsic_xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
        hsic_xx = np.linalg.norm(x.T @ x, ord="fro") ** 2
        hsic_yy = np.linalg.norm(y.T @ y, ord="fro") ** 2

        denom = math.sqrt(float(hsic_xx * hsic_yy)) + eps
        return float(hsic_xy / denom)


class SpikingBrainValidator:

    def __init__(
        self,
        device: torch.device,
        layer_map: Optional[dict[int, int]] = None,
        thresholds: Optional[dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.layer_map = layer_map or {0: 0, 1: 1, 2: 2}

        defaults: dict[str, Any] = {
            "dead": 0.05,
            "saturated": 0.10,
            "mi": 0.1,
            "cka": 0.3,
            "firing_rate_range": (0.2, 0.6),
        }
        if thresholds:
            defaults.update(thresholds)

        self.dead_threshold: float = defaults["dead"]
        self.saturated_threshold: float = defaults["saturated"]
        self.mi_threshold: float = defaults["mi"]
        self.cka_threshold: float = defaults["cka"]
        self.firing_rate_range: tuple[float, float] = defaults["firing_rate_range"]

        self.mi_estimator = MutualInformationEstimator()
        self.rep_analyzer = RepresentationAnalyzer(device)

    def compute_health_metrics(
        self,
        spike_data: dict[int, list[torch.Tensor]],
    ) -> SpikeHealthMetrics:
        alerts: list[str] = []
        all_firing_rates: list[np.ndarray] = []

        for layer_idx, spike_tensors in spike_data.items():
            if not spike_tensors:
                continue

            stacked = torch.cat(
                [t.reshape(-1, t.shape[-1]) for t in spike_tensors], dim=0
            )
            firing_rates = (stacked != 0).float().mean(dim=0).cpu().numpy()
            all_firing_rates.append(firing_rates)

        if not all_firing_rates:
            return SpikeHealthMetrics(
                dead_neuron_pct=0.0,
                saturated_neuron_pct=0.0,
                firing_rate_mean=0.0,
                firing_rate_std=0.0,
                health_pass=False,
                alerts=["No spike data collected"],
            )

        combined_rates = np.concatenate(all_firing_rates)

        dead_pct = float((combined_rates < 0.05).mean())
        if dead_pct > self.dead_threshold:
            alerts.append(
                f"ALERT: {dead_pct * 100:.1f}% dead neurons (>{self.dead_threshold * 100}%)"
            )

        saturated_pct = float((combined_rates > 0.95).mean())
        if saturated_pct > self.saturated_threshold:
            alerts.append(
                f"ALERT: {saturated_pct * 100:.1f}% saturated neurons (>{self.saturated_threshold * 100}%)"
            )

        fr_mean = float(combined_rates.mean())
        fr_std = float(combined_rates.std())

        if not (self.firing_rate_range[0] <= fr_mean <= self.firing_rate_range[1]):
            alerts.append(
                f"ALERT: Firing rate {fr_mean:.3f} outside [{self.firing_rate_range[0]}, {self.firing_rate_range[1]}]"
            )

        health_pass = len(alerts) == 0

        return SpikeHealthMetrics(
            dead_neuron_pct=dead_pct,
            saturated_neuron_pct=saturated_pct,
            firing_rate_mean=fr_mean,
            firing_rate_std=fr_std,
            health_pass=health_pass,
            alerts=alerts,
        )

    def validate_spikes(
        self,
        spike_data: dict[int, list[torch.Tensor]],
        pre_spike_data: dict[int, list[torch.Tensor]],
    ) -> SpikingBrainValidation:
        health = self.compute_health_metrics(spike_data)

        mi_results, cka_results = self._compute_information_and_cka(
            spike_data, pre_spike_data
        )

        mi_pass = mi_results.get("mutual_information", 0) > self.mi_threshold
        cka_pass = cka_results.get("cka_mean", 0) > self.cka_threshold

        overall_pass = health.health_pass and mi_pass and cka_pass

        summary = self._generate_summary(health, mi_results, cka_results, overall_pass)

        return SpikingBrainValidation(
            health=health,
            mutual_information=mi_results,
            cka=cka_results,
            overall_pass=overall_pass,
            summary=summary,
        )

    def _compute_information_and_cka(
        self,
        spike_data: dict[int, list[torch.Tensor]],
        pre_spike_data: dict[int, list[torch.Tensor]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        mi_results: dict[str, float] = {}
        cka_results: dict[str, float] = {}
        mi_metadata: dict[str, Any] = {}

        for s_layer, p_layer in self.layer_map.items():
            if s_layer not in spike_data or p_layer not in pre_spike_data:
                continue

            spike_tensors = spike_data[s_layer]
            pre_spike_tensors = pre_spike_data[p_layer]

            if not spike_tensors or not pre_spike_tensors:
                continue

            spikes_flat = torch.cat(
                [t.reshape(-1, t.shape[-1]) for t in spike_tensors], dim=0
            )
            pre_flat = torch.cat(
                [t.reshape(-1, t.shape[-1]) for t in pre_spike_tensors], dim=0
            )

            mi_payload = self.mi_estimator.estimate_mi(spikes_flat, pre_flat)
            mi_value = float(mi_payload.get("mutual_information", 0.0))
            mi_results[f"layer_{s_layer}_to_{p_layer}"] = mi_value

            if not mi_metadata:
                mi_metadata = {
                    k: v for k, v in mi_payload.items() if k != "mutual_information"
                }

            cka_value = self.rep_analyzer.linear_cka(spikes_flat, pre_flat)
            cka_results[f"layer_{s_layer}_to_{p_layer}"] = cka_value

        mi_mean = float(np.mean(list(mi_results.values()))) if mi_results else 0.0
        cka_mean = float(np.mean(list(cka_results.values()))) if cka_results else 0.0

        mi_summary: dict[str, Any] = {
            **mi_results,
            "mutual_information": mi_mean,
            "method": mi_metadata.get("method", "binning"),
        }
        for key in ("n_bins", "n_dims_analyzed"):
            if key in mi_metadata:
                mi_summary[key] = mi_metadata[key]

        cka_summary: dict[str, Any] = {
            **cka_results,
            "cka_mean": cka_mean,
            "method": "linear_cka",
        }

        return mi_summary, cka_summary

    def _generate_summary(
        self,
        health: SpikeHealthMetrics,
        mi: dict[str, float],
        cka: dict[str, float],
        overall_pass: bool,
    ) -> str:
        lines = [
            "=" * 60,
            "SPIKINGBRAIN VALIDATION SUMMARY",
            "=" * 60,
            "",
            "[HEALTH]",
            f"  Dead neurons: {health.dead_neuron_pct * 100:.1f}% {'PASS' if health.dead_neuron_pct < self.dead_threshold else 'FAIL'}",
            f"  Saturated neurons: {health.saturated_neuron_pct * 100:.1f}% {'PASS' if health.saturated_neuron_pct < self.saturated_threshold else 'FAIL'}",
            f"  Firing rate: {health.firing_rate_mean:.3f} +/- {health.firing_rate_std:.3f}",
            "",
            "[INFORMATION]",
            f"  Mutual Information: {mi.get('mutual_information', 0):.4f} ({mi.get('method', 'binning')}) {'PASS' if mi.get('mutual_information', 0) > self.mi_threshold else 'FAIL'}",
            "",
            "[REPRESENTATION]",
            f"  CKA (mean): {cka.get('cka_mean', 0):.4f} {'PASS' if cka.get('cka_mean', 0) > self.cka_threshold else 'FAIL'}",
        ]

        for key, value in cka.items():
            if key.startswith("layer_"):
                lines.append(f"    {key}: {value:.4f}")

        if health.alerts:
            lines.extend(["", "[ALERTS]"])
            for alert in health.alerts:
                lines.append(f"  {alert}")

        lines.extend([
            "",
            "=" * 60,
            f"OVERALL: {'PASS' if overall_pass else 'NEEDS ATTENTION'}",
            "=" * 60,
        ])

        return "\n".join(lines)
