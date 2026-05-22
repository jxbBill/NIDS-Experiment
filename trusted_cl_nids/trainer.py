from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle
from .drift import KSDriftDetector
from .losses import supervised_contrastive_loss, weighted_bce_with_logits
from .memory import ClassBalancedReplayBuffer
from .metrics import binary_metrics
from .model import AutoencoderClassifier
from .utils import ensure_dir, save_json


@dataclass
class TrainerConfig:
    batch_size: int = 128
    initial_epochs: int = 5
    online_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    window_size: int = 2000
    max_windows: int = 0
    tau: float = 0.85
    consistency_tau: float = 0.90
    noise_std: float = 0.01
    memory_size: int = 4096
    replay_batch_size: int = 256
    drift_alpha: float = 0.05
    reference_size: int = 5000
    lambda_recon: float = 0.2
    lambda_contrastive: float = 0.05
    lambda_consistency: float = 0.1
    lambda_distill: float = 0.2
    replay_weight: float = 1.0
    pseudo_weight: float = 1.0
    always_update: bool = False
    oracle_budget: int = 0
    output_dir: Path = Path("outputs/run")


class TrustedPseudoLabelCLTrainer:
    def __init__(self, bundle: DatasetBundle, config: TrainerConfig, device: torch.device) -> None:
        self.bundle = bundle
        self.cfg = config
        self.device = device
        self.model = AutoencoderClassifier(bundle.input_dim).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.memory = ClassBalancedReplayBuffer(config.memory_size, seed=17)
        self.detector = KSDriftDetector(
            reference_size=config.reference_size,
            alpha=config.drift_alpha,
        )
        ensure_dir(config.output_dir)

    def run(self) -> Dict[str, float]:
        save_json(self.cfg.output_dir / "config.json", asdict(self.cfg))
        self._train_initial()
        self.memory.add_many(
            self.bundle.x_initial,
            self.bundle.y_initial,
            confidence=np.ones(len(self.bundle.y_initial), dtype=np.float32),
            source="initial",
        )
        initial_scores = self._score_numpy(self.bundle.x_initial)[0]
        self.detector.fit(initial_scores)

        window_records = self._run_stream()
        pd.DataFrame(window_records).to_csv(self.cfg.output_dir / "window_log.csv", index=False)

        final_prob, _, _ = self._score_numpy(self.bundle.x_test)
        final_pred = (final_prob >= 0.5).astype(np.int64)
        final_metrics = binary_metrics(self.bundle.y_test, final_pred)
        final_metrics["num_windows"] = float(len(window_records))
        final_metrics["memory_total"] = float(len(self.memory))
        for cls, count in self.memory.class_counts().items():
            final_metrics[f"memory_class_{cls}"] = float(count)

        save_json(self.cfg.output_dir / "final_metrics.json", final_metrics)
        torch.save(self.model.state_dict(), self.cfg.output_dir / "model.pt")
        return final_metrics

    def _train_initial(self) -> None:
        x = torch.from_numpy(self.bundle.x_initial).float()
        y = torch.from_numpy(self.bundle.y_initial).long()
        loader = DataLoader(TensorDataset(x, y), batch_size=self.cfg.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.cfg.initial_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                features, reconstruction, logits = self.model(xb)
                loss = self.bce(logits, yb.float())
                loss = loss + self.cfg.lambda_recon * F.mse_loss(reconstruction, xb)
                loss = loss + self.cfg.lambda_contrastive * supervised_contrastive_loss(features, yb)
                loss.backward()
                self.optimizer.step()

    def _run_stream(self) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        x_stream = self.bundle.x_stream
        y_stream = self.bundle.y_stream
        num_windows = int(np.ceil(len(x_stream) / self.cfg.window_size))
        if self.cfg.max_windows > 0:
            num_windows = min(num_windows, self.cfg.max_windows)

        for window_id in range(num_windows):
            start = window_id * self.cfg.window_size
            end = min(start + self.cfg.window_size, len(x_stream))
            x_window = x_stream[start:end]
            y_window = y_stream[start:end]
            prob, recon_error, prob_aug = self._score_numpy(x_window, with_aug=True)
            pred = (prob >= 0.5).astype(np.int64)
            metrics = binary_metrics(y_window, pred)

            confidence = np.maximum(prob, 1.0 - prob)
            consistency = 1.0 - np.abs(prob - prob_aug)
            trusted_mask = (confidence >= self.cfg.tau) & (consistency >= self.cfg.consistency_tau)
            drift = self.detector.detect(prob)
            should_update = bool(drift.detected or self.cfg.always_update)

            pseudo_x = x_window[trusted_mask]
            pseudo_y = pred[trusted_mask]
            pseudo_conf = confidence[trusted_mask]

            oracle_used = 0
            if should_update and self.cfg.oracle_budget > 0:
                oracle_x, oracle_y, oracle_conf = self._select_oracle_samples(
                    x_window, y_window, confidence, self.cfg.oracle_budget
                )
                oracle_used = len(oracle_y)
                if oracle_used > 0:
                    pseudo_x = np.concatenate([pseudo_x, oracle_x], axis=0)
                    pseudo_y = np.concatenate([pseudo_y, oracle_y], axis=0)
                    pseudo_conf = np.concatenate([pseudo_conf, oracle_conf], axis=0)

            if should_update and len(pseudo_y) > 0:
                self.memory.add_many(pseudo_x, pseudo_y, pseudo_conf, source="pseudo")
                teacher = copy.deepcopy(self.model).to(self.device)
                teacher.eval()
                replay_x, replay_y, replay_conf = self.memory.sample(self.cfg.replay_batch_size)
                self._online_update(pseudo_x, pseudo_y, pseudo_conf, replay_x, replay_y, replay_conf, teacher)

            self.detector.update(prob)
            record = {
                "window": float(window_id),
                "start": float(start),
                "end": float(end),
                "drift": float(drift.detected),
                "drift_p_value": float(drift.p_value),
                "drift_statistic": float(drift.statistic),
                "trusted": float(len(pseudo_y)),
                "trusted_ratio": float(len(pseudo_y) / max(len(x_window), 1)),
                "oracle_used": float(oracle_used),
                "mean_confidence": float(confidence.mean()),
                "mean_recon_error": float(recon_error.mean()),
                "memory_total": float(len(self.memory)),
            }
            record.update({f"window_{k}": v for k, v in metrics.items()})
            records.append(record)
            print(
                f"window={window_id:03d} drift={int(drift.detected)} "
                f"trusted={len(pseudo_y)} f1={metrics['f1']:.4f} "
                f"memory={len(self.memory)}"
            )
        return records

    def _online_update(
        self,
        pseudo_x: np.ndarray,
        pseudo_y: np.ndarray,
        pseudo_conf: np.ndarray,
        replay_x: np.ndarray,
        replay_y: np.ndarray,
        replay_conf: np.ndarray,
        teacher: nn.Module,
    ) -> None:
        x_parts = [pseudo_x]
        y_parts = [pseudo_y]
        weight_parts = [self.cfg.pseudo_weight * pseudo_conf]
        if replay_x.size > 0:
            x_parts.append(replay_x)
            y_parts.append(replay_y)
            weight_parts.append(self.cfg.replay_weight * replay_conf)

        x = torch.from_numpy(np.concatenate(x_parts, axis=0)).float()
        y = torch.from_numpy(np.concatenate(y_parts, axis=0)).long()
        weights = torch.from_numpy(np.concatenate(weight_parts, axis=0)).float()
        loader = DataLoader(TensorDataset(x, y, weights), batch_size=self.cfg.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.cfg.online_epochs):
            for xb, yb, wb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)
                noisy_xb = self._augment_tensor(xb)

                self.optimizer.zero_grad()
                features, reconstruction, logits = self.model(xb)
                _, _, noisy_logits = self.model(noisy_xb)
                with torch.no_grad():
                    _, _, teacher_logits = teacher(xb)

                loss = weighted_bce_with_logits(logits, yb, wb)
                loss = loss + self.cfg.lambda_recon * F.mse_loss(reconstruction, xb)
                loss = loss + self.cfg.lambda_contrastive * supervised_contrastive_loss(features, yb)
                loss = loss + self.cfg.lambda_consistency * F.mse_loss(
                    torch.sigmoid(logits), torch.sigmoid(noisy_logits)
                )
                loss = loss + self.cfg.lambda_distill * F.mse_loss(logits, teacher_logits)
                loss.backward()
                self.optimizer.step()

    def _score_numpy(
        self,
        x: np.ndarray,
        with_aug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x).float()),
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        probs: List[np.ndarray] = []
        recon_errors: List[np.ndarray] = []
        aug_probs: List[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                _, reconstruction, logits = self.model(xb)
                prob = torch.sigmoid(logits)
                probs.append(prob.cpu().numpy())
                recon_errors.append(F.mse_loss(reconstruction, xb, reduction="none").mean(dim=1).cpu().numpy())
                if with_aug:
                    _, _, aug_logits = self.model(self._augment_tensor(xb))
                    aug_probs.append(torch.sigmoid(aug_logits).cpu().numpy())
        prob_arr = np.concatenate(probs).astype(np.float32)
        recon_arr = np.concatenate(recon_errors).astype(np.float32)
        if with_aug:
            aug_arr = np.concatenate(aug_probs).astype(np.float32)
        else:
            aug_arr = prob_arr.copy()
        return prob_arr, recon_arr, aug_arr

    def _augment_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.cfg.noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    @staticmethod
    def _select_oracle_samples(
        x: np.ndarray,
        y: np.ndarray,
        confidence: np.ndarray,
        budget: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if budget <= 0 or len(x) == 0:
            return x[:0], y[:0], confidence[:0]
        k = min(budget, len(x))
        idx = np.argsort(confidence)[:k]
        return x[idx], y[idx], np.ones(k, dtype=np.float32)
