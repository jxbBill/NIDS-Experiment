from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftResult:
    detected: bool
    p_value: float
    statistic: float


class KSDriftDetector:
    """Two-sample KS detector over model scores."""

    def __init__(self, reference_size: int = 5000, alpha: float = 0.05) -> None:
        self.reference_size = int(reference_size)
        self.alpha = float(alpha)
        self.reference: Deque[float] = deque(maxlen=self.reference_size)

    def fit(self, scores: np.ndarray) -> None:
        self.reference.clear()
        self.update(scores)

    def update(self, scores: np.ndarray) -> None:
        for score in np.asarray(scores, dtype=np.float32).ravel():
            self.reference.append(float(score))

    def detect(self, scores: np.ndarray) -> DriftResult:
        scores = np.asarray(scores, dtype=np.float32).ravel()
        if len(self.reference) < 20 or len(scores) < 20:
            return DriftResult(False, 1.0, 0.0)
        stat, p_value = ks_2samp(np.asarray(self.reference), scores)
        return DriftResult(bool(p_value < self.alpha), float(p_value), float(stat))
