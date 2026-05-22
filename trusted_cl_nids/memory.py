from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryItem:
    x: np.ndarray
    y: int
    confidence: float
    source: str


class ClassBalancedReplayBuffer:
    """Class-balanced replay memory inspired by CBRS/augmented replay code."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.storage: Dict[int, List[MemoryItem]] = {0: [], 1: []}
        self.seen = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self.storage.values())

    def class_counts(self) -> Dict[int, int]:
        return {cls: len(items) for cls, items in self.storage.items()}

    def add_many(
        self,
        x: np.ndarray,
        y: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        source: str = "pseudo",
    ) -> None:
        if len(x) == 0:
            return
        if confidence is None:
            confidence = np.ones(len(x), dtype=np.float32)
        for xi, yi, ci in zip(x, y, confidence):
            self.add_one(xi.astype(np.float32), int(yi), float(ci), source)

    def add_one(self, x: np.ndarray, y: int, confidence: float, source: str) -> None:
        self.seen += 1
        item = MemoryItem(x=x.copy(), y=int(y), confidence=float(confidence), source=source)
        if len(self) < self.capacity:
            self.storage[item.y].append(item)
            return

        counts = self.class_counts()
        largest_class = max(counts, key=counts.get)
        current_count = counts.get(item.y, 0)
        largest_count = counts[largest_class]

        should_replace = current_count < largest_count
        if not should_replace:
            should_replace = self.rng.random() < (self.capacity / max(self.seen, 1))
        if should_replace and self.storage[largest_class]:
            replace_idx = int(self.rng.integers(0, len(self.storage[largest_class])))
            if largest_class == item.y:
                self.storage[largest_class][replace_idx] = item
            else:
                self.storage[largest_class].pop(replace_idx)
                self.storage[item.y].append(item)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self) == 0 or batch_size <= 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        available_classes = [cls for cls, items in self.storage.items() if items]
        per_class = max(1, batch_size // len(available_classes))
        selected: List[MemoryItem] = []
        for cls in available_classes:
            items = self.storage[cls]
            k = min(len(items), per_class)
            idx = self.rng.choice(len(items), size=k, replace=False)
            selected.extend(items[int(i)] for i in idx)

        remaining = batch_size - len(selected)
        if remaining > 0:
            all_items = [item for items in self.storage.values() for item in items]
            k = min(remaining, len(all_items))
            idx = self.rng.choice(len(all_items), size=k, replace=False)
            selected.extend(all_items[int(i)] for i in idx)

        self.rng.shuffle(selected)
        x = np.stack([item.x for item in selected]).astype(np.float32)
        y = np.asarray([item.y for item in selected], dtype=np.int64)
        confidence = np.asarray([item.confidence for item in selected], dtype=np.float32)
        return x, y, confidence
