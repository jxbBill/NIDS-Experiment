from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DatasetBundle:
    x_initial: np.ndarray
    y_initial: np.ndarray
    x_stream: np.ndarray
    y_stream: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    dataset_name: str

    @property
    def input_dim(self) -> int:
        return int(self.x_initial.shape[1])


def _stratified_limit(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return x, y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(y)), y))
    selected = np.sort(selected)
    return x[selected], y[selected]


def _initial_stream_split(
    x: np.ndarray,
    y: np.ndarray,
    initial_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < initial_ratio < 1.0:
        raise ValueError("initial_ratio must be in (0, 1).")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=initial_ratio, random_state=seed)
    initial_idx, stream_idx = next(splitter.split(np.zeros(len(y)), y))
    return x[np.sort(initial_idx)], y[np.sort(initial_idx)], x[np.sort(stream_idx)], y[np.sort(stream_idx)]


def load_nsl_kdd(
    data_root: Path,
    initial_ratio: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DatasetBundle:
    train_path = data_root / "NSL_pre_data" / "PKDDTrain+.csv"
    test_path = data_root / "NSL_pre_data" / "PKDDTest+.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Cannot find NSL-KDD CSV files. Expected "
            f"{train_path} and {test_path}."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = (train_df["labels2"].to_numpy() != "normal").astype(np.int64)
    y_test = (test_df["labels2"].to_numpy() != "normal").astype(np.int64)
    x_train_df = train_df.drop(columns=["labels2", "labels5"])
    x_test_df = test_df.drop(columns=["labels2", "labels5"])

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_df).astype(np.float32)
    x_test = scaler.transform(x_test_df).astype(np.float32)

    x_train, y_train = _stratified_limit(x_train, y_train, max_train_samples, seed)
    x_test, y_test = _stratified_limit(x_test, y_test, max_test_samples, seed + 17)
    x_initial, y_initial, x_stream, y_stream = _initial_stream_split(
        x_train, y_train, initial_ratio=initial_ratio, seed=seed
    )

    return DatasetBundle(
        x_initial=x_initial,
        y_initial=y_initial,
        x_stream=x_stream,
        y_stream=y_stream,
        x_test=x_test,
        y_test=y_test,
        feature_names=list(x_train_df.columns),
        dataset_name="nsl",
    )


def load_dataset(
    dataset: str,
    data_root: Path,
    initial_ratio: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DatasetBundle:
    dataset = dataset.lower()
    if dataset != "nsl":
        raise ValueError("This prototype currently uses NSL-KDD. Pass --dataset nsl.")
    return load_nsl_kdd(
        data_root=data_root,
        initial_ratio=initial_ratio,
        seed=seed,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
