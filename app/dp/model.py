"""模型层：一个可增量“后训练”的分类器，资源占用小、CPU 秒级可训。"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "hgb", seed: int = 0, n_iter: int = 120):
    """n_iter 越小越便宜：严重漂移用 120，轻微漂移用 40 即可。"""
    if kind == "hgb":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=int(n_iter), max_depth=6, learning_rate=0.1,
                early_stopping=False, random_state=seed,
            ),
        )
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, n_jobs=1, random_state=seed),
        )
    raise ValueError(f"未知模型: {kind}")


@dataclass
class TrainRecord:
    seconds: float
    n_samples: int
    n_classes: int


def fit_model(model, X: np.ndarray, y: np.ndarray) -> TrainRecord:
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    return TrainRecord(seconds=round(dt, 3), n_samples=len(y), n_classes=len(np.unique(y)))


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float((predict(model, X) == y).mean())
