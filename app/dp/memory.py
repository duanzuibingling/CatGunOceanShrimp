"""长期记忆：近期缓冲 + 水库采样，解决「学了新的忘了旧的」。

原来的做法只保留最近 N 批，早期分布被整体丢掉，于是：
  - 后训练在旧子集上大幅退化（C-MAPSS 实测：留出集 0.8204 → 0.7910）；
  - 门禁只在最近缓冲上比较，天然偏向刚训过的新模型，几乎不放行失败。

现在改成两层记忆：
  recent     最近 N 批（保证新分布）
  reservoir  全历史水库采样（保证旧分布不丢）
  probe      旧+新各半的评估集（门禁用它，防止“新模型赢了新数据、输掉老数据”）
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class ReplayMemory:
    def __init__(self, max_recent_batches: int = 4, reservoir_size: int = 2000,
                 seed: int = 0):
        self.max_recent_batches = max_recent_batches
        self.reservoir_size = reservoir_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.recent_X: List[np.ndarray] = []
        self.recent_y: List[np.ndarray] = []
        self._res_X: Optional[np.ndarray] = None
        self._res_y: Optional[np.ndarray] = None
        self._seen = 0

    # ---------- 写入 ----------
    def add(self, X: np.ndarray, y: np.ndarray):
        self.recent_X.append(X)
        self.recent_y.append(y)
        while len(self.recent_X) > self.max_recent_batches:
            self.recent_X.pop(0)
            self.recent_y.pop(0)
        self._update_reservoir(X, y)

    def _update_reservoir(self, X: np.ndarray, y: np.ndarray):
        """经典水库采样：每条样本以 cap/seen 的概率替换库中一条。"""
        cap = self.reservoir_size
        res_X = self._res_X
        res_y = self._res_y
        for i in range(len(y)):
            self._seen += 1
            if res_X is None:
                res_X = X[i:i + 1].copy()
                res_y = np.asarray(y[i:i + 1])
                continue
            if len(res_y) < cap:
                res_X = np.vstack([res_X, X[i:i + 1]])
                res_y = np.concatenate([res_y, y[i:i + 1]])
            else:
                j = self.rng.integers(0, self._seen)
                if j < cap:
                    res_X[j] = X[i]
                    res_y[j] = y[i]
        self._res_X, self._res_y = res_X, res_y

    # ---------- 读取 ----------
    def recent(self, per_class: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        if not self.recent_X:
            raise RuntimeError("记忆为空")
        X = np.vstack(self.recent_X)
        y = np.concatenate(self.recent_y)
        return self._balanced_sample(X, y, per_class)

    def reservoir(self, cap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self._res_X is None:
            return self.recent()
        X, y = self._res_X, self._res_y
        cap = cap or len(y)
        if len(y) > cap:
            idx = self.rng.choice(len(y), cap, replace=False)
            return X[idx], y[idx]
        return X, y

    def train_set(self, per_class: int = 80, reservoir_cap: int = 1200):
        """训练用：近期样本 + 水库样本。"""
        Xr, yr = self.recent(per_class)
        Xo, yo = self.reservoir(reservoir_cap)
        return np.vstack([Xo, Xr]), np.concatenate([yo, yr])

    def probe(self, n_old: int = 500, n_new: int = 500):
        """门禁用：旧分布与新分布各取一半，分别返回，便于判断“是否赢新输旧”。"""
        Xo, yo = self.reservoir(n_old)
        try:
            Xn, yn = self.recent(per_class=10_000)
        except Exception:
            Xn, yn = Xo, yo
        if len(yn) > n_new:
            idx = self.rng.choice(len(yn), n_new, replace=False)
            Xn, yn = Xn[idx], yn[idx]
        return (Xo, yo), (Xn, yn)

    def _balanced_sample(self, X, y, per_class):
        idx = []
        for c in np.unique(y):
            ii = np.flatnonzero(y == c)
            if len(ii) > per_class:
                ii = self.rng.choice(ii, per_class, replace=False)
            idx.append(ii)
        idx = np.concatenate(idx)
        return X[idx], y[idx]

    @property
    def size(self) -> int:
        return 0 if self._res_y is None else len(self._res_y)
