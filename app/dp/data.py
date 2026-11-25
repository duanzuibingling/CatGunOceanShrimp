"""数据层：官方数据包适配器 + UCI 气体传感器漂移 + 合成漂移数据。

统一契约：
    Batch = (t: int, X: np.ndarray[n, d], y: np.ndarray[n], name: str)
所有数据源都返回 list[Batch]，按时间顺序排列。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Batch:
    t: int
    X: np.ndarray
    y: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def _read_numeric(path: str) -> np.ndarray:
    """读空格/制表符分隔的数值表，自动跳过非数值表头。"""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            parts = line.split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"空文件或无法解析: {path}")
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=np.float64)


def _read_uci_sparse(path: str) -> tuple[np.ndarray, np.ndarray]:
    """UCI 气体传感器原始格式：'类别 1:值 2:值 ...'。"""
    rows, labels, max_idx = [], [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                lab = int(float(parts[0]))
            except ValueError:
                continue
            vals = {}
            for tok in parts[1:]:
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    try:
                        vals[int(k)] = float(v)
                    except ValueError:
                        continue
            if not vals:
                continue
            rows.append(vals)
            labels.append(lab)
            max_idx = max(max_idx, max(vals))
    X = np.zeros((len(rows), max_idx), dtype=np.float64)
    for i, d in enumerate(rows):
        for k, v in d.items():
            X[i, k - 1] = v
    return X, np.asarray(labels, dtype=int)


def load_uci_gas(directory: str, base_batches: int = 2) -> List[Batch]:
    """UCI Gas Sensor Array Drift：batch1..batch10，最后一列是类别(1..6)。"""
    files = []
    for i in range(1, 11):
        for cand in (f"batch{i}.dat", f"batch{i}.txt", f"batch{i}.csv"):
            p = os.path.join(directory, cand)
            if os.path.exists(p):
                files.append((i, p))
                break
    if not files:
        raise FileNotFoundError(f"{directory} 下没有 batch1..10 数据文件")

    batches = []
    for i, p in files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        if ":" in first:
            X, y = _read_uci_sparse(p)
        else:
            arr = _read_numeric(p)
            X, y = arr[:, :-1], arr[:, -1].astype(int)
        batches.append(Batch(t=i, X=X, y=y, name=f"batch{i}"))
    return batches


def make_synthetic_drift(
    n_batches: int = 10,
    n_per_batch: int = 300,
    n_features: int = 24,
    n_classes: int = 6,
    drift_kind: str = "covariate",
    seed: int = 0,
) -> List[Batch]:
    """合成漂移流，用于在真实数据到位前验证管线。

    漂移类型：
      covariate —— 特征分布平移（传感器零漂/增益漂移）
      concept   —— 决策边界旋转（同分布不同标签规律）
      noise     —— 传感器噪声增大
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, size=(n_classes, n_features))
    coef = rng.normal(0, 1.0, size=(n_features, n_classes))
    batches = []
    for t in range(1, n_batches + 1):
        frac = (t - 1) / max(1, n_batches - 1)
        X, y = [], []
        for _ in range(n_per_batch):
            c = rng.integers(0, n_classes)
            x = centers[c] + rng.normal(0, 1.0, size=n_features)
            if drift_kind == "covariate":
                shift = frac * 1.2
                x = x + shift
                x[: n_features // 3] += frac * 1.5  # 前 1/3 传感器漂得厉害
                lab = c
            elif drift_kind == "concept":
                # 决策边界随时间旋转：P(x) 不变，P(y|x) 变（真正的概念漂移）
                th = frac * (np.pi / 2.5)
                R = np.eye(n_features)
                R[0, 0], R[0, 1] = np.cos(th), -np.sin(th)
                R[1, 0], R[1, 1] = np.sin(th), np.cos(th)
                lab = int(np.argmax((R @ x) @ coef) % n_classes)
            elif drift_kind == "noise":
                x = x + rng.normal(0, 1.0 + frac * 2.5, size=n_features)
                lab = c
            else:
                lab = c
            X.append(x)
            y.append(lab)
        batches.append(Batch(t=t, X=np.asarray(X), y=np.asarray(y), name=f"syn{t}"))
    return batches


def _npz_xy(path: str):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    xkey = next((k for k in ("X", "x", "features", "data") if k in keys), None)
    ykey = next((k for k in ("y", "label", "labels", "target") if k in keys), None)
    if xkey is None or ykey is None:
        raise ValueError(f"{os.path.basename(path)} 缺少特征或标签数组，实际键：{sorted(keys)}")
    return np.asarray(d[xkey], dtype=np.float64), np.asarray(d[ykey]).astype(int)


def _csv_xy(path: str):
    import csv
    with open(path, encoding="utf-8", errors="ignore") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} 为空")
    try:
        float(rows[0][0])
        body = rows
    except (ValueError, IndexError):
        body = rows[1:]
    arr = np.asarray([[float(v) for v in r] for r in body], dtype=np.float64)
    return arr[:, :-1], arr[:, -1].astype(int)


def load_official(data_dir: str) -> List[Batch]:
    """官方数据包适配器：自动识别常见的几种组织方式。

    支持：
      1) batch_000.npz（含 X / y，或 features / labels）
      2) batch_000/ 子目录，内含 X.npy + y.npy
      3) batch_000.csv（最后一列是标签，可带表头）
    数据到手后先跑 scripts/check_data.py 看清单，再决定要不要转换。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    entries = []
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        low = name.lower()
        if os.path.isfile(p) and low.endswith(".npz"):
            X, y = _npz_xy(p)
            entries.append(Batch(t=len(entries), X=X, y=y, name=name))
        elif os.path.isfile(p) and low.endswith(".csv"):
            X, y = _csv_xy(p)
            entries.append(Batch(t=len(entries), X=X, y=y, name=name))
        elif os.path.isdir(p):
            xp, yp = os.path.join(p, "X.npy"), os.path.join(p, "y.npy")
            if os.path.exists(xp) and os.path.exists(yp):
                entries.append(Batch(t=len(entries),
                                     X=np.load(xp).astype(np.float64),
                                     y=np.load(yp).astype(int), name=name))

    if not entries:
        raise FileNotFoundError(
            f"{data_dir} 下没有识别到批次数据。请把官方包按 batch_*.npz 或 batch_*/X.npy+y.npy 摆放，"
            f"或先跑 scripts/check_data.py 生成清单。"
        )
    return entries
