"""官方训练包适配层：把 A/B/C 三个包接进我们的 Harness。

官方协议（见包内 README / TRAINING_GUIDE）：
  - 训练：python -m baselines.train --profile X --train ... --validation ... [--resume ckpt] --output 新目录
  - 预测：python -m baselines.predict --checkpoint ckpt，stdin 逐行 {"task","inputs"}，stdout 逐行 {"outputs":...}
  - 产物：checkpoint.npz + training.json + history.jsonl（每轮一个新目录）

本模块只做三件事：切流、调预测、调续训。模型本体仍然是官方的 NumPy MLP。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = r"D:\竞赛"
OFFICIAL = os.path.join(ROOT, "data", "official")
TMP = os.path.join(ROOT, "work", "tmp")


# ---------------- 数据 ----------------
def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def batch_by_meta(rows: List[dict], key: str = "source_batch") -> List[tuple]:
    """按 meta 字段分批，保持时间顺序。"""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get("meta", {}).get(key)), []).append(r)

    def _sort_key(k):
        try:
            return (0, float(k))
        except ValueError:
            return (1, k)

    return [(k, groups[k]) for k in sorted(groups, key=_sort_key)]


# ---------------- 指标 ----------------
def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(list(y_true) + list(y_pred)))
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------- 包 ----------------
@dataclass
class OfficialPackage:
    profile: str
    root: str = ""
    python: str = ""

    def __post_init__(self):
        self.root = self.root or os.path.join(OFFICIAL, self.profile)
        self.python = self.python or sys.executable
        self.train_path = os.path.join(self.root, "public", "train.jsonl")
        self.val_path = os.path.join(self.root, "public", "validation.jsonl")
        self.baseline_ckpt = os.path.join(self.root, "checkpoint.npz")
        with open(os.path.join(self.root, "baselines", "configs", f"{self.profile}.json"),
                  encoding="utf-8") as f:
            self.config = json.load(f)
        self.task = self.config["task"]
        self.classes = self.config.get("classes", [])
        self.epochs = int(self.config.get("epochs", 5))

    # ---- 预测 ----
    def predict(self, checkpoint: str, rows: Sequence[dict]) -> List[dict]:
        """返回每个样本的预测：分类为标签字符串，回归为数值。"""
        payload = "\n".join(json.dumps({"task": self.task, "inputs": r["inputs"]},
                                       ensure_ascii=False) for r in rows) + "\n"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", BASELINE_CPU_THREADS="2")
        p = subprocess.run([self.python, "-m", "baselines.predict",
                            "--checkpoint", checkpoint],
                           cwd=self.root, input=payload, capture_output=True,
                           text=True, encoding="utf-8", errors="ignore", env=env)
        if p.returncode != 0:
            raise RuntimeError(f"predict 失败：{p.stderr[-300:]}")
        outs = []
        for line in p.stdout.strip().splitlines():
            obj = json.loads(line)["outputs"]
            if self.task == "classification":
                best = max(obj, key=lambda d: d["score"])
                outs.append(best["label"])
            else:
                outs.append(float(obj["value"]))
        return outs

    # ---- 训练一轮 ----
    def train_round(self, train_rows: Sequence[dict], out_dir: str,
                    resume: Optional[str] = None, from_baseline: Optional[str] = None,
                    epochs: Optional[int] = None, lr: Optional[float] = None,
                    val_rows: Optional[Sequence[dict]] = None) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        train_file = os.path.join(out_dir, "train_used.jsonl")
        write_jsonl(train_file, train_rows)
        val_file = self.val_path
        if val_rows is not None:
            val_file = os.path.join(out_dir, "val_used.jsonl")
            write_jsonl(val_file, val_rows)

        cmd = [self.python, "-m", "baselines.train", "--profile", self.profile,
               "--train", train_file, "--validation", val_file,
               "--epochs", str(epochs or self.epochs),
               "--output", os.path.join(out_dir, "ckpt"),
               "--stage-id", os.path.basename(out_dir)]
        if resume:
            cmd += ["--resume", resume]
        elif from_baseline:
            cmd += ["--from-baseline", from_baseline]
        if lr:
            cmd += ["--learning-rate", str(lr)]

        env = dict(os.environ, PYTHONIOENCODING="utf-8", BASELINE_CPU_THREADS="4",
                   TMP=TMP, TEMP=TMP)
        p = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", env=env)
        if p.returncode != 0:
            raise RuntimeError(f"train 失败：{p.stderr[-400:]}")
        rep = json.loads(p.stdout.strip().splitlines()[-1])
        rep["checkpoint"] = os.path.join(out_dir, "ckpt", "checkpoint.npz")
        return rep

    # ---- 指标 ----
    def score(self, rows: Sequence[dict], preds: Sequence) -> float:
        if self.task == "classification":
            return macro_f1([r["answer"] for r in rows], list(preds))
        return rmse([float(r["answer"]) for r in rows], [float(p) for p in preds])
