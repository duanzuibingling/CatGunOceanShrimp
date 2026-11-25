"""任务注册表：三个数据集 = 三种任务，指标与模型头都不同。

赛题原文：A 包含「气体传感器漂移、轴承退化识别、发动机剩余寿命评估」三个数据集。
- 气体传感器漂移：多分类（6 类气体）
- 轴承退化识别：多分类（退化阶段）
- 发动机剩余寿命：回归（RUL 连续值）

所以不能用同一个模型头通吃；共用的是编排层（监控/决策/门禁/审计），
这一层才是赛题真正要考的 Harness。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TaskSpec:
    key: str                 # gas / bearing / engine
    name: str                # 中文名
    kind: str                # classification / regression
    metric: str              # accuracy / balanced_f1 / rmse
    n_classes: Optional[int] = None
    note: str = ""

    @property
    def is_regression(self) -> bool:
        return self.kind == "regression"


REGISTRY = {
    "gas": TaskSpec("gas", "气体传感器漂移", "classification", "accuracy", 6,
                    "128 维特征，6 类气体，跨 36 个月 10 批"),
    "bearing": TaskSpec("bearing", "轴承退化识别", "classification", "balanced_f1", 5,
                        "XJTU-SY，振动特征，退化阶段 5 档"),
    "engine": TaskSpec("engine", "发动机剩余寿命", "regression", "accuracy_binned", None,
                       "C-MAPSS；当前用 RUL 分箱近似评准确率，回归 RMSE/PHM08 见 results/cmapss_reg"),
}


def detect(source: str, data_hint: str = "") -> TaskSpec:
    """按数据源名字猜任务类型；猜不出就退回气体（分类）并留痕。"""
    s = f"{source} {data_hint}".lower()
    if "cmapss" in s or "engine" in s or "rul" in s:
        return REGISTRY["engine"]
    if "xjtu" in s or "bearing" in s:
        return REGISTRY["bearing"]
    if "gas" in s or "uci" in s:
        return REGISTRY["gas"]
    return REGISTRY["gas"]


def balanced_f1(y_true, y_pred) -> float:
    """宏平均 F1：对类别不平衡的数据集，比 accuracy 更诚实。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def compute_metric(task: TaskSpec, y_true, y_pred) -> float:
    """统一的指标出口：分类走 accuracy / balanced_f1，回归走 RMSE。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if task.metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if task.metric == "balanced_f1":
        return balanced_f1(y_true.astype(int), y_pred.astype(int))
    return float((y_true == y_pred).mean())


def describe(task: TaskSpec) -> dict:
    return {"key": task.key, "name": task.name, "kind": task.kind,
            "metric": task.metric, "n_classes": task.n_classes, "note": task.note}
