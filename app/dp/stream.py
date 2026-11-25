"""漂移监测 + 后训练策略 + 门禁，跑一条时间序数据流。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .data import Batch
from .memory import ReplayMemory

try:  # 完整环境用 sklearn；瘦身镜像（仅 numpy）里只用到 Monitor/KS，不加载模型层
    from .model import accuracy, fit_model, make_model
except Exception:  # pragma: no cover
    accuracy = fit_model = make_model = None


def ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """两样本 KS 统计量（纯 NumPy 实现，避免引入 SciPy 依赖）。

    与 scipy.stats.ks_2samp 的 statistic 定义一致：经验分布函数的最大差。
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n
    cdf_b = np.searchsorted(b, grid, side="right") / m
    return float(np.max(np.abs(cdf_a - cdf_b)))


# ---------------- 漂移监测 ----------------
@dataclass
class DriftReport:
    t: int
    score: float          # 0~1，越大越漂
    kind: str             # covariate / concept / none
    top_features: List[int]
    triggered: bool


class Monitor:
    """特征分布 KS 统计量 + 真实性能下降，两个信号取或。

    只用特征会漏掉「纯概念漂移」（P(y|x) 变了但 P(x) 没变），
    所以再加一路：用上一批已见标签算出的真实准确率，跌破参考值就触发。
    """

    def __init__(self, threshold: float = 0.35, feature_quantile: float = 0.9,
                 perf_drop: float = 0.08):
        self.threshold = threshold
        self.q = feature_quantile
        self.perf_drop = perf_drop
        self.ref_X: Optional[np.ndarray] = None
        self.ref_conf: Optional[float] = None

    def set_reference(self, X: np.ndarray):
        self.ref_X = X

    def observe(self, t: int, X: np.ndarray, model=None,
                last_acc: Optional[float] = None,
                ref_acc: Optional[float] = None) -> DriftReport:
        if self.ref_X is None:
            self.ref_X = X
            return DriftReport(t=t, score=0.0, kind="none", top_features=[], triggered=False)

        n_feat = min(self.ref_X.shape[1], X.shape[1])
        sample = min(400, len(X), len(self.ref_X))
        idx_new = np.random.default_rng(t).choice(len(X), sample, replace=False)
        idx_ref = np.random.default_rng(1000 + t).choice(len(self.ref_X), sample, replace=False)

        ks = np.zeros(n_feat)
        for j in range(n_feat):
            ks[j] = ks_2samp(self.ref_X[idx_ref, j], X[idx_new, j])
        score = float(np.quantile(ks, self.q))
        top = list(np.argsort(ks)[-3:][::-1])

        feat_drift = score > self.threshold
        perf_drift = (last_acc is not None and ref_acc is not None
                      and (ref_acc - last_acc) > self.perf_drop)
        if feat_drift:
            kind = "covariate"
        elif perf_drift:
            kind = "concept"
        else:
            kind = "none"
        return DriftReport(t=t, score=round(score, 4), kind=kind, top_features=top,
                           triggered=bool(feat_drift or perf_drift))


# ---------------- 策略 ----------------
@dataclass
class StepResult:
    t: int
    n: int
    acc: float
    base_acc: float           # 静态基线在同一批上的准确率
    acted: bool
    action: str
    train_seconds: float
    drift_score: float
    promoted: bool


class Strategy:
    name = "base"

    def __init__(self, model_kind: str = "hgb", seed: int = 0):
        self.model_kind = model_kind
        self.seed = seed

    def initial_fit(self, X, y):
        self.model = make_model(self.model_kind, self.seed)
        self.train_seconds = fit_model(self.model, X, y).seconds

    def on_batch(self, b: Batch, rep: DriftReport) -> StepResult:
        raise NotImplementedError


class StaticBaseline(Strategy):
    """部署后不再更新：官方静态基线。"""

    name = "static"

    def on_batch(self, b, rep):
        acc = accuracy(self.model, b.X, b.y)
        return StepResult(b.t, len(b), acc, acc, False, "none", 0.0, rep.score, False)


class NaiveRetrain(Strategy):
    """每批都把历史全量重训一遍：效果上限，但算力最贵。"""

    name = "naive_retrain"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.buf_X, self.buf_y = [], []

    def initial_fit(self, X, y):
        super().initial_fit(X, y)
        self.buf_X, self.buf_y = [X], [y]

    def on_batch(self, b, rep):
        acc = accuracy(self.model, b.X, b.y)
        self.buf_X.append(b.X)
        self.buf_y.append(b.y)
        m = make_model(self.model_kind, self.seed)
        rec = fit_model(m, np.vstack(self.buf_X), np.concatenate(self.buf_y))
        self.model = m
        return StepResult(b.t, len(b), acc, acc, True, "full_retrain", rec.seconds, rep.score, True)


class DriftTriggered(Strategy):
    """只在漂移触发时重训：近期样本 + 水库样本一起训练（防遗忘）。"""

    name = "drift_triggered"

    def __init__(self, model_kind="hgb", seed=0, buffer_per_class=80, max_buffer_batches=4,
                 reservoir_size=2000, **kw):
        super().__init__(model_kind, seed, **kw)
        self.buffer_per_class = buffer_per_class
        self.max_buffer_batches = max_buffer_batches
        self.memory = ReplayMemory(max_recent_batches=max_buffer_batches,
                                   reservoir_size=reservoir_size, seed=seed)

    def initial_fit(self, X, y):
        super().initial_fit(X, y)
        self.memory.add(X, y)

    def _replay(self):
        """训练集 = 水库样本（旧分布）+ 近期样本（新分布）。"""
        return self.memory.train_set(per_class=self.buffer_per_class)

    def on_batch(self, b, rep):
        acc = accuracy(self.model, b.X, b.y)
        acted, action, secs = False, "none", 0.0
        if rep.triggered:
            Xr, yr = self._replay()
            m = make_model(self.model_kind, self.seed)
            rec = fit_model(m, np.vstack([Xr, b.X]), np.concatenate([yr, b.y]))
            self.model = m
            acted, action, secs = True, "replay_posttrain", rec.seconds
            self.memory.add(b.X, b.y)
        return StepResult(b.t, len(b), acc, acc, acted, action, secs, rep.score, acted)


class GuardedDriftTriggered(DriftTriggered):
    """漂移触发 + 双探针门禁。

    候选模型必须同时满足：
      ① 在**旧分布探针**上不明显退化（防止遗忘早期子集）
      ② 在**新分布探针**上不明显退化（防止越改越差）
    只在最近缓冲上比（旧做法）会天然偏向刚训过的新模型，导致门禁形同虚设。
    """

    name = "guarded"

    def __init__(self, model_kind="hgb", seed=0, buffer_per_class=80,
                 max_buffer_batches=4, reservoir_size=2000, old_tol=0.02, new_tol=0.005, **kw):
        super().__init__(model_kind, seed, buffer_per_class, max_buffer_batches,
                         reservoir_size, **kw)
        self.old_tol = old_tol      # 旧分布允许的最大跌幅
        self.new_tol = new_tol      # 新分布允许的最大跌幅

    def on_batch(self, b, rep):
        acc_before = accuracy(self.model, b.X, b.y)
        if not rep.triggered:
            return StepResult(b.t, len(b), acc_before, acc_before, False, "none", 0.0,
                              rep.score, False)

        Xr, yr = self._replay()
        cand = make_model(self.model_kind, self.seed)
        rec = fit_model(cand, np.vstack([Xr, b.X]), np.concatenate([yr, b.y]))

        (Xo, yo), (Xn, yn) = self.memory.probe()
        old_on_old = accuracy(self.model, Xo, yo)
        new_on_old = accuracy(cand, Xo, yo)
        old_on_new = accuracy(self.model, Xn, yn)
        new_on_new = accuracy(cand, Xn, yn)

        ok_old = new_on_old >= old_on_old - self.old_tol
        ok_new = new_on_new >= old_on_new - self.new_tol
        promoted = bool(ok_old and ok_new)
        if promoted:
            self.model = cand
            self.memory.add(b.X, b.y)
        return StepResult(b.t, len(b), acc_before, acc_before, True,
                          "replay_posttrain" if promoted else "rejected_by_gate",
                          rec.seconds, rep.score, promoted)


def run_stream(
    batches: List[Batch],
    strategy: Strategy,
    monitor: Monitor,
    base_count: int = 2,
) -> List[StepResult]:
    """前 base_count 个批次用于初始训练，其后逐批“先预测、后决策”。"""
    init_X = np.vstack([b.X for b in batches[:base_count]])
    init_y = np.concatenate([b.y for b in batches[:base_count]])
    strategy.initial_fit(init_X, init_y)
    monitor.set_reference(init_X)

    # 参考性能只用于“性能骤降”触发；策略若做了特征变换，这里容忍失败
    try:
        ref_acc = accuracy(strategy.model, init_X, init_y)
    except Exception:
        ref_acc = 0.0
    last_acc = ref_acc
    results = []
    for b in batches[base_count:]:
        rep = monitor.observe(b.t, b.X, strategy.model, last_acc=last_acc, ref_acc=ref_acc)
        res = strategy.on_batch(b, rep)
        last_acc = res.acc
        if res.promoted:  # 模型更新成功后，把参考性能抬到当前水平
            ref_acc = max(ref_acc, res.acc)
        results.append(res)
    return results



