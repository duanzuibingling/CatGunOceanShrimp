"""C-MAPSS 连续 RUL 回归评估通路（与 ``dp.adapters.cmapss`` 的分箱分类互补）。

为什么需要这条通路
------------------
``dp.adapters.cmapss`` 为了复用分类管线，把 RUL 分箱成 5 档，报出的"准确率"只说明
档位判对，**不能换算成 RMSE / PHM08 评分**（同档内差 1 个循环和差 19 个循环同罚）。
本模块复用其窗口/切批私有函数（不改动原文件），只把标签换成连续 RUL：

    RUL = (该发动机失效前的总循环数 - 1) - 窗口末端循环
    RUL = clip(RUL, 0, RUL_CAP)      # RUL_CAP = 125，PHM 标准截断

截断到 125 的理由：早期健康阶段 RUL 从 190 掉到 180 对运维毫无区别，但会让 RMSE
被这些"本来就不该预测准"的样本主导；PHM08 官方做法也是先截断再算分。

窗口/切批与 ``load_cmapss`` **完全一致**：窗口 30、步长 1、每 30 台发动机一批、
FD001→FD004 串联、批名 ``FD002#3`` 同格式。因此同一 ``t`` 在两条通路上指的是
同一批发动机，分类结论与回归结论可以逐批对齐。

留出集同理：``load_cmapss_reg_test`` 用 ``test_FD00x + RUL_FD00x``，
窗口末端 e 的 RUL = ``rul_last + ((n_cycle - 1) - e)``，同样截断到 125；
FD004 官方 test 有 249 台而 RUL 只有 248 个值，按较小值对齐并在 meta 里标注。

评估指标
--------
``evaluate_regression`` 返回 RMSE、MAE 和 PHM08 评分：

    d = pred - true（正数=预测偏晚，更危险）
    score = Σ_{d<0} (exp(-d/13) - 1) + Σ_{d>=0} (exp(d/10) - 1)

PHM08 对"预测偏晚"罚得更重（分母 10 < 13），这是该赛题评估函数的标准形式。
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .adapters.cmapss import (
    SENSOR_COLS,
    _check_files,
    _engine_slices,
    _read_table,
    _window_features,
)
from .data import Batch
from .stream import DriftReport, Monitor

__all__ = [
    "RUL_CAP",
    "RegressionMetrics",
    "evaluate_regression",
    "load_cmapss_reg",
    "load_cmapss_reg_test",
    "make_regressor",
    "fit_regressor",
    "predict_regressor",
    "RegReplayMemory",
    "StepResultReg",
    "StaticBaselineReg",
    "DriftTriggeredReg",
    "GuardedDriftTriggeredReg",
    "run_stream_regression",
]

#: 连续 RUL 的截断上限（PHM08 标准做法）
RUL_CAP: float = 125.0


# ---------------------------------------------------------------- 指标
@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    phm08: float
    n: int

    @property
    def phm08_per_sample(self) -> float:
        return float(self.phm08 / self.n) if self.n else float("nan")

    def as_dict(self, nd: int = 4) -> Dict[str, float]:
        d = asdict(self)
        return {k: (round(float(v), nd) if k != "n" else int(v)) for k, v in d.items()}


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """RMSE / MAE / PHM08。空输入返回 NaN 指标（n=0），不抛异常。"""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 长度不一致：{yt.shape} vs {yp.shape}")
    n = int(yt.size)
    if n == 0:
        return RegressionMetrics(rmse=float("nan"), mae=float("nan"),
                                 phm08=float("nan"), n=0)

    d = yp - yt
    rmse = float(np.sqrt(np.mean(d ** 2)))
    mae = float(np.mean(np.abs(d)))
    # 不对称罚分：预测偏晚（d>0）罚得更重
    pen = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return RegressionMetrics(rmse=rmse, mae=mae, phm08=float(pen.sum()), n=n)


# ---------------------------------------------------------------- 数据
def _subset_windows_reg(
    path: str, window: int, step: int, rul_cap: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """读一个 train 子集：每台发动机的 (特征, 连续 RUL, 窗口数)。

    窗口与标签口径和 ``cmapss._subset_windows`` 一致，只把 ``rul_to_class``
    换成 ``clip(rul, 0, rul_cap)``。
    """
    arr = _read_table(path)
    units = arr[:, 0].astype(np.int64)
    feats_list, rul_list, sizes = [], [], []
    for a, b in _engine_slices(units):
        track = arr[a:b, list(SENSOR_COLS)]
        n_cycle = b - a
        feats, end_idx = _window_features(track, window, step)
        if len(end_idx) == 0:          # 轨迹短于窗口，跳过（FD001–FD004 不会出现）
            continue
        rul = (n_cycle - 1) - end_idx  # 窗口末端距离失效的循环数
        feats_list.append(feats)
        rul_list.append(np.clip(rul.astype(np.float64), 0.0, rul_cap))
        sizes.append(int(len(end_idx)))
    return feats_list, rul_list, sizes


def load_cmapss_reg(
    root_dir: str,
    window: int = 30,
    step: int = 1,
    batch_size: int = 30,
    subsets: Sequence[int] = (1, 2, 3, 4),
    rul_cap: float = RUL_CAP,
) -> List[Batch]:
    """把 train_FD001..FD004 组织成一条漂移流，标签是**连续 RUL**（截断到 ``rul_cap``）。

    切批与 ``load_cmapss`` 逐字一致：每 ``batch_size`` 台发动机一批，子集顺序串联，
    ``t`` 从 1 递增，``name`` 形如 ``FD002#3``，``meta`` 带 subset / engines / n_engines。
    额外带 ``rul_cap`` 与 ``label_clipped_frac``（被截断的样本占比）。
    """
    _check_files(root_dir, "train", subsets)

    batches: List[Batch] = []
    t = 0
    for sub in subsets:
        path = os.path.join(root_dir, f"train_FD00{sub}.txt")
        feats_list, rul_list, _ = _subset_windows_reg(path, window, step, rul_cap)
        n_eng = len(feats_list)
        for g in range(0, n_eng, batch_size):
            X = np.concatenate(feats_list[g:g + batch_size], axis=0)
            y = np.concatenate(rul_list[g:g + batch_size], axis=0)
            t += 1
            clipped = float(np.mean(np.isclose(y, rul_cap)))
            batches.append(Batch(
                t=t, X=X, y=y,
                name=f"FD00{sub}#{g // batch_size + 1}",
                meta={
                    "subset": f"FD00{sub}",
                    "engines": list(range(g + 1, min(g + batch_size, n_eng) + 1)),
                    "n_engines": min(batch_size, n_eng - g),
                    "window": window,
                    "step": step,
                    "rul_cap": rul_cap,
                    "label_clipped_frac": round(clipped, 4),
                },
            ))
    return batches


def load_cmapss_reg_test(
    root_dir: str,
    window: int = 30,
    step: int = 1,
    subsets: Sequence[int] = (1, 2, 3, 4),
    rul_cap: float = RUL_CAP,
) -> List[Batch]:
    """``test_FD00x + RUL_FD00x`` 的连续 RUL 留出批次（每个子集一批）。

    与 ``load_cmapss_test`` 的窗口/对齐方式一致，只把标签换成连续 RUL 并截断。
    FD004 test 249 台 vs RUL 248 个值，取较小值对齐并在 ``meta['aligned']`` 标注。
    """
    _check_files(root_dir, "test", subsets)
    _check_files(root_dir, "RUL", subsets)

    batches: List[Batch] = []
    t = 0
    for sub in subsets:
        arr = _read_table(os.path.join(root_dir, f"test_FD00{sub}.txt"))
        rul_last = np.loadtxt(os.path.join(root_dir, f"RUL_FD00{sub}.txt"),
                              dtype=np.float64, ndmin=1)
        units = arr[:, 0].astype(np.int64)
        slices = _engine_slices(units)
        k = min(len(slices), len(rul_last))
        feats_all, rul_all = [], []
        for i in range(k):
            a, b = slices[i]
            track = arr[a:b, list(SENSOR_COLS)]
            n_cycle = b - a
            feats, end_idx = _window_features(track, window, step)
            if len(end_idx) == 0:
                continue
            rul = rul_last[i] + ((n_cycle - 1) - end_idx)
            feats_all.append(feats)
            rul_all.append(np.clip(rul, 0.0, rul_cap))
        if not feats_all:
            continue
        t += 1
        batches.append(Batch(
            t=t,
            X=np.concatenate(feats_all, axis=0),
            y=np.concatenate(rul_all, axis=0),
            name=f"FD00{sub}-test",
            meta={
                "subset": f"FD00{sub}",
                "n_engines": int(k),
                "engines_in_file": int(len(slices)),
                "rul_values": int(len(rul_last)),
                "aligned": bool(k == len(slices) == len(rul_last)),
                "rul_cap": rul_cap,
            },
        ))
    return batches


# ---------------------------------------------------------------- 模型
def make_regressor(kind: str = "hgb", seed: int = 0, max_iter: int = 120):
    """``hgb``：HistGradientBoosting 回归头；``dummy``：一直预测训练集均值。"""
    if kind == "hgb":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=int(max_iter), max_depth=6, learning_rate=0.1,
                early_stopping=False, random_state=seed,
            ),
        )
    if kind == "dummy":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"未知回归模型: {kind}")


def fit_regressor(model, X: np.ndarray, y: np.ndarray) -> float:
    """拟合并返回耗时（秒）。"""
    t0 = time.perf_counter()
    model.fit(X, y)
    return round(time.perf_counter() - t0, 3)


def predict_regressor(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=np.float64)


# ---------------------------------------------------------------- 回归版记忆
class RegReplayMemory:
    """近期缓冲 + 水库采样（回归版）。

    与 ``dp.memory.ReplayMemory`` 结构相同，但**不做类别均衡采样**——连续 RUL 的
    "类别"几乎每个样本都不同，按类均衡会退化成"全留"，既慢又没意义。这里统一用
    无放回随机下采样，接口保持 recent / reservoir / train_set / probe 四件套。
    """

    def __init__(self, max_recent_batches: int = 4, reservoir_size: int = 2000,
                 seed: int = 0):
        self.max_recent_batches = max_recent_batches
        self.reservoir_size = reservoir_size
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
        """向量化的 Algorithm R：每条样本以 cap/seen 概率替换库中一条。

        与逐条 vstack 的经典写法等价（后写入覆盖先写入），但 5000 条一批也只要
        毫秒级，避免 14 万条样本逐条 vstack 的开销。
        """
        cap = self.reservoir_size
        n = len(y)
        if n == 0:
            return
        seen = self._seen + np.arange(1, n + 1, dtype=np.int64)   # 每条样本的到达序号
        reserve_X = None if self._res_X is None else self._res_X

        if reserve_X is None:
            reserve_X = np.empty((0, X.shape[1]), dtype=np.float64)
            reserve_y = np.empty((0,), dtype=np.float64)
        else:
            reserve_y = self._res_y

        room = cap - len(reserve_y)
        if room >= n:
            reserve_X = np.vstack([reserve_X, X.astype(np.float64, copy=False)])
            reserve_y = np.concatenate([reserve_y, np.asarray(y, dtype=np.float64)])
        else:
            fill = max(0, room)
            if fill > 0:   # 先填满空位，再进入替换阶段
                reserve_X = np.vstack([reserve_X, X[:fill].astype(np.float64, copy=False)])
                reserve_y = np.concatenate([reserve_y, np.asarray(y[:fill], dtype=np.float64)])
            rest_idx = np.arange(fill, n)
            if rest_idx.size:
                s = seen[rest_idx].astype(np.float64)
                accept = self.rng.random(rest_idx.size) < (cap / s)
                acc_idx = rest_idx[accept]
                if acc_idx.size:
                    slots = self.rng.integers(0, cap, size=acc_idx.size)
                    reserve_X[slots] = X[acc_idx]
                    reserve_y[slots] = np.asarray(y, dtype=np.float64)[acc_idx]

        self._res_X, self._res_y = reserve_X, reserve_y
        self._seen += n

    # ---------- 读取 ----------
    @staticmethod
    def _subsample(X: np.ndarray, y: np.ndarray, cap: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        if len(y) <= cap:
            return X, y
        idx = rng.choice(len(y), cap, replace=False)
        return X[idx], y[idx]

    def recent(self, cap: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        if not self.recent_X:
            raise RuntimeError("记忆为空")
        X, y = np.vstack(self.recent_X), np.concatenate(self.recent_y)
        return self._subsample(X, y, cap, self.rng)

    def reservoir(self, cap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self._res_X is None:
            return self.recent()
        cap = cap or len(self._res_y)
        return self._subsample(self._res_X, self._res_y, cap, self.rng)

    def train_set(self, recent_cap: int = 3000, reservoir_cap: int = 1200):
        """训练用：水库样本（旧分布）+ 近期样本（新分布）。"""
        Xr, yr = self.recent(recent_cap)
        Xo, yo = self.reservoir(reservoir_cap)
        return np.vstack([Xo, Xr]), np.concatenate([yo, yr])

    def probe(self, n_old: int = 500, n_new: int = 500):
        """门禁用：旧分布（水库）与新分布（近期）探针，分开返回。"""
        Xo, yo = self.reservoir(n_old)
        try:
            Xn, yn = self.recent(cap=10_000)
        except RuntimeError:
            Xn, yn = Xo, yo
        if len(yn) > n_new:
            Xn, yn = self._subsample(Xn, yn, n_new, self.rng)
        return (Xo, yo), (Xn, yn)

    @property
    def size(self) -> int:
        return 0 if self._res_y is None else len(self._res_y)


# ---------------------------------------------------------------- 策略
@dataclass
class StepResultReg:
    t: int
    n: int
    rmse: float
    mae: float
    phm08: float
    acted: bool
    action: str
    train_seconds: float
    drift_score: float
    promoted: bool
    gate: Optional[Dict[str, object]] = None


class _Strategy:
    name = "base"

    def __init__(self, model_kind: str = "hgb", seed: int = 0, max_iter: int = 120):
        self.model_kind = model_kind
        self.seed = seed
        self.max_iter = max_iter
        self.model = None
        self.train_seconds = 0.0

    def _new_model(self):
        return make_regressor(self.model_kind, self.seed, self.max_iter)

    def initial_fit(self, X, y):
        self.model = self._new_model()
        self.train_seconds = fit_regressor(self.model, X, y)

    def on_batch(self, b: Batch, rep: DriftReport) -> StepResultReg:
        raise NotImplementedError


class StaticBaselineReg(_Strategy):
    """部署后不再更新：官方静态基线。"""

    name = "static"

    def on_batch(self, b, rep):
        m = evaluate_regression(b.y, predict_regressor(self.model, b.X))
        return StepResultReg(b.t, len(b), m.rmse, m.mae, m.phm08,
                             False, "none", 0.0, rep.score, False)


class DriftTriggeredReg(_Strategy):
    """漂移触发才重训：训练集 = 水库（旧分布）+ 近期（新分布）+ 当前批。"""

    name = "drift_triggered"

    def __init__(self, model_kind="hgb", seed=0, max_iter=120,
                 max_recent_batches=4, reservoir_size=2000,
                 recent_cap=3000, reservoir_cap=1200):
        super().__init__(model_kind, seed, max_iter)
        self.recent_cap = recent_cap
        self.reservoir_cap = reservoir_cap
        self.memory = RegReplayMemory(max_recent_batches=max_recent_batches,
                                      reservoir_size=reservoir_size, seed=seed)

    def initial_fit(self, X, y):
        super().initial_fit(X, y)
        self.memory.add(X, y)

    def _training_data(self, b: Batch):
        Xr, yr = self.memory.train_set(self.recent_cap, self.reservoir_cap)
        return np.vstack([Xr, b.X]), np.concatenate([yr, b.y])

    def on_batch(self, b, rep):
        m = evaluate_regression(b.y, predict_regressor(self.model, b.X))  # 先预测
        acted, action, secs = False, "none", 0.0
        if rep.triggered:
            Xt, yt = self._training_data(b)
            cand = self._new_model()
            secs = fit_regressor(cand, Xt, yt)
            self.model = cand
            acted, action = True, "replay_posttrain"
            self.memory.add(b.X, b.y)
        return StepResultReg(b.t, len(b), m.rmse, m.mae, m.phm08,
                             acted, action, secs, rep.score, acted)


class GuardedDriftTriggeredReg(DriftTriggeredReg):
    """漂移触发 + 双探针门禁（回归版：用 RMSE 判断"是否退化"）。

    候选模型必须同时满足：
      ① **旧分布探针**（初始训练数据里留出的 15%，三策略都不拿它训练）上，
         RMSE 不比现役模型差超过 ``old_tol`` 个循环——防遗忘早期子集；
      ② **新分布探针**（当前批留出的 20%，不参与候选训练）上，
         RMSE 不比现役模型差超过 ``new_tol`` 个循环——防"越改越差"。
    两项都过才替换现役模型，否则回滚（``rejected_by_gate``）并记下四项探针 RMSE。

    为什么探针要留出：``dp.memory.ReplayMemory.probe`` 的探针取自训练水库，
    现役模型在探针上有"记忆性"优势，任何被新数据稀释的候选都必然在探针上退化，
    门禁于是 100% 拒绝、退化成静态基线（分类通路实测即如此）。把探针换成
    真正留出的样本，比较的才是泛化能力，而不是"谁背得更熟"。
    """

    name = "guarded"

    def __init__(self, model_kind="hgb", seed=0, max_iter=120,
                 max_recent_batches=4, reservoir_size=2000,
                 recent_cap=3000, reservoir_cap=1200,
                 old_tol: float = 1.5, new_tol: float = 0.5,
                 old_probe: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 old_probe_frac: float = 0.15, new_probe_frac: float = 0.20):
        super().__init__(model_kind, seed, max_iter, max_recent_batches,
                         reservoir_size, recent_cap, reservoir_cap)
        self.old_tol = old_tol      # 旧分布允许的最大 RMSE 涨幅（循环）
        self.new_tol = new_tol      # 新分布允许的最大 RMSE 涨幅（循环）
        self.old_probe = old_probe
        self.old_probe_frac = old_probe_frac
        self.new_probe_frac = new_probe_frac

    def initial_fit(self, X, y):
        if self.old_probe is None:
            # 未从外部传入时自行留出 15%（此时只有 guarded 少用这部分数据，仅作兜底）
            n = len(y)
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            k = max(200, int(self.old_probe_frac * n))
            pidx, fidx = np.sort(perm[:k]), np.sort(perm[k:])
            self.old_probe = (X[pidx].copy(), y[pidx].copy())
            X, y = X[fidx], y[fidx]
        super().initial_fit(X, y)

    def _split_batch(self, b: Batch) -> Tuple[np.ndarray, np.ndarray]:
        """当前批切成（新分布探针, 候选训练用）两份，探针永不进入记忆。"""
        rng = np.random.default_rng(10_000 + b.t)
        idx = rng.permutation(len(b.y))
        k = max(200, int(self.new_probe_frac * len(b.y)))
        return np.sort(idx[:k]), np.sort(idx[k:])

    def on_batch(self, b, rep):
        m = evaluate_regression(b.y, predict_regressor(self.model, b.X))
        if not rep.triggered:
            return StepResultReg(b.t, len(b), m.rmse, m.mae, m.phm08,
                                 False, "none", 0.0, rep.score, False)

        gidx, tidx = self._split_batch(b)      # 新分布探针 / 候选训练
        Xr, yr = self.memory.train_set(self.recent_cap, self.reservoir_cap)
        Xb = np.vstack([Xr, b.X[tidx]])
        yb = np.concatenate([yr, b.y[tidx]])
        cand = self._new_model()
        secs = fit_regressor(cand, Xb, yb)

        Xo, yo = self.old_probe
        old_before = evaluate_regression(yo, predict_regressor(self.model, Xo)).rmse
        old_after = evaluate_regression(yo, predict_regressor(cand, Xo)).rmse
        new_before = evaluate_regression(b.y[gidx], predict_regressor(self.model, b.X[gidx])).rmse
        new_after = evaluate_regression(b.y[gidx], predict_regressor(cand, b.X[gidx])).rmse

        ok_old = bool(old_after <= old_before + self.old_tol)
        ok_new = bool(new_after <= new_before + self.new_tol)
        promoted = bool(ok_old and ok_new)
        if promoted:
            self.model = cand
            self.memory.add(b.X[tidx], b.y[tidx])
        gate = {
            "old_before": round(float(old_before), 4),
            "old_after": round(float(old_after), 4),
            "new_before": round(float(new_before), 4),
            "new_after": round(float(new_after), 4),
            "ok_old": ok_old, "ok_new": ok_new,
            "n_old_probe": int(len(yo)), "n_new_probe": int(gidx.size),
        }
        return StepResultReg(b.t, len(b), m.rmse, m.mae, m.phm08, True,
                             "replay_posttrain" if promoted else "rejected_by_gate",
                             secs, rep.score, promoted, gate)


def run_stream_regression(
    batches: List[Batch],
    strategy: _Strategy,
    monitor: Monitor,
    base_count: int = 2,
    perf_drop: float = 1.0,
) -> List[StepResultReg]:
    """前 ``base_count`` 批初始训练，其后逐批"先预测、后决策"。

    与 ``dp.stream.run_stream`` 语义一致；性能信号传**负 RMSE**（越大越好），
    因此 ``perf_drop`` 的单位是"RMSE 涨了多少个循环才触发"，默认 1.0。
    """
    init_X = np.vstack([b.X for b in batches[:base_count]])
    init_y = np.concatenate([b.y for b in batches[:base_count]])
    strategy.initial_fit(init_X, init_y)
    monitor.set_reference(init_X)
    monitor.perf_drop = perf_drop

    ref_rmse = evaluate_regression(init_y, predict_regressor(strategy.model, init_X)).rmse
    last_rmse = ref_rmse
    results: List[StepResultReg] = []
    for b in batches[base_count:]:
        rep = monitor.observe(b.t, b.X, strategy.model,
                              last_acc=-last_rmse, ref_acc=-ref_rmse)
        res = strategy.on_batch(b, rep)
        last_rmse = res.rmse
        if res.promoted:   # 模型更新成功后，把参考性能抬到当前水平
            ref_rmse = min(ref_rmse, res.rmse)
        results.append(res)
    return results
