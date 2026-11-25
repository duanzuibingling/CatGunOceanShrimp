"""评测指标体系：按赛题公布的 6.1 节，一次实验必报 7 项。

1. 流上平均性能        mean_perf
2. 最差窗口性能        worst_window（滑动窗口内的最低均值，看鲁棒性）
3. 恢复时延            adaptation_latency（漂移后回到基线以上所需批数）
4. 遗忘度              forgetting（相对历史峰值的平均跌落，看会不会学了新的忘了旧的）
5. 成本三项            gpu_sec / labels / tokens
6. 单位增益成本        gain_per_cost（核心 KPI）
7. 稳定性              stability（3 个 seed 的标准差）

说明：官方口径写的是「GPU 秒」，我们环境无 GPU，用 **CPU 训练秒** 等价替代，
并在报告里注明口径差异——这比含糊其辞更可信。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Cost:
    train_seconds: float = 0.0      # 等价 GPU 秒（CPU 计时）
    labels: int = 0                 # 标注配额消耗（本赛题标签免费，恒为 0）
    tokens: int = 0                 # 大模型 Token

    @property
    def total(self) -> float:
        # 权重可调：1 秒训练 ≈ 1 token 的千分之一（经验折算，答辩时说明）
        return self.train_seconds + self.tokens / 1000.0 + self.labels * 1.0


def worst_window(scores: Sequence[float], w: int = 3) -> float:
    """最差窗口：任意连续 w 批的平均性能里的最小值。"""
    s = list(scores)
    if len(s) < w:
        return float(np.min(s)) if s else float("nan")
    return float(min(np.mean(s[i:i + w]) for i in range(len(s) - w + 1)))


def adaptation_latency(ours: Sequence[float], baseline: Sequence[float],
                       drop: float = 0.05, recover_margin: float = 0.02,
                       cap: int = 10) -> Dict[str, float]:
    """漂移后，用多少批把性能拉回到「漂移前基线水平」。

    漂移事件由基线曲线的骤降定义（基线下滑 > drop）。
    """
    lat = []
    events = []
    for t in range(1, len(baseline)):
        if baseline[t - 1] - baseline[t] > drop:
            events.append(t)
            target = baseline[t - 1] - recover_margin
            k = None
            for j in range(t, min(len(ours), t + cap + 1)):
                if ours[j] >= target:
                    k = j - t
                    break
            lat.append(k if k is not None else cap)
    return {"events": len(events), "latency": float(np.mean(lat)) if lat else 0.0,
            "latencies": lat}


def forgetting(ours: Sequence[float], window: int = 5) -> float:
    """遗忘度：运行峰值与当前值的平均差（只看后半程，避免起步波动）。"""
    s = list(ours)
    if len(s) < window + 1:
        return 0.0
    peak = -1.0
    gaps = []
    start = len(s) // 2
    for i, v in enumerate(s):
        peak = max(peak, v)
        if i >= start:
            gaps.append(max(0.0, peak - v))
    return float(np.mean(gaps)) if gaps else 0.0


@dataclass
class RunMetrics:
    name: str
    mean_perf: float
    worst_window: float
    adaptation_events: int
    adaptation_latency: float
    forgetting: float
    cost: Dict[str, float]
    per_batch: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_run(name: str, scores: Sequence[float], cost: Cost,
                 baseline_scores: Optional[Sequence[float]] = None) -> RunMetrics:
    lat = (adaptation_latency(scores, baseline_scores)
           if baseline_scores is not None else
           {"events": 0, "latency": 0.0})
    return RunMetrics(
        name=name,
        mean_perf=round(float(np.mean(scores)), 4) if len(scores) else float("nan"),
        worst_window=round(worst_window(scores), 4),
        adaptation_events=int(lat["events"]),
        adaptation_latency=round(float(lat["latency"]), 2),
        forgetting=round(forgetting(scores), 4),
        cost={"train_seconds": round(cost.train_seconds, 2),
              "labels": int(cost.labels), "tokens": int(cost.tokens)},
        per_batch=[round(float(x), 4) for x in scores],
    )


def aggregate(runs: List[RunMetrics], baseline: Optional[RunMetrics] = None) -> dict:
    """把 3 个 seed 的同一策略合成一行：均值 ± 标准差 + 单位增益成本。"""
    if not runs:
        return {}
    means = np.array([r.mean_perf for r in runs])
    costs = np.array([r.cost["train_seconds"] + r.cost["tokens"] / 1000.0 for r in runs])
    out = {
        "n_seeds": len(runs),
        "mean_perf": round(float(means.mean()), 4),
        "std_perf": round(float(means.std(ddof=0)), 4),
        "worst_window": round(float(np.mean([r.worst_window for r in runs])), 4),
        "adaptation_latency": round(float(np.mean([r.adaptation_latency for r in runs])), 2),
        "adaptation_events": int(np.mean([r.adaptation_events for r in runs])),
        "forgetting": round(float(np.mean([r.forgetting for r in runs])), 4),
        "train_seconds": round(float(costs.mean()), 2),
        "tokens": int(np.mean([r.cost["tokens"] for r in runs])),
        "labels": int(np.mean([r.cost["labels"] for r in runs])),
    }
    if baseline is not None:
        gain = out["mean_perf"] - baseline.mean_perf
        out["gain_vs_baseline"] = round(float(gain), 4)
        out["gain_per_cost"] = round(float(gain / max(1e-6, costs.mean())), 5)
    return out
