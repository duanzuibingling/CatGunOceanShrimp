"""C-MAPSS（NASA 商用模块化航空推进系统仿真）涡扇发动机退化数据适配器。

统一契约与 ``dp.data`` 一致：返回 ``list[Batch]``，``t`` 从 1 开始按时间递增。

切批逻辑
--------
NASA C-MAPSS 的 FD001–FD004 是四组「run-to-failure」多变量时间序列，每个子集内部
工况与故障模式固定，子集之间变化：

    FD001  1 种工况  1 种故障（HPC 退化）    100 台train
    FD002  6 种工况  1 种故障（HPC 退化）    260 台train
    FD003  1 种工况  2 种故障（HPC+Fan）     100 台train
    FD004  6 种工况  2 种故障（HPC+Fan）     248 台train

因此把 FD001→FD002→FD003→FD004 按顺序串成一条流，**子集边界就是一次真实的
协变量/概念漂移**（工况数变化导致特征分布平移，故障模式变化导致 P(y|x) 改变）。
每个子集内部再按发动机编号顺序，每 ``batch_size``（默认 30）台发动机合成一批，
发动机内部严格保持循环（cycle）先后顺序。这样 4 个子集共切成约 26 批，
既保留了时间顺序，也让每批有足够的样本训练/评估。

特征
----
原始每行 26 列：``unit``、``cycle``、3 个操作条件、21 路传感器 + 1 个占位。
本适配器按题目约定取**第 3–26 列（0 基下标 2..25，共 24 列）**作为特征源，
做窗口 ``window``=30、步长 ``step``=1 的滑窗，每个窗口算 4 组统计量：

    均值 / 标准差 / 斜率（对窗口内 cycle 做最小二乘线性拟合） / 最后值

拼接成 24×4 = 96 维定长向量。

标签（近似，务必留意）
----------------------
RUL（剩余寿命）本身是**回归值**，这里为了复用现有分类管线强行分箱成 5 类：

    RUL ∈ (-inf, 20] → 0    (20, 40] → 1    (40, 60] → 2    (60, 80] → 3    (80, +inf) → 4

这是**近似**：同档内的 RUL 差异被抹平，相邻档的误判与差 100 个循环的误判同罚，
所以本文报告的"准确率"含义是"RUL 档位判断对了"，**不等于** RUL 回归指标
（RMSE / 评分函数）。要报 RMSE 需另接回归头，不在本适配器范围内。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data import Batch

__all__ = [
    "RUL_BINS",
    "LABEL_NAMES",
    "SENSOR_COLS",
    "load_cmapss",
    "load_cmapss_test",
    "rul_to_class",
]

#: RUL 分箱边界（含右端点归属左侧档）
RUL_BINS: Tuple[int, ...] = (20, 40, 60, 80)
#: 5 个档位的中文名，画图/写文档时用
LABEL_NAMES: Tuple[str, ...] = ("0-20", "20-40", "40-60", "60-80", ">80")
#: 参与特征计算的列（0 基）：第 3–26 列
SENSOR_COLS: Tuple[int, ...] = tuple(range(2, 26))
#: 子集定义：编号 → 中文/英文说明
SUBSET_INFO: Dict[int, Dict[str, str]] = {
    1: {"cond": "6→1", "fault": "HPC", "note": "1 工况 / 1 故障 / 100 台"},
    2: {"cond": "6", "fault": "HPC", "note": "6 工况 / 1 故障 / 260 台"},
    3: {"cond": "1", "fault": "HPC+Fan", "note": "1 工况 / 2 故障 / 100 台"},
    4: {"cond": "6", "fault": "HPC+Fan", "note": "6 工况 / 2 故障 / 248 台"},
}


# ---------------------------------------------------------------- 基础读取
def _read_table(path: str) -> np.ndarray:
    """读空格分隔的 C-MAPSS 表（无表头、行尾有填充空格）。"""
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape[1] != 26:
        raise ValueError(f"{os.path.basename(path)} 期望 26 列，实际 {arr.shape[1]} 列")
    return arr


def _check_files(root_dir: str, kind: str, subsets: Sequence[int]) -> List[str]:
    paths = []
    for sub in subsets:
        p = os.path.join(root_dir, f"{kind}_FD00{sub}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"缺少 {p}。请把 C-MAPSS 的 train_FD00x.txt / test_FD00x.txt / "
                f"RUL_FD00x.txt 放到 {root_dir} 下。"
            )
        paths.append(p)
    return paths


# ---------------------------------------------------------------- 特征
def _window_features(track: np.ndarray, window: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """单台发动机的滑窗统计特征。

    参数
    ----
    track : (n_cycle, d) 按循环先后排列的传感器矩阵
    window, step : 窗口长度 / 步长

    返回
    ----
    feats : (n_win, 4*d) 均值|标准差|斜率|最后值 拼接
    end_idx : (n_win,) 每个窗口结束位置在 track 中的下标（用于算 RUL）
    """
    n, d = track.shape
    if n < window:
        return np.empty((0, 4 * d), dtype=np.float32), np.empty((0,), dtype=int)

    starts = np.arange(0, n - window + 1, step)
    end_idx = starts + window - 1
    # (n_win, window, d)：单台发动机最多几百个窗口，内存可控
    win = track[starts[:, None] + np.arange(window)[None, :], :]

    t = np.arange(window, dtype=np.float64)
    t = t - t.mean()
    denom = float((t ** 2).sum())

    win64 = win.astype(np.float64, copy=False)
    mean = win64.mean(axis=1)
    std = win64.std(axis=1)
    slope = np.tensordot(win64, t, axes=([1], [0])) / denom   # (n_win, d)
    last = win64[:, -1, :]

    feats = np.concatenate([mean, std, slope, last], axis=1)
    return feats.astype(np.float32), end_idx


def rul_to_class(rul: np.ndarray) -> np.ndarray:
    """把连续 RUL 映射到 5 个档位（近似标签，见模块 docstring）。"""
    return np.digitize(np.asarray(rul, dtype=np.float64), RUL_BINS, right=True).astype(int)


def _engine_slices(units: np.ndarray) -> List[Tuple[int, int]]:
    """按 unit 列切出连续区间；文件本身按发动机分组写，这里做一次稳健性校验。"""
    if len(units) == 0:
        return []
    change = np.flatnonzero(np.diff(units) != 0) + 1
    bounds = np.concatenate([[0], change, [len(units)]])
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def _subset_windows(path: str, window: int, step: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """读一个子集，返回每台发动机的 (特征, 标签, 窗口数)。"""
    arr = _read_table(path)
    units = arr[:, 0].astype(np.int64)
    feats_list, label_list, sizes = [], [], []
    for a, b in _engine_slices(units):
        track = arr[a:b, list(SENSOR_COLS)]
        n_cycle = b - a
        feats, end_idx = _window_features(track, window, step)
        if len(end_idx) == 0:          # 轨迹短于窗口，跳过（FD001–FD004 不会出现）
            continue
        rul = (n_cycle - 1) - end_idx  # 窗口末端距离失效的循环数
        feats_list.append(feats)
        label_list.append(rul_to_class(rul))
        sizes.append(int(len(end_idx)))
    return feats_list, label_list, sizes


# ---------------------------------------------------------------- 对外接口
def load_cmapss(
    root_dir: str,
    window: int = 30,
    step: int = 1,
    batch_size: int = 30,
    subsets: Sequence[int] = (1, 2, 3, 4),
) -> List[Batch]:
    """把 train_FD001..FD004 组织成一条按时间排序的漂移流。

    参数
    ----
    root_dir : 存放 ``train_FD00x.txt`` 的目录
    window   : 滑窗长度（循环数），默认 30
    step     : 滑窗步长，默认 1
    batch_size : 每批包含多少台发动机，默认 30
    subsets  : 参与拼接的子集编号，默认 (1,2,3,4)

    返回
    ----
    list[Batch]，``t`` 从 1 递增，``name`` 形如 ``FD002#3``，
    ``meta`` 里带 subset / engines / n_engines。
    """
    _check_files(root_dir, "train", subsets)

    batches: List[Batch] = []
    t = 0
    for sub in subsets:
        path = os.path.join(root_dir, f"train_FD00{sub}.txt")
        feats_list, label_list, _ = _subset_windows(path, window, step)
        n_eng = len(feats_list)
        for g in range(0, n_eng, batch_size):
            X = np.concatenate(feats_list[g:g + batch_size], axis=0)
            y = np.concatenate(label_list[g:g + batch_size], axis=0)
            t += 1
            batches.append(Batch(
                t=t, X=X, y=y,
                name=f"FD00{sub}#{g // batch_size + 1}",
                meta={
                    "subset": f"FD00{sub}",
                    "engines": list(range(g + 1, min(g + batch_size, n_eng) + 1)),
                    "n_engines": min(batch_size, n_eng - g),
                    "window": window,
                    "step": step,
                },
            ))
    return batches


def load_cmapss_test(
    root_dir: str,
    window: int = 30,
    step: int = 1,
    subsets: Sequence[int] = (1, 2, 3, 4),
) -> List[Batch]:
    """把 test_FD00x.txt + RUL_FD00x.txt 组织成留出评估批次（每个子集一批）。

    test 轨迹在失效前被截断，最后一行对应的 RUL 由 ``RUL_FD00x.txt`` 给出，
    因此窗口末端 e 的 RUL = ``rul_last + (last_cycle - e)``。
    FD004 官方文件里 test 有 249 台而 RUL 只有 248 个值，按两者较小值对齐，
    并在 ``meta['aligned']`` 里标注，避免静默错位。
    """
    _check_files(root_dir, "test", subsets)
    _check_files(root_dir, "RUL", subsets)

    batches: List[Batch] = []
    t = 0
    for sub in subsets:
        arr = _read_table(os.path.join(root_dir, f"test_FD00{sub}.txt"))
        rul_last = np.loadtxt(os.path.join(root_dir, f"RUL_FD00{sub}.txt"), dtype=np.float64, ndmin=1)
        units = arr[:, 0].astype(np.int64)
        slices = _engine_slices(units)
        k = min(len(slices), len(rul_last))
        feats_all, label_all = [], []
        for i in range(k):
            a, b = slices[i]
            track = arr[a:b, list(SENSOR_COLS)]
            n_cycle = b - a
            feats, end_idx = _window_features(track, window, step)
            if len(end_idx) == 0:
                continue
            rul = rul_last[i] + ((n_cycle - 1) - end_idx)
            feats_all.append(feats)
            label_all.append(rul_to_class(rul))
        if not feats_all:
            continue
        t += 1
        batches.append(Batch(
            t=t,
            X=np.concatenate(feats_all, axis=0),
            y=np.concatenate(label_all, axis=0),
            name=f"FD00{sub}-test",
            meta={
                "subset": f"FD00{sub}",
                "n_engines": int(k),
                "engines_in_file": int(len(slices)),
                "rul_values": int(len(rul_last)),
                "aligned": bool(k == len(slices) == len(rul_last)),
            },
        ))
    return batches
