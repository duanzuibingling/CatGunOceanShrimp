"""XJTU-SY 滚动轴承全寿命数据集适配器。

数据来源（详见 docs/08-XJTU接入.md）：
    Wang Biao et al., XJTU-SY Bearing Datasets (西安交通大学 & 浙江长兴异阳机械装备健康监测
    联合实验室)，15 个 LDK UER204 轴承在 3 种工况下加速运行至失效，每个采样文件 1.28 s、
    25.6 kHz、水平/垂直两通道。
    本次实验用的是国内镜像 PHMbench/PHM-Vibench 的 RM_002_XJTU.h5（ModelScope），
    从中抽取 3 个轴承的原始片段落成紧凑子集，不下载全量 CSV。

契约（复用 src/dp/data.py，不修改）：
    Batch = (t: int, X: np.ndarray[n, d], y: np.ndarray[n], name: str)
    返回 list[Batch]，按时间顺序排列。

时间语义：
    一个 CSV/片段 = 一个"时间点"（轴承运行 1.28 s 的观测）。
    每个时间点内部再切 `segments` 个子窗，每个子窗出一行特征，
    同一时间点的所有行共享同一个 RUL 标签。

标签（近似，非官方标注）：
    以"距失效还剩多少寿命的比例"分箱，5 类：
        0 → 剩余 0%~10%（濒临失效）
        1 → 剩余 10%~30%
        2 → 剩余 30%~50%
        3 → 剩余 50%~70%
        4 → 剩余 70%~100%（健康，刚上机）
    这是把"退化程度"当成分类问题做的近似标注，真实工程里 RUL 是回归量。

支持两种目录布局（layout="auto" 自动识别）：
    1) 官方 CSV 布局：  <root>/<BearingX_Y>/<n>.csv      （两列：水平、垂直）
       也接受 <root>/<工况>/<BearingX_Y>/<n>.csv
    2) 紧凑子集布局：  <root>/<BearingX_Y>/signal.npy + meta.json
       signal.npy: int16/float32, 形状 (n_time, n_samples, 2)，可用 mmap 低内存按行读。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..data import Batch

# ---------------------------------------------------------------- 常量
FS_HZ = 25600.0          # 官方采样率 25.6 kHz（README 亦写 25.6kHz / 1.28s / 32768 点）
SEG_SAMPLES = 4096       # 每个子窗 4096 点 = 0.16 s
N_BANDS = 8              # FFT 频带数，每带 1.6 kHz
FEATURE_NAMES = (
    ["h_rms", "h_p2p", "h_kurt", "h_skew", "h_crest"]
    + ["h_band%d" % i for i in range(N_BANDS)]
    + ["v_rms", "v_p2p", "v_kurt", "v_skew", "v_crest"]
    + ["v_band%d" % i for i in range(N_BANDS)]
)
N_FEATURES = len(FEATURE_NAMES)   # 26 = (5 时域 + 8 频带) x 2 通道

RUL_BIN_EDGES = (0.0, 0.10, 0.30, 0.50, 0.70, 1.0001)
RUL_BIN_NAMES = ("0-10%", "10-30%", "30-50%", "50-70%", "70-100%")


# ---------------------------------------------------------------- 特征
def _channel_features(x: np.ndarray) -> np.ndarray:
    """单通道一维信号的 13 维特征：5 时域 + 8 频带相对能量。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.empty(5 + N_BANDS, dtype=np.float64)

    mean = x.mean()
    xc = x - mean
    std = xc.std()
    rms = float(np.sqrt(np.mean(x * x)))
    vmax = float(np.max(np.abs(x)))
    # 用标准差归一化的三/四阶矩，省掉 scipy.stats 的每次调用开销
    if std > 1e-12:
        s3 = float(np.mean(xc ** 3) / std ** 3)
        s4 = float(np.mean(xc ** 4) / std ** 4)
    else:
        s3 = s4 = 0.0
    out[0] = rms
    out[1] = float(x.max() - x.min())
    out[2] = s4                      # 峭度（非超额，正态≈3）
    out[3] = s3                      # 偏度
    out[4] = vmax / rms if rms > 1e-12 else 0.0   # 峰值因子

    # 频域：Hann 窗 + rFFT，按 8 个等宽频带累计功率占比
    win = np.hanning(n)
    spec = np.fft.rfft(xc * win)
    power = spec.real ** 2 + spec.imag ** 2
    total = power.sum()
    if total <= 0:
        out[5:] = 0.0
        return out
    nb = power.size
    edges = np.linspace(0, nb, N_BANDS + 1).astype(int)
    bands = np.empty(N_BANDS, dtype=np.float64)
    for i in range(N_BANDS):
        s, e = edges[i], max(edges[i] + 1, edges[i + 1])
        bands[i] = power[s:e].sum()
    out[5:] = bands / total
    return out


def segment_features(seg: np.ndarray) -> np.ndarray:
    """一段两通道信号 (n_samples, 2) → 26 维特征。"""
    seg = np.asarray(seg, dtype=np.float64)
    if seg.ndim == 1:
        seg = seg[:, None]
    return np.concatenate([_channel_features(seg[:, 0]), _channel_features(seg[:, 1])])


def _split_segments(x: np.ndarray, segments: int) -> List[np.ndarray]:
    """把一条 (L, 2) 的时间序列等分成 segments 个子窗，丢弃不足的尾段。"""
    L = x.shape[0]
    k = L // segments
    if k < 64:                      # 太短就不切，整段当一个样本
        return [x]
    return [x[i * k:(i + 1) * k] for i in range(segments)]


# ---------------------------------------------------------------- 标签
def rul_labels(n_time: int) -> np.ndarray:
    """按"剩余寿命比例"给一条全寿命序列打 5 类标签。

    i 从 0（上机）到 n_time-1（失效前最后一个文件）：
        remaining_frac = (n_time - 1 - i) / max(1, n_time - 1)
    分箱见 RUL_BIN_EDGES，0 类最接近失效、4 类最健康。
    """
    i = np.arange(n_time, dtype=np.float64)
    frac = (n_time - 1 - i) / max(1, n_time - 1)
    lab = np.digitize(frac, RUL_BIN_EDGES[1:-1], right=False)
    # digitize 给的是 0..4，0 对应 frac 最小 = 最接近失效
    return lab.astype(np.int64)


# ---------------------------------------------------------------- 读信号
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _sniff_sep(path: str) -> str:
    """只看文件开头几 KB 判断分隔符（逗号/制表符/分号/空白），避免整文件嗅探。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    line = next((ln for ln in head.splitlines() if ln.strip()), "")
    for cand in (",", "\t", ";", "|"):
        if cand in line:
            return cand
    return r"\s+"


def _read_csv_signal(path: str, max_rows: Optional[int] = None,
                     decimate: int = 1) -> np.ndarray:
    """低内存读一个 XJTU-SY 采样 CSV：只取前两列（水平/垂直），单精度。

    pandas 的 C 解析器 + usecols 只物化需要的两列；decimate>1 时按步长抽样，
    用于内存/耗时敏感的场景（默认不抽样，官方文件只有 32768 行）。
    """
    import pandas as pd

    sep = _sniff_sep(path)
    df = pd.read_csv(
        path, header=None, usecols=[0, 1], dtype=np.float64,
        nrows=max_rows, engine="c", sep=sep,
        comment="#", on_bad_lines="skip",
    )
    arr = df.to_numpy(dtype=np.float64)
    if decimate > 1:
        arr = arr[::decimate]
    return arr


def _bearing_dirs(root: str) -> List[str]:
    """找所有 'BearingX_Y' 目录（允许中间夹一层工况目录）。"""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if re.fullmatch(r"[Bb]earing\d+_\d+", base):
            if any(f.lower().endswith(".csv") for f in filenames) or \
                    "signal.npy" in filenames:
                found.append(dirpath)
                dirnames[:] = []
    return sorted(found, key=lambda p: _natural_key(os.path.basename(p)))


def _load_bearing(path: str, max_rows: Optional[int], decimate: int,
                  segments: int) -> tuple[str, List[np.ndarray], dict]:
    """读一个轴承的全部时间点，返回 (名字, 每时间点的 26 维特征矩阵列表, 元信息)。"""
    base = os.path.basename(path)
    meta = {"bearing": base, "dir": path}
    feats: List[np.ndarray] = []

    npy = os.path.join(path, "signal.npy")
    if os.path.exists(npy):
        arr = np.load(npy, mmap_mode="r")      # mmap：不把整轴承读进内存
        meta["layout"] = "npy"
        meta["n_time"] = int(arr.shape[0])
        meta["n_samples"] = int(arr.shape[1])
        meta["dtype"] = str(arr.dtype)
        mj = os.path.join(path, "meta.json")
        if os.path.exists(mj):
            with open(mj, encoding="utf-8") as f:
                meta.update(json.load(f))
        for i in range(arr.shape[0]):
            sig = np.asarray(arr[i], dtype=np.float64)
            if decimate > 1:
                sig = sig[::decimate]
            rows = [segment_features(s) for s in _split_segments(sig, segments)]
            feats.append(np.vstack(rows))
        return base, feats, meta

    files = sorted((f for f in os.listdir(path) if f.lower().endswith(".csv")),
                   key=_natural_key)
    meta["layout"] = "csv"
    meta["n_time"] = len(files)
    for fn in files:
        sig = _read_csv_signal(os.path.join(path, fn), max_rows=max_rows, decimate=decimate)
        rows = [segment_features(s) for s in _split_segments(sig, segments)]
        feats.append(np.vstack(rows))
    return base, feats, meta


# ---------------------------------------------------------------- 主入口
@dataclass
class XjtuMeta:
    bearings: List[dict]
    fs_hz: float
    n_features: int
    n_time_total: int


def load_xjtu(
    root_dir: str,
    files_per_batch: int = 16,
    segments: int = 8,
    max_rows: Optional[int] = None,
    decimate: int = 1,
    max_bearings: Optional[int] = None,
) -> List[Batch]:
    """把 XJTU-SY 子集读成按时间顺序排列的 Batch 列表。

    Args:
        root_dir:        数据根目录（官方 CSV 布局或紧凑 npy 子集布局均自动识别）
        files_per_batch: 每多少个"时间点（CSV）"打包成一个 Batch
        segments:        每个时间点内部切多少个子窗（每子窗一行特征）
        max_rows:        单个 CSV 最多读多少行（None=全读；低内存/快速试跑可用）
        decimate:        时间轴降采样步长（1=不抽）
        max_bearings:    只用前几个轴承（调试用）

    Returns:
        list[Batch]，t 从 1 递增，name 形如 "Bearing1_1#03"。
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"XJTU 数据目录不存在：{root_dir}")
    dirs = _bearing_dirs(root_dir)
    if not dirs:
        raise FileNotFoundError(
            f"{root_dir} 下没有找到 BearingX_Y 目录（需要 *.csv 或 signal.npy）"
        )
    if max_bearings:
        dirs = dirs[:max_bearings]

    batches: List[Batch] = []
    bearing_meta: List[dict] = []
    t = 0
    n_time_total = 0
    for d in dirs:
        name, feats, meta = _load_bearing(d, max_rows, decimate, segments)
        n_time = len(feats)
        labels = rul_labels(n_time)
        meta["segments"] = segments
        meta["label_counts"] = {RUL_BIN_NAMES[c]: int((labels == c).sum())
                                for c in range(len(RUL_BIN_NAMES))}
        meta["n_time"] = n_time
        bearing_meta.append(meta)
        n_time_total += n_time

        # 按时间顺序，每 files_per_batch 个时间点一个 Batch
        for s in range(0, n_time, files_per_batch):
            e = min(s + files_per_batch, n_time)
            if e - s == 0:
                continue
            t += 1
            X = np.vstack(feats[s:e])
            y = np.repeat(labels[s:e], [len(f) for f in feats[s:e]])
            batches.append(Batch(
                t=t, X=X, y=y,
                name=f"{name}#{t:02d}",
                meta={
                    "bearing": name,
                    "t_start": s,
                    "t_end": e - 1,
                    "n_time": e - s,
                    "layout": meta["layout"],
                },
            ))

    # 把元信息挂到每个 Batch 上，方便 run 脚本写 summary
    info = XjtuMeta(bearings=bearing_meta, fs_hz=FS_HZ, n_features=N_FEATURES,
                    n_time_total=n_time_total)
    for b in batches:
        b.meta["xjtu"] = {
            "root": os.path.abspath(root_dir),
            "fs_hz": FS_HZ,
            "segments": segments,
            "files_per_batch": files_per_batch,
            "n_bearings": len(bearing_meta),
            "n_time_total": n_time_total,
        }
    load_xjtu.last_info = info  # 供脚本读元信息（不修改 Batch 契约）
    return batches


load_xjtu.last_info = None


def describe(batches: Sequence[Batch]) -> str:
    """一行式清单，方便脚本打印。"""
    if not batches:
        return "空数据流"
    X = batches[0].X
    rows = ["%-14s n=%4d dim=%d 类别=%s" % (b.name, len(b), X.shape[1],
                                          sorted(set(b.y.tolist()))) for b in batches]
    return "\n".join(rows)


if __name__ == "__main__":  # 手工抽查：python -m dp.adapters.xjtu <dir>
    import sys

    bs = load_xjtu(sys.argv[1] if len(sys.argv) > 1 else r"D:\竞赛\data\raw\xjtu")
    print(describe(bs))
