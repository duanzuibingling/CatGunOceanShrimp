"""余弦退火长训（B 任务专用）。

来源：队友 B 闭环包的 fit_variant（余弦退火 + 梯度裁剪 + Adam）在 public/train 上
单发 600 epoch，公开验证集 RMSE 359.4（官方 681.2、我们此前交付 425.33）。
官方训练器是恒定学习率，600 epoch 只有 467；差异完全来自退火调度。

本模块只保留训练数学（约 40 行），不引入队友的整套闭环引擎；
checkpoint 由官方 baselines.Model 写出，格式与平台完全兼容。
"""
from __future__ import annotations

import math

import numpy as np


def cosine_train(model, x, y, epochs=600, lr_start=0.025, lr_end=0.00125,
                 batch=126, seed=42, clip=5.0, groups=None, warmup=2, target_cap=None):
    """在 model 上做 epochs 轮余弦退火训练（原地更新权重），返回 loss 曲线。

    model 须为官方 baselines.model.Model（有 normalized/forward/params/first/second/meta）。
    x/y 为原始特征与目标（未归一化）；内部按 model.meta 的 target_mean/scale 归一化目标。
    groups 非空时按组分层采样（每组 batch/组数 条），这是队友 B 包的关键做法；
    warmup > 0 时前若干 epoch 线性升到 lr_start；
    target_cap 非空时把训练目标截断到该值（C 任务用来压住极端高估，实测 cap350 最优）。
    """
    rng = np.random.default_rng(seed)
    xn = model.normalized(np.asarray(x, dtype=np.float64))
    if target_cap:
        y = np.clip(np.asarray(y, dtype=np.float64), 0, target_cap)
    target = ((np.asarray(y, dtype=np.float64) - model.meta["target_mean"])
              / model.meta["target_scale"])
    n = len(xn)
    batch = int(max(2, min(batch, n)))
    curve = []
    groups = np.asarray(groups) if groups is not None else None
    if groups is not None:
        uniq = np.unique(groups)
        per = max(1, batch // len(uniq))
        idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
        n_steps = max(1, n // batch)
    for ep in range(int(epochs)):
        # 预热 + 余弦退火：lr_start → lr_end
        if ep < warmup:
            lr = lr_start * (ep + 1) / max(1, warmup)
        else:
            t = (ep - warmup) / max(1, epochs - warmup - 1)
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))
        model.meta["learning_rate"] = float(lr)
        if groups is None:
            order = rng.permutation(n)
            batches = [order[s:s + batch] for s in range(0, n, batch)]
        else:
            batches = []
            for _ in range(n_steps):
                parts = [rng.choice(idx_by[g], size=per,
                                    replace=len(idx_by[g]) < per) for g in uniq]
                idx = np.concatenate(parts)[:batch]
                rng.shuffle(idx)
                batches.append(idx)
        total = 0.0
        for idx in batches:
            if len(idx) < 2:
                continue
            hidden, logits = model.forward(xn[idx])
            err = logits[:, 0] - target[idx]
            total += float(np.sum(err ** 2))
            delta = (2 * err / len(idx))[:, None]
            grads = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            grads.update(w1=xn[idx].T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            model.meta["steps"] += 1
            for name, g in grads.items():
                g *= min(1.0, clip / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * g
                model.second[name] = 0.999 * model.second[name] + 0.001 * g ** 2
                first = model.first[name] / (1 - 0.9 ** model.meta["steps"])
                second = model.second[name] / (1 - 0.999 ** model.meta["steps"])
                model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        model.meta["epochs"] += 1
        curve.append(total / n)
    return curve

def onecycle_train(model, x, y, epochs=1200, peak_lr=0.018, warmup_frac=0.10,
                   lr_end=1.8e-5, batch=256, seed=0, clip=5.0, groups=None,
                   target_cap=350, on_epoch=None, epoch_offset=0, total=None,
                   shape="cosine", start_lr=None):
    """one-cycle 训练（原地更新），返回 loss 曲线。

    前 warmup_frac 线性升到 peak_lr，其余按余弦降到 lr_end。
    groups 非空时按组分层采样；target_cap 对目标做截断（C 用 350）。
    on_epoch(epi_global, model) 可选回调，用于分块记录。
    """
    rng = np.random.default_rng(seed)
    xn = model.normalized(np.asarray(x, dtype=np.float64))
    if target_cap:
        y = np.clip(np.asarray(y, dtype=np.float64), 0, target_cap)
    target = ((np.asarray(y, dtype=np.float64) - model.meta["target_mean"])
              / model.meta["target_scale"])
    n = len(xn)
    batch = int(max(2, min(batch, n)))
    total = int(total or (epoch_offset + epochs))
    warmup = max(1, int(total * float(warmup_frac)))
    groups = np.asarray(groups) if groups is not None else None
    if groups is not None:
        uniq = np.unique(groups)
        per = max(1, batch // len(uniq))
        idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
        n_steps = max(1, n // batch)
    curve = []
    for k in range(int(epochs)):
        ep = int(epoch_offset) + k
        start = peak_lr * 0.05 if start_lr is None else float(start_lr)
        if ep < warmup:
            lr = start + (peak_lr - start) * (ep + 1) / warmup
        else:
            tt = (ep - warmup) / max(1, total - warmup - 1)
            if shape == "step60":
                # 前 60% 保持 peak，后 40% 余弦降到 lr_end（docs/68 最优）
                if tt < 0.6:
                    lr = peak_lr
                else:
                    q = (tt - 0.6) / 0.4
                    lr = lr_end + 0.5 * (peak_lr - lr_end) * (1 + math.cos(math.pi * q))
            else:
                lr = lr_end + 0.5 * (peak_lr - lr_end) * (1 + math.cos(math.pi * tt))
        model.meta["learning_rate"] = float(lr)
        if groups is None:
            order = rng.permutation(n)
            batches = [order[s:s + batch] for s in range(0, n, batch)]
        else:
            batches = []
            for _ in range(n_steps):
                parts = [rng.choice(idx_by[g], size=per, replace=len(idx_by[g]) < per)
                         for g in uniq]
                idx = np.concatenate(parts)[:batch]
                rng.shuffle(idx)
                batches.append(idx)
        total_loss = 0.0
        for idx in batches:
            if len(idx) < 2:
                continue
            hidden, logits = model.forward(xn[idx])
            err = logits[:, 0] - target[idx]
            total_loss += float(np.sum(err ** 2))
            delta = (2 * err / len(idx))[:, None]
            grads = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            grads.update(w1=xn[idx].T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            model.meta["steps"] += 1
            for name, g in grads.items():
                g *= min(1.0, clip / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * g
                model.second[name] = 0.999 * model.second[name] + 0.001 * g ** 2
                f1 = model.first[name] / (1 - 0.9 ** model.meta["steps"])
                f2 = model.second[name] / (1 - 0.999 ** model.meta["steps"])
                model.params[name] -= model.meta["learning_rate"] * f1 / (np.sqrt(f2) + 1e-8)
        model.meta["epochs"] += 1
        curve.append(total_loss / n)
        if on_epoch is not None:
            on_epoch(ep, model)
    return curve
