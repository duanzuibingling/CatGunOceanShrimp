#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B 任务「模型闭环优化」引擎（开发包主线版，自包含，仅依赖 numpy + 官方 baselines）。

闭环五步：
  感知（影子评估）→ 决策（是否训练/练多久）→ 执行（从线上 checkpoint 续训 60 epoch）
  → 发布验证（整体+按轴承平衡双指标，容差 5%）→ 采纳/保持不变 → 哈希链审计
另有 rollback()：上线后监控到退化，切回上一个已发布版本。

推荐配置（嵌套交叉验证定稿，防过拟合）：lr=0.025、按轴承分层 batch=126（每 batch 每根轴承 14 条）、
epochs=60/轮、跨轮 lr×0.90、发布门禁 5%、**直接部署原始权重**。
10 轮闭环 public validation：681.16 → 425.33（平衡 314.72，最差轴承 439.69）；
留一轴承嵌套 CV 的合并 OOF ≈ 422.0（与全验证 425.3 一致，无明显选择偏差）。
若 `--deploy ema` 则改为轮内 EMA(halflife=4) 部署（小 batch 场景更常用）。

关键设计（与 docs/22-B曲线抖动治理.md 对应）：
  - 只用官方 B 包数据与 checkpoint，不重新初始化（首轮父权重=官方 checkpoint，之后=线上版本）；
  - 每轮训练：lr 0.025 × 0.90^(轮次-1)，轮内余弦退火到 5%、warmup 2 epoch；
  - 按轴承分层采样（默认 batch=126，每 batch 每根轴承 14 条）；
  - EMA(halflife=4 epoch) 部署，避免"最后一步相位噪声"；
  - 被门禁拒绝时线上版本完全不变（步数也不前进），更贴近生产。

用法（在开发包根目录下）：
  D:/竞赛/.venv/Scripts/python.exe src/dp/loop_engine.py ^
      --out results/closed_loop_B --rounds 10 --epochs 40 --gate-margin 0.05 ^
      --export results/closed_loop_B_record --drill 2
本文件的 repro 实验版本：D:/竞赛/repro/v4_smooth/loop_engine.py（含 5 阶段扫描脚本）。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEV_ROOT = Path(__file__).resolve().parents[2]

def _resolve_official():
    """在容器/宿主里定位官方 B 包（含 baselines/ 与 checkpoint.npz）。
    优先 env DP_B_PKG，其次若干常见挂载点，最后退回开发包布局。"""
    cands = [os.environ.get("DP_B_PKG", ""), str(DEV_ROOT / "data" / "official" / "B"),
             str(ROOT / "data" / "official" / "B"), "/data/official/B", "/data/B",
             "/official/B", "/app/data/official/B"]
    for c in cands:
        if not c:
            continue
        p = Path(c)
        if (p / "checkpoint.npz").is_file() and (p / "baselines" / "model.py").is_file():
            return p
    return DEV_ROOT / "data" / "official" / "B"

OFFICIAL = _resolve_official()
TRAIN_PATH = OFFICIAL / "public" / "train.jsonl"
VAL_PATH = OFFICIAL / "public" / "validation.jsonl"
OFFICIAL_CKPT = OFFICIAL / "checkpoint.npz"

if OFFICIAL.is_dir():
    sys.path.insert(0, str(OFFICIAL))
from baselines.model import Model, checkpoint_digest  # noqa: E402
from baselines.data import read_dataset  # noqa: E402

PARAM_NAMES = ("w1", "b1", "w2", "b2")


# ---------------------------------------------------------------- 数据/评估
def load_data():
    train = read_dataset(TRAIN_PATH, "regression")
    val = read_dataset(VAL_PATH, "regression", train["paths"])
    train_groups = _bearings(TRAIN_PATH)
    val_groups = _bearings(VAL_PATH)
    if len(train_groups) != len(train["ids"]) or len(val_groups) != len(val["ids"]):
        raise ValueError("bearing meta mismatch")
    return train, val, np.asarray(train_groups), np.asarray(val_groups)


def _bearings(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["meta"]["bearing"])
    return out


def predict_params(model, xn, params):
    h = np.tanh(xn @ params["w1"] + params["b1"])
    logits = h @ params["w2"] + params["b2"]
    return np.maximum(0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])


def evaluate(model, x, y, groups, params=None):
    """official 口径：全体 RMSE/MAE + 按轴承分组 RMSE + 平衡 RMSE（分组平均）。"""
    params = model.params if params is None else params
    pred = predict_params(model, model.normalized(x), params)
    err = pred - y
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
           "n": int(len(y)), "per_group": {}, "pred_stats": [float(pred.min()), float(pred.mean()), float(pred.max())]}
    per = []
    for g in np.unique(groups):
        mask = groups == g
        r = float(np.sqrt(np.mean(err[mask] ** 2)))
        out["per_group"][str(g)] = {"rmse": r, "n": int(mask.sum())}
        per.append(r)
    out["balanced_rmse"] = float(np.mean(per))
    out["worst_group_rmse"] = float(np.max(per))
    return out


# ---------------------------------------------------------------- 训练核心
def _shuffled_batches(model, xn, target, cfg, rng):
    """返回本轮 epoch 的 batch 索引列表。stratified=True 时按轴承等量抽。"""
    batch = model.meta["batch_size"]
    if cfg.get("stratified"):
        groups = cfg["_train_groups"]
        uniq = np.unique(groups)
        per = batch // len(uniq)
        if per < 1:
            raise ValueError("batch too small for stratified sampling")
        n_steps = len(xn) // batch
        need = per * n_steps
        streams = {}
        for g in uniq:
            members = np.where(groups == g)[0]
            seq = []
            while len(seq) < need:
                seq.extend(rng.permutation(members).tolist())
            streams[g] = seq
        return [np.asarray([streams[g][s * per:(s + 1) * per] for g in uniq]).ravel()
                for s in range(n_steps)]
    order = rng.permutation(len(xn))
    return [order[s:s + batch] for s in range(0, len(order), batch)]


def fit_variant(model, x, y, groups, epochs, lr, batch, cfg, record_each_epoch=None):
    """在 model 上继续训练 epochs 个 epoch，返回 (history, ema_params, swa_params)。

    history 每项记录：epoch / step / train_loss / raw 验证 / ema 验证（若启用）。
    record_each_epoch: 可选的额外回调 fn(model, epoch_idx, ema_params) -> dict
    """
    xn = model.normalized(x)
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    batch = int(batch)
    model.meta["batch_size"] = batch
    cfg = dict(cfg)
    if cfg.get("stratified"):
        mode = cfg.get("stratify_by", "bearing")
        if mode == "bearing":
            cfg["_train_groups"] = groups
        elif mode == "rul":
            qs = np.quantile(y, [0.25, 0.5, 0.75])
            cfg["_train_groups"] = np.asarray([str(int(b)) for b in np.digitize(y, qs)])
            cfg["stratified"] = True
        elif mode == "bearing_rul":
            qs = np.quantile(y, [0.5])
            br = np.digitize(y, qs)
            cfg["_train_groups"] = np.asarray([f"{g}_{b}" for g, b in zip(groups, br)])
        else:
            raise ValueError(f"unknown stratify_by {mode}")
    clip = float(cfg.get("clip_norm", 5.0))
    huber = cfg.get("huber_delta")
    ema_decay = cfg.get("ema_decay")
    if cfg.get("ema_halflife"):
        steps_per_epoch = max(1, len(xn) // batch)
        ema_decay = 0.5 ** (1.0 / (float(cfg["ema_halflife"]) * steps_per_epoch))
    cfg["_ema_decay_effective"] = ema_decay
    sched = cfg.get("lr_schedule", "const")
    lr_min_ratio = float(cfg.get("lr_min_ratio", 0.0))
    warmup = int(cfg.get("warmup_epochs", 0) or 0)
    ema = {k: v.copy() for k, v in model.params.items()}
    ema_full = {"first": {k: v.copy() for k, v in model.first.items()},
                "second": {k: v.copy() for k, v in model.second.items()}}
    swa_buf = []
    snapshots = []
    hist = []
    for ep in range(epochs):
        # ---- 轮内 lr 调度（按 epoch 粒度，官方粒度是整轮，这里做轮内衰减）----
        if sched == "cosine":
            mult = lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * ep / max(epochs, 1)))
        elif sched == "linear":
            mult = 1 - (1 - lr_min_ratio) * ep / max(epochs - 1, 1)
        else:
            mult = 1.0
        if warmup and ep < warmup:
            mult *= (ep + 1) / warmup
        model.meta["learning_rate"] = float(lr) * mult
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        batches = _shuffled_batches(model, xn, target, cfg, rng)
        loss_sum = 0.0
        for indices in batches:
            b, t = xn[indices], target[indices]
            hidden, logits = model.forward(b)
            error = logits[:, 0] - t
            if huber:
                d = float(huber)
                abs_e = np.abs(error)
                quad = np.minimum(abs_e, d)
                loss_sum += float(np.sum(quad ** 2 + 2 * d * np.maximum(abs_e - d, 0)))
                delta = np.where(abs_e <= d, 2 * error, 2 * d * np.sign(error))
            else:
                loss_sum += float(np.sum(error ** 2))
                delta = 2 * error
            delta = (delta / len(indices))[:, None]
            grads = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hd = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            grads.update(w1=b.T @ hd, b1=hd.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            model.meta["steps"] += 1
            for name in PARAM_NAMES:
                g = grads[name]
                g *= min(1.0, clip / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * g
                model.second[name] = .999 * model.second[name] + .001 * g ** 2
                m1 = model.first[name] / (1 - .9 ** model.meta["steps"])
                m2 = model.second[name] / (1 - .999 ** model.meta["steps"])
                model.params[name] -= model.meta["learning_rate"] * m1 / (np.sqrt(m2) + 1e-8)
                if ema_decay:
                    ema[name] = ema_decay * ema[name] + (1 - ema_decay) * model.params[name]
                    if cfg.get("ema_full"):
                        ema_full["first"][name] = ema_decay * ema_full["first"][name] + (1 - ema_decay) * model.first[name]
                        ema_full["second"][name] = ema_decay * ema_full["second"][name] + (1 - ema_decay) * model.second[name]
            if not all(np.all(np.isfinite(p)) for p in model.params.values()):
                raise ValueError("nonfinite model state")
        model.meta["epochs"] += 1
        swa_buf.append({k: v.copy() for k, v in model.params.items()})
        rec = {"epoch": model.meta["epochs"], "step": model.meta["steps"],
               "train_loss": loss_sum / len(xn), "lr": model.meta["learning_rate"]}
        dep = None
        if cfg.get("eval_deploy") == "swa":
            k = int(cfg.get("swa_epochs", 3))
            use = swa_buf[-k:]
            dep = {name: np.mean([p[name] for p in use], axis=0) for name in PARAM_NAMES}
        elif cfg.get("eval_deploy") == "ema":
            dep = ema
        if record_each_epoch is not None:
            rec.update(record_each_epoch(model, ep, ema, dep))
        hist.append(rec)
        if cfg.get("snapshots"):
            snapshots.append({"raw": {k: v.copy() for k, v in model.params.items()},
                              "ema": {k: v.copy() for k, v in ema.items()}})
    model.validate()
    swa = None
    if cfg.get("deploy") == "swa":
        k = int(cfg.get("swa_epochs", 3))
        use = swa_buf[-k:]
        swa = {name: np.mean([p[name] for p in use], axis=0) for name in PARAM_NAMES}
    return {"hist": hist, "ema": ema, "swa": swa, "snapshots": snapshots, "ema_full": ema_full}


# ---------------------------------------------------------------- 单轮 / 链式
def run_round(parent_ckpt, out_dir, *, epochs, lr, batch, cfg, val_data, val_groups,
              train_data, train_groups, stage_id):
    """从 parent_ckpt 续训一轮；按 cfg['deploy'] 产出 checkpoint + history + report。"""
    model = Model.load(parent_ckpt)
    parent_sha = checkpoint_digest(parent_ckpt)
    parent_params = {k: v.copy() for k, v in model.params.items()}
    parent_first = {k: v.copy() for k, v in model.first.items()}
    parent_second = {k: v.copy() for k, v in model.second.items()}
    before = evaluate(model, val_data["x"], val_data["y"], val_groups)
    t0 = time.monotonic()

    def obs(m, ep, ema, dep):
        rec = {"validation": evaluate(m, val_data["x"], val_data["y"], val_groups)}
        if cfg.get("ema_decay") or cfg.get("ema_halflife"):
            rec["validation_ema"] = evaluate(m, val_data["x"], val_data["y"], val_groups, params=ema)
        if dep is not None:
            rec["validation_deploy"] = evaluate(m, val_data["x"], val_data["y"], val_groups, params=dep)
        return rec

    deploy = cfg.get("deploy", "raw")
    cfg = dict(cfg)
    if deploy in ("best_ema", "best_raw"):
        cfg["snapshots"] = True
    res = fit_variant(model, train_data["x"], train_data["y"], train_groups,
                      epochs, lr, batch, cfg, record_each_epoch=obs)
    hist, ema, swa, snaps = res["hist"], res["ema"], res["swa"], res["snapshots"]
    duration = time.monotonic() - t0
    best_epoch = None
    if deploy == "ema" and (cfg.get("ema_decay") or cfg.get("ema_halflife")):
        model.params = {k: v.copy() for k, v in ema.items()}
    elif deploy == "ema_full" and (cfg.get("ema_decay") or cfg.get("ema_halflife")):
        model.params = {k: v.copy() for k, v in ema.items()}
        model.first = {k: v.copy() for k, v in res["ema_full"]["first"].items()}
        model.second = {k: v.copy() for k, v in res["ema_full"]["second"].items()}
    elif deploy == "swa" and swa is not None:
        model.params = {k: v.copy() for k, v in swa.items()}
    elif deploy in ("round_ema", "blend"):
        alpha = float(cfg.get("deploy_alpha", 0.5))
        target = ema if deploy == "round_ema" else None
        for name in PARAM_NAMES:
            src = parent_params[name] if deploy == "round_ema" else model.params[name]
            if deploy == "round_ema":
                model.params[name] = alpha * parent_params[name] + (1 - alpha) * model.params[name]
            else:  # blend: 与 EMA 混合
                model.params[name] = alpha * model.params[name] + (1 - alpha) * ema[name]
    elif deploy in ("best_ema", "best_raw"):
        key = "validation_ema" if deploy == "best_ema" else "validation"
        best = int(min(range(len(hist)), key=lambda i: hist[i][key]["rmse"]))
        best_epoch = best + 1
        model.params = {k: v.copy() for k, v in snaps[best]["ema" if deploy == "best_ema" else "raw"].items()}
    model.validate()
    after = evaluate(model, val_data["x"], val_data["y"], val_groups)
    rolled_back = False
    if cfg.get("rollback_if_worse"):
        margin = float(cfg.get("rollback_margin", 0.0))
        mode = cfg.get("rollback_mode", "single")
        if mode == "both":
            accept = (after["rmse"] <= before["rmse"] * (1 + margin)
                      and after["balanced_rmse"] <= before["balanced_rmse"] * (1 + margin))
        else:
            key = cfg.get("rollback_metric", "rmse")
            accept = after[key] <= before[key] * (1 + margin)
        if not accept:
            model.params = {k: v.copy() for k, v in parent_params.items()}
            model.first = {k: v.copy() for k, v in parent_first.items()}
            model.second = {k: v.copy() for k, v in parent_second.items()}
            model.validate()
            after = evaluate(model, val_data["x"], val_data["y"], val_groups)
            rolled_back = True
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output must be new: {out_dir}")
    out_dir.mkdir(parents=True)
    ckpt = out_dir / "checkpoint.npz"
    model.save(ckpt)
    with (out_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for r in hist:
            f.write(json.dumps({"stage_id": stage_id, "seed": model.meta["seed"], **r},
                               ensure_ascii=False, allow_nan=False) + "\n")
    report = {"format": "contest.cpu-baseline-report.v1", "profile": "B", "task": "regression",
              "stage_id": stage_id, "seed": model.meta["seed"], "epochs": epochs,
              "learning_rate": lr, "batch_size": batch, "variant": cfg,
              "parent_checkpoint_sha256": parent_sha,
              "checkpoint_sha256": checkpoint_digest(ckpt),
              "validation_before": before, "validation_after": after,
              "training_seconds": duration,
              "deployed_weights": deploy, "deployed_epoch": best_epoch,
              "rolled_back": rolled_back}
    with (out_dir / "training.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, allow_nan=False, indent=2)
    return report, hist, after


def run_chain(out_root, *, rounds, epochs, lr, batch, cfg, initial_ckpt=OFFICIAL_CKPT,
              lr_decay_per_round=1.0, data=None, global_cosine=False, global_lr_min_ratio=0.05):
    """10 轮链式续训：第 1 轮父权重=官方 checkpoint，之后每轮=上一轮 deploy 产物。

    global_cosine=True 时，学习率在「整条 10 轮」上按一条余弦曲线下降（每轮只在轮内
    按 cfg 的 lr_schedule 微调），避免每轮重启高 lr 造成的锯齿。
    """
    train_data, val_data, train_groups, val_groups = data or load_data()
    out_root = Path(out_root)
    if out_root.exists():
        raise ValueError(f"output must be new: {out_root}")
    out_root.mkdir(parents=True)
    parent = Path(initial_ckpt)
    rows = []
    for r in range(1, rounds + 1):
        if global_cosine:
            frac = 0.5 * (1 + math.cos(math.pi * (r - 1) / max(rounds - 1, 1)))
            cur_lr = lr * (global_lr_min_ratio + (1 - global_lr_min_ratio) * frac)
        else:
            cur_lr = lr * (lr_decay_per_round ** (r - 1))
        rep, hist, after = run_round(parent, out_root / f"round-{r:02d}",
                                     epochs=epochs, lr=cur_lr, batch=batch, cfg=cfg,
                                     val_data=val_data, val_groups=val_groups,
                                     train_data=train_data, train_groups=train_groups,
                                     stage_id=f"round-{r}")
        rows.append({"round": r, "lr": cur_lr, "metrics": after,
                     "parent_sha": rep["parent_checkpoint_sha256"],
                     "checkpoint_sha": rep["checkpoint_sha256"],
                     "seconds": rep["training_seconds"],
                     "rolled_back": rep["rolled_back"],
                     "epoch_rmse": [h["validation"]["rmse"] for h in hist],
                     "epoch_rmse_ema": [h["validation_ema"]["rmse"] for h in hist]
                     if (cfg.get("ema_decay") or cfg.get("ema_halflife")) else None,
                     "epoch_rmse_deploy": [h["validation_deploy"]["rmse"] for h in hist]
                     if cfg.get("eval_deploy") else None,
                     "epoch_train_loss": [h["train_loss"] for h in hist]})
        parent = out_root / f"round-{r:02d}" / "checkpoint.npz"
    summary = {"rounds": rounds, "epochs_per_round": epochs, "lr": lr, "batch": batch,
               "cfg": cfg, "lr_decay_per_round": lr_decay_per_round, "rows": rows}
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, allow_nan=False, indent=2)
    return summary


# ---------------------------------------------------------------- 抖动指标
def curve_stats(values, warmup=0):
    v = np.asarray(values[warmup:], dtype=float)
    d = np.diff(v)
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
            "swing": float(v.max() - v.min()), "mean_abs_delta": float(np.mean(np.abs(d))) if len(d) else 0.0,
            "max_abs_delta": float(np.max(np.abs(d))) if len(d) else 0.0,
            "min": float(v.min()), "max": float(v.max()),
            "argmin": int(np.argmin(v)) + warmup, "first": float(v[0]), "last": float(v[-1])}



import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_CFG = {"lr_schedule": "cosine", "lr_min_ratio": 0.05, "warmup_epochs": 2,
               "stratified": True, "stratify_by": "bearing", "ema_halflife": 4,
               "eval_deploy": "ema", "deploy": "ema"}
RAW_CFG = {"lr_schedule": "cosine", "lr_min_ratio": 0.05, "warmup_epochs": 2,
           "stratified": True, "stratify_by": "bearing"}
LR, BATCH, DECAY = 0.025, 126, 0.90


def build_cfg(deploy="ema", halflife=4):
    """deploy='raw' 时直接部署原始权重（不适合再用 EMA 的短轮配置）；'ema' 时用轮内 EMA。"""
    if deploy == "raw":
        return dict(RAW_CFG)
    return dict(DEFAULT_CFG, ema_halflife=halflife)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class ClosedLoop:
    def __init__(self, out_root, *, rounds=10, epochs=40, lr=LR, batch=BATCH, decay=DECAY,
                 cfg=None, gate_margin=0.05, deploy="ema", halflife=4, data=None):
        self.root = Path(out_root)
        if self.root.exists():
            raise ValueError(f"output must be new: {self.root}")
        self.root.mkdir(parents=True)
        (self.root / "published").mkdir()
        (self.root / "candidates").mkdir()
        self.rounds, self.epochs, self.lr, self.batch, self.decay = rounds, epochs, lr, batch, decay
        self.deploy_mode = deploy
        self.cfg = dict(cfg) if cfg is not None else build_cfg(deploy, halflife)
        self.gate_margin = gate_margin
        # data 可传入 (train, val, train_groups, val_groups)，用于嵌套交叉验证等
        self.train, self.val, self.tg, self.vg = data if data is not None else load_data()
        self.audit_prev = "genesis"
        self.report = {"started": now(), "config": {"rounds": rounds, "epochs": epochs, "lr": lr,
                                                   "batch": batch, "lr_decay_per_round": decay,
                                                   "gate_margin": gate_margin,
                                                   "deploy": deploy, "cfg": self.cfg},
                       "rounds": [], "published": [], "rollbacks": []}
        # v00 = 官方 checkpoint（初始线上版本）
        v00 = self.root / "published" / "v00" / "checkpoint.npz"
        v00.parent.mkdir(parents=True)
        shutil.copy2(OFFICIAL_CKPT, v00)
        self.versions = [{"version": "v00", "checkpoint": str(v00),
                          "sha256": checkpoint_digest(v00), "metrics": self._metrics(v00),
                          "published_at": now(), "note": "官方 checkpoint（初始线上版本）"}]
        self._set_online(0)

    # ---------------- 闭环各步 ----------------
    def _metrics(self, ckpt):
        m = Model.load(ckpt)
        return evaluate(m, self.val["x"], self.val["y"], self.vg)

    def monitor(self):
        """感知：线上模型在影子集上的整体/分组指标（工业上还可接漂移/业务指标）。"""
        online = self.versions[-1]
        return {"online_version": online["version"], "metrics": self._metrics(online["checkpoint"])}

    def gate(self, candidate, online):
        """发布验证：双指标 + 容差；返回 (是否发布, 原因)。"""
        m = self.gate_margin
        checks = {
            "overall": candidate["rmse"] <= online["rmse"] * (1 + m),
            "balanced": candidate["balanced_rmse"] <= online["balanced_rmse"] * (1 + m),
        }
        ok = all(checks.values())
        reason = ("通过：整体与平衡均未变差超过容差"
                  if ok else "拒绝：" + "、".join(k for k, v in checks.items() if not v) + "变差超过容差")
        return ok, {"checks": checks, "reason": reason}

    def train_candidate(self, r):
        """执行：从当前线上版本续训一轮，产出候选四件套（lr 按跨轮衰减调度）。"""
        online_ckpt = Path(self.versions[-1]["checkpoint"])
        d = self.root / "candidates" / f"round-{r:02d}"
        cur_lr = self.lr * (self.decay ** (r - 1))
        rep, hist, after = run_round(
            online_ckpt, d, epochs=self.epochs, lr=cur_lr, batch=self.batch, cfg=dict(self.cfg),
            val_data=self.val, val_groups=self.vg, train_data=self.train, train_groups=self.tg,
            stage_id=f"round-{r}")
        return d, rep, after

    def publish(self, r, candidate_ckpt, metrics, audit_entry):
        """发布：候选 -> 新线上版本（保留旧版本快照以支持回滚）。"""
        v = f"v{len(self.versions):02d}"
        dst = self.root / "published" / v / "checkpoint.npz"
        dst.parent.mkdir(parents=True)
        shutil.copy2(candidate_ckpt, dst)
        self.versions.append({"version": v, "checkpoint": str(dst), "sha256": checkpoint_digest(dst),
                              "metrics": metrics, "published_at": now(),
                              "note": f"第 {r} 轮候选通过发布门禁"})
        self._set_online(len(self.versions) - 1)
        audit_entry["action"] = "publish"
        audit_entry["published_version"] = v
        return v

    def rollback(self, to_version=None, reason="上线后监控发现指标退化"):
        """回滚：把线上指针切回上一个（或指定）已发布版本。"""
        if to_version is None:
            idx = max(0, len(self.versions) - 2)
        else:
            idx = next(i for i, v in enumerate(self.versions) if v["version"] == to_version)
        cur = self.versions[-1]["version"]
        self._set_online(idx)
        entry = {"round": None, "action": "rollback", "from": cur,
                 "to": self.versions[idx]["version"], "reason": reason, "time": now()}
        self.report["rollbacks"].append(entry)
        self.audit(entry)
        return entry

    def _set_online(self, idx):
        (self.root / "online.json").write_text(json.dumps(
            {"online_version": self.versions[idx]["version"],
             "checkpoint": self.versions[idx]["checkpoint"],
             "sha256": self.versions[idx]["sha256"],
             "updated_at": now()}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 审计（哈希链） ----------------
    def audit(self, entry):
        prev = self.audit_prev
        payload = json.dumps({"prev": prev, **entry}, ensure_ascii=False, sort_keys=True)
        h = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        record = {"prev": prev, **entry, "hash": h}
        with (self.root / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.audit_prev = h
        return record

    def verify_audit(self):
        records, prev = [], "genesis"
        ok = True
        with (self.root / "audit.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                expect = "sha256:" + hashlib.sha256(
                    json.dumps({"prev": prev, **{k: v for k, v in r.items() if k != "hash"}},
                               ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
                if r.get("prev") != prev or r.get("hash") != expect:
                    ok = False
                prev = r.get("hash")
                records.append(r)
        return {"verified": ok, "n_records": len(records)}

    # ---------------- 跑闭环 ----------------
    def run(self):
        t0 = time.monotonic()
        for r in range(1, self.rounds + 1):
            online = self.versions[-1]
            mon = {"online_version": online["version"], "metrics": self._metrics(online["checkpoint"])}
            cand_dir, rep, cand_metrics = self.train_candidate(r)
            ok, ginfo = self.gate(cand_metrics, mon["metrics"])
            entry = {"round": r, "time": now(),
                     "online_version": online["version"], "online_metrics": mon["metrics"],
                     "candidate_dir": str(cand_dir),
                     "candidate_sha256": rep["checkpoint_sha256"],
                     "parent_sha256": rep["parent_checkpoint_sha256"],
                     "candidate_metrics": cand_metrics,
                     "gate": ginfo,
                     "train_seconds": rep["training_seconds"]
                     }
            if ok:
                v = self.publish(r, cand_dir / "checkpoint.npz", cand_metrics, entry)
                online_after = v
            else:
                entry["action"] = "hold"
                online_after = online["version"]
            entry["online_after"] = online_after
            entry = self.audit(entry)
            self.report["rounds"].append({k: v for k, v in entry.items() if k != "hash"})
            print(f"round {r:02d} online={online['version']:>4s} cand={cand_metrics['rmse']:7.2f} "
                  f"online={mon['metrics']['rmse']:7.2f} -> {entry['action']:7s} "
                  f"({ginfo['reason'][:18]})", flush=True)
        self.report["finished"] = now()
        self.report["seconds"] = round(time.monotonic() - t0, 2)
        self.report["versions"] = self.versions
        self.report["audit"] = self.verify_audit()
        # 闭环曲线：每轮线上模型的指标（发布后）与候选指标
        self.report["online_curve"] = [{"round": 0, "version": "v00", "rmse": self.versions[0]["metrics"]["rmse"],
                                        "balanced_rmse": self.versions[0]["metrics"]["balanced_rmse"]}]
        for rd in self.report["rounds"]:
            v = next(x for x in self.versions if x["version"] == rd["online_after"])
            self.report["online_curve"].append({"round": rd["round"], "version": v["version"],
                                                "rmse": v["metrics"]["rmse"],
                                                "balanced_rmse": v["metrics"]["balanced_rmse"]})
        self._write_report()
        return self.report

    def _write_report(self):
        (self.root / "loop_report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 导出为平台口径的 10 轮记录 ----------------
    def export_records(self, dst_root):
        """把闭环运行导出成「每轮一个目录」的记录（checkpoint=本轮采纳的线上版本）。

        被门禁拒绝的轮次：checkpoint 复制线上版本（本轮不改线上），并在 decision.json 里记录拒绝原因。
        training.json 重写为：父权重 sha、采纳后的 checkpoint sha、候选与线上指标、门禁结论。
        """
        dst_root = Path(dst_root)
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True)
        rows = []
        for rd in self.report["rounds"]:
            r = rd["round"]
            d = dst_root / f"round-{r:02d}"
            d.mkdir()
            src = self.root / "candidates" / f"round-{r:02d}"
            shutil.copy2(src / "history.jsonl", d / "history.jsonl")
            cand_rep = json.loads((src / "training.json").read_text(encoding="utf-8"))
            committed = next(v for v in self.versions if v["version"] == rd["online_after"])
            shutil.copy2(committed["checkpoint"], d / "checkpoint.npz")
            rep = {"format": "contest.cpu-baseline-report.v1", "profile": "B", "task": "regression",
                   "stage_id": f"round-{r}", "seed": cand_rep["seed"],
                   "parent_checkpoint_sha256": rd["parent_sha256"],
                   "checkpoint_sha256": committed["sha256"],
                   "validation_before": rd["online_metrics"],
                   "validation_after": committed["metrics"],
                   "candidate_checkpoint_sha256": rd["candidate_sha256"],
                   "candidate_metrics": rd["candidate_metrics"],
                   "gate": rd["gate"], "action": rd["action"],
                   "online_before": rd["online_version"], "online_after": rd["online_after"],
                   "epochs": self.epochs, "learning_rate": cand_rep.get("learning_rate", self.lr),
                   "batch_size": cand_rep.get("batch_size", self.batch),
                   "training_seconds": rd["train_seconds"],
                   "configuration": cand_rep.get("configuration", {
                       "hidden": 24, "activation": "tanh", "optimizer": "Adam",
                       "gradient_clip_norm": 5.0, "learning_rate": cand_rep.get("learning_rate"),
                       "batch_size": cand_rep.get("batch_size")}),
                   "deployed_weights": "ema",
                   "epochs_per_round": self.epochs, "lr_decay_per_round": self.decay,
                   "note": ("本轮候选通过发布门禁，采纳为新线上版本" if rd["action"] == "publish"
                            else "本轮候选被发布门禁拒绝（整体或平衡指标变差超过容差），线上版本保持不变")}
            (d / "training.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            (d / "decision.json").write_text(json.dumps(
                {"round": r, "action": rd["action"], **rd["gate"],
                 "online_before": rd["online_version"], "online_after": rd["online_after"]},
                ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({"round": r, "action": rd["action"], "metrics": committed["metrics"],
                         "candidate_metrics": rd["candidate_metrics"],
                         "online_after": rd["online_after"]})
        lines = ["# B · 模型闭环优化 10 轮记录（每轮 40 epoch，seed=42）", "",
                 "> 闭环：感知（影子评估）→ 执行（从线上版本续训候选）→ 发布验证（双指标+5%容差）→"
                 "采纳/保持不变 → 审计（audit.jsonl 哈希链）。", "",
                 "| 轮次 | 线上版本 | 候选 RMSE | 候选平衡 RMSE | 决策 | 采纳后线上版本 | 线上 RMSE |",
                 "|---|---|---|---|---|---|---|"]
        for x in rows:
            m, cm = x["metrics"], x["candidate_metrics"]
            lines.append(f"| {x['round']:02d} | {x['online_after']} | {cm['rmse']:.2f} | "
                         f"{cm['balanced_rmse']:.2f} | {'采纳' if x['action'] == 'publish' else '拒绝'} | "
                         f"{x['online_after']} | {m['rmse']:.2f} |")
        last = rows[-1]["metrics"]
        lines += ["", f"- 官方基线：681.16 / 平衡 447.87 / 最差轴承 710.81",
                  f"- 第 10 轮线上版本：{last['rmse']:.2f} / 平衡 {last['balanced_rmse']:.2f} / "
                  f"最差轴承 {last['worst_group_rmse']:.2f}",
                  f"- 审计链：{self.report['audit']}（audit.jsonl 逐条 prev/hash 可校验）", ""]
        (dst_root / "总表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (dst_root / "summary.json").write_text(json.dumps(
            {"rounds": rows, "config": self.report["config"],
             "loop_report": str(self.root / "loop_report.json")},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return dst_root

    # ---------------- 上线后回滚演练 ----------------
    def drill_rollback(self, candidate_round=2, slice_name="rul_ge_1000", threshold=0.05):
        """演练：强制发布一个坏候选 → 上线后监控发现退化 → 切回上一个已发布版本。

        用第 N 轮「被门禁拒绝」的候选（整体可能还行、但平衡/长尾切片变差）模拟一次发布事故。
        """
        src = self.root / "candidates" / f"round-{candidate_round:02d}"
        cand_ckpt = src / "checkpoint.npz"
        bad_metrics = self._metrics(cand_ckpt)
        prev = self.versions[-1]
        v = f"v{len(self.versions):02d}"
        dst = self.root / "published" / v / "checkpoint.npz"
        dst.parent.mkdir(parents=True)
        shutil.copy2(cand_ckpt, dst)
        self.versions.append({"version": v, "checkpoint": str(dst), "sha256": checkpoint_digest(dst),
                              "metrics": bad_metrics, "published_at": now(),
                              "note": f"演练：强制发布第 {candidate_round} 轮坏候选"})
        self._set_online(len(self.versions) - 1)
        entry = {"round": None, "action": "publish_forced", "published_version": v,
                 "candidate_round": candidate_round, "metrics": bad_metrics,
                 "reason": "回滚演练：强制发布一个被门禁拒绝的候选", "time": now()}
        self.audit(entry)
        self.report["drills"] = self.report.get("drills", []) + [entry]

        # 上线后监控：用长尾切片（RUL>=1000）与平衡口径
        y = self.val["y"]
        mask = y >= 1000 if slice_name == "rul_ge_1000" else np.ones(len(y), bool)
        x, yy, g = self.val["x"][mask], y[mask], self.vg[mask]
        per_version = []
        for ver in self.versions:
            m = Model.load(ver["checkpoint"])
            per_version.append({"version": ver["version"],
                                "slice_rmse": evaluate(m, x, yy, g)["rmse"],
                                "balanced_rmse": ver["metrics"]["balanced_rmse"]})
        hist = [p for p in per_version if p["version"] != v]
        best = min(hist, key=lambda p: p["slice_rmse"])
        cur = next(p for p in per_version if p["version"] == v)
        degraded = cur["slice_rmse"] > best["slice_rmse"] * (1 + threshold)
        result = {"candidate_round": candidate_round, "slice": slice_name, "n_slice": int(mask.sum()),
                  "forced_version": v, "per_version": per_version,
                  "degraded": bool(degraded), "best_previous": best,
                  "rollback": None}
        if degraded:
            result["rollback"] = self.rollback(to_version=best["version"],
                                               reason=f"回滚演练：上线后 {slice_name} 切片 "
                                                      f"{cur['slice_rmse']:.1f} > {best['version']} 的 "
                                                      f"{best['slice_rmse']:.1f}（超阈值 {threshold:.0%}）")
        self.report.setdefault("drills", []).append(result)
        self._write_report()
        return result

    def simulate_monitoring(self, hard_slice="rul_ge_1000", degradation=0.05):
        """演示「上线后监控到退化 -> 回滚」。

        用验证集里 RUL>=1000 的长尾样本当"上线后的新批次"（工业里是新数据到来），
        对每个已发布版本算该切片 RMSE；若当前线上版本比历史最好差 >degradation，就回滚。
        """
        y = self.val["y"]
        mask = y >= 1000 if hard_slice == "rul_ge_1000" else np.ones(len(y), bool)
        x, yy, g = self.val["x"][mask], y[mask], self.vg[mask]
        per_version = []
        for v in self.versions:
            m = Model.load(v["checkpoint"])
            ev = evaluate(m, x, yy, g)
            per_version.append({"version": v["version"], "rmse": ev["rmse"]})
        best = min(per_version, key=lambda d: d["rmse"])
        cur = per_version[-1]
        result = {"slice": hard_slice, "n": int(mask.sum()), "per_version": per_version,
                  "best": best, "current": cur}
        if cur["rmse"] > best["rmse"] * (1 + degradation) and best["version"] != cur["version"]:
            result["rollback"] = self.rollback(to_version=best["version"],
                                               reason=f"上线后监控：{hard_slice} 切片 {cur['rmse']:.1f} "
                                                      f"> 历史最好 {best['rmse']:.1f}（{best['version']}）")
        else:
            result["rollback"] = None
        self.report["monitoring"] = result
        self._write_report()
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60, help="每轮训练 epoch（推荐 60）")
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--batch", type=int, default=126,
                    help="分层采样 batch（9 的倍数，推荐 126 = 每 batch 每根轴承 14 条；嵌套 CV 最优）")
    ap.add_argument("--decay", type=float, default=0.90, help="跨轮学习率衰减")
    ap.add_argument("--deploy", choices=["raw", "ema"], default="raw",
                    help="发布哪种权重：raw=原始权重（推荐 bs90）；ema=轮内 EMA")
    ap.add_argument("--halflife", type=float, default=4.0, help="EMA 半衰期（epoch），仅 deploy=ema 有效")
    ap.add_argument("--gate-margin", type=float, default=0.05)
    ap.add_argument("--export", help="导出成平台口径的 10 轮记录目录")
    ap.add_argument("--drill", type=int, default=None, help="回滚演练：强制发布第 N 轮坏候选")
    ap.add_argument("--demo-rollback", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else Path.cwd() / out
    loop = ClosedLoop(out, rounds=a.rounds, epochs=a.epochs, lr=a.lr, batch=a.batch,
                      decay=a.decay, deploy=a.deploy, halflife=a.halflife,
                      gate_margin=a.gate_margin)
    rep = loop.run()
    print(f"\n闭环完成：发布 {len(rep['versions']) - 1} 次，拒绝 "
          f"{sum(1 for r in rep['rounds'] if r['action'] == 'hold')} 次，"
          f"审计链 {rep['audit']}，用时 {rep['seconds']}s")
    if a.export:
        exp = Path(a.export)
        exp = exp if exp.is_absolute() else Path.cwd() / exp
        d = loop.export_records(exp)
        print("导出记录:", d)
    if a.drill:
        r = loop.drill_rollback(a.drill)
        print("回滚演练:", json.dumps({k: v for k, v in r.items() if k != "per_version"},
                                    ensure_ascii=False))
    if a.demo_rollback:
        mon = loop.simulate_monitoring()
        print("上线后监控切片:", {k: v for k, v in mon.items() if k != "per_version"})
        print("回滚:", mon["rollback"])


if __name__ == "__main__":
    main()
