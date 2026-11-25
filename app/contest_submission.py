"""SDK 提交模块（contest.harness.workspace.v2）—— 第四轮审查后修复版（2026-11-17）。

每轮：父权重 + 训练行 → 续训 → 候选 → **提交（平台测量）** → **门禁裁决（合同指标）**
      → 通过则存 state 并选候选，否则选父产物 → 一次模型调用 → finish。
预测：逐行读 stdin、逐行写 stdout；分类 softmax、回归非负。

修复记录（D1–D9、E1–E7、F1–F6）：
  F1 门禁用合同指标：分类算 macro-F1（不再用 accuracy），回归算 RMSE；优先用平台 summary
  F2 门禁三态 pass/reject/unknown；unknown（validation 不可用/过小/无答案）→ **保留父**（fail-closed）
  F3 先提交后裁决：每轮都产生平台测量；被拒轮仍提交候选但不选它，且**不上传 state**
  F4 容差按噪声量级：A 0.005（macro-F1 绝对）、B/C 相对 2%（死区，防噪声误判）
  F5 门禁用全量 validation（A 1,244 / B 1,679 / C 23,049），不再抽样，消除 population 偏差
  F6 父权重不可读 → finish(父产物)（带一次模型调用），不再整轮失败
  D5 /state：只认 model_state.npz，校验 paths/shapes/epochs 单调，steps=max
  D7 自适应按"父是否官方起点（epochs ≤ 20）"；D8 显式 CONTEST_EPOCHS/LR 优先
"""
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
EPOCHS_BY_KEY = {"A": 6, "B": 60, "C": 100}
LR_BY_KEY = {"A": 0.003, "B": 0.025, "C": 0.003}
# A：第 4-10 轮不再归零（原缺陷：warm 配方 0 步，晋升一次后永久停训）
WARM_EPOCHS = {"A": 6, "B": 0, "C": 0}
WARM_LR = {"A": 0.003, "B": 0.025, "C": 0.003}
OFFICIAL_EPOCH_LIMIT = 20
# B/C：官方 train 在 round 1 就全量可用，长训一次做完；后续轮用新标注做小步增量。
LONG_EPOCHS = {"B": 600, "C": 1200}
C_LONG_CHUNK = 1200
INCR_EPOCHS = {"B": 60, "C": 20}
INCR_LR = {"B": 0.001, "C": 0.001}
# 门禁死区：分类用绝对 macro-F1，回归用相对 RMSE
GATE_MARGIN = {"A": 0.01, "B": 0.005, "C": 0.001}
# A 的漂移策略：候选不得比父低超过 A_GATE_TOL，才优先最新（不再是“≥0.60 就晋升”）
A_GATE_FLOOR = 0.60
A_GATE_TOL = 0.0
# B 最差轴承 / C 最差子集 与 NASA 的尾部护栏
B_WORST_TOL = 1.02
C_WORST_TOL = 1.02
C_NASA_TOL = 1.10
CLIP_MAX = {"C": 350.0}


def _log(msg):
    sys.stderr.write(msg + "\n")


# ------------------------------------------------------------------ checkpoint
def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.keys()}
    out["meta"] = json.loads(str(out["metadata"]))
    return out


def save_ckpt(model, path):
    arrays = {k: v for k, v in model.items() if k != "meta"}
    arrays["metadata"] = json.dumps(model["meta"], ensure_ascii=False, allow_nan=False)
    np.savez(path, **arrays)


def _finite(model):
    for k, v in model.items():
        if k == "meta":
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.all(np.isfinite(v)):
            raise ValueError("nonfinite model state: %s" % k)


# ------------------------------------------------------------------ features
def features_with_paths(inputs):
    """与官方 baselines/data.py::flatten 同序；非数值/非有限叶子直接拒绝。"""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [["key", key]], depth + 1)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + [["index", i]], depth + 1)
        elif isinstance(value, bool):
            raise ValueError("boolean input leaf is not allowed")
        elif isinstance(value, (int, float)):
            f = float(value)
            if not math.isfinite(f):
                raise ValueError("non-finite input leaf")
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(f)
        elif value is None:
            raise ValueError("null input leaf is not allowed")
        elif isinstance(value, str):
            raise ValueError("string input leaf is not allowed")
        else:
            raise ValueError("unsupported input leaf type")

    visit(inputs, [], 0)
    return np.asarray(values, dtype=np.float64), paths


# ------------------------------------------------------------------ training
def train_epochs(model, X, Y, epochs, lr, batch=128, task="classification"):
    n = len(X)
    if n < 2:
        raise RuntimeError("too few training rows: %d" % n)
    xn = np.clip((X - model["mean"]) / model["scale"], -10, 10)
    is_cls = task == "classification"
    yt = Y.astype(int) if is_cls else (Y - model["meta"]["target_mean"]) / model["meta"]["target_scale"]
    rng = np.random.default_rng(1000 + int(model["meta"].get("epochs", 0)))
    for ep in range(int(epochs)):
        prog = ep / max(1, epochs)
        cur_lr = lr * (1.0 - 0.95 * prog)
        model["meta"]["learning_rate"] = float(cur_lr)
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            if len(idx) < 2:
                continue
            h = np.tanh(xn[idx] @ model["w1"] + model["b1"])
            logits = h @ model["w2"] + model["b2"]
            if is_cls:
                z = logits - logits.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                p[np.arange(len(idx)), yt[idx]] -= 1.0
                delta = p / len(idx)
            else:
                err = logits[:, 0] - yt[idx]
                delta = (2.0 * err / len(idx))[:, None]
            grads = {"w2": h.T @ delta, "b2": delta.sum(axis=0)}
            hd = (delta @ model["w2"].T) * (1.0 - h ** 2)
            grads.update(w1=xn[idx].T @ hd, b1=hd.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            model["meta"]["steps"] = int(model["meta"].get("steps", 0)) + 1
            t = model["meta"]["steps"]
            for name, g in grads.items():
                g = g * min(1.0, 5.0 / max(norm, 1e-12))
                model["adam_first_" + name] = 0.9 * model["adam_first_" + name] + 0.1 * g
                model["adam_second_" + name] = 0.999 * model["adam_second_" + name] + 0.001 * g ** 2
                f1 = model["adam_first_" + name] / (1 - 0.9 ** t)
                f2 = model["adam_second_" + name] / (1 - 0.999 ** t)
                model[name] = model[name] - cur_lr * f1 / (np.sqrt(f2) + 1e-8)
        model["meta"]["epochs"] = int(model["meta"].get("epochs", 0)) + 1


# ------------------------------------------------------------------ /state
def _merge_state(model):
    p = Path("/state") / "model_state.npz"
    if not p.is_file():
        return 0
    try:
        st = load_ckpt(str(p))
    except Exception as exc:
        _log("state unreadable, ignored: %r" % (exc,))
        return 0
    if st.get("meta", {}).get("paths") != model["meta"].get("paths"):
        _log("state path mismatch, ignored")
        return 0
    if int(st["meta"].get("epochs", 0)) > int(model["meta"].get("epochs", 0)):
        _log("state newer than parent, ignored")
        return 0
    for name in ("w1", "b1", "w2", "b2"):
        if name not in st or st[name].shape != model[name].shape:
            _log("state shape mismatch on %s, ignored" % name)
            return 0
    for k in list(model.keys()):
        if k.startswith("adam_") and k in st and st[k].shape == model[k].shape:
            if not np.all(np.isfinite(st[k])):
                _log("state nonfinite on %s, ignored" % k)
                return 0
            model[k] = st[k]
    model["meta"]["steps"] = max(int(model["meta"].get("steps", 0)),
                                 int(st["meta"].get("steps", 0)))
    return 1


# ------------------------------------------------------------------ scoring
def _forward_logits(model, inputs):
    vec, _paths = features_with_paths(inputs)
    xn = np.clip((np.asarray(vec, dtype=np.float64) - model["mean"]) / model["scale"], -10, 10)
    h = np.tanh(xn @ model["w1"] + model["b1"])
    return (h @ model["w2"] + model["b2"]).reshape(-1)


def _score_rows(model, rows, task, metric):
    """按合同指标打分：macro_f1 / accuracy / rmse。只读公开 validation。"""
    gold, pred, errs = [], [], []
    classes = model["meta"].get("classes") or []
    for r in rows:
        y = r.get("answer")
        if y is None:
            continue
        v = _forward_logits(model, r.get("inputs"))
        if task == "regression":
            p = max(0.0, float(v[0]) * model["meta"]["target_scale"] + model["meta"]["target_mean"])
            errs.append((p - float(y)) ** 2)
        else:
            idx = int(np.argmax(v))
            gold.append(str(y))
            pred.append(str(classes[idx]) if idx < len(classes) else str(idx))
    _score_rows.last_rows = len(errs) if task == "regression" else len(gold)
    if task == "regression":
        if not errs:
            return None, None
        return "rmse", math.sqrt(sum(errs) / len(errs))
    if not gold:
        return None, None
    if metric == "accuracy":
        return "accuracy", sum(1 for a, b in zip(gold, pred) if a == b) / len(gold)
    labels = sorted(set(gold) | set(pred))
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(gold, pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(gold, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(gold, pred) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return "macro_f1", sum(f1s) / len(f1s) if f1s else 0.0


def _pick(d, names):
    if not isinstance(d, dict):
        return None
    for n in names:
        v = d.get(n)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    for v in d.values():
        if isinstance(v, dict):
            got = _pick(v, names)
            if got is not None:
                return got
    return None


class _CkptModel:
    """把 SDK 的 dict checkpoint 适配成 dp.cosine / dp.b_engine 需要的 Model 接口。"""

    def __init__(self, ck):
        self.ck = ck
        self.params = {k: ck[k] for k in ("w1", "b1", "w2", "b2")}
        self.first = {k: ck["adam_first_" + k] for k in ("w1", "b1", "w2", "b2")}
        self.second = {k: ck["adam_second_" + k] for k in ("w1", "b1", "w2", "b2")}
        self.meta = ck["meta"]
        self.meta.setdefault("seed", 1)
        self.meta.setdefault("steps", 0)
        self.meta.setdefault("epochs", 0)

    def normalized(self, x):
        return np.clip((np.asarray(x, dtype=np.float64) - self.ck["mean"]) / self.ck["scale"],
                       -10, 10)

    def forward(self, xn):
        h = np.tanh(xn @ self.params["w1"] + self.params["b1"])
        return h, h @ self.params["w2"] + self.params["b2"]

    def predict(self, x):
        _h, logits = self.forward(self.normalized(x))
        return logits.reshape(-1) * self.meta["target_scale"] + self.meta["target_mean"]

    def validate(self):
        for name, arr in self.params.items():
            if not np.all(np.isfinite(arr)):
                raise ValueError("nonfinite param %s" % name)
        for name in ("mean", "scale"):
            if not np.all(np.isfinite(self.ck[name])):
                raise ValueError("nonfinite %s" % name)
        lr = self.meta.get("learning_rate")
        if lr is not None and not (0.0 < float(lr) <= 1.0):
            raise ValueError("learning_rate out of range: %r" % (lr,))

    def export(self):
        for k in ("w1", "b1", "w2", "b2"):
            self.ck[k] = self.params[k]
            self.ck["adam_first_" + k] = self.first[k]
            self.ck["adam_second_" + k] = self.second[k]
        self.ck["meta"] = self.meta


def _reg_rows(rows, profile):
    X, Y, G = [], [], []
    for r in rows:
        y = r.get("answer")
        if y is None:
            continue
        vec, _paths = features_with_paths(r.get("inputs"))
        X.append(vec)
        Y.append(float(y))
        meta = r.get("meta") or {}
        G.append(str(meta.get("bearing") if profile == "B" else
                     meta.get("source_subset", "0")))
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(G)


def _reg_metrics(model, rows, profile):
    X, Y, G = _reg_rows(rows, profile)
    p = np.maximum(model.predict(X).reshape(-1), 0.0)
    cmax = model.meta.get("clip_max")
    if cmax:
        p = np.minimum(p, float(cmax))
    d = p - Y
    rmse = float(np.sqrt(np.mean(d ** 2)))
    mae = float(np.mean(np.abs(d)))
    dn = np.clip(np.where(d < 0, -d / 13.0, d / 10.0), 0.0, 50.0)
    nasa = float(np.sum(np.exp(dn) - 1.0))
    groups = {}
    for g in np.unique(G):
        m = G == g
        groups[str(g)] = float(np.sqrt(np.mean(d[m] ** 2)))
    return {"rmse": rmse, "mae": mae, "nasa": nasa,
            "worst": max(groups.values()) if groups else rmse,
            "groups": groups, "n": int(len(Y))}


def _train_b_long(model, rows):
    import dp.b_engine as be
    X, Y, G = _reg_rows(rows, "B")
    model.meta["batch_size"] = 126
    be.fit_variant(model, X, Y, G, int(LONG_EPOCHS["B"]), 0.025, 126,
                   be.build_cfg("raw"))
    model.meta["long_train_done"] = True
    model.meta["long_epochs"] = int(LONG_EPOCHS["B"])


def _train_c_long(model, rows, seed=7000, epochs=None, epoch_offset=0):
    from dp.cosine import onecycle_train
    X, Y, G = _reg_rows(rows, "C")
    total = int(LONG_EPOCHS["C"])
    epochs = int(epochs if epochs is not None else total)
    onecycle_train(model, X, Y, epochs=epochs, peak_lr=0.018,
                   warmup_frac=0.10, lr_end=1.8e-5, batch=256, seed=int(seed),
                   groups=G, target_cap=350, epoch_offset=int(epoch_offset), total=total,
                   shape="step60", start_lr=0.0)
    model.meta["clip_max"] = float(CLIP_MAX["C"])


def _gate(rnd, parent, cand, task, dataset):
    """返回 (verdict, detail)。verdict ∈ pass/reject/unknown；unknown 一律保留父。"""
    contract_metric = str((rnd.task or {}).get("metric") or "")
    direction = str((rnd.task or {}).get("metric_direction") or "")
    try:
        rows = list(rnd.data.validation_rows())
    except Exception as exc:
        return "unknown", "validation unavailable (%r)" % (exc,)
    if len(rows) < 30:
        return "unknown", "validation too small (%d)" % len(rows)
    if task == "regression" and dataset in ("B", "C"):
        try:
            p = _reg_metrics(_CkptModel(parent), rows, dataset)
            c = _reg_metrics(_CkptModel(cand), rows, dataset)
        except Exception as exc:
            return "unknown", "tail gate error (%r)" % (exc,)
        margin = GATE_MARGIN.get(dataset, 0.005)
        better = c["rmse"] < p["rmse"] * (1.0 - margin)
        worse = c["rmse"] > p["rmse"] * (1.0 + margin)
        tol = B_WORST_TOL if dataset == "B" else C_WORST_TOL
        tail_ok = c["worst"] <= p["worst"] * tol
        if dataset == "C":
            tail_ok = tail_ok and c["nasa"] <= p["nasa"] * C_NASA_TOL
        detail = ("parent rmse=%.6f nasa=%.4g worst=%.6f vs cand rmse=%.6f "
                  "nasa=%.4g worst=%.6f on %d rows, margin %.4f, tail_ok=%s"
                  % (p["rmse"], p["nasa"], p["worst"], c["rmse"], c["nasa"],
                     c["worst"], c["n"], margin, tail_ok))
        if not tail_ok:
            return "reject", detail + " [tail guard]"
        if worse:
            return "reject", detail
        return ("pass_better" if better else "pass_tie"), detail
    try:
        p_metric, p_val = _score_rows(parent, rows, task, contract_metric)
        c_metric, c_val = _score_rows(cand, rows, task, contract_metric)
    except Exception as exc:
        return "unknown", "gate error (%r)" % (exc,)
    if p_val is None or c_val is None:
        return "unknown", "validation has no answers"
    margin = GATE_MARGIN.get(dataset, 0.02)
    if direction == "higher":
        better = (c_val - p_val) > margin
        worse = (p_val - c_val) > margin
    elif direction == "lower":
        rel = (c_val - p_val) / max(abs(p_val), 1e-9)
        better = rel < -margin
        worse = rel > margin
    else:
        # 平台未给 direction 时按指标名推断（rmse/mae/loss → lower；f1/accuracy/auc → higher）
        name = contract_metric.lower()
        if any(k in name for k in ("rmse", "mae", "mse", "loss", "mape")):
            rel = (c_val - p_val) / max(abs(p_val), 1e-9)
            better, worse = rel < -margin, rel > margin
        elif any(k in name for k in ("f1", "accuracy", "acc", "auc")):
            better, worse = (c_val - p_val) > margin, (p_val - c_val) > margin
        else:
            return "unknown", "metric_direction unknown (%r)" % (direction,)
    scored = int(getattr(_score_rows, "last_rows", 0) or len(rows))
    detail = ("parent %s=%.6f vs cand %s=%.6f on %d rows, margin %.4f"
              % (p_metric, p_val, c_metric, c_val, scored, margin))
    if dataset == "A":
        # 容忍降幅：候选不得比父低超过 A_GATE_TOL；满足才取最新（不再“≥0.60 就晋升”）。
        if c_val < A_GATE_FLOOR:
            return "reject", detail + " [A below floor %.2f]" % A_GATE_FLOOR
        if c_val < p_val - A_GATE_TOL:
            return "reject", detail + " [A tolerance: drop %.4f > %.4f]" % (
                p_val - c_val, A_GATE_TOL)
        if better:
            return "pass_better", detail
        return "promote_latest", detail + " [A drop within tolerance, prefer latest]"
    if worse:
        return "reject", detail
    return ("pass_better" if better else "pass_tie"), detail


submitted_global = {}


# ------------------------------------------------------------------ train
def train(rnd):
    global submitted_global
    submitted_global = {}
    dataset = str(rnd.dataset_key)
    parent_id = None
    try:
        parent_id = rnd.parent.artifact_id
    except Exception:
        pass

    def finish_parent(reason, cfg=None, seed=None, submitted=False):
        if not rnd.call_ids:
            try:
                rnd.model([{"role": "user", "content": "保留父产物：" + reason}],
                          max_tokens=min(8, rnd.max_output_tokens), temperature=0)
            except Exception as exc:
                _log("fallback model call failed: %r" % (exc,))
        rnd.finish(selected_artifact_id=parent_id,
                   hypothesis=("已提交候选但本轮不选" if submitted else "本轮不提交新候选"),
                   change_summary=("候选已提交，门禁拒绝后保留父产物" if submitted
                                   else "保留父产物，不做权重更新"),
                   reason=reason,
                   configuration=cfg or {"dataset": dataset},
                   seed=int(seed if seed is not None else
                            os.environ.get("CONTEST_SEED", "1")),
                   history_ids=[],
                   model_call_ids=list(rnd.call_ids))

    try:
        parent_path = rnd.parent.path("checkpoint.npz")
        parent = load_ckpt(str(parent_path))
        model = load_ckpt(str(parent_path))
    except Exception as exc:
        finish_parent("父产物不可读或解析失败，保留父产物：%r" % (exc,))
        return

    declared = str(model["meta"].get("task") or "classification")
    asked = (rnd.task or {}).get("task") if isinstance(rnd.task, dict) else None
    task = str(asked or declared)
    if task != declared:
        raise RuntimeError("task mismatch: contract=%r checkpoint=%r" % (task, declared))

    used_state = _merge_state(model)

    try:
        rows = list(rnd.data.training_rows())
        classes = model["meta"].get("classes") or []
        expect_paths = model["meta"].get("paths")
        X, Y, skipped = [], [], 0
        for r in rows:
            y = r.get("answer")
            if isinstance(y, str):
                y = float(classes.index(y)) if y in classes else None
            if y is None:
                skipped += 1
                continue
            vec, paths = features_with_paths(r.get("inputs"))
            if expect_paths is not None and paths != expect_paths:
                raise RuntimeError("feature schema mismatch on row %r" % (r.get("id"),))
            X.append(vec)
            Y.append(float(y))
    except Exception as exc:
        finish_parent("训练行读取/解析失败，保留父产物：%r" % (exc,))
        return

    if len(X) < 2:
        finish_parent("本轮可用训练行 %d 条（跳过未知标签 %d），保留父产物" % (len(X), skipped))
        return

    round_index = int(getattr(rnd, "round_index", 1) or 1)
    wall_seconds = getattr(rnd, "wall_seconds", None)
    train_seed = int(os.environ.get("CONTEST_SEED", "1"))
    t_train = time.monotonic()
    if dataset in ("B", "C"):
        if model["meta"].get("long_train_done"):
            # 长训完成后，用本轮新标注做小步增量；C 继承 target_cap=350 与 v15 warm lr 序列。
            epochs = int(INCR_EPOCHS[dataset])
            warm = False
            if dataset == "C":
                base_lr = 0.0009 * (0.9 ** max(0, round_index - 2))
                Y_train = np.clip(np.asarray(Y), 0.0, 350.0)
            else:
                base_lr = 0.001
                Y_train = np.asarray(Y)
            lr = base_lr
            train_seed = 1000 + int(model["meta"].get("epochs", 0) or 0)
            try:
                train_epochs(model, np.asarray(X), Y_train,
                             epochs=epochs, lr=lr, task=task)
                _finite(model)
            except Exception as exc:
                finish_parent("增量训练失败，保留父产物：%r" % (exc,),
                              {"dataset": dataset, "epochs": 0,
                               "long_train_done": True, "wall_seconds": wall_seconds})
                return
        else:
            # 父产物还没长训过：一次性长训；R1 被拒/失败后下一轮换 seed 重试。
            try:
                cm = _CkptModel(model)
                model["meta"]["seed"] = int(model["meta"].get("seed", 1)) + max(0, round_index - 1)
                if dataset == "B":
                    train_seed = int(model["meta"].get("seed", 1))
                    _train_b_long(cm, rows)
                else:
                    train_seed = 7000 + max(0, round_index - 1)
                    progress = int(model["meta"].get("long_progress", 0) or 0)
                    chunk = min(C_LONG_CHUNK, int(LONG_EPOCHS["C"]) - progress)
                    _train_c_long(cm, rows, seed=train_seed, epochs=chunk,
                                  epoch_offset=progress)
                    progress += chunk
                    cm.meta["long_progress"] = progress
                    cm.meta["long_epochs"] = progress
                    cm.meta["long_train_done"] = progress >= int(LONG_EPOCHS["C"])
                    cm.meta["seed"] = train_seed
                cm.export()
                _finite(model)
            except Exception as exc:
                finish_parent("长训失败，保留父产物：%r" % (exc,),
                              {"dataset": dataset, "epochs": 0,
                               "long_train_done": False, "wall_seconds": wall_seconds})
                return
            if dataset == "B":
                epochs = int(LONG_EPOCHS["B"])
            else:
                epochs = chunk
            base_lr = 0.025 if dataset == "B" else 0.018
            lr = base_lr
            warm = False
    else:
        ep_raw = os.environ.get("CONTEST_EPOCHS")
        lr_raw = os.environ.get("CONTEST_LR")
        epochs = int(ep_raw) if (ep_raw is not None and ep_raw.strip() != "") else None
        base_lr = float(lr_raw) if (lr_raw is not None and lr_raw.strip() != "") else None
        parent_epochs = int(model["meta"].get("epochs", 0) or 0)
        warm = parent_epochs > OFFICIAL_EPOCH_LIMIT
        if epochs is None:
            epochs = (int(os.environ.get("CONTEST_WARM_EPOCHS", WARM_EPOCHS.get(dataset, 20)))
                      if warm else EPOCHS_BY_KEY.get(dataset, 3))
        if base_lr is None:
            base_lr = (float(os.environ.get("CONTEST_WARM_LR", WARM_LR.get(dataset, 0.001)))
                       if warm else LR_BY_KEY.get(dataset, 0.003))
        if epochs <= 0:
            finish_parent("本轮配方为 0 步（%s 已收敛），保留父产物" % dataset,
                          {"dataset": dataset, "epochs": 0, "wall_seconds": wall_seconds})
            return
        lr = base_lr * (0.9 ** max(0, round_index - 1))
        train_seed = 1000 + int(model["meta"].get("epochs", 0) or 0)
        try:
            train_epochs(model, np.asarray(X), np.asarray(Y), epochs=epochs, lr=lr, task=task)
            _finite(model)
        except Exception as exc:
            finish_parent("A 训练失败，保留父产物：%r" % (exc,), seed=train_seed)
            return

    train_seconds = time.monotonic() - t_train
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    cand = work / "candidate"
    try:
        shutil.rmtree(cand, ignore_errors=True)
        cand.mkdir(parents=True, exist_ok=True)
        save_ckpt(model, str(cand / "checkpoint.npz"))
    except Exception as exc:
        finish_parent("工作目录不可写，保留父产物：%r" % (exc,))
        return
    meta_round = {"dataset": dataset, "task": task, "train_rows": int(len(Y)),
                  "skipped_rows": int(skipped), "epochs_this_round": int(epochs),
                  "lr_base": float(base_lr), "lr_applied": float(lr),
                  "round_index": round_index, "warm": bool(warm),
                  "state_used": bool(used_state),
                  "long_train_done": bool(model["meta"].get("long_train_done")),
                  "wall_seconds": wall_seconds,
                  "train_seconds": round(float(train_seconds), 3)}
    (cand / "meta.json").write_text(json.dumps(meta_round, ensure_ascii=False), encoding="utf-8")

    # ---- 先提交（每轮都有平台测量）----
    try:
        submitted = rnd.submit_candidate(
            cand,
            hypothesis="父权重续训 %d epoch（%s，lr %.4f）" % (epochs, dataset, lr),
            change_summary="在父 checkpoint 上做 %d 个 epoch Adam 续训（保留优化器状态）" % epochs,
            configuration=meta_round,
            seed=int(train_seed),
            checkpoint_paths=["checkpoint.npz"], code_paths=[], predict_argv=PREDICT_ARGV)
        submitted_global = submitted or {}
    except Exception as exc:
        finish_parent("候选提交失败，保留父产物：%r" % (exc,), meta_round)
        return

    verdict, detail = _gate(rnd, parent, model, task, dataset)
    meta_round["gate"] = "local_validation"
    meta_round["gate_verdict"] = verdict
    meta_round["gate_detail"] = detail

    try:
        receipt = rnd.model(
            [{"role": "user", "content": "本轮 %s 门禁=%s（%s）" % (dataset, verdict, detail)}],
            max_tokens=min(32, rnd.max_output_tokens), temperature=0)
    except Exception as exc:
        finish_parent("门禁回执模型调用失败，保留父产物：%r" % (exc,), meta_round,
                      seed=train_seed, submitted=True)
        return

    if verdict in ("pass_better", "promote_latest"):
        artifact_id = (submitted or {}).get("artifact_id")
        if not artifact_id:
            finish_parent("候选回执缺 artifact_id，保留父产物", meta_round, seed=train_seed,
                          submitted=True)
            return
        state_dir = work / "state_round"
        try:
            shutil.rmtree(state_dir, ignore_errors=True)
            ws = rnd.state_workspace(state_dir)
            save_ckpt(model, str(ws / "model_state.npz"))
            rnd.save_state(ws)
        except Exception as exc:
            _log("state save skipped: %r" % (exc,))
        _why = "优于父" if verdict == "pass_better" else "不劣于父、优先最新"
        rnd.finish(
            selected_artifact_id=artifact_id,
            hypothesis="父权重续训 %d epoch（%s）" % (epochs, dataset),
            change_summary="在父 checkpoint 上做 %d 个 epoch Adam 续训" % epochs,
            reason="门禁%s（%s）；模型调用号 %s" % (_why, detail, receipt.call_id),
            configuration=meta_round,
            seed=int(train_seed),
            history_ids=[],
            model_call_ids=list(rnd.call_ids))
        return

    # tie / reject / unknown：保留父产物；候选已提交（平台已测量）但不选，删本地候选、state 不更新
    shutil.rmtree(cand, ignore_errors=True)
    _tag = {"reject": "拒绝", "unknown": "未验证", "pass_tie": "死区内持平（未超阈值）"}.get(verdict, verdict)
    finish_parent("门禁%s（%s），保留父产物" % (_tag, detail), meta_round,
                  seed=train_seed, submitted=True)


# ------------------------------------------------------------------ predict
_CACHE = {}


def _model(path):
    if path not in _CACHE:
        _CACHE[path] = load_ckpt(path)
    return _CACHE[path]


def predict(sample, model_dir):
    ckpt = os.environ.get("CONTEST_ARTIFACT_PATH") or os.path.join(model_dir, "checkpoint.npz")
    model = _model(ckpt)
    want = str(model["meta"].get("task") or "classification")
    got = str(sample.task)
    if got != want:
        raise RuntimeError("predict task mismatch: request=%r checkpoint=%r" % (got, want))
    vec, paths = features_with_paths(sample.inputs)
    expect = model["meta"].get("paths")
    if expect is not None and paths != expect:
        raise RuntimeError("predict feature schema mismatch (got %d leaves, expected %d)"
                           % (len(paths), len(expect)))
    xn = np.clip((np.asarray(vec, dtype=np.float64) - model["mean"]) / model["scale"], -10, 10)
    h = np.tanh(xn @ model["w1"] + model["b1"])
    logits = (h @ model["w2"] + model["b2"]).reshape(-1)
    if want == "regression":
        val = float(logits[0] * model["meta"]["target_scale"] + model["meta"]["target_mean"])
        if not math.isfinite(val):
            raise RuntimeError("nonfinite regression output")
        val = max(0.0, val)
        cmax = model["meta"].get("clip_max")
        if cmax:
            val = min(val, float(cmax))
        return {"value": val}
    z = np.exp(logits - logits.max())
    probs = z / z.sum()
    if not np.all(np.isfinite(probs)):
        raise RuntimeError("nonfinite classification output")
    best = int(np.argmax(probs))
    probs[best] = float(probs[best]) + 1e-9
    probs = probs / probs.sum()
    classes = model["meta"].get("classes") or ["c%d" % i for i in range(len(probs))]
    return [{"label": str(classes[i]), "score": float(probs[i])} for i in range(len(probs))]
