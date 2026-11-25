"""官方协议下的 Harness：观测 → 决策 → 续训 → 门禁 → 回滚 → 审计。

与 `stream.py` 里那些「自己训 HGB」的策略相比，这一层的差别只有一句：
**动作层不改模型，只发起一轮官方续训**（`baselines.train --resume`）。
协议顺序严格不可颠倒：先用当前模型预测本批 → 拿到标签后才允许用它训练。

四个关键时刻：
  1. 要不要续训 —— 性能掉超过阈值，或特征 KS 漂移，才动手；否则 noop（省算力）。
  2. 续训成什么样 —— 从当前 checkpoint resume，训练集 = 历史已见 + 本批 80%。
  3. 敢不敢上线 —— 本批留出 20% 当裁判，候选打不过现役就不晋升。
  4. 不行怎么退 —— 保留晋升前的 checkpoint；新模型连续两批打不过它，退回。

审计：每一步写 `audit.jsonl`（哈希链），每一轮写 `round_summary.json`。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audit import AuditLog
from .official import OfficialPackage, batch_by_meta
from .stream import Monitor


# ==================== DP_A_BATCH_WEIGHT 补丁（work/safe_image/patch_a_batch_weight.py）====================
BATCH_WEIGHT_ENV = "DP_A_BATCH_WEIGHT"


def _batch_weight() -> int:
    """读环境变量开关：1（默认/回滚）= 现状；2 = docs/60 的 d4_w2；上限 3。

    任何非法值都退化成 1（镜像里绝不能因为一个环境变量训练崩掉）。
    """
    raw = os.environ.get(BATCH_WEIGHT_ENV, "1").strip().lower()
    if raw in ("", "0", "off", "false", "no", "none"):
        return 1
    try:
        weight = int(float(raw))
    except Exception:
        return 1
    return max(1, min(3, weight))


def _weight_rows(rows: Sequence[dict], weight: int = 1) -> List[dict]:
    """把本批样本复制 weight-1 份，实现「近期批 ×w」的样本加权。

    官方 `baselines/data.py::read_dataset` 要求 sample ID **非空且唯一**，
    所以复制行**必须重铸 id**（只改 id，inputs/answer/meta 原样不动）。
    """
    weight = max(1, int(weight))
    out = list(rows)
    if weight <= 1:
        return out
    for k in range(2, weight + 1):
        for i, row in enumerate(rows):
            dup = dict(row)
            base = row.get("id")
            if not isinstance(base, str) or not base:
                base = f"row{i:07d}"
            dup["id"] = f"{base}#w{k}"
            out.append(dup)
    return out
# ==================== 补丁结束 ====================



# ---------------- 小工具 ----------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_inputs(inputs: Any) -> List[float]:
    """把 `inputs` 拍平成数值向量。

    顺序与官方 `baselines/data.py::flatten` 完全一致（dict 按 key 排序、list 按下标），
    这样 KS 统计看到的特征顺序和模型输入一致。只用来做漂移检测，不参与训练。
    """
    values: List[float] = []

    def visit(value, depth: int) -> None:
        if depth > 16:
            raise ValueError("inputs 嵌套过深")
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            values.append(number if math.isfinite(number) else 0.0)
        else:
            raise ValueError(f"非数值特征：{type(value).__name__}")

    visit(inputs, 0)
    return values


def feature_matrix(rows: Sequence[dict]) -> np.ndarray:
    feats = [flatten_inputs(r["inputs"]) for r in rows]
    width = min(len(f) for f in feats) if feats else 0
    return np.asarray([f[:width] for f in feats], dtype=float)


def build_batches(rows: List[dict], meta_key: str = "source_batch",
                  sort_key: str = "", n_batches: int = 8) -> List[Tuple[str, List[dict]]]:
    """分两路：A 用 meta 里的月份/批次天然分批；B/C 按 sort_key 排序后等分。"""
    if not sort_key:
        return batch_by_meta(rows, meta_key)
    ordered = sorted(rows, key=lambda r: float(r["meta"][sort_key]))
    n = max(1, n_batches)
    step = max(1, len(ordered) // n)
    batches: List[Tuple[str, List[dict]]] = []
    for i in range(n):
        seg = ordered[i * step:(i + 1) * step] if i < n - 1 else ordered[i * step:]
        if seg:
            batches.append((f"{i + 1}", seg))
    return batches


# ---------------- 结果 ----------------
def _worst(scores: Sequence[float], higher_better: bool) -> float:
    if not scores:
        return float("nan")
    return float(min(scores)) if higher_better else float(max(scores))


class OfficialHarness:
    """在官方 A/B/C 包上跑的闭环 Harness。

    参数（括号内为默认值，也是 docs/19 里那张表的口径）：
      epochs(3)           每轮续训的 epoch 数
      gate_frac(0.2)      本批留出多少当门禁裁判
      gate_tol(0.005)     门禁容忍度：允许候选小幅退步，超过就拒
      perf_drop(None)     性能下降阈值；None 时按任务取 0.08（分类）/0.10（回归相对值）
      ks_threshold(0.35)  特征漂移阈值（复用 stream.Monitor）
      rollback_patience(2) 新模型连续几批打不过旧的就退回
      rollback_margin(0.01) 「打不过」的判定幅度
      backoff_after_rejects(2) 连续被拒几次后歇一批（不再立刻重训）
    """

    def __init__(self, package: OfficialPackage, out_dir: str, init_ckpt: str,
                 epochs: int = 3, gate_frac: float = 0.2, gate_tol: float = 0.005,
                 min_train_rows: int = 100, min_gate_rows: int = 30,
                 perf_drop: Optional[float] = None, ks_threshold: float = 0.35,
                 ks_quantile: float = 0.9, rollback_patience: int = 2,
                 rollback_margin: float = 0.01, backoff_after_rejects: int = 2,
                 seed_rows: Optional[Sequence[dict]] = None, seed: int = 0,
                 run_id: str = "", verbose: bool = True):
        self.pkg = package
        self.out_dir = out_dir
        self.init_ckpt = init_ckpt
        self.epochs = int(epochs)
        self.min_train_rows = int(min_train_rows)
        self.min_gate_rows = int(min_gate_rows)
        self.epochs_base = int(epochs)
        self.epoch_scale = 1.0
        self.consec_reject = 0
        self.gate_frac = float(gate_frac)
        self.gate_tol = float(gate_tol)
        self.perf_drop = (0.08 if package.task == "classification" else 0.10) \
            if perf_drop is None else float(perf_drop)
        self.rollback_patience = int(rollback_patience)
        self.rollback_margin = float(rollback_margin)
        self.backoff_after_rejects = int(backoff_after_rejects)
        self.seed = int(seed)
        self.verbose = verbose

        self.higher_better = package.task == "classification"
        self.metric_name = "macro_f1" if self.higher_better else "rmse"

        os.makedirs(os.path.join(out_dir, "rounds"), exist_ok=True)
        self.audit = AuditLog(os.path.join(out_dir, "audit.jsonl"), run_id=run_id or
                              os.path.basename(os.path.normpath(out_dir)))

        # 现役 / 回滚点
        self.ckpt = init_ckpt
        self.rollback_ckpt: Optional[str] = None
        self.rollback_wins = 0
        self.reject_streak = 0          # 连续被门禁拒绝的次数
        self.backoff_left = 0           # 还要歇几批

        # 漂移参照：上一次「被接受」的分布与性能
        self.monitor = Monitor(threshold=float(ks_threshold), feature_quantile=float(ks_quantile),
                               perf_drop=self.perf_drop)
        self.ref_perf: Optional[float] = None
        self._reference_locked = False

        # 历史（已见标签）；可选用官方公开训练集垫底，防止遗忘早期分布
        self.seen: List[dict] = list(seed_rows or [])
        self.seed_row_count = len(self.seen)

        self.static: Dict[str, Any] = {}
        self.batches_log: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "promotions": 0, "rejections": 0, "rollbacks": 0,
            "noop_no_drift": 0, "noop_backoff": 0,
            "noop_small_batch": 0, "noop_small_gate": 0, "train_failures": 0,
            "train_seconds": 0.0, "gate_seconds": 0.0, "predict_seconds": 0.0,
            "train_rounds": 0, "epochs_used": 0, "early_stops": 0, "adaptive_short": 0,
        }
        self._started = 0.0

    # ---------- 基础动作 ----------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _score(self, ckpt: str, rows: Sequence[dict]) -> Tuple[float, float]:
        """返回 (指标, 预测耗时)。指标方向由任务决定（F1 越高越好 / RMSE 越低越好）。"""
        if not rows:
            raise ValueError("空批次无法评分")
        began = time.perf_counter()
        preds = self.pkg.predict(ckpt, rows)
        seconds = time.perf_counter() - began
        return float(self.pkg.score(rows, preds)), seconds

    def _improvement(self, new: float, old: float) -> float:
        """正数 = 变好，统一成「越高越好」的一维口径。回归用相对变化，避免量纲干扰。"""
        if self.higher_better:
            return new - old
        scale = max(abs(old), 1e-9)
        return (old - new) / scale

    def _acceptable(self, cand: float, base: float) -> bool:
        if self.higher_better:
            return cand >= base - self.gate_tol
        return cand <= base * (1.0 + self.gate_tol)

    def _pick_epochs(self, drift: dict) -> int:
        """轮数自适应：漂移越强练得越多，漂移弱就少练。

        依据 drift 里的 KS 分与真实性能下降两路信号。
        """
        ks = float(drift.get("ks", 0.0))
        drop = float(drift.get("perf_drop", 0.0))
        import os as _os
        strong = float(_os.environ.get("ADAPT_STRONG", "3.0"))
        mid = float(_os.environ.get("ADAPT_MID", "1.0"))
        weak = float(_os.environ.get("ADAPT_WEAK", "1.0"))
        if ks > 0.6 or drop > 0.25:
            mult = strong
        elif ks > 0.4 or drop > 0.12:
            mult = mid
        else:
            mult = weak
        n = int(round(self.epochs_base * mult * self.epoch_scale))
        n = max(1, min(n, self.epochs_base * 3))
        if n < self.epochs_base:
            self.stats["adaptive_short"] += 1
        return n

    def _perf_drop(self, score: float) -> float:
        """相对参照性能的退步量；正数表示变差。分类用绝对差，回归用相对比例。"""
        if self.ref_perf is None:
            return 0.0
        return -self._improvement(score, self.ref_perf)

    def _split_gate(self, rows: Sequence[dict], key: str) -> Tuple[List[dict], List[dict]]:
        """把本批切成「训练 80% / 裁判 20%」，用批次 key 派生种子，保证可复现。"""
        n = len(rows)
        if n < 5:
            return list(rows), list(rows[:1])
        idx = list(range(n))
        random.Random(f"{self.seed}:{key}").shuffle(idx)
        k = max(1, min(n - 1, int(round(n * self.gate_frac))))
        gate = set(idx[:k])
        gate_rows = [rows[i] for i in range(n) if i in gate]
        train_rows = [rows[i] for i in range(n) if i not in gate]
        return train_rows, gate_rows

    def _ckpt_path(self, round_dir: str) -> str:
        return os.path.join(round_dir, "ckpt", "checkpoint.npz")

    def _fresh_round_dir(self, index: int, key: str) -> str:
        """官方 train 要求 output 是新目录；残留目录先删，删不掉就换名字。"""
        round_dir = os.path.join(self.out_dir, "rounds", f"r{index:02d}_{key}")
        if os.path.isdir(round_dir):
            shutil.rmtree(round_dir, ignore_errors=True)
        if os.path.isdir(round_dir):
            round_dir = f"{round_dir}_{int(time.time()) % 100000}"
        return round_dir

    # ---------- 静态基线 ----------
    def run_static(self, batches: Sequence[Tuple[str, List[dict]]]) -> Dict[str, Any]:
        """同一初始 checkpoint、零续训：这就是「静态不续训」这条对照线。"""
        curve, total = [], 0.0
        for key, rows in batches:
            score, secs = self._score(self.init_ckpt, rows)
            total += secs
            curve.append({"batch": key, "n": len(rows), "score": round(score, 4)})
            self._log(f"  [static] batch {key:<4} n={len(rows):<5} {self.metric_name}={score:.4f}")
        scores = [c["score"] for c in curve]
        self.static = {
            "mean": round(float(np.mean(scores)), 4),
            "worst": round(_worst(scores, self.higher_better), 4),
            "last": round(scores[-1], 4),
            "seconds": round(total, 2),
            "curve": curve,
        }
        self.stats["predict_seconds"] += total
        self.audit.record("static_baseline", {"metric": self.metric_name, **{
            k: self.static[k] for k in ("mean", "worst", "last", "seconds")}})
        return self.static

    # ---------- 主循环 ----------
    def run(self, batches: Sequence[Tuple[str, List[dict]]],
            static: bool = True) -> Dict[str, Any]:
        self._started = time.perf_counter()
        batches = list(batches)
        if not batches:
            raise ValueError("没有批次可跑")

        self.audit.record("run_start", {
            "profile": self.pkg.profile, "task": self.pkg.task, "metric": self.metric_name,
            "init_ckpt": self.init_ckpt, "init_digest": _sha256_file(self.init_ckpt),
            "n_batches": len(batches), "batch_sizes": [len(b[1]) for b in batches],
            "epochs": self.epochs, "gate_frac": self.gate_frac, "gate_tol": self.gate_tol,
            "perf_drop": self.perf_drop, "ks_threshold": self.monitor.threshold,
            "rollback_patience": self.rollback_patience, "seed_rows": len(self.seen),
            "env": {"pythonpath": os.environ.get("PYTHONPATH", ""),
                    "tmp": os.environ.get("TMP", ""),
                    "python": sys.executable},
        })
        self._log(f"[{self.pkg.profile}] {sum(len(b[1]) for b in batches)} 行 → "
                  f"{len(batches)} 批 {[len(b[1]) for b in batches]}")

        if static:
            self.run_static(batches)

        # 漂移参照：优先用垫底的官方训练集；没有就用第一批
        if self.seen:
            ref_X = feature_matrix(self.seen)
        else:
            ref_X = feature_matrix(batches[0][1])
        self.monitor.set_reference(ref_X)

        for index, (key, rows) in enumerate(batches, start=1):
            self.batches_log.append(self._run_batch(index, key, rows))

        return self._finish()

    # ---------- 单批 ----------
    def _run_batch(self, index: int, key: str, rows: List[dict]) -> Dict[str, Any]:
        t = index
        round_dir = self._fresh_round_dir(index, key)
        X = feature_matrix(rows)
        entry: Dict[str, Any] = {"batch": key, "index": index, "n": len(rows),
                                 "round_dir": round_dir}

        # ---- ① 观测：先用现役模型预测、评分（标签只在预测之后才用）----
        score_before, predict_secs = self._score(self.ckpt, rows)
        self.stats["predict_seconds"] += predict_secs
        entry["score_before"] = round(score_before, 4)
        entry["predict_seconds"] = round(predict_secs, 3)
        self.audit.record("observe", {
            "batch": key, "n": len(rows), "score_before": round(score_before, 4),
            "metric": self.metric_name, "ckpt_digest": _sha256_file(self.ckpt),
            "n_seen": len(self.seen),
        })

        # ---- ② 回滚判定：现役 vs 上一轮 checkpoint（用本批做裁判）----
        rolled_back = self._maybe_rollback(rows, key, entry)
        if rolled_back:
            # 回滚后现役模型变了，本批的分数按退回后的模型重算（诚实地报告用户实际体验）
            score_before, predict_secs = self._score(self.ckpt, rows)
            self.stats["predict_seconds"] += predict_secs
            entry["score_after_rollback"] = round(score_before, 4)

        # ---- ③ 决策：要不要续训 ----
        rep = self.monitor.observe(t, X)          # KS 那一路（triggered=特征漂移）
        drop = self._perf_drop(score_before)
        perf_drift = drop > self.perf_drop
        triggered = bool(rep.triggered or perf_drift)
        kind = ("covariate" if rep.triggered else "concept") if triggered else "none"
        entry["drift"] = {"ks": round(float(rep.score), 4), "kind": kind,
                          "feat_drift": bool(rep.triggered),
                          "perf_drop": round(float(drop), 4),
                          "perf_drift": bool(perf_drift),
                          "triggered": triggered}

        entry["epochs_used"] = self._pick_epochs(entry["drift"])

        if rolled_back:
            action, promoted = "rollback", False
        elif triggered and len(rows) < self.min_train_rows:
            # 批太小：门禁分辨率不足，续训风险大于收益（实验结论 32 号文档）
            self.stats["noop_small_batch"] += 1
            entry["skip_reason"] = (f"batch too small: {len(rows)} < "
                                    f"{self.min_train_rows}")
            action, promoted = "noop_small_batch", False
            self.audit.record("skip_small_batch", {
                "batch": key, "n": len(rows), "min_train_rows": self.min_train_rows})
        elif self.backoff_left > 0:
            self.backoff_left -= 1
            self.stats["noop_backoff"] += 1
            action, promoted = "noop_backoff", False
        elif not triggered:
            self.stats["noop_no_drift"] += 1
            action, promoted = "noop", False
        else:
            action, promoted = self._train_and_gate(
                index, key, rows, entry, epochs=entry["epochs_used"])

        entry["action"] = action
        entry["promoted"] = bool(promoted)

        # ---- ④ 收尾：标签已到手，本批进入历史 ----
        self.seen.extend(rows)
        if promoted:
            self.monitor.set_reference(X)      # 接受新分布为参照，避免「永久漂移」误触发
            self.reject_streak = 0
            self.ref_perf = score_before if self.ref_perf is None else (
                max(self.ref_perf, score_before) if self.higher_better
                else min(self.ref_perf, score_before))
            self.rollback_wins = 0
        elif action in ("rejected_by_gate",):
            self.reject_streak += 1
            if self.reject_streak >= max(1, self.backoff_after_rejects):
                self.backoff_left = 1
                self.reject_streak = 0
        elif self.ref_perf is None:
            # 首批没有参照性能：把当前水平记为参照，后续才能算「掉了多少」
            self.ref_perf = score_before

        self._write_round_summary(round_dir, entry)
        self.audit.record("round_commit", {
            "batch": key, "action": action, "promoted": bool(promoted),
            "score_before": entry["score_before"], "drift": entry["drift"],
            "train_seconds": entry.get("train_seconds", 0.0),
            "ckpt_digest": _sha256_file(self.ckpt), "n_seen": len(self.seen),
        })
        self._log(f"  [{index}/{len(self.batches_log)}] batch {key:<4} n={len(rows):<5} "
                  f"{self.metric_name}={score_before:.4f} ks={rep.score:.3f} "
                  f"drop={drop:+.3f} → {action}"
                  + (f" ({entry['train_seconds']}s)" if entry.get("train_seconds") else ""))
        return entry

    # ---------- 续训 + 门禁 ----------
    def _train_and_gate(self, index: int, key: str, rows: List[dict],
                        entry: Dict[str, Any], epochs: int = 0) -> Tuple[str, bool]:
        train_rows, gate_rows = self._split_gate(rows, key)
        # ---- DP_A_BATCH_WEIGHT 补丁：本批载荷 ×w（w=1 时与打补丁前逐位一致）----
        batch = _weight_rows(train_rows, _batch_weight())
        payload = list(self.seen) + batch
        round_dir = entry["round_dir"]
        basis = self.ckpt

        began = time.perf_counter()
        try:
            use_epochs = int(epochs or self.epochs)
            self.stats["epochs_used"] += use_epochs
            report = self.pkg.train_round(payload, round_dir, resume=basis,
                                          epochs=use_epochs, val_rows=gate_rows)
        except Exception as exc:                      # 训练崩了也要有记录，不能静默
            elapsed = time.perf_counter() - began
            self.stats["train_failures"] += 1
            entry["train_seconds"] = round(elapsed, 3)
            entry["error"] = str(exc)[:400]
            self.audit.record("train_failed", {
                "batch": key, "error": str(exc)[:400], "train_rows": len(payload),
                "seconds": round(elapsed, 3), "kept_ckpt_digest": _sha256_file(self.ckpt)})
            self._log(f"    ! 训练失败，保持现役：{str(exc)[:120]}")
            return "train_failed", False
        train_secs = time.perf_counter() - began
        self.stats["train_seconds"] += train_secs
        self.stats["train_rounds"] += 1
        entry["train_seconds"] = round(train_secs, 3)
        entry["train_rows"] = len(payload)
        entry["gate_rows"] = len(gate_rows)
        entry["train_report"] = {k: report.get(k) for k in (
            "stage_id", "train_rows", "validation_rows", "initial_epochs", "final_epochs",
            "initial_steps", "final_steps", "training_seconds", "learning_rate",
            "parent_checkpoint_sha256", "checkpoint_sha256", "validation_before",
            "validation_after")}

        if len(gate_rows) < self.min_gate_rows:
            self.stats["noop_small_gate"] += 1
            entry["skip_reason"] = (f"gate too small: {len(gate_rows)} < "
                                    f"{self.min_gate_rows}")
            self.audit.record("skip_small_gate", {
                "batch": key, "n_gate": len(gate_rows), "n_rows": len(rows)})
            return "noop_small_gate", False

        cand = report["checkpoint"]
        # ---- 门禁：本批留出切片当裁判，候选 vs 现役 ----
        gate_began = time.perf_counter()
        try:
            old_gate, s1 = self._score(basis, gate_rows)
            new_gate, s2 = self._score(cand, gate_rows)
        except Exception as exc:
            self.stats["gate_seconds"] += time.perf_counter() - gate_began
            entry["gate"] = {"error": str(exc)[:300], "passed": False}
            self.audit.record("gate_error", {"batch": key, "error": str(exc)[:300]})
            return "gate_error", False
        gate_secs = time.perf_counter() - gate_began
        # 注意：s1/s2 的预测耗时已包含在 gate_secs 内，不能再加到 predict_seconds（否则重复计数）
        self.stats["gate_seconds"] += gate_secs

        passed = self._acceptable(new_gate, old_gate)
        gain = self._improvement(new_gate, old_gate)
        entry["gate"] = {"metric": self.metric_name, "n": len(gate_rows),
                         "old": round(old_gate, 4), "cand": round(new_gate, 4),
                         "gain": round(gain, 4), "tol": self.gate_tol,
                         "passed": bool(passed), "seconds": round(gate_secs, 3)}
        self.audit.record("gate", {
            "batch": key, "passed": bool(passed), "metric": self.metric_name,
            "old": round(old_gate, 4), "cand": round(new_gate, 4), "gain": round(gain, 4),
            "tol": self.gate_tol, "n_gate": len(gate_rows),
            "cand_digest": _sha256_file(cand), "basis_digest": _sha256_file(basis)})

        if passed:
            self.rollback_ckpt = basis          # 留好退路：上一轮 checkpoint
            self.rollback_wins = 0
            self.ckpt = cand
            self.stats["promotions"] += 1
            self.consec_reject = 0
            self.epoch_scale = min(1.0, self.epoch_scale * 1.25)  # 逐步恢复预算
            return "promoted", True
        # ---- 早停退避：连续被拒就缩短后续轮数 ----
        self.stats["rejections"] += 1
        self.consec_reject += 1
        if self.consec_reject >= 2:
            self.epoch_scale = max(0.25, self.epoch_scale / 2)
            self.stats["early_stops"] += 1
        if self.consec_reject >= 4 and self.backoff_left <= 0:
            self.backoff_left = 1          # 暂停一轮，别再烧算力
        entry["early_stop"] = {"consec_reject": self.consec_reject,
                               "epoch_scale": round(self.epoch_scale, 3)}
        return "rejected_by_gate", False

    # ---------- 回滚 ----------
    def _maybe_rollback(self, rows: List[dict], key: str, entry: Dict[str, Any]) -> bool:
        """新模型连续 `rollback_patience` 批打不过上一轮 checkpoint → 退回。"""
        if not self.rollback_ckpt:
            return False
        probe = rows if len(rows) <= 400 else rows[:: max(1, len(rows) // 400)]
        try:
            cur, s1 = self._score(self.ckpt, probe)
            old, s2 = self._score(self.rollback_ckpt, probe)
        except Exception as exc:
            self.audit.record("rollback_probe_error", {"batch": key, "error": str(exc)[:300]})
            return False
        self.stats["predict_seconds"] += s1 + s2

        better = self._improvement(old, cur) > self.rollback_margin
        self.rollback_wins = self.rollback_wins + 1 if better else 0
        info = {"batch": key, "n": len(probe), "current": round(cur, 4),
                "previous": round(old, 4), "previous_better": bool(better),
                "wins": self.rollback_wins, "patience": self.rollback_patience}
        entry["rollback_probe"] = info
        if self.rollback_wins < self.rollback_patience:
            self.audit.record("rollback_probe", {**info, "action": "hold"})
            return False

        self.ckpt, old_ckpt = self.rollback_ckpt, self.ckpt
        self.rollback_ckpt = None
        self.rollback_wins = 0
        self.ref_perf = old          # 参照性能回到退回后的模型水平
        self.monitor.set_reference(feature_matrix(rows))
        self.stats["rollbacks"] += 1
        self.audit.record("rollback", {**info, "action": "reverted",
                                       "reverted_to": _sha256_file(self.ckpt),
                                       "abandoned": _sha256_file(old_ckpt)})
        self._log(f"    ↺ 回滚：连续 {self.rollback_patience} 批旧模型更优 "
                  f"({old:.4f} vs {cur:.4f})，退回上一轮 checkpoint")
        return True

    # ---------- 落盘 ----------
    def _write_round_summary(self, round_dir: str, entry: Dict[str, Any]) -> None:
        os.makedirs(round_dir, exist_ok=True)
        payload = {
            "format": "dp.harness.round-summary.v1",
            "profile": self.pkg.profile, "task": self.pkg.task, "metric": self.metric_name,
            "higher_better": self.higher_better,
            "deployed_ckpt": self.ckpt, "deployed_digest": _sha256_file(self.ckpt),
            "rollback_ckpt": self.rollback_ckpt, "ref_perf": self.ref_perf,
            **entry,
        }
        with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    def _finish(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self._started
        scores = [b["score_before"] for b in self.batches_log]
        mean = float(np.mean(scores)) if scores else float("nan")
        worst = _worst(scores, self.higher_better)
        last = scores[-1] if scores else float("nan")

        final_dir = os.path.join(self.out_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        target = os.path.join(final_dir, "checkpoint.npz")
        shutil.copyfile(self.ckpt, target)
        deploy = {"format": "dp.harness.deploy.v1", "profile": self.pkg.profile,
                  "task": self.pkg.task, "metric": self.metric_name,
                  "source_ckpt": self.ckpt, "digest": _sha256_file(target),
                  "batches": len(self.batches_log), "epochs": self.epochs,
                  "promotions": self.stats["promotions"],
                  "rollbacks": self.stats["rollbacks"]}
        with open(os.path.join(final_dir, "deploy.json"), "w", encoding="utf-8") as f:
            json.dump(deploy, f, ensure_ascii=False, indent=2)

        delta = mean - self.static["mean"] if self.static else None
        if self.static and not self.higher_better:
            relative = (self.static["mean"] - mean) / max(abs(self.static["mean"]), 1e-9)
        elif self.static:
            relative = (mean - self.static["mean"]) / max(abs(self.static["mean"]), 1e-9)
        else:
            relative = None

        result = {
            "format": "dp.harness.metrics.v1",
            "profile": self.pkg.profile, "task": self.pkg.task,
            "metric": self.metric_name, "higher_better": self.higher_better,
            "run": {"out_dir": self.out_dir, "init_ckpt": self.init_ckpt,
                    "epochs": self.epochs, "gate_frac": self.gate_frac,
                    "gate_tol": self.gate_tol, "perf_drop": self.perf_drop,
                    "ks_threshold": self.monitor.threshold,
                    "rollback_patience": self.rollback_patience,
                    "backoff_after_rejects": self.backoff_after_rejects,
                    "seed_rows": self.seed_row_count,
                    "batches": len(self.batches_log),
                    "total_seconds": round(elapsed, 2)},
            "stream": {"mean": round(mean, 4), "worst": round(worst, 4),
                       "last": round(last, 4),
                       "mean_abs_gain": None if delta is None else round(delta, 4),
                       "relative_gain": None if relative is None else round(relative, 4)},
            "static": self.static,
            "counts": {k: self.stats[k] for k in
                       ("promotions", "rejections", "rollbacks", "noop_no_drift", "epochs_used",
                        "noop_backoff", "train_failures", "train_rounds")},
            "timing": {"train_seconds": round(self.stats["train_seconds"], 2),
                       "gate_seconds": round(self.stats["gate_seconds"], 2),
                       "predict_seconds": round(self.stats["predict_seconds"], 2),
                       "total_seconds": round(elapsed, 2)},
            "batches": self.batches_log,
            "final": {"checkpoint": target, "deploy": os.path.join(final_dir, "deploy.json"),
                      "digest": deploy["digest"]},
        }
        # docs/83 §九 P0-5：先把 run_end 写进审计链，再取 summary，否则 head/entries 会少最后一条
        self.audit.record("run_end", {
            "mean": result["stream"]["mean"], "worst": result["stream"]["worst"],
            "promotions": self.stats["promotions"], "rejections": self.stats["rejections"],
            "rollbacks": self.stats["rollbacks"], "seconds": round(elapsed, 2),
            "final_digest": deploy["digest"]})
        result["audit"] = self.audit.summary()
        with open(os.path.join(self.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        return result


# ---------------- 演示用的字典/文本输出 ----------------
def format_summary(result: Dict[str, Any]) -> str:
    metric = result["metric"]
    stream, static = result["stream"], result.get("static") or {}
    counts, timing = result["counts"], result["timing"]
    lines = [
        f"profile={result['profile']} task={result['task']} metric={metric} "
        f"batches={len(result['batches'])}",
        f"  Harness  : mean={stream['mean']:.4f} worst={stream['worst']:.4f} "
        f"last={stream['last']:.4f}",
    ]
    if static:
        lines.append(f"  Static   : mean={static['mean']:.4f} worst={static['worst']:.4f} "
                     f"last={static['last']:.4f}")
        lines.append(f"  Δ        : {stream['mean_abs_gain']:+.4f} "
                     f"({stream['relative_gain'] * 100:+.2f}%)")
    lines.append(f"  promoted={counts['promotions']} rejected={counts['rejections']} "
                 f"rollbacks={counts['rollbacks']} noop={counts['noop_no_drift']} "
                 f"backoff={counts['noop_backoff']} failed={counts['train_failures']}")
    lines.append(f"  time     : total={timing['total_seconds']}s "
                 f"train={timing['train_seconds']}s gate={timing['gate_seconds']}s")
    return "\n".join(lines)
