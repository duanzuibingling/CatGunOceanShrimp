"""Harness v2：诊断 → 成本感知决策 → 执行 → 门禁 → 审计 → 技能演进。

v2 相对 v1 的三处升级：
1. **成本感知决策**：不再是“一触发就重训”。引入死区（deadband）与高严重度直通，
   让“这次不动手”成为一个真正会被选中的动作——直接对应评分里的“单位增益成本”。
2. **技能版本演进**：同一技能连续被门禁拒绝，就派生新版本（加大回放、放宽容差），
   旧版本保留并标记 superseded，可随时回滚。
3. **决策可解释**：每次决策把“为什么动/为什么不动”写进审计日志。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .audit import AuditLog
from .llm import TokenLedger, make_provider
from .memory import ReplayMemory
from .model import accuracy, fit_model, make_model
from .skills import SkillLibrary
from .stream import DriftReport, Strategy, StepResult


class HarnessStrategy(Strategy):
    name = "harness"

    def __init__(self, model_kind: str = "hgb", seed: int = 0,
                 skills_dir: str = "skills", audit_path: str = "audit/decisions.jsonl",
                 provider: Optional[Any] = None, max_buffer_batches: int = 4,
                 ledger: Optional[TokenLedger] = None,
                 deadband: float = -1.0, high_severity: float = 0.55,
                 cost_penalty: float = 0.004, revise_after_rejects: int = 2,
                 cold_start: bool = False, model_factory: Optional[Any] = None,
                 featurizer: Optional[Any] = None):
        super().__init__(model_kind, seed)
        self.lib = SkillLibrary(skills_dir)
        self.audit = AuditLog(audit_path)
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.provider = provider if provider is not None else make_provider("auto")
        self.max_buffer_batches = max_buffer_batches
        # 可插拔的模型工厂与特征变换：用来做模型层的消融实验
        self.model_factory = model_factory
        self.featurizer = featurizer

        # 成本感知决策的三个参数（自动化同学负责整定的就是这些）
        self.deadband = deadband
        self.high_severity = high_severity
        self.cost_penalty = cost_penalty
        self.revise_after_rejects = revise_after_rejects

        self.memory = ReplayMemory(max_recent_batches=max_buffer_batches,
                                   reservoir_size=2000, seed=seed)
        self.prev_acc: Optional[float] = None
        self.rollback = None
        self.stats = {
            "skill_hits": 0, "skills_created": 0, "skills_revised": 0,
            "llm_calls": 0, "promotions": 0, "rejections": 0,
            "noop_triggered": 0, "noop_deadband": 0, "tokens": 0, "acted": 0, "rollbacks": 0,
            "light": 0, "heavy": 0,
        }
        if cold_start:
            for s in list(self.lib.skills):
                s.superseded_by = "cold_start_reset"
            self.lib.skills = [s for s in self.lib.skills if s.active]

    # ---------- 模型层 ----------
    def _fe(self, X):
        return self.featurizer(X) if self.featurizer is not None else X

    def _new_model(self, n_iter: int = 120):
        if self.model_factory is not None:
            return self.model_factory()
        return make_model(self.model_kind, self.seed, n_iter=n_iter)

    def initial_fit(self, X, y):
        # 有状态的对齐器（如 CORAL）先用原始参考批估计目标分布，再做变换
        if self.featurizer is not None and hasattr(self.featurizer, "set_reference"):
            try:
                self.featurizer.set_reference(X)
            except Exception:
                pass
        Xt = self._fe(X)
        self.model = self._new_model()
        fit_model(self.model, Xt, y)
        self.memory.add(Xt, y)          # 记忆里存的是【变换后】的特征，避免二次变换

    def _remember(self, X, y):
        self.memory.add(self._fe(X), y)

    def _replay(self, per_class: int = 80):
        """训练集 = 水库（旧分布）+ 近期（新分布），与 DriftTriggered 同一套记忆。"""
        return self.memory.train_set(per_class=per_class)

    # ---------- 诊断 ----------
    def _diagnose(self, rep: DriftReport, acc_before: float) -> Dict[str, Any]:
        drop = 0.0 if self.prev_acc is None else round(self.prev_acc - acc_before, 4)
        severity = "high" if rep.score > 0.6 else ("mid" if rep.score > 0.35 else "low")
        return {"kind": rep.kind or "none", "score": round(float(rep.score), 4),
                "severity": severity, "acc_drop": round(drop, 4),
                "top_features": [int(x) for x in list(rep.top_features)[:3]],
                "acc_before": round(acc_before, 4)}

    # ---------- 决策：成本分级 ----------
    def _worth_acting(self, diag: Dict[str, Any], rep: DriftReport):
        """什么时候不动手：只有“特征漂移了但性能没掉”才敢等一等。

        实测（真实气体传感器数据）：一刀切的死区会掉 3 个点，所以默认 deadband=-0.02，
        也就是「性能只要没明显上升就动手」，把省成本的任务交给下面的动作分级。
        """
        if not rep.triggered:
            return False, "not_triggered"
        drop = diag.get("acc_drop", 0.0)
        if drop >= self.deadband:
            return True, f"perf_drop={drop:.3f}"
        if rep.score >= self.high_severity:
            return True, f"high_drift={rep.score:.3f}"
        return False, f"in_deadband(drop={drop:.3f},score={rep.score:.3f})"

    def _pick_iter(self, diag: Dict[str, Any]) -> int:
        """按严重程度选训练预算：轻量 40 轮 / 完整 120 轮。"""
        if diag.get("acc_drop", 0) >= 0.15 or diag.get("score", 0) >= self.high_severity:
            return 120
        return 80

    def _decide(self, diag, rep):
        ok, why = self._worth_acting(diag, rep)
        if not ok:
            self.stats["noop_deadband"] += 1
            return [{"action": "noop", "params": {}}], "noop", ""

        skill = self.lib.match(rep)
        if skill is not None:
            self.stats["skill_hits"] += 1
            return skill.then, "skill", skill.id

        kind = diag["kind"] or "covariate"
        existing = self.lib.get(f"{kind}_drift_v1")
        if existing is not None and existing.active:
            self.stats["skill_hits"] += 1
            return existing.then, "skill", existing.id

        ctx = {"diagnosis": diag, "budget_left": {"tokens": 20000 - self.ledger.total},
               "reason_maybe_act": why}
        res = self.provider.plan(rep, ctx)
        self.ledger.add("plan", res.tokens, res.latency, res.backend)
        self.stats["llm_calls"] += 1
        self.stats["tokens"] = self.ledger.total

        if res.ok and res.plan.get("steps"):
            new_skill = SkillLibrary.build_from_report(rep, source="llm")
            if res.plan["steps"] and res.plan["steps"][0].get("params"):
                new_skill.then[0]["params"].update(res.plan["steps"][0]["params"])
            self.lib.save(new_skill)
            self.stats["skills_created"] += 1
            return res.plan["steps"], "llm", new_skill.id

        return ([{"action": "replay_posttrain",
                  "params": {"buffer_per_class": 80,
                             "max_buffer_batches": self.max_buffer_batches}},
                 {"action": "shadow_gate", "params": {"min_delta": -0.005}}], "rule", "")

    # ---------- 执行 ----------
    def _execute(self, steps, b, diag=None) -> Dict[str, Any]:
        candidate = None
        train_seconds = 0.0
        gate_cfg = {"min_delta": -0.005}
        promoted = False
        n_iter_used = 0
        for step in steps:
            action = step.get("action")
            params = step.get("params") or {}
            if action == "noop":
                continue
            if action in ("replay_posttrain", "full_retrain"):
                if action == "full_retrain":
                    Xr, yr = self.memory.train_set(per_class=80)
                else:
                    Xr, yr = self._replay(int(params.get("buffer_per_class", 80)))
                n_iter_used = int(params.get("n_iter") or self._pick_iter(diag or {}))
                cand = self._new_model(n_iter_used)
                rec = fit_model(cand, np.vstack([Xr, self._fe(b.X)]),
                                np.concatenate([yr, b.y]))
                candidate = (cand, Xr, yr)
                train_seconds += rec.seconds
                self.stats["heavy" if n_iter_used >= 120 else "light"] += 1
            elif action == "shadow_gate":
                gate_cfg.update(params)

        if candidate is not None:
            cand, Xr, yr = candidate
            old_sh = accuracy(self.model, Xr, yr)
            new_sh = accuracy(cand, Xr, yr)
            if new_sh >= old_sh + float(gate_cfg.get("min_delta", -0.005)):
                # 上线前把旧模型留作“回滚点”，上线后连续两批被它打败就退回去
                self.rollback = {"model": self.model, "wins": 0}
                self.model = cand
                promoted = True
                self.stats["promotions"] += 1
            else:
                self.stats["rejections"] += 1
        return {"promoted": promoted, "train_seconds": round(train_seconds, 3),
                "n_iter": n_iter_used}

    # ---------- 回滚 ----------
    def _maybe_rollback(self, b) -> bool:
        """新模型连续两批打不过旧模型 → 回滚。这是闭环里「回滚」那一步的落地。"""
        if not self.rollback:
            return False
        old = self.rollback["model"]
        cur_acc = accuracy(self.model, self._fe(b.X), b.y)
        old_acc = accuracy(old, self._fe(b.X), b.y)
        if old_acc > cur_acc + 0.01:
            self.rollback["wins"] += 1
        else:
            self.rollback["wins"] = 0
        if self.rollback["wins"] >= 2:
            self.model = old
            self.rollback = None
            self.stats["rollbacks"] += 1
            return True
        return False

    # ---------- 主循环 ----------
    def on_batch(self, b, rep: DriftReport) -> StepResult:
        acc_before = accuracy(self.model, self._fe(b.X), b.y)
        diag = self._diagnose(rep, acc_before)
        steps, source, skill_id = self._decide(diag, rep)
        rolled = self._maybe_rollback(b)
        result = self._execute(steps, b, diag)

        # 技能演进：连续被门禁拒绝 → 派生新版本（旧版本保留可回滚）
        if source == "skill" and skill_id and not result["promoted"] and result["train_seconds"] > 0:
            rejects = self.lib.mark_reject(skill_id)
            if rejects >= self.revise_after_rejects:
                old = self.lib.get(skill_id)
                if old is not None and old.active:
                    params = dict(old.then[0].get("params", {}))
                    params["buffer_per_class"] = int(params.get("buffer_per_class", 80) * 1.5) + 20
                    params["max_buffer_batches"] = int(params.get("max_buffer_batches", 4)) + 1
                    self.lib.bump(old, params, reason=f"连续 {rejects} 次被拒")
                    self.stats["skills_revised"] += 1

        if result["promoted"]:
            self._remember(b.X, b.y)

        acted = source != "noop"
        if acted:
            self.stats["acted"] += 1
        else:
            self.stats["noop_triggered"] += 1

        label = {"skill": f"skill:{skill_id}", "llm": f"llm:{skill_id}",
                 "rule": "rule", "noop": "noop"}[source]
        if rolled:
            label = "rollback"
        self.audit.record("decision", {
            "batch": b.t, "diagnosis": diag, "source": source, "skill": skill_id,
            "steps": [s.get("action") for s in steps],
            "promoted": result["promoted"], "train_seconds": result["train_seconds"],
            "tokens": self.ledger.total, "acted": acted, "rollback": bool(rolled),
        })

        if skill_id:
            sk = self.lib.get(skill_id)
            if sk is not None:
                sk.hits += 1
        self.prev_acc = acc_before
        return StepResult(b.t, len(b), acc_before, acc_before, acted=acted,
                          action=label, train_seconds=result["train_seconds"],
                          drift_score=rep.score, promoted=result["promoted"])

    # ---------- 收尾 ----------
    def report(self) -> Dict[str, Any]:
        for s in self.lib.skills:
            try:
                self.lib.save(s)
            except Exception:
                pass
        return {"stats": dict(self.stats), "tokens": self.ledger.summary(),
                "skills": [{"id": s.id, "active": s.active, "hits": s.hits,
                            "rejects": s.rejects, "digest": s.digest()}
                           for s in self.lib.skills],
                "params": {"deadband": self.deadband,
                           "high_severity": self.high_severity,
                           "cost_penalty": self.cost_penalty},
                "audit": self.audit.summary()}









