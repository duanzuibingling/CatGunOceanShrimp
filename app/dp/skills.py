"""技能库：把一次诊断固化成可复用、可版本化、可回滚的策略资产。

技能不是 Prompt，是 YAML + 执行器绑定。同类漂移再次出现时走技能，零 Token。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Skill:
    id: str
    when: Dict[str, Any]
    then: List[Dict[str, Any]]
    budget: Dict[str, Any] = field(default_factory=dict)
    fallback: str = "escalate"
    source: str = "llm"          # llm / human / template
    created_at: str = ""
    hits: int = 0
    superseded_by: str = ""      # 被新版本取代后填新版本 id，匹配时跳过
    rejects: int = 0             # 被门禁拒绝的次数，连续拒绝会触发版本修订

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    @property
    def version(self) -> int:
        try:
            return int(str(self.id).rsplit("_v", 1)[1])
        except Exception:
            return 1

    @property
    def active(self) -> bool:
        return not self.superseded_by

    def to_yaml(self) -> str:
        d = asdict(self)
        return yaml.safe_dump(d, allow_unicode=True, sort_keys=False)

    def digest(self) -> str:
        raw = json.dumps({"when": self.when, "then": self.then}, sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]


class SkillLibrary:
    """技能库：从目录加载 YAML，匹配 drift_report，持久化新技能。"""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.skills: List[Skill] = []
        self.load()

    # ---------- 读写 ----------
    def load(self):
        self.skills = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(self.root, name), encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if not d or "id" not in d:
                continue
            self.skills.append(Skill(
                id=d["id"], when=d.get("when", {}), then=d.get("then", []),
                budget=d.get("budget", {}), fallback=d.get("fallback", "escalate"),
                source=d.get("source", "llm"), created_at=d.get("created_at", ""),
                hits=int(d.get("hits", 0)),
                superseded_by=d.get("superseded_by", "") or "",
                rejects=int(d.get("rejects", 0) or 0),
            ))

    def save(self, skill: Skill) -> str:
        path = os.path.join(self.root, f"{skill.id}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(skill.to_yaml())
        if all(s.id != skill.id for s in self.skills):
            self.skills.append(skill)
        return path

    def get(self, skill_id: str) -> Optional[Skill]:
        for s in self.skills:
            if s.id == skill_id:
                return s
        return None

    def next_version(self, kind: str) -> int:
        n = 1
        for s in self.skills:
            if s.id.startswith(f"{kind}_drift_v"):
                n = max(n, s.version + 1)
        return n

    # ---------- 匹配 ----------
    def match(self, report) -> Optional[Skill]:
        best, best_score = None, -1.0
        for s in self.skills:
            if not s.active:
                continue
            sc = self._score(s.when, report)
            if sc is not None and sc > best_score:
                best, best_score = s, sc
        return best

    # ---------- 版本演进 ----------
    def bump(self, skill: Skill, patch: Dict[str, Any], reason: str = "") -> Skill:
        """基于旧技能派生新版本：参数按 patch 调整，旧版本标记 superseded（可回滚）。"""
        new_version = skill.version + 1
        new_id = f"{skill.id.rsplit('_v', 1)[0]}_v{new_version}"
        then_new = json.loads(json.dumps(skill.then))
        if then_new and patch:
            then_new[0].setdefault("params", {}).update(patch)
        new_skill = Skill(id=new_id, when=dict(skill.when), then=then_new,
                          budget=dict(skill.budget), source=skill.source,
                          fallback=skill.fallback)
        new_skill.rejects = 0
        skill.superseded_by = new_id
        self.save(skill)
        self.save(new_skill)
        return new_skill

    def mark_reject(self, skill_id: str) -> int:
        s = self.get(skill_id)
        if s is None:
            return 0
        s.rejects += 1
        self.save(s)
        return s.rejects

    @staticmethod
    def _score(when: Dict[str, Any], report) -> Optional[float]:
        kind = when.get("kind")
        if kind and kind != "any" and kind != getattr(report, "kind", None):
            return None
        score = getattr(report, "score", 0.0)
        if when.get("min_score") is not None and score < float(when["min_score"]):
            return None
        if when.get("max_score") is not None and score > float(when["max_score"]):
            return None
        feats = when.get("feature_any")
        if feats:
            top = set(getattr(report, "top_features", []) or [])
            if not (set(int(x) for x in feats) & top):
                return None
        return score

    # ---------- 生成 ----------
    @staticmethod
    def build_from_report(report, action: str = "replay_posttrain", source: str = "llm",
                          version: int = 1) -> Skill:
        kind = getattr(report, "kind", "covariate") or "covariate"
        score = float(getattr(report, "score", 0.4))
        top = [int(x) for x in getattr(report, "top_features", [])[:2]]
        sid = f"{kind}_drift_v{version}"
        # 只有协变量漂移才用“命中特征”做匹配条件；概念漂移靠性能信号，锁特征会永远匹配不上
        feature_any = top if kind == "covariate" else []
        # 概念漂移由“真实性能下降”触发，KS 分数与它无关，所以不锁分数/特征；
        # 协变量漂移才用“命中特征 + 分数下限”做精确匹配。
        if kind == "covariate":
            when = {"kind": kind, "min_score": round(max(0.20, score - 0.15), 2),
                    "feature_any": top}
        else:
            when = {"kind": kind}
        skill = Skill(
            id=sid,
            when=when,
            then=[
                {"action": action, "params": {"buffer_per_class": 80, "max_buffer_batches": 4}},
                {"action": "shadow_gate", "params": {"min_delta": -0.005}},
            ],
            budget={"tokens": 0, "gpu_sec": 0, "labels": 0},
            source=source,
        )
        return skill


DEFAULT_TEMPLATE = """# 技能卡模板
id: concept_drift_v1
when:
  kind: concept          # covariate / concept / sensor_fault / any
  min_score: 0.30
  feature_any: [0, 1]    # 可选：命中的 top 特征
then:
  - action: replay_posttrain
    params: {buffer_per_class: 80, max_buffer_batches: 4}
  - action: shadow_gate
    params: {min_delta: -0.005}
budget: {tokens: 0, gpu_sec: 0, labels: 0}
fallback: escalate
source: human
"""
