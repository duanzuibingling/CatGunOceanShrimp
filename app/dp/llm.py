"""大模型规划器：可插拔、可离线、可记账。

设计原则：
1. **LLM 是可选件**。没有 Key、没有网络、平台不给端点时，规则策略独立跑完全程。
2. **Token 是预算项**。每次调用都记进账本，直接对应评分里的“单位增益成本”。
3. 只做两件事：读诊断报告 → 给出动作方案；写完方案就固化成技能卡，下次不再调用。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResult:
    ok: bool
    plan: Dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    latency: float = 0.0
    backend: str = "none"
    error: str = ""


class TokenLedger:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def add(self, purpose: str, tokens: int, latency: float, backend: str):
        self.calls.append({"purpose": purpose, "tokens": tokens,
                           "latency": round(latency, 3), "backend": backend})

    @property
    def total(self) -> int:
        return sum(c["tokens"] for c in self.calls)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def summary(self) -> Dict[str, Any]:
        return {"calls": self.n_calls, "tokens": self.total}


class BaseProvider:
    name = "base"

    def plan(self, report, context: Dict[str, Any]) -> LLMResult:
        raise NotImplementedError


class MockProvider(BaseProvider):
    """离线可用：按诊断类型返回一份“合理但模板化”的方案，并模拟 Token 消耗。

    它存在的意义：在没有平台账号时，把 Harness 的调用链路、账本、技能沉淀全部跑通；
    换成真模型时，只需替换这一个类。
    """

    name = "mock"

    def __init__(self, tokens_per_call: int = 1200):
        self.tokens_per_call = tokens_per_call

    def plan(self, report, context: Dict[str, Any]) -> LLMResult:
        t0 = time.perf_counter()
        kind = getattr(report, "kind", "covariate")
        if kind == "concept":
            steps = [{"action": "replay_posttrain",
                      "params": {"buffer_per_class": 120, "max_buffer_batches": 5}},
                     {"action": "shadow_gate", "params": {"min_delta": -0.01}}]
            why = "概念漂移下旧样本的标签关系已失效，需扩大回放权重并放宽门禁容差"
        elif kind == "covariate":
            steps = [{"action": "replay_posttrain",
                      "params": {"buffer_per_class": 80, "max_buffer_batches": 4}},
                     {"action": "shadow_gate", "params": {"min_delta": -0.005}}]
            why = "协变量漂移，重训后需做特征标准化复核"
        else:
            steps = [{"action": "noop", "params": {}}]
            why = "证据不足，先观察一批再说（不动手也是决策）"
        latency = 0.05 + 0.02 * len(steps)
        time.sleep(0)  # 演示用，不真的等待
        return LLMResult(ok=True,
                         plan={"why": why, "steps": steps, "kind": kind},
                         tokens=self.tokens_per_call,
                         latency=round(time.perf_counter() - t0 + latency, 3),
                         backend=self.name)


class OpenAICompatProvider(BaseProvider):
    """对接平台模型服务（PyArrow / vLLM / 任意 OpenAI 兼容端点）。

    赛题限制：**只能调用 PyArrow 平台上的两个模型**
      - qwen3.8-27b（推荐，能力最强，Token 贵）
      - qwen3.5-4b （挑战选项，Token 便宜；用 4B 成功则资源分/客观分更高）
    默认走 4B：token 成本约为 27B 的 1/5，且是加分项。
    """

    name = "openai-compat"
    SUPPORTED = ("qwen3.5-4b", "qwen3.8-27b")

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 20.0,
                 max_tokens_per_run: int = 20000):
        self.base_url = (base_url or os.environ.get("DP_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("DP_LLM_API_KEY", "")
        self.model = model or os.environ.get("DP_LLM_MODEL", "qwen3.5-4b")
        self.timeout = timeout
        self.max_tokens_per_run = max_tokens_per_run
        self.used_tokens = 0

    def available(self) -> bool:
        return bool(self.base_url)

    def budget_left(self) -> int:
        return max(0, self.max_tokens_per_run - self.used_tokens)

    def plan(self, report, context: Dict[str, Any]) -> LLMResult:
        if not self.available():
            return LLMResult(ok=False, backend=self.name, error="no endpoint")
        if self.budget_left() <= 0:
            return LLMResult(ok=False, backend=self.name, error="token budget exhausted")
        try:
            import requests
            prompt = _build_prompt(report, context)
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            plan = json.loads(_extract_json(text))
            used = int(usage.get("total_tokens", 0))
            self.used_tokens += used
            return LLMResult(ok=True, plan=plan, tokens=used,
                             latency=0.0, backend=self.name)
        except Exception as e:
            return LLMResult(ok=False, backend=self.name, error=str(e)[:200])


def _build_prompt(report, context) -> str:
    return (
        "你是工业模型运维专家。下面是漂移诊断报告，请输出 JSON："
        '{"why": "一句话解释", "steps": [{"action": "...", "params": {...}}]}。\n'
        "可选动作：noop / replay_posttrain / full_retrain。必须考虑成本。\n"
        f"诊断：kind={getattr(report, 'kind', '?')}, score={getattr(report, 'score', 0)}, "
        f"top_features={getattr(report, 'top_features', [])}\n"
        f"上下文：{json.dumps(context, ensure_ascii=False)[:500]}"
    )


def _extract_json(text: str) -> str:
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if i >= 0 and j > i else "{}"


def make_provider(mode: str = "auto") -> BaseProvider:
    """mode: mock / openai / auto（有端点用端点，否则 mock，再否则空）。"""
    if mode == "mock":
        return MockProvider()
    prov = OpenAICompatProvider()
    if mode == "openai":
        return prov
    return prov if prov.available() else MockProvider()
