"""审计层：哈希链决策日志。每一次判断都可追溯、不可静默篡改。

这既是主观分「安全可控」的证据，也是复现实验的底账。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List


class AuditLog:
    def __init__(self, path: str, run_id: str = ""):
        self.path = path
        self.run_id = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._prev = "GENESIS"
        self._seq = 0
        # 续写：从已有文件尾部接上哈希链（多进程/多次实例化也不会断链）
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            e = json.loads(line)
                            self.entries.append(e)
                            self._prev = e["hash"]
                            self._seq = e["seq"]
            except Exception:
                pass

    @staticmethod
    def _digest(payload: Dict[str, Any], prev: str) -> str:
        raw = (prev + json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def record(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "payload": payload,
            "prev": self._prev,
        }
        entry["hash"] = self._digest(entry["payload"], self._prev)
        self._prev = entry["hash"]
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    # ---------- 校验 ----------
    def verify(self) -> bool:
        prev = "GENESIS"
        for e in self.entries:
            if e["prev"] != prev:
                return False
            if self._digest(e["payload"], prev) != e["hash"]:
                return False
            prev = e["hash"]
        return True

    def head(self) -> str:
        return self._prev

    def summary(self) -> Dict[str, Any]:
        return {"entries": len(self.entries), "head": self.head(), "run_id": self.run_id,
                "verified": self.verify()}
