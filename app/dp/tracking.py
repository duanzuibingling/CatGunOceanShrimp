"""实验追踪：优先 SwanLab，其次本地 JSONL（离线评测环境没有网络）。

设计原则：**追踪失败绝不能拖垮实验**。任何异常都降级为本地记录。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class Tracker:
    def __init__(self, project: str = "driftpilot", run_name: str = "",
                 config: Optional[Dict[str, Any]] = None, outdir: str = "results",
                 mode: str = "auto"):
        self.project = project
        self.run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
        self.config = config or {}
        self.mode = mode
        self.step = 0
        self.backend = "local"
        self.swan = None
        self.outdir = outdir

        os.makedirs(outdir, exist_ok=True)
        self.local_path = os.path.join(outdir, "track.jsonl")
        self._local_fp = open(self.local_path, "a", encoding="utf-8")
        self._write_local({"event": "init", "run": self.run_name, "config": self.config})

        if mode in ("auto", "swanlab"):
            self._try_swanlab()

    def _try_swanlab(self):
        try:
            import swanlab  # noqa: F401
        except Exception as e:
            if self.mode == "swanlab":
                print(f"[tracker] swanlab 不可用，降级为本地记录：{e}")
            return
        try:
            import swanlab
            # 有 key 或已登录时走云端；否则走离线模式，避免阻塞
            key = os.environ.get("SWANLAB_API_KEY")
            logdir = os.path.join(self.outdir, "swanlog")
            os.makedirs(logdir, exist_ok=True)
            # 赛题要求：SwanLab 记录必须【公开】，评委才能访问
            self.swan = swanlab.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                mode="cloud" if key else "offline",
                log_dir=logdir,
                public=True,
            )
            self.backend = "swanlab"
            print(f"[tracker] SwanLab 已启用（{'cloud' if key else 'offline'}，public=True）：{self.run_name}")
        except Exception as e:
            if self.mode == "swanlab":
                print(f"[tracker] SwanLab 初始化失败，降级为本地记录：{e}")
            self.swan = None

    def _write_local(self, payload: Dict[str, Any]):
        self._local_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._local_fp.flush()

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.step = step if step is not None else self.step + 1
        self._write_local({"event": "log", "step": self.step, "metrics": metrics})
        if self.swan is not None:
            try:
                import swanlab
                swanlab.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                            step=self.step)
            except Exception:
                pass

    def finish(self, summary: Optional[Dict[str, Any]] = None):
        if summary:
            self._write_local({"event": "finish", "summary": summary})
        self._local_fp.close()
        if self.swan is not None:
            try:
                import swanlab
                swanlab.finish()
            except Exception:
                pass
