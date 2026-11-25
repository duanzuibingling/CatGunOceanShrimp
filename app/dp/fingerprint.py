"""可复现性指纹：环境 + 代码 + 数据，三件套。

主观分里的「实验规范」和门槛里的「可审计、可复现」，靠的就是这份指纹：
任何一次结果，都能追到"哪台机器、哪个版本、哪份数据、哪个随机种子"。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from typing import Dict, List, Optional

KEY_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "rivern"]


def _pkg_version(name: str) -> str:
    try:
        import importlib
        m = importlib.import_module("sklearn" if name == "scikit-learn" else name.replace("-", "_"))
        return getattr(m, "__version__", "?")
    except Exception:
        return "missing"


def env_fingerprint() -> Dict[str, str]:
    pkgs = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "river"]:
        pkgs[p] = _pkg_version(p)
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu_count": str(os.cpu_count()),
        "packages": pkgs,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def file_sha256(path: str, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def data_fingerprint(paths: List[str]) -> Dict[str, str]:
    out = {}
    for p in paths:
        if os.path.isfile(p):
            out[os.path.basename(p)] = file_sha256(p)[:16] + f":{os.path.getsize(p)}"
    return out


def write_fingerprint(outdir: str, extra: Optional[Dict] = None) -> str:
    os.makedirs(outdir, exist_ok=True)
    payload = {"env": env_fingerprint()}
    if extra:
        payload.update(extra)
    path = os.path.join(outdir, "fingerprint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
