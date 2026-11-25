"""CLI：在官方 A/B/C 包上跑完整 Harness（观测→决策→续训→门禁→回滚→审计）。

用法（A 的漂移流，8 批 = 3–10 月）：
    python scripts/run_official_harness.py --profile A \
        --stream data/official/A/public/stream_uci_months3_10.jsonl \
        --init data/official/A/checkpoint.npz --out results/harness_A --epochs 3

用法（B 无天然月份，按 snapshot 排序等分 8 批）：
    python scripts/run_official_harness.py --profile B \
        --stream data/official/B/public/validation.jsonl --sort-key snapshot \
        --n-batches 8 --init data/official/B/checkpoint.npz \
        --out results/harness_B --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dp.official import OfficialPackage, load_jsonl                     # noqa: E402
from dp.official_harness import (OfficialHarness, build_batches,       # noqa: E402
                                 format_summary)

# 官方 train.py 用 os.mkdir(mode=0o700) 建 staging 目录；DSH 的 Windows 沙箱里
# 这种「受保护 DACL」目录不继承工作区授权，子目录创建 / tempfile 清理都会
# WinError 5。垫片只把训练子进程的 mkdir mode 放宽成 0o777，不碰训练逻辑与产物。
# 详见 docs/19-Harness官方适配.md 第 6 节。
SHIM_SRC = '''"""训练子进程的 Windows 沙箱兼容垫片：仅放宽 os.mkdir 的权限位。

官方 baselines/train.py 有两处用 mode=0o700 建目录：
    staging.mkdir(mode=0o700) 与 tempfile.TemporaryDirectory(dir=output.parent)
在受限令牌沙箱里这类目录的 DACL 不继承工作区 ACE，创建子目录即 WinError 5。
本垫片把 mode 统一成 0o777（默认继承 ACL），不改训练数学、不改产物内容。
"""
import os

_orig_mkdir = os.mkdir


def _mkdir(path, mode=0o777, *args, **kwargs):
    return _orig_mkdir(path, 0o777, *args, **kwargs)


os.mkdir = _mkdir
'''


def _ensure_sandbox_shim() -> str:
    """把垫片放进 work/compat 并挂到 PYTHONPATH（只影响训练/预测子进程）。"""
    shim_dir = os.path.join(ROOT, "work", "compat")
    os.makedirs(shim_dir, exist_ok=True)
    shim_file = os.path.join(shim_dir, "sitecustomize.py")
    if not os.path.isfile(shim_file):
        with open(shim_file, "w", encoding="utf-8") as f:
            f.write(SHIM_SRC)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if shim_dir not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([shim_dir] + parts)
    return shim_file


def _abs(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="A", choices=["A", "B", "C"])
    ap.add_argument("--stream", default="", help="模拟隐藏流的 JSONL（默认用公开训练集）")
    ap.add_argument("--init", default="", help="初始 checkpoint（默认用包内 checkpoint.npz）")
    ap.add_argument("--out", default="", help="输出目录（默认 results/harness_<profile>）")
    ap.add_argument("--epochs", type=int, default=3, help="每轮续训 epoch 数")
    ap.add_argument("--meta-key", default="source_batch", help="按该 meta 字段分批")
    ap.add_argument("--sort-key", default="", help="按该 meta 字段排序后等分成批（如 snapshot）")
    ap.add_argument("--n-batches", type=int, default=8, help="sort-key 模式下的批数")
    ap.add_argument("--gate-frac", type=float, default=0.2, help="本批留出多少当门禁裁判")
    ap.add_argument("--gate-tol", type=float, default=0.005, help="门禁容忍度")
    ap.add_argument("--perf-drop", type=float, default=None,
                    help="性能下降阈值（分类绝对差 / 回归相对比例；默认 0.08 / 0.10）")
    ap.add_argument("--ks-threshold", type=float, default=0.35, help="特征 KS 漂移阈值")
    ap.add_argument("--rollback-patience", type=int, default=2,
                    help="新模型连续几批打不过旧的就退回")
    ap.add_argument("--rollback-margin", type=float, default=0.01, help="「打不过」的判定幅度")
    ap.add_argument("--backoff-after-rejects", type=int, default=2,
                    help="连续被门禁拒绝几次后歇一批")
    ap.add_argument("--seed", type=int, default=0, help="门禁切片/随机种子")
    ap.add_argument("--no-seed-rows", action="store_true",
                    help="不把官方公开训练集垫进历史（默认垫，防遗忘）")
    ap.add_argument("--no-static", action="store_true", help="跳过「静态不续训」对照")
    ap.add_argument("--no-sandbox-shim", action="store_true",
                    help="不挂 Windows 沙箱兼容垫片（无沙箱环境可关）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    shim = "" if args.no_sandbox_shim else _ensure_sandbox_shim()
    pkg = OfficialPackage(args.profile)
    stream_path = _abs(args.stream) or pkg.train_path
    init_ckpt = _abs(args.init) or pkg.baseline_ckpt
    out_dir = _abs(args.out) or os.path.join(ROOT, "results", f"harness_{args.profile}")

    for label, path in (("stream", stream_path), ("init ckpt", init_ckpt)):
        if not os.path.isfile(path):
            print(f"[error] {label} 不存在：{path}")
            return 2

    rows = load_jsonl(stream_path)
    if args.sort_key:
        batches = build_batches(rows, sort_key=args.sort_key, n_batches=args.n_batches)
    else:
        batches = build_batches(rows, meta_key=args.meta_key)
    if len(batches) < 3:
        print(f"[error] 只有 {len(batches)} 批，无法构造漂移流")
        return 2

    seed_rows = []
    if not args.no_seed_rows:
        seed_path = pkg.train_path
        if os.path.isfile(seed_path) and os.path.abspath(seed_path) != os.path.abspath(stream_path):
            seed_rows = load_jsonl(seed_path)

    # 每轮 train 的 output 目录必须是新目录：清掉上一轮的产物
    os.makedirs(out_dir, exist_ok=True)
    for stale in ("rounds", "final"):
        target = os.path.join(out_dir, stale)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            if os.path.isdir(target):      # 沙箱里可能留下无权限的残骸
                print(f"[warn] 无法完全清理 {target}，本轮改用带时间戳的轮次目录")
    for stale in ("audit.jsonl", "metrics.json"):
        target = os.path.join(out_dir, stale)
        try:
            if os.path.isfile(target):
                os.remove(target)
        except OSError as exc:
            print(f"[warn] 无法删除 {target}: {exc}")

    if not args.quiet:
        print(f"[{args.profile}] stream={os.path.relpath(stream_path, ROOT)} "
              f"({len(rows)} 行) → {len(batches)} 批 "
              f"{[len(b[1]) for b in batches]}")
        print(f"           init={os.path.relpath(init_ckpt, ROOT)} "
              f"epochs={args.epochs} seed_rows={len(seed_rows)} out={out_dir}")
        if shim:
            print(f"           sandbox-shim={os.path.relpath(shim, ROOT)}"
                  f"（仅放宽训练子进程的 mkdir 权限位，不改训练逻辑）")

    harness = OfficialHarness(
        pkg, out_dir=out_dir, init_ckpt=init_ckpt, epochs=args.epochs,
        gate_frac=args.gate_frac, gate_tol=args.gate_tol, perf_drop=args.perf_drop,
        ks_threshold=args.ks_threshold, rollback_patience=args.rollback_patience,
        rollback_margin=args.rollback_margin,
        backoff_after_rejects=args.backoff_after_rejects,
        seed_rows=seed_rows, seed=args.seed, verbose=not args.quiet)
    result = harness.run(batches, static=not args.no_static)

    print("\n" + format_summary(result))
    print(f"[out] {os.path.join(out_dir, 'metrics.json')}")
    print(f"[out] {result['final']['checkpoint']}")
    print(f"[audit] entries={result['audit']['entries']} verified={result['audit']['verified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
