# 镜像 ↔ 源码一致性核对（final2b）

> 方法：对已交付镜像做**只读**检查，不改镜像、不改 tar。
> 时间：2026-11-24｜结论：**PASS**

## 1. 核对对象

| 项 | 值 |
|---|---|
| 镜像 | `driftpilot:sdk-final2b` |
| image id | `sha256:2f4014bb51450c3a9bebd5e072ad61907909df99db86b4cb20c1236a8b0bbb3e` |
| 层数 | 25 |
| tar | `dist/driftpilot-sdk-final2b.tar`，150,721,024 B |
| tar sha256 | `999a28e32814e7d9452ef107d2fde0bc242208f3b7ac4c0f3d0257435f52158c`（与 sidecar 同值，实测复算一致） |

## 2. 方法（可复现）

```bash
docker create --name m4inspect driftpilot:sdk-final2b
docker cp m4inspect:/app _inspect/app_image
docker cp m4inspect:/usr/local/lib/python3.12/site-packages/contest_harness _inspect/sdk_image
docker rm m4inspect
# 逐文件 sha256（PowerShell Get-FileHash / hashlib）后与源码包比对
```

同时执行官方校验器：

```bash
python tools/verify_final_image.py --tar dist/driftpilot-sdk-final2b.tar \
  --expect-image-id sha256:2f4014bb51450c3a9bebd5e072ad61907909df99db86b4cb20c1236a8b0bbb3e \
  --expect-layers 25 --docker-facts docker_facts.json
# RESULT: PASS (16 PASS / 0 WARN / 0 FAIL)
```

`verify_final_image.py` 覆盖：sidecar sha、config digest、层数、Entrypoint、Cmd 为空、WorkingDir、
6 项必需 ENV、是否混入 `CONTEST_*`、层内数据/权重成员数为 0、docker facts 的 id/层数/数据计数。
补充实测：容器内 `/app` 之外无任何 `.jsonl/.npz/.npy/.csv/.parquet/.pkl/.pt/.bin` 文件；
根目录无 `/out`、`/output`、`/results`、`/workspace` 残留。

## 3. 结果

| 检查 | 结果 |
|---|---|
| 镜像 `/app` 文件数 | 28 |
| 源码包 `app/` 对应文件 | 28 |
| 逐字节相同 | **28 / 28** |
| 缺失 | **0** |
| 字节不同 | **0** |
| SDK `contest_harness` 模块 | 源码 9 / 镜像 site-packages 9 / 包内 9，**三方逐字节相同** |
| 数据/权重类扩展名 | **0** |
| 凭据模式命中 | **0** |

明细：`docs/VERIFY-image-consistency.json` 的 `image_vs_package`（28 条，含两侧 sha256）与
`sdk_source_vs_package`（9 条，含三方 sha256）。

关键文件 sha256（前缀）：

| 文件 | sha256 |
|---|---|
| `app/contest_submission.py` | `b4bc4bb9dc398ac88b3129f04c7fd65890a579145558e2871766f92d44092d99` |
| `app/harness.py` | `6ef17e7dfd73558daf8464f7be8b1e963ccf103c76638c623911ec5c334c93f5` |
| `app/entrypoint.py` = `entrypoint_safe.py` | `e05d257f556193c18ee55922cd735cff1720465cc80cec4a403ab258e0f5dbd4` |
| `app/run_official_harness.py` | `472ae62410c4730e78c0176f61b1c75733bb88e998af1c441ea5a643e9e33cf1` |

## 4. 差异说明（包内多出 / 镜像内未收录）

**镜像 `/app` 全部文件均已收录，没有遗漏。** 包内其余条目不属于 `/app`（构建配方、文档、
SDK 副本与清单），逐项说明如下：

| 包内条目 | 为什么在包内 |
|---|---|
| `app/contest_harness/*.py`（9 个） | 镜像里由 wheel 装到 `site-packages/contest_harness/`，不在 `/app`；为便于阅读源码而 vendor。三方 sha 相同 |
| `Dockerfile`、`.dockerignore`、`requirements.txt` | 构建配方与依赖声明，镜像内无对应文件 |
| `config/env.list` | 镜像 ENV 的文本化记录（值取自 `docker inspect`），非镜像文件 |
| `README.md`、`DEPENDENCIES.md`、`MANIFEST.json` | 本包的说明书与清单 |
| `docs/HARNESS_SDK.md`、`docs/SDK-NOTICE.md` | SDK 官方文档原样副本与来源说明 |
| `docs/VERIFY-image-consistency.{md,json}` | 本核对报告 |

镜像内存在、包内未收录的只有 `__pycache__/*.pyc`（构建产物，非源码；镜像 `/app` 本身也没有
`__pycache__`，`site-packages` 下有，但 `.pyc` 不参与一致性判定）。

## 5. 限制

1. 本核对证明的是**内容一致**（文件字节与 ENV），不是 digest 可复现：用包内 `Dockerfile` 重建会得到
   不同的 image id（层历史/时间戳/基础镜像 digest 不同）。
2. `/app/entrypoint*.py` 是 legacy 死代码，平台执行的是 `harness.py`；保留它们是为了让源码包与
   已交付镜像完全对应。
3. 平台侧镜像存档不返回 digest（只有 id/created_at），因此"平台跑的代码 = 本包代码"依赖本包 +
   交付出 tar 的本地记录链，平台无法独立复核。
