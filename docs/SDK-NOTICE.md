# contest_harness 1.0.2 来源说明（SDK NOTICE）

本目录 `app/contest_harness/` 是赛题官方 SDK 的**源码副本**，用于让评审在不装 wheel 的情况下
直接阅读协议实现。

## 来源与身份

| 项 | 值 |
|---|---|
| 上游 | `contest-harness-sdk-1.0.2`（赛题官方下载包，本地路径 `work/sdk_1_0_2/contest-harness-sdk-1.0.2/`） |
| 版本 | 1.0.2 |
| 协议 | `contest.harness.workspace.v2` |
| 镜像内安装方式 | `pip install --no-index --no-deps contest_harness-1.0.2-py3-none-any.whl` |
| wheel sha256 | `c5a6cadcb418b98ed136be07953ed09669c27de0b2d57aea3a68b1c6cfc8dc68` |
| 镜像内落位 | `/usr/local/lib/python3.12/site-packages/contest_harness/` |

## 一致性

本副本的 9 个 `.py` 文件与镜像 `site-packages/contest_harness/` 下同名文件**逐字节相同**
（sha256 见 `MANIFEST.json` 与 `docs/VERIFY-image-consistency.json`）。

`__pycache__/` 未收录；镜像内存在 `.pyc`，属于构建产物，不影响一致性判定。

## 许可证

SDK 元数据未声明许可证（无 `License` 字段、dist-info 内无 LICENSE 文件）。
本副本按「赛题官方提供、原样引用」处理，未做任何修改；正式再分发前请向 SDK 提供方确认许可条款。

## 官方文档

`docs/HARNESS_SDK.md` 是 SDK 包内 `HARNESS_SDK.md` 的原样副本（11,864 B），
描述 `describe / train / predict` 三步协议、`rnd.model()` 调用、`/state` 语义与提交要求。
