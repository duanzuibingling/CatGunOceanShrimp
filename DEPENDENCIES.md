# 依赖与许可证清单（DEPENDENCIES）

> 对象：`driftpilot:sdk-final2b` 提交镜像的源码包。
> 口径：只记录**镜像内真实存在**的组件；开发期用到但未进镜像的依赖单列一节，不冒充运行时依赖。
> 许可证依据一律写「依据来源」：Python 发行包看 `*.dist-info/METADATA` 与包内许可证文件，
> 不猜、不从网上补。

## 1. 运行时（进镜像）

| 组件 | 版本 | 作用 | 许可证 | 依据 |
|---|---|---|---|---|
| Python | 3.12.16 | 解释器（`python:3.12-alpine` 基础镜像） | PSF-2.0 | 基础镜像 `ENV PYTHON_VERSION=3.12.16`；CPython 许可为 PSF |
| numpy | 2.0.2 | 唯一数值依赖（MLP 前向/反向、KS 统计） | **BSD-3-Clause** | `numpy-2.0.2.dist-info/METADATA`：`License:` 为 BSD 三条款全文（含 "Neither the name of the NumPy Developers…"），子组件另有 `License: BSD-3-Clause` |
| contest-harness（`contest_harness`） | 1.0.2 | 平台 SDK：协议帧、round 生命周期、transport、predict CLI | **未声明**（见 §4） | `contest_harness-1.0.2.dist-info/METADATA` 无 `License`/`License-Expression`/`Classifier: License`；dist-info 内无 LICENSE 文件；`pyproject.toml` 无 `license` 字段 |
| 赛题 baselines 代码（`app/baselines/{__init__,data,model}.py`） | 随官方数据包 | 官方 CPU 参考实现：NumPy MLP + Adam、可续训 checkpoint、特征展平 | **未声明**（见 §4） | 官方 `TRAINING_GUIDE.md` 与包内均无许可证声明；文件为官方原样，未改动（sha 见 MANIFEST） |
| Alpine Linux 基础系统 | 3.24 | libc(musl)、ca-certificates、busybox 等 | 各包各自许可（MIT / GPL-2.0 / Apache-2.0 / MPL-2.0 等） | 以 Alpine 3.24 包元数据为准；这些组件由基础镜像提供，未随本源码包分发 |

**运行时依赖结论**：平台评分路径只 import `numpy`、`contest_harness`，以及 `app/dp/b_engine.py`、
`app/dp/cosine.py`（两者均只用 numpy）。镜像内 `pip freeze` 实测只有
`contest-harness @ …whl#sha256=c5a6cadc…` 与 `numpy==2.0.2`。

## 2. 开发期 / 非镜像路径（未进镜像，不计入运行时）

`app/dp` 是完整的本地 Harness 代码树，其中下列模块**不被平台 SDK 路径导入**，镜像内也没有其依赖，
因此在镜像里 import 会失败——这是已知且有意保留的边界，不是缺陷：

| 模块 | 额外依赖 | 用途 | 是否进镜像 |
|---|---|---|---|
| `app/dp/model.py` | scikit-learn | 本地可增量后训练的分类器（HGB / LogisticRegression） | 否（代码在，依赖不在） |
| `app/dp/stream.py` | scikit-learn（可选） | 本地时间序列流 + 监控 + 门禁演示 | 否 |
| `app/dp/skills.py` | PyYAML | 技能库（YAML 策略资产） | 否 |
| `app/dp/official.py` | 官方 baselines（子进程） | 调官方 `baselines.train/predict` 的适配层 | 否 |
| `app/entrypoint.py`、`entrypoint_safe.py` | numpy | 旧版 legacy 入口（含 KS 漂移监测、预算准入、回滚）；平台走 `harness.py`，**不执行这两个文件** | 是（镜像内保留，死代码） |

## 3. 标准库

`json`、`math`、`os`、`pathlib`、`shutil`、`sys`、`time`、`hashlib`、`tarfile`、`subprocess`、
`dataclasses`、`typing`、`zipfile`、`io`、`re`、`platform`、`datetime` —— 均属 CPython 标准库，PSF-2.0。

## 4. 未决项（需主办方或上游确认）

1. **contest-harness 1.0.2 许可证未声明**：包元数据与发行文件里都没有许可证文本。
   本包按「官方赛题 SDK，随镜像原样使用」处理；若需要正式再分发，建议向 SDK 提供方索取许可证声明。
2. **官方 baselines 代码许可证未声明**：赛题数据包内的参考实现未附许可证文件。
   本包只原样引用、未改动，`MANIFEST.json` 记录了每个文件的 sha256 便于比对。
3. **基础镜像可复现性**：`python:3.12-alpine` 未固定 digest（原构建使用国内镜像源）。
   若要求逐字节复现基础层，需要固定 `python:3.12-alpine@sha256:…`。

## 5. 本包不含

- 训练/验证数据（`.jsonl`、`.csv`、`.npz`、`.npy` 等一律不在包内）；
- 模型权重 / checkpoint（`checkpoint.npz`、`model_state.npz`）；
- 隐藏测试集、样本预测、提交回执原文；
- 任何凭据（API key、token、账号口令）。代码只从环境变量读 key 名，不含值。
