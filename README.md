# 智训引擎 final2b · 项目源代码

> 赛道：第九赛道「智训引擎」｜队伍：猫枪海虾队（平台账号 `duanzuibingling`）
> 对象：提交镜像 `driftpilot:sdk-final2b` 的源码与构建配方
> 口径声明：本包内所有「本地验收」数字都来自官方公开 validation 的单轮实测，**不是平台隐藏分**；
> 平台侧数字单独标注 run id，两类数字 population 不同，不可相减、不可拼成一条趋势。

## 1. 身份

| 项 | 值 |
|---|---|
| 镜像 | `driftpilot:sdk-final2b` |
| 镜像 id | `sha256:2f4014bb51450c3a9bebd5e072ad61907909df99db86b4cb20c1236a8b0bbb3e` |
| 层数 | 25 |
| tar | `dist/driftpilot-sdk-final2b.tar`，150,721,024 B |
| tar sha256 | `999a28e32814e7d9452ef107d2fde0bc242208f3b7ac4c0f3d0257435f52158c` |
| 入口 | `Entrypoint=["python","/app/harness.py"]`，`Cmd=[]`，`WorkingDir=/app` |
| 协议 | `contest.harness.workspace.v2`（SDK `contest_harness` 1.0.2） |
| 适配器 | `app/contest_submission.py`，31,993 B，sha256 `b4bc4bb9dc398ac88b3129f04c7fd65890a579145558e2871766f92d44092d99` |
| 平台提交 | v16，`submission_id=f6829a94f5c3c414b68a196222236c43` |
| 平台基础模型身份 | `base:mspec_3ecdzfqhaml3`（平台报告 `model_identity` 原值） |

## 2. 目录结构

```
zhixun-engine-final2b-src/
├── README.md                 本文件
├── DEPENDENCIES.md           依赖与许可证清单（含未声明许可证的披露）
├── MANIFEST.json             每个文件的字节数与 sha256（不含自身）
├── requirements.txt          运行时依赖（numpy==2.0.2；SDK 为 vendor）
├── Dockerfile                从源码重建镜像的扁平化配方
├── .dockerignore
├── config/env.list           镜像 ENV 与可选项（与 docker inspect 对齐）
├── app/                      = 镜像 /app（逐字节一致）
│   ├── contest_submission.py 平台适配器：续训 / 候选提交 / 门禁 / 预测
│   ├── harness.py            SDK 固定入口（describe|train|predict）
│   ├── entrypoint.py         legacy 入口（含 KS 漂移监测；平台不执行）
│   ├── entrypoint_safe.py    同上（对照副本）
│   ├── run_official_harness.py legacy 本地跑官方包的驱动
│   ├── baselines/            官方 CPU 参考实现（NumPy MLP + Adam）
│   ├── dp/                   本地 Harness 代码树（20 个模块 + 2 个数据适配器）
│   └── contest_harness/      官方 SDK 源码副本（镜像内位于 site-packages）
└── docs/
    ├── HARNESS_SDK.md        SDK 官方协议文档（原样）
    ├── SDK-NOTICE.md         SDK 来源、wheel sha、许可证状态
    ├── VERIFY-image-consistency.md    镜像 vs 源码核对结论
    └── VERIFY-image-consistency.json  逐文件核对明细（28 + 9 条）
```

`app/` 下 28 个文件与镜像 `/app` **逐字节相同**；`app/contest_harness/` 的 9 个模块与镜像
`site-packages/contest_harness/` 及 SDK 源码三方**逐字节相同**。核对方法：对运行中的镜像
`docker cp` 出 `/app` 与 site-packages，逐文件算 sha256 后与包内文件比对（明细见
`docs/VERIFY-image-consistency.json`）。

## 3. 代码地图（训练 / 微调 / 预测 / 评测 / 漂移监测 / Harness）

平台评分只走 `harness.py → contest_harness.cli → contest_submission.py` 一条路径；
下表行号均指 `app/contest_submission.py`。

| 能力 | 位置 | 说明 |
|---|---|---|
| **Harness 协议入口** | `app/harness.py`；`app/contest_harness/{cli,round,transport,predict}.py` | `describe/train/predict` 三帧；round 生命周期、父产物、state、模型调用都由 SDK 管理 |
| **训练（续训）** | `train()` L426–658；`train_epochs()` L111–152 | 从父 checkpoint 继续 Adam，保留一阶/二阶动量；A 每轮 6 ep，B 首次 600 ep，C 首次 1200 ep |
| **微调 / 增量** | L499–520（B/C 已长训后的 60/20 ep 增量）、L554–578（A 的 warm 配方） | B/C 长训只做一次，之后用本轮新标注小步增量；A 每轮都续训 |
| **预测** | `predict()` L671–703；`contest_harness/predict.py` | stdin 逐行 JSON → 分类 softmax 概率 / 回归非负值；校验特征 schema 与 task |
| **评测 / 门禁** | `_gate()` L347–419；`_score_rows()` L194–228；`_reg_metrics()` L305–322 | 父/候选在全量官方 validation 上比较：A 用 macro-F1，B/C 用 RMSE；尾部另比最差轴承/子集与 NASA |
| **漂移监测（现行门禁策略）** | `_gate()` 的 `A_GATE_FLOOR/A_GATE_TOL`、`B_WORST_TOL/C_WORST_TOL/C_NASA_TOL`（L39–48） | 分类用绝对容差、回归用相对死区；unknown/不可信一律保留父（fail-closed） |
| **漂移监测（KS / 预算准入，legacy）** | `app/entrypoint.py`、`app/entrypoint_safe.py`、`app/dp/stream.py`、`app/dp/official_harness.py` | legacy 入口按 KS 统计量与性能跌落决定是否续训，并做标注预算准入；**平台 SDK 路径不执行**，保留仅为追溯与本地演示 |
| **失败兜底** | `finish_parent()` L436–452 及各处 `try/except` | 读行/训练/写候选/提交/回执/state 任一步失败都保留父产物，不让整轮 failed |
| **/state 续接** | `_merge_state()` L156–183 | 只认 `model_state.npz`，校验 paths、shape、epochs 单调，steps 取 max |
| **B 长训引擎** | `app/dp/b_engine.py::fit_variant`（由 L325–333 调用） | 按轴承分层、梯度裁剪、余弦退火 |
| **C 长训引擎** | `app/dp/cosine.py::onecycle_train`（由 L335–344 调用） | onecycle、warmup、cap 350、clip 5 |
| **审计 / 指纹 / 技能库 / 记忆** | `app/dp/{audit,fingerprint,skills,memory,tracking}.py` | 本地 Harness 的审计链与策略资产；平台路径不 import |

说明：平台路径只依赖 `numpy` 与 `contest_harness`。`app/dp` 里 `model.py`、`skills.py`、
`stream.py`、`official.py` 需要 scikit-learn / PyYAML / 子进程，镜像内没有这些依赖，因此在镜像里
import 会失败——它们不参与评分，属于本地完整 Harness 的保留代码。详见 `DEPENDENCIES.md` §2。

## 4. 构建与运行

```bash
# 从源码重建（本目录即构建上下文）
docker build -t driftpilot:sdk-final2b-src .

# 三帧自检（不需要数据；describe 只检查导入与任务声明）
echo '{}' | docker run --rm -i driftpilot:sdk-final2b-src describe
```

平台运行方式（镜像已固化，重建镜像同样适用）：

```bash
python /app/harness.py describe
python /app/harness.py train      # 宿主注入 /data /previous /workspace /model 与 RPC 通道
python /app/harness.py predict    # stdin 逐行 {"task","inputs"} → stdout 逐行 {"outputs":...}
```

核对已交付镜像（只读，不重新构建）：

```bash
python tools/verify_final_image.py --tar dist/driftpilot-sdk-final2b.tar \
  --expect-image-id sha256:2f4014bb51450c3a9bebd5e072ad61907909df99db86b4cb20c1236a8b0bbb3e \
  --expect-layers 25 --docker-facts docker_facts.json
# 实测 16 PASS / 0 WARN / 0 FAIL（含 sidecar、config digest、层数、入口、6 项 ENV、层内数据成员=0）
```

## 5. 验收摘要（本地单轮，官方公开 validation）

| 任务 | 指标 | 父产物 | 候选 | 门禁 |
|---|---|---|---|---|
| A 气体识别 | macro-F1 / accuracy | 0.714030 | **0.726306** / accuracy 0.885852 | pass_better（margin 0.0100，1,244 行） |
| B 轴承 RUL | RMSE | 681.159833 | **361.450651**（worst 372.973938） | pass_better（margin 0.005，1,679 行，tail_ok） |
| C 发动机 RUL | RMSE | 52.868573 | **49.508243**（worst 60.649103，NASA 2.135e9） | pass_better（margin 0.001，23,049 行，tail_ok） |

- 本地 SDK 验收：`sdk_acceptance.py --dataset all` → **3/3 PASS，280.6 s**；候选 checkpoint sha256
  依次为 `149762a1…`、`85b3472c…`、`514fb485…`。
- 平台试运行（run `231df08579172d727faacbdb798107fe`，trial）：A macro-F1 0.8789877670187155
  （accuracy 0.8845681539091281，4,886 行）、B RMSE 169.5863809106589（200 行）、
  C RMSE 54.33825722159884（9,687 行）。trial 不计分。
- 平台正式评测（run `3dfd43f0a1dd4d4b96ad2f6e0756fc5b`）：**61.91 / 100**，A 63.89 / B 63.95 / C 57.88，
  CPU 1,682.75 核秒、已结算 Token 13,383；`eligible=false`。

## 6. 数据 / 权重 / 凭据边界

- **不含数据**：无 `.jsonl/.csv/.npz/.npy/.parquet/.pkl` 等数据文件；训练行一律由平台
  `rnd.data.training_rows()` 下发，validation 只用于门禁测量，不进梯度。
- **不含权重**：无 `checkpoint.npz` / `model_state.npz`；父产物与候选都由平台产物目录提供。
- **不含隐藏测试与样本预测**：包内没有任何预测输出或回执原文。
- **不含凭据**：代码只读取环境变量名（如 `CONTEST_WORK`），不包含任何 key/token 值。
  打包脚本对 16 类数据/权重扩展名与 6 类凭据模式做了扫描，命中 0（见一致性报告 `banned_ext_files`/`secret_hits`）。

## 7. 已知限制与未决项（必须随包披露）

1. **重建不等于复现 digest**：`Dockerfile` 是扁平化重建配方，`/app` 内容与 ENV 可核对一致，
   但 image id 与交付出 tar 不同（层历史/时间戳差异）。逐字节复现请直接 `docker load` 交付 tar。
2. **A 的提升在噪声内**：同配方 10 次抽样 macro-F1 0.7216±0.0045；b03–b10 人口外父/候选均 ≈0.40。
   对外只应写「val 不劣化」，不能写「隐藏集提升」。
3. **B 是单 seed 读数**：361.45 为单次；同配方 3 个批序 seed 为 396.86±37.71。平台隐藏集 B 仅 200 行，
   单次波动未知（本地 3 批序 seed sd ≈38 分钟量级）。
4. **C 的隐藏集会放大**：历史同模型 val→hidden 倍率 1.06–1.17；NASA 由单行主导，不能承诺。
5. **legacy 死代码**：`entrypoint*.py` 仍在 `/app` 内但不被平台调用；`DP_*` ENV 是它的兼容开关，
   SDK 路径不读取。清理会改变镜像身份，故本版不动。
6. **许可证未决**：`contest_harness` 与官方 `baselines` 代码的许可证在官方包内均未声明，
   正式再分发前需向主办方确认（`DEPENDENCIES.md` §4）。
7. **平台 archive 无 digest**：平台侧镜像存档只返回 id/created_at，代码身份依赖本地记录（本包 + 交付 tar）。

## 8. MANIFEST.json

`MANIFEST.json` 记录除自身外每个文件的 `path / bytes / sha256 / origin`，可离线复核：
`python -c "import json,hashlib;..."` 或直接比对 `docs/VERIFY-image-consistency.json` 的 28 条
`image_vs_package` 与 9 条 `sdk_source_vs_package` 记录。
