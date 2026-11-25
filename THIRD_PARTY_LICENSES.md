# 第三方许可证清单（THIRD_PARTY_LICENSES）

> **本文件由脚本从已安装包的元数据直接生成，未手工填写任何许可证名称。**
>
> 数据来源：`.venv/Lib/site-packages/*.dist-info/METADATA`（Python 包元数据，RFC822 头部）
> 生成时间：2026-11-10 22:07　｜　覆盖包数：**56**（site-packages 下所有带 METADATA 的发行包）

## 一、读取规则（可复核，不做推断）

| 优先级 | 字段 | 采用条件 |
|---|---|---|
| 1 | `License-Expression` | PEP 639 的 SPDX 表达式，原样照抄 |
| 2 | `License` | 文本里出现 MIT / BSD / Apache / MPL / GPL / PSF / ISC / Zlib / CC0 等许可证标识词；≤120 字符照抄，超长截断并标注 |
| 3 | `Classifier: License ::` | 上一级不成立时回退到 Trove 分类器，保留原始字符串 |
| 4 | 以上皆无 | 记为 **未声明**（不猜、不从网上补） |

第 2 级的判断是必要的：有些包的 `License` 字段写的是版权声明（如 `Copyright (c) …`）、协议标题（如 `License agreement for …`）甚至分隔符 `====`，这类值不是许可证标识，表里会标注「License 字段非许可证标识」并改用分类器。

「包内许可证文件」列是该包 `*.dist-info/` 目录里**实际存在**的文件名，供法务/评委翻原文。

## 二、赛题点名要求的 10 个包

| 包 | 版本 | 许可证 | 依据字段 | 包内许可证文件 | 作用域 |
|---|---|---|---|---|---|
| **numpy** | 2.5.4 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | License-Expression（SPDX） | licenses/LICENSE.txt, licenses/numpy/_core/include/numpy/libdivide/LICENSE.txt, licenses/numpy/_core/src/common/pythoncapi-compat/COPYING 等 16 个 | 运行时（进镜像） |
| **pandas** | 3.1.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE, licenses/subprojects/fast_float-8.2.10/LICENSE-APACHE, licenses/subprojects/fast_float-8.2.10/LICENSE-BOOST 等 4 个 | 运行时（进镜像） |
| **scipy** | 1.18.1 | `OSI: BSD License` | Classifier（License 字段非许可证标识） | LICENSE.txt | 运行时（进镜像） |
| **scikit-learn** | 1.9.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/COPYING | 运行时（进镜像） |
| **lightgbm** | 4.7.0 | `MIT` | License-Expression（SPDX） | licenses/LICENSE | 运行时（进镜像） |
| **matplotlib** | 3.11.2 | `OSI: Python Software Foundation License` | Classifier（License 字段非许可证标识） | LICENSE | 运行时（进镜像） |
| **river** | 0.26.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE | 运行时（进镜像） |
| **pydantic** | 2.14.3 | `MIT` | License-Expression（SPDX） | licenses/LICENSE | 运行时（进镜像） |
| **PyYAML** | 6.0.3 | `MIT` | License | licenses/LICENSE | 运行时（进镜像） |
| **swanlab** | 0.8.4 | `Apache-2.0` | License-Expression（SPDX） | licenses/LICENSE | 开发期（不上镜像） |

## 三、全部依赖（按作用域，作用域由 Requires-Dist 传递闭包算出）

### 运行时（进镜像）　（28 个）

| 包 | 版本 | 许可证 | 依据字段 | 包内许可证文件 |
|---|---|---|---|---|
| annotated-types | 0.8.0 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| cloudpickle | 3.1.2 | `BSD-3-Clause` | License | licenses/LICENSE |
| colorama | 0.4.6 | `OSI: BSD License` | Classifier | licenses/LICENSE.txt |
| contourpy | 1.4.0 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE |
| cycler | 0.12.1 | `OSI: BSD License` | Classifier（License 字段非许可证标识） | LICENSE |
| fonttools | 4.68.0 | `MIT` | License | licenses/LICENSE, licenses/LICENSE.external |
| joblib | 1.6.0 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE.txt |
| kiwisolver | 1.5.1 | `OSI: BSD License` | Classifier（License 字段非许可证标识） | licenses/LICENSE |
| lightgbm | 4.7.0 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| matplotlib | 3.11.2 | `OSI: Python Software Foundation License` | Classifier（License 字段非许可证标识） | LICENSE |
| narwhals | 2.28.0 | `MIT` | License-Expression（SPDX） | licenses/LICENSE.md |
| numpy | 2.5.4 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | License-Expression（SPDX） | licenses/LICENSE.txt, licenses/numpy/_core/include/numpy/libdivide/LICENSE.txt, licenses/numpy/_core/src/common/pythoncapi-compat/COPYING 等 16 个 |
| packaging | 26.3 | `Apache-2.0 OR BSD-2-Clause` | License-Expression（SPDX） | licenses/LICENSE, licenses/LICENSE.APACHE, licenses/LICENSE.BSD |
| pandas | 3.1.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE, licenses/subprojects/fast_float-8.2.10/LICENSE-APACHE, licenses/subprojects/fast_float-8.2.10/LICENSE-BOOST 等 4 个 |
| pillow | 13.0.0 | `MIT-CMU` | License-Expression（SPDX） | licenses/LICENSE |
| pydantic | 2.14.3 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| pydantic_core | 2.50.4 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| pyparsing | 3.3.3 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| python-dateutil | 4.0.0 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE |
| PyYAML | 6.0.3 | `MIT` | License | licenses/LICENSE |
| river | 0.26.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE |
| scikit-learn | 1.9.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/COPYING |
| scipy | 1.18.1 | `OSI: BSD License` | Classifier（License 字段非许可证标识） | LICENSE.txt |
| threadpoolctl | 3.8.0 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE |
| tqdm | 4.70.1 | `MPL-2.0 AND MIT` | License | licenses/LICENCE |
| typing_extensions | 4.16.0 | `PSF-2.0` | License-Expression（SPDX） | licenses/LICENSE |
| typing-inspection | 0.4.5 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| tzdata | 2026.6 | `Apache-2.0` | License | licenses/LICENSE, licenses/licenses/LICENSE_APACHE |

### 开发期（不上镜像）　（27 个）

| 包 | 版本 | 许可证 | 依据字段 | 包内许可证文件 |
|---|---|---|---|---|
| certifi | 2026.10.23 | `MPL-2.0` | License | licenses/LICENSE |
| charset-normalizer | 3.5.2 | `MIT` | License | licenses/LICENSE |
| click | 8.5.1 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE.txt |
| idna | 3.20 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE.md |
| Jinja2 | 3.1.6 | `OSI: BSD License` | Classifier | licenses/LICENSE.txt |
| markdown-it-py | 4.2.0 | `OSI: MIT License` | Classifier | licenses/LICENSE, licenses/LICENSE.markdown-it |
| MarkupSafe | 3.0.4 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE.txt |
| mdurl | 0.1.2 | `OSI: MIT License` | Classifier | LICENSE |
| nvidia-ml-py | 13.615.71 | `BSD` | License | 无独立文件 |
| orjson | 3.13.0 | `MPL-2.0 AND (Apache-2.0 OR MIT)` | License-Expression（SPDX） | licenses/LICENSE-APACHE, licenses/LICENSE-MIT, licenses/LICENSE-MPL-2.0 |
| platformdirs | 4.14.5 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| prettytable | 3.18.0 | `BSD-3-Clause` | License-Expression（SPDX） | licenses/LICENSE |
| protobuf | 6.33.6 | `3-Clause BSD License` | License | LICENSE |
| psutil | 7.2.2 | `BSD-3-Clause` | License | LICENSE |
| pydantic-settings | 2.16.2 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| pyecharts | 2.1.0 | `MIT` | License | LICENSE |
| Pygments | 2.21.0 | `BSD-2-Clause` | License-Expression（SPDX） | licenses/LICENSE |
| python-dotenv | 1.2.4 | `BSD-3-Clause` | License | licenses/LICENSE |
| requests | 2.34.2 | `Apache-2.0` | License | licenses/LICENSE, licenses/NOTICE |
| rich | 15.0.0 | `MIT` | License | licenses/LICENSE |
| setuptools | 84.2.1 | `MIT` | License-Expression（SPDX） | licenses/LICENSE |
| simplejson | 4.2.0 | `MIT OR AFL-2.1` | License | licenses/LICENSE.txt |
| swanlab | 0.8.4 | `Apache-2.0` | License-Expression（SPDX） | licenses/LICENSE |
| urllib3 | 2.8.0 | `MIT` | License-Expression（SPDX） | licenses/LICENSE.txt |
| watchdog | 7.1.0 | `Apache-2.0` | License | licenses/COPYING, licenses/LICENSE |
| wcwidth | 0.9.4 | `OSI: MIT License` | Classifier | licenses/LICENSE |
| wrapt | 2.5.2 | `BSD-2-Clause` | License-Expression（SPDX） | licenses/LICENSE |

### 环境自带 / 未归类　（1 个）

| 包 | 版本 | 许可证 | 依据字段 | 包内许可证文件 |
|---|---|---|---|---|
| pip | 26.3 | `MIT` | License-Expression（SPDX） | licenses/LICENSE.txt, licenses/src/pip/_vendor/cachecontrol/LICENSE.txt, licenses/src/pip/_vendor/certifi/LICENSE 等 21 个 |

## 四、许可证分布与履约动作

| 许可证（METADATA 原值） | 包数 | 需要做什么 |
|---|---|---|
| `MIT` | 17 | 保留版权声明与许可全文 |
| `BSD-3-Clause` | 14 | 保留版权声明、条件与免责声明；不得用作者名义背书 |
| `OSI: BSD License` | 5 | 保留版权声明、条件与免责声明；不得用作者名义背书 |
| `Apache-2.0` | 4 | 保留 LICENSE 与 NOTICE；标注修改过的文件；不暗示商标授权 |
| `OSI: MIT License` | 3 | 保留版权声明与许可全文 |
| `BSD-2-Clause` | 2 | 保留版权声明、条件与免责声明；不得用作者名义背书 |
| `3-Clause BSD License` | 1 | 保留版权声明、条件与免责声明；不得用作者名义背书 |
| `Apache-2.0 OR BSD-2-Clause` | 1 | 保留 LICENSE 与 NOTICE；标注修改过的文件；不暗示商标授权 |
| `BSD` | 1 | 保留版权声明、条件与免责声明；不得用作者名义背书 |
| `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | 1 | 保留版权声明、条件与免责声明；不得用作者名义背书 |
| `MIT OR AFL-2.1` | 1 | 保留版权声明与许可全文 |
| `MIT-CMU` | 1 | 保留版权声明与许可全文 |
| `MPL-2.0` | 1 | 保留许可文本；被修改过的文件需以 MPL 公开（我们未修改第三方源码） |
| `MPL-2.0 AND (Apache-2.0 OR MIT)` | 1 | 保留许可文本；被修改过的文件需以 MPL 公开（我们未修改第三方源码） |
| `MPL-2.0 AND MIT` | 1 | 保留许可文本；被修改过的文件需以 MPL 公开（我们未修改第三方源码） |
| `OSI: Python Software Foundation License` | 1 | 见下方通用动作——保留版权声明与许可原文 |
| `PSF-2.0` | 1 | 保留 PSF 许可与版权声明 |

### 我们实际做的四件事

1. **不打包 wheel / site-packages**：提交镜像通过 `pip install -r requirements.txt` 重装依赖，许可证由各包自带（`dist-info` 内的 LICENSE），不改变其分发形式。
2. **随交付物附本清单**：`THIRD_PARTY_LICENSES.md` 与镜像、源码同目录，每个包都能回溯到具体 `*.dist-info/METADATA`。
3. **不修改第三方源码**：`src/` 只 `import`，不打补丁；因此不触发 Apache-2.0 的「标注修改文件」与 MPL 的「修改文件公开」义务。
4. **不留模型权重**：镜像内无预训练权重，不涉及第三方模型许可。

## 五、版本一致性核对

`requirements.txt` 锁定的 11 个包，版本与本地 `.venv` 实装**逐条一致**，因此上表可作为镜像的许可证清单使用。

## 六、口径与已知缺口

- **未声明 ≠ 无许可证**：仅表示 METADATA 没写许可证字段，需人工查包内 LICENSE 文件；本清单如实标注，不推断。
- `License` 字段为纯文本（非 SPDX 标识符）的包，表中保留原值或注明回退原因，SPDX 归一化留给合规岗复核。
- 作用域由 `Requires-Dist` 传递闭包算出；“环境自带 / 未归类”多为 pip、setuptools 这类基础工具，以及未被任何 `Requires-Dist` 引用的包。
- 传递依赖版本由 pip 解析，未在 `requirements.txt` 显式锁定；需要完全可复现的许可证快照时，应在构建镜像后重新生成本文件。

---

复核命令（只读、不联网；任取一行都能对回源文件）：

```powershell
$env:PYTHONIOENCODING="utf-8"
Get-ChildItem .venv\Lib\site-packages\*.dist-info\METADATA | Select-Object -First 3 |
  ForEach-Object { Get-Content $_ -TotalCount 30 | Select-String "^(Name|Version|License)" }
```
