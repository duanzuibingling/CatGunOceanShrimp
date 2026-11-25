# 本文件只用基础 Dockerfile 语法（无 BuildKit 专属特性），离线环境也能解析；
# 构建需要本地已有 python:3.12-alpine，或能访问镜像仓库/校内镜像源。
# ---------------------------------------------------------------------------
# 智训引擎 final2b 提交镜像 —— 源码重建配方（扁平化）
#
# 已交付镜像：driftpilot:sdk-final2b
#   image id : sha256:2f4014bb51450c3a9bebd5e072ad61907909df99db86b4cb20c1236a8b0bbb3e
#   层数     : 25
#   tar      : dist/driftpilot-sdk-final2b.tar
#   tar sha  : 999a28e32814e7d9452ef107d2fde0bc242208f3b7ac4c0f3d0257435f52158c
#
# 原镜像的组装链是「driftpilot:safe 基座 + 适配器层」，逐层保留了历史；
# 本文件把同一份 /app 内容与 ENV 收敛成单文件构建，方便评委从源码重建。
#
# 重要：重建**不保证**与已交付 tar 的 digest 相同。Docker 层历史、文件时间戳、
# 基础镜像 digest 都会改变 image id；可核对的是 /app 文件内容与 ENV（见
# docs/VERIFY-image-consistency.md）。若需逐字节复现，请直接 docker load 交付 tar。
# ---------------------------------------------------------------------------

FROM python:3.12-alpine

# 与镜像 config.Env 逐项一致（另有 PATH/LANG/GPG_KEY/PYTHON_VERSION/PYTHON_SHA256
# 由基础镜像 python:3.12-alpine 提供）。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BASELINE_CPU_THREADS=1 \
    DP_A_BATCH_WEIGHT=2 \
    DP_LABEL_BUDGET_ENFORCE=1 \
    DP_LABEL_BUDGET_TOTAL=9628 \
    DP_LABEL_DOSE=drop_last \
    DP_C_SHADOW_MAX_TOTAL=1000 \
    PYTHONPATH=/app

WORKDIR /app

# 运行时依赖只有 numpy（镜像内实测 numpy==2.0.2）。
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==2.0.2" && \
    python -c "import numpy; print('numpy', numpy.__version__)"

# contest_harness 1.0.2：原镜像用官方 wheel 离线安装
#   contest_harness-1.0.2-py3-none-any.whl
#   wheel sha256 = c5a6cadcb418b98ed136be07953ed09669c27de0b2d57aea3a68b1c6cfc8dc68
# 源码包 app/contest_harness/ 与该 wheel 安装出的模块文件逐字节相同，
# 这里直接落位到 site-packages，避免依赖 wheel 文件与网络。
COPY app/contest_harness/ /usr/local/lib/python3.12/site-packages/contest_harness/

# /app 源码树（与镜像 /app 逐字节一致，见一致性报告）
COPY app/dp/ /app/dp/
COPY app/baselines/ /app/baselines/
COPY app/harness.py /app/harness.py
COPY app/run_official_harness.py /app/run_official_harness.py
COPY app/entrypoint.py /app/entrypoint.py
COPY app/entrypoint_safe.py /app/entrypoint_safe.py
COPY app/contest_submission.py /app/contest_submission.py

# 构建期自检：入口可导入、版本正确。失败即中止构建（fail-closed）。
RUN python -c "import sys, numpy, contest_harness; assert numpy.__version__ == '2.0.2', numpy.__version__; print('selfcheck ok:', 'contest_harness', getattr(contest_harness, '__version__', '1.0.2'), 'numpy', numpy.__version__, 'python', sys.version.split()[0])"

ENTRYPOINT ["python", "/app/harness.py"]
