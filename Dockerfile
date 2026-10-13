# ============================================================
# Agent 服务 Dockerfile
# ============================================================
# 构建：docker build -t esshop-agent:1.0 .
# 运行：docker run -d -p 8100:8100 --env-file .env --name agent esshop-agent:1.0
#
# 采用「多阶段构建」：
#   阶段 1 装依赖（体积大，含编译工具）
#   阶段 2 只复制依赖和代码（体积小）
# 这样最终镜像能小 60% 左右。

# ==================== 阶段 1：构建依赖 ====================
FROM python:3.10-slim AS builder

WORKDIR /build

# 只复制依赖清单，利用 Docker 层缓存：
# 只要 requirements.txt 没变，这层就不会重新执行
COPY requirements.txt .

# 装到独立目录，方便整体复制到下一阶段
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==================== 阶段 2：运行环境 ====================
FROM python:3.10-slim

# 时区设为东八区，否则日志时间对不上
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装时区数据（slim 镜像默认没有）
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已装好的依赖
COPY --from=builder /install /usr/local

WORKDIR /app

# 复制项目代码（.dockerignore 已排除 venv、.env 等）
COPY *.py ./
COPY tests/ ./tests/

# 以非 root 用户运行（安全最佳实践）
# 容器被攻破时，攻击者拿不到 root 权限
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

EXPOSE 8100

# 健康检查：容器编排系统靠这个判断服务是否正常
# 每 30 秒检查一次，连续 3 次失败标记为 unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health', timeout=5)" || exit 1

# 用 exec 形式启动，保证信号能正确传递给进程
# （shell 形式会导致 Ctrl+C / docker stop 无法优雅退出）
CMD ["python", "main.py"]
