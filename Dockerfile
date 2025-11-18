# 使用官方Python镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

ENV http_proxy=http://192.168.244.188:7897
ENV https_proxy=http://192.168.244.188:7897

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY faiss_server_optimized.py .
COPY search_optimization.py .
COPY config.py .
COPY entrypoint.sh .
COPY healthcheck.sh .

# 创建数据目录和日志目录
RUN mkdir -p /app/data /app/logs

# 设置文件权限
RUN chmod +x faiss_server_optimized.py entrypoint.sh healthcheck.sh

# 设置默认环境变量
ENV BUSINESS_ID=default
ENV FAISS_DATA_DIR=/app/data
ENV API_PORT=8001

# 暴露端口 - 所有容器内部都运行在8001，外部映射由docker-compose处理
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./healthcheck.sh

# 启动命令
ENTRYPOINT ["./entrypoint.sh"]
