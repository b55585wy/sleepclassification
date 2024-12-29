# 使用官方 TensorFlow GPU 基础镜像
FROM tensorflow/tensorflow:2.9.0-gpu


# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*




# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# 设置容器启动命令
CMD ["bash"] 