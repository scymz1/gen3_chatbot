FROM ollama/ollama:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt-get update && \
    apt-get install -y bash tzdata python3 python3-pip python3-venv build-essential cmake git wget curl && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# 设置工作目录
WORKDIR /app
COPY . .

# 创建虚拟环境并安装依赖
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 暴露端口
EXPOSE 8000 11434
ENTRYPOINT ["bash", "-c"]
# 启动 Ollama + FastAPI
CMD ["ollama serve & sleep 3 && ollama pull llama3 && uvicorn backend:app --host 0.0.0.0 --port 8000"]
