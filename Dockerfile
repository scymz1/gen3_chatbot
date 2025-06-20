# Stage 1: Base image with Ollama
FROM ollama/ollama as base

# Stage 2: App layer
FROM python:3.10-slim


# Copy ollama binaries from base image
COPY --from=base /usr/bin/ollama /usr/bin/ollama

# 安装构建 llama-cpp-python 所需的依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Expose ports
EXPOSE 8000 11434

# Start Ollama + FastAPI
CMD ollama serve & \
    sleep 3 && \
    ollama pull llama3 && \
    uvicorn backend:app --host 0.0.0.0 --port 8000
# CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
