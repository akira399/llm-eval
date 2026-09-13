# llm-eval 平台镜像（Phase 2 交付形态）
# 构建：docker build -t llm-eval .
# 运行：docker compose up          （api: 8800 / 演示被测服务: 8766）
# 注意：本仓库主开发机无 Docker，镜像构建待有 Docker 的环境验证后标注版本。
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 默认启动 HTTP API；演示被测服务用 compose 里的第二个 service
EXPOSE 8800
CMD ["uvicorn", "api_server.app:app", "--host", "0.0.0.0", "--port", "8800"]
