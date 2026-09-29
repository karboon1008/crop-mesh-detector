# Farmer alerts server (FastAPI + SQLite). Built from the repo root by
# apps/hivemind_demo/docker-compose.yml; serves the iPhone app on port 8080.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "httpx[http2]>=0.27" "pyjwt[crypto]>=2.8"
COPY services ./services
ENV ALERTS_CONFIG=/app/services/alerts/config.example.json
EXPOSE 8080
CMD ["uvicorn", "--factory", "services.alerts.app:build_app", "--host", "0.0.0.0", "--port", "8080"]
