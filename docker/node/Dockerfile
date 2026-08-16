# docker/node/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/node/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docker/node/main.py docker/node/node_runner.py docker/node/knowledge_codec.py ./

EXPOSE 8000
CMD ["python", "main.py"]
