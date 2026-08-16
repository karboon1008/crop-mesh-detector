# docker/coordinator/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/coordinator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py ./src/__init__.py
COPY src/config.py ./src/config.py
COPY src/energy/ ./src/energy/
COPY docker/coordinator/main.py docker/coordinator/coordinator_runner.py ./

EXPOSE 9000
CMD ["python", "main.py"]
