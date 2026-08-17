# docker/coordinator/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/coordinator/requirements.txt .
RUN sh -c 'i=0; until pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt; do i=$((i+1)); [ $i -ge 8 ] && exit 1; echo "pip install failed, retrying ($i)..."; sleep 3; done'

COPY src/__init__.py ./src/__init__.py
COPY src/config.py ./src/config.py
COPY src/energy/ ./src/energy/
COPY docker/coordinator/main.py docker/coordinator/coordinator_runner.py ./

EXPOSE 9000
CMD ["python", "main.py"]
