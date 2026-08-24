# docker/dashboard/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/dashboard/requirements.txt .
RUN sh -c 'i=0; until pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt; do i=$((i+1)); [ $i -ge 8 ] && exit 1; echo "pip install failed, retrying ($i)..."; sleep 3; done'

COPY src/__init__.py ./src/__init__.py
COPY src/energy/ ./src/energy/
COPY docker/dashboard/app.py docker/dashboard/data.py ./

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
