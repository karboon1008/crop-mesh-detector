# docker/dashboard/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py ./src/__init__.py
COPY src/energy/ ./src/energy/
COPY docker/dashboard/app.py docker/dashboard/data.py ./

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
