# HiveMind crop-disease demo. Build and run from this folder:
#   docker compose up --build        -> http://localhost:8502
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py inference.py db.py leaf_detection.py live_camera.py phone_alerts.py farm_server.py ./
COPY live_camera_web ./live_camera_web
COPY sample_images ./sample_images
COPY models ./models
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
