# apps/crop_disease_detection/Dockerfile
#
# Build from the repo root (needs apps/ at the top of the build context):
#   docker build -f apps/crop_disease_detection/Dockerfile -t crop-disease-detection .
#
# Run, bind-mounting the model and DB so they live on the host rather than
# inside the container -- replacing apps/model/model.onnx on the host and
# restarting the container deploys an updated model, same as restarting
# `streamlit run` directly. On Windows + Git Bash, use `pwd -W` (Windows-style
# path) rather than `pwd` -- the plain POSIX form makes Docker Desktop
# misparse the volume spec and litter the host with stray "<name>;C"
# directories instead of mounting correctly:
#   docker run --rm -p 8502:8501 \
#     -v "$(pwd -W)/apps/model:/app/apps/model" \
#     -v "$(pwd -W)/apps/crop_disease_detection/data:/app/apps/crop_disease_detection/data" \
#     crop-disease-detection
FROM python:3.11-slim

WORKDIR /app

COPY apps/crop_disease_detection/requirements.txt ./apps/crop_disease_detection/requirements.txt
RUN sh -c 'i=0; until pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r apps/crop_disease_detection/requirements.txt; do i=$((i+1)); [ $i -ge 8 ] && exit 1; echo "pip install failed, retrying ($i)..."; sleep 3; done'

COPY apps/crop_disease_detection/*.py ./apps/crop_disease_detection/
COPY apps/model/model.onnx ./apps/model/model.onnx

EXPOSE 8501
CMD ["streamlit", "run", "apps/crop_disease_detection/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
