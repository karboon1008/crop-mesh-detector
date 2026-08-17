# docker/node/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/node/requirements.txt .
# --extra-index-url (not --index-url) points pip at PyTorch's CPU-only wheel
# index while KEEPING PyPI for everything else in requirements.txt. Without
# it, `torch>=2.1`/`torchvision>=0.16` resolve to PyPI's default CUDA-enabled
# wheels, pulling multi-GB nvidia-* libraries into an image whose Node is
# hardcoded to device="cpu" (see docker/node/main.py) and never uses a GPU.
RUN sh -c 'i=0; until pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt; do i=$((i+1)); [ $i -ge 8 ] && exit 1; echo "pip install failed, retrying ($i)..."; sleep 3; done'

COPY src/ ./src/
COPY docker/node/main.py docker/node/node_runner.py docker/node/knowledge_codec.py ./

EXPOSE 8000
CMD ["python", "main.py"]
