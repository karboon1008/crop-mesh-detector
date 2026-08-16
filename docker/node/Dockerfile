# docker/node/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/node/requirements.txt .
# --extra-index-url (not --index-url) points pip at PyTorch's CPU-only wheel
# index while KEEPING PyPI for everything else in requirements.txt. Without
# it, `torch>=2.1`/`torchvision>=0.16` resolve to PyPI's default CUDA-enabled
# wheels, pulling multi-GB nvidia-* libraries into an image whose Node is
# hardcoded to device="cpu" (see docker/node/main.py) and never uses a GPU.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src/ ./src/
COPY docker/node/main.py docker/node/node_runner.py docker/node/knowledge_codec.py ./

EXPOSE 8000
CMD ["python", "main.py"]
