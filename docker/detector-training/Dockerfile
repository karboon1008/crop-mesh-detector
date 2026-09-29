# Leaf-detector training environment (CPU). The repo is mounted at /work by docker-compose.yml,
# so datasets, checkpoints and the exported ONNX all stay on your computer. See README.md here.
FROM python:3.11-slim
# CPU-only PyTorch wheels are ~10x smaller than the default CUDA ones.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir onnx onnxruntime onnxscript codecarbon pillow numpy pyyaml tqdm
WORKDIR /work
