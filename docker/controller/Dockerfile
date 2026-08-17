# docker/controller/Dockerfile
# docker:27-cli ships the `docker` CLI and the `compose` plugin, needed to
# drive the host daemon over the mounted socket -- this image never runs
# its OWN daemon, only the client talking out through the socket.
FROM docker:27-cli

RUN apk add --no-cache python3 py3-pip

WORKDIR /app
COPY docker/controller/requirements.txt .
# --break-system-packages: Alpine's system python3 is PEP-668
# externally-managed; this is a throwaway container image, not a shared
# host Python, so overriding that guard here is safe.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY docker/controller/main.py docker/controller/controller_runner.py ./

EXPOSE 9100
CMD ["python3", "main.py"]
