# Sailfish appliance — the image the installer pulls: `docker pull deepbluedynamics/sailfish`.
#
# ONE image, ONE port (22343): the llama.cpp Tier B engine (measured-working on a 12 GB 3060 —
# 76 tool-runs / 177 repetitive TPS with zero-VRAM n-gram speculation) + the FastAPI gateway
# (site at /, OpenAI /v1 proxy, /api/status). The entrypoint autodetects the GPU, launches the
# engine on an internal port, and fronts it with the gateway.
#
# Tier A (>=16 GB: vLLM + MTP drafter, the challenge stack) ships as a separate image in P4 —
# see Dockerfile.vllm. This image reports Tier-A *capability* on big cards via /api/status.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

LABEL org.opencontainers.image.title="deepbluedynamics/sailfish"
LABEL org.opencontainers.image.description="Sovereign fast gemma-4-E4B on your own GPU (Tier B: llama.cpp + n-gram speculation)"
LABEL org.opencontainers.image.source="https://github.com/DeepBlueDynamics/sailfish"

# Python for the gateway + curl for the health probe. venv avoids PEP-668 externally-managed errors.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV SAILFISH_PY=/opt/sailfish/venv/bin/python
RUN python3 -m venv /opt/sailfish/venv \
    && /opt/sailfish/venv/bin/pip install --no-cache-dir \
        fastapi==0.115.0 "uvicorn[standard]==0.30.6" httpx==0.27.2

WORKDIR /opt/sailfish
# The llama.cpp libs live in /app (the base image ran from WORKDIR=/app); we run from /opt/sailfish,
# so put /app on the library path explicitly or llama-server can't find libllama-server-impl.so.
ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64 \
    PYTHONPATH=/opt/sailfish \
    PYTHONUNBUFFERED=1 \
    SAILFISH_PORT=22343 \
    SAILFISH_ENGINE_PORT=8080 \
    SAILFISH_MODEL=gemma4-e4b

COPY app/ ./app/
COPY site/ ./site/
COPY scrape/ ./scrape/
COPY container/entrypoint.sh /opt/sailfish/entrypoint.sh
RUN chmod +x /opt/sailfish/entrypoint.sh
# writable corpus dir (the data plane scrapes local transcripts into here)
RUN mkdir -p /opt/sailfish/data

EXPOSE 22343
# override the base image's llama-server entrypoint
ENTRYPOINT ["/opt/sailfish/entrypoint.sh"]
