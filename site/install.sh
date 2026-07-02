#!/bin/sh
# Sailfish local installer (macOS / Linux).
#   curl -fsSL https://sailfish.nuts.services/install.sh | sh
set -e
echo "sailfish - sovereign fast inference"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Install Docker Engine (with the NVIDIA Container Toolkit for GPU) then re-run."
  echo "  https://docs.docker.com/engine/install/"
  exit 1
fi

echo "checking GPU passthrough into containers..."
GPU=""
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 \
     nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >/tmp/sf_gpu 2>/dev/null; then
  GPU=$(cat /tmp/sf_gpu)
  echo "GPU: $GPU"
  GPUFLAG="--gpus all"
else
  echo "No GPU visible to Docker (install nvidia-container-toolkit for acceleration). CPU fallback is slow."
  GPUFLAG=""
fi

echo "pulling deepbluedynamics/sailfish ..."
docker pull deepbluedynamics/sailfish:latest
docker rm -f sailfish >/dev/null 2>&1 || true
docker run -d --name sailfish $GPUFLAG -p 22343:22343 \
  -v sailfish-cache:/root/.cache \
  --restart unless-stopped deepbluedynamics/sailfish:latest

echo ""
echo "sailfish is up -> http://localhost:22343"
echo "OpenAI-compatible API at http://localhost:22343/v1"
command -v open >/dev/null 2>&1 && open http://localhost:22343 || true
