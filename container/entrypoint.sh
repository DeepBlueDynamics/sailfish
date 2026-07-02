#!/usr/bin/env bash
# Sailfish appliance entrypoint — one container, one port (22343).
#
#   1. autodetect the GPU (VRAM + arch)
#   2. launch the serving engine on an internal port (8080)
#   3. launch the gateway (site + /v1 proxy + /api/status) on the published port (22343)
#
# P1 ships the Tier B engine (llama.cpp + zero-VRAM n-gram speculation) for ALL cards — it is the
# measured, working path (76 tool-runs / 177 repetitive TPS on a 3060). A >=16 GB card is *Tier-A
# capable* (vLLM + MTP drafter, the challenge stack); that engine ships as a separate image in P4,
# and /api/status tells the user their card can run it.
set -uo pipefail

ENGINE_PORT="${SAILFISH_ENGINE_PORT:-8080}"
GATEWAY_PORT="${SAILFISH_PORT:-22343}"
MODEL_HF="${SAILFISH_GGUF:-ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M}"
ALIAS="${SAILFISH_MODEL:-gemma4-e4b}"
CTX="${SAILFISH_CTX:-8192}"
SPEC="${SAILFISH_SPEC:-ngram-mod}"   # zero-VRAM n-gram speculation (Tier B default)

# This image serves Tier B. Report it honestly; the gateway notes Tier-A capability separately.
export SAILFISH_TIER="${SAILFISH_TIER:-B}"
export SAILFISH_ENGINE_URL="http://127.0.0.1:${ENGINE_PORT}/v1"

# --- GPU probe (informational; the gateway does its own for /api/status) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  echo "[sailfish] GPU: ${GPU_LINE:-<none visible>}"
else
  echo "[sailfish] WARNING: nvidia-smi not found — no GPU passed into the container. CPU fallback is slow."
fi

# --- resolve the llama-server binary (path differs across ggml-org image tags) ---
LLAMA_BIN="$(command -v llama-server 2>/dev/null || true)"
[ -z "$LLAMA_BIN" ] && [ -x /app/llama-server ] && LLAMA_BIN=/app/llama-server
[ -z "$LLAMA_BIN" ] && [ -x /llama-server ] && LLAMA_BIN=/llama-server
if [ -z "$LLAMA_BIN" ]; then
  echo "[sailfish] FATAL: llama-server binary not found in image"; exit 1
fi

echo "[sailfish] tier=$SAILFISH_TIER engine=llama.cpp spec=$SPEC model=$MODEL_HF ctx=$CTX"
echo "[sailfish] starting engine on :${ENGINE_PORT} ..."
"$LLAMA_BIN" \
  -hf "$MODEL_HF" \
  --spec-type "$SPEC" \
  --alias "$ALIAS" \
  -ngl 99 -c "$CTX" -fa on \
  --host 127.0.0.1 --port "$ENGINE_PORT" &
ENGINE_PID=$!
trap 'kill $ENGINE_PID 2>/dev/null' EXIT INT TERM

# --- wait for the engine to answer (first run downloads the GGUF ~5.3 GB) ---
echo "[sailfish] waiting for engine health (first start downloads the model) ..."
for i in $(seq 1 600); do
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "[sailfish] FATAL: engine exited during startup"; exit 1
  fi
  if curl -fsS "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null 2>&1; then
    echo "[sailfish] engine healthy after ${i}s"; break
  fi
  sleep 1
done

echo "[sailfish] starting gateway on :${GATEWAY_PORT} -> http://localhost:${GATEWAY_PORT}"
exec "${SAILFISH_PY:-python3}" -m uvicorn app.main:app --host 0.0.0.0 --port "$GATEWAY_PORT"
