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

# --- resolve the model source: baked file > URL (gs:// | https, for trained models) > HF repo (stock) ---
MODEL_ARGS=()
if [ -n "${SAILFISH_MODEL_PATH:-}" ] && [ -f "${SAILFISH_MODEL_PATH:-}" ]; then
  echo "[sailfish] model: baked-in ($SAILFISH_MODEL_PATH)"
  MODEL_ARGS=(-m "$SAILFISH_MODEL_PATH")                    # bundled image — no download, offline-ready
elif [ -n "${SAILFISH_MODEL_URL:-}" ]; then
  url="$SAILFISH_MODEL_URL"
  case "$url" in gs://*) url="https://storage.googleapis.com/${url#gs://}" ;; esac   # GCS object → https
  dest="/root/.cache/sailfish/model.gguf"; mkdir -p "$(dirname "$dest")"
  if [ ! -s "$dest" ]; then
    echo "[sailfish] model: downloading from $SAILFISH_MODEL_URL"
    AUTH=(); [ -n "${SAILFISH_MODEL_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${SAILFISH_MODEL_TOKEN}")
    curl -fSL "${AUTH[@]}" -o "$dest" "$url" || { echo "[sailfish] FATAL: model download failed"; exit 1; }
  else
    echo "[sailfish] model: cached ($dest)"
  fi
  MODEL_ARGS=(-m "$dest")                                    # trained model pulled from GCS/HF/any https
else
  echo "[sailfish] model: Hugging Face ($MODEL_HF)"
  MODEL_ARGS=(-hf "$MODEL_HF")                               # stock default — first-run download into the cache volume
fi

# --- optional trained MTP draft head (VSD): spec=draft-mtp + a tiny GGUF beats ngram on fresh content.
# Measured on a 3060 (2026-07-03): 76.5 tok/s agentic / 82.4 prose vs 65.8/63.7 ngram-mod (+16%/+29%).
# NOTE: gemma4-assistant draft + flash-attn fatals on sm_86 (ggml fattn.cu head-dim case) -> when a
# draft is set, FA defaults to off (the FA tax is ~2 tok/s; the draft more than pays it back).
FA="${SAILFISH_FA:-on}"
DRAFT_ARGS=()
if [ -n "${SAILFISH_DRAFT_GGUF:-}" ]; then
  DRAFT_ARGS=(-md "$SAILFISH_DRAFT_GGUF" --spec-draft-n-max "${SAILFISH_DRAFT_NMAX:-3}" -ngld 99)
  FA="${SAILFISH_FA:-off}"
  echo "[sailfish] draft head: $SAILFISH_DRAFT_GGUF (n-max ${SAILFISH_DRAFT_NMAX:-3}, fa=$FA)"
fi

echo "[sailfish] tier=$SAILFISH_TIER engine=llama.cpp spec=$SPEC ctx=$CTX"
echo "[sailfish] starting engine on :${ENGINE_PORT} ..."
"$LLAMA_BIN" \
  "${MODEL_ARGS[@]}" \
  --spec-type "$SPEC" \
  "${DRAFT_ARGS[@]}" \
  --alias "$ALIAS" \
  -ngl 99 -c "$CTX" -fa "$FA" \
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
