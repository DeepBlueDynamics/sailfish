#!/usr/bin/env bash
# Sailfish sovereign serve — gemma-4-E4B-it INT4 target + speculative drafter on one Ampere GPU.
# OpenAI-compatible API on :8000. Every knob is an env var so the container is the same on any card.
set -euo pipefail

TARGET="${SAILFISH_TARGET:-ciocan/gemma-4-E4B-it-W4A16}"   # INT4 GPTQ, fits a 12GB Ampere card
DRAFTER="${SAILFISH_DRAFTER:-google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant}"
K="${SAILFISH_SPEC_K:-5}"
QUANT="${SAILFISH_QUANT:-auto}"   # 'auto' = let vLLM detect from the checkpoint config
MAXLEN="${SAILFISH_MAXLEN:-8192}"
UTIL="${SAILFISH_GPU_UTIL:-0.92}"
PORT="${SAILFISH_PORT:-22343}"   # Sailfish's own port

echo "[sailfish] target=$TARGET drafter=$DRAFTER K=$K quant=$QUANT maxlen=$MAXLEN util=$UTIL port=$PORT"

SPEC='{"model":"'"$DRAFTER"'","num_speculative_tokens":'"$K"'}'

# DRAFTER=none -> serve the bare target (baseline to beat: Ollama high-60s TPS)
SPEC_ARGS=(--speculative-config "$SPEC")
if [[ "$DRAFTER" == "none" ]]; then SPEC_ARGS=(); echo "[sailfish] spec-decode DISABLED (baseline mode)"; fi

# QUANT=auto -> omit the flag, let vLLM detect from the checkpoint config
QUANT_ARGS=(--quantization "$QUANT")
if [[ "$QUANT" == "auto" ]]; then QUANT_ARGS=(); fi

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$TARGET" \
  "${QUANT_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --host 0.0.0.0 --port "$PORT"
