# Sailfish :: container — sovereign fast model on one 12 GB Ampere card (RTX 3060, sm_86).
#
# vLLM OpenAI server: gemma-4-E4B-it INT4 target + speculative drafter (spec-decode baked in).
# The drafter is swappable via SAILFISH_DRAFTER — point it at the tuned tool-drafter when ready.
FROM vllm/vllm-openai:latest

LABEL org.opencontainers.image.title="sailfish.nuts.services"
LABEL org.opencontainers.image.description="Sovereign fast gemma-4-E4B with speculative decoding"

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    SAILFISH_TARGET=ciocan/gemma-4-E4B-it-W4A16 \
    SAILFISH_DRAFTER=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant \
    SAILFISH_SPEC_K=5 \
    SAILFISH_QUANT=auto \
    SAILFISH_MAXLEN=8192 \
    SAILFISH_GPU_UTIL=0.92 \
    SAILFISH_PORT=22343

COPY serve.sh /opt/sailfish/serve.sh
RUN chmod +x /opt/sailfish/serve.sh

EXPOSE 22343
ENTRYPOINT ["/opt/sailfish/serve.sh"]
