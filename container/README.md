# Sailfish container

The appliance the installer pulls — **one image, one port (22343)**: the llama.cpp Tier B engine
+ the FastAPI gateway (web UI at `/`, OpenAI API at `/v1`, control plane at `/api/*`). The entrypoint
autodetects the GPU, launches the engine internally, and fronts it with the gateway.

```
container/
  Dockerfile         →  THE appliance: llama.cpp + gateway (deepbluedynamics/sailfish)   ✅ ships P1
  entrypoint.sh      →  autodetect GPU → launch engine (:8080) → launch gateway (:22343)
  Dockerfile.vllm    →  Tier A engine: vLLM + INT4 + MTP drafter (the challenge stack)    🧱 P4, blocked
  serve.sh           →  vLLM launch script (used by Dockerfile.vllm)
  llamacpp/*.yml     →  bare-engine compose files for local A/B of drafter modes (no gateway)
```

## Build & run (the appliance)
Build context is the **repo root** (the image COPYs `app/`, `site/`, `container/entrypoint.sh`):
```bash
docker build -f container/Dockerfile -t deepbluedynamics/sailfish .
docker run -d --name sailfish --gpus all -p 22343:22343 \
  -v sailfish-cache:/root/.cache deepbluedynamics/sailfish
# -> http://localhost:22343  (web UI), http://localhost:22343/v1 (OpenAI API)
pwsh ../harness/tool_harness.ps1 -Backend openai -Port 22343
```
First start downloads the GGUF (~5.3 GB) into the named volume; subsequent starts are instant.
Publish to Docker Hub via `cloudbuild-appliance.yaml` (grubcrawler's `dockerhub-token` pattern).

## Tiers
- **Tier B (this image, all cards at P1):** llama.cpp + stock E4B Q4_K_M + `--spec-type ngram-mod`
  (zero-VRAM n-gram speculation). Measured on a 12 GB RTX 3060: **75.7 avg TPS** tool-runs (6/6
  accuracy), **177 TPS** repetitive (2.7×), ~67 prose (no regression). An e2b *draft model* was a wash
  (68.5, 8.8 GB) — don't use it on 12 GB cards.
- **Tier A (`Dockerfile.vllm`, P4):** vLLM + our INT4 W4A16 target + MTP drafter K=7 — the challenge
  stack (reference 472 TPS @ 2.377 PPL on A10G). Blocked until we publish `DeepBlueDynamics/sailfish-e4b-int4`:
  vLLM can't load community-quantized gemma-4 (audio-tower bf16 loader bug, vLLM #40247). `/api/status`
  reports Tier-A *capability* on ≥16 GB cards so users know it's coming.

## Bare-engine compose (for drafter A/B, no gateway)
`llamacpp/docker-compose.yml` runs `llama-server` directly on 22343 with the Tier B flags — handy for
measuring drafter modes in isolation (`ngram` vs `mtp` vs bare). The appliance image is the product;
these composes are a bench.

## Knobs (appliance)
| var | default | note |
|---|---|---|
| `SAILFISH_PORT` | `22343` | published gateway port |
| `SAILFISH_ENGINE_PORT` | `8080` | internal engine port (not published) |
| `SAILFISH_GGUF` | `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` | swap to our fine-tuned GGUF when it ships |
| `SAILFISH_SPEC` | `ngram-mod` | speculation mode |
| `SAILFISH_CTX` | `8192` | context length |
| `SAILFISH_TIER` | `B` | set by the image; `auto`/`A`/`B` override for /api/status |
