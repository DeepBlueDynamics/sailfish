# Sailfish container

The appliance the installer pulls — **one image, one port (22343)**: the llama.cpp Tier B engine
+ the FastAPI gateway (web UI at `/`, OpenAI API at `/v1`, control plane at `/api/*`). The entrypoint
autodetects the GPU, launches the engine internally, and fronts it with the gateway.

```
container/
  Dockerfile         →  THIN appliance: llama.cpp + gateway, model downloaded on first run  → :latest
  Dockerfile.bundled →  BUNDLED appliance: same + the GGUF baked in (offline)                → :bundled
  entrypoint.sh      →  autodetect GPU → resolve model (baked>URL>HF) → engine (:8080) → gateway (:22343)
  Dockerfile.vllm    →  Tier A engine: vLLM + INT4 + MTP drafter (the challenge stack)    🧱 P4, blocked
  serve.sh           →  vLLM launch script (used by Dockerfile.vllm)
  llamacpp/*.yml     →  bare-engine compose files for local A/B of drafter modes (no gateway)
```

## Two variants
| tag | model | size | use when |
|---|---|---|---|
| **`:latest`** (thin) | downloaded on first run into a volume | ~7 GB | default — smaller pull, **swappable model** |
| **`:bundled`** | baked into the image | ~12 GB | air-gapped / no first-run download / pinned weights |

The thin image resolves its model source in priority order (see `entrypoint.sh`):
1. `SAILFISH_MODEL_PATH` — a baked-in file (this is what `:bundled` sets).
2. `SAILFISH_MODEL_URL` — download a **trained** model from `gs://bucket/model.gguf` (rewritten to
   `storage.googleapis.com`, optional `SAILFISH_MODEL_TOKEN` bearer) **or** any `https://` URL; cached in
   the volume. This is the "serve the model a user trained" path.
3. `SAILFISH_GGUF` — the stock model from Hugging Face (default).
```bash
# serve a trained model from your GCS bucket instead of the stock one:
docker run -d --gpus all -p 22343:22343 -v sailfish-cache:/root/.cache \
  -e SAILFISH_MODEL_URL=gs://my-bucket/gemma4-e4b-toolft.gguf \
  deepbluedynamics/sailfish:latest
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
| `SAILFISH_MODEL_PATH` | *(set in `:bundled`)* | serve a baked-in local GGUF; no download (highest priority) |
| `SAILFISH_MODEL_URL` | — | download a trained GGUF from `gs://…` or `https://…` (2nd priority) |
| `SAILFISH_MODEL_TOKEN` | — | optional bearer for a private `SAILFISH_MODEL_URL` |
| `SAILFISH_GGUF` | `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` | stock HF repo:quant (default, lowest priority) |
| `SAILFISH_SPEC` | `ngram-mod` | speculation mode |
| `SAILFISH_CTX` | `8192` | context length |
| `SAILFISH_TIER` | `B` | set by the image; `auto`/`A`/`B` override for /api/status |
