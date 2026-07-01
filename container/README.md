# Sailfish container — two backends, pick your fish

Sovereign `gemma-4-E4B-it` on one 12 GB Ampere card (RTX 3060). **Two serve options**, same OpenAI API on **:22343** — choose by which folder you `docker compose up` in.

```
container/
  Dockerfile · serve.sh · docker-compose.yml   →  BACKEND #1: vLLM   (high-perf, currently blocked — see below)
  llamacpp/docker-compose.yml                   →  BACKEND #2: llama.cpp (works on the 3060 today)
```

## Backend #2 — llama.cpp  ✅ the one that runs today
```bash
cd container/llamacpp && docker compose up -d        # -> http://localhost:22343 (OpenAI API)
pwsh ../../harness/tool_harness.ps1 -Backend openai -Port 22343
```
- Serves the **e4b GGUF Ollama already proved works** (mounted read-only from `~/.ollama/models/blobs`) — no re-download, no quant bug.
- Spec-decode here is **zero-VRAM prompt-lookup**, because a draft *model* won't fit (e4b 8.95 G + e2b 6.67 G > 12 G). That lookup slot is **exactly where the Sailfish n-gram tool-drafter (2.56×) plugs in** — its corpus is the upgrade from llama.cpp's context-only lookup to a tool-call-primed one.
- Runs *alongside* Ollama (own port), so you can A/B them.

## Backend #1 — vLLM  🧱 blocked, scaffolded
```bash
cd container && docker compose up --build -d
```
- The real spec-decode machinery (MTP/assistant drafter, async, Marlin INT4) — **faster ceiling** when it loads.
- **Blocked today:** vLLM 0.24 can't load a quantized gemma4 — every community INT4 checkpoint quantizes the audio/vision towers and the loader demands bf16 there (`no parameter audio_tower…weight`, vLLM #40247). `ciocan` and `Vishva007` both fail identically.
- **Unblock path:** swap the base image to a **gemma4-capable vLLM tag** (build ARG `SAILFISH_VLLM_IMAGE`) or a tower-bf16 INT4 checkpoint. Until then, use Backend #2.

## Knobs
| var | applies to | default | note |
|---|---|---|---|
| `SAILFISH_PORT` | both | `22343` | own port; runs beside Ollama (11434) |
| `OLLAMA_MODELS` | llama.cpp | `~/.ollama/models` | where the GGUF blobs live |
| `SAILFISH_TARGET` / `SAILFISH_DRAFTER` / `SAILFISH_SPEC_K` | vLLM | see Dockerfile | target / drafter / spec tokens |
| `SAILFISH_GPU_UTIL` | vLLM | `0.85` | leave KV headroom on 12 GB |

## The benchmark that matters
Baseline (bare e4b) should land near **Ollama's high-60s TPS**. The win is spec-decode on top — and on this card that means **prompt-lookup primed with the tool-call corpus**, not a draft model. Measure with the harness `-Backend openai`.
