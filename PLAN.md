# Sailfish — Product Plan v2

> Your agents' tool calls → a fast private model on your own GPU.
> Everything we learned in the Fast Gemma Challenge (472 TPS @ 2.377 PPL, verified reproduction),
> productized: a local appliance, a training pipeline, and a hosted service at `sailfish.nuts.services`.

---

## 1. Mission

Take the Gemma-challenge toolkit — INT4 bandwidth engineering, MTP speculative decoding, SeqKD drafter
training — and ship it three ways:

1. **Local appliance** (Docker Hub image): runs on any consumer GPU, autodetects the card, serves the
   fastest valid configuration on port `22343` with a local web UI.
2. **Training pipeline**: harvest the user's own agent tool-call history (via nemesis8), curate it with a
   frontier model, fine-tune Gemma-4-E4B on it — on *our* L4, or shipped to *their* Google Cloud A100.
3. **Hosted service** (`sailfish.nuts.services`): our tool-tuned model on serverless L4, scale-to-zero,
   nuts-auth login — for people who don't want to run containers at all.

---

## 2. The serving decision tree (the core product logic)

**The exclusivity rule:** the MTP drafter is *coupled to its target* — fine-tuning the target invalidates
the drafter. So a card either runs the **fine-tuned model bare** or the **stock model + drafter**, never
both. (v2 unlock, later: SeqKD-retrain the drafter *against our fine-tuned target* → tool-sharp AND fast.)

```
GPU autodetect (VRAM + architecture + driver) at container start:

< 16 GB   (e.g. RTX 3060 12GB)      TIER B — "bare & sharp"
    engine: llama.cpp (server-cuda)
    model:  fine-tuned E4B GGUF (tool-sharp), Q4_K_M
    no drafter (12GB measured: e2b draft = no win on prose; fused MTP head won't run in llama.cpp)
    optimizations: -fa on, full offload (-ngl 99), tuned ctx, everything from the challenge that ports
    measured floor today: ~66 TPS bare on the 3060

>= 16 GB  (A10G-class, L4, 3090/4090, 5090-32GB)   TIER A — "stock & fast"
    engine: vLLM
    model:  STOCK E4B INT4 (properly-built checkpoint, towers unquantized) + MTP drafter
    the Gemma-challenge technique: K=7 speculation, our trained drafter
    reference: 472 TPS @ 2.377 PPL on A10G (24GB, Ampere)

TIER RULE IS VRAM + CHECKPOINT, NOT SILICON. The challenge ran vLLM+MTP on *Ampere* (A10G).
Architecture matters only for kernel/feature selection:
    sm_86 Ampere  — Marlin INT4, no FP8
    sm_89 Ada/L4  — + FP8 KV/weights paths available
    sm_120 Blackwell (5090) — needs recent vLLM build; verify wheel support in P1
```

Fallback ladder if a tier fails to load: Tier A → Tier B → CPU-only refusal with a clear message.

---

## 3. Components

### 3.1 Local appliance (`container/`) — the Docker Hub image
- **Image:** `deepbluedynamics/sailfish` on Docker Hub (reuse grubcrawler's publish pipeline:
  `cloudbuild.yaml` + `dockerhub-token` secret + `deploy.ps1 -Target dockerhub-cloud`).
- **One port, 22343:** a small **gateway** process serves the web UI at `/`, the OpenAI-compatible API at
  `/v1/*` (proxied to the engine), and the Sailfish control API at `/api/*`. One URL to click:
  `http://localhost:22343`.
- **Startup sequence:** GPU autodetect → pick tier → pull/verify model artifacts (cached volume) → start
  engine → gateway up. The UI's landing view says exactly what tier you're on and why.
- **Engines shipped in-image:** llama.cpp server-cuda + vLLM; only one starts.
- Windows-first: docs and installer assume Docker Desktop + NVIDIA runtime on Windows.

### 3.2 Local web UI (in the gateway)
Views:
1. **Status** — card detected, tier, model loaded, live TPS, endpoint URL for harnesses.
2. **Data** — nemesis8 connection status; discovered sources with counts; local-fallback scrape
   (Claude Code transcripts) if nemesis8 absent; size warning before ingest.
3. **Curate** — frontier-model config (provider: Anthropic / OpenAI / Gemini / GLM; token pasted into
   container env — explicitly not hardened for now); run curation; **review the proposed training set**
   before accepting.
4. **Train** — choose backend: (a) *our hosted trainer* (default for stock users), (b) **their Google
   Cloud** (paste org/project ID + region; page shows the exact `gcloud` commands to run in their own
   shell — auth stays on their machine, we never hold their credentials), (c) advanced/local.
5. **Deploy** — when a fine-tune completes: download GGUF, hot-swap the serving model, restart engine.

### 3.3 Data plane — nemesis8 integration
- **Discovery:** gateway probes the nemesis8 controller from inside the container at
  `host.docker.internal:18042` (best current candidate port — **verify against nemesis8 source**; it must
  be host-exposed, not container-internal). Not listening → show "nemesis8 not running" + fallback.
- **Contract:** `NEMESIS8_INTEGRATION.md` (already written) — `/v1/training/sources`,
  `/v1/training/export` (streamed, ANSI-scrubbed), `/v1/training/stats`. nemesis8 owns per-agent parsers
  (all agents it supports: antigravity, opencode, Claude Code local, …); Sailfish keeps its own Claude
  Code scraper only as the fallback bridge.
- **Handoff doc for the nemesis8 agent** stays the spec above + discovery/port requirements.

### 3.4 Curation agent (new component, `curate/`)
- Input: raw exported traces (may be tens of MB — surface size + estimated token cost *before* sending).
- A configurable frontier model reviews samples + statistics and returns: cleaned/deduped examples in the
  **exact serving chat format** (tool schemas in system prompt, gemma tool-call syntax in completions —
  this formatting is what makes the fine-tune transfer), quality flags, and a summary the user reviews.
- Provider adapter: Anthropic / OpenAI / Gemini / GLM top models; token via env; batched map-reduce over
  the corpus so it works on big exports.

### 3.5 Training plane (`train/`)
- **Objective:** SFT/LoRA on curated tool-call data in serving format (fix the current
  `finetune_target.py` formatting gap first). Drafter work (Tier A) uses **SeqKD** — the challenge lesson:
  train on the target's own greedy tokens, never a surrogate.
- **Backends:**
  - **Company (default): Cloud Run Job + L4** on `gnosis-459403` — LoRA on E4B fits L4 24GB; ephemeral,
    pay-per-run, no instance lifecycle. Mirrors shivvr's Cloud Build pattern.
  - **BYO Google Cloud A100:** we generate a copy-paste script (org/project ID templated in): enable APIs,
    `gcloud compute instances create` (a2-highgpu-1g, spot), pull trainer image, run, upload result to
    their GCS/HF, tear down. Runs in *their* shell where *their* gcloud is authed.
  - Output artifacts: merged HF weights + converted GGUF (Q4_K_M) + provenance manifest.
- **Company model:** we fine-tune our own E4B on DeepBlue tool-run data (Hyperia + nemesis8 call patterns)
  — this is the model the hosted service serves, and the default local Tier B model.

### 3.6 Hosted service — `sailfish.nuts.services`
- **Serving:** Cloud Run, `--gpu-type nvidia-l4`, `--min-instances 0` (scale-to-zero; cold-start lag
  accepted for now; dedicated instance later if traffic warrants). **Models baked into the image**
  (shivvr's `Dockerfile.models` / `--rebuild-models` trick) so cold start ≠ model download.
- **Auth:** nuts-auth (JWKS gate like shivvr: `NUTS_AUTH_JWKS_URL=https://auth.nuts.services/...`).
  Logged-in users hit our tool-tuned small model.
- **Landing page:** public, styled to match the nuts.services family (grubcrawler/shivvr layout). Sailfish
  branding — 🗡️🐟, "the fastest fish in the ocean." Page contents: what it is, live demo/login, **the
  installer** (one-line PowerShell + docker pull for local), docs for the BYO-A100 path.
- **DNS:** wildcard exists → map `sailfish.nuts.services` via Cloud Run domain-mapping (shivvr's
  OPERATIONS.md has the exact commands). **Deploy early** — landing page + hello-world service first, so
  the URL is live while the guts are built.

### 3.7 Hyperia integration (SKETCH ONLY — for the Hyperia agent, not our work)
Detection ladder Hyperia should implement:
1. Probe `localhost:22343/api/status` → local Sailfish found → use it directly.
2. Not running → ask nemesis8 (`:18042`) to offer install / `docker run` of the appliance.
3. User declines containers → fall back to `https://sailfish.nuts.services` with nuts-auth login.
The endpoint is OpenAI-compatible either way, so Hyperia's client code is identical across all three.
Deliverable to the Hyperia agent: a short `HYPERIA_INTEGRATION.md` with this ladder + the status API shape.

---

## 4. Build phases

- **P0 — Site live (deploy early, per Kord):** landing page + scale-to-zero Cloud Run stub + DNS mapping
  `sailfish.nuts.services` + installer link. Mirrors shivvr/grubcrawler scaffolding. *Ship first.*
- **P1 — Local appliance MVP:** gateway (UI + proxy) + GPU autodetect + Tier B (llama.cpp, stock E4B GGUF
  until our fine-tune exists) + Docker Hub publish. Click-a-URL works on the 3060.
- **P2 — Data + curation:** nemesis8 probe + fallback scraper + size warnings + frontier-model curation
  with review UI. (Parallel: hand nemesis8 agent the integration spec; hand Hyperia agent the sketch.)
- **P3 — Training:** fixed serving-format trainer → Cloud Run Job (L4) for company runs → **our tool-tuned
  E4B v1** (Hyperia/nemesis8-aware) → swap into hosted service + Tier B default. BYO-A100 script generator.
- **P4 — Tier A (drafter):** proper public INT4 E4B checkpoint (towers unquantized) + our drafter via the
  challenge stack on 16GB+; arch matrix validation (Ampere / Ada+FP8 / Blackwell wheels); test on the
  32GB 5090.
- **P5 — Hardening:** dedicated L4 fallback if hot, token security, multi-model hosting, drafter-on-
  fine-tuned-target (the v2 unlock).

## 5. Division of labor
- **Sailfish agents (us):** everything in §3.1–3.6, the plan's P0–P5.
- **nemesis8 agent (Tyrannosaurus):** implement `NEMESIS8_INTEGRATION.md` API on the controller; confirm
  the host-exposed port (candidate: 18042).
- **Hyperia agent:** implement the §3.7 detection ladder from `HYPERIA_INTEGRATION.md` (we write the doc,
  they write the code).

## 6. Open questions for review
1. **Hosted-model shape:** one shared tool-tuned E4B for all users at launch (simplest), per-user
   fine-tunes hosted later?
2. **Public INT4 checkpoint for Tier A:** the challenge's frontier weights are org-gated. Plan assumes we
   build/publish our own properly-quantized E4B INT4 (llm-compressor W4A16, towers in bf16). Confirm we're
   OK publishing that artifact under DeepBlueDynamics on HF.
3. **Installer form:** PowerShell one-liner (`irm sailfish.nuts.services/install.ps1 | iex`) that checks
   Docker Desktop + NVIDIA runtime and runs the container — acceptable for v1?
4. **Curation cost guardrail:** hard cap on tokens sent to the frontier model per run (default ~$5-worth,
   user-raisable)?
5. **nemesis8 port 18042:** best candidate from source grep — needs confirmation from Tyrannosaurus.
