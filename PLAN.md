# Sailfish — Product Plan (v3, the long one)

> **Your agents' tool calls → a fast private model on your own GPU.**
> Everything learned in the Fast Gemma Challenge (verified 472 TPS @ 2.377 PPL reproduction on A10G),
> productized three ways: a local appliance, a training pipeline, and a hosted service at
> `sailfish.nuts.services`.

This is the working plan. Decisions in §10 are settled — don't re-litigate them casually. Measured
numbers are cited wherever they exist; estimates are labeled as estimates.

---

## Build status — 2026-07-02

Code is on `main`. Not yet deployed to Cloud Run / published to Docker Hub (those are actions, below).

- **P0 (ship surface) — DEPLOYED.** Live on Cloud Run (`sailfish` service, gnosis-459403/us-central1);
  `sailfish.nuts.services` domain-mapping created + DNS resolves via the wildcard (Google cert
  provisioning). Landing page verified: GitHub links (topbar/hero/footer, grub-style), login-gated
  Ollama guide, nuts.services family footer (sdrrand's, "Auth Dashboard"→"Dashboard"). The
  **nuts.services family page lists sailfish** (nuts-site redeployed). Local appliance opens the
  console at `/`; hosted opens the landing. `install.ps1`/`install.sh` mount `~/.claude` read-only.
  Kaniko note: no top-level `images:` (Kaniko pushes via --destination); use `/health` not `/healthz`
  (GFE swallows the exact path). Console + landing both carry the family footer.
- **P1 (appliance) — RUNTIME-VERIFIED on a real RTX 3060; only the publish is BLOCKED.** Built the
  image and ran it end-to-end: autodetect → tier B → llama.cpp+ngram engine (:8080) → gateway (:22343).
  `/api/status` returns the full contract (RTX 3060, 12 GB, ampere). **Harness: 6/6 tool-call accuracy,
  74 avg decode TPS** (clears the ≥70 bar; zero tool errors). Fixed a load-bearing boot bug found only by
  running: overriding the base image's `WORKDIR=/app` left `llama-server` unable to find its `.so`; fix =
  `LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64` in the Dockerfile (+ a `.dockerignore`). **Remaining
  BLOCKER: the `dockerhub-token` secret doesn't exist in gnosis-459403 — needs Kord's Docker Hub write
  token to publish `deepbluedynamics/sailfish`; until then the installer one-liner pulls a nonexistent image.**
- **P2 (data + curation) — code done, verified end-to-end.** Python scraper twin (25,960 calls/85
  tools/0 errors on real transcripts); `app/data.py` (nemesis8 probe + sources + scrape + zip);
  `app/curate.py` (estimate-free, per-batch cap halt, Anthropic + OpenAI-compat); `finetune_target.py`
  serving-format fix (tools in prompt, gemma-native tool_calls, completion-only loss). Cost gate
  proven ($18.88 > $5 cap → refused; run without confirm → 400).
- **P3 (training) — in progress.** BYO-cloud zip export exists; templated gcloud script generator next.
- **UI shell (Status/Data/Curate/Train/Deploy views): TODO** — backend APIs all exist; only the
  landing page is built so far.

---

## 1. Mission

Agent users generate thousands of tool calls a day. That history is (a) the perfect training data for a
small model that's *sharp at tool-calling*, and (b) the perfect food for a speculative drafter, because
tool calls are structured and repetitive. Sailfish turns that history into a fast, private model:

- **Private**: runs on your card, your data never leaves unless *you* ship it to *your* cloud.
- **Fast**: the Gemma-challenge toolkit — INT4 bandwidth engineering, speculative decoding, SeqKD —
  applied to consumer hardware.
- **Self-serve**: click a URL, see what your card can do, pull your data, curate, train, swap, serve.

## 2. The evidence base (what we proved before writing this)

From the challenge (A10G, 24 GB, Ampere):
- Verified reproduction: **472 TPS @ 2.377 PPL** (INT4 target + MTP drafter K=7 + split-KV + 12k lm-head
  + CUDA-graph capture). Submitted as `weber-moe/splitkv-k7-byteshark-repro-v0`.
- **The drafter lesson:** our VSD-trained drafter lost to a 1-epoch SeqKD drafter (319 vs 413 TPS) on
  byte-identical architecture → acceptance is everything, and the objective must be the target's own
  greedy tokens (**SeqKD**), never a surrogate.
- **The coupling rule:** the MTP drafter is trained against a specific target. Fine-tune the target and
  the drafter's acceptance collapses. Hence the exclusivity rule in §4.

From local measurement (RTX 3060 12 GB, Ampere, llama.cpp `server-cuda`, ggml-org E4B Q4_K_M):
- Bare: **~66 TPS** (55–71 range).
- `--spec-type ngram-mod` (zero-VRAM n-gram speculation): **75.7 avg TPS on the tool-run harness**
  (6/6 tool accuracy), **177 TPS on repetitive output (2.7×)**, ~67 on prose (no regression). This is now
  the Tier B default.
- e2b as a draft *model*: fits (8.8 GB) but **no speedup** (68.5) — heavy drafts lose on 12 GB. Rejected.
- vLLM cannot load community-quantized gemma-4 anywhere (audio-tower layout bug) and nothing loadable
  fits 12 GB → the vLLM/MTP stack needs a 16 GB+ card *and* a properly-built INT4 checkpoint (§5).
- `--spec-type draft-mtp` exists in current llama-server — **the sleeper**: if the MTP head runs in
  llama.cpp, challenge-grade drafting reaches small cards. Test in flight; plan updates on result.
- Corpus: **25,479 real tool calls** scraped from local agent transcripts, ANSI-scrubbed; n-gram analysis
  showed 61% of tool-call tokens are free-draftable (2.56 tokens/verify).

## 3. Product shapes (three ways to use Sailfish)

1. **Local appliance** — `docker pull deepbluedynamics/sailfish`, one container, autodetects the GPU,
   serves `http://localhost:22343` (web UI at `/`, OpenAI API at `/v1`). Windows-first; Mac/Linux
   supported by the installer.
2. **Training pipeline** — from the appliance UI: pull tool-call data (nemesis8), curate with a frontier
   model, then train (a) on our hosted trainer later, or (b) on **their own Google Cloud** via a
   generated script + a data zip. Output swaps into the appliance.
3. **Hosted service** — `sailfish.nuts.services`: our model on serverless L4, scale-to-zero, nuts-auth
   login. For people who won't run containers. (Launch = stock model; our tool-tuned model lands later.)

## 4. Serving tiers — the decision tree

**Exclusivity rule:** fine-tuned target ⊕ drafter — never both (coupling, §2). A card runs either the
*fine-tuned model bare* or the *stock model + drafter*. (v2 unlock, later: SeqKD-retrain the drafter
against our fine-tuned target → both.)

```
GPU autodetect at container start (VRAM, architecture, driver):

TIER B  — < 16 GB (e.g. RTX 3060 12 GB)          "bare & sharp"
   engine: llama.cpp server-cuda
   model:  E4B Q4_K_M GGUF — STOCK at launch; swap to our fine-tuned when it ships (§10.1)
   drafter: ngram-mod (zero VRAM, free) — measured 76 tool-runs / 177 repetitive / 67 prose
   (+ draft-mtp if the sleeper test proves it — would add prose/novel acceleration)

TIER A  — ≥ 16 GB (L4, 3090/4090, A10G-class, 5090-32GB)   "stock & fast"
   engine: vLLM
   model:  STOCK E4B INT4 (our published checkpoint, §5) + trained MTP drafter, K=7
   the challenge technique; reference 472 @ 2.377 on 24 GB Ampere
   16 GB is the entry (INT4 ~9 GB + drafter + KV fits); 24 GB is comfortable

TIER RULE IS VRAM + CHECKPOINT, NOT SILICON. The challenge ran vLLM+MTP on Ampere.
Architecture only selects kernels/features:
   sm_86 Ampere      Marlin INT4; no FP8
   sm_89 Ada (L4)    + FP8 KV/weight paths
   sm_120 Blackwell  needs recent vLLM wheels — validate on the 32 GB 5090 in P4
Fallback ladder: Tier A → Tier B → clear "no supported GPU" message. Never half-load.
```

## 5. Model artifacts & distribution — **Hugging Face, DeepBlueDynamics org**

All model artifacts live on HF under `DeepBlueDynamics/` (decision §10.3; GCS idea dropped):

| artifact | contents | tier | status |
|---|---|---|---|
| `DeepBlueDynamics/sailfish-e4b-gguf` | mirror/pin of stock E4B Q4_K_M GGUF | B | P1 (or pull ggml-org direct at first) |
| `DeepBlueDynamics/sailfish-e4b-int4` | **our** properly-built INT4 W4A16 (llm-compressor, towers bf16) | A | P4 — must build; community quants are broken |
| `DeepBlueDynamics/sailfish-drafter-e4b` | MTP drafter (SeqKD-trained; challenge technique) | A | P4 |
| `DeepBlueDynamics/sailfish-e4b-toolft` | our shared fine-tuned E4B (tool-sharp) + GGUF | B, hosted | P3 — later, per §10.1 |

The appliance downloads from HF on first start into a named docker volume (cache survives restarts).
The hosted service **bakes models into the image** (shivvr `Dockerfile.models` pattern) to kill
cold-start downloads.

## 6. Components

### 6.1 Local appliance (`container/`)
- **Image:** `deepbluedynamics/sailfish` on Docker Hub. Publish via **grubcrawler's existing pipeline**
  (`cloudbuild.yaml` + `dockerhub-token` in Secret Manager + `deploy.ps1 -Target dockerhub-cloud`).
- **Gateway** (small service in-container, likely Rust or Node — pick in P1): one port **22343**:
  - `/` → web UI (§6.2)
  - `/v1/*` → proxied to the engine (OpenAI-compatible, tools supported)
  - `/api/*` → control plane: `status`, `detect`, `data/*`, `curate/*`, `train/*`, `model/swap`
- **Startup:** detect GPU (`nvidia-smi` — VRAM, name, compute capability) → choose tier → fetch/verify
  artifacts → launch engine (llama.cpp or vLLM) → gateway up. Status page explains the choice in plain
  language ("12 GB Ampere → Tier B: fine-tuned-ready bare model + free n-gram speculation").
- **Engines in-image:** llama.cpp server-cuda; vLLM. Only the selected one runs.
- Config via env: `SAILFISH_PORT` (22343), `SAILFISH_TIER` (auto|A|B override), `SAILFISH_N8_URL`
  (default `http://host.docker.internal:18042`), `HF_TOKEN` (optional, for gated pulls),
  `SAILFISH_CURATOR` + `SAILFISH_CURATOR_KEY` (frontier model provider + token).

### 6.2 Local web UI (served by the gateway)
Five views. Style: nuts.services family look (see §6.6), sailfish branding.
1. **Status** — card, tier + why, model id, drafter mode, live TPS, endpoint URL, copy-paste snippets
   for common harnesses.
2. **Data** — nemesis8 probe result (`host.docker.internal:18042`, wired per §10.5); discovered sources
   + counts (`/v1/training/sources`); local fallback scraper for Claude Code transcripts when nemesis8 is
   absent; ingest with **size warning** before anything big moves.
3. **Curate** — provider picker (Anthropic / OpenAI / Gemini / GLM top models), token entry (stored in
   container env only — explicitly not hardened yet, §10.4-note), **cost estimate + cap before running**
   (default cap ≈ $5-equivalent tokens, user-raisable), then a **review screen**: proposed training set,
   stats, reject/accept. Output = curated dataset in **exact serving chat format** (tool schemas in
   system prompt, gemma tool-call syntax in completions — the formatting that makes fine-tunes transfer).
   *Alternative path (§10.4): logged-in users can run curation through our hosted wiring instead of
   bringing a token — post-P3.*
4. **Train** — two buttons at launch:
   (a) **BYO Google Cloud** — form: org ID + project ID + region → generates (1) a **zip** of the curated
   dataset for download and (2) a **copy-paste script** with their values templated in. They run it in
   their own gcloud-authed shell; it uploads the zip, spins an ephemeral A100 (spot), trains, writes the
   artifact to their bucket/HF, tears down. **We never touch their credentials.** Cost note printed.
   (b) **Hosted trainer** — disabled at launch ("coming soon"); post-P3, logged-in users submit the
   curated set to our L4 Cloud Run Job queue.
5. **Deploy** — when a fine-tune artifact exists (their HF repo or uploaded file): download → convert to
   GGUF if needed → hot-swap Tier B model → restart engine → status shows new model id.

### 6.3 Data plane — nemesis8 (spec: `NEMESIS8_INTEGRATION.md`, root)
- Probe `http://host.docker.internal:18042` (wired; env-overridable). Endpoints: `/v1/training/sources`,
  `/v1/training/export` (JSONL stream or **zip**), `/v1/training/stats`. ANSI/control scrubbing at source.
- nemesis8 also owns **container lifecycle** (install/start Sailfish on request — used by Hyperia's
  ladder).
- Until the nemesis8 API exists: Sailfish's built-in scraper (`scrape/scrape_toolcalls.mjs`) covers
  Claude Code transcripts (25k calls proven). The UI treats it as one source among many.
- No other agents are working right now (§10.6) — **we build against the spec and ship the fallback**;
  nemesis8 catches up later.

### 6.4 Training plane (`train/`)
- **Objective:** LoRA SFT on curated tool-call data **in serving format** (fix `finetune_target.py`'s
  formatting gap — currently trains without tool schemas in context; must mirror serving exactly).
  Drafter training uses **SeqKD** (target's own greedy tokens) — `train/distill_e4b.mjs` builds that
  corpus through the local target.
- **Backends:**
  - **BYO A100 (launch):** generated script, their shell, their credentials, our zip. Template lives in
    `train/byo_gcloud/` — parameterized `{ORG_ID}`, `{PROJECT}`, `{REGION}`, `{ZIP_PATH}`. Ephemeral
    a2-highgpu-1g spot instance; teardown in a `trap`. Cost warning in the script header.
  - **Company (post-P3): Cloud Run Job + L4** on `gnosis-459403` — LoRA on E4B fits 24 GB; ephemeral,
    no instance lifecycle; mirrors shivvr's Cloud Build pattern.
- **Dogfood loop (per Kord):** we use the BYO path ourselves to test it end-to-end, and that run produces
  the **shared company model** — fine-tuned on OUR tools (hyperia, nemesis8, grubcrawler, shivvr call
  patterns filtered from our own data). That model then serves hosted + becomes the Tier B default.
- Outputs: merged HF weights + Q4_K_M GGUF + provenance manifest (data sources, curation model, params).

### 6.5 Hosted service — `sailfish.nuts.services`
- **Serving:** Cloud Run, `--gpu 1 --gpu-type nvidia-l4 --min-instances 0` (scale-to-zero; slight
  cold-start accepted; if usage grows it stays warm naturally; dedicated instance only if problems —
  §10 decisions). Models **baked into image**. Many users share one L4 (the model is small); concurrency
  tuned in P0.
- **Auth:** nuts-auth JWKS gate exactly like shivvr (`NUTS_AUTH_JWKS_URL=https://auth.nuts.services/...`).
  Public landing page; API + app behind login.
- **At launch serves the STOCK model** (§10.1); swaps to `sailfish-e4b-toolft` when P3 delivers it.
- **DNS:** wildcard exists → Cloud Run domain-mapping `sailfish.nuts.services` (commands in shivvr's
  OPERATIONS.md). **Ship this first** (§10.7).

### 6.6 Site & installer
- **Landing page:** matches the nuts.services family (grubcrawler/shivvr layout patterns). Branding:
  🗡️🐟 sailfish — "the fastest fish in the ocean." Content: what it is, measured numbers (66 → 76 → 177
  on a $300 card; 472 on datacenter), login for hosted, **installer**, BYO-cloud docs, GitHub link.
- **Installer:** all three OSes (§10.4-installer):
  - Windows: `irm https://sailfish.nuts.services/install.ps1 | iex`
  - Mac/Linux: `curl -fsSL https://sailfish.nuts.services/install.sh | sh`
  - Both: check Docker (+ NVIDIA container runtime on Linux/Windows), `docker pull deepbluedynamics/sailfish`,
    run with `-p 22343:22343 --gpus all` + named volume, open `http://localhost:22343`.
    (Mac = CPU/Metal note: llama.cpp runs, no CUDA — set expectations honestly on the page.)

### 6.7 Integration specs (root of this repo, per Kord)
- `NEMESIS8_INTEGRATION.md` — data API + lifecycle role (port 18042 wired).
- `HYPERIA_INTEGRATION.md` — detection ladder local → nemesis8-install → hosted; endpoint contract.
- Both are **specs we own**; the other teams implement their side whenever they spin up. Nothing in
  P0–P3 blocks on them.

## 7. The drafter program (the speed roadmap)

1. **NOW (shipped):** `ngram-mod` zero-VRAM speculation as Tier B default — measured 76/177/67.
2. **Sleeper — TESTED 2026-07-01, near-miss with a clear path:** llama-server **does** support
   `--spec-type draft-mtp` (the MTP context engaged), but the ggml-org Q4_K_M GGUF **lacks the MTP head
   tensors** ("model doesn't contain MTP layers"). "MTP is vLLM-only" is now outdated upstream. **New
   task (P4, promotes to P2 if cheap):** convert `google/gemma-4-E4B-it` to GGUF *with MTP layers*,
   quantize Q4_K_M, publish as `DeepBlueDynamics/sailfish-e4b-gguf-mtp`, retest — challenge-grade
   drafting on 12 GB cards if it works.
3. **P3-P4: SeqKD tool-drafter.** Distill the target's own tool-call behavior (`distill_e4b.mjs` corpus)
   into (a) better n-gram tables (corpus-primed, if server support for preloaded caches lands) and
   (b) a trained drafter for Tier A.
4. **P4: Tier A drafter** — our MTP drafter (challenge technique, SeqKD objective) against our published
   INT4 target.
5. **v2 unlock:** retrain the drafter against the *fine-tuned* target → tool-sharp AND fast on one card.

## 8. Build phases

**Commit cadence: small and frequent, straight to `main` (new repo, no ceremony).**

### P0 — Site live first (per Kord)
- [ ] Landing page (nuts.services style) + hello-world Cloud Run service (no GPU yet is fine)
- [ ] DNS: `sailfish.nuts.services` domain-mapping
- [ ] `install.ps1` + `install.sh` served from the site (they can point at the image as soon as P1 pushes it)
- [ ] Reuse: shivvr `deploy.sh` shape, grubcrawler site scaffold + Docker Hub pipeline

### P1 — Local appliance MVP
- [ ] Gateway (UI shell + `/v1` proxy + `/api/status`) — one port, 22343
- [ ] GPU autodetect → tier choice (Tier B path only at first)
- [ ] Tier B engine: llama.cpp + stock E4B Q4_K_M + `ngram-mod` (already measured/locked)
- [ ] Dockerfile (single image), publish `deepbluedynamics/sailfish` via grubcrawler pipeline
- [ ] Harness passes against the container out-of-the-box (6/6, ~76 TPS on the 3060)

### P2 — Data + curation
- [ ] nemesis8 probe (18042) + graceful absent-state; built-in Claude Code scraper as fallback source
- [ ] Ingest + size warnings; corpus stats view
- [ ] Curation agent: provider adapter (Anthropic/OpenAI/Gemini/GLM), token via env, cost cap + estimate,
      map-reduce over corpus, **review UI**, output in exact serving format
- [ ] `finetune_target.py` serving-format fix (tool schemas in system prompt, gemma tool-call syntax)

### P3 — Training
- [ ] BYO Google Cloud: zip download + templated script generator (org/project/region), ephemeral spot
      A100, teardown-in-trap, cost header — **dogfood it ourselves end-to-end**
- [ ] The dogfood run → **company shared model v1** (tool-tuned on DBD tools: hyperia, nemesis8,
      grubcrawler, shivvr)
- [ ] Deploy view: download artifact → GGUF convert → hot-swap → Tier B default becomes the fine-tune
- [ ] Hosted trainer (Cloud Run Job + L4) for logged-in users
- [ ] Hosted service swaps stock → `sailfish-e4b-toolft`

### P4 — Tier A (the fast tier)
- [ ] Build + publish `DeepBlueDynamics/sailfish-e4b-int4` (llm-compressor W4A16, towers bf16) on HF
- [ ] SeqKD drafter → `DeepBlueDynamics/sailfish-drafter-e4b`
- [ ] vLLM path in the appliance; tier autodetect goes live for ≥16 GB
- [ ] Arch validation: Ampere ✓ (challenge), Ada/L4 (+FP8), **Blackwell on the 32 GB 5090** (wheel check)

### P5 — Hardening & v2
- [ ] Token security for curator keys; per-user hosted fine-tunes; dedicated L4 if hot
- [ ] Drafter-on-fine-tuned-target (the v2 unlock)
- [ ] draft-mtp / corpus-primed ngram upgrades as upstream allows

## 9. Division of labor (current reality)
- **Everything above: us.** No other agents are running right now (Tyrannosaurus is down, Hyperia team
  is busy) — the two specs in the root are self-serve docs for whoever picks them up later. Nothing
  blocks on them; the fallback scraper covers data until nemesis8's API exists.

## 10. Decisions log (settled 2026-07-01, Kord)
1. **Launch with the STOCK model everywhere** (Tier B + hosted). Our shared fine-tuned E4B comes later
   (P3) and then swaps in. One shared model for all hosted users at launch — per-user models later.
2. **Tier A (≥16 GB): yes** — we build and publish our own INT4 E4B under **DeepBlueDynamics on HF**.
3. **Models are stored on Hugging Face** (DeepBlueDynamics org), not GCS.
4. **Installer:** PowerShell one-liner **and** Mac/Linux shell script. **Curation:** users bring a
   frontier-model token into the container (not hardened yet — accepted); a hosted-curation path for
   logged-in users may come later ("curate it ourselves using our own wiring harness"). **Cost cap: yes.**
   Training requires running the container (no container → hosted-only usage, no training).
5. **nemesis8 port 18042: wire it up as-is**; fix later if wrong.
6. **No agent coordination right now** — Tyrannosaurus down; build solo; specs live in repo root
   (`NEMESIS8_INTEGRATION.md`, `HYPERIA_INTEGRATION.md`) so Kord knows where they are.
7. **Ship the site first** (P0), including DNS `sailfish.nuts.services` and the installer.
8. **Docker Hub via grubcrawler's existing pipeline.** ✔
9. **BYO-cloud shape:** we hand the user a **zip of curated training data + a templated script**; they
   upload/run in their own gcloud shell; we never hold credentials; cost warning included. We dogfood
   this path to produce the company model.
10. **No hosted credits burn:** we don't run per-user hosted models people didn't ask for; hosted =
    one shared small model on scale-to-zero L4; dedicated instance only if real usage demands it.

## 11. Risks & honest unknowns
- **draft-mtp sleeper**: RESOLVED to "needs an MTP-bearing GGUF" (llama.cpp support confirmed; ggml-org
  file lacks the head). Upside intact; conversion task added to §7. Plan doesn't depend on it.
- **Our INT4 build (P4)**: llm-compressor on gemma-4 with towers-in-bf16 is believed-right but unbuilt;
  the community's broken quants show the pitfall. Budget real time.
- **Blackwell wheels**: 5090 (sm_120) needs recent vLLM; validate before promising Tier A there.
- **Cloud Run cold start**: L4 + baked model still means ~1-2 min cold spins; acceptable at launch,
  monitor; mitigation is min-instances=1 (costs) or a warming ping.
- **Fine-tune transfer**: the serving-format fix (P2) is load-bearing — a mis-formatted fine-tune will
  bench well and fail in real harnesses.
- **Curation cost surprises**: cap + estimate mandatory before any tokens leave the box.

## 12. Success criteria
- P0: `sailfish.nuts.services` resolves, styled page, installer downloads.
- P1: fresh Windows box + 3060 → one-liner → clicking `localhost:22343` → chatting with a model at
  ≥70 TPS on tool runs, 6/6 harness accuracy.
- P3: our company model measurably beats stock on DBD-tool harness accuracy; swap is one click.
- P4: a 16 GB+ card runs stock+drafter meaningfully faster than Tier B (target: ≥1.5× on mixed agentic
  output); 5090 validated.
- Hosted: login → chat works; scale-to-zero bill ≈ $0 idle.
