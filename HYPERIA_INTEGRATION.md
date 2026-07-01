# Sailfish ↔ Hyperia — integration spec

> **Audience:** the agent working on Hyperia (`Code/DeepBlueDynamics/hyperia`). Sailfish exposes the
> endpoints; Hyperia implements the client side described here. Sailfish's team does not modify Hyperia.

## What Hyperia gets out of this
A fast, local (or hosted) OpenAI-compatible model endpoint specialized for **tool-calling** — for
completions, tool-run assistance, and any place Hyperia wants low-latency private inference instead of a
frontier API. One client codepath works against all three deployment shapes.

## The detection ladder (implement in this order)

```
1. LOCAL:   GET http://localhost:22343/api/status
            → 200 = Sailfish appliance is running. Use http://localhost:22343/v1 (OpenAI API).
2. INSTALL: not running → ask nemesis8's controller (http://localhost:18042) to offer
            install/start of the Sailfish container (docker pull deepbluedynamics/sailfish + run).
            nemesis8 owns container lifecycle; Hyperia just requests it.
3. HOSTED:  user declines containers → https://sailfish.nuts.services/v1 with nuts-auth login
            (same JWKS flow as other nuts.services; the user logs in, Hyperia attaches the bearer token).
```

## Endpoint contract (identical local & hosted)

- `GET /api/status` → `{ "service":"sailfish", "tier":"A|B", "engine":"llama.cpp|vllm",
  "model":"<id>", "drafter":"ngram-mod|mtp|none", "gpu":{"name":"...","vram_gb":12,"arch":"sm_86"},
  "tps_recent": 76.0, "version":"..." }`
- `POST /v1/chat/completions` — standard OpenAI chat completions **with `tools` support** (function
  calling). Greedy-friendly; supports `stream`.
- `GET /v1/models` — lists the served model id.
- Local is unauthenticated (localhost-only). Hosted requires `Authorization: Bearer <nuts-auth JWT>`.

## Usage notes for Hyperia
- **Model id:** read it from `/v1/models` — do not hardcode (stock vs fine-tuned swaps change it).
- **Tool-calling:** pass Hyperia's tool schemas in the standard OpenAI `tools` array. Measured 6/6
  tool-selection accuracy on the local stack; treat it as a capable-but-4B model: keep tool descriptions
  crisp, prefer ≤ ~20 tools per call.
- **Latency shape:** first request after idle may be slow (model load locally; cold-start on hosted
  scale-to-zero). Show a "warming up" state on the first call rather than timing out (allow 120 s).
- **Speed expectations (local, RTX 3060 12 GB, measured):** ~76 TPS on tool runs, up to ~177 TPS in long
  repetitive/agentic sessions (n-gram speculation feeds on repeated context). Bigger cards run Tier A
  (stock model + trained drafter) and go faster.
- **Health/again:** if `/api/status` starts failing mid-session, fall back down the ladder gracefully
  (local → hosted) rather than erroring the user's flow.

## Nice-to-have (later, optional)
- Hyperia settings pane: "Sailfish endpoint" (auto / local / hosted URL override).
- Surface Sailfish's `tier` + `tps_recent` in Hyperia's status bar so users see what they're running on.
- A "train on my tool runs" deep-link that opens `http://localhost:22343/` (the appliance UI) on the
  Data tab.
