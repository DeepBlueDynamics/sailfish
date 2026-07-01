# Sailfish ↔ nemesis8 — training-data API (spec for the nemesis8 build agent)

Sailfish should NOT parse agent logs itself. nemesis8 already owns the logs and runs a **host-level
controller**; it should expose a small local API that Sailfish (running in a container) calls to
discover and export tool-call training data. This is the function to build.

## Where it lives
On the **nemesis8 controller** (the process that starts with nemesis8, bound at host level). Sailfish
reaches it from inside its container at `http://host.docker.internal:18042` — **port 18042 is the wired
default** (best candidate from source recon, per Kord: wire it up this way; adjust later if wrong; env
override `SAILFISH_N8_URL`). Localhost-only; optional bearer token (`SAILFISH_N8_TOKEN`) since it's
local, but keep the door for it.

## Second responsibility: container lifecycle
Beyond data export, nemesis8 is the component that **installs/starts the Sailfish appliance** when asked
(e.g. by Hyperia's detection ladder, see `HYPERIA_INTEGRATION.md`): `docker pull deepbluedynamics/sailfish`
+ `docker run -p 22343:22343 --gpus all ...`. Expose that as a controller action.

## Export packaging
Exports must also be downloadable as a **zip** (`GET /v1/training/export?...&format=zip`) — the BYO-cloud
training path hands the user a zip + a templated script; they upload it to their own Google Cloud account
themselves. Sailfish never holds their credentials.

## Endpoints

### `GET /v1/training/sources`
List agents/tools that have tool-call data, with counts so the UI can show what's trainable.
```json
{ "sources": [
  { "id": "antigravity", "label": "Antigravity", "tool_calls": 8123, "sessions": 41, "last": "2026-07-01T..." },
  { "id": "opencode",    "label": "OpenCode",    "tool_calls": 2044, "sessions": 12, "last": "..." },
  { "id": "claude-code", "label": "Claude Code", "tool_calls": 25479, "sessions": 346, "last": "..." }
]}
```

### `GET /v1/training/export?source=<id>&format=jsonl&clean=1`
Stream the tool-call traces for a source as JSONL — **one record per tool call**, already scrubbed of
ANSI/control chars (see `scrape/analyze_ctrl.mjs` for what to strip). Record shape:
```json
{ "context": [ {"role":"user|assistant","text":"..."} ],   // last N messages, control-clean
  "tool": "Bash", "arguments": { "command": "...", "description": "..." },
  "result_preview": "…first 400 chars, ANSI-stripped…",
  "source": "antigravity", "ts": "..." }
```
Sailfish then runs these `context`s through the local target (SeqKD self-distill) to make the actual
training set — so nemesis8 only needs to supply **clean traces**, not model outputs.

### `GET /v1/training/stats?source=<id>` (optional, nice-to-have)
Tool-name histogram + arg-key frequencies (for the UI preview + minimum-data warnings).

## Notes for the builder
- **Clean at the source:** strip CSI/OSC escapes, CR, NUL, C0. Tool *arguments* are already clean;
  the mess is in terminal *results* (Bash/PowerShell/terminal_screen). See `scrape/analyze_ctrl.mjs`.
- **Dedup/window:** cap `context` to the last ~6 text messages; drop empty turns.
- **Streaming:** export can be large (25k+ rows) — stream JSONL, don't buffer.
- **Sources = whatever nemesis8 supports:** antigravity, opencode, claude-code, and future agents.
  The list is dynamic; Sailfish just renders whatever `/sources` returns.

## Fallback (until the API exists)
Sailfish ships `scrape/scrape_toolcalls.mjs` (Claude Code transcripts) as a **parser fallback**, so the
appliance works before nemesis8 exposes the API. The API is the preferred path; the parser is the bridge.
