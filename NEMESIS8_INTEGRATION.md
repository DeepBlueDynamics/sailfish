# Sailfish ↔ nemesis8 — data + training spec (handoff)

> **Audience:** the agent building the **nemesis8 controller**. **PART 1 is what you build** — a small
> local API that lets Sailfish discover and extract tool-run training data. PARTS 2–3 are *context*:
> how Sailfish consumes that API (local training config) and the hosted training service we're sketching,
> so you understand what the data feeds. Build Part 1 to the shapes below; Parts 2–3 are ours.
>
> Sailfish must NOT parse agent logs itself in production — nemesis8 owns the logs and the host-level
> controller. Until this API exists, Sailfish falls back to its built-in scraper (see §1.7).

---

## 0. Concepts (the vocabulary this spec uses)

- **tool-bag** *(a.k.a. provider)* — a named source/namespace of tools. Examples: `hyperia` (its
  `mcp__hyperia__*` surface), `claude-code` (built-in Bash/Edit/Read/…), `antigravity`, `opencode`.
  One agent/host = one tool-bag (or a few). This is the "who exposed the tool."
- **tool** — an individual tool within a bag: `terminal_run`, `Bash`, `Edit`, `web_open`.
- **tool-run** — one actual invocation: the lead-up **context**, the **tool** called, its **arguments**,
  and a **result preview**. This is the atom of training data.

The workflow Sailfish drives against this API:
**search** tools that match what we want to train on → that **defines the provider(s)/tools** → **extract**
the tool-runs for those (provider, tool) pairs → curate → train.

---

## 1. PART 1 — the nemesis8 API  *(BUILD THIS)*

Bound at host level (starts with nemesis8). Sailfish reaches it from inside its container at
`http://host.docker.internal:18042` — **port 18042 is the wired default** (env override `SAILFISH_N8_URL`).
Localhost-only; optional bearer token (`SAILFISH_N8_TOKEN`) — keep the door for it.

### 1.1 `GET /v1/providers` — which tool-bags have tool-run data
"Which providers have been used for tool runs," with counts so the UI/config can pick.
```json
{ "providers": [
  { "id": "hyperia",     "label": "Hyperia",     "tool_runs": 9137,  "tools": 121, "last": "2026-07-02T18:00:00Z" },
  { "id": "claude-code", "label": "Claude Code", "tool_runs": 26708, "tools": 85,  "last": "2026-07-02T17:40:00Z" },
  { "id": "antigravity", "label": "Antigravity", "tool_runs": 812,   "tools": 40,  "last": "2026-06-30T..." }
]}
```

### 1.2 `GET /v1/tools/search` — find tools matching a request, across all logs
The discovery step. All params optional; combine them.
| param | meaning |
|---|---|
| `q` | free-text over tool **name** + description (fuzzy/substring is fine) |
| `provider` | restrict to one tool-bag (e.g. `hyperia`) |
| `tool` | exact tool name (e.g. `terminal_run`) |
| `limit` | cap results (default 100) |
```
GET /v1/tools/search?q=terminal&provider=hyperia
```
```json
{ "query": { "q": "terminal", "provider": "hyperia" },
  "matches": [
    { "tool": "terminal_run",    "provider": "hyperia", "runs": 589, "description": "Run a command in a pane" },
    { "tool": "terminal_screen", "provider": "hyperia", "runs": 704, "description": "Capture a pane" }
  ]}
```
This is what "define the provider" means: the caller searches for the tools they want, and the response
tells them which tool-bag(s) + tools to then extract.

### 1.3 `GET /v1/tool-runs` — extract the runs (the training records)
The extraction step. Filter by provider/tool, stream the runs. **This is the training data.**
| param | meaning |
|---|---|
| `provider` | repeatable — one or more tool-bags |
| `tool` | repeatable — one or more tool names (glob ok, e.g. `terminal_*`) |
| `since` | ISO timestamp — only runs after this |
| `limit` | cap total records |
| `format` | `jsonl` (stream, default) or `zip` (downloadable — for the BYO-cloud path) |
| `clean` | `1` = ANSI/control-scrubbed (default on) |
```
GET /v1/tool-runs?provider=hyperia&tool=terminal_run&tool=terminal_screen&format=jsonl
```
Streams one record per line (already scrubbed — see §1.6):
```json
{ "provider": "hyperia", "tool": "terminal_run",
  "arguments": { "command": "ls -la", "pane": 2 },
  "context": [ {"role":"user","text":"list the files"} , {"role":"assistant","text":"…"} ],
  "result_preview": "…first 400 chars, ANSI-stripped…",
  "ts": "2026-07-02T18:00:00Z" }
```
Record shape is stable across providers — Sailfish's trainer reads exactly `{context, tool, arguments}`
(result_preview optional). Sailfish runs these `context`s through the local target for the training set,
so **you only supply clean traces, not model outputs.**

### 1.4 `GET /v1/stats` — histograms (optional but wanted)
`?provider=` optional. Tool-name histogram + arg-key frequencies, for UI preview + minimum-data warnings.

### 1.5 Container lifecycle (second responsibility)
Beyond data, nemesis8 **installs/starts the Sailfish appliance** on request (used by Hyperia's ladder,
see `HYPERIA_INTEGRATION.md`): `docker pull deepbluedynamics/sailfish` + `docker run -p 22343:22343
--gpus all -v ~/.claude:/root/.claude:ro …`. Expose that as a controller action.

### 1.6 Cleaning / streaming / dedup (builder notes)
- **Clean at the source:** strip CSI/OSC escapes, CR, NUL, C0 from tool *results* (Bash/PowerShell/
  terminal_screen are the mess; arguments are usually clean). See `scrape/analyze_ctrl.mjs`.
- **Window/dedup:** cap `context` to the last ~6 text messages; drop empty turns; de-dup identical runs.
- **Stream:** exports can be 25k+ rows — stream JSONL, don't buffer; support `zip` for download.
- **Providers are dynamic:** whatever nemesis8 supports (hyperia, claude-code, antigravity, opencode,
  future agents). Sailfish renders whatever `/providers` returns.

### 1.7 Fallback (until this API exists)
Sailfish ships `scrape/scrape_toolcalls.py` (Claude Code transcripts) as the bridge — it already produces
the §1.3 record shape (proven: 26,708 runs / 85 tools / 0 parse errors on a real box). The nemesis8 API
is the **preferred** path (more providers, incl. Hyperia's own sessions); the scraper is the stopgap.

---

## 2. PART 2 — how Sailfish consumes it (local training config)  *(ours; here for context)*

Sailfish keeps a local config selecting **which agent's / tool-bag's tool-runs become training prompts**.
Lives at `train/training.toml` (appliance-editable via the Curate/Train UI).

```toml
# which tool-runs become training data
[source]
providers  = ["hyperia", "claude-code"]   # tool-bags from GET /v1/providers
tools      = ["*"]                          # or ["terminal_run","Bash","Edit"] — globs ok
min_runs   = 5                              # drop rare/noisy tools
max_examples = 20000
since       = "2026-01-01T00:00:00Z"        # optional recency window

[curate]
provider     = "anthropic"                  # frontier model that filters runs → clean prompts
cost_cap_usd = 5                             # estimate-first, hard-capped

[train]
base        = "google/gemma-4-E4B-it"
epochs      = 1
menu_size   = 14                            # tools shown per example (the Doors sweet spot)
output      = "hf://you/gemma4-e4b-toolft"  # or gs://bucket/gemma4-e4b-toolft.gguf

[serve]
relaunch    = true                          # hot-swap the appliance to the new model when done
```

**The loop:** `GET /v1/tools/search` (pick) → `GET /v1/tool-runs?provider=…&tool=…` (extract) → curate
(cost-capped) → render to serving format (`train/finetune_target.py`) → train → publish → (if
`relaunch`) restart the appliance pointed at the new model via `SAILFISH_MODEL_URL`.

### 2.1 Local training on a big card (5090)
If the box has a ≥24 GB card (5090 32 GB is ideal), train **locally**: build the dataset from the config,
run the LoRA, merge → GGUF, then **relaunch the container** at the new model (`SAILFISH_MODEL_PATH` for a
baked local file, or `SAILFISH_MODEL_URL` for one it fetched). Auto-restart the engine so the sharper
model is serving with no manual step. (A 3060 can serve but not comfortably train E4B — big-card only.)

---

## 3. PART 3 — hosted training service (SKETCH, `nuts.services`)  *(design, not built)*

For users who won't run their own trainer: a paid nuts.services account trains on **our** A100 box.

**Flow:**
1. **Log in** (nuts-auth JWT, same as the rest of the fleet).
2. **Permission check** — is this a **paid** account? Read a `plan`/entitlement claim (or a billing
   lookup). **Free users are rejected here.** *(Payment gateway enforcement is a later session — for now
   this is a stub gate: an allow-list / a `paid` claim.)*
3. **Enqueue a training job** — the user's `training.toml` + their extracted/curated dataset (from
   nemesis8 via §1, or an uploaded bundle) go into a queue.
4. **Provision** — fire up the **A100 cloud box** (GCP a2, like the BYO-cloud path but *we* own it and
   the credentials). Run **in batches** where possible — drain the queue on one warm box to amortize
   spin-up, tear down when idle.
5. **Train** — same trainer as local (`finetune_target.py`), producing a merged model + GGUF.
6. **Publish** — push the artifact to the user's HF repo or a GCS object they can pull (the appliance's
   `SAILFISH_MODEL_URL` then serves it).
7. **Notify** — **agent-mail** the user when it's ready (job done / failed / link to the model). No
   polling required on their end.

**Sketch endpoints (Sailfish/site side, all logged-in + paid-gated):**
| endpoint | does |
|---|---|
| `POST /api/train/hosted` | submit a job `{training_toml, dataset_ref}`; 402 if not paid; returns `job_id` |
| `GET  /api/train/jobs/:id` | status: queued / provisioning / training / publishing / done / failed |
| `POST /api/train/hosted/estimate` | cost/time estimate before submit (free) |

**Guards & niceties:**
- **No free training:** the paid gate in step 2 is load-bearing — never provision an A100 for an
  unentitled user. Rate-limit per account.
- **Batch economics:** queue + batch on a warm A100; cap concurrent boxes; `--max-run-duration` backstop
  on every VM (same safety net as the BYO-cloud script).
- **Notify via agent-mail** on done/failed; include the model link + a one-line "how to serve it"
  (the `SAILFISH_MODEL_URL` docker run).
- **Data residency:** hosted training uses data the user shipped us (their curated bundle); we don't
  reach into their box. Local training keeps everything on their machine.

---

## 4. Division of labor
- **nemesis8 agent builds:** Part 1 (`/v1/providers`, `/v1/tools/search`, `/v1/tool-runs`, `/v1/stats`)
  + the container-lifecycle action. To the shapes above.
- **Sailfish builds:** Part 2 (local training config + loop + 5090 local train/relaunch) and Part 3 (the
  hosted service). Nothing in Sailfish blocks on nemesis8 — the fallback scraper (§1.7) covers data until
  the API lands.
