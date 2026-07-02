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
            (the user logs in from the Electron app, Hyperia attaches the bearer token).
            → see "Authentication from the Electron app" below.
```

**Login is a hosted-tier concern only.** Tiers 1–2 (local appliance) are unauthenticated
(localhost-only) — do **not** prompt for login while a local endpoint works. Only prompt when the
ladder falls through to HOSTED. And note the split: **data / curate / train are local-appliance-only**
(they touch the box's filesystem + GPU); the hosted service is **inference (`/v1`) only**. So the
Electron login exists to unlock hosted *inference* when the user has no local card.

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

## Authentication from the Electron app (hosted path)

Hyperia is an Electron desktop app, so there's no browser tab to stash a token in — the login flow has
to be a native one. **Sailfish already validates whatever you send** (its gateway exchanges/validates
both token types against nuts-auth); Hyperia's only job is to *acquire* a token, *store* it securely,
and *attach* it as `Authorization: Bearer <token>` on hosted calls. Two options, in preference order.

### Recommended: paste a long-lived `ahp_` token
This is the nuts.services fleet convention (nuts-site: "get an `ahp_` token from the dashboard, pass it
as a Bearer header"), and it's the right fit for a desktop app because **magic-link JWTs expire in 30
minutes** (`nuts-auth` `ACCESS_TOKEN_EXPIRE_MINUTES=30`) — you'd be re-logging-in constantly. `ahp_`
tokens are long-lived, revocable, and dashboard-managed.

```
Settings → "Connect to nuts.services":
  1. button → shell.openExternal('https://auth.nuts.services/dashboard')   // system browser; user is
                                                                            // likely already logged in
  2. user creates/copies an ahp_… token there
  3. paste field in Hyperia → store with Electron safeStorage (OS keychain), NOT plaintext/localStorage
  4. attach `Authorization: Bearer ahp_…` to hosted /v1 (and /api) requests
```
No redirect plumbing, no protocol handler, survives restarts, one paste. Sailfish's `AuthClient` does the
`ahp_` → JWT exchange server-side; you never call nuts-auth directly.

### Optional: one-click "Log in with browser" (magic-link → JWT)
If you want a no-paste button, use the RFC 8252 native-app loopback pattern. nuts-auth returns the token
as a **query param** (`return_url?token=<JWT>`), so a localhost server can capture it (a URL *fragment*
couldn't — it never reaches the server; nuts-auth uses query, confirmed in `web/routes/auth.py`).

```js
// Electron main process (sketch)
const http = require('http'); const { shell, safeStorage } = require('electron');
const srv = http.createServer((req, res) => {
  const token = new URL(req.url, 'http://127.0.0.1').searchParams.get('token');
  if (token) { store(safeStorage.encryptString(token));           // keychain-backed
               res.end('<h3>Signed in — return to Hyperia.</h3>'); srv.close(); }
});
srv.listen(0, '127.0.0.1', () => {
  const port = srv.address().port;
  const cb = encodeURIComponent(`http://127.0.0.1:${port}/cb`);
  shell.openExternal(`https://auth.nuts.services/login?return_url=${cb}`);
});
// user enters email → clicks the emailed magic link → nuts-auth 302s to 127.0.0.1:<port>/cb?token=JWT
```
Because the JWT is 30-min, cache it and **transparently re-run this on a 401**, or just prefer `ahp_`.
Alternatives to loopback: a custom protocol (`app.setAsDefaultProtocolClient('hyperia')`, `return_url =
hyperia://auth`) or an embedded `BrowserWindow` intercepting the redirect — but the magic-link email
round-trip means the user leaves an embedded window anyway, so external-browser + loopback is cleanest.

### Token handling rules
- **Scope the bearer to the hosted origin.** Send it only to `https://sailfish.nuts.services`; do **not**
  attach it to the `localhost:22343` appliance (unnecessary, and avoids leaking a token to a local process).
- **On 401 from hosted:** clear the stored token and re-prompt (paste again / re-run browser login).
  `ahp_` tokens are revocable from the dashboard — treat any 401 as "reconnect", not a hard error.
- **Never** log the token or write it to disk unencrypted; `safeStorage` (or the OS keychain) only.
- Reuse this exact acquisition + storage code for any other nuts.services the app talks to — it's the
  same bearer everywhere (email-partitioned server-side).

## Nice-to-have (later, optional)
- Hyperia settings pane: "Sailfish endpoint" (auto / local / hosted URL override).
- Surface Sailfish's `tier` + `tps_recent` in Hyperia's status bar so users see what they're running on.
- A "train on my tool runs" deep-link that opens `http://localhost:22343/` (the appliance UI) on the
  Data tab.
