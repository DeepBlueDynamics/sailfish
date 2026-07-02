# Hyperia → Sailfish: the tool-call loop (complete, verified worked example)

> Companion to `HYPERIA_INTEGRATION.md` (endpoints, detection ladder, Electron auth). This doc is the
> concrete "how the agent sends tools and reads back calls." **Every payload below is copied from the
> running container** (`deepbluedynamics/sailfish` on an RTX 3060, `gemma4-e4b` = gemma-4-E4B-it Q4_K_M),
> not invented — verified 2026-07-02.

## TL;DR
Sailfish is an OpenAI-compatible endpoint. Hyperia POSTs `messages` + `tools` to
`http://localhost:22343/v1/chat/completions`; the model replies with `tool_calls`; Hyperia executes them
and appends `role:"tool"` results; repeat until `finish_reason:"stop"`. Three things bite people — all
confirmed against the live engine:

1. **`tool_calls[].function.arguments` is a JSON *string*, not an object** — you must `JSON.parse()` it.
2. **`tool_calls[].id` is a bare token** (e.g. `3JPe108v7sZaDgmDpB8Q74oNpDiw38F8`, no `call_` prefix) —
   echo it back verbatim in the matching `role:"tool"` message's `tool_call_id`.
3. **gemma-4 emits `message.reasoning_content`** (its thinking) next to the tool call — display it or
   drop it, but know it's there and it is *not* the tool call.

## The loop (algorithm)
```
messages = [ {role:"user", content: <task>} ]
loop (bounded, e.g. 8 turns):
  resp = POST /v1/chat/completions { model, messages, tools, temperature:0, stream:false }
  msg  = resp.choices[0].message
  if resp.choices[0].finish_reason != "tool_calls":   return msg.content     # final answer
  messages.push(msg)                                   # push assistant turn VERBATIM (keeps tool_calls+ids)
  for tc in msg.tool_calls:
      args   = JSON.parse(tc.function.arguments)        # <-- string!
      result = executeYourTool(tc.function.name, args)  # guard unknown names
      messages.push({role:"tool", tool_call_id: tc.id, content: String(result)})
```

## Verified wire format

### Turn 1 — request (Hyperia → Sailfish)
```json
{ "model":"gemma4-e4b", "temperature":0, "stream":false,
  "messages":[{"role":"user","content":"List the files in the current directory, then tell me how many there are."}],
  "tools":[
    {"type":"function","function":{"name":"run_shell","description":"Run a shell command and return its stdout.",
      "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a file's contents.",
      "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
  ]}
```

### Turn 1 — response (Sailfish → Hyperia)  *(trimmed)*
```json
{ "choices":[{ "finish_reason":"tool_calls", "index":0, "message":{
    "role":"assistant", "content":"",
    "reasoning_content":"The user wants to … I will start by calling run_shell with ls.",
    "tool_calls":[{ "type":"function",
      "function":{"name":"run_shell","arguments":"{\"command\":\"ls\"}"},
      "id":"3JPe108v7sZaDgmDpB8Q74oNpDiw38F8" }] }}],
  "usage":{"completion_tokens":145,"prompt_tokens":129,"total_tokens":274},
  "timings":{"predicted_per_second":71.4,"prompt_per_second":279.7} }
```
Note: `content` is empty on a tool turn; the plan is in `reasoning_content`; `arguments` is a string.

### Turn 2 — request: append the assistant turn + the tool result
```json
{ "model":"gemma4-e4b","temperature":0,"stream":false,
  "messages":[
    {"role":"user","content":"List the files in the current directory, then tell me how many there are."},
    {"role":"assistant","content":"","tool_calls":[{"type":"function","id":"3JPe108v7sZaDgmDpB8Q74oNpDiw38F8",
      "function":{"name":"run_shell","arguments":"{\"command\":\"ls\"}"}}]},
    {"role":"tool","tool_call_id":"3JPe108v7sZaDgmDpB8Q74oNpDiw38F8","content":"app.py\nauth.py\nconfig.py\ncurate.py\ndata.py\ngpu.py\nmain.py\ntrain.py"}
  ],
  "tools":[ … same tools … ] }
```

### Turn 2 — response: final answer
```json
{ "choices":[{ "finish_reason":"stop", "index":0, "message":{
    "role":"assistant",
    "content":"The files in the current directory are:\n\n* app.py\n* auth.py … \n\nThere are **8** files in total.",
    "reasoning_content":"… I have already executed ls … There are 8 files." }}],
  "usage":{"completion_tokens":237,"prompt_tokens":181},
  "timings":{"predicted_per_second":70.8,"draft_n":64,"draft_n_accepted":8} }
```
`finish_reason:"stop"` + populated `content` = you're done. `draft_n/draft_n_accepted` = the n-gram
drafter working (acceptance climbs on repeated/structured output — that's the 5–12× in `harness/PROFILE_RESULTS.md`).

## Reference client (TypeScript — Hyperia is Electron/Node)
```ts
const BASE = "http://localhost:22343/v1";   // local appliance. hosted: https://sailfish.nuts.services/v1 + bearer
let MODEL = "gemma4-e4b";                    // better: fetch once from GET /v1/models (don't hardcode; it swaps)

// Hyperia's tools are MCP tools → OpenAI is a mechanical map (MCP inputSchema IS JSON Schema):
const toOpenAI = (t: {name:string; description?:string; inputSchema?:any}) => ({
  type: "function",
  function: { name: t.name, description: t.description ?? "",
              parameters: t.inputSchema ?? { type: "object", properties: {} } },
});

async function chat(messages: any[], tools: any[], token?: string) {
  const r = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ model: MODEL, messages, tools, temperature: 0, stream: false }),
  });
  if (r.status === 401) throw new Error("auth");        // hosted only → re-prompt login
  if (!r.ok) throw new Error(`sailfish ${r.status}`);
  return r.json();
}

// exec(name,args) runs the actual MCP/Hyperia tool and returns a string result
async function runAgent(task: string, tools: any[], exec: (n:string,a:any)=>Promise<string>, maxTurns = 8) {
  const oai = tools.map(toOpenAI);
  const names = new Set(tools.map(t => t.name));
  const messages: any[] = [{ role: "user", content: task }];
  for (let i = 0; i < maxTurns; i++) {
    const j = await chat(messages, oai);
    const choice = j.choices[0], msg = choice.message;
    if (choice.finish_reason !== "tool_calls" || !msg.tool_calls?.length) return msg.content;  // done
    messages.push(msg);                                  // push assistant VERBATIM (tool_calls + ids intact)
    for (const tc of msg.tool_calls) {
      let args: any = {};
      try { args = JSON.parse(tc.function.arguments); }  // arguments is a STRING
      catch { args = {}; }                               // malformed → decide: retry / error result
      const result = names.has(tc.function.name)
        ? await exec(tc.function.name, args)
        : `ERROR: unknown tool "${tc.function.name}"`;    // guard hallucinated names
      messages.push({ role: "tool", tool_call_id: tc.id, content: String(result) });  // echo tc.id
    }
  }
  return "(max turns reached)";
}
```

## gemma-4 E4B specifics (this is a 4B-class model — respect its limits)
- **Cap the menu.** Keep tools per request to **≤ ~20** with crisp descriptions. Hyperia exposes 100+ MCP
  tools; do **not** dump them all — a 4B model's selection accuracy degrades with a huge menu. Subset by
  relevance per task (your existing tool-search/gating is the right upstream step). With a tight menu we
  measured **6/6** tool-selection accuracy (`harness/tool_harness.ps1`).
- **`temperature: 0`** for tool-decision turns → deterministic, reliable selection.
- **`reasoning_content`** is emitted; if Hyperia shows a "thinking" affordance, wire it there, else ignore.
- **Parallel tool calls:** the model *may* return several `tool_calls` in one turn — execute all, append one
  `role:"tool"` per `tool_call_id`, then continue. (Our example returned one; handle N.)
- **System prompt:** gemma's template folds a `system` role into the first user turn — you can send a
  `{role:"system",...}` and it works, but keep it short; the tool schemas already carry most structure.
- **Multimodal:** `/v1/models` reports `capabilities:["completion","multimodal"]` — image inputs are
  possible later, but Tier B is tuned for text/tool traffic; don't rely on vision yet.

## Recommended: gate tools behind doors (gather + expand as entered)
The ≤~20-tools rule isn't a limitation to grudgingly accept — it's a design win if you **gate the tools
behind doors** instead of flattening the whole catalog at the model. Present a small menu; expand it only
as the agent walks through a door. Hyperia already has the machinery for this — `ToolSearch` is its own
9th-most-used tool — so this is a wiring choice, not new infrastructure.

**The pattern (progressive disclosure / faceted tool routing):**
1. **Level-0 menu = core + doors.** Every turn starts with a handful of always-on tools (the ones used
   constantly — `run_shell`, `read_file`, …) **plus a few "door" tools**: coarse category openers
   (`open_terminal_tools`, `open_web_tools`, `open_task_tools`, …) and/or a single `search_tools` door
   that takes a natural-language query. Doors are cheap: name + one-line description of what's behind them.
2. **Gather on entry.** When the model calls a door — `open_web_tools` or `search_tools("scrape a page")`
   — Hyperia *gathers* that door's tool schemas (by namespace, or by relevance from `ToolSearch`) and
   returns a tool result naming what's now available.
3. **Expand as entered.** On the **next** request, inject those gathered schemas into the `tools` array so
   the model can actually call them. The live menu grows only along the path the agent is walking.
4. **Keep the live set bounded (≤~20).** Collapse doors the agent has left (drop their schemas back to
   name-only), so depth is unlimited but breadth at any step stays inside the 4B model's sweet spot.

**Why it's the right move here:**
- **Accuracy:** a 4B model picks correctly from 20 relevant tools, not 100 mixed ones (we measured 6/6 on
  a tight menu). Doors keep every decision small.
- **Context:** Tier B is 8k tokens. 100 full tool schemas can eat thousands of tokens before the task even
  starts; doors cost ~a line each until opened.
- **Scale:** total tool count becomes irrelevant — you can expose thousands behind doors and the model
  never sees more than a room at a time.
- **It mirrors how Hyperia already works** (ToolSearch = a door; deferred tools = tools behind doors whose
  schemas load on demand). This just makes that the *front door* of the Sailfish tool loop.

Mechanically it's the same loop above — a door is just a tool whose "result" is "here's what's now
open," and the expansion happens by editing the `tools` array you send on the next turn.

## Performance you can expect (measured, RTX 3060)
- **First call after idle:** slow (model/graph warm) — allow ~120 s, show "warming up," don't time out.
- **Tool/agentic traffic:** ~74 tok/s steady on tool runs; **5–12× faster than stock Ollama on repeated /
  structured output** (JSON, tables, repeated edits) because n-gram speculation drafts predictable tokens
  for free. The longer and more repetitive an agent session, the more the drafter pays off. Full numbers:
  `harness/PROFILE_RESULTS.md`.
- **Idle:** the appliance holds ~6 GB VRAM while loaded. `tps_recent` on `/api/status` reflects live speed.

## Failure handling (fall down the ladder, never hard-error the user)
- **Unknown tool name** in a `tool_call` → return an `ERROR: unknown tool …` tool result; the model
  recovers or you cap turns.
- **Malformed `arguments` JSON** → guard the `JSON.parse`; return an error tool result rather than crashing.
- **Engine warming / 5xx** → retry with backoff (first-call warmup) before falling back.
- **`/api/status` starts failing mid-session** → drop to the next ladder tier (local → hosted) gracefully.
- **401 (hosted only)** → clear the stored token, re-prompt login (see `HYPERIA_INTEGRATION.md`).
