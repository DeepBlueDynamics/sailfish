#!/usr/bin/env node
/**
 * Sailfish :: train — self-distill e4b's own tool-call behavior into a SeqKD corpus.
 *
 * The lesson from the leaderboard drafter: acceptance = "did the drafter emit the TARGET's exact
 * next token." So we don't train on Claude's tokens (wrong tokenizer) or a surrogate objective.
 * We run the LOCAL e4b *greedily* over the scraped tool-call contexts, with the tools defined, and
 * capture e4b's OWN tool-call sequences. That corpus — gemma's own tokens — is exactly what the
 * acceptance objective wants, and it feeds BOTH the n-gram lookup and the SeqKD model drafter.
 *
 *   node train/distill_e4b.mjs [--endpoint http://localhost:22343/v1] [--max 2000] [--top-tools 15]
 *
 * In:  ../data/tool_calls.jsonl   (for inferring tool schemas)
 *      ../data/agentic_prompts.jsonl (the contexts to replay)
 * Out: ../data/gemma_toolcalls.jsonl  {context, tool, arguments, text}  ← the SeqKD corpus
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(__dirname, "..", "data");
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const ENDPOINT = opt("--endpoint", "http://localhost:22343/v1");
const MAX = parseInt(opt("--max", "2000"), 10);
const TOP_TOOLS = parseInt(opt("--top-tools", "15"), 10);
const MODEL = opt("--model", "gemma4-e4b");

const readJsonl = (f) => fs.readFileSync(f, "utf8").split(/\r?\n/).filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);

// 1. infer tool schemas from real usage (name -> arg keys + a value sample for the description)
function buildToolSchemas() {
  const rows = readJsonl(path.join(DATA, "tool_calls.jsonl"));
  const tools = new Map(); // name -> {count, keys:Set}
  for (const r of rows) {
    const t = tools.get(r.tool) || { count: 0, keys: new Set() };
    t.count++;
    for (const k of Object.keys(r.arguments || {})) t.keys.add(k);
    tools.set(r.tool, t);
  }
  const top = [...tools.entries()].sort((a, b) => b[1].count - a[1].count).slice(0, TOP_TOOLS);
  return top.map(([name, t]) => ({
    type: "function",
    function: {
      name,
      description: `Agent tool ${name}`,
      parameters: {
        type: "object",
        properties: Object.fromEntries([...t.keys].map(k => [k, { type: "string" }])),
        required: [],
      },
    },
  }));
}

async function callE4B(messages, tools) {
  const body = { model: MODEL, messages, tools, stream: false, temperature: 0, max_tokens: 256 };
  const res = await fetch(`${ENDPOINT}/chat/completions`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`e4b ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const j = await res.json();
  return j.choices?.[0]?.message || {};
}

const tools = buildToolSchemas();
console.log(`Sailfish distill: e4b @ ${ENDPOINT}, ${tools.length} tools, greedy`);
const contexts = readJsonl(path.join(DATA, "agentic_prompts.jsonl")).slice(0, MAX);
const out = fs.createWriteStream(path.join(DATA, "gemma_toolcalls.jsonl"));

let n = 0, withTool = 0, fails = 0;
for (const c of contexts) {
  const messages = (c.messages || []).filter(m => m.content && m.content.trim()).slice(-6);
  if (!messages.length) continue;
  if (messages[messages.length - 1].role !== "user") messages.push({ role: "user", content: "Continue." });
  try {
    const msg = await callE4B(messages, tools);
    const tc = (msg.tool_calls || [])[0];
    const rec = {
      tool: tc?.function?.name || null,
      arguments: tc?.function?.arguments ? (typeof tc.function.arguments === "string" ? JSON.parse(tc.function.arguments || "{}") : tc.function.arguments) : null,
      text: msg.content || "",
      expect_tool: c.expect_tool || null,
    };
    out.write(JSON.stringify(rec) + "\n");
    if (rec.tool) withTool++;
  } catch (e) { fails++; if (fails <= 3) console.error("  fail:", e.message); }
  if (++n % 100 === 0) process.stdout.write(`\r  ${n}/${contexts.length}  tool-calls=${withTool}  fails=${fails}`);
}
out.end();
console.log(`\nDone: ${n} contexts → data/gemma_toolcalls.jsonl  (${withTool} produced tool calls, ${fails} fails)`);
console.log("Next: node ../drafter/build_ngram_gemma.mjs  (index gemma's own tokens → the lookup drafter)");
