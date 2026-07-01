#!/usr/bin/env node
/**
 * Sailfish :: scrape — harvest real tool-call traces from Claude Code transcripts.
 *
 * Walks ~/.claude/projects (+ the nemesis8 sandbox) and extracts every tool call with its
 * lead-up context, arguments, and result. Emits a clean corpus for training/measuring a
 * tool-specialized speculative drafter.
 *
 *   node scrape/scrape_toolcalls.mjs [--max-context 6] [--include-builtins]
 *
 * Outputs (./data):
 *   tool_calls.jsonl      one record per tool call: {project, ts, tool, arguments, result_preview, context[]}
 *   agentic_prompts.jsonl chat-format prompts (context -> expected tool call) for VSD replay-training
 *   stats.json           counts by tool, by project, arg-key frequencies
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(__dirname, "..", "data");
fs.mkdirSync(OUT, { recursive: true });

const argv = process.argv.slice(2);
const getOpt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const MAX_CTX = parseInt(getOpt("--max-context", "6"), 10);
const CTX_CHARS = 1200;

const ROOTS = [
  path.join(os.homedir(), ".claude", "projects"),
  path.join(os.homedir(), ".nemesis8", "home", ".claude", "projects"),
];

function* walk(dir) {
  let ents = [];
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) yield* walk(p);
    else if (e.name.endsWith(".jsonl")) yield p;
  }
}

function textOf(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.filter(b => b && b.type === "text").map(b => b.text).join("\n");
  return "";
}
const clip = (s, n) => (s && s.length > n ? s.slice(0, n) + "…" : (s || ""));
// strip ANSI/OSC escapes + control chars (terminal gunk in tool results); keep \t \n
const stripCtrl = (s) => (s || "")
  .replace(/\x1b\[[0-9;?]*[ -\/]*[@-~]/g, "")            // CSI (colors, cursor)
  .replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, "")      // OSC (titles, hyperlinks)
  .replace(/[\x1b\r\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, ""); // stray ESC, CR, NUL, other C0

const tcOut = fs.createWriteStream(path.join(OUT, "tool_calls.jsonl"));
const promptOut = fs.createWriteStream(path.join(OUT, "agentic_prompts.jsonl"));

const byTool = {}, byProject = {}, argKeys = {};
let nFiles = 0, nCalls = 0, nLines = 0, nBad = 0;

for (const file of [].concat(...ROOTS.map(r => [...walk(r)]))) {
  nFiles++;
  let lines;
  try { lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean); } catch { continue; }

  // pass 1: parse events + index tool results by tool_use_id
  const events = [];
  const resultById = {};
  for (const line of lines) {
    nLines++;
    let o; try { o = JSON.parse(line); } catch { nBad++; continue; }
    const m = o.message;
    if (!m || !m.content) { events.push({ role: o.type, text: "" }); continue; }
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b && b.type === "tool_result") {
          resultById[b.tool_use_id] = clip(stripCtrl(textOf(b.content) || (typeof b.content === "string" ? b.content : "")), 400);
        }
      }
    }
    events.push({ role: m.role || o.type, content: m.content, project: o.cwd, ts: o.timestamp });
  }

  // pass 2: rolling context -> emit a record at each tool_use
  const ctx = [];
  for (const ev of events) {
    const t = textOf(ev.content);
    if (Array.isArray(ev.content)) {
      for (const b of ev.content) {
        if (b && b.type === "tool_use") {
          const project = ev.project ? path.basename(ev.project) : "unknown";
          const rec = {
            project, ts: ev.ts || null, tool: b.name,
            arguments: b.input || {},
            result_preview: resultById[b.id] || null,
            context: ctx.slice(-MAX_CTX),
          };
          tcOut.write(JSON.stringify(rec) + "\n");
          // chat-format prompt for replay-training (context messages -> the tool the model should call)
          promptOut.write(JSON.stringify({
            messages: ctx.slice(-MAX_CTX).map(c => ({ role: c.role === "assistant" ? "assistant" : "user", content: c.text })),
            expect_tool: b.name, arguments: b.input || {}, project,
          }) + "\n");
          nCalls++;
          byTool[b.name] = (byTool[b.name] || 0) + 1;
          byProject[project] = (byProject[project] || 0) + 1;
          for (const k of Object.keys(b.input || {})) argKeys[`${b.name}.${k}`] = (argKeys[`${b.name}.${k}`] || 0) + 1;
        }
      }
    }
    if (t && t.trim()) ctx.push({ role: ev.role === "assistant" ? "assistant" : "user", text: clip(stripCtrl(t).trim(), CTX_CHARS) });
    if (ctx.length > 40) ctx.splice(0, ctx.length - 40);
  }
}
tcOut.end(); promptOut.end();

const top = (obj, n) => Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, n);
const stats = {
  generated: new Date().toISOString(),
  files: nFiles, lines: nLines, parse_errors: nBad, tool_calls: nCalls,
  unique_tools: Object.keys(byTool).length,
  top_tools: top(byTool, 25), top_projects: top(byProject, 15), top_arg_keys: top(argKeys, 25),
};
fs.writeFileSync(path.join(OUT, "stats.json"), JSON.stringify(stats, null, 2));

console.log(`\nSailfish scrape complete:`);
console.log(`  files=${nFiles}  lines=${nLines}  parse_errors=${nBad}`);
console.log(`  tool_calls=${nCalls}  unique_tools=${stats.unique_tools}`);
console.log(`  -> data/tool_calls.jsonl, data/agentic_prompts.jsonl, data/stats.json`);
console.log(`  top tools:`, top(byTool, 8).map(([k, v]) => `${k}:${v}`).join("  "));
