#!/usr/bin/env node
/**
 * Sailfish :: scrape — control-character audit of the harvested corpus.
 * Terminal tool results (Bash, terminal_screen, PowerShell) are full of ANSI escapes, cursor
 * moves, CR, NUL — garbage for training. This quantifies the damage per field and per tool.
 *
 *   node scrape/analyze_ctrl.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FILE = path.join(__dirname, "..", "data", "tool_calls.jsonl");

// classifiers
const ANSI = /\x1b\[[0-9;?]*[ -\/]*[@-~]/g;   // CSI escape sequences (colors, cursor)
const OSC = /\x1b\][^\x07\x1b]*(\x07|\x1b\\)/g; // OSC sequences (titles, hyperlinks)
const ESC = /\x1b/g;                            // any remaining ESC
const CR = /\r/g;
const NUL = /\x00/g;
const C0 = /[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]/g; // other control chars (excl \t \n)

const count = (s, re) => { const m = s.match(re); return m ? m.length : 0; };
const lenCtrl = (s) => { let n = 0; for (const m of s.matchAll(ANSI)) n += m[0].length; return n; };

const lines = fs.readFileSync(FILE, "utf8").split(/\r?\n/).filter(Boolean);

const agg = { records: 0, dirty: 0, bytes: 0, ansiBytes: 0,
  cat: { ansi: 0, osc: 0, esc: 0, cr: 0, nul: 0, c0: 0 },
  field: { arguments: 0, result_preview: 0, context: 0 } };
const perTool = {};
const samples = [];

for (const line of lines) {
  let r; try { r = JSON.parse(line); } catch { continue; }
  agg.records++;
  const fields = {
    arguments: JSON.stringify(r.arguments || {}),
    result_preview: r.result_preview || "",
    context: (r.context || []).map(c => c.text).join("\n"),
  };
  let dirtyHere = 0, ansiHere = 0;
  for (const [fname, s] of Object.entries(fields)) {
    agg.bytes += s.length;
    const c = { ansi: count(s, ANSI), osc: count(s, OSC), esc: count(s, ESC), cr: count(s, CR), nul: count(s, NUL), c0: count(s, C0) };
    const total = c.ansi + c.osc + c.esc + c.cr + c.nul + c.c0;
    for (const k in c) agg.cat[k] += c[k];
    if (total) { agg.field[fname] += total; dirtyHere += total; }
    const ab = lenCtrl(s); ansiHere += ab; agg.ansiBytes += ab;
  }
  if (dirtyHere) {
    agg.dirty++;
    const pt = perTool[r.tool] || (perTool[r.tool] = { dirty: 0, ctrl: 0, n: 0 });
    pt.dirty++; pt.ctrl += dirtyHere;
  }
  const ptn = perTool[r.tool] || (perTool[r.tool] = { dirty: 0, ctrl: 0, n: 0 }); ptn.n++;
  if (ansiHere > 50 && samples.length < 4) {
    const raw = (fields.result_preview || fields.context).slice(0, 160);
    samples.push({ tool: r.tool, raw: JSON.stringify(raw) });
  }
}

const pct = (a, b) => (b ? (100 * a / b).toFixed(1) : "0") + "%";
console.log(`\nSailfish corpus — control-character audit  (${FILE.split(/[\\/]/).pop()})`);
console.log(`  records          : ${agg.records}`);
console.log(`  dirty records    : ${agg.dirty}  (${pct(agg.dirty, agg.records)} have control junk)`);
console.log(`  text scanned     : ${(agg.bytes / 1e6).toFixed(1)} MB`);
console.log(`  ANSI escape bytes: ${(agg.ansiBytes / 1e6).toFixed(2)} MB  (${pct(agg.ansiBytes, agg.bytes)} of all text is ANSI gunk)`);
console.log(`\n  control chars by type:`);
for (const [k, v] of Object.entries(agg.cat)) console.log(`    ${k.padEnd(6)}: ${v.toLocaleString()}`);
console.log(`\n  where it lives (control-char count by field):`);
for (const [k, v] of Object.entries(agg.field)) console.log(`    ${k.padEnd(15)}: ${v.toLocaleString()}`);
console.log(`\n  dirtiest tools (dirty / total, ctrl-chars):`);
Object.entries(perTool).sort((a, b) => b[1].ctrl - a[1].ctrl).slice(0, 10)
  .forEach(([k, p]) => console.log(`    ${String(p.ctrl).padStart(8)}  ${k}  (${p.dirty}/${p.n} dirty)`));
console.log(`\n  samples (escaped):`);
samples.forEach(s => console.log(`    [${s.tool}] ${s.raw.slice(0, 150)}`));
