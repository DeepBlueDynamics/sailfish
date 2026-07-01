#!/usr/bin/env node
/**
 * Sailfish :: drafter — training-free n-gram tool-drafter + acceptance evaluator.
 *
 * Builds a prompt-lookup drafter from the scraped tool-call corpus and measures, on a held-out
 * split, how many tokens a target would accept per verify when this drafter proposes them —
 * i.e. the speculative-decode speedup you'd get on tool-call generation, for ZERO training cost.
 *
 *   node drafter/ngram_tool_drafter.mjs [--k 8] [--order 4] [--test 0.15]
 *
 * The premise: tool calls are structured + repetitive, so a lookup over prior calls predicts the
 * next token cheaply. This is the "dumb drafter" landing in the venue where it actually wins.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(__dirname, "..", "data", "tool_calls.jsonl");

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const K = parseInt(opt("--k", "8"), 10);          // max draft length per verify
const ORDER = parseInt(opt("--order", "4"), 10);  // n-gram context length
const TEST = parseFloat(opt("--test", "0.15"));   // held-out fraction

// What the assistant actually emits for a tool call: name + argument JSON.
const serialize = (rec) => `${rec.tool} ${JSON.stringify(rec.arguments)}`;
const tokenize = (s) => s.match(/[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) || [];

// ---- load + split (deterministic, by hash so it's reproducible) ----
const records = fs.readFileSync(DATA, "utf8").split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l));
const hash = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 2 ** 32; };
const train = [], test = [];
for (const r of records) (hash(r.tool + JSON.stringify(r.arguments)) < TEST ? test : train).push(r);

// ---- build backoff n-gram lookup over the training serializations ----
// maps "tok|tok|tok" (orders 1..ORDER) -> {nextToken: count}
const tables = Array.from({ length: ORDER + 1 }, () => new Map());
function bump(map, key, tok) { let m = map.get(key); if (!m) { m = new Map(); map.set(key, m); } m.set(tok, (m.get(tok) || 0) + 1); }
for (const r of train) {
  const toks = tokenize(serialize(r));
  for (let i = 0; i < toks.length; i++)
    for (let o = 1; o <= ORDER; o++)
      if (i - o >= 0) bump(tables[o], toks.slice(i - o, i).join(""), toks[i]);
}
const argmax = (m) => { let best = null, bc = -1; for (const [t, c] of m) if (c > bc) { bc = c; best = t; } return best; };
function predict(prev) {                   // backoff: longest matching context wins
  for (let o = Math.min(ORDER, prev.length); o >= 1; o--) {
    const m = tables[o].get(prev.slice(prev.length - o).join(""));
    if (m) return argmax(m);
  }
  return null;
}

// ---- evaluate: simulate spec-decode accept length per verify on held-out tool calls ----
function evalStream(toks) {
  let i = 0, accepted = 0, verifies = 0;
  while (i < toks.length) {
    let a = 0;
    while (a < K && i + a < toks.length && predict(toks.slice(0, i + a)) === toks[i + a]) a++;
    accepted += a; verifies++; i += a + 1;   // target accepts `a` drafted + emits 1
  }
  return { accepted, verifies, tokens: toks.length };
}

let A = 0, V = 0, T = 0;
const perTool = {};
for (const r of test) {
  const toks = tokenize(serialize(r));
  const e = evalStream(toks);
  A += e.accepted; V += e.verifies; T += e.tokens;
  const pt = perTool[r.tool] || (perTool[r.tool] = { a: 0, v: 0, n: 0 });
  pt.a += e.accepted; pt.v += e.verifies; pt.n++;
}

const meanAccept = A / V;                       // avg drafted tokens accepted per verify
const tokensPerVerify = meanAccept + 1;         // + the target's own token => speedup factor
const acceptFrac = A / T;                       // fraction of all tokens that were free-drafted

console.log(`\nSailfish n-gram tool-drafter — held-out acceptance`);
console.log(`  corpus: ${records.length} calls  (train ${train.length} / test ${test.length})   order=${ORDER}  K=${K}`);
console.log(`  mean accepted/verify : ${meanAccept.toFixed(2)} tokens`);
console.log(`  tokens per verify    : ${tokensPerVerify.toFixed(2)}   <-- spec-decode speedup factor on tool calls`);
console.log(`  drafted-for-free     : ${(acceptFrac * 100).toFixed(1)}% of all tool-call tokens`);
console.log(`\n  most-draftable tools (tokens/verify):`);
Object.entries(perTool).filter(([, p]) => p.n >= 5).map(([k, p]) => [k, (p.a / p.v) + 1, p.n])
  .sort((a, b) => b[1] - a[1]).slice(0, 12)
  .forEach(([k, s, n]) => console.log(`    ${s.toFixed(2)}x  ${k}  (n=${n})`));

// persist the drafter table (compact) for llama.cpp/vLLM lookup integration later
const dump = {};
for (let o = 1; o <= ORDER; o++) for (const [key, m] of tables[o]) dump[`${o}${key}`] = argmax(m);
fs.writeFileSync(path.join(__dirname, "..", "data", "ngram_drafter.json"), JSON.stringify({ order: ORDER, table: dump }));
console.log(`\n  saved drafter table -> data/ngram_drafter.json (${Object.keys(dump).length} entries)`);
