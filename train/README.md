# Sailfish — training the tool-drafter

**One rule (learned the hard way):** train on the **target's own greedy tokens** (SeqKD). Acceptance =
"did the drafter emit e4b's exact next token," so that's the only objective worth optimizing. VSD/other
surrogates underperform (see `../ARCHITECTURE.md`).

## Flow

```
1. scrape    node ../scrape/scrape_toolcalls.mjs        → data/tool_calls.jsonl  (real tool-call traces)
2. distill   node distill_e4b.mjs --max 2000            → data/gemma_toolcalls.jsonl
             (replays contexts through the LOCAL e4b, greedy, captures e4b's OWN tool-call tokens)
3a. lookup   node ../drafter/build_ngram_gemma.mjs      → data/ngram_gemma.* (training-free drafter)
3b. seqkd    (later) SFT a small draft model on gemma_toolcalls.jsonl → --model-draft checkpoint
```

## Why distill through the *local* e4b
The drafter must predict **e4b's** tokens, not Claude's (the scrape source uses a different tokenizer).
`distill_e4b.mjs` runs the exact model we're accelerating (`:22343`, greedy/temperature-0) over the
scraped task contexts, so the corpus is in the target's own voice. That corpus is the acceptance target
for both the n-gram lookup (3a) and any SeqKD model drafter (3b).

## Requirements
- The Sailfish container serving e4b on `:22343` (`../container/llamacpp`).
- The scraped corpus (`../data/tool_calls.jsonl`, `../data/agentic_prompts.jsonl`).

## Prereq to wire 3a onto the container
llama.cpp must expose a **static lookup cache** (`--lookup-cache-static`, built via `llama-lookup-create`
over `gemma_toolcalls.jsonl`). If the server build doesn't support it, fall back to the model-draft route
(quantize e2b→Q4, SeqKD-tune it, `--model-draft`). Verify support before investing in the index.
