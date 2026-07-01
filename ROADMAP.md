# Sailfish — build roadmap & status

**Bar:** bare `gemma4:e4b` on Ollama = **high-60s TPS** on the RTX 3060. Sailfish must beat it.

## Done ✅
- **Scraper** (`scrape/scrape_toolcalls.mjs`) — harvests Claude Code tool-call traces.
  → **25,479 tool calls**, 346 transcripts, 85 unique tools, 0 parse errors.
- **Data** (`data/`) — `tool_calls.jsonl`, `agentic_prompts.jsonl` (replay-training format), `stats.json`.
- **n-gram tool-drafter** (`drafter/ngram_tool_drafter.mjs`) — training-free, held-out:
  → **2.56× tokens/verify**, **61% of tool-call tokens free**; Read 4.86×, Grep 3.34×, Edit 2.93×.
  → table saved to `data/ngram_drafter.json` for lookup integration.
- **Container** — dual backend (`container/` vLLM, `container/llamacpp/` llama.cpp), port **22343**.
  → **llama.cpp backend LIVE on the 3060: 71 TPS bare e4b (beats Ollama's ~66), no spec yet.** Official
    `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M`. vLLM backend blocked on the gemma4 tower-quant bug (scaffolded).
- **Harness** (`harness/tool_harness.ps1`) — dual backend (Ollama / Sailfish-OpenAI), 6/6 tool accuracy.

## Next 🔜
1. **Trained tool-drafter (higher ceiling than n-gram).** Upload `data/agentic_prompts.jsonl` to the
   A100 box, VSD-train a gemma MTP drafter on the agentic distribution (replay through the target).
   The n-gram is the floor; the trained MTP is the ceiling. Swap into the container via `SAILFISH_DRAFTER`.
2. **First container run on the 3060.** Point `SAILFISH_TARGET` at an INT4/AWQ gemma checkpoint, build,
   measure TPS vs the Ollama bar. Baseline mode: `SAILFISH_DRAFTER=none`.
3. **Push the card.** Quant sweep (Q4/AWQ/INT4) × spec-decode K-sweep → find the 3060's ceiling TPS.
4. **n-gram → llama.cpp lookup.** Wire the free drafter into a prompt-lookup path for a no-GPU-cost speedup.
5. **Submit the challenge candidate** — the reasoning drafter (`vsd_full2`/`vsd_curated2`) once A10G-validated.

## Honest open questions
- Exact INT4 gemma-4-E4B repo + `--quantization` flag that loads cleanly on `sm_86` (first-run validation).
- n-gram numbers are a text-token proxy; real speedup is on gemma's tokenizer — structure transfers, values vary.
- vLLM on a 12 GB card: confirm target INT4 + drafter co-resident fits with headroom for KV.
