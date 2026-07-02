# Sailfish — architecture & hard-won lessons

The distilled knowledge behind this service, so a new agent doesn't re-learn it the expensive way.

## What Sailfish is
A **sovereign fast-inference** service: run `gemma-4-E4B-it` locally (OpenAI API on `:22343`) and make
it *fast at tool-calling* via speculative decoding, with a drafter specialized on real tool-call traces.

## The core mechanism: speculative decoding = lookahead + verify
A small **drafter** guesses the next K tokens; the big **target** verifies all K in one forward pass.
Every agreed token is free; the first miss, the target corrects. **Lossless** — output is byte-identical
to normal decoding. The drafter is a lossy *compressed predictor*; the target catches its mistakes.

## The compression stack (where speed actually comes from)
1. **Weights bf16→INT4** (4× fewer bytes hauled — the biggest lever; decode is memory-bandwidth-bound).
2. **LM-head 262,144→12,288 vocab** (~21× thinner final projection; the frontier's `pck04`/SlimSpec).
3. **Sliding-window attention** (bounded KV read; minor for short prompts).
4. **The drafter itself** (tiny 4-layer model approximating the target's next token).

## The drafter lesson — the expensive one we already paid
Our own drafter served **319 TPS** vs kduma's **413** on the *byte-identical* architecture and the same
target. So the gap is **100% acceptance** — how often the drafter's guesses match the target's argmax:
- Ours: **41.2%** draft acceptance (2.88 of 7 accepted/verify).
- kduma's: higher (~50%+, inferred from +29% serving on identical arch).

**Why ours lost:** we trained with **VSD (variational — a surrogate objective)** for 4,000 steps.
kduma used **straight SeqKD** (fine-tune the drafter to reproduce the target's greedy tokens) for **one
epoch** — and won. Direct objective beats surrogate. **Ruled out:** it is NOT the data (we cleaned it),
NOT the epochs (we did 12× more), NOT the head (identical). **It is the training objective.**
→ **Always train the drafter with SeqKD on the target's own greedy completions.** (See `train/`.)

## Local-fit constraints (the 12 GB reality)
| drafter | fits 12 GB | runs in llama-server | note |
|---|---|---|---|
| MTP head (~200 MB) | yes | **NO** — fused to the target, vLLM-only | the "right" draft, unusable locally |
| e2b via `--spec-draft-hf` | borderline (e4b-Q4 4.97 + e2b-Q8 4.63 ≈ 9.6 GB + KV) | ✅ yes | **heavy** draft (~½ of e4b) → speedup uncertain; off-the-shelf test |
| **SeqKD small draft** via `--spec-draft-hf` | yes (small) | ✅ yes | **the goal** — train a small draft on the gemma tool corpus |
| n-gram prompt-lookup | trivial (zero VRAM) | **NO** — llama-*server* has no `--lookup-cache-static` (CLI-only) | corpus feeds the SeqKD draft instead |

**Confirmed (2026-07-01):** `llama-server` (ggml-org server-cuda) exposes `--spec-draft-hf`/`--model-draft`
(draft *model* spec-decode) but **no lookup/ngram option** — that decoder lives in the `llama-lookup` CLI,
not the server. So the container drafter must be a **model**. The 25k tool corpus → `distill_e4b.mjs` →
SeqKD-tune a small draft → `--spec-draft-hf`. e2b is the off-the-shelf plumbing test.

vLLM **cannot** load a quantized gemma4 on Ampere (audio-tower quant bug, vLLM #40247). Ollama's MTP
speedup is **Apple-MLX-only**. Ollama's own GGUFs are a *forked* build — use **ggml-org GGUFs** for
mainline llama.cpp (`ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M`, 4.97 GB, what the container runs).

## The tool-call insight (why a "dumb" drafter wins here)
Cheap/n-gram drafters get **1.2%** acceptance on *reasoning* generation (they die) but **2.56×** on
*tool calls* — because tool calls are structured and repetitive (same names, JSON arg keys). The dumb
drafter found its ocean. This is a **local** play (agentic tool-runs), never the leaderboard (reasoning).

## Vocabulary curation — what the leaderboard keepset war taught us (2026-07-02)
The frontier prunes gemma's 262,144-token lm_head down to ~12k rows ("keepset") for bandwidth. Sailfish
inherits this lever, but with hard-won caveats — we ran the full experiment on A100s:
- **The shared 12k keepset is fitted to the public eval's answer key** (floor + specials + ascending-ID
  fill that happens to cover all 5,892 public target tokens perfectly). It is NOT frequency-profiled.
- **Regenerating answers on disjoint prompts puts ~1-6% of target tokens OUTSIDE the 12k** → those score
  `-inf` → perplexity detonates. This is real out-of-distribution tail, not noise.
- **SKG/embedding-diffusion curation helps only marginally.** We built keepsets by cosine-diffusing from
  observed tokens across the model's tied embedding graph. Tournament (16k→26k budgets): held-out miss
  only fell 5.8%→4.3% and **never reached zero**. The private tail is genuinely OOD — diffusion reaches
  *unobserved-but-nearby*, not *unobserved-and-far*. Raw keepset SIZE is the only real coverage lever,
  and size costs bandwidth.
- **The lm_head prune can only go SMALLER than its source tier** (subset check in the serve). To field a
  differently-composed head you must slice from a full-vocab (262k) target, not a pre-pruned one.
- **PPL is denominator-sensitive**: a smaller head = smaller softmax denominator = *lower* reported PPL,
  independent of real fidelity. Never trust a sub-bf16-reference PPL (e.g. our 1.99 artifact) — it's the
  truncated denominator, not a better model. Always sanity-check against the bf16 reference (~2.30).

**Sailfish takeaway:** for the LOCAL product, keepset choice is a **speed knob with a fidelity floor**, not
a magic lever. Ship a keepset sized to the user's *own* tool-call token distribution (which we CAN profile
from their real data — unlike the challenge's hidden private set), and it's honest and robust *because the
user's future traffic is the same distribution as their past*. The OOD problem that dooms the leaderboard
keepset doesn't exist for a personal model trained on personal data. That asymmetry is a Sailfish advantage.

## The pipeline
```
scrape/scrape_toolcalls.mjs   ~/.claude + agent sandboxes  → data/tool_calls.jsonl (25k, ANSI-scrubbed)
train/distill_e4b.mjs         replay contexts through LOCAL e4b greedily → data/gemma_toolcalls.jsonl
                              (e4b's OWN tokens = the SeqKD/acceptance target)
drafter/build_ngram_gemma.mjs  index gemma sequences → the lookup drafter   (training-free, zero-VRAM)
   (later) train/seqkd.*       SFT a small draft model on the same corpus   (→ --model-draft)
container/llamacpp             e4b-Q4 target + the drafter → OpenAI API :22343
harness/tool_harness.ps1       measure tool-call accuracy + TPS (-Backend openai -Port 22343)
```

## Status (what's proven vs theory)
- ✅ Local e4b sovereign endpoint LIVE (`:22343`, ~55-71 TPS, **bare target, NO drafter yet**).
- ✅ n-gram drafter concept: 2.56× on Claude-token tool calls (needs re-index on *gemma* tokens).
- ✅ 25k tool-call corpus scraped + ANSI-cleaned.
- ⬜ First real drafter on the container (verify `--lookup-cache-static` support → wire it).
- ⬜ SeqKD self-distill run (`train/distill_e4b.mjs`) → gemma corpus.
- ⬜ Optimize: quantize e2b→Q4 if we go the model-draft route; tune K / lookup depth.
