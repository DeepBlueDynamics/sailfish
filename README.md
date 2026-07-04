# 🗡️🐟 Sailfish

> `sailfish.nuts.services` — the fastest fish in the ocean, ported to your own card.

Sailfish is a **sovereign fast-inference stack**: it runs `gemma-4-E4B-it` locally on a single
consumer GPU (built and tuned for a 12 GB Ampere RTX 3060, `sm_86`) and makes it *fast* at the
one thing a local agent actually does all day — **calling tools** — by spearing the next tokens
with a speculative drafter before the big model has to emit them.

A sailfish runs down its prey with its bill. That's the whole idea: the **drafter is the bill.**

## The bar, and the result

**The bar:** bare `gemma4:e4b` on Ollama does high-60s tok/s on a 3060. Anything ≤ Ollama is a fail.

**The result (measured 2026-07-04, RTX 3060 12 GB, greedy, llama.cpp):**

| config | agentic tok/s | prose tok/s | tool-call accuracy |
|---|---|---|---|
| bare llama.cpp Q4_K_M | 66.6 | 67.6 | 6/6 |
| n-gram lookup drafter (prior default) | 65.8 | 63.7 | 6/6 |
| **VSD-trained drafter (shipped)** | **82.6** | **81.6** | **6/6 — unchanged** |

Product harness average: **76 → 98 tok/s (+29%)**, tool-call bursts 105–108. The drafter is a
**156 MB GGUF** — about 1% of the VRAM budget. Accuracy does not move: speculative decoding is
lossless by construction (the target model verifies every proposed token; output is bit-identical
with any drafter). We additionally verified this on an independent A10G harness: swapping drafters
moved measured perplexity by 0.0002 over 61,797 scored tokens — noise.

Physics note: bare decode on this card runs at ~99% of the memory-bandwidth ceiling
(360 GB/s ÷ 5.34 GB of weights ≈ 67 tok/s). Past that wall, speculation is the only door.

## How the drafter was made (facts, not vibes)

Google ships a small reference drafter head alongside gemma (~183 MB bf16 / ~99 MB Q4). Stock, it
gets **1.35** of its 7-token guesses accepted per draft on real tool-calling work. Ours — the same
model, retrained — gets **2.64**. Same size, ~2× the survivor rate. The recipe:

1. **Harvest your own workload.** `scrape/` pulls real tool-call traces out of local Claude Code
   transcripts (`~/.claude/projects`). Our corpus: 25,479 tool calls across 346 sessions.
2. **Generate on-policy data.** Run the *target model itself* over those contexts (greedy) and keep
   its outputs — ~910k tokens. The drafter must learn to guess *this* model on *this* work; generic
   pretraining text is the classic failure mode.
3. **Train for acceptance, not likelihood.** Variational Speculative Decoding (VSD, arXiv
   [2602.05774](https://arxiv.org/abs/2602.05774)) optimizes *expected accepted length* — the
   quantity that actually buys speed — with a variance-regularization term.
4. **Two-phase schedule: burn, then anneal.** The hot phase (lr 3e-5) peaks early and then
   diverges — checkpoint every 100 steps and keep the peak. Restart *from the peak* at ~6× lower
   lr with fresh sampling. Held-out acceptance jumped 1.914 → 2.805 in the anneal on our reasoning
   head; the shipped agentic head went 1.35 → 2.64 vs stock.
5. **Right-size the draft at serve time.** Drafting 7 tokens per round was a net **loss** (26%
   survived → 51.6 tok/s). Drafting 3 flipped it (46–49% survived → 82.6 tok/s). One integer
   separated −22% from +26%.

Two hard-won operational laws, both measured twice:
- **Train against the target you serve.** A head trained vs an INT4-quantized oracle lost at serve
  time to the same head trained vs the stock model. Distribution match beats pedigree.
- **Bare-protocol scores don't transfer across serving environments.** A leaderboard-champion
  drafter that drives 416 tok/s inside its custom serving stack accepts ~0.08/7 against the stock
  model — functionally useless on a standard card. (And ours loses inside *its* cage. Train where
  you serve, run where you trained.)

## Run it

```bash
cd container && docker compose up --build        # OpenAI API on :22343

# with the trained drafter (the shipped config):
#   SAILFISH_SPEC=draft-mtp
#   SAILFISH_DRAFT_GGUF=/root/.cache/sailfish/vsd_tool_f16.gguf
#   SAILFISH_DRAFT_NMAX=3
# note: flash-attn defaults off when a draft head is set (ggml fattn head-dim case on sm_86;
# costs ~2 tok/s, the draft pays it back 8x)
```

## Train one on your own sessions

Everything above is reproducible from this repo against **your** workload — the corpus scraper,
the acceptance evaluator, and the training scaffold are here:

```bash
node scrape/scrape_toolcalls.mjs        # harvest your own tool-call history -> data/
node drafter/ngram_tool_drafter.mjs     # measure how draftable your workload is (training-free)
# drafter/ holds the VSD training scaffold; full recipe writeup + drafter weights: soon
```

If your agent spends its day in different tools than ours, your drafter will learn *your* register
— that's the point. The method is workload-shaped by design.

## Layout

```
sailfish/
  scrape/      harvest real tool-call traces from Claude Code transcripts  ->  data/
  data/        the harvested corpus (tool_calls.jsonl, agentic_prompts.jsonl, stats.json)
  drafter/     ngram tool-drafter (training-free) + acceptance evaluator; VSD training scaffold
  container/   sovereign-serve image for the 3060 (llama.cpp engine + gateway, OpenAI API :22343)
  harness/     agentic tool-run test rig (measures tool-call accuracy + TPS)
  docs/        story/ (how this happened, long-form) · integrations/
  ROADMAP.md   build order + status
```

## References

- Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding*,
  [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al. — *Accelerating LLM Decoding with Speculative Sampling*,
  [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- *Variational Speculative Decoding* (VSD), [arXiv:2602.05774](https://arxiv.org/abs/2602.05774) —
  the training objective used here
- DSpark (semi-autoregressive drafting + confidence-scheduled verification) — the paper that
  reshaped our verification budget thinking
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — serving engine (`--spec-type draft-mtp`)
- [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) + Google's reference assistant
  drafter — the frozen target and the base we retrain
- The long version, receipts included: [`docs/story/BACKSTORY.md`](docs/story/BACKSTORY.md)

**More soon — drafter weights, the full training writeup, and bigger cards. Stay tuned.**

*Named by Claude. Built with Kord. Make it punch above class.*
