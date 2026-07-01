# 🗡️🐟 Sailfish

> `sailfish.nuts.services` — the fastest fish in the ocean, ported to your own card.

**The bar:** bare `gemma4:e4b` on Ollama does **high-60s TPS** on the 3060. Sailfish has to beat
that on the same card — that's the entire point of the speculative stack. Anything ≤ Ollama is a fail.

Sailfish is a **sovereign fast-inference stack**: it runs `gemma-4-E4B-it` locally on a single
consumer GPU (built and tuned for a 12 GB Ampere RTX 3060, `sm_86`) and makes it *fast* at the
one thing a local agent actually does all day — **calling tools** — by spearing the next tokens
with a speculative drafter before the big model has to emit them.

A sailfish runs down its prey with its bill. That's the whole idea: the **drafter is the bill**.

## Why this exists

The cloud leaderboard rewards raw decode speed on a frozen model under a brutal stability gate.
Your own card has *no gate* — which means every trick that's illegal up there is fair game here:
adaptive drafting, quant choice, a dumb-but-fast lookup drafter, agentic knobs. Sailfish is where
those live.

## The three speed levers (all stack)

1. **Speculative decoding** — a small drafter proposes K tokens, the target verifies. Lossless by
   construction (the target accepts/rejects), so quality is identical — just faster.
2. **A *tool-specialized* drafter** — tool calls are highly structured and repetitive (`{"command":
   …, "description": …}` every time). A drafter trained on **real tool-call traces** hits very high
   acceptance on exactly the tokens an agent emits. This is the opposite of reasoning text, where
   cheap drafters die — here they *win*.
3. **Quantization** — INT4 weights = fewer bytes hauled per token on a bandwidth-bound card.

## Layout

```
sailfish/
  scrape/      harvest real tool-call traces from Claude Code transcripts  ->  data/
  data/        the harvested corpus (tool_calls.jsonl, agentic_prompts.jsonl, stats.json)
  drafter/     ngram tool-drafter (training-free) + acceptance evaluator; VSD training scaffold
  container/   vLLM sovereign-serve image for the 3060 (OpenAI API, spec-decode baked in)
  harness/     agentic tool-run test rig (measures tool-call accuracy + TPS)
  ROADMAP.md   build order + status
```

## Quickstart

```bash
# 1. harvest the goldmine (your own Claude Code tool-call history)
node scrape/scrape_toolcalls.mjs

# 2. measure how draftable tool calls are (training-free lookup drafter)
node drafter/ngram_tool_drafter.mjs

# 3. stand up the sovereign fast model on the card
cd container && docker compose up --build      # OpenAI API on :8000

# 4. point the harness at it (or at Ollama) and watch it fly
pwsh harness/ollama_tool_harness.ps1
```

## Moving into DeepBlueDynamics

This directory is self-contained. Drop it under `nuts/services/` (addressed as `sailfish.nuts.services`);
nothing here reaches outside its own folder except reading `~/.claude/projects` (the scrape source).

*Named by Claude. Built with Kord. Make it punch above class.*
