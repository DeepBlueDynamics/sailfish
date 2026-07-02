# Sailfish vs Ollama — decode-speed profile (measured 2026-07-02)

Head-to-head on **one RTX 3060 (12 GB, Ampere sm_86)**, greedy decode, 512 tokens, median of 3 reps,
same prompts both backends. Reproduce: `pwsh harness/profile.ps1 -Backend openai -Port 22343` and
`pwsh harness/profile.ps1 -Backend ollama`. Backends were run **separately** (each with the full card;
Ollama's 9.6 GB + Sailfish's ~6 GB won't co-reside on 12 GB).

| regime | Ollama `gemma4:e4b` | Sailfish (llama.cpp + ngram) | speedup |
|---|---|---|---|
| prose (story) | 68.4 tok/s | 88.3 tok/s | **1.3×** |
| code (BST impl) | 68.7 tok/s | 76.2 tok/s | **1.1×** |
| repetitive (numbered list) | 69.8 tok/s | **853.5 tok/s** | **12.2×** |
| agentic-json (repeated objects) | 67.4 tok/s | **340.2 tok/s** | **5.0×** |
| overall median | 68.6 tok/s | 339.6 tok/s | — |

## What it shows
- **Ollama is flat (~68–70 tok/s) across every regime** — it has no speculative decoding, so output shape
  doesn't change its speed.
- **Sailfish's win is regime-dependent, exactly as designed.** n-gram speculation drafts predictable
  tokens for free: neutral-to-better on prose/code, **5–12× on the structured/repetitive output that
  agentic workloads actually generate** (tool calls, JSON, tables, repeated edits). This is the whole
  Tier-B thesis, measured.

## Honest caveats
- **Quant differs.** Sailfish serves the official ggml-org **Q4_K_M (5.3 GB)**; Ollama's shipped
  `gemma4:e4b` is **9.6 GB** (heavier quant). So the prose/code edge (1.1–1.3×) mixes a lighter quant +
  llama.cpp efficiency and is *not* purely the stack. **The repetitive/agentic blowout is purely
  speculation** — Ollama shows *zero* regime sensitivity (68→70), so the 5–12× cannot be a quant effect;
  it's the drafter. A same-quant run (point Ollama at the same GGUF) would tighten the prose number but
  not change the regime story.
- **Warmup:** the first rep of each Sailfish regime measured ~66 tok/s (cold CUDA graph / cache); steady
  state (the median) is what's reported. Ollama had no such warmup swing.
- Tool-calling correctness (separate harness): **6/6** on the same card, 74 tok/s.

## Takeaway
For chat-shaped prose the two are close (with a real quality/speed quant tradeoff to be honest about).
For **agent traffic** — structured, repetitive, tool-heavy — Sailfish is **5–12× faster on the same
$300 card**, because that output is exactly what free n-gram speculation eats.
