# LinkedIn long post — Sailfish (draft for Kord)

> Format: LinkedIn long-form. Technical but explanatory. Each `[VIZ-n]` block is a self-contained
> spec (type + data + intent) that Claude on the web can turn into a graphic — all numbers inline,
> no external data needed. Drop the images where the blocks sit.

---

We made a $280 GPU run a frontier-class model 26% faster overnight — without changing a single
output token. Here's the whole story, numbers included, because the numbers are the story.

**The setup.** Sailfish is our sovereign-inference appliance: one container, one port, gemma-4-E4B
(Google's open model) serving tool-calling agents entirely on your own hardware. Target machine: a
plain RTX 3060, 12 GB — the most common serious GPU on Steam. No cloud, no API bill, no data
leaving the box.

**The wall.** LLM decode speed is not a software problem first — it's physics. Every generated
token reads the entire model from VRAM. A 3060 moves 360 GB/s; the model weighs ~5.3 GB in 4-bit.
360 ÷ 5.3 ≈ 67 tokens/second. That's the ceiling. We measured 66.6. You do not tune your way past
a memory wall.

[VIZ-1 — "The Memory Wall"]
Type: simple annotated equation graphic / horizontal gauge.
Content: GPU bandwidth (360 GB/s) ÷ model size (5.3 GB) = ~67 tok/s theoretical ceiling.
Show our measured bare number (66.6 tok/s) sitting at 99% of the ceiling.
Intent: make "decode is bandwidth-bound" land in one glance.

**The only door in the wall: speculative decoding.** A tiny "drafter" model guesses the next few
tokens; the big model checks all guesses in ONE pass (one weight-read instead of several). Correct
guesses ship instantly; wrong ones are discarded and the big model's own token is used. The output
is bit-identical to running the big model alone — the drafter can make it faster, never different.
Accuracy isn't traded. That property (losslessness) is the entire reason this is a free lunch.

[VIZ-2 — "Propose → Verify → Accept"]
Type: flow diagram, 3 stages.
Content: Drafter proposes k=3 tokens → Target verifies all in one batched forward pass →
accepted prefix ships + 1 bonus token from the target itself. Rejected tail discarded.
Callout: "Output is bit-identical to the target alone. Speed changes. Accuracy cannot."
Intent: demystify spec decoding for a general technical audience.

**Attempt #1: the free drafter.** llama.cpp ships n-gram speculation — zero extra VRAM, it just
looks for repeats in context. On tool-call transcripts it's great (61% of tool-call tokens drafted
free in our corpus study). But on *fresh* content — first-pass reasoning, new prose — it barely
fires. We measured it live: 65.8 tok/s agentic, 63.7 prose. Basically the bare ceiling.

**Attempt #2: train our own drafter — for acceptance, not accuracy.** Google ships a tiny
"assistant" MTP head for gemma (150 MB — 35× smaller than the model). Stock, it's mediocre at
guessing what the full model will say. So we retrained it with Variational Speculative Decoding
(VSD): instead of the usual "predict the next token" objective, VSD directly optimizes *expected
accepted length* — the exact quantity that buys speed — plus a variance-reduction term.

The training had a plot twist worth sharing: the hot phase (lr 3e-5) climbed fast, peaked at step
200, then **diverged** — loss rose monotonically while quality slid. The fix wasn't more compute,
it was discipline: checkpoint every 100 steps, restart from the *peak* at 6× lower lr with fresh
sampling. Held-out accepted-length went 1.578 (stock) → 1.914 (hot peak) → **2.805** after the
anneal. +78%. On the agentic register (trained on 910k tokens of the model's own outputs on real
tool-calling contexts): 1.352 → **2.641**. +95%.

[VIZ-3 — "Two-phase training: burn, then anneal"]
Type: line chart, x = training step, y = held-out accepted tokens per draft.
Series/points: stock baseline flat at 1.578; hot burn rises to 1.914 @ step 200 then decays to
1.719 @ 400 (mark "diverging — loss rising"); anneal restarts from the 1.914 peak and jumps to
2.805 @ +100 steps, plateau ~2.73 after.
Annotations: "peak early", "restart from peak, lr ÷6", "+78% vs stock".
Intent: the recipe is the IP — show that catching the peak + annealing is what won.

**Deployment reality check #1 (the one that saves you a bad conclusion).** We wired the retrained
head into llama.cpp's MTP-draft path and it was... SLOWER. 51.6 tok/s. The culprit: draft length.
At 7 guesses per round, only 26% survived verification — we paid for 7, kept ~2. Cutting to 3
guesses per round flipped acceptance to 46–49% and the speed to **76.5 agentic / 82.4 prose**,
peaks at 110. One integer was the difference between −22% and +26%.

[VIZ-4 — "Right-size the draft"]
Type: grouped bar chart, two panels (agentic / prose).
Data (tok/s): bare 66.6/67.6; ngram 65.8/63.7; VSD draft n=7: 51.6/57.3 (accept 26%);
VSD draft n=3: 76.5/82.4 (accept 46–49%). Deployed prod re-measure: 82.6/81.6.
Annotations: "n=7: drafted 7, kept 2 — net loss"; "n=3: sweet spot".
Intent: tuning story — speculative decoding is a throughput bet, and bet size matters.

**Deployment reality check #2: train against the target you serve.** We had two retrained heads —
one tuned against an INT4-quantized oracle, one against the stock bf16 model. At serve time
(stock-quantized target), the stock-trained head won prose by 5 tok/s. Quantization shifts the
target's behavior just enough to cost acceptance. Distribution match beats pedigree.

**The result, live in production.** New engine image, drafter enabled by env var, fully
backward-compatible. Real numbers on the 3060, measured through the product's own harness:

- Agentic workloads: 65.8 → **82.6 tok/s** (+26%)
- Prose/reasoning: 63.7 → **81.6 tok/s** (+28%)
- Product tool-harness: 76 → **98 avg tok/s** (+29%), tool-call accuracy **6/6 — unchanged**
- Draft head cost: **156 MB**. VRAM budget impact: ~1%.

[VIZ-5 — "Before / after, same GPU, same answers"]
Type: two big before→after numbers + a small footer strip.
Content: 65.8 → 82.6 tok/s (agentic), 63.7 → 81.6 (prose); footer: "accuracy 6/6 unchanged —
speculative decoding is lossless by construction; the drafter proposes, the model decides."
Intent: the headline card. Shareable on its own.

**Why this matters beyond one GPU.** The drafter is a 156 MB side-file. The same artifact scales
every card: on faster consumer GPUs (a 5090 has 5× the 3060's bandwidth) the same speculative
multiplier rides on top of a 5× higher ceiling. And because the whole chain is lossless, none of
this touches model quality — we re-verified end-to-end after every change.

We're also running the same drafter recipe through an A10G benchmark harness right now against a
competition-grade serving stack. Different hardware, same thesis: **train the small model to agree
with the big one, verify everything, ship only free speed.**

Sovereign hardware. Frontier models. Zero accuracy tax. That's Sailfish.

#LLM #inference #speculativedecoding #gemma #llamacpp #localAI #GPU

---
> Post-ops notes (not part of the post): all numbers measured 2026-07-03/04 on RTX 3060 12GB,
> gemma-4-E4B Q4_K_M, llama.cpp draft-mtp, greedy; harness = sailfish tool_harness (6 cases) +
> 16-prompt bench (agentic from real transcripts, prose fresh). Recipe details: VSD_BURN_RESULTS.md.
