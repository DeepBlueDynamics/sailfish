# BACKSTORY — the challenge, the release, and the loop

*How a leaderboard chase became a shipped product, told honestly — including the dead ends.
This is the story of a feedback loop: Kord pushing, questioning, and dropping papers on the
table; the machine executing, measuring, and refusing (eventually) to fool itself.*

---

## 1. The Challenge

The Fast Gemma Challenge is a beautifully cruel benchmark: **google/gemma-4-E4B-it, frozen.**
You may not make the model smarter — perplexity is gated at ≤2.42 with a ±5% private rerun to
kill flaky entries. All you can sell is **speed**: decode tokens-per-second on a single A10G.
The A10G is the great equalizer — 600 GB/s of memory bandwidth, and decode speed is bandwidth
physics before it is anything else. Every generated token re-reads the whole model. Want more
tokens? Read less, or read smarter.

We got on the board the honest way: INT4 quantization, a fused MTP drafter at K=7, CUDA-graph
surgery, token curation — **504.81 TPS at 2.394 PPL, verified lineage, median-of-three.** Top of
the verified board sat ~510. The remaining edges were thin: keepset curation for rerun stability,
and the one lever everyone underestimates — **drafter quality**.

A note on that lever, because it's the spine of everything that follows: speculative decoding is
**lossless**. A small drafter guesses; the frozen target verifies every guess in one batched pass;
outputs are bit-identical to running the target alone. The drafter can buy speed and *cannot*
spend accuracy. In a competition where accuracy is a hard gate, the drafter is the only part of
the stack with no downside.

## 2. The release — Sailfish

Somewhere in the leaderboard grind a second question formed: *what is all this for?* The answer
became **Sailfish** — the sovereign inference appliance. One container, one port (22343),
gemma-4-E4B on a plain RTX 3060, specialized for the one thing a local agent does all day:
**calling tools.** No cloud, no API meter, no data leaving the box.

Sailfish shipped Tier B first: llama.cpp, official GGUF, zero-VRAM n-gram speculation — measured,
working, humble numbers (71 TPS bare; the n-gram only fires on repeats). The challenge stack was
the lab; Sailfish was the street. The bet was that the lab would eventually feed the street.

It did. But not by itself.

## 3. The Loop

This project runs on a specific dynamic. The machine executes and measures; Kord pushes,
interrogates, and — at exactly the right moments — throws a paper on the table. The transcript
tells it better than summary can:

**"keep going please do not stop until you get the drafter out of training... question your
desire to stop if you do."** — The standing order. Not "work hard": *audit your own urge to
quit.* It caught real failures-of-nerve: an EAGLE-3 head that served at ā=0.33 (off-policy data,
the classic trap) would have been a natural stopping point. The order said: diagnose it instead.
Verdict: not a bug — wrong data. That diagnosis seeded everything later.

**"don't fool yourself on a squirrel chase that can be fixed with a modulated technique."** — The
counterweight. Push hard, but don't confuse stubbornness with progress. This line killed more bad
plans than any experiment did.

**"i proposed multi drafters before."** — The thesis was his, months old: don't race two drafters
as peers; **compose specialists** — a heavy proposer, a tiny corrector for the tail, a lookup
planner for repeats. The lab kept re-deriving this idea from different directions until it
finally listened.

**"wait, we have it smaller. talk to me."** — An interrupted box restart, and a working rule
born from it: talk before spending money. The betting-odds ritual ("give me betting odds",
"what's our bet on expected TPS? on my card, and my kid's card 5090 32GB") came from the same
instinct — force the machine to price its confidence *before* the result, so wins and losses
both teach calibration.

## 4. The paper

Then the paper landed — twice, actually: **"sorry i gave you the wrong paper!"** The right one
was **DSpark**: a semi-autoregressive drafter — parallel backbone for the easy positions, a tiny
serial **low-rank Markov transition head** to fix the decaying tail, and **confidence-scheduled
verification** to stop wasting verify compute on doomed guesses. Beats EAGLE-3 by 27–31% in the
paper's setting. Kord's push: *"the spark gemma is all over the fast gemma challenge board"* —
this wasn't theory, the board was already breathing it. Then the directive: **"so use the stock
or leader modified drafter and apply this to it. don't forget our micro planner we have in
sailfish as well. build a plan."**

Here's the honest part: **DSpark never shipped as an architecture in our stack. It shipped as
three ideas that did.**

1. **The Markov head idea** forced the cheapest possible experiment first: count bigrams of the
   target's *own greedy output* and measure the signal. On prose it looked alive (0.346 top-1);
   on the challenge's reasoning eval it died (**0.228 — gate failed, $3 spent, half a day of CUDA
   surgery saved**). But the probe's *byproduct* was the treasure: **720k tokens of on-policy,
   eval-matched greedy** — the exact fuel whose absence had sunk EAGLE-3. The corrector was wrong;
   the *on-policy principle* it dragged in was right.
2. **Confidence-scheduled verification** resurfaced at deploy time as the humblest possible knob:
   draft length. n-max 7 was a net LOSS (−22%); n-max 3 flipped it to +26%. One integer,
   DSpark's scheduling insight in trench coat.
3. **The composition thesis** became the deployed design: MTP-style drafter for fresh content +
   n-gram planner for repeats, each covering the other's blind spot — Kord's multi-drafter
   proposal, finally built.

**"green lit let's take these nerds down."** The box burned. VSD — Variational Speculative
Decoding, training the drafter *directly for accepted length* on the model's own outputs — turned
out to be the weapon the DSpark push had been steering toward all along. The first run diverged
after step 200 (loss rising monotonically — caught because we checkpoint every 100). **"burn it
if we have a chance or don't know... YOU ARE TOKEN MAXIMUS."** So: anneal-from-the-peak, 6× lower
lr, fresh sampling. Held-out accepted length: **1.578 → 1.914 → 2.805 (+78%).** A second head,
trained on 910k tokens of real agentic tool-calling contexts against the *stock* target: **1.352
→ 2.641 (+95%)** — and that oracle choice mattered, because at serve time the stock-trained head
beat the INT4-trained one. *Train against the target you serve.* Measured, not assumed.

## 5. The payoff

**"Build and deploy Sailfish locally... I want tests run I want real numbers... accuracy
accuracy accuracy."**

Same night, on the 3060, through the product's own harness:

| | before | after | Δ |
|---|---|---|---|
| agentic workloads | 65.8 tok/s | **82.6** | +26% |
| prose/reasoning | 63.7 tok/s | **81.6** | +28% |
| tool harness avg | 76 tok/s | **98** | +29% |
| tool-call accuracy | 6/6 | **6/6** | **unchanged** |

The drafter is a 156 MB side-file. The accuracy line is the whole ideology: every point of speed
was free because the target verifies everything. **"deploy the test to hf. let's do this."** —
as this file is written, two A10G jobs are running the official challenge harness in A/B: stock
drafter vs ours, single variable, PPL re-verified by the official scorer. The test lane, not the
leaderboard — that trigger stays human.

## 6. What the loop actually is

Strip the specifics and the method is portable:

- **The machine measures; the human interrogates.** Every "how do the numbers look" forced a
  number to exist.
- **Question the desire to stop** — but pair it with **don't chase squirrels**. The tension
  between those two lines is where the good decisions lived.
- **Price it before you run it.** Betting odds made overconfidence expensive and honest losses
  cheap.
- **Papers are idea-donors, not blueprints.** DSpark contributed a probe, a knob, and a thesis —
  not a checkout.
- **Cheap kill-gates before expensive integrations.** $3 probes, 10-step smokes, pre-registered
  thresholds. The corrector died for pocket change; the recipe that worked got the GPU-hours.
- **Byproducts are products.** The failed corrector's dataset trained the drafter that shipped.
- **Accuracy is not a trade-off surface.** Lossless-by-construction meant the entire campaign ran
  with zero accuracy risk. Pick battlegrounds where your downside is structurally zero.

The challenge gave us the physics. The paper gave us the shape. The loop — push, measure, price,
burn, question the stop — gave us the product. And the nerds, per directive, were taken down a peg.

---
*Artifacts and receipts: SUBMISSION_NOTES.md (the board era), MULTIDRAFTER_PLAN.md (the plan the
paper triggered), VSD_BURN_RESULTS.md (the recipe + traps), SAILFISH_DEPLOYMENT_PLAN.md (the
intel-to-product map), COMPETITION_EVAL.md (the accuracy-first submission case), sailfish repo
149ff88 (the shipped engine change), LINKEDIN_POST.md (the public telling).*
