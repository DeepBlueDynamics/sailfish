# Sailfish — build plan

A self-serve appliance: pull your agents' tool-call logs from **nemesis8**, fine-tune (and/or train a
drafter) on an ephemeral cloud GPU, download the model back, and serve it fast on your own Windows box.
GPU-**platform**-aware. Windows-first.

## Guiding decisions (locked)
- **Differentiate by silicon, not VRAM.** **Ampere → llama.cpp** (fine-tuned model, optional small draft
  model). **Blackwell → vLLM** (real MTP-head drafter — the Gemma-challenge path). Runtime follows the
  architecture because vLLM can't load quantized gemma4 on Ampere; llama.cpp can.
- **Two independent products, never conflated:**
  1. **Fine-tune the target** on your tool calls → *better* tool-calling (accuracy). Everyone.
  2. **Train a drafter** → target frozen, *faster* (lossless). Blackwell (MTP) or bigger llama.cpp (small draft).
- **Data via nemesis8**, not our own parsers (parser is a fallback). See `NEMESIS8_INTEGRATION.md`.
- **Rust** extractor/parser · **Python** trainer.
- **Both training backends:** HF Jobs (simple auth, smaller boxes) *and* GCloud ephemeral A100 (BYO account).
- **SeqKD is the training objective** (target's own greedy tokens). See `ARCHITECTURE.md`.

## Platform / product matrix
```
silicon    default runtime   fine-tuned target   drafter (speed)         notes
Ampere     llama.cpp         yes (GGUF)          small SeqKD draft model  vLLM can't load quant gemma4 here
Blackwell  vLLM              yes                 MTP head (Gemma-chal.)   the real drafter path
(any, alt) Ollama            drop-in GGUF        —                        easy-mode serving
```
Tiny tier: fine-tuned **e2b** (dense — fine-tunable) for 8 GB / low-end. Autodetect VRAM+arch → pick setup.

## Components
1. **`extractor/` (Rust)** — talks to the nemesis8 controller (`host.docker.internal`), pulls clean
   tool-call traces per source; ships a Claude-Code transcript parser as fallback. Emits `tool_calls.jsonl`.
2. **`train/` (Python)** — `distill_e4b.mjs` (self-distill via local target) → SeqKD trainer → checkpoint.
   Two launchers: `hfjobs_launch.py` and `gcloud_a100_launch.py`. Outputs a fine-tuned target GGUF and/or
   a drafter checkpoint.
3. **`container/`** — `llamacpp/` (Ampere default) + `vllm/` (Blackwell). Autodetect picks one. Serves
   OpenAI API on `:22343`; drops in the downloaded model + optional draft.
4. **`web/`** — local UI: "where's nemesis8?" (default localhost) → list sources → pick dataset → preview
   → enter GCloud org/token OR use HF Jobs → launch train → progress → download → restart → serving.
5. **nemesis8 API** — the data source (spec handed to the nemesis8 build agent).

## Training backends
- **HF Jobs (MVP):** what we already use. Org-paid or user token. A10G/A100 flavors. Simplest auth.
- **GCloud ephemeral A100 (v2):** BYO Google account + token; launch ephemeral instance, train, download,
  tear down (cheaper than static). Heavier auth story — ships after the HF-Jobs path is proven.

## Build phases
- **P0 (now):** prove the Ampere drafter path (e2b-draft on llama.cpp — *running*). Decide draft viability.
- **P1:** Python SeqKD trainer + HF-Jobs launcher → produce a fine-tuned e4b GGUF from the tool corpus,
  drop into the running container, measure tool-call accuracy lift on the harness.
- **P2:** Rust extractor + nemesis8 API client (parser fallback first, live API when Tyrannosaurus ships it).
- **P3:** web UI (local, single-page) wiring P1+P2 into click-to-train-and-serve.
- **P4:** GCloud ephemeral A100 backend; Blackwell/vLLM drafter path; Ollama drop-in.

## Open decisions (will proceed with these defaults unless told otherwise)
- **Base models:** e4b (main), e2b (tiny). Target GGUFs from `ggml-org`.
- **Fine-tune method:** LoRA/QLoRA on the target (cheap, mergeable to GGUF) for the "better tool-calling"
  product; SeqKD for the drafter. Default LoRA unless full-FT proves needed.
- **Serving default per platform** as in the matrix; user can override runtime in the UI.
