#!/usr/bin/env python3
"""
Sailfish :: train — LoRA fine-tune the target on a tool-call corpus (the "better tool-calling" product).

This is the Ampere/12GB product: teach the served model to pick the right tool and emit valid arguments,
by SFT on real tool-call examples. LoRA (cheap, mergeable), runs on an A10G/A100 (HF Jobs or GCloud).

  python finetune_target.py \
      --base google/gemma-4-E4B-it \
      --data data/tool_calls.jsonl \
      --out  out/e4b-toolft \
      --epochs 1 --lora-r 16

SERVING-FORMAT FIDELITY (this is load-bearing — PLAN.md §6.4, §11):
A fine-tune only transfers if training input == serving input. At serve time the model sees, in every
turn, the **tool schemas** (rendered by the chat template from a `tools=[...]` list) and must emit the
call in **gemma's native tool-call syntax**. So we:
  1. synthesize a compact tool schema per tool from observed argument keys (the corpus doesn't ship
     schemas), and present each example with a realistic menu = the called tool + sampled distractors;
  2. render prompt AND completion through `tokenizer.apply_chat_template(..., tools=tools)` so the
     assistant turn is a real `tool_calls` object in gemma syntax — never a raw json.dumps in content;
  3. train **completion-only** (prompt masked) via a prompt/completion dataset.

Data: one JSON object per line, {context:[{role,text}], tool, arguments}  (nemesis8 / scraper format).

Output: a LoRA adapter (+ optional merged weights). Convert to GGUF for llama.cpp with llama.cpp's
convert_hf_to_gguf.py after merging; keep as adapter for vLLM.
"""
import argparse, json, os, random
from collections import defaultdict
from pathlib import Path

# how many candidate tools to show per example (called tool + distractors) — a realistic menu that
# teaches discrimination without blowing context. Menu is deterministic per row index.
MENU_SIZE = 14


def _py_type(v):
    if isinstance(v, bool): return "boolean"
    if isinstance(v, (int, float)): return "number"
    if isinstance(v, list): return "array"
    if isinstance(v, dict): return "object"
    return "string"


def load_corpus(path):
    """Return (rows, tool_registry, tool_freq).
    rows: [{context:[{role,text}], tool, arguments}]
    tool_registry: {tool -> {arg_key -> json-schema type}}   (synthesized from observed args)
    tool_freq: {tool -> count}   (for weighted distractor sampling)"""
    rows, registry, freq = [], defaultdict(dict), defaultdict(int)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tool = r.get("tool")
        if not tool:
            continue
        args = r.get("arguments") or {}
        ctx = [c for c in (r.get("context") or []) if c.get("text") and c.get("role") in ("user", "assistant")]
        if not any(c["role"] == "user" for c in ctx):
            continue
        reg = registry[tool]  # touch -> every tool gets an entry, even no-arg tools (e.g. *_status)
        for k, v in args.items():
            reg.setdefault(k, _py_type(v))
        freq[tool] += 1
        rows.append({"context": ctx, "tool": tool, "arguments": args})
    return rows, dict(registry), dict(freq)


def tool_schema(name, params):
    """OpenAI-style function tool the chat template understands. Descriptions are intentionally terse —
    the corpus has none, and the point is the SHAPE (name + arg keys), which is what transfers."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool.",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": t} for k, t in sorted(params.items())},
                "required": sorted(params.keys()),
            },
        },
    }


def build_menu(called, registry, freq, rng):
    """called tool + (MENU_SIZE-1) frequency-weighted distractors, shuffled — a realistic tool menu."""
    others = [t for t in registry if t != called]
    weights = [freq.get(t, 1) for t in others]
    picks = set()
    # weighted sampling without replacement (small N; simple loop is fine)
    pool, w = others[:], weights[:]
    while pool and len(picks) < MENU_SIZE - 1:
        i = rng.choices(range(len(pool)), weights=w, k=1)[0]
        picks.add(pool.pop(i)); w.pop(i)
    names = list(picks) + [called]
    rng.shuffle(names)
    return [tool_schema(n, registry[n]) for n in names]


def sanitize_context(ctx):
    """Gemma requires the conversation to start with user and alternate. Collapse to a clean,
    alternating history ending on a user turn (so the next turn is the assistant tool call)."""
    msgs = []
    for c in ctx:
        role, text = c["role"], (c["text"] or "").strip()
        if not text:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text[:2000]
        else:
            msgs.append({"role": role, "content": text[:2000]})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    while msgs and msgs[-1]["role"] != "user":
        msgs.pop()
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--data", default="data/tool_calls.jsonl")
    ap.add_argument("--out", default="out/e4b-toolft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into base weights on save")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.base, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows, registry, freq = load_corpus(args.data)
    print(f"[sailfish-ft] {len(rows)} tool-call examples, {len(registry)} unique tools from {args.data}")
    if not rows:
        raise SystemExit("no usable tool-call examples — check the data format")

    def render(row, idx):
        """Return {prompt, completion} in EXACT serving format: tools in the prompt, gemma-native
        tool-call in the completion, so the LoRA transfers to the real harness."""
        rng = random.Random(args.seed * 1_000_003 + idx)
        history = sanitize_context(row["context"])
        if not history:
            return None
        tools = build_menu(row["tool"], registry, freq, rng)
        assistant = {"role": "assistant",
                     "tool_calls": [{"type": "function",
                                     "function": {"name": row["tool"], "arguments": row["arguments"]}}]}
        prompt = tok.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(history + [assistant], tools=tools, tokenize=False)
        if not full.startswith(prompt):
            # template didn't render as a clean prefix (rare) — skip rather than mis-mask
            return None
        completion = full[len(prompt):]
        if not completion.strip():
            return None
        return {"prompt": prompt, "completion": completion}

    pairs = [p for p in (render(r, i) for i, r in enumerate(rows)) if p]
    print(f"[sailfish-ft] {len(pairs)} examples rendered in serving format (prompt/completion, masked)")
    ds = Dataset.from_list(pairs)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
        attn_implementation="eager",
    )
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft_cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len, bf16=True, logging_steps=10, save_strategy="epoch",
        warmup_ratio=0.03, lr_scheduler_type="cosine", report_to=[], seed=args.seed,
        gradient_checkpointing=True, packing=False,
        completion_only_loss=True,  # prompt/completion dataset -> train only on the tool call
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"[sailfish-ft] merged weights -> {args.out}  (convert to GGUF for llama.cpp)")
    else:
        trainer.model.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"[sailfish-ft] LoRA adapter -> {args.out}  (--merge for a full-weight GGUF path)")


if __name__ == "__main__":
    main()
