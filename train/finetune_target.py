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

Data: one JSON object per line with {context:[{role,text}], tool, arguments}. Each becomes a chat where
the assistant turn is the tool call — we mask the prompt and train only the completion.

Output: a LoRA adapter (+ optional merged weights). Convert to GGUF for llama.cpp with llama.cpp's
convert_hf_to_gguf.py after merging; keep as adapter for vLLM.
"""
import argparse, json, os
from pathlib import Path


def load_examples(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("tool"):
            continue
        ctx = r.get("context") or []
        user_msgs = [c["text"] for c in ctx if c.get("role") == "user" and c.get("text")]
        if not user_msgs:
            continue
        # completion = the tool call, rendered as the assistant should emit it
        call = {"name": r["tool"], "arguments": r.get("arguments") or {}}
        rows.append({"user": user_msgs[-1][:2000], "assistant": json.dumps(call, ensure_ascii=True)})
    return rows


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

    rows = load_examples(args.data)
    print(f"[sailfish-ft] {len(rows)} tool-call examples from {args.data}")

    def to_text(r):
        # gemma chat: user asks, assistant emits the tool call. train on the completion.
        msgs = [{"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]}]
        return tok.apply_chat_template(msgs, tokenize=False)

    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

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
        warmup_ratio=0.03, lr_scheduler_type="cosine", report_to=[],
        gradient_checkpointing=True, packing=False,
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
