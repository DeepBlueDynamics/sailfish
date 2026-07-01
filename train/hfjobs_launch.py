#!/usr/bin/env python3
"""
Sailfish :: train — launch finetune_target.py on HF Jobs (MVP backend; GCloud-A100 backend is P4).

Uploads the corpus to your scratch bucket, then runs the LoRA fine-tune on a rented GPU. The job pulls
the corpus, trains, and pushes the adapter/merged weights back to the bucket for download + GGUF convert.

  export HF_TOKEN=hf_...
  python hfjobs_launch.py --data ../data/tool_calls.jsonl --flavor a100-large --bucket <you>/sailfish

Requires the `hf` CLI (huggingface_hub) authenticated with gemma-challenge/bucket write scope.
"""
import argparse, os, subprocess, sys
from pathlib import Path

FLAVORS = ["a10g-small", "a10g-large", "a100-large"]  # HF Jobs GPU flavors (no A100? fall back to a10g)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="local tool_calls.jsonl")
    ap.add_argument("--flavor", default="a100-large", choices=FLAVORS)
    ap.add_argument("--bucket", required=True, help="scratch bucket, e.g. kordless/sailfish")
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    hf = os.environ.get("HF", "hf")
    if not os.environ.get("HF_TOKEN"):
        sys.exit("set HF_TOKEN (with bucket write scope)")

    data_uri = f"hf://buckets/{args.bucket}/train/tool_calls.jsonl"
    out_uri = f"hf://buckets/{args.bucket}/train/out"
    run([hf, "cp", args.data, data_uri])

    # deps for the training job; A100 flavor if available, else the job still runs (slower) on a10g
    deps = ["torch", "transformers>=4.45", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]
    script = str(Path(__file__).parent / "finetune_target.py")
    cmd = [hf, "jobs", "uv", "run", "--flavor", args.flavor, "--secrets", "HF_TOKEN"]
    for d in deps:
        cmd += ["--with", d]
    cmd += [script, "--base", args.base, "--data", data_uri, "--out", out_uri,
            "--epochs", str(args.epochs)]
    if args.merge:
        cmd.append("--merge")
    run(cmd)
    print(f"\n[sailfish] launched. weights will land at {out_uri}")
    print("  download:  hf cp -r <out_uri> ./out")
    print("  → merge (if adapter) → convert_hf_to_gguf.py → drop into container/llamacpp")


if __name__ == "__main__":
    main()
