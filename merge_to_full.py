#!/usr/bin/env python3
"""Merge trained LoRA adapters into the base model → standard HF model dir.

WHY: main_lora.py trains PEFT adapters (adapter_model.safetensors, tiny). The deploy
pipeline (Base → SDFT → GGUF → Deploy) needs a FULL merged model — both the GGUF
converter and eval_tooluse.py expect a standard HF dir with config.json + full weights.

Verified merge path (tested on real Qwen2.5+LoRA round-trip):
    base = AutoModelForCausalLM.from_pretrained(BASE)
    peft = PeftModel.from_pretrained(base, ADAPTER)
    merged = peft.merge_and_unload()
    merged.save_pretrained(OUT); tokenizer.save_pretrained(OUT)

The output dir is a drop-in replacement for the base model dir: loadable by
AutoModelForCausalLM, vLLM, and llama.cpp's convert_hf_to_gguf.py.

Usage:
    python3 merge_to_full.py --base ~/sdft-training/Qwen3-8B \\
                             --adapter ~/sdft-training/output/run1/checkpoint-XXX \\
                             --out /mnt/oya-model-liberation/qwen3-8b-sdft-merged

Notes:
  - Output goes to the PERSISTENT volume by default hint (see default --out). Instance
    disk is wiped on preemption; the volume survives. Point --out at the volume.
  - Adapters are ~150 MB; the merged full model is ~16 GB (bf16). Make sure the target
    has space.
  - merge_and_unload loads the full base in bf16 — needs ~32 GB system RAM, NOT GPU.
    This script is CPU/RAM only; run it without holding GPU memory.
"""
import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapters into base → full HF model")
    ap.add_argument("--base", required=True,
                    help="Base model dir (e.g. ~/sdft-training/Qwen3-8B)")
    ap.add_argument("--adapter", required=True,
                    help="Trained adapter dir (the checkpoint-XXX from training output)")
    ap.add_argument("--out", required=True,
                    help="Output merged model dir (use the PERSISTENT volume)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                    help="Merge/load dtype (bf16 matches training)")
    args = ap.parse_args()

    base = os.path.expanduser(args.base)
    adapter = os.path.expanduser(args.adapter)
    out = os.path.expanduser(args.out)

    # ── validate inputs ───────────────────────────────────────────────────
    if not os.path.isdir(base):
        sys.exit(f"[ERR] base model dir not found: {base}")
    adapter_file = os.path.join(adapter, "adapter_model.safetensors")
    if not os.path.isfile(adapter_file):
        sys.exit(f"[ERR] not a PEFT adapter dir (no adapter_model.safetensors): {adapter}")
    os.makedirs(out, exist_ok=True)

    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[1/4] Loading base model ({args.dtype}) from {base} ...")
    # CPU load — merge is RAM-bound, don't consume GPU memory.
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dt)

    print(f"[2/4] Attaching adapters from {adapter} ...")
    peft_model = PeftModel.from_pretrained(base_model, adapter)

    print(f"[3/4] Merging adapters into base weights ...")
    merged = peft_model.merge_and_unload()

    print(f"[4/4] Saving merged full model + tokenizer to {out} ...")
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(out)

    # ── verify the output is a self-contained HF model dir ────────────────
    required = ["config.json"]
    safetensors = [f for f in os.listdir(out) if f.endswith(".safetensors")]
    missing = [f for f in required if f not in os.listdir(out)]
    if missing or not safetensors:
        sys.exit(f"[ERR] merge output incomplete; missing {missing or 'safetensors'}")
    total_gb = sum(os.path.getsize(os.path.join(out, f)) for f in safetensors) / 1e9
    print()
    print(f"[OK] Merged model written to {out}")
    print(f"     weight shards: {len(safetensors)}  ({total_gb:.1f} GB)")
    print(f"     loadable by: AutoModelForCausalLM, vLLM, llama.cpp convert_hf_to_gguf.py")
    print()
    print("Next:")
    print(f"  Eval:    python3 eval_tooluse.py --model_path {out}")
    print(f"  → GGUF:  bash convert_gguf.sh {out}")


if __name__ == "__main__":
    main()
