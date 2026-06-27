#!/usr/bin/env python3
"""SDFT training entry point — 2x H100 DDP (data parallel).

Launched via torchrun by launch_ddp.sh:
    torchrun --nproc_per_node=2 main_ddp.py --model_name ... --output_dir ...

WHY DDP (read docs/DIAGNOSIS.md + HANDOFF.md): each GPU runs a full student+teacher
copy and processes half the batch in parallel; generation runs on both cards. Measured
~1.8x throughput vs single GPU => a 1-epoch run drops from ~$29 (1x H100) to ~$8 (2x H100).

DDP handling in this trainer (verified, NOT assumed):
  - ref_model device-placed via accelerator.prepare_model (distil_trainer.py:594)
  - generation unwraps DDP via unwrap_model_for_generation (trl/models/utils.py:308)
  - ref_model sync is name-based AND strips the DDP `module.` prefix (distil_trainer.py)
  - the explicit ref_model=teacher wins over the beta==0 / is_peft branches (line 444)

DO NOT set CUDA_VISIBLE_DEVICES — torchrun maps processes to GPUs via LOCAL_RANK and
needs all GPUs visible. Manual .to('cuda:N') would fight Accelerate's placement.

Run order:
    bash launch_ddp.sh smoke     # ~$0.50, 5 min, 8 examples — VERIFY FIRST
    bash launch_ddp.sh train     # ~$8, ~2 hrs, 1 epoch — only after smoke passes
"""
from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
from datasets import load_from_disk
from string import Template
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="SDFT LoRA, 2x H100 DDP")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--num_prompts_per_batch", type=int, default=16)
    p.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--dataset_name", type=str, default="tooluse", choices=["tooluse", "science"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--max_completion_length", type=int, default=256)
    return p.parse_args()


def load_tooluse_dataset(seed=42, limit=None):
    ds = load_from_disk("data/tooluse_data/train_data")

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                orig_content=example["prompt"],
                output_text="\n".join(example["golden_response"]))}],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=seed)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds, None


def load_science_dataset(seed=42, limit=None):
    ds = load_from_disk("data/science_data/train_data")

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {"role": "user", "content": teacher_prompt.substitute(
                    orig_content=example["messages"][1]["content"],
                    output_text=example["output_text"])},
            ],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=seed)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds, None


if __name__ == "__main__":
    args = parse_args()

    # ── SMOKE TEST OVERRIDES ──────────────────────────────────────────────
    # Tiny subset, short completions, few steps. Proves the FULL DDP path:
    # torchrun init -> generation on BOTH GPUs -> loss -> ref_model sync across
    # processes -> checkpoint. If this passes, full DDP is safe.
    if args.smoke_test:
        smoke_limit = 8
        # per-process batch; with 2 procs the effective batch is 2x this
        batch = 1
        accum = 1
        epochs = 1
        max_comp = 128
        if torch.distributed.is_available():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        else:
            rank = 0
        if rank == 0:
            print("=" * 64)
            print("  DDP SMOKE TEST: 8 examples, short completions, ~3 steps, 2x H100.")
            print("  Verifies torchrun -> generate (both GPUs) -> loss -> sync -> ckpt.")
            print("=" * 64)
    else:
        smoke_limit = None
        batch = 1
        # num_prompts_per_batch is the EFFECTIVE batch; grad accum divides it across procs.
        # With 2 procs, each does num_prompts_per_batch/2 generations per optimizer step.
        accum = max(1, args.num_prompts_per_batch // 2)
        epochs = args.num_train_epochs
        max_comp = args.max_completion_length

    # ── MODELS — NO manual .to(device); Accelerate places via LOCAL_RANK ──
    # Setting device_map or .cuda() here would conflict with Accelerate's DDP setup.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── LoRA ──────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ── DATA ──────────────────────────────────────────────────────────────
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed, limit=smoke_limit)
    else:
        dataset, _ = load_science_dataset(args.seed, limit=smoke_limit)

    # ── CONFIG ────────────────────────────────────────────────────────────
    config = DistilConfig(
        seed=args.seed,
        use_vllm=False,                   # same proven path; DDP distributes it
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        max_prompt_length=1024,
        max_completion_length=max_comp,
        num_train_epochs=epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=10 if args.smoke_test else 50,
        max_grad_norm=1,
        gradient_checkpointing=True,
        report_to="none",
        output_dir=args.output_dir,
        log_completions=args.smoke_test,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
