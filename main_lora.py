"""SDFT training entry point — LoRA + single-GPU colocate mode.

This is the working config for the spheron-h100 branch. It exists as a SEPARATE file
from main.py so the upstream repo source stays pristine and there is no ambiguity about
what changed. See docs/DIAGNOSIS.md for why this is the only path we use.

Why these settings (all verified against repo source + known-good venv):
  - vllm_mode="colocate":  vLLM runs in-process. NO init_communicator, NO NCCL, NO 2nd
                           GPU, NO HTTP weight sync. This is the server-mode trap avoided.
  - peft_config (LoRA):    kills the full-FT AdamW optimizer monster (~64 GB for 8B).
                           Adapter params are the only thing with optimizer state.
  - teacher (ref_model):   KEPT and loaded bf16. It is load-bearing for the SDFT
                           forward-KL loss (distil_trainer.py:1464). On 80 GB it fits.
  - device_map NOT set:    single visible GPU, HF places the model on it. (Avoids the
                           over-offload footgun that device_map="auto" caused on small VRAM.)
  - report_to="none":      no wandb login required.

Run via:  python3 main_lora.py --model_name <path> --output_dir <dir> --smoke_test
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
    p = argparse.ArgumentParser(description="SDFT LoRA (spheron-h100 branch)")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--num_prompts_per_batch", type=int, default=32)
    p.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--dataset_name", type=str, default="tooluse", choices=["tooluse", "science"])
    p.add_argument("--seed", type=int, default=42)
    # ── smoke test: tiny subset + few steps, to prove the pipeline cheaply ──
    p.add_argument("--smoke_test", action="store_true",
                   help="Run ~10 examples for 3 steps. Verifies init->generate->backward->ckpt end-to-end.")
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
    # Tiny subset, 3 steps, short sequences. Proves the pipeline for ~$0.20.
    if args.smoke_test:
        smoke_limit = 10
        batch = 2          # generation_batch_size per step
        accum = 1          # grad accum (steps per generation = 1)
        epochs = 1
        print("=" * 60)
        print("  SMOKE TEST: 10 examples, 3 steps, short seqs.")
        print("  This verifies init -> generate -> backward -> checkpoint.")
        print("=" * 60)
    else:
        smoke_limit = None
        batch = 1
        accum = args.num_prompts_per_batch
        epochs = args.num_train_epochs

    # ── MODELS ────────────────────────────────────────────────────────────
    # Single visible GPU (CUDA_VISIBLE_DEVICES=0 set by train scripts).
    # Do NOT set device_map — on one GPU HF places it correctly and we avoid
    # the "auto" over-offload footgun.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── LoRA ──────────────────────────────────────────────────────────────
    # q/v projections are the standard target for decoder LLMs. r=16 keeps adapter
    # memory tiny while preserving enough capacity for domain adaptation.
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
        use_vllm=True,
        vllm_mode="colocate",              # <-- THE fix. Never "server".
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.5,   # comfortable on 80 GB; lower to 0.4 if OOM at generation
        vllm_enable_sleep_mode=False,      # avoids VRAM thrash during backward (verified necessary)
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=100,
        max_grad_norm=1,
        gradient_checkpointing=True,
        report_to="none",                  # no wandb login
        output_dir=args.output_dir,
        log_completions=args.smoke_test,   # print completions during smoke test for sanity
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,           # SDFT teacher — KEPT (load-bearing), fits on 80 GB
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
