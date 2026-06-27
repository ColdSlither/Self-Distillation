"""SDFT training entry point — LoRA + use_vllm=False on a single H100 80GB.

WHY use_vllm=False (read H100_POSTMORTEM.md + docs/DIAGNOSIS.md):
  On a single 80 GB card, vLLM colocate mode pre-allocates its full KV cache at init
  (~24 GB) and ignores vllm_gpu_memory_utilization in colocate mode. Combined with three
  model copies (student 16 GB + teacher ~16 GB + vLLM weights 15 GB), the resident set
  hits ~66-76 GB. The forward-KL loss then materializes a full-vocab log-softmax
  ([B, completion_len, 151936]) for BOTH student and teacher at compute_loss
  (distil_trainer.py:1659), needing ~3-5 GB of peak transient that has nowhere to go
  → consistent 78.24 GB OOM at torch.cat(all_logps) regardless of batch/utilization.

  use_vllm=False drops the vLLM weight copy AND its pre-allocated KV cache (~39 GB
  freed). Generation uses HF model.generate() instead — slower, but the OOM is gone
  and a full training step completes. This is the repo author's own escape hatch:
  the non-vLLM generation path (distil_trainer.py:1236-1270) is a complete first-class
  branch, not a stub.

Required repo change (committed, not re-patched per instance):
  distil_trainer.py _sync_param shape guard — LoRA adapter params have shapes the
  teacher lacks; the ref-model sync callback would crash without the guard.

Run via:
  python3 main_lora.py --model_name <path> --output_dir <dir>            # full run
  python3 main_lora.py --model_name <path> --output_dir <dir> --smoke_test
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
    p = argparse.ArgumentParser(description="SDFT LoRA, use_vllm=False (H100 single-GPU)")
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
    # ── completion length: the OOM tensor scales with this. 512 halves it vs 1024. ──
    p.add_argument("--max_completion_length", type=int, default=512,
                   help="Lower = smaller full-vocab logps tensor at compute_loss (the former OOM site).")
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
    # Tiny subset, short completions, few steps. Proves the WHOLE pipeline
    # (init -> HF generate -> forward/backward -> checkpoint) for ~$0.20.
    if args.smoke_test:
        smoke_limit = 8
        batch = 1
        accum = 1
        epochs = 1
        max_comp = 256          # short completions, tiny OOM tensor
        print("=" * 64)
        print("  SMOKE TEST (use_vllm=False): 8 examples, short completions, 3 steps.")
        print("  Verifies init -> model.generate() -> compute_loss -> checkpoint.")
        print("=" * 64)
    else:
        smoke_limit = None
        batch = 1
        accum = args.num_prompts_per_batch
        epochs = args.num_train_epochs
        max_comp = args.max_completion_length

    # ── MODELS ────────────────────────────────────────────────────────────
    # use_vllm=False: NO vLLM weight copy, NO pre-allocated KV cache. Resident set
    # is just student + teacher (~32 GB) + activations, leaving ~40 GB headroom for
    # the full-vocab KL loss tensor that OOM'd at 80 GB under vLLM.
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
        use_vllm=False,                   # <-- THE FIX. Drops ~39 GB. Generation via HF model.generate().
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
        save_steps=100,
        max_grad_norm=1,
        gradient_checkpointing=True,
        report_to="none",                 # no wandb login
        output_dir=args.output_dir,
        log_completions=args.smoke_test,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        # vllm_importance_sampling_correction is irrelevant under use_vllm=False;
        # the trainer guards it (distil_trainer.py:1438,1454,1678 all key on use_vllm).
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,          # SDFT teacher — KEPT (load-bearing for forward-KL)
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
