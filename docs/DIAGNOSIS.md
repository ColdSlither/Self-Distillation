# Why Training Kept Failing — Root Cause & The Exit

> Read this BEFORE touching any Spheron instance. Every claim below was verified
> against the repo source and the known-good local `venv/` on 2026-06-26.

## TL;DR (the one-paragraph version)

The training loop was caused by **installing the wrong library versions** on every
Spheron instance, then applying per-instance patches for the symptoms of those wrong
versions. Five of the six cataloged "patches" are fixes for bugs that **only exist with
the wrong vllm/transformers**. The repo's own `requirements.txt` already contains a
known-good version set, and a working `venv/` exists locally proving those versions
import cleanly. The "2-GPU + server mode + NCCL" line of attack in the handoff chases a
problem that **does not exist in colocate mode** — and colocate is what the repo ships by
default. We are abandoning server mode permanently.

---

## Verified version matrix (from `venv/` — known good)

| Package     | Handoff/scripts said | Repo `requirements.txt` | LOCAL VENV (proven) |
|-------------|----------------------|-------------------------|---------------------|
| torch       | 2.5.1                | 2.9.0                   | **2.9.0+cu128**     |
| vllm        | 0.10.2               | 0.12.0                  | **0.12.0**          |
| trl         | 0.24.0               | 0.24.0                  | **0.24.0**          |
| transformers| (unpinned → 5.x)     | 4.57.1                  | **4.57.1**          |
| accelerate  | 1.2.1                | 1.11.0                  | **1.11.0**          |
| peft        | 0.14.0               | 0.17.1                  | **0.17.1**          |
| datasets    | 3.2.0                | 4.3.0                   | **4.3.0**           |

`venv/bin/python -c "import trl, vllm, transformers, torch"` → **IMPORT OK**.
This is the target state for every Spheron instance.

### The known mismatch (be aware, don't "fix" it)

TRL 0.24 emits an advisory warning at import:
> "TRL currently only supports vLLM version 0.10.2. You have version 0.12.0 installed."

This is **baked into the upstream repo** — its own `requirements.txt` pins vllm 0.12.0 while
TRL 0.24 nominally wants 0.10.2. It is the seed of the entire debugging loop: every prior
attempt "fixed" the warning by pinning vllm 0.10.2, which then *actually broke* things
(`truncate_prompt_tokens`, `vllm_ascend` import, tokenizer API) — which then got "patched"
one by one. **Do not pin vllm 0.10.2.** The 0.12.0 + 0.24.0 combo imports cleanly,
instantiates `DistilConfig`/`DistilTrainer` cleanly, and the installed `vllm_client.py`
already supports the 0.12 API surface. The warning is advisory and harmless. Treat it as
expected output, not an error.

## The six "patches" — verdict on each

| # | Patch                    | Verdict     | Why                                                                 |
|---|--------------------------|-------------|---------------------------------------------------------------------|
| 1 | transformers==4.57.1     | **REAL**    | transformers 5.x removed `all_special_tokens_extended`. Repo already pins 4.57.1 in `requirements.txt`. **Fix = install the repo's requirements, not add a patch.** |
| 2 | vllm_ascend import guard | **NOT NEEDED** | The installed `trl/extras/vllm_client.py:37-42` already guards it correctly inside `if is_vllm_ascend_available():`. The bug only appears with the wrong vllm. |
| 3 | warnings_issued guard    | **NOT NEEDED** | Verified `hasattr(model, "warnings_issued")` → True on real Qwen2.5 with transformers 4.57.1. The crash only happens on transformers 5.x. |
| 4 | gpu_memory_utilization   | context-only | Real tuning, but only painful at the 47 GB knife-edge. On 80 GB (H100) the default is comfortable. |
| 5 | init_communicator dance  | **AVOID ENTIRELY** | Only exists in `vllm_mode="server"`. Colocate mode (`distil_trainer.py:461-513`) never calls it. The whole Instance-4 saga is server-mode-only. |
| 6 | report_to="none"         | REAL (minor) | Keep `report_to="none"` and `WANDB_MODE=disabled`. |
| (extra) | truncate_prompt_tokens removal | **HARMFUL — DO NOT** | The installed `vllm_client.py:185,254` **supports** `truncate_prompt_tokens`. The trainer relies on it (`distil_trainer.py:1109,1142`). The old `setup_spheron.sh` patch that removes it breaks truncation. Remove that patch. |

## Why server mode is the trap

`distil_trainer.py` has two branches:

- **`vllm_mode="server"`** (lines 452-459): launches a *separate* vLLM process (2nd model
  copy, ~16 GB), requires `VLLMClient`, `init_communicator()`, NCCL broadcast for weight
  sync (`update_named_param`). This is where every hang / HTTP 500 / OOM came from.
- **`vllm_mode="colocate"`** (lines 461-513): vLLM runs **in-process**. One model copy.
  Weight sync is a direct `llm_model.load_weights(...)` call — no NCCL, no HTTP, no second
  GPU. This is the repo default and the only mode we will use.

The handoff's own Session-2 post-mortem (`2026-06-26-session2.md`) states:
> "I changed vllm_mode='colocate' to 'server'. This was the WRONG fix."

…then the handoff's "EXACT FIX" section prescribes server mode anyway. **That
contradiction is the loop.** We resolve it: **colocate only, single GPU, forever.**

## Memory reality for Qwen3-8B (why the card matters)

LoRA + colocate on one card (teacher is the SDFT ref model — it is load-bearing and stays):

| Component                       | VRAM    |
|---------------------------------|---------|
| Student base, frozen bf16        | ~16 GB  |
| Teacher (ref), forward-only bf16 | ~16 GB  |
| vLLM colocate (weights + KV)     | ~16-22 GB |
| LoRA params + AdamW (small)      | ~1-2 GB |
| Activations (grad ckpt on)       | ~4-8 GB |
| **Total**                        | **~50-60 GB** |

- **A6000 47 GB**: ~1-10 GB short → OOM at a *different* point each run → looks like
  "I just need a bigger setting." This is the false signal that drove the 2-GPU idea.
- **H100 80 GB (~$2/hr on Spheron)**: ~20-30 GB headroom. Comfortable. This is the card
  the repo was designed for (README: "All experiments can be run with a single H200").

**We are not fighting 47 GB anymore. Rent the H100.**

## The exit (what this branch implements)

1. `requirements.txt` (repo) is the source of truth — install it verbatim.
2. `main_lora.py` — LoRA config (`peft_config`), `vllm_mode="colocate"`, single GPU, no
   teacher offload needed on 80 GB, `report_to="none"`.
3. `setup_spheron_h100.sh` — idempotent, installs repo requirements (no version drift),
   downloads Qwen3-8B, writes the launch script. **No source patching** — the repo source
   is correct as-shipped.
4. `smoke_test.sh` — ~10 examples, 3 steps, <5 min, ~$0.20. Proves the whole pipeline
   (init → generate → backward → checkpoint) before any full paid run.
5. `train.sh` — full run launcher with the right env vars.

## If it still fails: the verification loop

Run, read the **full** error, classify:
- *Import/AttributeError* → version drift. Diff `pip freeze` against the matrix above.
- *OOM at generation* → lower `vllm_gpu_memory_utilization` (0.5 → 0.4).
- *OOM at backward* → enable `gradient_checkpointing=True` (already on), lower batch.
- *Hang* → you're in server mode. Switch to colocate.

**Do not add new one-off patches.** If a fix is needed, it goes in the repo on this branch
and is committed, so the next instance inherits it automatically. That is how the loop ends.
