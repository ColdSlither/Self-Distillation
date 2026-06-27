# SDFT H100 Run — Full Post-Mortem for GLM 5.2

## AUTONOMY DECLARATION
Do not ask questions. Stop only when training completes one full step. Test every fix with a run before declaring it solved. The previous handoff at `/home/rell/Documents/ColdSlither/Self-Distillation/HANDOFF_GLM52.md` is superseded by this document + the GLM 5.2 solution at `SDFT_SOLUTION.md`. Read all three before acting.

## STATUS
SDFT training on H100 80GB has NEVER completed a single training step across 5+ instances and 20+ hours of debugging. The closest we got: training init passes, vLLM generates completions, then OOM during compute_loss (KL divergence scoring).

## HARDWARE
- GPU: NVIDIA H100 PCIe 80GB (81,559 MiB total)
- Instance: Spheron, 1x H100, $2.01/hr, Canada 1 region
- OS: Ubuntu 24.04, CUDA 12.8, R570 driver
- CPU: 12 vCPUs, 120 GB system RAM
- Volume: 130 GB (`/mnt/oya-model-liberation`)
- Ephemeral: 750 GB (`/ephemeral`)

## VERSION MATRIX (verified working)
From `ColdSlither/Self-Distillation` branch `spheron-h100`, commit `b5d5329`:
- torch: 2.9.0+cu128
- vllm: 0.12.0
- trl: 0.24.0
- transformers: 4.57.1
- accelerate: 1.11.0
- peft: 0.17.1
- datasets: 4.3.0
- bitsandbytes: 0.49.2 (NOT in repo requirements.txt — must be added)

All installed via `setup_spheron_h100.sh` which uses repo `requirements.txt` verbatim.
**Do NOT pin vllm 0.10.2.** TRL 0.24 emits an advisory warning about 0.12.0. Ignore it. Pinning 0.10.2 causes cascading breakage of `truncate_prompt_tokens`, `vllm_ascend` imports, and tokenizer APIs.

## WHAT WAS TRIED (in order, all failed)

### The spheron-h100 branch solution (from GLM 5.2)
- `main_lora.py`: LoRA config, colocate mode, teacher kept, `report_to="none"`
- `setup_spheron_h100.sh`: Idempotent, installs repo requirements verbatim
- `smoke_test.sh`: 10 examples, 3 steps, `--num_prompts_per_batch 4`
- `train.sh`: Full run

### Blocker 1: LoRA param shape mismatch
**Error:** `RuntimeError: The size of tensor a (1024) must match the size of tensor b (16)`
**Where:** `MemoryEfficientSyncRefModelCallback._sync_param` in `distil_trainer.py:112`
**Why:** LoRA adds adapter parameters (shape [16, 4096], [1024, 16]) to the student model. The teacher (ref_model) doesn't have these. The sync callback iterates all named_parameters and tries to sync shapes that don't match.
**Fix:** Add shape guard:
```python
if model_param.shape == ref_param.shape:
    ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)
```
**Status:** APPLIED and verified working.

### Blocker 2: expandable_segments incompatible with vLLM memory pool
**Error:** `AssertionError: Expandable segments are not compatible with memory pool.`
**Where:** vLLM 0.12.0 `CuMemAllocator.__init__`
**Why:** The `smoke_test.sh` exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. vLLM's memory pool cannot work with expandable segments.
**Fix:** Remove `PYTORCH_CUDA_ALLOC_CONF` entirely. `PYTORCH_ALLOC_CONF` (without CUDA) is OK but not needed.
**Status:** APPLIED.

### Blocker 3: bitsandbytes not installed
**Error:** Teacher model loaded in bf16 (16 GB) instead of 4-bit (~6 GB). `BitsAndBytesConfig` was set but bitsandbytes was not pip-installed.
**Why:** Repo `requirements.txt` doesn't include bitsandbytes. It was added to `main_lora.py` imports but the package wasn't in the venv.
**Fix:** `pip install bitsandbytes`
**Status:** APPLIED (confirmed working: `python3 -c "from bitsandbytes.nn import Linear4bit; print('OK')"` passes).

### Blocker 4: 78 GB OOM at compute_loss (UNSOLVED)
**Error:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 79.19 GiB of which 964.50 MiB is free.`
**Where:** `distil_trainer.py:815, _get_per_token_logps_and_entropies, logps = torch.cat(all_logps, dim=0)`
**Memory fingerprint: IDENTICAL every run (78.24 GB/79.19 GB, 964.50 MB free) regardless of:**
- `num_prompts_per_batch`: 32 → 8 → 4 → 2 (zero effect)
- `vllm_gpu_memory_utilization`: 0.5 → 0.35 → 0.20 → 0.15 (zero effect)
- `vllm_enable_sleep_mode`: False → True (zero effect)
- Teacher precision: bf16 → 4-bit (zero effect — 4-bit tested working in isolation)
- Tensor shapes when tested: `torch.cat(all_logps, dim=0)` during scoring

**vLLM init logs show:**
- Model loading: 15.27 GiB
- Available KV cache: 23.79 GiB
- GPU KV cache size: 173,184 tokens
- Maximum concurrency: 84.56x at 2048 tokens

**Memory budget breakdown (estimated):**
- vLLM model weights: ~15.3 GB
- vLLM KV cache: ~23.8 GB
- Teacher (4-bit, on GPU): ~6 GB
- Student (moved to GPU by Trainer.__init__): ~15.3 GB
- CUDA context + torch caches: ~5 GB
- Training activations during scoring: ~8-12 GB
- **Total:** ~73-77 GB

**Why the OOM is consistent:**
78.24 GB in use at the `torch.cat` call. The `all_logps` list accumulates per-step log probability tensors. When concatenated, the result needs 1.16 GB more. With only 964 MB free, it fails by ~200 MB.

**Why config changes don't affect memory:**
The memory is dominated by model weights + vLLM infrastructure. Batch size, sequence length, and KV cache utilization have minimal impact because:
1. vLLM pre-allocates KV cache at init time based on free memory, not based on the utilization factor (the setting may be ignored in colocate mode)
2. The student model is always 15.3 GB when moved to GPU
3. Teacher 4-bit is always ~6 GB
4. vLLM model is always ~15.3 GB

### What was NOT tried
1. **Student on CPU** — `device_map="auto"` for the student model. Accelerate would keep active layers on GPU, offload rest to CPU. Previous attempt on A6000 showed this over-offloads, but on H80 it might leave enough layers on GPU.
2. **gradient_checkpointing=True** — already on in the config. But maybe not reaching the HF Trainer.
3. **Disable torch.compile** — vLLM 0.12.0 uses torch.compile by default for its backbone (7.46 seconds in logs). This adds GPU memory overhead for compiled kernels.
4. **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — the non-CUDA version. Conflicting message says `PYTORCH_CUDA_ALLOC_CONF` is deprecated in favor of `PYTORCH_ALLOC_CONF`. Maybe the new version works with vLLM.
5. **`vllm_enable_sleep_mode=True` with `level=0` (not level=1)** — `sleep(level=1)` frees KV cache. `sleep(level=0)` frees everything including model weights. Level 1 might not free enough.
6. **Manually call `self.llm.sleep()` before scoring** — the sleep is called in `_generate_single_turn` AFTER the `_generate` call returns. But `_compute_loss` is called after that. Maybe the sleep IS happening but releasing memory is async and takes time.
7. **Use `generate_from_teacher=True`** — vLLM uses teacher model instead of loading its own copy. Saves one 15 GB model load. But teacher is 4-bit, and vLLM expects bf16.
8. **Don't use vLLM at all** — set `use_vllm=False` in the config. The trainer falls back to `model.generate()` for generation. Slower but no second model copy.
9. **Single A100 80GB** — different architecture, different memory layout. Might work where H100 doesn't.

## THE ROOT PROBLEM
Three model copies on a single 80 GB card:
- Student (training): 16 GB
- Teacher (reference): 6 GB 4-bit
- vLLM (generation): 15 GB + 24 GB KV cache
- Total: ~61 GB just for models + infrastructure
- Remaining for training: ~18 GB
- But `torch.cat` at scoring time needs 1.16 GB more than available (964 MB free)

This is NOT a batch-size or sequence-length problem. It's a model-count problem.

## FILES AND COMMANDS
**Repo location:** `/home/rell/Documents/ColdSlither/Self-Distillation/` (branch `spheron-h100`)
**H100 instance:** `ssh ubuntu@69.19.137.64`
**Training location on H100:** `/home/ubuntu/sdft-training/`
**Smoke test:** `bash /home/ubuntu/smoke_test.sh`
**Full train:** `bash /home/ubuntu/train.sh`

**Main files (key lines):**
- `main_lora.py:119` — student model load (needs device_map)
- `main_lora.py:126` — teacher model load (4-bit with BitsAndBytesConfig)
- `main_lora.py:153` — vllm_gpu_memory_utilization (0.15)
- `main_lora.py:154` — vllm_enable_sleep_mode (True)
- `distil_trainer.py:112` — LoRA param shape guard (PATCHED)
- `distil_trainer.py:1067-1070` — sleep/wake for generation
- `distil_trainer.py:815` — THE OOM SITE: `torch.cat(all_logps, dim=0)`

## PATCHES APPLIED (to spheron-h100 branch on H100)
1. `distil_trainer.py:112` — shape guard on `_sync_param` (indentation must be exact: 8 spaces for docstring, 8 for if, 12 for body)
2. `smoke_test.sh` and `train.sh` — removed `expandable_segments:True`
3. `main_lora.py:22` — BitsAndBytesConfig import (already in repo)
4. `main_lora.py:126` — teacher 4-bit via BitsAndBytesConfig
5. `main_lora.py:153` — vllm_gpu_memory_utilization set to 0.15 (had no effect)

## ACCEPTANCE CRITERIA (still unmet)
1. Training init completes without hanging — PARTIAL (init passes, generation works)
2. Model weights sync from training to vLLM — UNTESTED (fails before this)
3. First training step completes — FAILED (OOM at compute_loss)
4. Training log written to output directory — FAILED
5. All 252 steps complete — FAILED
6. Checkpoint resume — FAILED
7. Coherent completions — FAILED

## INSTANCE RUNNING
H100 is live at 69.19.137.64 ($2.01/hr). Smoke test has been run ~8 times with different configs. No training steps completed.
