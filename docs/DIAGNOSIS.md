# SDFT — Root Cause Across Both Walls (v2)

> Two distinct walls were hit. Both are now diagnosed. This doc is the single source
> of truth. Read it before touching any instance.
> **Current solution:** branch `spheron-h100`, `use_vllm=False` path.

---

## Wall 1 (solved) — The version/server-mode trap

The first ~15 hours of failure came from **installing the wrong library versions on every
Spheron instance, then patching the symptoms.** Five of six cataloged "patches" were fixes
for bugs that only exist with wrong vllm/transformers. The repo's own `requirements.txt` is
a known-good matrix, and a local `venv/` proves it imports cleanly.

The "2-GPU + server mode + NCCL" attack chased a problem **that does not exist in colocate
mode** — colocate is the repo default and never calls `init_communicator`.

### Verified version matrix (from `venv/` — known good)

| Package | Wrong (handoff) | Repo `requirements.txt` | Proven in venv |
|---------|-----------------|-------------------------|----------------|
| torch | 2.5.1 | 2.9.0 | **2.9.0+cu128** |
| vllm | 0.10.2 | 0.12.0 | **0.12.0** |
| trl | 0.24.0 | 0.24.0 | **0.24.0** |
| transformers | (→5.x) | 4.57.1 | **4.57.1** |

TRL 0.24 warns it "only supports vllm 0.10.2" — **ignore it.** Pinning 0.10.2 is what
*caused* the `truncate_prompt_tokens`/`vllm_ascend`/tokenizer breakage. 0.12.0 works.

### Patch verdicts

| Patch | Verdict |
|-------|---------|
| transformers==4.57.1 | REAL — but repo already pins it. Install repo reqs. |
| vllm_ascend guard | NOT NEEDED — installed source already guards it |
| warnings_issued guard | NOT NEEDED — transformers 4.57.1 provides the attr |
| init_communicator/NCCL/2-GPU | AVOID — colocate never calls it |
| truncate_prompt_tokens removal | HARMFUL — installed client supports it, trainer uses it |
| report_to="none" | REAL (minor) — kept |

---

## Wall 2 (solved) — The 80 GB OOM at compute_loss

After the version/server-mode trap was cleared, the H100 run hit a **genuinely different,
structural wall.** The signature: an OOM at `torch.cat(all_logps)` in `_get_per_token_logps_and_entropies`
(`distil_trainer.py:814`) with a fingerprint **identical every run (78.24 GB / 964 MB free)
regardless of batch size, `vllm_gpu_memory_utilization`, sleep mode, or teacher precision.**

When *nothing* moves memory, it's not tuning — it's a structural floor.

### The mechanism (verified against code)

1. **vLLM colocate pre-allocates its full KV cache at init (~24 GB) and ignores
   `vllm_gpu_memory_utilization` in colocate mode.** That's why turning the knob did nothing.
2. Resident set at the moment of OOM: vLLM weights ~15 GB + vLLM KV ~24 GB + student ~16 GB +
   teacher ~6-16 GB + CUDA ~5 GB = **~66-76 GB locked.**
3. The forward-KL loss (`distil_trainer.py:1659`) materializes a **full-vocabulary
   log-softmax** over Qwen3's 151,936 vocab for every completion token, for **both** student
   and teacher simultaneously: `[B, completion_len, 151936]`. At batch 2 / 1024 tokens that's
   ~1.24 GB/model in fp32, ~3-5 GB peak transient total (incl. `kl_div`).
4. With only 964 MB free, the `torch.cat` that assembles these needs ~1 GB more → **OOM by
   ~200 MB, every time.**

### Why every "not tried" option except use_vllm=False was a dead end

| Option | Verdict |
|--------|---------|
| student device_map="auto" | Dead — PCIe offload of the loss forward, 10-50× slower |
| PYTORCH_ALLOC_CONF (non-CUDA) | Dead — out of *total* memory, not fragmentation |
| sleep(level=0) | Weak — frees vLLM weights but reload every step kills throughput |
| generate_from_teacher=True | Dead — teacher is 4-bit, vLLM wants bf16; doesn't touch OOM tensor |
| **use_vllm=False** | **FIX** — drops ~39 GB (vLLM weights + KV). HF `model.generate()` instead. |
| H200 141GB | Also a fix — but costs ~2x; use_vllm=False keeps the $2/hr H100 |
| reduce max_completion_length | Partial — halves the OOM tensor; helps, doesn't fully solve 80 GB |

**The fix chosen:** `use_vllm=False`. It's the repo author's own escape hatch — the
non-vLLM generation path (`distil_trainer.py:1236-1270`) is a complete first-class branch,
not a stub. Generation is slower (HF vs vLLM), but a full training step completes.

---

## What's committed on `spheron-h100`

1. **`distil_trainer.py`** — `_sync_param` shape guard. With LoRA, the student has adapter
   params the teacher lacks; the ref-sync callback crashes on shape mismatch without this.
   (Postmortem found this as "Blocker 1"; it was never committed before — now it is, so it
   never needs re-patching.)
2. **`main_lora.py`** — `use_vllm=False`, LoRA, teacher kept (load-bearing), `report_to="none"`,
   `gradient_checkpointing=True`, `--smoke_test` mode, `--max_completion_length` (default 512)
   to keep the former-OOM tensor small even as insurance.
3. **`setup_spheron_h100.sh`** — idempotent; installs repo reqs verbatim + bitsandbytes;
   verifies the version matrix; generates `train.sh` + `smoke_test.sh` with NO
   `expandable_segments` (postmortem blocker #2: incompatible with the memory pool) and NO
   NCCL env (no server mode).

---

## Path forward

1. **H100 80GB on Spheron** (~$2/hr).
2. `git clone -b spheron-h100 … && bash setup_spheron_h100.sh`
3. `bash ~/smoke_test.sh` — proves init → HF generate → compute_loss (former OOM site) →
   checkpoint. If this passes, the wall is broken.
4. `bash ~/train.sh` — full run, only after smoke passes.
5. **Snapshot the instance** after setup.

## If it still fails

- *OOM again at compute_loss* → lower `--max_completion_length` (256) or batch. The tensor
  scales linearly with completion length.
- *Shape mismatch in sync* → the guard is in place; if it still fires, the LoRA target modules
  diverge from teacher — check `target_modules`.
- *Import/AttributeError* → version drift. Diff `pip freeze` against the matrix above.
- *Hang* → impossible under `use_vllm=False` (no NCCL). If it hangs, something re-enabled vLLM.
