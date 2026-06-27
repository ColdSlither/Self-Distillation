# ColdSlither SDFT Training — Handoff (v3, authoritative)

> **This document supersedes** `HANDOFF_GLM52.md`, `H100_POSTMORTEM.md`, and the
> narrative in `SDFT_SOLUTION.md`. If any of those conflict with this one, **this one wins**
> — it was written after verifying every claim against the actual repo source and a
> known-good local venv.
> Branch: `spheron-h100` (through commit `2758bf3`) in `ColdSlither/Self-Distillation`.

---

## AUTONOMY DECLARATION (read first)

Do not ask questions. Do not stop early. Do not re-derive anything that is already solved and
committed. Two structural walls and one silent correctness bug have already been diagnosed and
fixed on this branch — your job is to **run it and verify**, not to re-debug.

Before you act, read these in order, then trust them:
1. This file.
2. `docs/DIAGNOSIS.md` — both walls, with verified mechanisms.
3. `docs/BLOCKERS.md` — every potential failure point, classified FIXED / NON-ISSUE / RESIDUAL.

Rules:
- **Do not edit `vllm_client.py`** (the installed package). It is correct for vllm 0.12.0. Every
  prior "patch" to it fixed symptoms of the wrong vllm version. Obsolete.
- **Do not pin `vllm==0.10.2`.** TRL 0.24 warns it "only supports 0.10.2" — ignore the warning.
  Pinning 0.10.2 is the root cause of the entire first debugging loop. The repo pins 0.12.0; keep it.
- **Do not switch to `vllm_mode="server"`** and do not reintroduce 2-GPU / NCCL /
  `init_communicator`. Server mode is the trap; colocate / no-vLLM is the exit. Every instance
  that chased server mode circled.
- **Do not add one-off patches to a live instance.** If a real fix is needed, commit it to the
  `spheron-h100` branch so the next instance inherits it. Per-instance patching is why fixes kept
  respawning.
- **Test before you declare.** A crash being gone is not the same as the code being correct — the
  LoRA sync bug (below) "passed" two prior sessions because they only checked that the crash stopped.

The success criterion is concrete: **`bash ~/smoke_test.sh` completes one full training step and
writes a checkpoint.** Not "init passes." Not "generation works." One full step: generate →
compute_loss → on_step_end sync → checkpoint. Until that happens, nothing is solved.

---

## PROJECT GOAL (unchanged)

Train Qwen3-8B via Self-Distillation Fine-Tuning (SDFT) on adult domain data (Literotica, 243
stories) using the ColdSlither fork of `Continual-Intelligence/Self-Distillation`. The end goal is
a complete NC-17 model — reasoning, coding, tool calling preserved, trained on adult material via
on-policy self-distillation. This is NOT abliteration; it is fine-tuning with on-policy
self-distillation recovery.

- **Repo:** https://github.com/ColdSlither/Self-Distillation (branch `spheron-h100`)
- **Pipeline:** Base → SDFT recovery (on-policy self-distillation) → Quantize (GGUF) → Deploy
- **Target:** Qwen3-8B (8B, 36 layers, 4096 hidden, 151936 vocab) — text-only (no vision to preserve)
- **Hardware:** Spheron H100 80GB (~$2/hr)
- **Data:** tooluse dataset (4046 examples, pre-packaged) + Literotica (243 stories)

---

## CURRENT STATE — what is solved

Three things are fixed and committed. Do not redo them.

### Fix 1 — the version/server-mode trap (`b5d5329`)
The first ~15 hrs of failure came from installing the wrong library versions every instance, then
patching the symptoms. The repo's own `requirements.txt` is the known-good matrix:

| Package | Wrong (prior) | Repo / proven |
|---|---|---|
| torch | 2.5.1 | **2.9.0** |
| vllm | 0.10.2 | **0.12.0** |
| trl | 0.24.0 | **0.24.0** |
| transformers | (→5.x) | **4.57.1** |

5 of the 6 cataloged "patches" (vllm_ascend guard, warnings_issued guard, truncate_prompt_tokens
removal, etc.) were fixes for bugs that only exist with the wrong versions. **They are obsolete.
Do not re-apply them.** Server mode / 2-GPU / NCCL chases a problem that does not exist in
colocate mode. Abandoned permanently.

### Fix 2 — the structural 80GB OOM (`a86f069`)
The H100 run hit a different wall: a consistent 78.24 GB OOM at `torch.cat(all_logps)`
(`distil_trainer.py:814`), identical across batch size / `vllm_gpu_memory_utilization` / sleep mode /
teacher precision. Verified mechanism: vLLM colocate pre-allocates its full ~24 GB KV cache
(knob ignored in colocate) + three model copies; the forward-KL loss then materializes a full-vocab
log-softmax (`[B, completion_len, 151936]`) for both student and teacher at once.
**Fix: `use_vllm=False`.** Drops ~39 GB (vLLM weights + KV). Generation uses HF
`model.generate()` — a complete first-class branch in the trainer (`distil_trainer.py:1236-1270`),
the repo author's own escape hatch. Slower generation, but a full step completes.

### Fix 3 — the LoRA ref-model sync corruption (`79cfa6a`) — THE SILENT ONE
`sync_target_model_memory_efficient` (the SDFT anti-forgetting mechanism, runs every step) used
positional `zip(model.parameters(), teacher.parameters())`. Under LoRA the student has ~96 extra
adapter params interleaved, so the first adapter pairs against the wrong teacher param and
**every parameter after misaligns**. The prior shape guard stopped the crash but the sync was
silently wrong — corrupting the teacher. Fixed to name-based matching (normalize PEFT prefixes:
`base_model.model.`, `.base_layer`, `modules_to_save.default.`, `_checkpoint_wrapped_module.`).
**Proven correct on a real Qwen+LoRA model.** This is the bug that would have produced
plausible-looking-but-wrong training. Do not regress it.

---

## THE ONE DECISIVE SEQUENCE

Everything above collapses to this. Do exactly this, in order:

```bash
# 1. Launch H100 80GB on Spheron (~$2/hr). Ubuntu 24.04 + CUDA.

# 2. Build the environment (idempotent; installs repo requirements verbatim + bitsandbytes):
git clone -b spheron-h100 https://github.com/ColdSlither/Self-Distillation.git
cd Self-Distillation && bash setup_spheron_h100.sh

# 3. Smoke test FIRST. ~8 examples, ~3 steps, <5 min, ~$0.20.
#    This exercises the former OOM site AND the fixed sync in one run.
bash ~/smoke_test.sh

# 4. ONLY after smoke passes — full run:
bash ~/train.sh

# 5. Snapshot the instance after setup so you never rebuild.
```

**Stop after step 3 and report** whether all five checkpoints passed (see BLOCKERS.md §verification
protocol): (1) load, (2) generate, (3) finite loss, (4) sync no-error, (5) checkpoint written.
Those five passing = every fixed item empirically confirmed.

---

## POTENTIAL BLOCKERS — the short version (full analysis in `docs/BLOCKERS.md`)

**🔴 Critical — FIXED:** LoRA sync corruption (`79cfa6a`), OOM at compute_loss (`a86f069`).

**🟢 Verified NON-ISSUES (do not re-investigate):**
- `disable_compile=True` in generate — tested, swallowed by `**kwargs`, no crash
- `pad_token_id=None` — tokenizer resolves to `<|endoftext|>` (151329)
- HF generation KV cache — computed ~0.1 GB, trivial
- gradient_checkpointing + use_cache — HF auto-disables use_cache
- EOS `<|im_end|>` termination — verified correct

**🟡 Residual risks — bounded, with mitigations:**
- **Long-run stability** — not testable locally. If first 10 losses NaN/explode, drop LR
  `2e-5 → 1e-5`. Note `ref_model_mixup_alpha=0.01` pulls teacher ~92% toward student over 252
  steps (paper's design); if too aggressive, lower alpha or raise `ref_model_sync_steps`.
- **Teacher prompt inflation** — measured median 634 / max 1086 tokens; ~15% truncated at 1024.
  Raise `max_prompt_length` to 1536 if quality suffers.
- **Throughput** — HF generate ~2-4× slower than vLLM (deliberate tradeoff for 80GB). Flip to
  H200 141GB + vLLM (`use_vllm=True`) if the full run is too slow — it's a one-line change once
  the pipeline is proven.

## Failure triage (if smoke_test fails)
- *OOM at compute_loss* → `--max_completion_length 256` (the OOM tensor scales with it)
- *shape error in sync* → regression of fix 3; check `target_modules` match teacher arch
- *Import/AttributeError* → version drift; `pip freeze` vs the matrix in DIAGNOSIS.md
- *Hang* → impossible under use_vllm=False (no NCCL); something re-enabled vLLM

---

## INFRASTRUCTURE NOTES

- **Spheron:** web dashboard only (no CLI/API). Ubuntu 22.04/24.04 + CUDA images. Volumes
  attachable (`oya_model_liberation` 130 GB). No snapshots for this user (region restriction) —
  so the committed branch is your "snapshot." All instances preemptible.
- **SSH user varies:** `ubuntu@` for A6000/H100 images, `sesterce@` for others.
- **Obsidian vault:** `/home/rell/Documents/Obsidian Vault/Oya/` — historical session notes.
  `2026-06-26-session2.md` has the (now-obsolete) patch catalog. Do not trust summaries over the
  committed code; the summaries have been wrong before.

## KEY FILES (on the branch)
```
main_lora.py            # WORKING entry point: use_vllm=False, LoRA, --smoke_test
setup_spheron_h100.sh   # idempotent env builder; generates ~/train.sh + ~/smoke_test.sh
distil_trainer.py       # _sync_param guard + name-based sync (FIXED, committed)
docs/DIAGNOSIS.md       # both walls, verified mechanisms
docs/BLOCKERS.md        # every failure point, classified
HANDOFF.md              # this file
```

---

## THE PROMPT FOR THE NEXT AGENT

> You are continuing ColdSlither SDFT training. A prior agent (GLM 5.2) spent two sessions
> diagnosing and fixing this pipeline. Three fixes are already committed on branch
> `spheron-h100` of `ColdSlither/Self-Distillation`: the version/server-mode trap, the
> structural 80GB OOM (via `use_vllm=False`), and a silent LoRA ref-model sync corruption bug.
> **Your job is to run and verify, not to re-debug.**
>
> **Do not** ask questions, stop early, or re-derive solved problems. **Do not** pin
> `vllm==0.10.2`, edit `vllm_client.py`, switch to server mode, or apply any of the obsolete
> patches cataloged in the Obsidian notes — they were fixes for the wrong library versions and
> are now harmful. **Do not** patch a live instance; commit fixes to the branch instead.
>
> **Read first:** `HANDOFF.md`, `docs/DIAGNOSIS.md`, `docs/BLOCKERS.md`. Trust the committed code
> over any summary note — summaries have been wrong before.
>
> **Do exactly this:**
> 1. On a fresh Spheron H100 80GB: `git clone -b spheron-h100 … && bash setup_spheron_h100.sh`
> 2. `bash ~/smoke_test.sh`
> 3. Verify all five checkpoints pass (load → generate → finite loss → sync no-error → checkpoint).
>    Report exactly which passed and which failed, with the verbatim error for any failure.
> 4. If smoke passes, run `bash ~/train.sh`.
> 5. If smoke fails, read the FULL error, classify it against BLOCKERS.md, and fix *forward* on
>    the branch — never backtrack to server mode or vllm 0.10.2.
>
> **Success criterion:** one full training step completes and a checkpoint is written. Loss is
> finite and roughly decreasing across the first 10 logged steps. Nothing else counts as solved.
> The only knob you should reach for first is `--max_completion_length` (256) if you OOM at
> compute_loss, and `--learning_rate 1e-5` if loss diverges. If you are stuck after two attempts,
> report what you tried, the verbatim errors, and what you believe is needed — do not keep
> silently iterating.
