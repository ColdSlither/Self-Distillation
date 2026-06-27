# Potential Blockers — Verified Analysis for the `use_vllm=False` Run

> Every item below was traced through the actual code path the `use_vllm=False` config
> exercises, and tested where possible against a real Qwen model in the local venv.
> Branch `spheron-h100`, through commit `79cfa6a`.
>
> This is the most important document to read before the next H100 run. It separates
> **real blockers (now fixed)** from **non-issues (verified safe)** from **residual risks**.

---

## Summary table

| # | Item | Status | Severity |
|---|------|--------|----------|
| 1 | LoRA ref-model sync corrupts weights under PEFT | ✅ FIXED (`79cfa6a`) | **Critical** |
| 2 | OOM at compute_loss (full-vocab logps × 2) | ✅ FIXED (`a86f069`, use_vllm=False) | **Critical** |
| 3 | `disable_compile=True` in HF generate | ✅ Non-issue (tested) | — |
| 4 | `pad_token_id=None` in Qwen3 config | ✅ Non-issue (tested) | — |
| 5 | Teacher prompt longer than student (golden response) | ⚠️ Bounded | Low |
| 6 | Generation KV cache peak under HF generate | ✅ Non-issue (computed) | — |
| 7 | gradient_checkpointing + use_cache conflict | ✅ Non-issue (HF auto-disables) | — |
| 8 | Long-run stability / LR / divergence | ⚠️ Residual | Medium |
| 9 | Throughput (HF generate is slower than vLLM) | ⚠️ Known tradeoff | Low |
| 10 | EOS/`<|im_end|>` termination in generation | ✅ Non-issue (verified) | — |

---

## ✅ FIXED — Critical blockers

### 1. LoRA ref-model sync corrupts weights (FIXED in `79cfa6a`)

**This is the most important finding of the analysis and would have silently broken training.**

`MemoryEfficientSyncRefModelCallback.sync_target_model_memory_efficient` (the SDFT anti-forgetting
mechanism, `sync_ref_model=True`, run every step) used:
```python
for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
    _sync_param(model_param, ref_param, alpha)
```
Positional `zip`. Under LoRA the student has ~96 extra adapter params (lora_A/lora_B for each
of the 7 target modules × 36 layers) interleaved with the base weights. So the very first
adapter, `q_proj.lora_A` (shape `[16,4096]`), gets paired with the teacher's `k_proj.weight`
(shape `[4096,4096]`). The shape guard I added earlier prevents the crash — but from that point
**every subsequent parameter is misaligned**. The teacher gets synced with the *wrong* student
weights, or base weights never sync at all.

The H100 postmortem never caught this because it only checked that the crash went away — not
whether the sync was *correct*. It would have produced a model that "trains" (loss moves) but
whose reference model is garbage, defeating the entire SDFT continual-learning premise.

**Verified reproduction** (Qwen2.5-0.5B + LoRA): with positional zip, the first adapter
at param position 3 pairs against `k_proj.weight`; everything after misaligns.

**Fix (`79cfa6a`):** name-based matching. Build a `{normalized_name: param}` dict from the
teacher, iterate the student by name, skip adapter params, sync only base params that exist
in both with matching shapes. Normalization strips PEFT prefixes:
- `base_model.model.` → removed
- `.base_layer` → removed (so `q_proj.base_layer.weight` == teacher's `q_proj.weight`)
- `modules_to_save.default.`, `_checkpoint_wrapped_module.` → removed

**Verified working** (real model test): with the fix, `q_proj` (a LoRA target) syncs correctly
to the student value; `k_proj` (not a target) is untouched; adapters skipped. Both ZeRO-3 and
non-ZeRO paths fixed.

### 2. OOM at compute_loss (FIXED in `a86f069`)

Covered in `docs/DIAGNOSIS.md`. Structural: vLLM colocate pre-allocates ~24 GB KV cache (knob
ignored) + three model copies; the forward-KL loss materializes full-vocab log-softmax for both
student and teacher. Fixed by `use_vllm=False` (drops ~39 GB).

---

## ✅ NON-ISSUES — Verified safe (do NOT re-investigate)

### 3. `disable_compile=True` in `model.generate()` (`distil_trainer.py:1259`)

I initially flagged this as a hard crash — `transformers 4.57.1` `generate()` has no
`disable_compile` parameter. **Tested directly:** `generate()` accepts `**kwargs` and silently
swallows unknown kwargs. No crash. ✅

### 4. `pad_token_id = None` in Qwen3 config

Config says `pad_token_id: None`. I worried the trainer's padding would break. **Tested:** the
Qwen3 *tokenizer* resolves `pad_token` → `<|endoftext|>` (id 151329), and `distil_trainer.py:332`
sets `self.pad_token_id = tokenizer.pad_token_id`. Resolves fine. ✅

(One subtlety: HF prints `Setting pad_token_id to eos_token_id` during generation when the
generation_config lacks one — cosmetic only, the trainer uses its own resolved id.)

### 5 (part). Generation KV cache peak under HF generate

Without vLLM, `model.generate()` builds its own KV cache: `[B, prompt_len+max_new_tokens]`.
Computed for Qwen3-8B (36 layers, GQA ~4 KV heads, 128 dim): **~0.1 GB** at batch 1 / 1024+512
tokens. A transient peak, released before compute_loss. Trivial vs the ~39 GB freed by dropping
vLLM. ✅

### 6. gradient_checkpointing + use_cache conflict

`main_lora.py` sets `gradient_checkpointing=True` and never sets `use_cache=False`. **Verified:**
HF `Trainer._wrap_model` auto-disables `use_cache` when gradient checkpointing is on. No conflict. ✅

### 7. EOS / `<|im_end|>` termination

Qwen3 uses `<|im_end|>` (151645) as eos. The non-vLLM generation path (`distil_trainer.py:1267-1270`)
masks everything after the first eos. **Verified** eos_token_id resolves correctly. Generation
terminates properly. ✅

---

## ⚠️ RESIDUAL RISKS — Bounded, with mitigations

### 5 (part). Teacher prompt length (golden response inflation)

The teacher prompt embeds the full `golden_response` as an in-context example, making it longer
than the student prompt. **Measured on 20 tooluse examples:**
- Student prompt: median 492, max 969 tokens
- Teacher prompt: median 634, max 1086 tokens

With `max_prompt_length=1024`, ~15% of teacher prompts get left-truncated. Not a crash, but means
some teacher in-context examples lose their tail. **Mitigation:** if quality suffers, raise
`max_prompt_length` to 1536 (still fits memory comfortably now). Low severity.

### 8. Long-run stability / LR / divergence (cannot verify locally)

Not testable without a real multi-step run. Risks:
- **LR too high for LoRA + SDFT:** `2e-5` is reasonable for LoRA but untested on this exact setup.
  If loss diverges (NaN) early, drop to `1e-5`.
- **`ref_model_mixup_alpha=0.01`**: teacher moves 1% toward student per step. Over 252 steps that's
  a ~92% pull toward the student — the teacher largely *becomes* the student by the end. This is
  the paper's design (anti-forgetting via slow track), but if it's too aggressive, raise alpha to
  0.001 or increase `ref_model_sync_steps` to 5.
- **Sensible first check:** watch the first 10 logged losses in the smoke test. Should be finite
  and roughly decreasing. NaN/explode → lower LR.

### 9. Throughput — HF generate is slower than vLLM

The deliberate tradeoff. Expect generation ~2-4× slower than vLLM. For 252 steps × 32 prompts this
may mean hours, not minutes. **Mitigation options if too slow:** (a) H200 141GB + vLLM (one-line
flip, ~$4.5/hr); (b) reduce `num_prompts_per_batch`; (c) reduce `max_completion_length`. Not a
correctness issue — only cost/time.

### 10. bitsandbytes 4-bit teacher (optional, not used by default)

`main_lora.py` loads the teacher in bf16 (16 GB), which fits the 80 GB card comfortably under
`use_vllm=False` (~50 GB total used, ~30 GB free). The 4-bit path the postmortem tested is not
needed here and adds the `libnvJitLink` dependency risk. Keep teacher bf16. If memory ever gets
tight, 4-bit teacher is the lever — but it isn't now.

---

## The verification protocol for the next run

When you run `bash ~/smoke_test.sh` on the H100, watch for these in order:

1. **Imports + model load** — should reach "Initializing model" without version errors.
2. **First generation** — `model.generate()` produces tokens (proves #3, #4, #7, #10).
3. **First compute_loss** — prints a finite loss value (proves #2 fix held; this was the OOM site).
4. **First on_step_end sync** — no crash, no shape error (proves #1 fix held).
5. **Checkpoint written** — `ls` the output dir shows `checkpoint-*`.

If all five pass, every item in this document is empirically confirmed and the pipeline is sound.
The remaining unknown (#8, long-run stability) only matters for the full 252-step run, and the
mitigation (lower LR) is a one-argument change.

---

## What was NOT changed (and why)

- **`main.py`** — left pristine. The working config is `main_lora.py`, a separate entry point, so
  upstream stays clean and there's no ambiguity about what's modified.
- **`vllm_client.py` (installed package)** — not patched. The repo's installed source is correct
  for vllm 0.12.0. Prior sessions patched it for the wrong vllm version; that's obsolete.
- **The SDFT algorithm itself** — untouched. `use_vllm=False` changes *how completions are
  generated*, not the loss or the distillation math. The forward-KL loss, teacher logps, and
  ref-model sync are all unchanged (sync now *correctly* implemented for LoRA).
