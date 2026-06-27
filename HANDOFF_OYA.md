# Handoff for Oya — Live Run + Deploy (2026-06-27)

> Operational handoff. You ran the smoke test and the full training; this covers the
> **current live state**, **one risk to act on**, and **what to do when training finishes**.
> (The debugging history, fixes, and analysis live in `HANDOFF.md` / `docs/`.)

---

## LIVE STATE (as observed via SSH, ~04:01 UTC)

- **Instance:** `38.128.232.33` (the `69.19.137.64` one was preempted — relaunching on the
  volume was the right call).
- **Training:** PID 3514, **healthy** — 105% CPU, `R` state, 11+ min elapsed, GPU steady at
  46.5 GB / 97% util. Doing real forward/backward passes, not stuck.
- **Output:** writing to **`/mnt/oya-model-liberation/sdft-output/full-run-1`** (persistent
  volume ✅ — survives preemption). Dir is currently empty; first checkpoint lands at
  `save_steps=100`.
- **Command run:**
  `python3 main_lora.py --model_name .../Qwen3-8B --output_dir /mnt/oya-model-liberation/sdft-output/full-run-1 --learning_rate 2e-5 --num_train_epochs 2 --num_prompts_per_batch 32 --max_completion_length 512`

## ⚠️ ONE RISK TO ACT ON — the run is tied to an SSH session

The training process's parent is `sshd: ubuntu@notty`. Its stdout/stderr go to a pipe. There is
**no `tmux`, no `screen`, no `nohup`, no log file.** If the SSH session holding it drops (laptop
sleep, network blip), the process gets SIGHUP and **dies** — and since the dir is empty until
step 100, you'd lose everything back to the last restart.

You can't cleanly retro-move a running process into tmux. Two options:

1. **Ride it out** if you're confident the session stays up — but at minimum know the exposure.
2. **Relaunch protected** next time (only worth it if you expect the current one to drop):
   ```bash
   cd ~/sdft-training/Self-Distillation
   nohup bash ~/train.sh /mnt/oya-model-liberation/sdft-output/full-run-1 \
     > /mnt/oya-model-liberation/full-run-1.log 2>&1 &
   disown
   ```
   This survives the session *and* writes a log. You lose the ~12 min already done (no checkpoint
   yet to resume from) — so weigh that against how stable the session feels.

**Recommendation:** ride it out, but launch the *next* epoch/restart with the `nohup ... &` form.

## WHEN TRAINING FINISHES — deploy pipeline (3 steps)

**0. Pull first** — your instance is at `00fa40c`, the deploy scripts are in `a787a0f`:
```bash
cd ~/sdft-training/Self-Distillation && git pull
```

**1. Run the deploy pipeline** (auto-picks the latest checkpoint on the volume):
```bash
bash deploy.sh
```
This does: **merge adapters → full model → tooluse eval → GGUF (Q4_K_M)**. All output to the
volume. The merge + GGUF are CPU-only; **eval needs the full GPU** — so either wait for training
to fully finish, or run `bash deploy.sh --no-eval` first and eval later.

**2. Verify the eval** — `cat /mnt/oya-model-liberation/eval-checkpoint-*/eval_results.json`.
The whole point of SDFT is preserving capabilities while shifting the domain, so the tooluse
accuracy number is the real acceptance signal, not just "it trained."

**3. Deploy** — the GGUF is at `/mnt/oya-model-liberation/gguf/*.q4_k_m.gguf`. Loadable in any
llama.cpp runtime (Ollama, LM Studio, `llama-cli`).

## What the deploy scripts do (already built + pushed)

| Script | Step | Resource |
|---|---|---|
| `merge_to_full.py` | LoRA adapters + base → standard HF dir (`PeftModel.merge_and_unload`) | CPU/RAM only |
| `convert_gguf.sh` | HF → GGUF via llama.cpp, Q4_K_M default | CPU only |
| `deploy.sh` | orchestrates all three, auto-finds latest checkpoint, GPU-busy guard | eval needs GPU |

Merge round-trip was verified on a real Qwen+LoRA model — output is exactly the HF dir format
that llama.cpp and `eval_tooluse.py` expect.

## Residual risks to watch while training runs

- **Loss curve:** first 10-20 logged losses should be finite and roughly decreasing. NaN/explode →
  the only knob is `--learning_rate 1e-5`.
- **`ref_model_mixup_alpha=0.01`:** over 252 steps the teacher gets pulled ~92% toward the student
  (the paper's anti-forgetting design). This is **now correct** because of the sync fix (`79cfa6a`)
  — before, the corrupted sync would have made it silently wrong. If generation quality degrades
  late in the run, lower alpha or raise `ref_model_sync_steps`.

## SSH access (for whoever monitors)

- `ssh -i ~/.ssh/id_ed25519 ubuntu@38.128.232.33` — the `id_ed25519` key is authorized.
- The `id_ed25519_spheron` key is also authorized on this instance.
- Check progress: `nvidia-smi`; `ls /mnt/oya-model-liberation/sdft-output/full-run-1/`

## Branch state (on GitHub)

```
a787a0f deploy: LoRA merge → eval → GGUF pipeline
00fa40c docs: H100 post-mortem   ← your instance is here (git pull to get deploy scripts)
fb46522 HANDOFF.md v3 (authoritative)
2758bf3 BLOCKERS.md
79cfa6a fix LoRA ref-model sync (name-based)
a86f069 use_vllm=False (OOM fix)
b5d5329 LoRA + colocate (version/server-mode fix)
```

That's it. Smoke test passed, full run is live and healthy on the volume, deploy pipeline is ready.
The only open item is the SSH-session exposure — your call whether to ride it out or relaunch
protected.
