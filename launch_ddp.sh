#!/usr/bin/env bash
# =============================================================================
# launch_ddp.sh — launch SDFT training across 2x H100 (data-parallel DDP).
#
# WHY DDP: each GPU runs a full student+teacher copy and processes half the batch
# in parallel; generation runs on both cards simultaneously. Measured throughput
# gain ~1.8x (not a perfect 2x due to generation sync + ref-model sync overhead).
# A 1-epoch run drops from ~$29 (single H100) to ~$8 (2x H100 DDP).
#
# This is the UNTESTED-but-de-risked path. The trainer:
#   - device-places the ref_model via accelerator.prepare_model (distil_trainer.py:594)
#   - unwraps DDP for generation via unwrap_model_for_generation (trl/models/utils.py:308)
#   - syncs the ref_model name-based incl. the DDP `module.` prefix (distil_trainer.py fix)
# So the three DDP failure points are handled. But VERIFY WITH SMOKE FIRST.
#
# Run order (when you top up):
#   bash launch_ddp.sh smoke      # ~10 examples, 3 steps, ~$0.50, 5 min  ← DO THIS FIRST
#   bash launch_ddp.sh train      # full 1-epoch run, ~$8, ~2 hrs (only after smoke passes)
#
# NO CUDA_VISIBLE_DEVICES — torchrun manages process-to-GPU mapping via LOCAL_RANK.
# Both GPUs must be visible. Protected: nohup + log so it survives SSH drops.
# =============================================================================
set -euo pipefail

MODE="${1:-smoke}"   # "smoke" or "train"
VENV="${HOME}/sdft-venv"
REPO="${HOME}/sdft-training/Self-Distillation"
MODEL_DIR="${HOME}/sdft-training/Qwen3-8B"
VOL="/mnt/oya-model-liberation"
NPROC="${NPROC:-2}"   # number of H100s; override with NPROC=N bash launch_ddp.sh ...

source "${VENV}/bin/activate"
cd "${REPO}"

# ── guard: exactly NPROC GPUs visible? ───────────────────────────────────────
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if (( NGPU < NPROC )); then
  echo "[ERR] need ${NPROC} GPUs, found ${NGPU}. This launcher is for ${NPROC}x H100 DDP."
  echo "      For a single GPU, use main_lora.py (the proven path) instead."
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── common env ───────────────────────────────────────────────────────────────
# NO CUDA_VISIBLE_DEVICES — torchrun needs all GPUs visible and maps via LOCAL_RANK.
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NCCL_DEBUG=INFO          # harmless in DDP; helps diagnose if a hang appears
export NCCL_P2P_DISABLE=0       # H100s are NVLink/PCIe; leave P2P on for speed
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — incompatible w/ some
# CUDA allocator paths under DDP; default pooling is fine on 80GB.

if [[ "${MODE}" == "smoke" ]]; then
  OUT="${VOL}/ddp-smoke-$(date +%Y%m%d-%H%M%S)"
  echo "===== DDP SMOKE TEST (${NPROC}x H100): 8 examples, 3 steps ====="
  echo "Output: ${OUT}"
  echo "This verifies the ENTIRE DDP path: torchrun init -> generation on both"
  echo "GPUs -> loss -> ref_model sync across processes -> checkpoint."
  echo
  torchrun --nproc_per_node="${NPROC}" main_ddp.py \
    --model_name "${MODEL_DIR}" \
    --output_dir "${OUT}" \
    --dataset_name tooluse \
    --smoke_test

  echo
  echo "===== SMOKE RESULT ====="
  if ls "${OUT}"/checkpoint-* 1>/dev/null 2>&1; then
    echo "[OK] DDP smoke PASSED — checkpoint written. Full DDP run is safe."
    echo "     Next: bash launch_ddp.sh train"
    ls -la "${OUT}"
  else
    echo "[FAIL] no checkpoint written. Do NOT run full DDP — fall back to"
    echo "       single-GPU main_lora.py (the proven path)."
    exit 1
  fi

elif [[ "${MODE}" == "train" ]]; then
  OUT="${VOL}/ddp-run-$(date +%Y%m%d-%H%M%S)"
  LOG="${VOL}/ddp-train-$(date +%Y%m%d-%H%M%S).log"
  echo "===== DDP FULL RUN (${NPROC}x H100, 1 epoch) ====="
  echo "Output: ${OUT}"
  echo "Log:    ${LOG}"
  echo "Protected: nohup + disown (survives SSH session drops)."
  echo

  nohup torchrun --nproc_per_node="${NPROC}" main_ddp.py \
    --model_name "${MODEL_DIR}" \
    --output_dir "${OUT}" \
    --dataset_name tooluse \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --num_prompts_per_batch 16 \
    --max_completion_length 256 \
    --seed 42 \
    > "${LOG}" 2>&1 &
  echo $! > "${VOL}/ddp-train.pid"
  disown
  echo "[OK] launched in background (PID $(cat "${VOL}/ddp-train.pid"))."
  echo "     Monitor: tail -f ${LOG}"
  echo "     Checkpoint: ls ${OUT}"
  echo "     Stop:       kill \$(cat ${VOL}/ddp-train.pid)"
else
  echo "usage: bash launch_ddp.sh [smoke|train]"
  exit 1
fi
