#!/usr/bin/env bash
# deploy.sh — full post-training pipeline: latest checkpoint → merge → eval → GGUF.
#
# Run AFTER training completes. Picks the latest checkpoint on the persistent volume,
# merges adapters into the base, evaluates on tooluse, and converts to deployable GGUF.
# All outputs go to the persistent volume (survives preemption).
#
# Usage:
#   bash deploy.sh                              # auto-pick latest checkpoint
#   bash deploy.sh <checkpoint_dir>             # explicit checkpoint
#   bash deploy.sh <checkpoint_dir> --no-eval   # skip eval (saves time/GPU)
#   bash deploy.sh <checkpoint_dir> --no-gguf   # skip GGUF, just merge
#
# Idempotent: skips merge/eval/gguf if the output already exists.
set -euo pipefail

VOL="/mnt/oya-model-liberation"
BASE="${HOME}/sdft-training/Qwen3-8B"
VENV="${HOME}/sdft-venv"
REPO="${HOME}/sdft-training/Self-Distillation"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*"; }

# ── parse args ──────────────────────────────────────────────────────────────
CKPT="${1:-}"
DO_EVAL=1
DO_GGUF=1
for a in "$@"; do
  case "$a" in
    --no-eval) DO_EVAL=0 ;;
    --no-gguf) DO_GGUF=0 ;;
  esac
done

# ── auto-pick latest checkpoint if not given ────────────────────────────────
if [[ -z "${CKPT}" ]] || [[ ! -d "${CKPT}" ]]; then
  info "no checkpoint given — finding latest on ${VOL}..."
  CKPT=$(ls -dt "${VOL}"/sdft-output/*/checkpoint-* 2>/dev/null | head -1)
  if [[ -z "${CKPT}" ]]; then
    err "no checkpoint found under ${VOL}/sdft-output/*/checkpoint-*"
    err "training may still be running, or no save_steps has been hit yet."
    exit 1
  fi
fi

if [[ ! -f "${CKPT}/adapter_model.safetensors" ]]; then
  err "not a PEFT checkpoint (no adapter_model.safetensors): ${CKPT}"
  exit 1
fi

MERGED="${VOL}/merged-$(basename "${CKPT}")"
EVAL_DIR="${VOL}/eval-$(basename "${CKPT}")"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"

info "checkpoint: ${CKPT}"
info "base model: ${BASE}"
info "merged out: ${MERGED}"
echo

# ── 1. merge adapters into base → full HF model ─────────────────────────────
if [[ -f "${MERGED}/config.json" ]] && ls "${MERGED}"/*.safetensors &>/dev/null; then
  warn "merged model exists — skipping merge: ${MERGED}"
else
  info "[1/3] merging LoRA adapters into base..."
  # CPU/RAM only — does NOT touch the GPU.
  python3 merge_to_full.py --base "${BASE}" --adapter "${CKPT}" --out "${MERGED}"
fi
ok "merged full model: ${MERGED}"
echo

# ── 2. evaluate on tooluse (verifies capabilities preserved) ────────────────
if [[ "${DO_EVAL}" == "1" ]]; then
  if [[ -f "${EVAL_DIR}/eval_results.json" ]]; then
    warn "eval results exist — skipping: ${EVAL_DIR}/eval_results.json"
  else
    info "[2/3] evaluating on tooluse (vLLM, GPU)..."
    # NOTE: eval uses vLLM. If training is still running and holding the GPU,
    # this will OOM — wait for training to finish first, or pass --no-eval.
    if [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -gt 5000 ]]; then
      err "GPU is in use (>5GB) — eval needs the full GPU. Wait for training to finish,"
      err "or re-run with: bash deploy.sh ${CKPT} --no-eval"
      exit 1
    fi
    python3 eval_tooluse.py --model_path "${MERGED}" --output_dir "${EVAL_DIR}"
    ok "eval done: ${EVAL_DIR}/eval_results.json"
    echo "  accuracy: $(python3 -c "import json;print(json.load(open('${EVAL_DIR}/eval_results.json'))['accuracy'])" 2>/dev/null || echo '?')"
  fi
else
  warn "[2/3] eval skipped (--no-eval)"
fi
echo

# ── 3. convert to GGUF for deployment ───────────────────────────────────────
if [[ "${DO_GGUF}" == "1" ]]; then
  info "[3/3] converting to GGUF (Q4_K_M, CPU-only)..."
  bash convert_gguf.sh "${MERGED}" q4_k_m
else
  warn "[3/3] GGUF skipped (--no-gguf)"
fi

echo
ok "DEPLOY PIPELINE COMPLETE"
echo "  merged:  ${MERGED}"
[[ -f "${EVAL_DIR}/eval_results.json" ]] && echo "  eval:    ${EVAL_DIR}/eval_results.json"
echo "  gguf:    ${VOL}/gguf/  (run: ls ${VOL}/gguf/)"
