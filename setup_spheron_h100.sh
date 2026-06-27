#!/usr/bin/env bash
# =============================================================================
# setup_spheron_h100.sh — idempotent SDFT setup, use_vllm=False path (H100 80GB)
#
# Run on a FRESH Spheron H100 80GB instance:
#   bash setup_spheron_h100.sh
#
# THE FIX (read H100_POSTMORTEM.md + docs/DIAGNOSIS.md for full rationale):
#   use_vllm=False. vLLM colocate mode pre-allocates ~24 GB KV cache that the
#   utilization knob does NOT control, colliding with three model copies + the
#   full-vocab KL loss tensor → consistent 78 GB OOM at compute_loss. Disabling
#   vLLM frees ~39 GB (weights + KV cache); generation uses HF model.generate().
#   Slower generation, but a full training step actually completes.
#
# Design principles:
#   1. NO VERSION DRIFT.   Installs repo requirements.txt verbatim (known-good:
#                          torch 2.9 / vllm 0.12 / trl 0.24 / transformers 4.57.1).
#   2. NO SOURCE PATCHING. The _sync_param LoRA shape guard is committed in the
#                          repo now — never re-patch per instance.
#   3. IDEMPOTENT.         Re-runnable.
#   4. NO expandable_segments. Incompatible with the memory pool (postmortem
#                          blocker #2). Not needed under use_vllm=False anyway.
#
# NO TRAINING HAPPENS HERE. Build once, snapshot, train many times.
# =============================================================================

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_DIR="${HOME}/sdft-training/Qwen3-8B"
REPO_DIR="${HOME}/sdft-training/Self-Distillation"
VENV_DIR="${HOME}/sdft-venv"
TRAIN_SCRIPT="${HOME}/train.sh"
SMOKE_SCRIPT="${HOME}/smoke_test.sh"
GIT_REMOTE="https://github.com/ColdSlither/Self-Distillation.git"
GIT_BRANCH="spheron-h100"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*"; }

phase_system() {
    info "Phase 0: system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential cmake git curl wget htop nvtop tmux \
        python3-pip python3-venv > /dev/null 2>&1
    ok "system packages"
    command -v nvidia-smi &>/dev/null || { err "no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
}

phase_repo() {
    info "Phase 1: repo (branch ${GIT_BRANCH})..."
    mkdir -p "${HOME}/sdft-training"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        warn "repo exists — fetching"
        cd "${REPO_DIR}" && git fetch origin && git checkout "${GIT_BRANCH}" && git pull
    else
        git clone --depth 1 --branch "${GIT_BRANCH}" "${GIT_REMOTE}" "${REPO_DIR}"
    fi
    ok "repo at ${REPO_DIR} on ${GIT_BRANCH}"
}

phase_venv() {
    info "Phase 2: venv..."
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip setuptools wheel

    if ! python3 -c "import torch" 2>/dev/null; then
        info "Installing deps from repo requirements.txt..."
        pip install --quiet -r "${REPO_DIR}/requirements.txt" 2>&1 | tail -15
    fi

    # bitsandbytes is NOT in repo requirements.txt but IS safe to have available
    # (4-bit teacher option). Postmortem blocker #3 found it missing.
    if ! python3 -c "import bitsandbytes" 2>/dev/null; then
        info "Installing bitsandbytes (not in repo requirements)..."
        pip install --quiet bitsandbytes 2>&1 | tail -3
    fi

    # Verify the known-good matrix (matches docs/DIAGNOSIS.md).
    python3 - << 'PYEOF'
import importlib.metadata as m, sys
want = {"torch":"2.9.0","vllm":"0.12.0","trl":"0.24.0","transformers":"4.57.1",
        "accelerate":"1.11.0","peft":"0.17.1","datasets":"4.3.0"}
bad=False
for p,exp in want.items():
    try:
        got=m.version(p); flag="OK " if got==exp else "DRIFT"
        if got!=exp: bad=True
        print(f"  [{flag}] {p:14s} got {got:10s} exp {exp}")
    except Exception:
        print(f"  [MISS] {p:14s} NOT INSTALLED"); bad=True
sys.exit(1 if bad else 0)
PYEOF
    ok "versions verified"
}

phase_model() {
    info "Phase 3: model..."
    source "${VENV_DIR}/bin/activate"
    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        warn "model exists — skipping"
    else
        mkdir -p "${MODEL_DIR}"
        hf download "${MODEL_NAME}" --local-dir "${MODEL_DIR}" 2>&1 | tail -3
    fi
    python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('${MODEL_DIR}')
assert c.architectures == ['Qwen3ForCausalLM']
print(f'Model OK: {c.architectures}, {c.num_hidden_layers}L, hidden {c.hidden_size}')
"
    ok "model ready"
}

phase_scripts() {
    info "Phase 4: writing launch scripts..."
    # NOTE: no PYTORCH_CUDA_ALLOC_CONF=expandable_segments (postmortem blocker #2).
    #       no NCCL_* env (we are not using vLLM server mode — no distributed comm).
    cat > "${TRAIN_SCRIPT}" << EOF
#!/usr/bin/env bash
# Full SDFT training, use_vllm=False (HF generation). Run AFTER smoke test passes.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

OUT="\${1:-${HOME}/sdft-training/output/run-\$(date +%Y%m%d-%H%M%S)}"
echo "Output: \$OUT  (generation via HF model.generate — slower than vLLM but fits 80GB)"

python3 main_lora.py \\
  --model_name "${MODEL_DIR}" \\
  --output_dir "\$OUT" \\
  --dataset_name tooluse \\
  --learning_rate 2e-5 \\
  --num_train_epochs 2 \\
  --num_prompts_per_batch 32 \\
  --max_completion_length 512 \\
  --seed 42
EOF
    chmod +x "${TRAIN_SCRIPT}"

    cat > "${SMOKE_SCRIPT}" << EOF
#!/usr/bin/env bash
# SMOKE TEST. 8 examples, short completions, ~3 steps, <5 min, ~\$0.20.
# Proves: init -> model.generate() -> compute_loss (the former OOM site) -> ckpt.
# Run FIRST on every fresh instance.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

OUT="${HOME}/sdft-training/smoke-test"
rm -rf "\$OUT"
python3 main_lora.py --model_name "${MODEL_DIR}" --output_dir "\$OUT" --dataset_name tooluse --smoke_test

echo ""
echo "SMOKE TEST PASSED if you saw: an HF generate, a loss value, a checkpoint."
echo "Former OOM site (torch.cat(all_logps) at compute_loss) did NOT fire -> fix works."
ls -la "\$OUT"
EOF
    chmod +x "${SMOKE_SCRIPT}"
    ok "scripts: ${TRAIN_SCRIPT} , ${SMOKE_SCRIPT}"
}

phase_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  SETUP COMPLETE — H100 use_vllm=False                 ║"
    echo "╠════════════════════════════════════════════════════════╣"
    echo "║  1. Run smoke test FIRST:                              ║"
    echo "║       bash ${SMOKE_SCRIPT}                             ║"
    echo "║                                                        ║"
    echo "║  2. If smoke passes, full run:                         ║"
    echo "║       bash ${TRAIN_SCRIPT}                             ║"
    echo "║                                                        ║"
    echo "║  3. Take a Spheron snapshot NOW.                       ║"
    echo "╚════════════════════════════════════════════════════════╝"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    ok "ready."
}

main() {
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  SDFT Setup — Spheron H100 (use_vllm=False)   ║"
    echo "╚═══════════════════════════════════════════════╝"
    phase_system
    phase_repo
    phase_venv
    phase_model
    phase_scripts
    phase_summary
}
main "$@"
