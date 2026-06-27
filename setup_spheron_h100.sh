#!/usr/bin/env bash
# =============================================================================
# setup_spheron_h100.sh — idempotent SDFT environment setup for Spheron H100 80GB
#
# Run on a FRESH Spheron H100 (Ubuntu 22.04/24.04) instance:
#   bash setup_spheron_h100.sh
#
# Design principles (read docs/DIAGNOSIS.md for the full rationale):
#   1. NO VERSION DRIFT.   Installs the repo's own requirements.txt verbatim —
#                          the known-good matrix (torch 2.9 / vllm 0.12 / trl 0.24 /
#                          transformers 4.57.1). Do NOT hand-roll pins. 5 of the old
#                          6 "patches" were fixes for the wrong versions this avoids.
#   2. NO SOURCE PATCHING. The repo source is correct as-shipped. We do not edit
#                          distil_trainer.py or trl/extras/vllm_client.py. The old
#                          setup_spheron.sh patched for symptoms of bad versions — gone.
#   3. IDEMPOTENT.         Re-running skips finished phases. Safe to resume after a
#                          network blip.
#   4. COLOCATE ONLY.      vllm_mode="colocate" is in main_lora.py. There is no server
#                          mode, no NCCL, no 2nd GPU, no init_communicator. Ever.
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
GIT_BRANCH="spheron-h100"   # <-- the branch with main_lora.py + this setup

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*"; }

# ── Phase 0: system packages ────────────────────────────────────────────────
phase_system() {
    info "Phase 0: system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential cmake git curl wget htop nvtop tmux \
        python3-pip python3-venv > /dev/null 2>&1
    ok "system packages"

    if ! command -v nvidia-smi &>/dev/null; then
        err "nvidia-smi not found — are you on a GPU instance?"; exit 1
    fi
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader
    # Warn (don't fail) if < 60 GB — H100 80GB expected, A100 80GB also fine.
    local vram; vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if (( vram < 60000 )); then
        warn "Only ${vram} MiB VRAM. LoRA+colocate Qwen3-8B needs ~50-60 GB. May OOM. H100 80GB recommended."
    fi
}

# ── Phase 1: venv from repo requirements (no version drift) ─────────────────
phase_venv() {
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        warn "venv exists — skipping creation"
    else
        info "Phase 1: venv..."
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip setuptools wheel

    info "Verifying known-good versions are present..."
    # Only install torch first if the venv is empty (torch+vllm must resolve together).
    if ! python3 -c "import torch" 2>/dev/null; then
        info "Installing deps from repo requirements.txt (known-good matrix)..."
        pip install --quiet -r "${REPO_DIR}/requirements.txt" 2>&1 | tail -15
    fi

    # Verify the matrix matches docs/DIAGNOSIS.md. Fail loudly if drifted.
    python3 - << 'PYEOF'
import importlib.metadata as m, sys
want = {"torch":"2.9.0","vllm":"0.12.0","trl":"0.24.0","transformers":"4.57.1",
        "accelerate":"1.11.0","peft":"0.17.1","datasets":"4.3.0"}
bad=False
for p,exp in want.items():
    try:
        got=m.version(p)
        flag = "OK " if got==exp else "DRIFT"
        if got!=exp: bad=True
        print(f"  [{flag}] {p:14s} got {got:10s} expected {exp}")
    except Exception:
        print(f"  [MISS] {p:14s} NOT INSTALLED"); bad=True
sys.exit(1 if bad else 0)
PYEOF
    ok "venv versions verified against known-good matrix"
}

# ── Phase 2: repo (this branch) ─────────────────────────────────────────────
phase_repo() {
    info "Phase 2: repo (branch ${GIT_BRANCH})..."
    mkdir -p "${HOME}/sdft-training"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        warn "repo exists — fetching ${GIT_BRANCH}"
        cd "${REPO_DIR}" && git fetch origin && git checkout "${GIT_BRANCH}" && git pull
    else
        git clone --depth 1 --branch "${GIT_BRANCH}" "${GIT_REMOTE}" "${REPO_DIR}"
    fi
    ok "repo at ${REPO_DIR} on ${GIT_BRANCH}"
}

# ── Phase 3: model ──────────────────────────────────────────────────────────
phase_model() {
    info "Phase 3: model..."
    source "${VENV_DIR}/bin/activate"
    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        warn "model exists — skipping download"
    else
        mkdir -p "${MODEL_DIR}"
        hf download "${MODEL_NAME}" --local-dir "${MODEL_DIR}" 2>&1 | tail -3
    fi
    python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('${MODEL_DIR}')
print(f'Model: {c.architectures}, {c.num_hidden_layers}L, hidden {c.hidden_size}')
assert c.architectures == ['Qwen3ForCausalLM'], 'wrong model!'
print('Verified OK')
"
    ok "model ready at ${MODEL_DIR}"
}

# ── Phase 4: write launch scripts ───────────────────────────────────────────
phase_scripts() {
    info "Phase 4: writing launch scripts..."

    # --- train.sh: full run ---
    cat > "${TRAIN_SCRIPT}" << EOF
#!/usr/bin/env bash
# Full SDFT LoRA training. Run AFTER setup + a passing smoke test.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export NCCL_DEBUG=INFO         # harmless in colocate (no NCCL); helps if ever debugged

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

OUT="\${1:-${HOME}/sdft-training/output/run-\$(date +%Y%m%d-%H%M%S)}"
echo "Output: \$OUT"

python3 main_lora.py \\
  --model_name "${MODEL_DIR}" \\
  --output_dir "\$OUT" \\
  --dataset_name tooluse \\
  --learning_rate 2e-5 \\
  --num_train_epochs 2 \\
  --num_prompts_per_batch 32 \\
  --seed 42
EOF
    chmod +x "${TRAIN_SCRIPT}"

    # --- smoke_test.sh: 10 examples, 3 steps ---
    cat > "${SMOKE_SCRIPT}" << EOF
#!/usr/bin/env bash
# SMOKE TEST. ~10 examples, 3 steps, <5 min, ~\$0.20.
# Proves: init -> vLLM generate -> forward/backward -> checkpoint.
# Run this FIRST on every fresh instance before paying for a full run.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

OUT="${HOME}/sdft-training/smoke-test"
rm -rf "\$OUT"

python3 main_lora.py \\
  --model_name "${MODEL_DIR}" \\
  --output_dir "\$OUT" \\
  --dataset_name tooluse \\
  --smoke_test

echo ""
echo "SMOKE TEST PASSED if you saw: a vLLM generate, loss decreasing, a checkpoint written."
ls -la "\$OUT"
EOF
    chmod +x "${SMOKE_SCRIPT}"
    ok "scripts: ${TRAIN_SCRIPT} , ${SMOKE_SCRIPT}"
}

# ── Phase 5: summary ────────────────────────────────────────────────────────
phase_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  SETUP COMPLETE — H100 LoRA colocate                  ║"
    echo "╠════════════════════════════════════════════════════════╣"
    echo "║  1. Run smoke test FIRST:                              ║"
    echo "║       bash ${SMOKE_SCRIPT}                             ║"
    echo "║                                                        ║"
    echo "║  2. If smoke passes, full run:                         ║"
    echo "║       bash ${TRAIN_SCRIPT}                             ║"
    echo "║                                                        ║"
    echo "║  3. Take a Spheron snapshot NOW so you never rebuild.  ║"
    echo "╚════════════════════════════════════════════════════════╝"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    ok "ready."
}

main() {
    echo ""
    echo "╔═════════════════════════════════════════════╗"
    echo "║  SDFT Setup — Spheron H100 (LoRA colocate)  ║"
    echo "╚═════════════════════════════════════════════╝"
    phase_system
    phase_repo     # repo first so requirements.txt exists for venv phase
    phase_venv
    phase_model
    phase_scripts
    phase_summary
}

main "$@"
