#!/usr/bin/env bash
# convert_gguf.sh — convert a merged HF model to GGUF for deployment.
#
# Pipeline step: merged HF model (from merge_to_full.py) → GGUF (for llama.cpp / Ollama / etc.)
# Uses llama.cpp's convert_hf_to_gguf.py (Python, no CUDA build needed for conversion).
# Quantization to Q4_K_M by default — the standard deploy balance of size/quality for 8B models.
#
# Usage:
#   bash convert_gguf.sh <merged_model_dir> [outtype]
#     outtype: q4_k_m (default, ~5GB, deploy) | q8_0 (~9GB, near-lossless) | f16 (~16GB, no quant)
#
# Example:
#   bash convert_gguf.sh /mnt/oya-model-liberation/qwen3-8b-sdft-merged
set -euo pipefail

MERGED="${1:?usage: convert_gguf.sh <merged_model_dir> [outtype]}"
OUTTYPE="${2:-q4_k_m}"
MERGED="${MERGED%/}"  # strip trailing slash

LLAMA_DIR="${HOME}/llama.cpp"
VENV="${HOME}/sdft-venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*"; }

if [[ ! -f "${MERGED}/config.json" ]]; then
  err "not a HF model dir (no config.json): ${MERGED}"
  err "run merge_to_full.py first to produce a merged full model"
  exit 1
fi

# ── 1. get llama.cpp + build the converter (Python build, fast, no CUDA) ─────
if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  info "cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_DIR}"
fi

CONVERTER="${LLAMA_DIR}/convert_hf_to_gguf.py"
if [[ ! -f "${CONVERTER}" ]]; then
  err "convert_hf_to_gguf.py not found at ${CONVERTER}"
  exit 1
fi

# install converter deps into the existing venv (transformers is already there)
# shellcheck disable=SC1091
source "${VENV}/bin/activate" 2>/dev/null || true
pip install --quiet -r "${LLAMA_DIR}/requirements/requirements-convert_hf_to_gguf.txt" 2>&1 | tail -3

# ── 2. convert HF → unquantized GGUF (f16), then quantize ───────────────────
MODEL_NAME="$(basename "${MERGED}")"
OUTDIR="/mnt/oya-model-liberation/gguf"
mkdir -p "${OUTDIR}"
F16_GGUF="${OUTDIR}/${MODEL_NAME}.f16.gguf"
FINAL_GGUF="${OUTDIR}/${MODEL_NAME}.${OUTTYPE}.gguf"

info "converting HF → f16 GGUF: ${MERGED} → ${F16_GGUF}"
# CPU + RAM only; runs alongside training without touching the GPU.
python3 "${CONVERTER}" "${MERGED}" --outtype f16 --outfile "${F16_GGUF}"
ok "f16 GGUF written: $(du -h "${F16_GGUF}" | cut -f1)"

if [[ "${OUTTYPE}" == "f16" ]]; then
  ok "done (no quantization requested): ${F16_GGUF}"
  exit 0
fi

# ── 3. quantize (needs the llama-quantize binary; build it once) ────────────
if ! command -v llama-quantize &>/dev/null && [[ ! -x "${LLAMA_DIR}/build/bin/llama-quantize" ]]; then
  info "building llama-quantize (one-time, ~2 min)..."
  sudo apt-get install -y -qq cmake build-essential > /dev/null 2>&1 || true
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -DGGML_CUDA=OFF > /dev/null 2>&1
  cmake --build "${LLAMA_DIR}/build" --target llama-quantize -j > /dev/null 2>&1
fi
QUANTIZE="${LLAMA_DIR}/build/bin/llama-quantize"
command -v llama-quantize &>/dev/null && QUANTIZE="$(command -v llama-quantize)"

info "quantizing f16 → ${OUTTYPE}: ${FINAL_GGUF}"
"${QUANTIZE}" "${F16_GGUF}" "${FINAL_GGUF}" "${OUTTYPE}"
ok "final GGUF: ${FINAL_GGUF} ($(du -h "${FINAL_GGUF}" | cut -f1))"
echo
echo "Deploy-ready. Test with:"
echo "  llama-cli -m ${FINAL_GGUF} -p 'Hello' -n 64"
echo "  # or load in Ollama / LM Studio / any llama.cpp runtime"
