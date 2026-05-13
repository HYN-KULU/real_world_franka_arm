#!/usr/bin/env bash
# Install Grounded-SAM2 (sam2 + groundingdino) into the frankapanda_3dfa conda env
# so frankapanda/perception/segmenter.py can run.
#
# Idempotent: each step is skipped when its artifact is already present.
#
# - Patches GroundingDINO C++ for PyTorch 2.x (skips if already patched).
# - Builds the GroundingDINO CUDA extension (skips if _C.so already exists).
# - Installs SAM2/gdino runtime python deps (skips packages already importable).
# - Verifies a Segmenter() instance loads.

set -euo pipefail

ENV_NAME="frankapanda_3dfa"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSAM2_DIR="${REPO_ROOT}/gsam2"
GDINO_DIR="${GSAM2_DIR}/grounding_dino"
GDINO_CSRC="${GDINO_DIR}/groundingdino/models/GroundingDINO/csrc/MsDeformAttn"

if [ ! -d "${GSAM2_DIR}/sam2" ] || [ ! -d "${GDINO_DIR}/groundingdino" ]; then
    echo "ERROR: gsam2 submodule missing at ${GSAM2_DIR}."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# ---- conda activate ------------------------------------------------------- #
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH."
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "===> Activating conda env: ${ENV_NAME}"
conda activate "${ENV_NAME}"
echo "===> Python: $(which python)  ($(python --version))"

# ---- torch + cuda sanity --------------------------------------------------- #
echo "===> Torch + CUDA sanity check"
python - <<'PY'
import torch
print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device={torch.cuda.get_device_name(0)}  capability={torch.cuda.get_device_capability(0)}")
PY

# ---- patch gdino C++ if needed -------------------------------------------- #
# PyTorch >= 2.1 dropped implicit conversion from at::DeprecatedTypeProperties
# to c10::ScalarType. The IDEA GroundingDINO C++ source uses the old
# tensor.type() API. Patch only if the broken pattern still exists.
if grep -q '\.type()\.is_cuda()\|AT_DISPATCH_FLOATING_TYPES(value\.type(),' \
        "${GDINO_CSRC}/ms_deform_attn_cuda.cu" \
        "${GDINO_CSRC}/ms_deform_attn.h" 2>/dev/null; then
    echo "===> Patching GroundingDINO C++ sources for PyTorch 2.x"
    sed -i -E 's/([A-Za-z_][A-Za-z0-9_]*)\.type\(\)\.is_cuda\(\)/\1.is_cuda()/g' \
        "${GDINO_CSRC}/ms_deform_attn_cuda.cu" \
        "${GDINO_CSRC}/ms_deform_attn.h"
    sed -i -E 's/AT_DISPATCH_FLOATING_TYPES\(value\.type\(\),/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/g' \
        "${GDINO_CSRC}/ms_deform_attn_cuda.cu"
else
    echo "===> GroundingDINO C++ already patched, skipping"
fi

# ---- build groundingdino C extension if needed ---------------------------- #
# The compiled extension lands in the source dir as groundingdino/_C*.so when
# installed in editable mode (it also gets symlinked from site-packages).
GDINO_CEXT_FOUND=$(ls "${GDINO_DIR}/groundingdino/"_C*.so 2>/dev/null | head -1 || true)
GDINO_PY_OK=$(python -c "from groundingdino import _C; print('ok')" 2>/dev/null || true)
if [ -z "${GDINO_PY_OK}" ] || [ -z "${GDINO_CEXT_FOUND}" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "WARNING: nvcc not on PATH; GroundingDINO extension build may fall back or fail."
        echo "         Set CUDA_HOME or install cuda-toolkit if the next step errors."
    fi
    # Wipe stale build artifacts before retrying.
    rm -rf "${GDINO_DIR}/build" "${GDINO_DIR}/groundingdino.egg-info" \
           "${GDINO_DIR}/groundingdino/"_C*.so 2>/dev/null || true
    echo "===> pip install -e ${GDINO_DIR}   (builds CUDA C extension)"
    pip install -e "${GDINO_DIR}"
else
    echo "===> GroundingDINO C extension already built, skipping pip install"
fi

# ---- install only missing python deps ------------------------------------- #
# SAM2's pyproject.toml requires Python >=3.10, but the source itself runs
# on 3.9. We don't pip-install sam2; segmenter.py adds ${GSAM2_DIR} to
# sys.path. We DO need SAM2's runtime deps and the gdino extras.
# (pkg_to_import_name pairs)
DEPS=(
    "iopath:iopath"
    "tqdm:tqdm"
    "pillow:PIL"
    "supervision:supervision"
    "opencv-python:cv2"
    "pycocotools:pycocotools"
)
MISSING=()
for entry in "${DEPS[@]}"; do
    pip_name="${entry%%:*}"
    import_name="${entry##*:}"
    if python -c "import ${import_name}" >/dev/null 2>&1; then
        echo "  [ok]      ${pip_name}  (import ${import_name})"
    else
        echo "  [missing] ${pip_name}  (import ${import_name})"
        MISSING+=("${pip_name}")
    fi
done

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "===> Installing missing deps: ${MISSING[*]}"
    pip install "${MISSING[@]}"
else
    echo "===> All python deps already installed, skipping"
fi

# ---- verify Segmenter loads ----------------------------------------------- #
echo "===> Verifying Segmenter loads end-to-end"
python - <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}")
sys.path.insert(0, "${GSAM2_DIR}")
from frankapanda.perception.segmenter import Segmenter
seg = Segmenter(device="cuda:0")
print("Segmenter ready")
PY

echo
echo "All set. Run perception pipeline with segmentation:"
echo "  python -m frankapanda.perception.perception_pipeline --continuous --segment \\"
echo "         --seg_labels \"blue block. red cup. lego.\""
