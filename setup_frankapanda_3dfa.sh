#!/usr/bin/env bash
# Create frankapanda_3dfa: clone of frankapanda + inference libs only.
# No sim stack, no extra editable installs. A .pth file is added to the
# env site-packages so `models.flowmatch_actor.*` and
# `visplan.submodules.robo_utils.*` (PEP 420 namespace packages under
# visplanWM/) resolve from any cwd.

set -euo pipefail

ENV_NAME="frankapanda_3dfa"
BASE_ENV="frankapanda"
VISPLAN_WM="/home/ksaha/Research/ModelBasedPlanning/visplanWM"

echo "[1/4] Cloning ${BASE_ENV} -> ${ENV_NAME}"
conda create --name "${ENV_NAME}" --clone "${BASE_ENV}" -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -V

echo "[2/4] Installing inference-only pip libs"
pip install \
    einops==0.8.1 \
    diffusers==0.35.2 \
    transformers==4.57.1 \
    tokenizers==0.22.1 \
    huggingface-hub==0.36.0 \
    safetensors==0.6.2 \
    ftfy==6.3.1 \
    regex==2025.10.23 \
    kornia==0.8.1 \
    kornia_rs==0.1.9 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    antlr4-python3-runtime==4.9.3

# OpenAI CLIP (pkg name `clip`, install from git).
pip install git+https://github.com/openai/CLIP.git

echo "[3/4] Adding visplanWM to sys.path via .pth file"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH_FILE="${SITE_PACKAGES}/visplanWM.pth"
echo "${VISPLAN_WM}" > "${PTH_FILE}"
echo "Wrote ${PTH_FILE}"

echo "[4/4] Sanity import check"
python - <<'PY'
import importlib, sys
mods = [
    "torch", "einops", "diffusers", "transformers", "tokenizers",
    "huggingface_hub", "safetensors", "clip", "ftfy", "regex",
    "kornia", "hydra", "omegaconf",
    # resolved via visplanWM.pth
    "models.flowmatch_actor.modeling.policy_grasp_place.denoise_actor_3d_packing",
    "models.flowmatch_actor.modeling.policy_grasp_place.value_network",
    "visplan.submodules.robo_utils.robo_utils.visualization.plotting",
]
ok, fail = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        fail.append((m, repr(e)))
print(f"OK ({len(ok)}): {ok}")
if fail:
    print(f"FAIL ({len(fail)}):")
    for m, e in fail:
        print(f"  {m}: {e}")
    sys.exit(1)
PY

echo "Done. Activate: conda activate ${ENV_NAME}"
echo "Imports of models.flowmatch_actor.* and visplan.submodules.robo_utils.*"
echo "resolve from any cwd via ${PTH_FILE} -> ${VISPLAN_WM}"
