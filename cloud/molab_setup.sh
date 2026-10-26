#!/bin/bash
# Setup sandbox molab untuk training RVC v2 40k. Idempoten: aman diulang.
#
# Pakai di sandbox:
#   nohup bash /marimo/molab_setup.sh > /marimo/logs/setup.log 2>&1 &
#
# Hasil:
#   /marimo/rvc          repo RVC (main)
#   /marimo/venv         venv yang mewarisi torch sistem (cu130), bukan torch unduhan
#   /marimo/rvc/assets   hubert_base, rmvpe/rmvpe.pt, pretrained_v2/f0{G,D}40k.pth
#   /marimo/rvc/configs/v2/40k.json
#
# Catatan:
#   - JANGAN pasang torch ke venv. Torch sistem 2.11+cu130 sudah mendukung sm_120
#     (Blackwell); torch unduhan uv akan menimpa dan merusak torchvision sistem.
#   - numpy tetap 2.x (numpy<2 tidak punya wheel cp313).
set -uo pipefail

RVC=${RVC:-/marimo/rvc}
VENV=${VENV:-/marimo/venv}
PY=$VENV/bin/python
HF=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main
REPO=https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

step() { echo; echo "===== $1 ($(date -u +%H:%M:%S)) ====="; }
fail() { echo "!!! GAGAL: $1"; exit "${2:-1}"; }
t0=$(date +%s)

# ---------------------------------------------------------------- 1. repo RVC
step "1/5 repo RVC"
if [ -d "$RVC/.git" ]; then
    git -C "$RVC" fetch --depth 1 origin main && git -C "$RVC" reset --hard origin/main \
        || echo "  (fetch gagal, pakai salinan yang ada)"
else
    rm -rf "$RVC"
    git clone --depth 1 "$REPO" "$RVC" || fail "git clone RVC" 10
fi
[ -f "$RVC/train/preprocess.py" ] || fail "repo RVC tidak lengkap" 10
echo "commit: $(git -C "$RVC" rev-parse --short HEAD 2>/dev/null)"
# config v2 40k hilang di upstream, isinya identik dengan v1 40k
[ -f "$RVC/configs/v2/40k.json" ] || { cp "$RVC/configs/v1/40k.json" "$RVC/configs/v2/40k.json" && echo "configs/v2/40k.json dibuat"; }

# ------------------------------------------------------- 2. venv tanpa torch
step "2/5 venv (mewarisi torch sistem)"
PKGS=(
    "scipy>=1.13.1,<1.16"
    "librosa>=0.10.2,<0.11"
    "soundfile>=0.13"
    "av>=15.1.0,<16"
    "scikit-learn>=1.6,<2"
    "tensorboard>=2.19"
    "transformers==4.49.0"
    "einops>=0.8"
    "local-attention>=1.11"
    "tqdm>=4.67"
    "PyYAML>=6.0.1"
    "matplotlib>=3.8.2,<4"
    "ffmpeg-python"
    "faiss-cpu"
    "praat-parselmouth>=0.4.5,<1"
)

if [ ! -x "$PY" ]; then
    ok=0
    for base in /usr/local/bin/python3 "$(command -v python3)"; do
        [ -x "$base" ] || continue
        rm -rf "$VENV"
        uv venv --system-site-packages --python "$base" "$VENV" >/dev/null 2>&1 || continue
        if "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then ok=1; break; fi
        echo "  $base: torch sistem tidak terlihat, coba kandidat lain"
    done
    [ "$ok" = 1 ] || fail "tidak ada interpreter dengan torch CUDA" 11
fi

uv pip install --python "$PY" "${PKGS[@]}" || fail "install paket" 12

# uv menarik torch dari PyPI (2.14.1, tanpa sm_120) -> buang, pakai torch sistem
if "$PY" -c "import torch" 2>/dev/null && [ "$("$PY" -c 'import torch,sys; print("venv" if "venv" in torch.__file__ else "system")')" = "venv" ]; then
    echo "  membuang torch unduhan dari venv"
    uv pip uninstall --python "$PY" torch triton torchvision 2>/dev/null
    nvidia=$(uv pip list --python "$PY" 2>/dev/null | awk '/^(nvidia-|cuda-)/{print $1}' | tr '\n' ' ')
    [ -n "$nvidia" ] && uv pip uninstall --python "$PY" $nvidia 2>/dev/null
fi

"$PY" - <<'PYEOF' || fail "verifikasi env" 13
import torch, numpy, librosa, soundfile, sklearn, transformers, faiss, parselmouth, ffmpeg
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "| librosa", librosa.__version__, "| faiss", faiss.__version__)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.cuda.is_available()
PYEOF

# ------------------------------------------------------------- 3. aset model
step "3/5 aset pretrained (~585 MB)"
fetch() {  # url dst minimal_size
    local url=$1 dst=$2 min=$3
    mkdir -p "$(dirname "$dst")"
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" -ge "$min" ]; then
        echo "  OK   $(basename "$dst") $(stat -c %s "$dst")"
        return 0
    fi
    curl -L --fail -C - --retry 5 --connect-timeout 20 -m 3600 -o "$dst" "$url" \
        || { echo "  FAIL $(basename "$dst")"; return 1; }
    local have; have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    [ "$have" -ge "$min" ] || { echo "  FAIL $(basename "$dst") $have < $min"; return 1; }
    echo "  OK   $(basename "$dst") $have"
}

# rmvpe harus di assets/rmvpe/rmvpe.pt (train/dataset/extract_f0.py memakai path itu)
[ -f "$RVC/assets/rmvpe.pt" ] && [ ! -f "$RVC/assets/rmvpe/rmvpe.pt" ] && mkdir -p "$RVC/assets/rmvpe" && mv "$RVC/assets/rmvpe.pt" "$RVC/assets/rmvpe/rmvpe.pt"

fetch "$HF/hubert_base/config.json"              "$RVC/assets/hubert_base/config.json" 1492
fetch "$HF/hubert_base/preprocessor_config.json" "$RVC/assets/hubert_base/preprocessor_config.json" 225
fetch "$HF/hubert_base/pytorch_model.bin"        "$RVC/assets/hubert_base/pytorch_model.bin" 189206711
fetch "$HF/rmvpe.pt"                             "$RVC/assets/rmvpe/rmvpe.pt" 181184272
fetch "$HF/pretrained_v2/f0G40k.pth"             "$RVC/assets/pretrained_v2/f0G40k.pth" 73106273
fetch "$HF/pretrained_v2/f0D40k.pth"             "$RVC/assets/pretrained_v2/f0D40k.pth" 142875703

# ---------------------------------------------------------- 4. uji import RVC
step "4/5 uji import modul RVC"
cd "$RVC" || fail "cd $RVC" 14
mkdir -p logs assets/weights /marimo/out
"$PY" - <<'PYEOF' || fail "import modul RVC" 15
import os, sys
os.chdir("/marimo/rvc")
sys.path.insert(0, "/marimo/rvc")
from train.dataset.slicer2 import Slicer
from infer.hubert import load_hubert_model
from infer.rmvpe import RMVPE
from configs.config import get_device_dtype_sm, get_training_dtype
print("import RVC ok | dtype training:", get_training_dtype(), "| device:", get_device_dtype_sm(0)[:2])
PYEOF

step "5/5 selesai"
echo "durasi setup: $(( $(date +%s) - t0 ))s"
echo "langkah berikutnya: bash /marimo/molab_fetch.sh lalu bash /marimo/molab_pipeline.sh"
echo SETUPDONE
