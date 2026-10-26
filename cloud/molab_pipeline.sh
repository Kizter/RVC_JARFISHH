#!/bin/bash
# Pipeline training RVC v2 40k di molab.
# Dijalankan detached: nohup bash cloud/molab_pipeline.sh > /marimo/logs/pipeline.log 2>&1 &
#
# Prasyarat di sandbox:
#   /marimo/rvc            repo RVC + venv /marimo/venv
#                          + assets/hubert_base, assets/rmvpe/rmvpe.pt, assets/pretrained_v2
#   /marimo/data/clips     klip 40 kHz mono 16-bit 3-10 detik
set -uo pipefail

cd /marimo/rvc || exit 1
PY=/marimo/venv/bin/python
EXP=${EXP:-myvoice}
DATA=${DATA:-/marimo/data/clips}
EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-8}
SAVE_EVERY=${SAVE_EVERY:-25}
WORKERS=${WORKERS:-8}
EXP_DIR="logs/$EXP"
OUT=/marimo/out

# savee() menulis ke assets/weights tanpa membuat direktorinya sendiri
mkdir -p "$EXP_DIR" "$OUT" /marimo/logs assets/weights
echo "=== mulai $(date -u +%FT%TZ) ==="
echo "exp=$EXP data=$DATA epochs=$EPOCHS batch=$BATCH workers=$WORKERS"
echo "klip: $(ls -1 "$DATA"/*.wav 2>/dev/null | wc -l)"

step() { echo; echo "===== $1 ($(date -u +%H:%M:%S)) ====="; }
fail() { echo "!!! GAGAL di tahap $1${2:+ (exit $2)}"; exit "${2:-1}"; }

# extract_f0.py dan train script memakai path relatif terhadap cwd /marimo/rvc
[ -f assets/rmvpe/rmvpe.pt ] || fail "aset rmvpe (assets/rmvpe/rmvpe.pt tidak ada)" 2
[ -f assets/hubert_base/pytorch_model.bin ] || fail "aset hubert_base" 2

step "1/7 preprocess (slicing + 16k)"
$PY train/preprocess.py "$DATA" 40000 "$WORKERS" "$EXP_DIR" False 12 || fail preprocess 10
GT=$(ls -1 "$EXP_DIR"/0_gt_wavs/*.wav 2>/dev/null | wc -l)
W16=$(ls -1 "$EXP_DIR"/1_16k_wavs/*.wav 2>/dev/null | wc -l)
echo "gt_wavs=$GT wavs16k=$W16"
[ "$GT" -gt 0 ] && [ "$W16" -eq "$GT" ] || fail "preprocess (gt=$GT, 16k=$W16)" 10

step "2/7 ekstraksi f0 (rmvpe)"
$PY train/dataset/extract_f0.py cuda 1 0 0 "$EXP_DIR" True || fail f0 11
F0=$(ls -1 "$EXP_DIR"/2a_f0/*.npy 2>/dev/null | wc -l)
NSF=$(ls -1 "$EXP_DIR"/2b-f0nsf/*.npy 2>/dev/null | wc -l)
echo "f0=$F0 f0nsf=$NSF (harus $GT)"
[ "$F0" -eq "$GT" ] && [ "$NSF" -eq "$GT" ] || fail "f0 (f0=$F0, f0nsf=$NSF, gt=$GT)" 11

step "3/7 ekstraksi fitur HuBERT v2"
$PY train/dataset/extract_hubert_feature.py cuda:0 1 0 "$EXP_DIR" v2 True || fail hubert 12
FEAT=$(ls -1 "$EXP_DIR"/3_feature768/*.npy 2>/dev/null | wc -l)
echo "fitur=$FEAT (harus $GT)"
[ "$FEAT" -eq "$GT" ] || fail "hubert (fitur=$FEAT, gt=$GT)" 12

step "4/7 config eksperimen"
cp configs/v2/40k.json "$EXP_DIR/config.json" || fail config 13

step "5/7 filelist (tidak dibuat otomatis oleh RVC)"
$PY - "$EXP_DIR" <<'PYEOF' || fail filelist 14
import os
import sys

exp_dir = sys.argv[1].rstrip("/")
gt_dir = os.path.join(exp_dir, "0_gt_wavs")
feat_dir = os.path.join(exp_dir, "3_feature768")
f0_dir = os.path.join(exp_dir, "2a_f0")
nsf_dir = os.path.join(exp_dir, "2b-f0nsf")

rows, missing = [], []
for name in sorted(os.listdir(gt_dir)):
    if not name.endswith(".wav"):
        continue
    stem = os.path.splitext(name)[0]
    paths = [
        os.path.join(gt_dir, name),
        os.path.join(feat_dir, stem + ".npy"),
        os.path.join(f0_dir, name + ".npy"),
        os.path.join(nsf_dir, name + ".npy"),
    ]
    absent = [p for p in paths if not os.path.exists(p)]
    if absent:
        missing.append((name, absent))
        continue
    rows.append("|".join(paths + ["0"]))

if not rows:
    print("filelist kosong: tidak ada klip dengan gt+fitur+f0+f0nsf lengkap")
    sys.exit(1)

out = os.path.join(exp_dir, "filelist.txt")
with open(out, "w", encoding="utf8") as f:
    f.write("\n".join(rows) + "\n")

print(f"filelist: {len(rows)} entri -> {out}")
if missing:
    print(f"dilewati {len(missing)} klip (fitur tidak lengkap)")
    for name, absent in missing[:10]:
        print(f"  - {name}: {absent}")
PYEOF

step "6/7 training"
$PY train/train.py -se "$SAVE_EVERY" -te "$EPOCHS" \
  -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth \
  -g 0 -bs "$BATCH" -e "$EXP" -sr 40k -sw 1 -v v2 -f0 1 -l 1 -c 0 \
  || fail train 15

step "7/7 index faiss"
$PY train/train_index.py "$EXP" v2 "$EXP_DIR" "$WORKERS" || fail index 16

shopt -s nullglob
PWTS=(assets/weights/"$EXP"*.pth)
[ "${#PWTS[@]}" -gt 0 ] || fail "copy-pth (tidak ada .pth di assets/weights)" 17
cp "${PWTS[@]}" "$OUT"/ || fail copy-pth 17
[ -f "$OUT/$EXP.pth" ] || echo "PERINGATAN: $EXP.pth final tidak ada, hanya checkpoint per-epoch"
for idx in "$EXP_DIR"/added_*.index; do
  cp "$idx" "$OUT/$EXP.index"
done

echo
echo "=== selesai $(date -u +%FT%TZ) ==="
ls -l "$OUT"
