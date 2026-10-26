# Catatan run training `myvoice`

Log faktual run penuh pertama (26 Oktober 2026) di marimo molab, GPU
NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB. Semua waktu di UTC.

## Timeline

| Waktu | Tahap | Hasil |
|---|---|---|
| 17:39:12 | `cloud/molab_setup.sh` | repo RVC `81eed5e`, venv + 15 paket, aset 585 MB, `configs/v2/40k.json` |
| 17:41:49 | uji import RVC | lolos, torch 2.11.0+cu130, device capability (12, 0) / `sm_120` |
| 17:42:57 | `cloud/molab_fetch.sh` | 4 zip (131,7 MB) diunduh dari filebin |
| 17:44:37 | unzip | 351 klip, 168 MB, QC format 0 error |
| 17:45:21 | `cloud/molab_pipeline.sh` mulai | 351 klip terdeteksi |
| 17:45:39 | 1/7 preprocess | `0_gt_wavs` 351, `1_16k_wavs` 351, 18 detik |
| ~17:46 | 2/7 f0 RMVPE | `2a_f0` 351, `2b-f0nsf` 351 |
| ~17:46 | 3/7 HuBERT v2 | `3_feature768` 351 |
| ~17:47 | 4–5/7 config + filelist | `filelist.txt` 351 baris (semua klip lengkap) |
| 17:47:30 | 6/7 epoch 1 | 19,3 detik (warm-up) |
| 17:47:38 | epoch 2 | 7,5 detik |
| ~17:48 | steady state | ~6,1 detik/epoch, GPU 62–80 %, VRAM ~9,9 GB |
| 18:18:32 | epoch 300 | total ~31 menit, 12.300 step |
| 18:18:43 | 7/7 index FAISS | `added_IVF1627_Flat_nprobe_1_myvoice_v2.index` |
| 18:18:43 | selesai | 13 `.pth` (12 checkpoint + final) + 1 index di `/marimo/out` |

Setup bisa diulang dari nol dalam ~92 detik; fetch ~100 detik; training ~31
menit. Total satu sesi kerja ±35 menit setelah sandbox hidup.

## Angka training

| Item | Nilai |
|---|---|
| Klip / durasi | 351 / 35,0 menit |
| Epoch / batch | 300 / 8 |
| Step | 12.300 (`_s12300`) |
| Waktu per epoch | 6,0–6,4 detik (epoch 1: 19,3 detik) |
| Checkpoint | e25 … e300, tiap 25 epoch, 55 MB/file |
| VRAM | ~9,9 GB dari 96 GB |
| Utilisasi GPU | 62–80 % (sampel) |

Loss yang tercatat (bukan kurva penuh, hanya baris log):

| Epoch | loss_disc | loss_gen | loss_mel | loss_kl |
|---|---|---|---|---|
| 5 | 3,872 | 3,192 | 26,927 | 2,656 |
| 10 | 4,135 | 3,192 | 24,671 | 2,090 |
| 157 | 3,660 | 3,824 | 22,298 | 1,216 |
| 161 | 3,147 | 3,860 | 21,286 | 1,364 |

`loss_mel` turun ~35 → ~21 dan G/D tetap seimbang. Tidak ada tanda mode
collapse atau discriminator yang menang telak, jadi 300 epoch tidak
overfit parah untuk dataset 35 menit.

## Artefak

| File | Ukuran | Catatan |
|---|---|---|
| `myvoice.pth` | 55.223.661 B (55,2 MB) | checkpoint final, salinan `assets/weights/myvoice.pth` |
| `myvoice.index` | 200.480.419 B (200,5 MB) | FAISS IVF1627, `nprobe=1`, dim 768 |
| `myvoice_e25..e300.pth` | 55 MB masing-masing | hanya ada di sandbox, tidak diunduh |

Verifikasi lokal setelah unduh:

```
zip ok, entri: 463
['myvoice/data.pkl']
index head: b'IwFl\x00\x03...'   # 0x0300 = 768 dimensi
```

`.pth` adalah arsip zip torch yang valid (463 entri, `testzip()` bersih) dan
`.index` punya header FAISS `IwFl` dengan dimensi 768, sesuai keluaran HuBERT.

## Masalah yang muncul saat run dan perbaikannya

| Gejala | Penyebab | Perbaikan |
|---|---|---|
| `syntax error near unexpected token '>'` di setup | `PKGS=(scipy>=1.13 …)` — bash membaca `>`/`<` sebagai redirection | tiap spec diberi tanda kutip |
| `GET rvc-clips-1.zip → 404` | `seq -w 1 4` menghasilkan `1`, bukan `01` | `seq -f "%02g" 1 4` |
| `venv` punya torch tanpa `sm_120` | `uv pip install` menarik torch PyPI | torch sistem dipertahankan, torch/triton/nvidia unduhan dibuang |
| `extract_f0.py` gagal cari model | path hardcode `assets/rmvpe/rmvpe.pt`, file HF bernama `assets/rmvpe.pt` | setup memindahkan file |
| `train.py` langsung exit | `logs/<exp>/filelist.txt` tidak dibuat otomatis | pipeline membuat 351 baris dari irisan gt + fitur + f0 |
| checkpoint "Success" tapi file tidak ada | `savee()` tidak membuat `assets/weights/` | pipeline `mkdir -p assets/weights` |
| filebin balas halaman peringatan, bukan zip | butuh cookie per client | dua request: pertama ambil cookie, kedua unduh byte |

## Cara mengulang

Urutan lengkap ada di `cloud/README.md`. Ringkasnya:

```powershell
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin> --start setup
# jalankan probe lewat .agents/skills/marimo-pair/scripts/execute-code.sh
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin> --start fetch
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin> --start pipeline `
  --env EXP=myvoice EPOCHS=300 BATCH=8 SAVE_EVERY=25 WORKERS=8
```

Sebelum sesi ditutup, unduh `.pth` + `.index` (dan checkpoint per-epoch kalau
ingin membandingkan). Sandbox kembali `410 sandbox terminated` setelah idle
90 menit dan seluruh isinya hilang.

## Catatan kualitas (uji dengar)

Uji dengar saat dipakai menilai hasilnya **cukup memuaskan dan cocok**, dengan catatan
suara masih **agak robot**. Kandidat penyebab dan langkah berikutnya:

1. **Kebersihan sampel** — vokal berasal dari campuran yang tidak sepenuhnya
   bersih, dan pemotongan otomatis ikut membawa sisa noise/ruang. Dataset
   bersih (rekaman langsung tanpa musik dan reverb) biasanya paling terasa
   efeknya.
2. **Setelan inference** — `index rate` terlalu tinggi membuat suara "keras"
   dan metalik; pitch yang tidak cocok menambah artefak. Uji dulu 0,5–0,75
   dan pitch 0 sebelum memutuskan melatih ulang.
3. **Durasi data** — 35 menit sudah cukup untuk RVC v2, tetapi menambah
   10–20 menit vokal bersih umumnya memperbaiki naturalitas.
