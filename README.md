# RVC_JARFISHH — buat model untuk voice changer real-time

test model **RVC v2 40k** dari konten kreator https://www.youtube.com/@JARFISH di GPU
cloud, lalu memakainya sebagai voice changer real-time di PC lokal.


Repository ini berisi **kode dan dokumentasi** saja. Audio sumber, dataset,
dan bobot model tidak ikut di-commit (lihat [Isi repo](#isi-repo)).

- Status: Fase 0–2 selesai, model `myvoice` sudah jadi dan dipakai.
- Training: marimo molab, RTX Pro 6000 Blackwell 96 GB.
- Inference: PC lokal, GTX 1650 4 GB / i5-13400F / 16 GB RAM.
- Fase 3 (pasang voice changer di PC) belum dikerjakan.

## Hasil

| Item | Nilai |
|---|---|
| Model | `myvoice` — RVC v2, sample rate 40 kHz, pakai f0 (RMVPE) |
| Dataset | 351 klip, 35,0 menit, 40 kHz mono 16-bit, 3–10 detik/klip |
| Training | 300 epoch, batch 8, 12.300 step, ~31 menit di molab |
| Artefak | `myvoice.pth` 55,2 MB + `myvoice.index` 200,5 MB (FAISS IVF1627, dim 768) |

Penilaian pemilik model setelah uji dengar: **cukup memuaskan dan cocok**,
tetapi suara masih terasa **agak robot**. Dugaan penyebab: kebersihan sampel
(vokal diambil dari campuran audio yang tidak sepenuhnya bersih, plus sedikit
artefak dari pemotongan otomatis), bukan dari arsitektur atau jumlah epoch.
Perbaikan yang mungkin di kemudian hari: dataset yang lebih bersih/noise-free,
pengaturan `index rate` dan `pitch` di sisi inference, atau latih ulang dengan
epoch lebih banyak sambil membandingkan checkpoint.

## Alur kerja

```
raw/ + model/            audio sumber (lokal, read-only)
      │
      ▼  scripts/prepare_dataset.py     potong adaptif + QC
dataset/clips/           351 klip 40 kHz mono
      │
      ▼  scripts/make_upload_zips.py → filebin.net → cloud/molab_fetch.sh
sandbox molab            /marimo/data/clips
      │
      ▼  cloud/molab_pipeline.sh        7 tahap dengan gate per tahap
      │    preprocess → f0 RMVPE → HuBERT v2 → config → filelist → train → FAISS
      ▼
marimo/out/myvoice.pth + myvoice.index
      │
      ▼  unduh lewat /api/files/download
models/myvoice/          (lokal)
      │
      ▼  Fase 3: w-okada/voice-changer
suara real-time di PC
```

Kenapa dibagi dua: training butuh GPU besar (PC lokal tidak kuat), sedangkan
inference harus lokal supaya latensinya rendah dan tidak terkena batas sesi
molab (12 jam / idle 90 menit).

## Isi repo

| Path | Isi |
|---|---|
| `scripts/prepare_dataset.py` | Pemotong dataset (VAD energi adaptif) + QC klip |
| `scripts/inspect_dataset.py` | Gate kualitas: durasi, level, clipping, format |
| `scripts/make_upload_zips.py` | Bagi klip jadi 4 zip untuk diunggah ke filebin |
| `scripts/molab_push.py` | Kirim script cloud ke sandbox molab (base64) + jalankan tahap |
| `cloud/molab_setup.sh` | Bangun ulang sandbox dari nol (repo, venv, aset, config) |
| `cloud/molab_fetch.sh` | Unduh + unzip dataset di sandbox |
| `cloud/molab_pipeline.sh` | Pipeline training 7 tahap dengan gate jumlah file |
| `cloud/README.md` | Runbook satu sesi molab dari awal sampai unduh artefak |
| `docs/training-run.md` | Catatan run training `myvoice` (angka, log, masalah yang ditemui) |
| `AGENTS.md` | Konvensi kerja, perintah, dan daftar jebakan teknis |
| `.agents/skills/` | Skill `marimo-pair` (kendali kernel molab dari lokal) |

## Yang tidak ada di repo

`.gitignore` menahan semuanya di bawah ini, jadi repo tetap kecil dan audio
pribadi tidak ikut terunggah:

| Path | Isi | Alasan |
|---|---|---|
| `raw/`, `model/` | Audio sumber (mix dan stem vokal) | Data pribadi, read-only |
| `dataset/` | 351 klip hasil potong + manifest | Data pribadi, bisa dibuat ulang |
| `models/` | `myvoice.pth` (55 MB), `myvoice.index` (200 MB) | Biner besar; GitHub menolak file >100 MB |
| `tools/` | `jq.exe` portable | Biner pihak ketiga |
| `.venv/` | Python lokal untuk script dataset | Bisa dibuat ulang |

Model disimpan lokal di `models/myvoice/` beserta `README.md` berisi metadata
training dan setelan inference awal.

## Reproduksi

Butuh: Git Bash, `ffmpeg`/`ffprobe`, Python 3.13 (`.venv`), akun molab dengan
GPU, dan koneksi internet (filebin + Hugging Face).

1. **Dataset** — potong audio sumber dan cek kualitasnya:

   ```powershell
   .venv\Scripts\python.exe scripts\prepare_dataset.py --exclude POKEMON --clean
   .venv\Scripts\python.exe scripts\inspect_dataset.py
   ```

2. **Sesi molab** — buka notebook baru, ambil URL + token, lalu:

   ```powershell
   $env:MARIMO_TOKEN='<token>'
   $env:PATH="E:\Voice-Model\tools;$env:PATH"
   .venv\Scripts\python.exe scripts\molab_push.py --bin <bin-filebin> --start setup
   ```

   Jalankan probe yang dicetak lewat `.agents/skills/marimo-pair/scripts/execute-code.sh`.
   `cloud/molab_setup.sh` selesai dalam ~92 detik.

3. **Dataset ke sandbox** — unggah zip ke filebin, lalu `--start fetch`.

4. **Training** — `--start pipeline --env EXP=myvoice EPOCHS=300 BATCH=8`.

5. **Unduh artefak** sebelum sesi berakhir (sandbox tidak persisten):

   ```powershell
   curl.exe -H "Authorization: Bearer $env:MARIMO_TOKEN" `
     -o models\myvoice\myvoice.pth `
     "https://<sandbox>/api/files/download?path=/marimo/out/myvoice.pth"
   ```

Detail perintah dan urutan lengkap ada di `cloud/README.md`.

## Hyperparameter training

| Parameter | Nilai |
|---|---|
| Arsitektur | RVC v2, `40k`, `f0=1` (RMVPE), `version=v2` |
| Pretrained | `f0G40k.pth` (G) + `f0D40k.pth` (D) dari `lj1995/VoiceConversionWebUI` |
| Epoch / batch | 300 / 8 (`save_every` 25) |
| Optimizer | default RVC (`lr` G 1e-4, D 1e-4), `cache_all_data=false` |
| Data loader | 8 worker, `per=12` di preprocess (klip sudah 3–10 detik) |
| Output | `assets/weights/myvoice.pth` + `logs/myvoice/added_*.index` |

## Setelan inference awal (GTX 1650 4 GB)

| Setelan | Nilai | Catatan |
|---|---|---|
| Model | `myvoice.pth` + `myvoice.index` | Nama dasar harus sama |
| Pitch | 0 | Geser ±12 kalau register suara berbeda |
| Index rate | 0,5–0,75 | Naikkan kalau karakter suara kurang mirip |
| Chunk | 112–192 ms | Turunkan kalau putus-putus, naikkan kalau suara pecah |
| Extra | 8192 | Beban VRAM paling ringan di 4 GB |
| f0 method | RMVPE | Paling akurat untuk suara manusia |

Tutup aplikasi lain yang memakai GPU (browser dengan video, game) karena VRAM
4 GB cepat penuh.

## Jebakan teknis yang sudah ditemui

Daftar lengkap ada di `AGENTS.md`; yang paling mahal waktunya:

- **Payload kernel marimo < 32 KB.** File besar lewat `execute-code.sh` gagal
  dengan `Argument list too long`; script dikirim sebagai base64 dalam satu probe.
- **Jangan pasang torch ke venv molab.** `uv` menarik torch dari PyPI yang tidak
  punya kernel `sm_120` untuk Blackwell dan menimpa torch sistem (2.11+cu130).
- **`configs/v2/40k.json` tidak ada di upstream RVC**; isinya identik dengan
  `configs/v1/40k.json`.
- **`filelist.txt` tidak dibuat otomatis** oleh RVC, dan `savee()` tidak membuat
  `assets/weights/`; tanpa keduanya training "sukses" di log tapi tidak ada file.
- **`extract_f0.py` hardcode `assets/rmvpe/rmvpe.pt`**, sedangkan file dari
  Hugging Face bernama `assets/rmvpe.pt`.
- **Sandbox molab tidak persisten** (HTTP 410 setelah idle 90 menit); semua
  langkah setup dibuat idempoten supaya sesi baru siap dalam ~1,5 menit.

## Privasi dan lisensi

- Token molab hanya lewat environment variable, tidak pernah ditulis ke file
  atau commit; token dirotasi tiap sesi.
- Audio sumber dan model tidak dipublikasikan. Model ini hanya untuk suara
  pemiliknya sendiri.
- RVC dan w-okada/voice-changer punya lisensi masing-masing; periksa sebelum
  mendistribusikan ulang kode atau model turunan.
