# Voice-Model — RVC Voice Changer

## Overview

Melatih model suara sendiri (RVC v2) di marimo molab, lalu memakainya sebagai
voice changer real-time di PC lokal dengan w-okada/voice-changer.

- Training: molab, RTX Pro 6000 Blackwell 96 GB (sesi maks. 12 jam, idle 90 menit).
- Inference: PC lokal, GTX 1650 4 GB / i5-13400F / 16 GB RAM.
- Tujuan: fun, single user, tanpa web app.

## Status

- Fase 0: selesai. `jq`, skill `marimo-pair`, struktur folder, `cloud/molab_setup.sh`
  (idempoten), `cloud/molab_fetch.sh`, `scripts/molab_push.py`.
- Fase 1: selesai. 351 klip, 35.0 menit, 160 MB di `dataset/clips/`, lolos
  `scripts/inspect_dataset.py`. `Pokemon` dikecualikan (music bed).
- Fase 2: selesai. Run penuh 26 Okt 2026 di molab: 351 klip, 300 epoch
  (~31 menit, 6.1 s/epoch, batch 8), 13 checkpoint + final. Artefak di
  `models/myvoice/`: `myvoice.pth` 55.2 MB (valid, 463 entri zip) dan
  `myvoice.index` 200.5 MB (FAISS IVF1627, dim 768). Sesi sandbox itu sudah
  tidak dipakai lagi; sumber bisa diulang dari `cloud/README.md`.
- Fase 3: belum mulai (real-time). Uji dengar model: **cukup memuaskan**,
  suara masih agak robot — dugaan sampel kurang bersih, bukan kurang epoch.

## Repo

Publik: <https://github.com/Kizter/RVC_JARFISHH> (kode + dokumentasi saja;
`raw/`, `model/`, `dataset/`, `models/`, `tools/`, `.venv/` di-ignore lewat
`.gitignore`). Dokumentasi proyek: `README.md`, `docs/training-run.md`,
`cloud/README.md`.

## Stack

| Bagian | Tool |
|---|---|
| Dataset | Python 3.13 (`.venv`), ffmpeg/ffprobe 7.1, numpy 2.5 |
| Training | RVC main + torch 2.11.0+cu130 sistem (Blackwell sm_120), Python 3.13 |
| Remote kernel | marimo-pair (`execute-code.sh`), payload < 32 KB |
| Transfer dataset | filebin.net (sandbox tidak punya endpoint upload) |
| Shell | Git Bash + `tools\jq.exe` |
| Inference | w-okada/voice-changer (Windows) |

## Commands

### Lokal

```powershell
# Fase 1: potong dataset + QC
.venv\Scripts\python.exe scripts\prepare_dataset.py --exclude POKEMON --clean
.venv\Scripts\python.exe scripts\inspect_dataset.py
.venv\Scripts\python.exe scripts\inspect_dataset.py --sources

# Kirim script ke molab (menulis probe; jalankan hasilnya lewat execute-code.sh)
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin-filebin> --start setup
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin-filebin> --start fetch
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin-filebin> --start pipeline --env EXP=myvoice EPOCHS=300

# Cek tool
tools\jq.exe --version
ffprobe -version
```

### Molab (marimo-pair)

```powershell
$env:MARIMO_TOKEN='<token>'
$env:PATH="E:\Voice-Model\tools;$env:PATH"
& "C:\Program Files\Git\bin\bash.exe" -c "cd /e/Voice-Model && bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <URL> -c '<code>'"
```

Panggilan pertama ke kernel wajib:

```python
import marimo._code_mode as cm; help(cm)
```

Jangan edit file notebook `.py` molab langsung; gunakan `cm` lewat
`execute-code.sh`.

## Struktur

```
Voice-Model/
├─ model/            # stem vokal sumber (read-only)
├─ raw/              # mix sumber (read-only)
├─ dataset/
│  ├─ clips/         # 40 kHz mono 16-bit, 3-10 detik (hasil Fase 1)
│  ├─ rvc-clips-0*.zip  # paket unggah ke filebin (dibuat scripts/make_upload_zips.py)
│  └─ manifest.json  # asal tiap klip + statistik per sumber
├─ models/           # .pth + .index hasil training
├─ scripts/          # prepare_dataset, inspect_dataset, molab_push
├─ cloud/            # molab_setup.sh, molab_fetch.sh, molab_pipeline.sh, README.md
├─ tools/            # jq portable
├─ .venv/            # Python lokal (numpy) untuk script dataset
├─ .agents/skills/   # marimo-pair
├─ AGENTS.md
└─ README.md
```

## Konvensi

- Token molab hanya lewat `MARIMO_TOKEN`; jangan tulis ke file, jangan commit.
- `model/` dan `raw/` read-only. Semua audio turunan ke `dataset/`.
- Klip training: 40 kHz, mono, 16-bit, 3-10 detik, tanpa clipping.
- Artefak model ke `models/<nama-model>/`.
- Satu fase selesai dan diverifikasi sebelum fase berikutnya.
- Jangan hapus sumber audio; dataset bisa dibuat ulang.

## Gotchas

- **Molab 410 `sandbox terminated`** = sesi sudah ditutup (idle 90 menit atau
  sesi 12 jam). Buka notebook baru, ambil URL/token baru, lalu jalankan ulang
  `cloud/molab_setup.sh` — isi sandbox lama hilang semua.
- **Payload kernel < 32 KB**: jangan kirim file besar lewat `-c` /
  `execute-code.sh` (`Argument list too long`). Pakai `scripts/molab_push.py`.
- **Torch**: jangan pasang ke venv molab. uv menarik torch PyPI (tanpa sm_120)
  dan menimpa torch sistem 2.11+cu130; setup membuangnya otomatis. Blackwood
  butuh cu128/cu130 — w-okada pin torch 2.0.1, jadi jangan dipakai di molab.
- **numpy**: tidak ada wheel numpy<2 untuk Python 3.13; RVC diuji jalan di 2.x.
- **Path rmvpe**: `extract_f0.py` membaca `assets/rmvpe/rmvpe.pt`, file dari HF
  bernama `assets/rmvpe.pt` — setup memindahkannya.
- **filelist.txt**: tidak dibuat otomatis oleh RVC; `train.py` langsung gagal
  tanpanya. Pipeline membuatnya (gt|fitur|f0|f0nsf|speaker).
- **assets/weights/**: `savee()` tidak membuat direktorinya; tanpa folder itu
  checkpoint "Success" palsu di log tapi file tidak ada.
- **Storage molab**: hasil training adalah scratch. Unduh `.pth` + `.index`
  sebelum sesi berakhir (`GET /api/files/download?path=...`).
- **Transfer dataset**: sandbox bisa mengunduh dari internet (HF, filebin).
  Uguu/tmpfiles menolak `.zip`; filebin butuh dua request (cookie peringatan).
- **VRAM 4 GB**: chunk 112-192, extra 8192, tutup aplikasi GPU lain.
- **Dataset**: file `Pokemon` punya hampir tanpa jeda (1 silence event di 12.9
  menit, mean -16.5 dBFS) = ada music bed, dikecualikan.
