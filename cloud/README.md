# Runbook molab (Fase 2 — training)

Semua perintah dijalankan dari `E:\Voice-Model` di PowerShell. Sandbox molab
hanya hidup selama sesi (12 jam, mati setelah 90 menit idle) — begitu sesi
hilang, URL/health mengembalikan **410** dan seluruh isi sandbox ikut hilang
(repo, venv, aset, dataset). Karena itu setup dibuat idempoten dan dataset bisa
diunduh ulang dari filebin.

## 0. Siapkan sesi

1. Buka notebook baru di molab, salin URL sandbox (`https://sb-xxxxxxxx.sb.molab.run/`)
   dan token.
2. Set di PowerShell (token tidak disimpan ke file):

```powershell
$env:MARIMO_TOKEN='<token>'
$env:PATH="E:\Voice-Model\tools;$env:PATH"
$URL='https://sb-xxxxxxxx.sb.molab.run/'
```

## 1. Pasang script ke sandbox

```powershell
.venv\Scripts\python.exe scripts\molab_push.py --bin <bin-filebin> --start setup
& "C:\Program Files\Git\bin\bash.exe" -c "cd /e/Voice-Model && bash .agents/skills/marimo-pair/scripts/execute-code.sh --url $URL '/c/Users/user/AppData/Local/Temp/opencode/molab_probe.py'"
```

`--start setup` menjalankan `cloud/molab_setup.sh` detached di sandbox:
clone RVC, venv yang mewarisi torch sistem (cu130, sm_120), ~15 paket Python,
unduh aset HuBERT/RMVPE/pretrained (~585 MB), buat `configs/v2/40k.json`,
lalu uji import modul training. Log: `/marimo/logs/setup.log`.

Cek progres (probe generik ada di `C:\Users\user\AppData\Local\Temp\opencode\`):

```powershell
& "C:\Program Files\Git\bin\bash.exe" -c "cd /e/Voice-Model && bash .agents/skills/marimo-pair/scripts/execute-code.sh --url $URL '/c/Users/user/AppData/Local/Temp/opencode/check_smoke.py'"
```

## 2. Ambil dataset

Dataset (351 klip, 35 menit, 4 zip ≈ 132 MB) diunggah dari PC lokal ke
filebin.net, karena sandbox tidak punya endpoint upload HTTP. Unggah ulang
kalau bin kedaluwarsa (filebin menyimpan 6 hari):

```powershell
$bin = "voicemodel-" + (Get-Date -Format "MMddHHmm")
1..4 | ForEach-Object {
  $f = "E:\Voice-Model\dataset\rvc-clips-{0:d2}.zip" -f $_
  curl.exe -sS -m 1800 -X POST --data-binary "@$f" -H "filename: $(Split-Path $f -Leaf)" `
    -H "Content-Type: application/octet-stream" "https://filebin.net/$bin/$(Split-Path $f -Leaf)"
}
```

Lalu di sandbox:

```powershell
.venv\Scripts\python.exe scripts\molab_push.py --bin $bin --start fetch
& "C:\Program Files\Git\bin\bash.exe" -c "cd /e/Voice-Model && bash .agents/skills/marimo-pair/scripts/execute-code.sh --url $URL '/c/Users/user/AppData/Local/Temp/opencode/molab_probe.py'"
```

Hasil: `/marimo/data/clips/*.wav` (351 file). Log: `/marimo/logs/fetch.log`.

## 3. Training

```powershell
.venv\Scripts\python.exe scripts\molab_push.py --bin $bin --start pipeline --env EXP=myvoice EPOCHS=300 BATCH=8 SAVE_EVERY=25
& "C:\Program Files\Git\bin\bash.exe" -c "cd /e/Voice-Model && bash .agents/skills/marimo-pair/scripts/execute-code.sh --url $URL '/c/Users/user/AppData/Local/Temp/opencode/molab_probe.py'"
```

`cloud/molab_pipeline.sh` menjalankan 7 tahap: preprocess → f0 RMVPE → fitur
HuBERT v2 → config → filelist → train → index FAISS. Setiap tahap punya gate
jumlah file, jadi kegagalan berhenti di situ (bukan lanjut dengan data kosong).
Log: `/marimo/logs/pipeline.log`; hasil: `/marimo/out/myvoice.pth` (+ checkpoint
per-epoch `myvoice_e*_s*.pth`) dan `/marimo/out/myvoice.index`.

## 4. Unduh artefak (sebelum sesi berakhir)

```powershell
$base = "https://sb-xxxxxxxx.sb.molab.run"
New-Item -ItemType Directory -Force models\myvoice | Out-Null
curl.exe -H "Authorization: Bearer $env:MARIMO_TOKEN" -o models\myvoice\myvoice.pth "$base/api/files/download?path=/marimo/out/myvoice.pth"
curl.exe -H "Authorization: Bearer $env:MARIMO_TOKEN" -o models\myvoice\myvoice.index "$base/api/files/download?path=/marimo/out/myvoice.index"
```

Verifikasi: `.pth` ≈ 55 MB, `.index` beberapa MB, lalu QC lokal (lihat Fase 3).

## Catatan yang sudah terbukti

- Torch **jangan** dipasang ke venv: uv akan menarik torch dari PyPI (tanpa
  sm_120) dan menimpa torch sistem. `molab_setup.sh` membuangnya otomatis.
- numpy tetap 2.x — tidak ada wheel numpy<2 untuk Python 3.13.
- `extract_f0.py` membaca `assets/rmvpe/rmvpe.pt`, sedangkan file dari HF bernama
  `assets/rmvpe.pt`; setup memindahkannya.
- `train.py` gagal kalau `assets/weights/` belum ada (`savee()` tidak membuat
  direktorinya) dan kalau `logs/<exp>/filelist.txt` belum dibuat — pipeline
  membuat keduanya.
- `preprocess.py` memakai `per=12` supaya slicer RVC tidak memotong klip lokal
  yang sudah 3–10 detik.
