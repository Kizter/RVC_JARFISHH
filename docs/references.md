# Referensi & inspirasi

Kumpulan sumber yang dipakai, dibaca, atau jadi pembanding proyek ini. Semua
tautan diperiksa hidup per Oktober 2026.

Urutan bagian mengikuti kebutuhan proyek: proyek inti → komponen model →
inference real-time → kebersihan dataset → infrastruktur → proyek sejenis.

## 1. Proyek inti (RVC)

| Sumber | Kegunaan |
|---|---|
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Repo yang di-clone di sandbox molab. Klaim: model bagus dari data ≥10 menit, latensi real-time 170 ms (90 ms dengan ASIO). MIT. |
| [docs/en/README.en.md](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md) | Daftar dependensi per GPU (cu118/cu128), struktur `assets/`, cara unduh aset — dasar dari `cloud/molab_setup.sh`. |
| [docs/en/training_tips_en.md](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/training_tips_en.md) | Alur 7 tahap versi upstream: split per 4 detik (overlap 0,3), f0, HuBERT, faiss index. |
| [docs/en/faiss_tips_en.md](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/faiss_tips_en.md) | Cara kerja retrieval + rumus `n_ivf = 4·√N … 16·√N`; menjelaskan kenapa index run ini `IVF1627` dengan `nprobe=1` sudah wajar. |
| [Wiki FAQ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) | Jawaban masalah umum training/inference. |
| [Retrieval-based-Voice-Conversion (framework)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion) | Kerangka asli berbasis VITS; berguna untuk memahami arsitektur, bukan untuk pemakaian harian. |

## 2. Komponen di dalam model

| Sumber | Peran di proyek ini |
|---|---|
| [HuBERT (arXiv 2106.07447)](https://arxiv.org/abs/2106.07447) | Content encoder: audio → 768-dim fitur (`3_feature768`), dasar pencarian index. |
| [ContentVec](https://github.com/auspicious3000/contentvec) · [arXiv 2207.06088](https://arxiv.org/abs/2207.06088) | Varian content encoder yang membuat fitur lebih bebas dari warna suara; konsep yang sama dipakai RVC v2. |
| [VITS](https://github.com/jaywalnut310/vits) · [arXiv 2106.06103](https://arxiv.org/abs/2106.06103) | Generator/discriminator yang diadopsi RVC (`f0G40k.pth` / `f0D40k.pth` adalah turunannya). |
| [HiFi-GAN](https://github.com/jik876/hifi-gan) | Vocoder di dalam generator; sumber artefak "metalik" kalau data latih kotor. |
| [RMVPE (arXiv 2306.15412)](https://arxiv.org/abs/2306.15412) · [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) · [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE) | Ekstraksi pitch; `assets/rmvpe/rmvpe.pt` yang dipakai pipeline. |
| [faiss](https://github.com/facebookresearch/faiss) | Approximate nearest neighbour untuk retrieval fitur; `train_index.py` menulis `myvoice.index`. |
| [audio-slicer](https://github.com/openvpi/audio-slicer) | Algoritma pemotongan berbasis silence yang menginspirasi `scripts/prepare_dataset.py`. |

## 3. Inference real-time

| Sumber | Kegunaan |
|---|---|
| [w-okada/voice-changer](https://github.com/w-okada/voice-changer) | VCClient — rencana Fase 3. Mendukung RVC `.pth` (v1) dan torch+ONNX (v2). |
| [Tutorial RVC untuk VCClient](https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_en_latest.md) | Penjelasan `CHUNK`, `EXTRA`, `INDEX`, `TUNE`, pilihan f0 (dio/harvest/crepe/rmvpe), dan menu **export to onnx**. Sumber utama setelan di README proyek ini. |
| [Binary VCClient (Hugging Face)](https://huggingface.co/wok000/vcclient000) | Unduhan rilis Windows: `vcclient_win_cuda_*` untuk NVIDIA. |
| [Branch v.2](https://github.com/w-okada/voice-changer/tree/v.2) | Versi yang sedang dikembangkan; mendukung Beatrice v2 dan torch/onnx sekaligus. |
| [RVC Realtime VST (resmi, di dalam repo RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main/RVCRealtimeVST) | Plugin VST2/VST3 Windows yang memuat `.pth` + `.index` langsung (tanpa ONNX). Block 20–1000 ms, context 500–3000 ms, f0 RMVPE/FCPE/PM. Alternatif VCClient kalau ingin dipakai di DAW/OBS. |
| [dev6699/rvc-onnx](https://github.com/dev6699/rvc-onnx) | Wrapper ONNX Runtime untuk RVC; jalur cepat kalau ingin inference tanpa PyTorch. |
| [Panduan W-Okada di Hugging Face](https://huggingface.co/blog/Lenylvt/w-okada) | Panduan komunitas langkah demi langkah, termasuk perbandingan `.pth` vs `.onnx`. |
| [VB-Audio Cable](https://vb-audio.com/Cable/) | Routing output voice changer ke Discord/OBS sebagai mikrofon virtual. |

Catatan ekspektasi latensi (dari dokumentasi masing-masing): RVC real-time GUI
170 ms (90 ms dengan ASIO), VCClient bergantung `CHUNK` + `EXTRA`, VST RVC
melaporkan latensi = 2× block, Seed-VC ~300 ms algoritma + ~100 ms perangkat.

## 4. Kebersihan dataset (paling relevan untuk suara "agak robot")

| Sumber | Isi yang berguna |
|---|---|
| [AI Hub — Dataset & Isolation](https://docs.aihub.gg/rvc/resources/dataset-isolation/) | Panduan paling lengkap: 15 menit minimum / 40+ menit untuk natural, variasi pitch, dan daftar penyebab artefak robotik (audio kotor, reverb, suara tumpang tindih, overfit/undertrain). Urutan bersih-bersih: hapus instrumen → de-reverb → ambil vokal utama → de-noise. |
| [AI Hub — Inference settings](https://docs.aihub.gg/rvc/resources/inference-settings/) | Arti *search feature ratio* (index rate), protection, dan pengaruhnya ke artefak. |
| [AI Hub — Applio](https://docs.aihub.gg/rvc/local/applio/) | Alternatif UI training yang lebih ramah; berguna kalau nanti ingin membandingkan hasil. |
| [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui) | Pemisah vokal/de-reverb/de-noise; dipakai untuk membersihkan sumber sebelum dipotong ulang. |
| [Dokumentasi RVC — training tips](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/training_tips_en.md) | Menegaskan split pendek (4 detik) dan normalisasi per potongan; di proyek ini `per=12` dipakai karena klip sudah 3–10 detik. |

## 5. Infrastruktur yang dipakai repo ini

| Sumber | Peran |
|---|---|
| [marimo](https://marimo.io) · [docs](https://docs.marimo.io) | Notebook reaktif tempat sandbox molab berjalan. |
| [marimo-team/marimo-pair](https://github.com/marimo-team/marimo-pair) | Skill yang dipakai `scripts/molab_push.py` + `execute-code.sh` untuk mengirim kode ke kernel. |
| [Hugging Face — lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | Sumber `hubert_base`, `rmvpe.pt`, `f0G40k.pth`, `f0D40k.pth`. |
| [ffmpeg](https://ffmpeg.org) | Decode/encode audio di `scripts/prepare_dataset.py` dan RVC. |
| [filebin.net](https://filebin.net) | Perantara transfer dataset ke sandbox (tanpa endpoint upload). |

## 6. Proyek sejenis (inspirasi)

| Proyek | Yang bisa dipinjam |
|---|---|
| [Applio](https://github.com/IAHispano/Applio) — fork RVC | UI training lebih rapi, normalisasi LUFS, preset; kandidat kalau ingin melatih ulang tanpa pipeline sendiri. |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) | Pendekatan SoftVC + VITS + NSF untuk suara nyanyi; arah kalau nanti ingin model yang bisa menyanyi. |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) · [arXiv 2411.09943](https://arxiv.org/abs/2411.09943) | Zero-shot VC + real-time (~300 ms) tanpa training; pembanding menarik untuk suara yang natural, walau bukan `.pth` RVC. |
| [RVC Realtime VST](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main/RVCRealtimeVST) | Contoh arsitektur audio real-time: ring buffer lock-free + worker Python terpisah, tanpa HTTP. |
| [dev6699/rvc-onnx](https://github.com/dev6699/rvc-onnx) | Contoh menjalankan RVC tanpa PyTorch, cocok untuk GPU kecil. |

## Kalau suara masih robot

Urutan percobaan yang paling murah lebih dulu:

1. **Turunkan index rate** ke 0,3–0,5. AI Hub menyebut menurunkan *search
   feature ratio* bisa mengurangi artefak; VCClient menjelaskan `INDEX=0`
   memakai output HuBERT apa adanya dan `INDEX=1` memberi bobot penuh ke fitur
   latih.
2. **Cek pitch** (`TUNE`). Selisih register membuat hasil terdengar tegang dan
   metalik; coba 0, ±2, ±12 tergantung suara asal vs target.
3. **Mainkan CHUNK/EXTRA.** `EXTRA` terlalu besar menambah waktu konversi dan
   kadang menggeser timbre; `CHUNK` kecil menambah pecah-pecah.
4. **Bersihkan dataset**: ulangi pemisahan vokal dengan UVR (de-reverb lalu
   de-noise, jangan menumpuk model), buang ruang/reverb, hindari audio hasil
   upscale, simpan lossless. Artefak robotik paling sering berasal dari sini.
5. **Bandingkan checkpoint.** Overfit membuat sibilance robotik; kalau ada
   `myvoice_e150`/`myvoice_e200`, uji beberapa epoch, bukan hanya final.
6. **Tambah data bersih** ke 40+ menit bila masih kurang, dengan variasi pitch
   dan ekspresi.

## Lisensi dan etika

- RVC WebUI: MIT. Aset pretrained mengikuti ketentuan repo Hugging Face masing-masing.
- VCClient dan RVCRealtimeVST: lisensi per proyek (VCClient menyertakan syarat
  penggunaan untuk karakter bawaan Tsukuyomi-chan/Amitaro/Kikoto Mahiro —
  tidak berlaku untuk model sendiri, tapi tetap baca sebelum distribusi).
- Seed-VC, so-vits-svc, Applio: lisensi masing-masing; periksa sebelum memakai
  kode atau modelnya secara komersial.
- Model `myvoice` dilatih dari audio publik konten kreator
  [@JARFISH](https://www.youtube.com/@JARFISH) dan tidak dipublikasikan; dipakai
  sendiri sebagai eksperimen. Referensi di atas bukan izin untuk mengkloning
  suara orang lain tanpa persetujuan, apalagi menyebarkannya.
