#!/usr/bin/env python
"""Potong stem vokal menjadi klip training RVC v2.

Input : model/*.wav   (44.1 kHz stereo, read-only)
Output: dataset/clips/*.wav  (40 kHz mono 16-bit, 3-10 detik)
        dataset/manifest.json

Alur per file:
  1. decode lewat ffmpeg -> 40 kHz mono float32 (tanpa file perantara)
  2. VAD energi adaptif (frame 25.6 ms) + median filter + merge gap
  3. potong tiap region bicara jadi klip 3-10 detik, batas di titik energi
     terendah supaya tidak memotong suku kata
  4. buang klip yang terlalu sepi / terlalu pelan / clipping parah
  5. tulis WAV 16-bit + fade 5 ms

Contoh:
  .venv\\Scripts\\python.exe scripts\\prepare_dataset.py --exclude POKEMON
  .venv\\Scripts\\python.exe scripts\\prepare_dataset.py --limit 1 --target-minutes 3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 40000
FRAME = 1024  # 25.6 ms @ 40 kHz
HOP = 1024


# --------------------------------------------------------------------------- io


def decode(path: Path, sr: int) -> np.ndarray:
    """Decode audio apa pun ke mono float32 lewat ffmpeg."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg gagal: {path.name}\n{proc.stderr.decode('utf-8', 'replace')[:800]}"
        )
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if audio.size == 0:
        raise RuntimeError(f"audio kosong: {path.name}")
    return audio


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# -------------------------------------------------------------------------- vad


def frame_db(audio: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (audio.size - FRAME) // HOP)
    if n <= 0:
        return np.array([-120.0], dtype=np.float32)
    trimmed = audio[: FRAME + (n - 1) * HOP]
    frames = np.lib.stride_tricks.sliding_window_view(trimmed, FRAME)[::HOP]
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-8)


def median_filter(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or mask.size < size:
        return mask
    pad = size // 2
    padded = np.pad(mask.astype(np.int8), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return windows.sum(axis=1) * 2 > size


def speech_mask(db: np.ndarray) -> np.ndarray:
    p10, p50 = np.percentile(db, 10), np.percentile(db, 50)
    threshold = min(max(p10 + 12.0, -55.0), -30.0)
    threshold = min(threshold, p50 - 3.0)  # jangan sampai semua dianggap bicara
    mask = db > threshold
    return median_filter(mask, 7)  # ~180 ms


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index frame awal/akhir (eksklusif) untuk setiap blok True."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def merge_runs(blocks: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not blocks:
        return []
    merged = [list(blocks[0])]
    for start, end in blocks[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


# ----------------------------------------------------------------------- slicing


@dataclass
class Stats:
    source: str
    duration: float = 0.0
    speech_ratio: float = 0.0
    peak_dbfs: float = 0.0
    clip_ratio: float = 0.0
    kept: int = 0
    kept_seconds: float = 0.0
    skipped_quiet: int = 0
    skipped_clipping: int = 0
    skipped_short: int = 0
    notes: list[str] = field(default_factory=list)


def slug(name: str, limit: int = 42) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return base[:limit].strip("-") or "audio"


def slice_file(
    path: Path,
    sr: int,
    min_len: float,
    max_len: float,
    target_len: float,
    max_gap: float,
    max_peak: float,
    min_peak_dbfs: float,
    min_speech_ratio: float,
    max_clip_ratio: float,
) -> tuple[list[tuple[int, int, np.ndarray]], Stats]:
    audio = decode(path, sr)
    db = frame_db(audio)
    mask = speech_mask(db)
    stats = Stats(source=path.name)
    stats.duration = audio.size / sr
    stats.speech_ratio = float(mask.mean()) if mask.size else 0.0
    peak = float(np.abs(audio).max())
    stats.peak_dbfs = 20.0 * np.log10(peak + 1e-9)
    # clipping sejati = sampel yang menempel di puncak file (bukan ambang tetap,
    # karena sumber float bisa dinormalisasi di atas 0 dBFS tanpa distorsi)
    stats.clip_ratio = float(np.mean(np.abs(audio) >= peak * 0.999)) if peak > 0 else 0.0

    min_frames = int(round(0.35 * sr / HOP))
    blocks = [(a, b) for a, b in runs(mask) if b - a >= min_frames]
    blocks = merge_runs(blocks, int(round(max_gap * sr / HOP)))

    min_samples = int(min_len * sr)
    max_samples = int(max_len * sr)
    target_samples = int(target_len * sr)

    pieces: list[tuple[int, int]] = []
    join_gap = int(3.0 * sr)  # blok pendek boleh disambung lewat jeda maks 3 detik
    i = 0
    while i < len(blocks):
        start_f, end_f = blocks[i]
        start = start_f * HOP
        end = min(audio.size, end_f * HOP + FRAME)
        # blok < min_len disambung dengan blok berikutnya, bukan dibuang
        while (
            end - start < min_samples
            and i + 1 < len(blocks)
            and blocks[i + 1][1] * HOP + FRAME - start <= max_samples
            and (blocks[i + 1][0] - blocks[i][1]) * HOP <= join_gap
        ):
            i += 1
            end = min(audio.size, blocks[i][1] * HOP + FRAME)
        if end - start < min_samples:
            stats.skipped_short += 1
        else:
            pieces.append((start, end))
        i += 1

    sliced: list[tuple[int, int]] = []
    for start, end in pieces:
        pos = start
        while end - pos > max_samples:
            lo = max(pos + min_samples, pos + target_samples - int(1.5 * sr))
            hi = min(pos + max_samples, pos + target_samples + int(1.5 * sr))
            lo_f, hi_f = lo // HOP, max(lo // HOP + 1, hi // HOP)
            cut = (lo_f + int(np.argmin(db[lo_f:hi_f]))) * HOP
            cut = min(max(cut, pos + min_samples), pos + max_samples)
            sliced.append((pos, cut))
            pos = cut
        if end - pos >= min_samples:
            sliced.append((pos, end))
        else:
            stats.skipped_short += 1
    pieces = sliced

    # fade 5 ms supaya tidak ada klik di batas potongan
    fade = int(0.005 * sr)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)

    kept: list[tuple[int, int, np.ndarray]] = []
    for start, end in pieces:
        clip = audio[start:end].copy()
        clip_peak = float(np.abs(clip).max())
        if clip_peak < 10 ** (min_peak_dbfs / 20.0):
            stats.skipped_quiet += 1
            continue
        frames = clip[: (clip.size // HOP) * HOP].reshape(-1, HOP)
        ratio = float(np.mean(np.sqrt(np.mean(frames**2, axis=1)) > 10 ** (-42 / 20.0)))
        if ratio < min_speech_ratio:
            stats.skipped_quiet += 1
            continue
        if float(np.mean(np.abs(clip) >= clip_peak * 0.999)) > max_clip_ratio:
            stats.skipped_clipping += 1
            continue
        if clip_peak > max_peak:  # turunkan linear, tanpa distorsi
            clip *= max_peak / clip_peak
        if clip.size > 2 * fade:
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        kept.append((start, end, clip))
        stats.kept += 1
        stats.kept_seconds += clip.size / sr

    if stats.clip_ratio > 0.001:
        stats.notes.append(f"clipping {stats.clip_ratio * 100:.2f}% sampel")
    return kept, stats


# -------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="model", help="folder sumber (default: model)")
    ap.add_argument("--out", default="dataset/clips", help="folder output")
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--min-len", type=float, default=3.0, dest="min_len")
    ap.add_argument("--max-len", type=float, default=10.0, dest="max_len")
    ap.add_argument("--target-len", type=float, default=7.5, dest="target_len")
    ap.add_argument(
        "--max-gap",
        type=float,
        default=0.6,
        dest="max_gap",
        help="jeda maksimum yang tetap digabung dalam satu klip (detik)",
    )
    ap.add_argument("--max-peak", type=float, default=0.95, dest="max_peak")
    ap.add_argument("--min-peak-dbfs", type=float, default=-30.0, dest="min_peak_dbfs")
    ap.add_argument("--min-speech-ratio", type=float, default=0.5, dest="min_speech_ratio")
    ap.add_argument("--max-clip-ratio", type=float, default=0.01, dest="max_clip_ratio")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="substring nama file yang dilewati (boleh diulang)",
    )
    ap.add_argument("--limit", type=int, default=0, help="proses maksimal N file")
    ap.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        dest="max_per_source",
        help="batasi jumlah klip per file sumber (0 = semua)",
    )
    ap.add_argument(
        "--target-minutes",
        type=float,
        default=0.0,
        dest="target_minutes",
        help="berhenti setelah ±N menit klip terkumpul (0 = semua)",
    )
    ap.add_argument("--clean", action="store_true", help="hapus isi folder output dulu")
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    if not src.is_dir():
        print(f"folder sumber tidak ada: {src}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for old in out.glob("*.wav"):
            old.unlink()
            removed += 1
        if removed:
            print(f"hapus {removed} klip lama")

    files = sorted(p for p in src.glob("*.wav") if p.is_file())
    if args.exclude:
        files = [p for p in files if not any(pat.lower() in p.name.lower() for pat in args.exclude)]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("tidak ada file sumber yang cocok", file=sys.stderr)
        return 1

    budget = args.target_minutes * 60.0
    total_seconds = 0.0
    manifest: dict = {
        "sr": args.sr,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "target_len": args.target_len,
        "exclude": args.exclude,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": [],
        "clips": [],
    }
    used_slugs: dict[str, int] = {}

    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}")
        try:
            kept, stats = slice_file(
                path,
                sr=args.sr,
                min_len=args.min_len,
                max_len=args.max_len,
                target_len=args.target_len,
                max_gap=args.max_gap,
                max_peak=args.max_peak,
                min_peak_dbfs=args.min_peak_dbfs,
                min_speech_ratio=args.min_speech_ratio,
                max_clip_ratio=args.max_clip_ratio,
            )
        except Exception as exc:  # noqa: BLE001 - satu file gagal tidak boleh menghentikan batch
            print(f"  ! gagal: {exc}")
            manifest["sources"].append({"file": path.name, "error": str(exc)})
            continue

        if args.max_per_source and len(kept) > args.max_per_source:
            pick = np.linspace(0, len(kept) - 1, args.max_per_source).round().astype(int)
            kept = [kept[i] for i in sorted(set(pick.tolist()))]

        base = slug(path.stem)
        if base in used_slugs:
            used_slugs[base] += 1
            base = f"{base}-{used_slugs[base]}"
        else:
            used_slugs[base] = 0

        written = 0
        for n, (start, end, clip) in enumerate(kept):
            if budget and total_seconds >= budget:
                break
            name = f"{base}__{n:04d}.wav"
            write_wav(out / name, clip, args.sr)
            total_seconds += clip.size / args.sr
            written += 1
            manifest["clips"].append(
                {
                    "clip": name,
                    "source": path.name,
                    "start_s": round(start / args.sr, 3),
                    "end_s": round(end / args.sr, 3),
                    "duration_s": round(clip.size / args.sr, 3),
                }
            )

        stats.kept = written
        stats.kept_seconds = sum(c["duration_s"] for c in manifest["clips"][-written:]) if written else 0.0
        manifest["sources"].append(stats.__dict__)
        print(
            f"  durasi {stats.duration / 60:.1f} mnt | bicara {stats.speech_ratio * 100:.0f}% "
            f"| peak {stats.peak_dbfs:.1f} dBFS | klip {written} "
            f"({stats.kept_seconds / 60:.1f} mnt) | skip senyap {stats.skipped_quiet} "
            f"clipping {stats.skipped_clipping} pendek {stats.skipped_short}"
        )
        if stats.notes:
            print(f"  catatan: {'; '.join(stats.notes)}")
        if budget and total_seconds >= budget:
            print(f"target {args.target_minutes:g} menit tercapai, berhenti")
            break

    manifest["total_clips"] = len(manifest["clips"])
    manifest["total_seconds"] = round(total_seconds, 2)
    manifest["total_minutes"] = round(total_seconds / 60.0, 2)
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    size_mb = sum(p.stat().st_size for p in out.glob("*.wav")) / 1e6
    print(
        f"\nselesai: {manifest['total_clips']} klip, "
        f"{manifest['total_minutes']:.1f} menit, {size_mb:.0f} MB -> {out}"
    )
    print(f"manifest: {out.parent / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
