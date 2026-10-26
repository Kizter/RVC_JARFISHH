#!/usr/bin/env python
"""QC dataset klip RVC atau file sumber.

Mode klip (default):
  .venv\\Scripts\\python.exe scripts\\inspect_dataset.py
  cek 40 kHz / mono / 16-bit / 3-10 detik, peak, RMS, rasio senyap, DC offset.

Mode sumber:
  .venv\\Scripts\\python.exe scripts\\inspect_dataset.py --sources
  durasi, sample rate, channel, peak, jumlah jeda, rasio senyap per file di model/.

Keluar dengan kode 1 kalau ada masalah, supaya bisa dipakai sebagai gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPECT_SR = 40000
EXPECT_MIN = 3.0
EXPECT_MAX = 10.0


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-9))


def read_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as w:
        channels, width, sr, frames = (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getnframes(),
        )
        raw = w.readframes(frames)
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"lebar sampel tidak didukung: {width} byte")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr, channels, width * 8


def silence_ratio(audio: np.ndarray, sr: int, threshold_db: float = -45.0) -> float:
    hop = max(1, int(0.02 * sr))
    n = audio.size // hop
    if n == 0:
        return 1.0
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12)
    return float(np.mean(20.0 * np.log10(rms + 1e-8) < threshold_db))


def inspect_clips(folder: Path) -> int:
    files = sorted(folder.glob("*.wav"))
    if not files:
        print(f"tidak ada klip di {folder}")
        return 1

    groups: dict[str, list[dict]] = defaultdict(list)
    problems: list[str] = []
    total = 0.0

    for path in files:
        try:
            audio, sr, channels, bits = read_wav(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.name}: tidak bisa dibaca ({exc})")
            continue
        seconds = audio.size / sr
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-12))
        info = {
            "name": path.name,
            "seconds": seconds,
            "sr": sr,
            "channels": channels,
            "bits": bits,
            "peak_db": db(peak),
            "rms_db": db(rms),
            "silence": silence_ratio(audio, sr),
            "dc": float(np.mean(audio)),
        }
        groups[path.name.split("__")[0]].append(info)
        total += seconds

        if sr != EXPECT_SR:
            problems.append(f"{path.name}: sample rate {sr} != {EXPECT_SR}")
        if channels != 1:
            problems.append(f"{path.name}: {channels} channel != 1")
        if bits != 16:
            problems.append(f"{path.name}: {bits}-bit != 16-bit")
        if not EXPECT_MIN <= seconds <= EXPECT_MAX + 0.05:
            problems.append(f"{path.name}: durasi {seconds:.2f}s di luar {EXPECT_MIN}-{EXPECT_MAX}s")
        if peak >= 0.9999:
            problems.append(f"{path.name}: peak {db(peak):.2f} dBFS (clipping)")
        if peak < 0.03:
            problems.append(f"{path.name}: peak {db(peak):.1f} dBFS terlalu pelan")
        if info["silence"] > 0.6:
            problems.append(f"{path.name}: {info['silence'] * 100:.0f}% senyap")
        if abs(info["dc"]) > 0.01:
            problems.append(f"{path.name}: DC offset {info['dc']:+.3f}")

    width = max((len(k) for k in groups), default=10)
    print(f"{'sumber':<{width}}  {'klip':>4}  {'menit':>6}  {'peak dB':>8}  {'RMS dB':>7}  {'senyap':>7}")
    print("-" * (width + 44))
    for key in sorted(groups):
        items = groups[key]
        secs = sum(i["seconds"] for i in items)
        peak = max(i["peak_db"] for i in items)
        rms = float(np.mean([i["rms_db"] for i in items]))
        sil = float(np.mean([i["silence"] for i in items]))
        print(
            f"{key:<{width}}  {len(items):>4}  {secs / 60:>6.1f}  {peak:>8.1f}  "
            f"{rms:>7.1f}  {sil * 100:>6.0f}%"
        )
    print("-" * (width + 44))

    durations = np.array([i["seconds"] for items in groups.values() for i in items])
    print(
        f"total: {len(files)} klip | {total / 60:.1f} menit | "
        f"durasi min/median/maks: {durations.min():.2f} / "
        f"{float(np.median(durations)):.2f} / {durations.max():.2f} s"
    )
    print(f"ukuran: {sum(p.stat().st_size for p in files) / 1e6:.0f} MB")

    if problems:
        print(f"\n{len(problems)} masalah:")
        for line in problems[:40]:
            print(f"  - {line}")
        if len(problems) > 40:
            print(f"  ... {len(problems) - 40} lagi")
        return 1
    print("\nOK: semua klip sesuai spesifikasi")
    return 0


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels,codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {
        "sr": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "codec": stream.get("codec_name", "?"),
        "duration": float((data.get("format") or {}).get("duration", 0.0) or 0.0),
    }


def ffmpeg_stats(path: Path) -> tuple[float, float, int]:
    """Kembalikan (peak_dbfs, mean_dbfs, jumlah event senyap >1s)."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stderr or ""
    peak = float(re.search(r"max_volume: ([-0-9.]+) dB", text).group(1)) if "max_volume" in text else float("nan")
    mean = float(re.search(r"mean_volume: ([-0-9.]+) dB", text).group(1)) if "mean_volume" in text else float("nan")

    silent = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-40dB:d=1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    events = (silent.stderr or "").count("silence_start:")
    return peak, mean, events


def inspect_sources(folder: Path) -> int:
    files = sorted(p for p in folder.glob("*.wav") if p.is_file())
    if not files:
        print(f"tidak ada file di {folder}")
        return 1
    print(f"{'file':<58} {'menit':>6} {'sr':>6} {'ch':>3} {'peak':>7} {'mean':>7} {'jeda':>5}")
    print("-" * 100)
    for path in files:
        meta = ffprobe(path)
        peak, mean, events = ffmpeg_stats(path)
        name = path.name if len(path.name) <= 56 else path.name[:53] + "..."
        print(
            f"{name:<58} {meta['duration'] / 60:>6.1f} {meta['sr']:>6} "
            f"{meta['channels']:>3} {peak:>7.1f} {mean:>7.1f} {events:>5}"
        )
    print("-" * 100)
    print("jeda = jumlah senyap >1 detik pada -40 dB (banyak jeda = cocok untuk klip)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="dataset/clips", help="folder klip (default: dataset/clips)")
    ap.add_argument("--sources", action="store_true", help="periksa file sumber di --src")
    ap.add_argument("--src", default="model", help="folder sumber untuk --sources")
    args = ap.parse_args()
    folder = ROOT / (args.src if args.sources else args.dir)
    if not folder.is_dir():
        print(f"folder tidak ada: {folder}", file=sys.stderr)
        return 1
    return inspect_sources(folder) if args.sources else inspect_clips(folder)


if __name__ == "__main__":
    raise SystemExit(main())
