#!/usr/bin/env python
"""Bagi dataset/clips menjadi beberapa zip untuk diunggah ke molab.

Sandbox molab tidak punya endpoint upload HTTP, jadi dataset dikirim lewat
filebin.net: zip diunggah dari PC lokal, lalu diunduh sandbox dengan
`cloud/molab_fetch.sh`.

  .venv\\Scripts\\python.exe scripts\\make_upload_zips.py

Empat bagian (masing-masing ~30 MB) lebih tahan gangguan daripada satu zip
132 MB. Nama file harus `rvc-clips-0N.zip` karena molab_fetch.sh mengulang
01..PARTS.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "dataset" / "clips"
PARTS = 4


def main() -> int:
    files = sorted(CLIPS.glob("*.wav"))
    if not files:
        print(f"tidak ada klip di {CLIPS}")
        return 1

    for old in (ROOT / "dataset").glob("rvc-clips-*.zip"):
        old.unlink()

    per = -(-len(files) // PARTS)
    chunks = [files[i : i + per] for i in range(0, len(files), per)]

    for i, chunk in enumerate(chunks, 1):
        out = ROOT / "dataset" / f"rvc-clips-{i:02d}.zip"
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for path in chunk:
                zf.write(path, arcname=path.name)
        print(f"{out.name}: {len(chunk)} klip, {out.stat().st_size / 1e6:.1f} MB")

    total = sum(p.stat().st_size for p in (ROOT / "dataset").glob("rvc-clips-*.zip"))
    print(f"total {len(chunks)} zip, {total / 1e6:.1f} MB, {len(files)} klip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
