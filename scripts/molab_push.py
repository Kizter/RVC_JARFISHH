#!/usr/bin/env python
"""Kirim script cloud/*.sh ke sandbox molab dan (opsional) jalankan satu tahap.

execute-code.sh membatasi payload ~32 KB, jadi script dikirim sebagai base64 di
dalam satu probe kecil, bukan sebagai file terpisah.

Contoh:
  python scripts\\molab_push.py --bin <bin-filebin> --start setup
  python scripts\\molab_push.py --start fetch
  python scripts\\molab_push.py --start pipeline --env EXP=myvoice EPOCHS=300

Perintah ini hanya MENULIS probe dan mencetak lokasinya; jalankan probe itu
lewat execute-code.sh (butuh MARIMO_TOKEN dan URL sesi aktif).
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOUD = ROOT / "cloud"
SCRIPTS = (
    ("molab_setup.sh", "/marimo/molab_setup.sh"),
    ("molab_fetch.sh", "/marimo/molab_fetch.sh"),
    ("molab_pipeline.sh", "/marimo/run_pipeline.sh"),
)
LAUNCH = {
    "setup": "bash /marimo/molab_setup.sh",
    "fetch": "bash /marimo/molab_fetch.sh",
    "pipeline": (
        'env EXP="${EXP:-myvoice}" DATA=/marimo/data/clips EPOCHS="${EPOCHS:-300}" '
        'BATCH="${BATCH:-8}" SAVE_EVERY="${SAVE_EVERY:-25}" WORKERS="${WORKERS:-8}" '
        "bash /marimo/run_pipeline.sh"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", choices=["setup", "fetch", "pipeline", "none"], default="none")
    ap.add_argument("--bin", default="", help="id bin filebin berisi rvc-clips-0*.zip")
    ap.add_argument("--env", nargs="*", default=[], help="variabel tambahan, mis. EXP=myvoice")
    ap.add_argument("--out", default="", help="path probe (default: %%TEMP%%/molab_probe.py)")
    args = ap.parse_args()

    missing = [name for name, _ in SCRIPTS if not (CLOUD / name).is_file()]
    if missing:
        print(f"script tidak ada: {', '.join(missing)}", file=sys.stderr)
        return 1

    payload = {
        target: base64.b64encode((CLOUD / name).read_bytes()).decode()
        for name, target in SCRIPTS
    }

    lines = [
        "import base64, os, subprocess",
        "",
        f"FILES = {payload!r}",
        "",
        "for target, b64 in FILES.items():",
        "    os.makedirs(os.path.dirname(target), exist_ok=True)",
        "    with open(target, 'wb') as f:",
        "        f.write(base64.b64decode(b64))",
        "    os.chmod(target, 0o755)",
        "    print('tulis', target, os.path.getsize(target), 'byte')",
        "",
        "os.makedirs('/marimo/logs', exist_ok=True)",
    ]

    if args.start != "none":
        extra = " ".join(args.env)
        launch = f"env BIN={args.bin} {extra} {LAUNCH[args.start]}".replace("  ", " ").strip()
        log = f"/marimo/logs/{args.start}.log"
        cmd = f"nohup {launch} > {log} 2>&1 &"
        lines += [
            "",
            f"subprocess.Popen(['bash', '-lc', {cmd!r}], start_new_session=True)",
            f"print('tahap {args.start} dijalankan -> {log}')",
        ]
    else:
        lines += ["", "print('script terpasang, tidak ada tahap yang dijalankan')"]

    out = Path(args.out) if args.out else Path(os.environ.get("TEMP", "/tmp")) / "molab_probe.py"
    out.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"probe: {out} ({out.stat().st_size} byte)")
    if args.start != "none":
        print(f"tahap: {args.start} -> /marimo/logs/{args.start}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
