#!/bin/bash
# Ambil dataset klip dari filebin ke sandbox molab, lalu unzip.
#
#   BIN=<id filebin> bash /marimo/molab_fetch.sh
#
# Struktur filebin: satu bin berisi rvc-clips-01.zip .. rvc-clips-04.zip.
# Filebin menampilkan halaman peringatan sekali per client, jadi tiap file
# diunduh dua kali: pertama untuk mengambil cookie, kedua untuk byte-nya.
set -uo pipefail

BIN=${BIN:-}
PARTS=${PARTS:-4}
DIR=/marimo/data/zips
OUT=/marimo/data/clips
JAR=/marimo/data/.filebin.jar

[ -n "$BIN" ] || {
    echo "!!! BIN kosong. Jalankan: BIN=<id filebin> bash /marimo/molab_fetch.sh"
    echo "    (atau salin zip lewat file browser ke $DIR lalu: unzip -o $DIR/'*.zip' -d $OUT)"
    exit 2
}

mkdir -p "$DIR" "$OUT" /marimo/logs
echo "=== unduh mulai $(date -u +%FT%TZ) bin=$BIN ==="

for i in $(seq -f "%02g" 1 "$PARTS"); do
    name="rvc-clips-$i.zip"
    url="https://filebin.net/$BIN/$name"
    curl -sS -L -m 300 -A "Mozilla/5.0" -c "$JAR" -b "$JAR" -o /dev/null "$url" || true
    code=$(curl -sS -L -m 1800 -A "Mozilla/5.0" -c "$JAR" -b "$JAR" \
                -o "$DIR/$name" -w "%{http_code}" "$url")
    size=$(stat -c %s "$DIR/$name" 2>/dev/null || echo 0)
    echo "$name http=$code size=$size"
    if [ "$code" != "200" ] || [ "$size" -lt 1000000 ]; then
        echo "!!! gagal unduh $name (bin kedaluwarsa? unggah ulang zips lalu ulangi)"
        exit 20
    fi
    unzip -oq "$DIR/$name" -d "$OUT" || { echo "!!! unzip gagal $name"; exit 21; }
done

count=$(ls -1 "$OUT"/*.wav 2>/dev/null | wc -l)
echo "=== selesai $(date -u +%FT%TZ) ==="
echo "klip: $count"
du -sh "$OUT"
[ "$count" -ge 300 ] || { echo "!!! klip kurang dari 300, cek isi zip"; exit 22; }
echo FETCHDONE
