"""Download a batch of Prelinger / educational films from archive.org.

Hits the archive.org metadata API to find h.264 mp4 files of reasonable size,
then downloads. Skips duplicates. Designed to scale the LA-JEPA training
corpus from 13 -> ~30 films without manual curation.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


SEARCH_URL = (
    "https://archive.org/advancedsearch.php"
    "?q=collection%3Aprelinger+mediatype%3Amovies+format%3A%22h.264%22"
    "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=runtime"
    "&rows={rows}&page={page}"
    "&sort%5B%5D=downloads+desc"
    "&output=json"
)


def fetch_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def pick_file(ident, size_max=250_000_000, size_min=5_000_000):
    """Return filename of an h.264 mp4 inside ident."""
    meta = fetch_json(f"https://archive.org/metadata/{ident}/files")
    files = meta.get("result", [])
    cands = []
    for f in files:
        fmt = (f.get("format") or "").lower()
        name = f.get("name") or ""
        if not name.lower().endswith((".mp4", ".mov")):
            continue
        sz = int(f.get("size") or 0)
        if sz < size_min or sz > size_max:
            continue
        if "h.264" in fmt or "mpeg4" in fmt:
            cands.append((sz, name))
    cands.sort()  # smallest first
    return cands[0][1] if cands else None


def download(url, dest):
    tmp = dest + ".tmp"
    print(f"  -> downloading {os.path.basename(dest)}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.rename(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/jepa_cache")
    ap.add_argument("--target_count", type=int, default=20)
    ap.add_argument("--rows", type=int, default=80)
    ap.add_argument("--max_pages", type=int, default=3)
    ap.add_argument("--size_max", type=int, default=250_000_000)
    ap.add_argument("--size_min", type=int, default=5_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    existing = {f.lower() for f in os.listdir(args.out_dir)}

    successes = 0
    seen_ids = set()
    for page in range(1, args.max_pages + 1):
        url = SEARCH_URL.format(rows=args.rows, page=page)
        try:
            j = fetch_json(url)
        except Exception as e:
            print(f"search page {page} failed: {e}")
            continue
        docs = j.get("response", {}).get("docs", [])
        print(f"page {page}: {len(docs)} candidates")
        for d in docs:
            if successes >= args.target_count:
                return
            ident = d.get("identifier")
            if not ident or ident in seen_ids:
                continue
            seen_ids.add(ident)
            try:
                fname = pick_file(ident, args.size_max, args.size_min)
            except Exception as e:
                print(f"  metadata fail {ident}: {e}")
                continue
            if not fname:
                continue
            local = os.path.join(args.out_dir, f"{ident}.mp4")
            if os.path.basename(local).lower() in existing or os.path.exists(local):
                continue
            url = f"https://archive.org/download/{ident}/{urllib.request.quote(fname)}"
            try:
                download(url, local)
                successes += 1
                existing.add(os.path.basename(local).lower())
                print(f"    [{successes}/{args.target_count}] {ident}.mp4")
            except Exception as e:
                print(f"  download fail {ident}: {e}")
            time.sleep(0.5)
    print(f"done: {successes} new files")


if __name__ == "__main__":
    main()
