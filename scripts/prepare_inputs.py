#!/usr/bin/env python3
"""Verify local benchmark inputs or explicitly download missing inputs.

Raw files may be subject to source terms. No transfers occur unless --download
is supplied. Checksums identify the exact local copies used in this project.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]

def fetch(entry, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "ssrank-audit-reproduction/1.0"})
    # The archive member is explicitly named; never extract archive paths wholesale.
    with tempfile.TemporaryDirectory(dir=dest.parent) as td:
        download = Path(td) / "download"
        with urllib.request.urlopen(req, timeout=120) as response, download.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if "archive_member" in entry:
            member = Path(td) / "member"
            with zipfile.ZipFile(download) as archive, archive.open(entry["archive_member"]) as src, member.open("wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            download = member
        if not valid(download, entry):
            raise ValueError("download does not match expected byte count/SHA-256")
        os.replace(download, dest)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("CARDIAC_DATA_DIR", REPO / "data")))
    parser.add_argument("--download", action="store_true", help="download missing files; never replace an existing mismatched file")
    args = parser.parse_args()
    manifest = json.loads((REPO / "data/input_files.json").read_text(encoding="utf-8"))
    errors = 0
    for entry in manifest["files"]:
        dest = args.data_dir / entry["path"]
        if valid(dest, entry):
            print("OK", entry["path"])
            continue
        if dest.exists():
            print("MISMATCH (left untouched)", entry["path"], file=sys.stderr)
            errors += 1
        elif not args.download:
            print("MISSING", entry["path"], file=sys.stderr)
            errors += 1
        else:
            try:
                print("DOWNLOAD", entry["path"], flush=True)
                fetch(entry, dest)
                print("OK", entry["path"])
            except Exception as exc:
                print("DOWNLOAD FAILED", entry["path"], str(exc), file=sys.stderr)
                errors += 1
    print(f"Checked {len(manifest['files'])} inputs; {errors} missing, mismatched or failed.")
    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
