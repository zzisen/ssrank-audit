#!/usr/bin/env python3
"""Reconstruct the frozen program file from public Enrichr libraries.

Library content is obtained from its provider rather than redistributed with this
software. Check the original resources' terms before use or redistribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    meta = json.loads((ROOT / "data/reference/theory/fixed_programs_2026-11-17/manifest.json").read_text(encoding="utf-8"))
    cache = {}
    for filename, expected in meta["sources_sha256"].items():
        name = filename.removesuffix(".gmt")
        url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?" + urllib.parse.urlencode({"mode": "text", "libraryName": name})
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{filename} content changed; obtain the archived original by matching its SHA-256")
        cache[filename] = data.decode("utf-8-sig").splitlines()
    resolved = {}
    for source in meta["sources"]:
        rows = [line.split("\t") for line in cache[source["file"]] if line.split("\t", 1)[0] == source["exact_entry"]]
        if len(rows) != 1:
            raise ValueError("Expected exactly one source entry: " + source["exact_entry"])
        genes = rows[0][2:]
        while genes and genes[-1] == "":
            genes.pop()
        if not genes or any(not g or g.strip() != g for g in genes) or len(set(genes)) != len(genes):
            raise ValueError("Invalid gene list: " + source["exact_entry"])
        resolved[source["alias"]] = set(genes)
    overlap = resolved["structural_original"] & resolved["conduction_original"]
    resolved["structural_exclusive"] = resolved["structural_original"] - overlap
    resolved["conduction_exclusive"] = resolved["conduction_original"] - overlap
    content = "".join(name + "\tpreexisting_annotation\t" + "\t".join(sorted(genes)) + "\n" for name, genes in resolved.items()).encode("utf-8")
    if hashlib.sha256(content).hexdigest() != meta["output_sha256"]:
        raise ValueError("Reconstructed GMT does not match the frozen checksum")
    dest = args.data_dir / "reference/theory/fixed_programs_2026-11-17/fixed_programs.gmt"
    if dest.exists() and dest.read_bytes() != content:
        raise ValueError(f"{dest} already exists with different content; left untouched")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    print("Verified and reconstructed", dest, meta["output_sha256"])

if __name__ == "__main__":
    main()
