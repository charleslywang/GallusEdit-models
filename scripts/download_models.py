#!/usr/bin/env python3
"""Download GallusEdit model artifacts from GitHub Release assets.

Usage:
    export GH_TOKEN=<token with repo read access>
    python scripts/download_models.py --tag v1.0.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "charleslywang/GallusEdit-models"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "model_manifest.tsv"


def api_request(url: str, token: str, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "gallusedit-downloader",
        },
    )
    return urllib.request.urlopen(req, timeout=120)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST}")
    rows = []
    lines = MANIFEST.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        rows.append(dict(zip(header, line.split("\t"))))
    return rows


def list_assets(tag: str, token: str) -> dict[str, str]:
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    with api_request(url, token) as response:
        release = json.load(response)
    return {asset["name"]: asset["url"] for asset in release.get("assets", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        sys.exit("GH_TOKEN or GITHUB_TOKEN is required (private repository).")

    assets = list_assets(args.tag, token)
    rows = read_manifest()
    failed = []
    for row in rows:
        target = ROOT / row["model_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        name = target.name
        if target.exists() and not args.force:
            if sha256(target) == row["sha256"]:
                print(f"[skip] {row['display_name']}: already present and verified")
                continue
            print(f"[redo] {row['display_name']}: checksum mismatch, re-downloading")
        if name not in assets:
            failed.append(f"{name}: not found in release {args.tag}")
            continue
        print(f"[get ] {row['display_name']} -> {target}")
        with api_request(assets[name], token, accept="application/octet-stream") as response:
            tmp = target.with_suffix(target.suffix + ".part")
            with tmp.open("wb") as handle:
                for chunk in iter(lambda: response.read(8 << 20), b""):
                    handle.write(chunk)
            tmp.replace(target)
        if sha256(target) != row["sha256"]:
            failed.append(f"{name}: SHA-256 mismatch after download")
        else:
            print(f"[ ok ] {row['display_name']}: verified")

    if failed:
        print("\n".join(["FAILED:"] + failed), file=sys.stderr)
        sys.exit(1)
    print(f"All {len(rows)} model artifacts are present and verified.")


if __name__ == "__main__":
    main()
