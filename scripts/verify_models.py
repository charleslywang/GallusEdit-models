#!/usr/bin/env python3
"""Verify GallusEdit model artifacts against models/model_manifest.tsv.

Checks file presence, byte size, SHA-256 and metadata n_train / n_features.
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "model_manifest.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    lines = MANIFEST.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    failures: list[str] = []

    for row in rows:
        label = f"{row['display_name']} ({row['short_name']})"
        model_path = ROOT / row["model_file"]
        meta_path = model_path.with_name(
            model_path.name.replace(".joblib", ".metadata.json")
        )
        row_failures: list[str] = []
        if not model_path.exists():
            row_failures.append(f"{label}: missing {model_path.relative_to(ROOT)}")
        else:
            if model_path.stat().st_size != int(row["size_bytes"]):
                row_failures.append(f"{label}: size mismatch")
            if sha256(model_path) != row["sha256"]:
                row_failures.append(f"{label}: SHA-256 mismatch")
            if not meta_path.exists():
                row_failures.append(f"{label}: missing {meta_path.name}")
            else:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                for key in ("n_train", "n_features"):
                    if int(metadata[key]) != int(row[key]):
                        row_failures.append(
                            f"{label}: metadata {key}={metadata[key]} != manifest {row[key]}"
                        )
        if row_failures:
            failures.extend(row_failures)
        else:
            print(f"[ ok ] {label}")

    if failures:
        print("\n".join(["FAILED:"] + failures))
        return 1
    print(f"All {len(rows)} models verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
