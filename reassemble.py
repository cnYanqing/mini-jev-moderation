"""Download, verify, and reassemble the Mini-Jev image v6 release archive."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = "lettery/mini-jev-moderation-image-v6"
HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch parts from Hugging Face")
    parser.add_argument("--output", type=Path, help="Output ZIP path")
    args = parser.parse_args()
    manifest = json.loads((HERE / "PARTS.json").read_text(encoding="utf-8"))
    source = HERE
    if args.download:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(REPO, allow_patterns=["*.part-*", "PARTS.json"]))
    target = args.output or (Path.cwd() / manifest["archive_name"])
    target = target.resolve()
    if target.exists() and sha256(target) == manifest["archive_sha256"]:
        print(f"Already verified: {target}")
        return
    with target.open("wb") as out:
        for item in manifest["parts"]:
            part = source / item["name"]
            if part.stat().st_size != item["size"] or sha256(part) != item["sha256"]:
                raise ValueError(f"Part failed integrity check: {part}")
            with part.open("rb") as file:
                shutil.copyfileobj(file, out, 8 * 1024 * 1024)
    if sha256(target) != manifest["archive_sha256"]:
        target.unlink()
        raise ValueError("Reassembled ZIP failed integrity check")
    with zipfile.ZipFile(target) as archive:
        print(f"Verified {len(archive.namelist())} files: {target}")


if __name__ == "__main__":
    main()
