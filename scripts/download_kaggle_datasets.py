from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "datasets" / "kaggle_datasets.json"


def load_manifest() -> list[dict[str, object]]:
    with MANIFEST_PATH.open() as handle:
        return json.load(handle)


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "The Kaggle CLI is not installed or not on PATH. Install it with `pip install kaggle`."
        )


def download_dataset(entry: dict[str, object], force: bool = False) -> None:
    target_dir = ROOT / str(entry["local_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        str(entry["slug"]),
        "-p",
        str(target_dir),
        "--unzip",
    ]
    if force:
        command.append("--force")

    print(f"Downloading {entry['name']} from {entry['slug']} -> {target_dir}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle datasets used by this repository.")
    parser.add_argument("--only", action="append", help="Download only one named dataset from the manifest.")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing archives.")
    args = parser.parse_args()

    ensure_kaggle_cli()
    manifest = load_manifest()

    selected = manifest
    if args.only:
        requested = {name.lower() for name in args.only}
        selected = [entry for entry in manifest if str(entry["name"]).lower() in requested]
        if not selected:
            raise SystemExit(f"No manifest entries matched: {', '.join(args.only)}")

    for entry in selected:
        download_dataset(entry, force=args.force)


if __name__ == "__main__":
    main()
