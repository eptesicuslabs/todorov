from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import yaml


REPORTS_DIR = Path("reports")
STATE_DIR = Path("state")
DOCS_DIR = Path("docs")


def register_run(bundle_path: str, run_id: str | None = None) -> None:
    bundle = Path(bundle_path)
    if not bundle.exists():
        print(f"[{datetime.now().isoformat()}] ERROR: Bundle not found: {bundle}")
        sys.exit(1)

    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if bundle.suffix == ".zip":
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(run_dir)
        print(f"[{datetime.now().isoformat()}] Extracted bundle to {run_dir}")
    else:
        shutil.copy2(bundle, run_dir)
        print(f"[{datetime.now().isoformat()}] Copied bundle to {run_dir}")

    with open(bundle, "rb") as f:
        bundle_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    config_path = run_dir / "config.json"
    config_data = {}
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)

    index_path = REPORTS_DIR / "index.md"
    entry = f"| {run_id} | {datetime.now().strftime('%Y-%m-%d')} | {bundle_hash} | {run_dir} |\n"
    with open(index_path, "a") as f:
        f.write(entry)

    print(f"[{datetime.now().isoformat()}] Registered run {run_id} (hash: {bundle_hash})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    register_run(args.bundle, args.run_id)


if __name__ == "__main__":
    main()
