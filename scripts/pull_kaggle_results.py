from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


KERNEL_REF = "dttdrv/todorov-autoresearch"
OUTPUT_DIR = Path("notebooks/autoresearch/output")


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_api():
    os.environ.setdefault("KAGGLE_API_TOKEN", os.environ.get("KAGGLE_API_TOKEN", ""))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def parse_kaggle_log(log_path: Path, tail_lines: int = 50) -> str:
    with open(log_path) as f:
        raw = f.read()
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[-3000:]

    lines: list[str] = []
    for entry in entries:
        stream = entry.get("stream_name", "")
        data = entry.get("data", "")
        prefix = "[ERR] " if stream == "stderr" else ""
        for sub in data.split("\n"):
            stripped = sub.rstrip()
            if stripped:
                lines.append(prefix + stripped)

    if tail_lines and len(lines) > tail_lines:
        lines = lines[-tail_lines:]
    return "\n".join(lines)


def pull_output() -> dict | None:
    api = get_api()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{timestamp()}] pulling output from {KERNEL_REF}...", flush=True)

    try:
        api.kernels_output(KERNEL_REF, str(OUTPUT_DIR))
    except Exception as e:
        print(f"[{timestamp()}] pull error: {e}", flush=True)
        return None

    pulled_files = list(OUTPUT_DIR.iterdir())
    print(f"[{timestamp()}] pulled {len(pulled_files)} files:", flush=True)
    for f in sorted(pulled_files):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)", flush=True)

    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n[{timestamp()}] === RESULTS ===", flush=True)
        for key, val in results.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}", flush=True)
            else:
                print(f"  {key}: {val}", flush=True)
        return results

    print(f"\n[{timestamp()}] no results.json -- kernel likely errored.", flush=True)

    log_path = OUTPUT_DIR / "todorov-autoresearch.log"
    if log_path.exists():
        print(f"[{timestamp()}] === KERNEL LOG (last 50 lines) ===", flush=True)
        parsed = parse_kaggle_log(log_path, tail_lines=50)
        print(parsed, flush=True)
        print(f"[{timestamp()}] === END LOG ===", flush=True)

    return None


if __name__ == "__main__":
    tail = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    result = pull_output()
    if result is None:
        sys.exit(1)
