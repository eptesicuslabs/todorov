from __future__ import annotations

import json
import os
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KAGGLE_API_TOKEN", os.environ.get("KAGGLE_API_TOKEN", ""))

KERNEL_REF = "dttdrv/todorov-autoresearch"
NOTEBOOK_DIR = Path("notebooks/autoresearch")
OUTPUT_DIR = Path("notebooks/autoresearch/output")


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def parse_log(log_path: Path, tail: int = 0) -> tuple[str, str]:
    with open(log_path) as f:
        raw = f.read()
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[-5000:], ""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    for entry in entries:
        stream = entry.get("stream_name", "")
        data = entry.get("data", "")
        for sub in data.split("\n"):
            stripped = sub.rstrip()
            if stripped:
                if stream == "stderr":
                    stderr_lines.append(stripped)
                else:
                    stdout_lines.append(stripped)
    if tail:
        stdout_lines = stdout_lines[-tail:]
        stderr_lines = stderr_lines[-tail:]
    return "\n".join(stdout_lines), "\n".join(stderr_lines)


def run_training(
    code: str | None = None,
    file_path: str | None = None,
    timeout_minutes: int = 30,
    poll_interval: int = 15,
) -> dict:
    if file_path:
        with open(file_path) as f:
            code = f.read()
    if code is None:
        code = (NOTEBOOK_DIR / "train.py").read_text()

    train_path = NOTEBOOK_DIR / "train.py"
    backup = train_path.read_text() if train_path.exists() else None
    train_path.write_text(code)

    api = get_api()

    print(f"[{timestamp()}] pushing {KERNEL_REF}...", flush=True)
    api.kernels_push(str(NOTEBOOK_DIR), acc="NvidiaTeslaT4")
    print(f"[{timestamp()}] pushed. polling (timeout={timeout_minutes}m)...", flush=True)

    deadline = time.time() + timeout_minutes * 60
    start_time = time.time()
    final_status = "timeout"

    while time.time() < deadline:
        time.sleep(poll_interval)
        try:
            r = api.kernels_status(KERNEL_REF)
            if hasattr(r, "_status"):
                status = str(r._status).split(".")[-1]
            elif isinstance(r, dict):
                status = r.get("status", "unknown")
            else:
                status = str(r)
        except Exception as e:
            print(f"[{timestamp()}] poll error: {e}", flush=True)
            continue

        elapsed = time.time() - start_time
        print(f"[{timestamp()}] {status} ({elapsed:.0f}s)", flush=True)

        if status == "COMPLETE":
            final_status = "complete"
            break
        if status in ("ERROR", "CANCELLED"):
            final_status = "error"
            break

    elapsed = time.time() - start_time

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.iterdir():
        f.unlink()

    stdout, stderr = "", ""
    output_files: list[str] = []
    results_data = None

    try:
        api.kernels_output(KERNEL_REF, str(OUTPUT_DIR))
        output_files = [f.name for f in sorted(OUTPUT_DIR.iterdir())]
        print(f"[{timestamp()}] pulled {len(output_files)} files: {output_files}", flush=True)

        log_file = OUTPUT_DIR / "todorov-autoresearch.log"
        if log_file.exists():
            stdout, stderr = parse_log(log_file)

        results_file = OUTPUT_DIR / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results_data = json.load(f)
    except Exception as e:
        stderr += f"\nFailed to pull output: {e}\n"

    return {
        "status": final_status,
        "stdout": stdout,
        "stderr": stderr,
        "execution_time": elapsed,
        "output_files": output_files,
        "results": results_data,
    }


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    result = run_training(file_path=file_path, timeout_minutes=timeout)

    print(f"\n{'='*60}", flush=True)
    print(f"STATUS: {result['status']} ({result['execution_time']:.0f}s)", flush=True)
    print(f"FILES: {result['output_files']}", flush=True)

    if result["results"]:
        print(f"\nRESULTS:", flush=True)
        for k, v in result["results"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}", flush=True)
            else:
                print(f"  {k}: {v}", flush=True)
    elif result["stdout"]:
        print(f"\nSTDOUT (last 30 lines):", flush=True)
        lines = result["stdout"].split("\n")
        for line in lines[-30:]:
            print(f"  {line}", flush=True)

    if result["stderr"]:
        err_lines = [l for l in result["stderr"].split("\n") if l.strip() and "Warning" not in l and "SyntaxWarning" not in l]
        if err_lines:
            print(f"\nERRORS:", flush=True)
            for line in err_lines[-10:]:
                print(f"  {line}", flush=True)
