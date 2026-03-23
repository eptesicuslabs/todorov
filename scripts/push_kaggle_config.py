from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


KERNEL_REF = "dttdrv/todorov-autoresearch"
NOTEBOOK_DIR = Path("notebooks/autoresearch")
POLL_INTERVAL = 30


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_api():
    os.environ.setdefault("KAGGLE_API_TOKEN", os.environ.get("KAGGLE_API_TOKEN", ""))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def push_kernel(accelerator: str = "NvidiaTeslaT4") -> bool:
    api = get_api()
    print(f"[{timestamp()}] pushing kernel to Kaggle (acc={accelerator})...", flush=True)
    try:
        api.kernels_push(str(NOTEBOOK_DIR), acc=accelerator)
        print(f"[{timestamp()}] kernel pushed: {KERNEL_REF}", flush=True)
        return True
    except Exception as e:
        print(f"[{timestamp()}] push failed: {e}", flush=True)
        return False


def poll_status(timeout_minutes: int = 120) -> str:
    api = get_api()
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        try:
            status_response = api.kernels_status(KERNEL_REF)
            if hasattr(status_response, "_status"):
                status = str(status_response._status).split(".")[-1]
                failure = getattr(status_response, "_failure_message", None)
            elif isinstance(status_response, dict):
                status = status_response.get("status", "unknown")
                failure = status_response.get("failureMessage")
            else:
                status = str(status_response)
                failure = None
        except Exception as e:
            print(f"[{timestamp()}] poll error: {e}", flush=True)
            time.sleep(POLL_INTERVAL)
            continue

        print(f"[{timestamp()}] status: {status}", flush=True)

        if status == "COMPLETE":
            return "complete"
        if status in ("ERROR", "CANCELLED"):
            if failure:
                print(f"[{timestamp()}] failure: {failure}", flush=True)
            return "error"

        time.sleep(POLL_INTERVAL)

    print(f"[{timestamp()}] poll timed out after {timeout_minutes} minutes", flush=True)
    return "timeout"


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "push"

    if action == "push":
        push_kernel()
    elif action == "poll":
        status = poll_status()
        print(f"[{timestamp()}] final status: {status}", flush=True)
    elif action == "run":
        if push_kernel():
            status = poll_status()
            print(f"[{timestamp()}] final status: {status}", flush=True)
    else:
        print(f"usage: {sys.argv[0]} [push|poll|run]", flush=True)
