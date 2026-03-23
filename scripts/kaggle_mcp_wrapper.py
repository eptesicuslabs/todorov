import json
import os
import sys
from pathlib import Path

mcp_config = Path(__file__).resolve().parent.parent / ".mcp.json"
if mcp_config.exists():
    with open(mcp_config) as f:
        cfg = json.load(f)
    token = cfg.get("mcpServers", {}).get("kaggle-exec", {}).get("env", {}).get("KAGGLE_API_TOKEN", "")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token

if not os.environ.get("KAGGLE_API_TOKEN"):
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            kj = json.load(f)
        token = kj.get("api_token", "")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token

from mcp_server_kaggle_exec.server import main

main()
