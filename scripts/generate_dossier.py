from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


REPORTS_DIR = Path("reports")
STATE_DIR = Path("state")
TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>Todorov Run Dossier - {run_id}</title>
<style>
body {{ font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; }}
h1, h2 {{ border-bottom: 1px solid #333; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #666; padding: 8px; text-align: left; }}
th {{ background: #222; color: #fff; }}
.pass {{ color: green; font-weight: bold; }}
.fail {{ color: red; font-weight: bold; }}
.pending {{ color: orange; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>Todorov Run Dossier</h1>
<p>Run ID: {run_id}</p>
<p>Generated: {timestamp}</p>

<h2>Configuration</h2>
<pre>{config_json}</pre>

<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>final_val_bpb</td><td>{final_val_bpb}</td></tr>
<tr><td>best_val_bpb</td><td>{best_val_bpb}</td></tr>
<tr><td>total_steps</td><td>{total_steps}</td></tr>
<tr><td>kda_state_norm</td><td>{kda_state_norm}</td></tr>
<tr><td>mamba3_utilization</td><td>{mamba3_utilization}</td></tr>
<tr><td>mla_compression_ratio</td><td>{mla_compression_ratio}</td></tr>
<tr><td>spike_firing_rate</td><td>{spike_firing_rate}</td></tr>
<tr><td>spike_dead_ratio</td><td>{spike_dead_ratio}</td></tr>
<tr><td>param_count</td><td>{param_count}</td></tr>
</table>

<h2>Gate Results</h2>
{gate_table}

<h2>Artifacts</h2>
{artifact_list}
</body>
</html>
"""


def generate_dossier(run_id: str) -> None:
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        print(f"[{datetime.now().isoformat()}] ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    config_path = run_dir / "config.json"
    config_json = "{}"
    if config_path.exists():
        with open(config_path) as f:
            config_json = json.dumps(json.load(f), indent=2)

    results_path = run_dir / "results.json"
    metrics = {}
    if results_path.exists():
        with open(results_path) as f:
            metrics = json.load(f)

    gate_results = {}
    gate_file = STATE_DIR / "gate_results.yaml"
    if gate_file.exists():
        with open(gate_file) as f:
            gate_results = yaml.safe_load(f) or {}

    gate_rows = []
    for phase_key, gates in gate_results.items():
        for gate_name, gate_data in gates.items():
            status = gate_data.get("status", "PENDING")
            css_class = status.lower()
            gate_rows.append(
                f'<tr><td>{phase_key}</td><td>{gate_name}</td>'
                f'<td class="{css_class}">{status}</td></tr>'
            )

    gate_table = "<table><tr><th>Phase</th><th>Gate</th><th>Status</th></tr>"
    gate_table += "\n".join(gate_rows) if gate_rows else "<tr><td colspan='3'>No results</td></tr>"
    gate_table += "</table>"

    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(run_dir)
            if path.suffix in (".png", ".jpg"):
                artifacts.append(f'<p>{rel_path}</p><img src="{rel_path}">')
            else:
                artifacts.append(f"<p>{rel_path} ({path.stat().st_size} bytes)</p>")

    artifact_list = "\n".join(artifacts) if artifacts else "<p>No artifacts found</p>"

    html = TEMPLATE.format(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        config_json=config_json,
        gate_table=gate_table,
        artifact_list=artifact_list,
        final_val_bpb=metrics.get("final_val_bpb", "N/A"),
        best_val_bpb=metrics.get("best_val_bpb", "N/A"),
        total_steps=metrics.get("total_steps", "N/A"),
        kda_state_norm=metrics.get("kda_state_norm", "N/A"),
        mamba3_utilization=metrics.get("mamba3_utilization", "N/A"),
        mla_compression_ratio=metrics.get("mla_compression_ratio", "N/A"),
        spike_firing_rate=metrics.get("spike_firing_rate", "N/A"),
        spike_dead_ratio=metrics.get("spike_dead_ratio", "N/A"),
        param_count=metrics.get("param_count", "N/A"),
    )

    output_path = run_dir / "dossier.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"[{datetime.now().isoformat()}] Dossier written to {output_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()

    generate_dossier(args.run_id)


if __name__ == "__main__":
    main()
