# aesthetic logger prototype

status: current (as of 2026-04-16).

date: 2026-04-14
implementation status: implemented, not yet wired into train_model
module: `neuroloc/model/aesthetic_logger.py`
test: `tests/test_aesthetic_logger.py`
related spec: `neuroloc/spec/next_gen.md` phase 6a

## what this is

phase 6a dual-stream logging prototype. reads a god_machine `metrics.jsonl`
artifact and produces four matplotlib png panels to disk. standalone module
plus cli entry point; train_model wiring is a later commit.

## module surface

- `AestheticLogger(jsonl_path, output_dir)`: class with `load_records`,
  `render_loss_bpb`, `render_state_frobenius_heatmap`,
  `render_alpha_eff_timeseries`, `render_layer_snapshot`, `render_all`.
- `render_from_jsonl(jsonl_path, output_dir, step_range=None)`: top-level
  helper that builds the logger and calls `render_all`.
- `python -m neuroloc.model.aesthetic_logger --jsonl PATH --out-dir PATH
  [--step-range LO:HI]`: cli.

## panels produced

- `loss_bpb.png`: two stacked axes, train loss vs step on top and val bpb
  vs step on bottom.
- `state_frobenius_heatmap.png`: heatmap of `state_frobenius_per_layer`,
  layers on y axis, steps on x axis.
- `alpha_eff_timeseries.png`: per-layer `alpha_eff_mean_per_layer` line
  plot, colormap by layer depth.
- `layer_snapshot.png`: bars of `delta_erasure_flag_per_layer`,
  `delta_path_per_layer`, and `beta_mean_per_layer` at the latest step.

filenames are deterministic. missing keys are handled gracefully: any panel
whose source metric is absent from the jsonl is silently skipped.

## validation

ten unit tests pass on the synthetic jsonl harness, including missing
keys, non finite values, empty files, malformed lines, step range
filtering, the cli entry point, and a graceful integration pass over a
checked in simulation artifact at
`neuroloc/output/simulation_suites/correction_field_capacity/`.

```
pytest tests/test_aesthetic_logger.py -v
```

## next step

wire into `train_model` in `neuroloc/model/god_machine.py` so that each
validation tick triggers a render into
`output/<run_name>/aesthetic/step_<N>/`. that commit also updates the
run card template and adds the png artifacts to the prosecutor check
list.

## see also

- `wiki/tests/index.md` — tests/ catalog
- `wiki/PROJECT_PLAN.md` — canonical project state
- `wiki/INDEX.md` — full wiki navigation map
- `wiki/OPERATING_DIRECTIVE.md` — rules governing this article
