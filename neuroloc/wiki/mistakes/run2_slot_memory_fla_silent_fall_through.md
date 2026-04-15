# mistake: paid run launched with FLA silently disabled, projected 56h instead of 75min

## what happened

2026-04-15. the second paid run2_slot_memory launch (after the retention-bug
fix) was authorized and started on the runpod h200. user directive was
explicitly "FLA on". the preset had been edited so `use_fla_if_available=True`,
the prosecutor had verified the wiring through `SlotMemory.__init__`, and the
launch passed all gates.

after 17 minutes, only step 0 had been logged. the metrics jsonl confirmed
training was progressing but at 655 tokens/sec — exactly the speed of the
pure-python `_recurrent_slot_update` loop, NOT the fused triton kernel. at
that rate the 4000-step run would take 56 hours of paid h200 time.

## root cause

the `fla` package (`flash-linear-attention`) was not installed on the pod.
`from fla.ops import fused_recurrent_simple_gla` failed at module import,
the import-time guard set `FLA_AVAILABLE = False` and `fused_recurrent_simple_gla = None`,
and `SlotMemory.use_fla = bool(cfg.use_fla_if_available) and FLA_AVAILABLE`
evaluated to `False`. the forward pass fell through to the python recurrent
loop with no warning at any layer of the stack: not in the import, not in the
config, not in the model construction, not in the run header.

an isolated benchmark on the same pod after `pip install flash-linear-attention`
showed: SlotMemory forward at d_model=1024, batch=16, seq=2048 takes 0.012s
with FLA, ~30s without. ratio: 2500x.

## why the prosecutor missed it

the prosecutor reviewed the source tree and the launch surface. it could not
detect a missing pip package on the runtime pod because the pod state is
external to the repository. the project's `requirements.txt` listed only
`torch, numpy, scipy, pyyaml, matplotlib, jinja2, pytest, brian2`. neither
`flash-linear-attention` nor `datasets` were pinned, so any fresh pod was
guaranteed to silently degrade SlotMemory performance and crash on fineweb-edu
streaming.

## the fix

commit `edcfe5d` adds `flash-linear-attention>=0.4.0` and `datasets>=2.19.0`
to `requirements.txt`. a fresh pod that runs `pip install -r requirements.txt`
now gets both, and the FLA fall-through path becomes impossible to reach.

structural improvement to consider in a future commit: at god_machine.py
startup, log a clear warning when any layer's `use_fla` is False after a
preset asks for it. silent fall-through to a 2500x slower path is a class of
bug the project should detect at launch time, not by reverse-engineering
metrics jsonl after burning paid compute.

## cost of this mistake

20 minutes of paid h200 wall clock at the slow path before the issue was
caught and the run killed. approximately $1.30 in direct compute. the
opportunity cost was higher: this was the second-time-this-week paid run that
got launched in a degraded state and had to be killed, and each cycle
consumes user attention and trust.

## rule (self-imposed)

before any paid launch:
1. verify on the pod that every architecture-critical dependency is importable.
   one `python -c "from fla.ops import fused_recurrent_simple_gla; print('ok')"`
   per critical kernel.
2. verify the model's runtime use_fla flag matches the preset intent. one
   `python -c "from neuroloc.model.god_machine import _resolve_preset, Config, GodMachine; ..."`
   line printing each block's `mixer.use_fla` after construction.
3. only then issue the training launch.

these three checks add ~30 seconds of pod time and would have prevented the
entire 20-minute fall-through episode.
