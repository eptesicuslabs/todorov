import json, sys
cap = json.load(open('/root/neuroloc/output/simulation_runs/misc/root/slot_buffer_capacity/slot_buffer_capacity_metrics.json'))
sw = json.load(open('/root/neuroloc/output/simulation_runs/misc/root/slot_surprise_writes/slot_surprise_writes_metrics.json'))
cs = cap['summary']
ss = sw['summary']
print("=== SLOT BUFFER CAPACITY ===")
print("pass_criterion_met:", cs['pass_criterion_met'])
print("slot_cosine:", cs['pass_criterion_slot_cosine_mean'])
print("slot_exact:", cs['pass_criterion_slot_exact_mean'])
print("best_temperature:", cs['pass_criterion_best_temperature'])
print()
print("=== SLOT SURPRISE WRITES ===")
print("pass_criterion_met:", ss['pass_criterion_met'])
print("best_cosine:", ss['pass_criterion_best_cosine'])
print("reference_cell:", ss['pass_reference_cell'])
