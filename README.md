# todorov

todorov is a research repository around one mathematical object:
the compressed rotational bilinear recurrence (crbr).

every layer instantiates
`z_t = Q(R(B(C(x_t), C(h_{t-1}))))`,
with compression, bilinear interaction, rotational structure,
and output shaping treated as one composable family.

the repository now has two distinct layers of state:

- the archived architecture track, which established the main todorov results and later hit a retrieval wall in the neural-machine branch
- the active curriculum track, which is building a 36-chapter teaching pdf in english latex so the architecture backlog can be resumed from a stronger theoretical base

## current status

the active workstream is the teaching pdf curriculum in `pdf_curriculum/`.
paid compute is paused indefinitely while the curriculum is in progress.
the architectural-intervention track remains in the research backlog and does
not resume until the curriculum is complete and a cpu-validated intervention
has been chosen.

current curriculum state:

- chapter 1 package is review-ready at `pdf_curriculum/chapters/ch_01_what_a_number_means/`
- chapter 2 is outline-only at `pdf_curriculum/chapters/ch_02_how_things_change/ch_02_outline.md`
- the curriculum status index is `pdf_curriculum/index/curriculum_status.md`

canonical persistent project state lives in
`neuroloc/wiki/PROJECT_PLAN.md`.

## archived architecture results

the strongest archived todorov result remains the 267m phase-5 baseline:

- bpb ratio vs matched transformer: `0.663x`
- spike mutual information: `1.168`
- spike cka similarity: `0.732`
- spike firing rate: `40.8%`

the later neural-machine branch achieved strong byte-level fit but failed
retrieval across six paid runs. that branch is paused pending architectural
changes. the canonical write-up is in
`neuroloc/wiki/synthesis/substrate_requires_architectural_change.md`.

## neuroloc

neuroloc (`neuroloc/`) is now both:

- the repo-native research wiki and simulation corpus that records the architecture backlog
- a source base for the curriculum and for later llm-assisted wiki maintenance

current neuroloc surface:

- `216` wiki markdown files
- `13` synthesis articles
- `45` simulation scripts
- `17` memory simulation scripts

the main entry points are:

- `neuroloc/wiki/Home.md`
- `neuroloc/wiki/INDEX.md`
- `neuroloc/wiki/PROJECT_PLAN.md`
- `neuroloc/HANDOFF.md`

## repository map

- `pdf_curriculum/` active curriculum production
- `neuroloc/` wiki, simulations, specs, and historical architecture backlog
- `src/` todorov library code
- `tests/` repo test suite
- `docs/` historical run summaries and status documents
- `state/` machine-readable project state

## funding

the architecture track is compute-limited, but compute spending is currently
paused by design. the immediate project need is not another paid run. it is
curriculum completion and theory consolidation.

eptesicus laboratories.
