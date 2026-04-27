# neuroloc

neuroloc is the repository-native neuroscience wiki, simulation corpus,
and architecture-backlog memory for todorov.

## current role

neuroloc currently serves four jobs:

- it preserves the canonical architecture history and the retrieval-failure diagnosis
- it acts as the source corpus for curriculum research and drafting
- it holds the cpu simulation surface for future backlog validation
- it provides the wiki structure for llm-maintained project memory

the canonical persistent project state is
`neuroloc/wiki/PROJECT_PLAN.md`.

## current state

the last paid neural-machine branch ended with six consecutive 0% passkey
results across two substrates and two corpora. that branch is paused.
the current backlog restart point is:

- the implemented `biology_phase1` simulation battery
- the remaining latent-world deliberation probe
- the model-side neural-model evaluation surface

no paid compute is authorised while the curriculum is active.

## current neuroloc surface

- `216` wiki markdown files
- `61` mechanism articles
- `19` bridge documents
- `13` comparison articles
- `13` synthesis articles
- `22` wiki test records
- `45` simulation scripts
- `17` memory simulation scripts

main entry points:

- `neuroloc/wiki/Home.md`
- `neuroloc/wiki/INDEX.md`
- `neuroloc/wiki/PROJECT_PLAN.md`
- `neuroloc/wiki/OPERATING_DIRECTIVE.md`
- `neuroloc/HANDOFF.md`

## what todorov is

todorov is built around the compressed rotational bilinear recurrence:

`z_t = Q(R(B(C(x_t), C(h_{t-1}))))`

where:

- `c` is compression
- `b` is bilinear interaction
- `r` is rotational structure
- `q` is output shaping or quantization

the archived architecture results established that these constraints can beat
matched transformers at scale. the later neural-machine work showed that good
distribution fit did not imply trained retrieval. neuroloc exists to explain
that gap well enough to fix it later.

## directory guide

- `neuroloc/wiki/` canonical research wiki and state articles
- `neuroloc/simulations/` cpu-first biological and neural-model simulations
- `neuroloc/spec/` neural-machine design documents and backlog plans
- `neuroloc/results/` historical experiment summaries
- `neuroloc/print/` print-oriented neuroloc documents
- `neuroloc/raw/` immutable source material

## what to read first

1. `neuroloc/wiki/PROJECT_PLAN.md`
2. `neuroloc/wiki/OPERATING_DIRECTIVE.md`
3. `neuroloc/HANDOFF.md`
4. `pdf_curriculum/index/curriculum_status.md`

## constraints

- zero comments in code
- zero emojis
- zero ai attribution
- lowercase in docs and commit text
- use the wiki operating directive for any wiki or state change
