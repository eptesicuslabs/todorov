# neuroloc

this directory is the research memory and simulation substrate for the todorov
project.

the active project lane is not a neuroloc experiment run. it is the teaching
pdf curriculum in `pdf_curriculum/`. neuroloc remains active as reference
material, simulation infrastructure, and backlog state.

canonical persistent project state lives in `neuroloc/wiki/PROJECT_PLAN.md`.

## current status

- curriculum is the only active lane
- chapter 1 is review-ready in `pdf_curriculum/chapters/ch_01_what_a_number_means/`
- chapter 2 is outline-only in `pdf_curriculum/chapters/ch_02_how_things_change/`
- no paid compute is authorised during this curriculum cycle
- the architecture backlog resumes later from the implemented `biology_phase1` battery plus the remaining latent-world deliberation and model-side evaluation gaps

## structure

```text
neuroloc/
  wiki/           canonical wiki and project state
  simulations/    cpu-first biology and neural-model simulations
  spec/           neural-machine design and backlog plans
  results/        historical experiment summaries
  print/          print-oriented neuroloc documents
  raw/            immutable source material
  HANDOFF.md      operator handoff for neuroloc work
  init.md         this file
```

## current counts

- `216` wiki markdown files
- `13` synthesis articles
- `45` simulation scripts
- `17` memory simulation scripts

## entry points

1. `neuroloc/wiki/PROJECT_PLAN.md`
2. `neuroloc/wiki/OPERATING_DIRECTIVE.md`
3. `neuroloc/HANDOFF.md`
4. `neuroloc/wiki/Home.md`
5. `pdf_curriculum/index/curriculum_status.md`

## note on history

older neuroloc documents may describe the pre-pivot architecture phase in
detail. keep them as historical record unless they are an overview or handoff
surface that must match the current state.
