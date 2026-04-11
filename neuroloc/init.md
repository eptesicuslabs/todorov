# neuroloc

biological grounding for the todorov hybrid spiking neural architecture.

canonical persistent project state lives in `neuroloc/wiki/PROJECT_PLAN.md`. that file holds the current run, current question, hypothesis under test, decision rules, prior runs index, open research questions, and the project glossary. read it first when picking up the project. updates to PROJECT_PLAN.md are mandatory whenever the project state changes (see the rule in `CLAUDE.md`).

## what this is

neuroloc is a computational neuroscience research project for eptesicus
laboratories. it maps how the brain computes to how todorov computes, so
every architectural decision has a principled biological basis -- or an
honest acknowledgment that the analogy breaks.

todorov development is paused until this research phase produces sufficient
biological grounding. todorov resumes when the bridge documents are complete
and the top interventions are validated.

## structure

```
neuroloc/
  wiki/           obsidian vault (185 articles)
  simulations/    brian2 + pytorch scripts (28)
  raw/            immutable source material
  init.md         this file
```

### wiki/

open `neuroloc/wiki/` as an obsidian vault. start at Home.md.

- mechanisms/ -- 61 articles on biological neural computation
- bridge/ -- 18 documents mapping biology to todorov source code
- comparisons/ -- 13 side-by-side analyses (bio vs artificial)
- concepts/ -- 7 introductory articles (start here if new)
- entities/ -- 33 researcher/lab notes
- knowledge/ -- 39 ML architecture and neuroscience research files
- synthesis/ -- 5 cross-domain integration articles
- tests/ -- 5 test records + 1 index

reading order for newcomers:
1. concepts/start_here.md
2. concepts/the_brain_in_one_page.md
3. concepts/neuroscience_for_ml_engineers.md
4. bridge documents (these connect biology to code)
5. individual mechanism articles as needed

### simulations/

```
pip install brian2 matplotlib numpy
cd simulations/
python single_neuron/lif_fi_curve.py
```

each simulation runs standalone on CPU in under 10 minutes. each outputs
PNG figures. organized by domain: single_neuron, plasticity, sparse_coding,
predictive_coding, cortical_microcircuit, neuromodulation, lateral_inhibition,
oscillations, memory, attention, dendritic, energy, development, spatial,
consciousness.

### raw/

immutable source material. ingested papers, excerpts, reference data.
never modify files here after creation.

## key findings

15 adversarial analyses tested whether todorov's biological analogies hold.
the dominant pattern: most are superficial. the architecture works not
because of biological fidelity but because the biological constraints
(ternary spikes, recurrent state, adaptive thresholds) are independently
useful engineering choices.

the one genuine correspondence: the outer-product associative memory in
KDA (k_t * v_t^T) mirrors Hebbian learning at the mathematical level.

## top 5 interventions (ranked by bridge authors)

1. ATMN leak term (faithful LIF) -- HIGH priority, phase 5a
2. activity-dependent alpha (BCM-like) -- 25-35%, phase 5b+
3. k-WTA ternary spikes -- 20-30%, phase 5+
4. progressive spike activation -- 20-30%, phase 5+
5. neuromodulator network (130 params/layer) -- 15-25%, phase 6+

## current state

- 61 mechanism articles with hooks, inline definitions, ML analogs
- 18 bridge documents with adversarial analysis and implementation specs
- 13 comparison articles
- 7 introductory/concept articles
- 33 entity notes
- 39 knowledge articles
- 5 synthesis articles
- 5 test records
- 28 simulations across 17 domains
- 1124 wikilinks, 0 broken links, 0 duplicate filenames
- 15 articles had missing challenges sections (retrofitted)

## what comes next

tier 2 simulations: small-scale pytorch experiments testing the top 5
interventions at 6M params before committing H200 time. each answers a
binary question: does this modification help at small scale, yes or no.

- ATMN with leak vs without
- k-WTA vs threshold spiking
- progressive spike activation training curves
- BCM-like adaptive alpha state dynamics
- GP vs random bilinear control

## rules

zero comments in code. zero emojis. lowercase in docs. no AI attribution.
sole author: Deyan Todorov.
