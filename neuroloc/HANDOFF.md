# Neuroloc Project Handoff

You are continuing work on the Neuroloc project inside the Todorov repository
at `C:\Users\deyan\Projects\todorov\neuroloc\`. Read this entire document
before taking any action.

## What Neuroloc Is

Neuroloc is a computational neuroscience research project for Eptesicus
Laboratories. It maps biological neural computation mechanisms to Todorov's
CRBR (Compressed Rotational Bilinear Recurrence) architecture. Todorov
development is paused while this research phase provides biological grounding
for every architectural decision.

Todorov achieves 0.663x BPB at 267M parameters (33.7% better than transformer
baseline). The CRBR equation: z_t = Q(R(B(C(x_t), C(h_{t-1})))). Three layer
types: KDA (delta-rule recurrence), Mamba3 (state-space model), MLA (compressed
softmax attention). Layer schedule: (KDA, KDA, KDA, Mamba3, KDA, KDA, KDA, MLA)
x 3. See `NEUROLOC.md` at the repo root for the full component registry.

## Project Rules (Non-Negotiable)

Read `CLAUDE.md` at the repo root. The critical rules:
- Zero comments in code. No inline, no block, no docstrings, no TODOs.
- Zero emojis anywhere.
- Zero AI attribution. No Co-authored-by. Sole author: Deyan Todorov.
- Lowercase in all docs, commits, and readme.
- The Adversarial Thinking Rule applies to every claim in the wiki.

## Directory Structure

```
neuroloc/
  wiki/              Obsidian vault -- 141 markdown articles
    mechanisms/      61 articles on biological neural computation
    bridge/          18 documents mapping biology to Todorov source code
    comparisons/     13 side-by-side analyses (biological vs artificial)
    concepts/        7 introductory and reference articles
    entities/        33 researcher and lab notes
    knowledge/       13 ML architecture research files (copied from Todorov)
    synthesis/       5 cross-domain integration articles
    Home.md          Vault home page with navigation links
    index.md         Flat reference catalog of all articles
    log.md           Chronological operation log (append-only)
  simulations/       18 Brian2 + Python scripts organized by domain
  print/             5 print-ready markdown files (cleaned for PDF)
  raw/               Immutable source material (never modify)
  init.md            Project overview and current state
  neuroloc_guide.tex LaTeX source for the printed guide
  neuroloc_guide.pdf 32-page compiled guide
  build_latex.py     Script to regenerate PDF from print/ sources
  build_pdf.py       Legacy reportlab builder (superseded by LaTeX)
  HANDOFF.md         This file
```

## What Exists

### Wiki (neuroloc/wiki/)

Open as Obsidian vault at `neuroloc/wiki/`. No nested `.obsidian/` conflicts.
All wikilinks use shortest-path resolution. Zero broken links. Zero duplicate
basenames. 1124 wikilinks total, 9.3 cross-reference density per article.

**61 mechanism articles** across 17 domains: single neuron models (4),
dendritic computation (4), synaptic plasticity (6), development and learning
(3), energy and metabolism (3), neural coding (4), predictive processing (3),
lateral inhibition (4), neuromodulation (5), cortical microcircuits (4), memory
systems (4), spatial computation (4), attention (3), oscillatory dynamics (3),
consciousness and integration (4), inhibitory signaling (2), action selection (1).
added 2026-04-07: serotonin_system, gaba_signaling, nmda_receptors,
basal_ganglia, three_factor_learning.

Every mechanism article was retrofitted with:
- A "why this matters" hook connecting to ML/architecture design
- Inline definitions for first-use technical terms
- ML analog notes mapping biological concepts to transformers/attention
- A mandatory challenges and counter-arguments section (at least 3 criticisms)

**18 bridge documents** mapping biological mechanisms to specific Todorov
components with adversarial analysis. Each contains: the biological mechanism,
the current Todorov implementation (with source file paths), the proposed
change, implementation spec, expected impact, risk assessment.
added 2026-04-07: positional_encoding_to_rope, normalization_to_rmsnorm.

**5 synthesis articles** (added 2026-04-07): sparsity_from_biology_to_ternary_spikes,
timescale_separation, local_vs_global_computation, compression_and_bottlenecks,
recurrence_vs_feedforward. cross-domain integration articles weaving multiple
mechanism and bridge findings into unified narratives.

**13 comparison articles** with structured side-by-side analyses and verdicts.

**7 concept articles**: start_here, the_brain_in_one_page,
neuroscience_for_ml_engineers (7-part primer), mathematical_foundations
(with worked examples), todorov_biology_map (master component mapping),
glossary (55 terms with ML analogs), notation.

**33 entity notes** on key researchers.

### Simulations (neuroloc/simulations/)

18 scripts across 15 domains. Most use Brian2. Each runs standalone on CPU
in under 10 minutes. Each outputs PNG figures. Each has a README.md.
Zero comments in code.

### Printed Guide (neuroloc/neuroloc_guide.pdf)

32-page LaTeX document compiled from `print/` sources. Four sections:
I. Study Guide, II. Reference (Glossary), III. Architecture Mapping,
IV. Key Findings. Proper LaTeX math throughout. Rebuild with:

```
python build_latex.py
```

Requires MiKTeX at `C:\Users\deyan\AppData\Local\Programs\MiKTeX\`.

## 15 Adversarial Findings

Every bridge document stress-tested whether its biological analogy holds.
The dominant finding: most are superficial. The architecture works because
the biological constraints are independently useful engineering choices,
not because they faithfully reproduce biology.

1. ATMN is not a faithful LIF -- batch reset during training eliminates temporal dynamics
2. KDA is not STDP -- it is a Hopfield-like associative memory; no timing dependence
3. 41% firing rate is not sparse -- cortical neurons fire at 1-5%
4. Adaptive threshold is not divisive normalization -- global scalar, not per-neuron
5. Next-token prediction is not predictive coding -- but they learn the same weights (Millidge et al.)
6. 3:1 layer ratio is not cortical -- derived from ML benchmarks (Kimi, Qwen3, OLMo)
7. Mamba3 rotation is not oscillatory dynamics -- likely positional encoding
8. KDA alpha is not neuromodulatory gain -- wrong granularity, static after training
9. KDA+MLA is not complementary learning systems -- no consolidation
10. SwiGLU IS dendritic coincidence detection mathematically -- but 1 branch vs 30-50
11. 354x energy claim is per-op correct, system-level misleading -- data movement dominates
12. Biological and transformer attention are different operations -- selection vs retrieval
13. Spike threshold is not critical period plasticity -- no closing mechanism
14. PGA-to-grid-cell connection is weak to nonexistent
15. Residual stream is a bus, not a global workspace -- no ignition, no selectivity

The one genuine correspondence: KDA's outer product (k_t * v_t^T) IS Hebbian learning.

## Top 5 Interventions (Ranked)

These have implementation specs in their bridge documents.

1. **ATMN Leak Term** -- HIGH priority, Phase 5a
   Bridge: `bridge/neuron_models_to_atmn.md`
   Add explicit exponential decay: h_t = (1 - alpha) * u_{t-1} + x_t

2. **Activity-Dependent Alpha** (BCM-like) -- 25-35%, Phase 5b+
   Bridge: `bridge/plasticity_to_kda_delta_rule.md`
   alpha_eff = sigmoid(alpha_log + gamma * log(||S_t||))

3. **k-WTA Ternary Spikes** -- 20-30%, Phase 5+
   Bridge: `bridge/lateral_inhibition_to_adaptive_threshold.md`
   Replace threshold with top-k selection by absolute value

4. **Progressive Spike Activation** -- 20-30%, Phase 5+
   Bridge: `bridge/development_to_training_curriculum.md`
   No spikes during warmup, linear ramp over 10% of training

5. **Neuromodulator Network** (130 params/layer) -- 15-25%, Phase 6+
   Bridge: `bridge/neuromodulation_to_learning_and_gating.md`
   Small network modulates alpha and beta from global state signals

## Known Issues (From Lint Pass)

FIXED on 2026-04-07:
- head_dim 128->64 in 3 files (biological_attention_to_mla, memory_systems_to_kda_mla, memory_kda_vs_hippocampus)
- energy savings unit error: bridge table had 2,100,000 pJ instead of 2,100,000,000 pJ. corrected to ~0.0005% system-level (was 0.5%). comparison harmonized.
- internal contradiction in memory_systems_to_kda_mla.md: removed "indiscriminate" claim, alpha is per-channel per-head
- 12 unsourced claims cited across 11 files (3 requested + 9 additional from systematic search)
- 15 challenges sections verified present with 3-5 substantive criticisms each
- RoPE and RMSNorm bridge notes written (positional_encoding_to_rope.md, normalization_to_rmsnorm.md)
- synthesis/ directory populated with 5 articles
- head_dim=128 in RoPE bridge corrected to 64
- 25% MLA claim in recurrence synthesis corrected to 12.5%
- ATMN "leaky integration" in nmda_receptors.md corrected (ATMN has no leak)
- corrupted author list in three_factor_learning.md partially fixed
- "prefrontal cortex" / "mushroom body" contradiction in three_factor_learning.md fixed

REMAINING issues (from 2026-04-07 prosecutor audit):
- all-caps emphasis (NOT, BEFORE, etc.) in several bridge and mechanism articles (style, not factual)
- inconsistent section headings across mechanism articles (## key references vs ## source bibliography, ## see also vs ## related mechanisms)
- no inbound wikilinks from existing articles to new articles (new articles link out but old articles don't link in)
- synthesis articles do not cross-reference each other
- no simulations for the 5 new mechanism articles

NEW finding (2026-04-07):
- **KDA does NOT implement the delta rule erasure term.** the state update is alpha * S + beta * k * v^T without targeted erasure. the "delta rule" name is aspirational. the erasure term k_t * (k_t^T * S_{t-1}) proposed in plasticity_to_kda_delta_rule.md is not in the code.

## What Comes Next

### Immediate (Fix Quality) -- DONE 2026-04-07

all 6 items completed. see "Known Issues" section above for details.

### Medium-Term (Extend Wiki)

DONE 2026-04-07:
- 5 synthesis articles written (sparsity, timescale, local/global, compression, recurrence)
- 5 missing mechanism articles written (serotonin, GABA, NMDA, basal ganglia, three-factor learning)

REMAINING:
1. Write 5 priority comparison articles: Phase Coding vs RoPE, Dendritic
   Branches vs Attention Heads, Three-Factor Learning vs BPTT, Spike Timing
   vs Token Position, Neuromodulatory Broadcast vs Layer Normalization
2. Add inbound wikilinks from existing articles to new articles
3. Standardize section headings across all mechanism articles
4. Add cross-references between synthesis articles

### Long-Term (Validate Interventions)

Build and run Tier 2 PyTorch simulations on Kaggle T4 (use eARA autoresearch
loop -- see `scripts/autoresearch_loop.md`). Each tests one intervention at
6M scale before committing H200 time:

1. ATMN with leak vs without -- toy sequence task
2. k-WTA vs threshold spiking -- gradient flow + MI/CKA
3. Progressive spike activation -- training curves with vs without ramp
4. BCM-like adaptive alpha -- state norm dynamics over long sequences
5. GP vs random bilinear control -- the critical experiment from the spatial bridge

## How to Use Subagents

For any substantial work on the wiki, use the Agent tool with specialized subagents:

**Research** (writing new mechanism or bridge articles):
```
Agent(subagent_type="general-purpose", prompt="you are a computational
neuroscience researcher... [domain brief, article format, adversarial
thinking rule, CLAUDE.md rules]")
```
Each researcher agent needs: the domain brief, the article format from the
AGENTS.md schema (user provided it in the original conversation), the full
NEUROLOC.md component registry, the adversarial thinking rule, and the
CLAUDE.md rules (zero comments, zero emojis, lowercase, no AI attribution).

**Lint** (quality checking):
```
Agent(subagent_type="general-purpose", prompt="cross-reference validator /
contradiction detector / gap analyzer... [read-only task]")
```

**Pedagogical retrofit** (improving existing articles):
Add hooks, inline definitions, ML analogs, challenges sections, break long
sentences. The dzipobel research files at `C:\Users\deyan\Projects\dzipobel\
docs\research\` contain the pedagogical framework (plain language, cognitive
load theory, 80/20 principle, Gagne's nine events).

**Prosecutor** (adversarial quality audit):
```
Agent(subagent_type="prosecutor-claude", prompt="audit the neuroloc wiki...")
```

**PDF regeneration**:
Edit `print/` files -> run `python build_latex.py`. The print files are
separate from wiki files. The wiki uses lowercase, wikilinks, markdown. The
print files use proper capitalization, no links, tighter prose.

## Key Files to Read First

1. This file (HANDOFF.md)
2. `NEUROLOC.md` at repo root (component registry, rules, simulation constraints)
3. `CLAUDE.md` at repo root (project rules, phase sequencing, bug history)
4. `wiki/concepts/start_here.md` (wiki entry point)
5. `wiki/index.md` (flat catalog)
6. `wiki/log.md` (operation history)
7. `init.md` (project overview)

## Critical Context

- The AGENTS.md schema (provided by the user in the original conversation, not
  stored as a file) defines the article format, simulation conventions, bridge
  document format, and the 15 subagent briefs. If the user references AGENTS.md,
  they mean the neuroloc wiki schema document.

- The Karpathy LLM Wiki gist (April 3, 2026) uses the same raw/ + wiki/ + index
  + log architecture. This is intentional -- the user is aware of the parallel.

- The wiki is designed for Obsidian. Open `neuroloc/wiki/` as a vault. The
  `.obsidian/` config has graph view, backlinks, and outline enabled. Use
  `[[page_name]]` wikilinks for all internal cross-references.

- Simulations and the wiki are separate. The wiki is `neuroloc/wiki/`. The
  simulations are `neuroloc/simulations/`. They are not mixed in the same
  Obsidian vault.

- The user has an assistant who reads the wiki in Obsidian and does source
  research. The AI agent runs experiments, writes articles, and maintains
  the infrastructure.
