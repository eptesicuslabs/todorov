# operating directive

status: current (as of 2026-04-16).

this file defines how the project maintains its wiki and state documentation.
it is binding for every agent and human who touches the project. it
supersedes any previous convention that contradicts it.

this directive is itself a living document; its own update_history section
appends every revision.

## scope and authority

the directive governs:

- every file under `neuroloc/wiki/`
- the state files: `neuroloc/wiki/PROJECT_PLAN.md`, `state/program_status.yaml`, `docs/STATUS_BOARD.md`
- the entry-point documents: `CLAUDE.md`, `neuroloc/HANDOFF.md`
- the run cards and mistake docs

it does not govern:

- python source files under `neuroloc/model/*.py` and `neuroloc/simulations/**/*.py` (the code's own rules are in `CLAUDE.md`)
- external dependencies and packaging (`requirements.txt`, etc.)
- the `neuroloc/output/` tree (gitignored local artifacts)

markdown documentation files under `neuroloc/model/` (e.g. legacy run cards)
are in scope for this directive. the exemption applies only to source code,
not to documents that happen to live next to source code.

authority order when rules conflict:

1. the user's explicit written instruction
2. for questions in the code domain: `CLAUDE.md` > this directive
3. for questions in the wiki/state domain: this directive > `CLAUDE.md`
4. individual wiki articles
5. machine-generated summaries

the code vs wiki/state distinction is the same one used throughout this
directive: wiki/state covers every file under `neuroloc/wiki/`, the state
files (`PROJECT_PLAN.md`, `program_status.yaml`, `STATUS_BOARD.md`), the
entry points (`CLAUDE.md`, `HANDOFF.md`), run cards, and mistake docs.
code covers the python source tree. the user's instruction always overrides
both.

## source-of-truth hierarchy

when two documents disagree, resolve in this order:

1. **code wins over docs.** if `god_machine.py` says `alpha_log_mean=5.0` and a wiki article says it's `-0.5`, the wiki article is stale. fix the article.
2. **commit date wins.** when two wiki articles make incompatible claims, the one with the later commit takes precedence. the older article is either superseded (add banner and forward pointer) or still valid in a narrower scope (add scoping qualifier).
3. **state files have named canonical roles:**
   - `PROJECT_PLAN.md` is canonical for current project state, the current question under test, decision rules, and the append-only history.
   - `program_status.yaml` is canonical for machine-readable state: `latest_run`, `latest_run_results`, `next_action`, `run_history`.
   - `STATUS_BOARD.md` is canonical for human-readable status snapshots.
   - `CLAUDE.md` is canonical for agent-facing rules and the run summary trail.
4. **wiki articles are not canonical on their own.** they are the reasoning record. when a wiki claim needs to be authoritative for a decision, promote it to a decision rule in `PROJECT_PLAN.md` or a binding constraint in `CLAUDE.md`.

## article lifecycle

every wiki article has exactly one of four lifecycle states:

### `current`

the article's claims reflect the project's present understanding. it is
the active reference for its topic. may be edited in place as the
understanding evolves.

banner: `status: current (as of YYYY-MM-DD).`

eligible directories: `synthesis/`, `tests/`, `knowledge/`, `concepts/`, `mechanisms/`, `bridge/`, `comparisons/`, `entities/`.

### `superseded by <link>`

the article was current at one point but has been overtaken by a newer
article that makes the opposing or more complete claim. the article is
retained for evidence continuity but readers are redirected.

banner: `status: superseded by <path>. retained for evidence continuity.`

the banner must be the first non-heading content after the title. the
superseding article must carry a reverse link in its `see also` section
pointing back at the superseded article.

### `historical context only`

the article documents something that was true at a particular time (a
run that happened, an experiment that was performed, a hypothesis that
was considered). the underlying event does not change, but the
interpretation may. these articles are not intended to be kept current;
they are frozen evidence.

banner: `status: historical context only. frozen as of YYYY-MM-DD. do not edit.`

eligible directories: `mistakes/`, `tests/` (run-specific entries).

the only permitted edits to a `historical context only` article are:

- typo / formatting fixes that do not change meaning
- appending a `see also` forward pointer if the article is later superseded
- appending a clarification at the bottom if the original content is factually wrong but must not be rewritten

the format for an appended correction is a new heading `## correction (YYYY-MM-DD)` at the bottom of the article (before `## see also`), containing the correcting text. a correction may be of any length. it must not edit any prior text; all prior text including the now-falsified claim is preserved as evidence. if the correction itself is later falsified, a new `## correction (YYYY-MM-DD)` heading is appended after the prior one, never replacing it. this is consistent with the append-only rule for the mistakes/ directory.

### `definitional`

the article defines a term, a mechanism, an entity, or a concept. it
changes only when the underlying definition changes (new evidence
corrects a claim about a biological mechanism, a mathematical identity
is revised, etc.). not intended to carry a commit-date banner because
staleness is rare by construction.

banner: `status: definitional. last fact-checked YYYY-MM-DD.`

eligible directories: `concepts/`, `entities/`, `mechanisms/` (primary), `bridge/` (secondary).

## banner format (strict)

every article's first non-heading line must be the banner. acceptable
formats:

```
status: current (as of 2026-04-16).
```

```
status: superseded by wiki/synthesis/training_objective_vs_architectural_goal.md. retained for evidence continuity.
```

```
status: historical context only. frozen as of 2026-04-15. do not edit.
```

```
status: definitional. last fact-checked 2026-04-10.
```

no other opening content is permitted above the banner except the article
title. any article whose first non-heading line is not the banner fails
the prosecutor on any wiki change TO THAT ARTICLE. see the migration
policy section below for how banner enforcement works on pre-existing
articles.

## migration policy for pre-existing articles

at the time this directive was committed (2026-04-16), the wiki contained
~200 articles without banners. a retroactive enforcement that declared
every one of them a violation would produce a wall of findings on the
first commit, making the prosecutor protocol unusable. the directive
therefore defines a staged migration.

**pre-migration state.** an article that existed before this directive
and has not yet been touched is in the `pre-migration` state. it is
assumed to be `current` in meaning but is not yet subject to banner
enforcement. a prosecutor pass that does not touch a pre-migration
article does not flag it for a missing banner.

**migration trigger.** an article leaves pre-migration under any of these
conditions:

1. the article is modified in a commit for any reason. the same commit
   must add the correct banner and `see also` section, or the prosecutor
   fails.
2. the scheduled refactor touches the article. the refactor (phase 2 of
   the 2026-04-16 plan) walks the wiki directory by directory, adds
   banners in bulk, and closes each directory with a prosecutor pass.
3. a new article in the same directory references the pre-migration
   article via `see also` and needs the reverse link. the reverse link
   addition is a modification and triggers the rule.

**scheduled migration completion.** the 2026-04-16 refactor is expected
to complete the migration within a bounded number of sessions. until then,
the directory-by-directory completion order is synthesis/ > tests/ >
mistakes/ > knowledge/ > bridge/ > comparisons/ > concepts/ > entities/
> mechanisms/. the refactor's progress is tracked in `PROJECT_PLAN.md`
under the update_history section.

**post-migration enforcement.** once the refactor completes, the
pre-migration state no longer exists. any article without a banner is a
violation on any commit that references it, including an unmodified
commit whose diff does not touch the article. the prosecutor's surface
expands to the full wiki.

## append-only sections

the following sections are append-only. never edit prior entries; append
a correction or clarification as a new entry instead.

- `PROJECT_PLAN.md` > `update history`
- `CLAUDE.md` > `results summary`, `bug history`, `phase sequencing`, and the section headed by `# architecture rules`
- every article under `mistakes/`

the append-only rule prevents evidence laundering: a mistake recorded
and later silently corrected is worse than a mistake recorded and later
openly overruled. keep the trail.

## cross-references (bidirectional)

every article must carry a `see also` section as its last section (before
any references or glossary). entries under `see also` are links to other
articles in the wiki that inform the current article's content.

bidirectional requirement:

- when article A supersedes article B, both must link each other. A's
  banner links to B (stated as "prior"); B's banner and `see also` both
  link to A.
- when article A cites article B as evidence, both must cross-link.
- when article A is the run card for a paid run, and article B is the
  mistake doc for that same run, both must cross-link.

the prosecutor treats a one-way link as a finding. it is often caught by
changing article A and forgetting to update article B. both sides of a
link are part of the same commit.

## prosecutor protocol on wiki changes

the prosecutor runs on any commit that touches:

- any file in `wiki/synthesis/`
- any file in `wiki/mistakes/`
- any file in `wiki/tests/`
- `PROJECT_PLAN.md`
- `OPERATING_DIRECTIVE.md` (this file)
- any state file (`program_status.yaml`, `STATUS_BOARD.md`, `CLAUDE.md`)
- the run cards in `wiki/tests/`

the prosecutor cycles to zero findings. process:

1. commit the change
2. launch `feature-dev:code-reviewer` with the commit hash and the list of changed files
3. fix every finding the prosecutor reports (critical, important, minor — all of them, per the existing `CLAUDE.md` rule)
4. commit the fixes
5. re-run the prosecutor on the new commit
6. repeat until zero findings

wiki changes that do not touch the above directories (e.g., a typo fix
in `entities/`, a citation update in `knowledge/`) may skip the
prosecutor unless the change alters a factual claim. when in doubt, run
the prosecutor.

## run cards

every paid run produces a run card. the canonical location is
`wiki/tests/<run_name>_results.md`.

the `neuroloc/output/` tree is gitignored. anything a run writes there
is local and ephemeral. if a run's result is worth preserving, copy the
key artifact (the run card) to `wiki/tests/`.

the run card's banner is `status: historical context only. frozen as of <date>. do not edit.`

the run card must cross-link to:

- any synthesis article that was informed by the run's result
- any mistake doc that covers a bug encountered during the run
- the prior runs in the index that it compares against

## file naming

- lowercase with underscores: `training_objective_vs_architectural_goal.md`
- one article per claim or event. do not mix "the substrate design" and "the paid run's result" in one file.
- capitalised top-level files are reserved for meta-documents: `PROJECT_PLAN.md`, `OPERATING_DIRECTIVE.md`, `INDEX.md`, `HANDOFF.md`, `CLAUDE.md`.

## how to handle disagreement

between agent and user: the user's explicit written instruction wins. if
the agent disagrees, it must say so, cite the rule or evidence it
disagrees on, and stop the action until the user overrides or concurs.

between this directive and the user: the user's explicit written
instruction wins. the directive can be amended (append to update_history)
but the amendment is retained.

between this directive and `CLAUDE.md`: `CLAUDE.md` wins on code rules.
this directive wins on wiki / state rules. when the two collide on
something that could be either (e.g., "are comments allowed in a wiki
code block?"), default to the stricter.

between two prior commits: the later commit wins by default. if the
later commit was itself a mistake (e.g., I3 in the run2 documentation
cycle was an invented claim), the correction is appended, not the prior
commit overwritten.

## update history

append-only. every change to this directive gets a new entry with date,
author, and a one-line description of what changed and why.

- **2026-04-16** — deyan todorov — file created. first draft after the five-paid-runs diagnosis cycle concluded. scope, source-of-truth hierarchy, four-state article lifecycle, banner format, append-only sections, bidirectional cross-reference rule, prosecutor protocol for wiki changes, run-card location rule, file naming, disagreement resolution. supersedes the ad-hoc practice documented in `CLAUDE.md` under "keeping this file current" and the scattered rules across prior mistake docs.
- **2026-04-16** — deyan todorov — first-round prosecutor fixes. C1: directive's own banner was two sentences; trimmed to the single line the format mandates with explanatory prose moved below. C2: added the "migration policy for pre-existing articles" section defining pre-migration state, migration triggers, scheduled completion order, and post-migration enforcement. C3: narrowed the scope exemption from `neuroloc/model/` and `neuroloc/simulations/` to their `*.py` files only, so run cards or other markdown that happen to live under those directories are in scope. I1: authority-order table was unconditional (CLAUDE.md > directive always) while the conflict-resolution section split by domain; reconciled by rewriting the table to carry the domain split itself. I2: added a concrete format for appended corrections on `historical context only` articles (`## correction (YYYY-MM-DD)` heading, append-only, preserving original text). M1: removed the see-also entry pointing at the planned `INDEX.md` (the directive's own rule forbids see-also entries to non-existent articles).
- **2026-04-16** — deyan todorov — second-round prosecutor fix. the first-round C1 fix trimmed the banner to one line but left three lines of introductory prose ABOVE the banner at lines 3-5. the directive's own banner-format rule says the banner must be the first non-heading content after the title, with no other opening content permitted. the directive itself was in violation. fix: moved the three lines of intro prose to below the banner, so the file now reads title, blank, banner, blank, intro prose, blank, living-document note. no content removed; only relocated.

## see also

- `neuroloc/wiki/PROJECT_PLAN.md` — canonical project state. this directive governs how that file is maintained.
- `CLAUDE.md` — agent-facing rules for code. this directive covers the wiki/state side.
- `neuroloc/wiki/INDEX.md` — human-readable navigation map of the wiki.
